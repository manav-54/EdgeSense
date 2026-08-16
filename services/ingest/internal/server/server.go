// Package server exposes the edge-facing WebSocket endpoint and its HTTP siblings.
package server

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"

	"github.com/manav-54/edgesense/ingest/internal/contracts"
	"github.com/manav-54/edgesense/ingest/internal/obs"
	"github.com/manav-54/edgesense/ingest/internal/session"
)

// MaxMessageBytes caps a single WebSocket frame. Generous for a transcript
// segment, small enough that a hostile client cannot allocate freely.
const MaxMessageBytes = 64 * 1024

// ReadTimeout closes a connection that has gone quiet. A live call produces a
// segment every few seconds; a minute of silence means the edge is gone and
// the connection is holding resources for nothing.
const ReadTimeout = 60 * time.Second

// Publisher and Sessions are narrow interfaces over the concrete Kafka and
// Redis implementations. They exist so the handler's decision logic --
// validate, dedupe, shed under pressure -- can be tested against a real
// WebSocket connection without standing up a broker, which is the part of this
// service most likely to break and hardest to reason about from the code.
type Publisher interface {
	Publish(ctx context.Context, seg *contracts.TranscriptSegment) error
}

type Sessions interface {
	SeenBefore(ctx context.Context, key string) bool
	Touch(ctx context.Context, callID, agentID string, seq int64, duplicate bool)
	EndCall(ctx context.Context, callID string)
	Get(ctx context.Context, callID string) (*session.State, error)
	Ping(ctx context.Context) error
}

type Server struct {
	pub      Publisher
	sessions Sessions
	log      *slog.Logger
	ready    atomic.Bool
}

type control struct {
	Type string `json:"type"`
}

func New(pub Publisher, sessions Sessions, log *slog.Logger) *Server {
	s := &Server{pub: pub, sessions: sessions, log: log}
	s.ready.Store(true)
	return s
}

func (s *Server) Routes(mux *http.ServeMux) {
	mux.HandleFunc("/v1/stream", s.handleStream)
	mux.HandleFunc("/healthz", s.handleHealth)
	mux.HandleFunc("/readyz", s.handleReady)
	mux.HandleFunc("/v1/calls/", s.handleCall)
}

func (s *Server) SetReady(v bool) { s.ready.Store(v) }

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}

func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	if !s.ready.Load() {
		http.Error(w, `{"status":"draining"}`, http.StatusServiceUnavailable)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), time.Second)
	defer cancel()
	if err := s.sessions.Ping(ctx); err != nil {
		// Degraded, not down: dedupe fails open, so we still accept traffic.
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"degraded","detail":"redis unavailable; dedupe failing open"}`))
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"status":"ready"}`))
}

func (s *Server) handleCall(w http.ResponseWriter, r *http.Request) {
	callID := strings.TrimPrefix(r.URL.Path, "/v1/calls/")
	if callID == "" {
		http.Error(w, `{"error":"call_id required"}`, http.StatusBadRequest)
		return
	}
	st, err := s.sessions.Get(r.Context(), callID)
	if err != nil {
		http.Error(w, `{"error":"not found"}`, http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(st)
}

func (s *Server) handleStream(w http.ResponseWriter, r *http.Request) {
	callID := r.URL.Query().Get("call_id")
	if callID == "" {
		http.Error(w, "call_id query parameter is required", http.StatusBadRequest)
		obs.SegmentsRejected.WithLabelValues("missing_call_id").Inc()
		return
	}

	conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{
		// The edge agent is a first-party client on the operator's machine,
		// not a browser. Origin checking is handled by network policy; see
		// DESIGN.md on why authentication is a named non-goal here.
		InsecureSkipVerify: true,
		CompressionMode:    websocket.CompressionContextTakeover,
	})
	if err != nil {
		s.log.Warn("websocket upgrade failed",
			slog.String("call_id", callID), slog.String("error", err.Error()))
		return
	}
	conn.SetReadLimit(MaxMessageBytes)

	obs.ActiveConnections.Inc()
	defer obs.ActiveConnections.Dec()

	ctx := r.Context()
	s.log.InfoContext(ctx, "call stream opened", slog.String("call_id", callID))

	closeStatus := websocket.StatusNormalClosure
	closeReason := ""
	accepted, rejected, dupes := 0, 0, 0

	for {
		readCtx, cancel := context.WithTimeout(ctx, ReadTimeout)
		typ, data, err := conn.Read(readCtx)
		cancel()

		if err != nil {
			if !isExpectedClose(err) {
				s.log.InfoContext(ctx, "call stream closed",
					slog.String("call_id", callID), slog.String("error", err.Error()))
			}
			break
		}
		if typ != websocket.MessageText {
			obs.SegmentsRejected.WithLabelValues("binary_frame").Inc()
			rejected++
			continue
		}

		// A control frame ends the call cleanly so post-call analysis can start
		// without waiting for a TCP timeout.
		var ctl control
		if err := json.Unmarshal(data, &ctl); err == nil && ctl.Type == "end_call" {
			s.sessions.EndCall(ctx, callID)
			closeReason = "end_call"
			break
		}

		outcome := s.handleSegment(ctx, callID, data)
		switch outcome {
		case outcomeAccepted:
			accepted++
		case outcomeDuplicate:
			dupes++
		case outcomeRejected:
			rejected++
		case outcomeShed:
			// The queue is saturated and we dropped a final. Tell the client
			// rather than failing silently -- it can slow down or reconnect.
			_ = conn.Write(ctx, websocket.MessageText,
				[]byte(`{"type":"throttle","reason":"publish_queue_full"}`))
			rejected++
		}
	}

	s.sessions.EndCall(ctx, callID)
	_ = conn.Close(closeStatus, closeReason)
	s.log.InfoContext(ctx, "call stream finished",
		slog.String("call_id", callID),
		slog.Int("accepted", accepted),
		slog.Int("duplicates", dupes),
		slog.Int("rejected", rejected),
	)
}

type outcome int

const (
	outcomeAccepted outcome = iota
	outcomeRejected
	outcomeDuplicate
	outcomeShed
)

func (s *Server) handleSegment(ctx context.Context, callID string, data []byte) outcome {
	start := time.Now()
	defer func() { obs.IngestLatency.Observe(time.Since(start).Seconds()) }()

	var seg contracts.TranscriptSegment
	if err := json.Unmarshal(data, &seg); err != nil {
		obs.SegmentsRejected.WithLabelValues("malformed_json").Inc()
		s.log.WarnContext(ctx, "malformed segment",
			slog.String("call_id", callID), slog.String("error", err.Error()))
		return outcomeRejected
	}

	obs.SegmentsReceived.WithLabelValues(boolLabel(seg.IsFinal)).Inc()

	// The URL is authoritative. A payload claiming a different call_id is
	// either a client bug or an attempt to write into another call's stream.
	if seg.CallID != "" && seg.CallID != callID {
		obs.SegmentsRejected.WithLabelValues("call_id_mismatch").Inc()
		s.log.WarnContext(ctx, "segment call_id does not match connection",
			slog.String("connection_call_id", callID),
			slog.String("payload_call_id", seg.CallID))
		return outcomeRejected
	}
	seg.CallID = callID

	if err := seg.Validate(); err != nil {
		obs.SegmentsRejected.WithLabelValues("invalid").Inc()
		s.log.WarnContext(ctx, "segment failed validation",
			slog.String("call_id", callID), slog.String("error", err.Error()))
		return outcomeRejected
	}

	// Continue the edge's trace rather than starting a new one, so one call_id
	// is followable from microphone to ClickHouse.
	spanCtx := ctx
	if seg.Traceparent != nil && *seg.Traceparent != "" {
		spanCtx = otel.GetTextMapPropagator().Extract(ctx,
			propagation.MapCarrier{"traceparent": *seg.Traceparent})
	}
	spanCtx, span := obs.Tracer().Start(spanCtx, "ingest.segment",
		trace.WithSpanKind(trace.SpanKindServer),
		trace.WithAttributes(
			attribute.String("edgesense.call_id", callID),
			attribute.Int("edgesense.seq", seg.Seq),
			attribute.Bool("edgesense.is_final", seg.IsFinal),
			attribute.Int("edgesense.redactions", len(seg.Redactions)),
		))
	defer span.End()

	if emitted, err := time.Parse(time.RFC3339Nano, seg.EmittedAt); err == nil {
		if d := time.Since(emitted); d > 0 && d < time.Hour {
			obs.EdgeToIngestLatency.Observe(d.Seconds())
		}
	}

	key := seg.IdempotencyKey()
	if s.sessions.SeenBefore(spanCtx, key) {
		obs.SegmentsDuplicate.Inc()
		s.sessions.Touch(spanCtx, callID, deref(seg.AgentID), int64(seg.Seq), true)
		span.SetAttributes(attribute.Bool("edgesense.duplicate", true))
		return outcomeDuplicate
	}

	if err := s.pub.Publish(spanCtx, &seg); err != nil {
		span.RecordError(err)
		s.log.ErrorContext(spanCtx, "failed to publish segment",
			slog.String("call_id", callID), slog.Int("seq", seg.Seq),
			slog.String("error", err.Error()))
		return outcomeShed
	}

	s.sessions.Touch(spanCtx, callID, deref(seg.AgentID), int64(seg.Seq), false)
	return outcomeAccepted
}

func isExpectedClose(err error) bool {
	status := websocket.CloseStatus(err)
	if status == websocket.StatusNormalClosure || status == websocket.StatusGoingAway {
		return true
	}
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

func boolLabel(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}
