// Package obs holds structured logging, Prometheus metrics, and tracing.
package obs

import (
	"context"
	"log/slog"
	"os"
	"regexp"
	"strings"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/resource"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"
)

var longDigits = regexp.MustCompile(`\d[\d\s\-.]{6,}\d`)

// Scrub strips digit runs from a string before it reaches a log.
//
// Ingest should never hold raw PII -- everything arriving here is already
// redacted. This exists for the case where that assumption is wrong: an edge
// build with a redaction bug, or a client that is not our edge agent at all.
// Logging its payload verbatim would take a client-side leak and write it into
// server-side log storage, turning a contained bug into a retained one.
func Scrub(s string) string {
	return longDigits.ReplaceAllString(s, "[redacted:digits]")
}

type scrubHandler struct{ slog.Handler }

func (h scrubHandler) Handle(ctx context.Context, r slog.Record) error {
	r.Message = Scrub(r.Message)
	if span := trace.SpanContextFromContext(ctx); span.IsValid() {
		r.AddAttrs(
			slog.String("trace_id", span.TraceID().String()),
			slog.String("span_id", span.SpanID().String()),
		)
	}
	return h.Handler.Handle(ctx, r)
}

func (h scrubHandler) WithAttrs(as []slog.Attr) slog.Handler {
	return scrubHandler{h.Handler.WithAttrs(as)}
}

func (h scrubHandler) WithGroup(name string) slog.Handler {
	return scrubHandler{h.Handler.WithGroup(name)}
}

// SetupLogging installs a JSON logger that scrubs and adds trace correlation.
func SetupLogging(level string) *slog.Logger {
	var lvl slog.Level
	switch strings.ToLower(level) {
	case "debug":
		lvl = slog.LevelDebug
	case "warn", "warning":
		lvl = slog.LevelWarn
	case "error":
		lvl = slog.LevelError
	default:
		lvl = slog.LevelInfo
	}

	base := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: lvl,
		ReplaceAttr: func(_ []string, a slog.Attr) slog.Attr {
			if a.Key == slog.TimeKey {
				a.Key = "ts"
			}
			if a.Key == slog.MessageKey {
				a.Key = "msg"
			}
			if a.Key == slog.LevelKey {
				a.Value = slog.StringValue(strings.ToLower(a.Value.String()))
			}
			return a
		},
	})
	logger := slog.New(scrubHandler{base}).With(slog.String("service", "ingest"))
	slog.SetDefault(logger)
	return logger
}

// SetupTracing configures OTLP export. Returns a shutdown func.
//
// A missing collector degrades to a no-op tracer. Ingest must keep accepting
// calls when the observability stack is down; the alternative is an
// availability incident caused by a monitoring outage.
func SetupTracing(ctx context.Context, endpoint string) (func(context.Context) error, error) {
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{}, propagation.Baggage{},
	))
	if endpoint == "" {
		return func(context.Context) error { return nil }, nil
	}

	exp, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithEndpoint(strings.TrimPrefix(strings.TrimPrefix(endpoint, "http://"), "https://")),
		otlptracegrpc.WithInsecure(),
	)
	if err != nil {
		slog.Warn("tracing disabled: exporter setup failed", slog.String("error", err.Error()))
		return func(context.Context) error { return nil }, nil
	}

	res, _ := resource.Merge(resource.Default(), resource.NewWithAttributes(
		semconv.SchemaURL,
		semconv.ServiceName("ingest"),
		attribute.String("service.namespace", "edgesense"),
	))
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exp),
		sdktrace.WithResource(res),
	)
	otel.SetTracerProvider(tp)
	return tp.Shutdown, nil
}

func Tracer() trace.Tracer { return otel.Tracer("ingest") }

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

var (
	SegmentsReceived = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "edgesense_ingest_segments_received_total",
		Help: "Segments read off a WebSocket, before validation.",
	}, []string{"is_final"})

	SegmentsRejected = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "edgesense_ingest_segments_rejected_total",
		Help: "Segments refused, by reason.",
	}, []string{"reason"})

	SegmentsDuplicate = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edgesense_ingest_segments_duplicate_total",
		Help: "Segments dropped by the idempotency check.",
	})

	SegmentsPublished = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "edgesense_ingest_segments_published_total",
		Help: "Segments written to Kafka.",
	}, []string{"topic"})

	PublishErrors = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "edgesense_ingest_publish_errors_total",
		Help: "Kafka write failures, by kind.",
	}, []string{"kind"})

	BackpressureEvents = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "edgesense_ingest_backpressure_events_total",
		Help: "Times the publish queue was full, by action taken.",
	}, []string{"action"})

	QueueDepth = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "edgesense_ingest_publish_queue_depth",
		Help: "Segments waiting to be written to Kafka.",
	})

	ActiveConnections = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "edgesense_ingest_active_connections",
		Help: "Open edge WebSocket connections.",
	})

	IngestLatency = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "edgesense_ingest_handle_seconds",
		Help:    "Time from reading a segment to handing it to the publisher.",
		Buckets: []float64{0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5},
	})

	PublishLatency = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "edgesense_ingest_publish_seconds",
		Help:    "Time for one Kafka write to be acknowledged.",
		Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5},
	})

	EdgeToIngestLatency = promauto.NewHistogram(prometheus.HistogramOpts{
		Name: "edgesense_edge_to_ingest_seconds",
		Help: "Wall-clock from the edge's emitted_at to arrival here. " +
			"Depends on clock sync between hosts; treat as indicative.",
		Buckets: []float64{0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5},
	})

	SessionErrors = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "edgesense_ingest_session_errors_total",
		Help: "Redis failures, by operation. Ingest fails open on these.",
	}, []string{"op"})
)
