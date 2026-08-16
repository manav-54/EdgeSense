package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"

	"github.com/manav-54/edgesense/ingest/internal/contracts"
	"github.com/manav-54/edgesense/ingest/internal/session"
)

// --- doubles ---------------------------------------------------------------

type fakePublisher struct {
	mu       sync.Mutex
	got      []contracts.TranscriptSegment
	failWith error
}

func (f *fakePublisher) Publish(_ context.Context, seg *contracts.TranscriptSegment) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.failWith != nil {
		return f.failWith
	}
	f.got = append(f.got, *seg)
	return nil
}

func (f *fakePublisher) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.got)
}

type fakeSessions struct {
	mu       sync.Mutex
	seen     map[string]bool
	touches  int
	ended    int
	pingErr  error
	failOpen bool // simulate Redis down: SeenBefore always reports "new"
}

func newFakeSessions() *fakeSessions {
	return &fakeSessions{seen: map[string]bool{}}
}

func (f *fakeSessions) SeenBefore(_ context.Context, key string) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.failOpen {
		return false
	}
	if f.seen[key] {
		return true
	}
	f.seen[key] = true
	return false
}

func (f *fakeSessions) Touch(_ context.Context, _, _ string, _ int64, _ bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.touches++
}

func (f *fakeSessions) EndCall(_ context.Context, _ string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.ended++
}

func (f *fakeSessions) Get(_ context.Context, callID string) (*session.State, error) {
	return &session.State{CallID: callID, Segments: 3}, nil
}

func (f *fakeSessions) Ping(_ context.Context) error { return f.pingErr }

// --- harness ---------------------------------------------------------------

func newTestServer(t *testing.T, pub Publisher, sess Sessions) (*httptest.Server, *Server) {
	t.Helper()
	log := slog.New(slog.NewJSONHandler(io.Discard, nil))
	srv := New(pub, sess, log)
	mux := http.NewServeMux()
	srv.Routes(mux)
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return ts, srv
}

func dial(t *testing.T, ts *httptest.Server, callID string) *websocket.Conn {
	t.Helper()
	url := "ws" + strings.TrimPrefix(ts.URL, "http") + "/v1/stream"
	if callID != "" {
		url += "?call_id=" + callID
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, url, nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	return conn
}

func segmentJSON(callID string, seq int, final bool, text string) []byte {
	seg := contracts.TranscriptSegment{
		SchemaVersion: contracts.ContractVersion,
		CallID:        callID,
		Seq:           seq,
		Speaker:       contracts.SpeakerCustomer,
		Text:          text,
		IsFinal:       final,
		StartMs:       seq * 1000,
		EndMs:         seq*1000 + 900,
		EmittedAt:     time.Now().UTC().Format(time.RFC3339Nano),
		ASRConfidence: 0.9,
	}
	b, _ := json.Marshal(seg)
	return b
}

func send(t *testing.T, conn *websocket.Conn, payload []byte) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := conn.Write(ctx, websocket.MessageText, payload); err != nil {
		t.Fatalf("write: %v", err)
	}
}

func eventually(t *testing.T, want int, get func() int) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if get() >= want {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for count %d, last was %d", want, get())
}

// --- tests -----------------------------------------------------------------

func TestAcceptsValidSegments(t *testing.T) {
	pub, sess := &fakePublisher{}, newFakeSessions()
	ts, _ := newTestServer(t, pub, sess)

	conn := dial(t, ts, "call-1")
	for i := 0; i < 3; i++ {
		send(t, conn, segmentJSON("call-1", i, true, fmt.Sprintf("segment %d", i)))
	}
	eventually(t, 3, pub.count)
	_ = conn.Close(websocket.StatusNormalClosure, "")
}

// TestDedupesOnCallIDAndSeq covers the reconnect case: an edge agent that
// loses its socket and replays the last few segments must not double-count
// them downstream.
func TestDedupesOnCallIDAndSeq(t *testing.T) {
	pub, sess := &fakePublisher{}, newFakeSessions()
	ts, _ := newTestServer(t, pub, sess)

	conn := dial(t, ts, "call-2")
	payload := segmentJSON("call-2", 7, true, "hello")
	send(t, conn, payload)
	eventually(t, 1, pub.count)

	for i := 0; i < 4; i++ {
		send(t, conn, payload)
	}
	time.Sleep(200 * time.Millisecond)

	if got := pub.count(); got != 1 {
		t.Fatalf("expected 1 published segment after 5 identical sends, got %d", got)
	}
	_ = conn.Close(websocket.StatusNormalClosure, "")
}

func TestPartialAndFinalWithSameSeqBothPublish(t *testing.T) {
	pub, sess := &fakePublisher{}, newFakeSessions()
	ts, _ := newTestServer(t, pub, sess)

	conn := dial(t, ts, "call-3")
	send(t, conn, segmentJSON("call-3", 0, false, "partial text"))
	send(t, conn, segmentJSON("call-3", 0, true, "final text"))
	eventually(t, 2, pub.count)
	_ = conn.Close(websocket.StatusNormalClosure, "")
}

func TestRejectsSegmentClaimingAnotherCall(t *testing.T) {
	pub, sess := &fakePublisher{}, newFakeSessions()
	ts, _ := newTestServer(t, pub, sess)

	conn := dial(t, ts, "call-mine")
	send(t, conn, segmentJSON("call-someone-else", 0, true, "cross-stream write"))
	time.Sleep(200 * time.Millisecond)

	if got := pub.count(); got != 0 {
		t.Fatalf("expected cross-call segment to be rejected, published %d", got)
	}
	_ = conn.Close(websocket.StatusNormalClosure, "")
}

func TestRejectsLyingRedactionOverTheWire(t *testing.T) {
	pub, sess := &fakePublisher{}, newFakeSessions()
	ts, _ := newTestServer(t, pub, sess)

	// Claims a redaction while leaving the raw card in the text.
	payload := []byte(`{"schema_version":"1.0","call_id":"call-4","seq":0,` +
		`"speaker":"customer","text":"card 4242424242424242 ok","is_final":true,` +
		`"start_ms":0,"end_ms":900,"emitted_at":"2026-08-16T12:00:00Z",` +
		`"redactions":[{"type":"CARD","placeholder":"<CARD_1>","start":5,"end":13,` +
		`"detector":"regex","confidence":0.99}],"asr_confidence":0.9}`)

	conn := dial(t, ts, "call-4")
	send(t, conn, payload)
	time.Sleep(200 * time.Millisecond)

	if got := pub.count(); got != 0 {
		t.Fatalf("expected a lying redaction to be rejected, published %d", got)
	}
	_ = conn.Close(websocket.StatusNormalClosure, "")
}

func TestMalformedJSONDoesNotKillTheConnection(t *testing.T) {
	pub, sess := &fakePublisher{}, newFakeSessions()
	ts, _ := newTestServer(t, pub, sess)

	conn := dial(t, ts, "call-5")
	send(t, conn, []byte("{not json"))
	send(t, conn, segmentJSON("call-5", 0, true, "still alive"))

	eventually(t, 1, pub.count)
	_ = conn.Close(websocket.StatusNormalClosure, "")
}

// TestShedFinalTellsTheClient asserts backpressure is visible rather than
// silent: when a final is dropped the client is told to throttle.
func TestShedFinalTellsTheClient(t *testing.T) {
	pub := &fakePublisher{failWith: errors.New("publish queue full")}
	ts, _ := newTestServer(t, pub, newFakeSessions())

	conn := dial(t, ts, "call-6")
	send(t, conn, segmentJSON("call-6", 0, true, "under pressure"))

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_, data, err := conn.Read(ctx)
	if err != nil {
		t.Fatalf("expected a throttle message, got read error: %v", err)
	}
	var msg struct {
		Type   string `json:"type"`
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal(data, &msg); err != nil {
		t.Fatalf("unmarshal throttle: %v", err)
	}
	if msg.Type != "throttle" {
		t.Fatalf("expected type=throttle, got %q", msg.Type)
	}
	_ = conn.Close(websocket.StatusNormalClosure, "")
}

// TestDedupeFailsOpenWhenRedisIsDown encodes the availability choice: losing
// the dedupe cache must not lose live call content.
func TestDedupeFailsOpenWhenRedisIsDown(t *testing.T) {
	pub := &fakePublisher{}
	sess := newFakeSessions()
	sess.failOpen = true
	ts, _ := newTestServer(t, pub, sess)

	conn := dial(t, ts, "call-7")
	payload := segmentJSON("call-7", 0, true, "redis is down")
	send(t, conn, payload)
	send(t, conn, payload)

	eventually(t, 2, pub.count)
	if got := pub.count(); got != 2 {
		t.Fatalf("expected both segments through while failing open, got %d", got)
	}
	_ = conn.Close(websocket.StatusNormalClosure, "")
}

func TestEndCallControlMessage(t *testing.T) {
	pub, sess := &fakePublisher{}, newFakeSessions()
	ts, _ := newTestServer(t, pub, sess)

	conn := dial(t, ts, "call-8")
	send(t, conn, segmentJSON("call-8", 0, true, "bye"))
	eventually(t, 1, pub.count)
	send(t, conn, []byte(`{"type":"end_call"}`))

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		sess.mu.Lock()
		ended := sess.ended
		sess.mu.Unlock()
		if ended > 0 {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("end_call did not mark the session ended")
}

func TestStreamRequiresCallID(t *testing.T) {
	ts, _ := newTestServer(t, &fakePublisher{}, newFakeSessions())
	url := "ws" + strings.TrimPrefix(ts.URL, "http") + "/v1/stream"
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if _, _, err := websocket.Dial(ctx, url, nil); err == nil {
		t.Fatal("expected dial without call_id to be refused")
	}
}

func TestReadyReportsDegradedRatherThanDown(t *testing.T) {
	sess := newFakeSessions()
	sess.pingErr = errors.New("connection refused")
	ts, _ := newTestServer(t, &fakePublisher{}, sess)

	resp, err := http.Get(ts.URL + "/readyz")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("redis down should be degraded-but-ready, got %d", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "degraded") {
		t.Fatalf("expected degraded status, got %s", body)
	}
}

func TestDrainingFailsReadiness(t *testing.T) {
	ts, srv := newTestServer(t, &fakePublisher{}, newFakeSessions())
	srv.SetReady(false)

	resp, err := http.Get(ts.URL + "/readyz")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("draining should fail readiness, got %d", resp.StatusCode)
	}
}
