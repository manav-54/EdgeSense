package contracts

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestPythonFixturesValidate is the cross-language contract check.
//
// The fixtures are produced by the real Python redactor and the real encoder
// (tools/fixtures/make_segments.py). If the Go struct drifts from the pydantic
// model -- a renamed field, a changed offset convention, a tightened bound --
// this fails rather than silently rejecting live traffic in production.
func TestPythonFixturesValidate(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "segments.jsonl")
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("fixtures missing (%v); run: python -m tools.fixtures.make_segments", err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024)

	count, withRedactions := 0, 0
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var seg TranscriptSegment
		if err := json.Unmarshal([]byte(line), &seg); err != nil {
			t.Fatalf("segment %d: unmarshal failed: %v", count, err)
		}
		if err := seg.Validate(); err != nil {
			t.Fatalf("segment %d (call %s seq %d): %v\npayload: %s",
				count, seg.CallID, seg.Seq, err, line)
		}
		if len(seg.Redactions) > 0 {
			withRedactions++
		}
		count++
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	if count == 0 {
		t.Fatal("no fixtures loaded; the test would prove nothing")
	}
	// Guard against a fixture regeneration that accidentally strips redactions,
	// which would make the span checks below vacuous.
	if withRedactions < 10 {
		t.Fatalf("only %d/%d fixtures carry redactions; expected the corpus to be PII-rich",
			withRedactions, count)
	}
	t.Logf("validated %d Python-produced segments (%d with redactions)", count, withRedactions)
}

func baseSegment() TranscriptSegment {
	return TranscriptSegment{
		SchemaVersion: ContractVersion,
		CallID:        "call-1",
		Seq:           3,
		Speaker:       SpeakerCustomer,
		Text:          "my card is <CARD_1> thanks",
		IsFinal:       true,
		StartMs:       1000,
		EndMs:         2000,
		EmittedAt:     "2026-08-16T12:00:00.000000Z",
		ASRConfidence: 0.9,
		Redactions: []RedactionRef{
			{Type: "CARD", Placeholder: "<CARD_1>", Start: 11, End: 19,
				Detector: "regex+checksum", Confidence: 0.99},
		},
	}
}

func TestValidateAcceptsWellFormed(t *testing.T) {
	seg := baseSegment()
	if err := seg.Validate(); err != nil {
		t.Fatalf("expected valid, got %v", err)
	}
}

// TestValidateRejectsLyingRedaction is the important one.
//
// A client can claim a span is redacted while leaving the raw value in the
// text. The portal would render a redaction badge over readable PII, and every
// downstream consumer would treat the segment as clean. Ingest re-derives the
// truth from the text itself instead of trusting the claim.
func TestValidateRejectsLyingRedaction(t *testing.T) {
	seg := baseSegment()
	seg.Text = "my card is 4242424242424242 thanks"
	// span left pointing at [11,19), which now holds real digits
	if err := seg.Validate(); err == nil {
		t.Fatal("expected rejection: redaction span does not hold its placeholder")
	}
}

func TestValidateRejectsOutOfRangeSpan(t *testing.T) {
	seg := baseSegment()
	seg.Redactions[0].End = 9999
	if err := seg.Validate(); err == nil {
		t.Fatal("expected rejection for span past end of text")
	}
}

func TestValidateRejectsInvertedSpan(t *testing.T) {
	seg := baseSegment()
	seg.Redactions[0].Start, seg.Redactions[0].End = 19, 11
	if err := seg.Validate(); err == nil {
		t.Fatal("expected rejection for inverted span")
	}
}

func TestValidateRejectsBadInputs(t *testing.T) {
	cases := map[string]func(*TranscriptSegment){
		"empty call_id":       func(s *TranscriptSegment) { s.CallID = "" },
		"negative seq":        func(s *TranscriptSegment) { s.Seq = -1 },
		"unknown speaker":     func(s *TranscriptSegment) { s.Speaker = "operator" },
		"inverted times":      func(s *TranscriptSegment) { s.StartMs, s.EndMs = 2000, 1000 },
		"negative start":      func(s *TranscriptSegment) { s.StartMs = -1 },
		"confidence over one": func(s *TranscriptSegment) { s.ASRConfidence = 1.5 },
		"missing emitted_at":  func(s *TranscriptSegment) { s.EmittedAt = "" },
		"future schema":       func(s *TranscriptSegment) { s.SchemaVersion = "2.0" },
		"oversized text": func(s *TranscriptSegment) {
			s.Text = strings.Repeat("a", MaxTextBytes+1)
			s.Redactions = nil
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			seg := baseSegment()
			mutate(&seg)
			if err := seg.Validate(); err == nil {
				t.Fatalf("expected rejection for %s", name)
			}
		})
	}
}

// TestValidateHandlesMultibyteOffsets guards the rune-vs-byte trap: Python
// offsets index runes, Go indexes bytes by default.
func TestValidateHandlesMultibyteOffsets(t *testing.T) {
	seg := baseSegment()
	seg.Text = "café card <CARD_1> ok"
	seg.Redactions[0].Start = 10
	seg.Redactions[0].End = 18
	if err := seg.Validate(); err != nil {
		t.Fatalf("multibyte offsets should validate by rune, got %v", err)
	}
}

func TestIdempotencyKeySeparatesPartialsFromFinals(t *testing.T) {
	partial := baseSegment()
	partial.IsFinal = false
	final := baseSegment()

	if partial.IdempotencyKey() == final.IdempotencyKey() {
		t.Fatal("a final must not be deduped against its own partial")
	}
	if final.IdempotencyKey() != "call-1:3:f" {
		t.Fatalf("unexpected key %q", final.IdempotencyKey())
	}
}

// TestNoOriginalFieldOnTheWire asserts the struct has nowhere to put a raw
// value, mirroring the pydantic side. If someone adds one, this fails.
func TestNoOriginalFieldOnTheWire(t *testing.T) {
	blob, err := json.Marshal(baseSegment().Redactions[0])
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"original", "raw", "value", "plaintext"} {
		if strings.Contains(strings.ToLower(string(blob)), `"`+forbidden+`"`) {
			t.Fatalf("RedactionRef serialises a %q field; it must not carry originals", forbidden)
		}
	}
}
