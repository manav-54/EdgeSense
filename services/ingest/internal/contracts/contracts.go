// Package contracts mirrors the pydantic wire types in packages/edgesense_core.
//
// The duplication is deliberate and bounded: a Go service cannot import Python
// models, and code-generating them would add a build step for four structs.
// The contract test in contracts_test.go loads the JSON Schema exported from
// the Python side and fails if these structs drift from it, so the duplication
// cannot rot silently.
package contracts

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

const ContractVersion = "1.0"

type Speaker string

const (
	SpeakerAgent    Speaker = "agent"
	SpeakerCustomer Speaker = "customer"
	SpeakerUnknown  Speaker = "unknown"
)

func (s Speaker) Valid() bool {
	switch s {
	case SpeakerAgent, SpeakerCustomer, SpeakerUnknown:
		return true
	}
	return false
}

// RedactionRef points at a placeholder in the redacted text. As on the Python
// side, there is no field for the original value: the struct cannot carry one.
type RedactionRef struct {
	Type        string  `json:"type"`
	Placeholder string  `json:"placeholder"`
	Start       int     `json:"start"`
	End         int     `json:"end"`
	Detector    string  `json:"detector"`
	Confidence  float64 `json:"confidence"`
}

type TranscriptSegment struct {
	SchemaVersion string         `json:"schema_version"`
	CallID        string         `json:"call_id"`
	Seq           int            `json:"seq"`
	Speaker       Speaker        `json:"speaker"`
	Text          string         `json:"text"`
	IsFinal       bool           `json:"is_final"`
	StartMs       int            `json:"start_ms"`
	EndMs         int            `json:"end_ms"`
	EmittedAt     string         `json:"emitted_at"`
	Redactions    []RedactionRef `json:"redactions"`
	ASRConfidence float64        `json:"asr_confidence"`
	AgentID       *string        `json:"agent_id"`
	Traceparent   *string        `json:"traceparent"`
}

// MaxTextBytes bounds a single segment. A streaming ASR segment is a sentence
// or two; anything past this is a bug or an attack, and accepting it would let
// one connection push arbitrary memory through the pipeline.
const MaxTextBytes = 8 * 1024

// MaxRedactions bounds the span list similarly.
const MaxRedactions = 256

// Validate enforces every invariant the producer claims to have upheld.
//
// The edge agent already validated this payload on the way out. Doing it again
// here is not redundant: ingest is a trust boundary, and a segment can arrive
// from a stale edge build, a replayed capture, or something that is not the
// edge agent at all. The span checks in particular are what stop a caller from
// declaring a redaction that does not correspond to any placeholder -- which
// would let raw text through while looking redacted in the portal.
func (s *TranscriptSegment) Validate() error {
	if s.SchemaVersion != "" {
		if major(s.SchemaVersion) != major(ContractVersion) {
			return fmt.Errorf("incompatible schema_version %q (this build speaks %q)",
				s.SchemaVersion, ContractVersion)
		}
	}
	if s.CallID == "" || len(s.CallID) > 64 {
		return fmt.Errorf("call_id must be 1..64 chars, got %d", len(s.CallID))
	}
	if s.Seq < 0 {
		return fmt.Errorf("seq must be non-negative, got %d", s.Seq)
	}
	if !s.Speaker.Valid() {
		return fmt.Errorf("unknown speaker %q", s.Speaker)
	}
	if len(s.Text) > MaxTextBytes {
		return fmt.Errorf("text is %d bytes, limit %d", len(s.Text), MaxTextBytes)
	}
	if !utf8.ValidString(s.Text) {
		return fmt.Errorf("text is not valid UTF-8")
	}
	if s.EndMs < s.StartMs {
		return fmt.Errorf("end_ms %d precedes start_ms %d", s.EndMs, s.StartMs)
	}
	if s.StartMs < 0 {
		return fmt.Errorf("start_ms must be non-negative, got %d", s.StartMs)
	}
	if s.ASRConfidence < 0 || s.ASRConfidence > 1 {
		return fmt.Errorf("asr_confidence %v outside [0,1]", s.ASRConfidence)
	}
	if s.EmittedAt == "" {
		return fmt.Errorf("emitted_at is required")
	}
	if len(s.Redactions) > MaxRedactions {
		return fmt.Errorf("%d redactions, limit %d", len(s.Redactions), MaxRedactions)
	}

	// Offsets index runes on the Python side, so compare against runes here.
	runes := []rune(s.Text)
	for i, r := range s.Redactions {
		if r.Start < 0 || r.End <= r.Start {
			return fmt.Errorf("redaction %d has empty or inverted span [%d,%d)", i, r.Start, r.End)
		}
		if r.End > len(runes) {
			return fmt.Errorf("redaction %d ends at %d, past text length %d", i, r.End, len(runes))
		}
		if got := string(runes[r.Start:r.End]); got != r.Placeholder {
			return fmt.Errorf("redaction %d span holds %q, expected placeholder %q",
				i, got, r.Placeholder)
		}
		if !strings.HasPrefix(r.Placeholder, "<") || !strings.HasSuffix(r.Placeholder, ">") {
			return fmt.Errorf("redaction %d placeholder %q is not <TYPE_N> shaped", i, r.Placeholder)
		}
		if r.Confidence < 0 || r.Confidence > 1 {
			return fmt.Errorf("redaction %d confidence %v outside [0,1]", i, r.Confidence)
		}
	}
	return nil
}

// IdempotencyKey identifies a segment for dedupe.
//
// Partials and finals are keyed separately: a final legitimately follows a
// partial with the same seq, and collapsing them would drop the final.
func (s *TranscriptSegment) IdempotencyKey() string {
	kind := "p"
	if s.IsFinal {
		kind = "f"
	}
	return fmt.Sprintf("%s:%d:%s", s.CallID, s.Seq, kind)
}

func major(v string) string {
	if i := strings.IndexByte(v, '.'); i >= 0 {
		return v[:i]
	}
	return v
}
