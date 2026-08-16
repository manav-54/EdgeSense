# Changelog

All notable changes to EdgeSense. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Repository scaffolding: service layout, environment template, license.
- Shared wire contracts (`edgesense_core`) for every process boundary, with
  `extra="forbid"` and a `RedactionRef` type that has no field for an original
  value. Signals require at least one evidence span.
- Golden corpus: 30 hand-authored scenarios across six categories, expanded to
  48 calls / 446 turns / 110 labelled PII spans. Adversarial coverage includes
  spelled-out digits, paired digit words, disfluent readback, and values split
  across segment boundaries.
- Real audio synthesis from the corpus (macOS `say`, piper, or espeak-ng) at
  16 kHz mono: 48 WAVs, 33 minutes, with turn-boundary manifests.
- Edge agent: 20 ms frame streaming, energy VAD segmentation, local
  faster-whisper transcription, hybrid PII redaction, and a bounded-queue
  WebSocket transport.
- Egress test suite proving no raw PII crosses the network boundary, including
  a test that captures bytes off a real WebSocket server.

### Fixed
- Trailing digits outside a claimed span were emitted in the clear
  (`<CARD_1>84`). Leftover digits inside a run are now absorbed into the
  adjacent claim.
- A held fragment orphaned by an intervening turn ("Go ahead.") was released
  raw. Unclaimed carry is now redacted rather than emitted.
- Spoken emails mangled by ASR (`taconquo at example.com`) escaped detection.
- NER trigger phrases were case-sensitive and missed sentence-initial
  "My name is ...".
