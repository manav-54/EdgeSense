# Changelog

All notable changes to EdgeSense. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Commits are incremental and reviewable; each phase below corresponds to one
commit whose message explains the reasoning, not just the change.

## [0.1.0] — 2026-08-17

### Contracts and corpus

- Shared wire contracts (`edgesense_core`) for every process boundary, with
  `extra="forbid"` and a `RedactionRef` type that has **no field** for an
  original value — privacy enforced by the type system rather than by reviewer
  discipline. Signals require at least one evidence span, so an unsourced claim
  fails validation instead of being published.
- Golden corpus: 30 hand-authored scenarios across six categories, expanded to
  48 calls / 446 turns / 110 labelled PII spans. Spans are recorded at
  substitution time, so ground truth is exact rather than annotated.
  Adversarial coverage includes spelled-out digits, paired digit words,
  disfluent readback, and values split across segment boundaries.
- Real audio synthesis (macOS `say`, piper, or espeak-ng) at 16 kHz mono: 48
  WAVs, 33 minutes, with turn-boundary manifests. Numbers are re-spaced for
  speech so a TTS engine reads a card the way a caller does, not as "four
  quadrillion".

### Edge agent

- Hybrid PII detection: compiled regex and Luhn for anything numeric, a small
  NER pass only for names and street addresses.
- Spoken-number normalisation with preserved character alignment. Digits are
  extracted into a stream whether they arrived as numerals or words, so
  "four two four two", "forty two forty two" and "double four" are all
  detectable, and redaction still happens in place rather than rewriting the
  transcript.
- Streaming hold buffer: a trailing digit run that could still be growing is
  withheld rather than emitted, because a retraction over the network is a
  fiction. A card split across segments joins, validates against Luhn, and
  emits as one `<CARD_1>`.
- 20 ms frame streaming, energy VAD segmentation, faster-whisper transcription,
  bounded-queue WebSocket transport.
- Egress test suite escalating from redactor output, to encoded wire bytes, to
  bytes read off a **real WebSocket server**, plus a corpus-independent shape
  invariant and a check that the contract itself cannot model a raw value.

### Ingest (Go)

- WebSocket server with server-side re-validation, Redis idempotency, and
  asymmetric backpressure: partials shed immediately, finals block the reader
  (propagating into TCP backpressure), and a shed final sends the client an
  explicit throttle message.
- Dedupe fails open — a duplicate double-counts one row, while dropping
  segments because a cache is down loses live content that cannot be recovered.
- Cross-language contract test validating 106 segments produced by the actual
  Python redactor, so Go/pydantic drift fails CI rather than production.

### Worker

- Tool-using agent loop where `flag_risk` rejects evidence that does not
  resolve to a real transcript turn, and a compliance claim requires
  `lookup_policy` to have returned the real text first.
- Two live paths: deterministic rules (~0.3 ms, cannot fail when a provider
  throttles) and a bounded agent loop for judgement. Signals dedupe across
  windows on change rather than repeating.
- Post-call summaries validated against the strict schema with up to two
  repairs, feeding the validator's own error text back to the model. On final
  failure the worker publishes nothing.
- Offline provider implements the same tool protocol rather than stubbing it,
  so the loop runs by default and the eval has a real baseline. Azure OpenAI
  drives generation and Azure AI Search backs policy retrieval when configured.

### Analytics

- ClickHouse schema validated against a live server, with sort keys chosen for
  granule skipping and the choice **verified**: on 4M rows the chosen key
  narrows 136 granules to 18, while a time-first key leaves the primary index
  contributing nothing.
- Batched sink with requeue-on-failure, five analytical queries, and a script
  that captures real `EXPLAIN` output into `docs/clickhouse-explain.md`.

### Portal

- Live call view with streaming transcript, real-time signals, and risk badges
  that reveal their transcript evidence on hover.
- Supervisor dashboard and a latency panel showing p50/p95/p99 per stage
  against the 2 s budget.
- Chart colours drawn from a validated palette — the categorical pair and the
  ordinal latency ramp were both run through a CVD/contrast validator in light
  and dark mode. Severity badges pair a glyph and the severity word with the
  colour so meaning survives colour blindness and greyscale.

### Evaluation

- Redaction scored at three levels: leak rate (the safety metric), span
  coverage, and type accuracy. Headline is F2, not F1 — a missed card and an
  over-redacted order number are not equally bad.
- Prompt regression suite with hard gates on leak rate, redaction recall,
  citation validity and evidence rate.
- Two measurement modes (text and audio), reported separately because they
  answer different questions.

### Fixed

- **Trailing digits outside a claimed span were emitted in the clear**
  (`<CARD_1>84`). ASR digit errors broke Luhn and the length rules, so a
  15-digit Amex was claimed as a 10-digit phone and the remainder shipped raw.
  Leftover digits inside a run are now absorbed into the adjacent claim.
- **A held fragment orphaned by an intervening turn was released raw.** When
  the agent said "Go ahead." between two halves of a card, the hold released
  and eight digits went out. Unclaimed carry is now redacted.
- **Punctuation was treated as proof a spoken number ended.** faster-whisper
  renders a chunked card readback as four segments each ending `"4242."`; the
  hold released every group and the card was reconstructible from consecutive
  segments. Found only by running real audio through the real model — authored
  transcripts never punctuate mid-number. Audio-mode leak rate 1.84% → 0.00%.
- **Partial segments leaked, and the test suite only covered finals.** Running
  the pipeline with previews enabled — the default — shipped six digits of a
  card (`"It's 520808."`) because the preview path used a scratch redactor
  that did not inherit the accumulated context, and shipped truncated emails
  (`"...at priya.nair@e"`) because a half-arrived address matches no pattern.
  Partials now withhold any unclaimed trailing digit run or in-progress email
  outright: a preview is superseded within a second, so holding costs nothing
  and releasing is unrecoverable. The egress suite now re-derives partials
  against progressively longer prefixes of every turn.
- Spoken emails mangled by ASR (`taconquo at example.com`) escaped detection.
- NER trigger phrases were case-sensitive and missed sentence-initial
  "My name is ...".
- `avgState(Float32 - Float32)` produced a Float64 aggregate state the column
  could not accept — caught by applying the schema to a real server.
- SQL alias shadowing in two dashboard queries: `countIf(...) AS escalated`
  made the next reference an aggregate-inside-an-aggregate, and
  `uniqMerge(calls) AS calls` made a later `uniqMerge` see a UInt64.
- `"intent in [collections]".split("in", 1)` split inside the word "intent", so
  intent-scoped policies (MINI-003, RTC-007) never applied to any call.
- The read API shared one ClickHouse client across FastAPI's threadpool,
  producing intermittent 500s under the dashboard's concurrent panel loads that
  never reproduced under sequential requests. Clients are now per-thread.
- `Path("") or default` silently resolved the prompt directory to the working
  directory, because `Path("")` is truthy.

### Eval methodology corrections

Two measurement bugs found while building the harness, both of which were
understating the system:

- Span coverage was measured without the carried context the streaming path
  actually has, so values the shipped pipeline catches scored as misses.
- Coverage required exact containment, so redacting `ACCT-1029384`'s digits
  while deliberately leaving `ACCT-` readable scored as a miss.

Coverage is now digit-complete for numeric spans and 90% character coverage for
names and addresses. Reported recall moved 82.6% → 98.2% with no change to the
redactor: the measurement was wrong, not the system.
