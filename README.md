# EdgeSense

Privacy-first, real-time conversation intelligence. Transcription and PII
removal run on the operator's machine; only redacted, structured data reaches
the cloud.

The cloud never receives a sample of the customer's voice or a digit of their
card — not by policy, but because the data is not sent.

```
zero PII leaks across 109 labelled spans, in text mode and through real ASR
100% of published signals carry a verifiable transcript quote
16 concurrent calls per host inside a 2 s p95 budget
```

Full numbers, including where it fails: **[EVAL.md](EVAL.md)**.
Rationale, alternatives and failure modes: **[DESIGN.md](DESIGN.md)**.

---

## Architecture

```
   OPERATOR'S MACHINE                    │                 CLOUD
 ═══════════════════════════════════════ │ ════════════════════════════════════
                                         │
  ┌──────────┐   20 ms frames            │
  │   WAV    │──────────────┐            │
  │  audio   │              │            │
  └──────────┘              ▼            │
                     ┌─────────────┐     │
                     │ VAD         │     │
                     │ segmenter   │     │
                     └──────┬──────┘     │
                            │ utterance  │
                            ▼            │
                     ┌─────────────┐     │      ╔═══════════════════════════╗
                     │faster-whisper│    │      ║  audio never crosses      ║
                     │  tiny.en CPU │    │      ║  this line                ║
                     └──────┬──────┘     │      ╚═══════════════════════════╝
                            │ text       │
                            ▼            │
              ┌──────────────────────────┐
              │        REDACTOR          │
              │  regex + Luhn checksum   │
              │  spoken-digit normaliser │
              │  small NER (names/addr)  │
              │  cross-segment hold      │
              └───┬──────────────────┬───┘
                  │                  │      │
        ┌─────────▼────────┐   redacted     │
        │    PIIVault      │   segments     │
        │  in-memory only  │        │       │
        │  never sent      │        └───────┼──▶ ┌──────────────┐
        └──────────────────┘                │    │ INGEST (Go)  │
         raw values stop here               │    │ validate     │
                                            │    │ dedupe/Redis │
                                            │    │ backpressure │
                                            │    └──────┬───────┘
                                            │           ▼
                                            │    ╔══════════════╗
                                            │    ║    Kafka     ║
                                            │    ║ transcript.  ║
                                            │    ║   segments   ║
                                            │    ╚══════┬═══════╝
                                            │           ▼
                                            │    ┌──────────────────────┐
                                            │    │  WORKER (agentic)    │
                                            │    │ ┌──────────────────┐ │
                                            │    │ │ fast path ~0.3ms │ │
                                            │    │ │ rules: policy,   │ │
                                            │    │ │ escalation       │ │
                                            │    │ ├──────────────────┤ │
                                            │    │ │ agent loop       │ │
                                            │    │ │ lookup_policy    │ │
                                            │    │ │ search_transcript│ │
                                            │    │ │ flag_risk        │ │
                                            │    │ └──────────────────┘ │
                                            │    └──────────┬───────────┘
                                            │               ▼
                                            │        ╔══════════════╗
                                            │        ║    Kafka     ║
                                            │        ║ call.insights║
                                            │        ╚══════┬═══════╝
                                            │               ▼
                                            │        ┌──────────────┐
                                            │        │  SINK + API  │
                                            │        └──────┬───────┘
                                            │               ▼
                                            │        ┌──────────────┐    ┌────────┐
                                            │        │  ClickHouse  │◀───│ PORTAL │
                                            │        └──────────────┘    │ React  │
                                            │                            └────────┘
                                            │
        OpenTelemetry: one call_id traced edge → ingest → worker → store
```

**Five services**, plus the observability stack:

| Service | Language | Responsibility |
|---|---|---|
| **edge-agent** | Python | Frame streaming, local ASR, PII redaction, transport |
| **ingest** | Go | WebSocket termination, validation, dedupe, Kafka publish |
| **worker** | Python | Agentic live analysis + post-call summaries |
| **sink** | Python | Kafka → ClickHouse, plus the portal's read API |
| **portal** | React + TS | Live call view, supervisor dashboard, latency panel |

---

## Quick start

```bash
make install          # Python venv + all service deps + spaCy model
make corpus           # 48 labelled synthetic calls
make audio            # synthesise real audio (33 min of WAVs)
make test             # egress suite + Go contract tests
make eval             # full evaluation, prints the table in EVAL.md
```

The full stack:

```bash
make demo             # compose up, generate audio, stream a real call
# portal   http://localhost:5173
# grafana  http://localhost:3000
# jaeger   http://localhost:16686
```

No cloud credentials are needed. Without Azure the worker uses a deterministic
offline provider that walks the same tool-calling loop, so the pipeline and the
eval both run end to end. To use Azure OpenAI, copy `.env.example` to `.env`
and fill in the Azure block.

> **Verification status.** Everything above the compose line was executed on the
> host described in EVAL.md: real audio, real faster-whisper, real ClickHouse
> (schema applied and queried, plans captured in
> [docs/clickhouse-explain.md](docs/clickhouse-explain.md)), real Go build and
> tests, portal rendered and screenshotted. Docker was not available on this
> machine, so `docker-compose.yml` and the Dockerfiles are written and
> YAML-validated but have not been executed.

---

## How a call flows through the system

Follow one real call — `gold-pii_fraud_report_full_profile-v0`, a customer
reporting fraud who reads out a card number, a date of birth and a social
security number.

### 1. Audio becomes frames (edge)

`stream_wav` reads the 16 kHz mono WAV and yields **20 ms frames**, paced to
wall clock. Pacing uses an absolute schedule rather than `sleep(20ms)` per
frame, so processing time inside the loop does not accumulate into drift.

An energy-based VAD accumulates frames into an utterance and closes it after
240 ms of silence.

### 2. Utterance becomes text (edge)

The closed utterance goes to faster-whisper `tiny.en`, int8, on CPU. This is
where the latency budget goes: **ASR p95 is 1105 ms**, everything else together
is under 6 ms.

The model produces, for one turn:

```
"Also my social is 9 0 0 4 5 6 7 8 9 if you need it for the report."
```

### 3. Text becomes safe (edge) — the important step

The redactor runs three families of detector:

- **Surface regex** — formatted emails, `xxx-xx-xxxx` SSNs, dates, labelled
  account numbers.
- **Digit-stream scanning** — every digit in the utterance, whether it arrived
  as a numeral or a word, is extracted into an ordered stream where each digit
  remembers its character range. `"four two four two"`, `"forty two forty
  two"`, `"double four"` and `"4242"` all normalise to the same digits, so a
  card spoken aloud is as detectable as one typed. Luhn validates; failing Luhn
  drops to a context tier rather than dismissing the candidate, because ASR
  digit errors break checksums on genuine cards.
- **A small NER pass** — names and street addresses only, where regex has
  nothing to match on.

Then the **hold buffer** decides what is not safe to emit yet. If the utterance
ends on a digit run that could still be growing, that run is withheld and
carried into the next segment. This is what stops a card split across two
segments from shipping as two clear halves.

The raw value goes into the `PIIVault` — in memory, per call, never serialised,
with `__repr__` overridden so a stray log line or traceback cannot render it.

What leaves the process:

```json
{
  "call_id": "gold-pii_fraud_report_full_profile-v0",
  "seq": 14, "speaker": "customer", "is_final": true,
  "text": "Also my social is <SSN_1> if you need it for the report.",
  "redactions": [{
    "type": "SSN", "placeholder": "<SSN_1>", "start": 18, "end": 25,
    "detector": "regex", "confidence": 0.97
  }],
  "start_ms": 48200, "end_ms": 52100,
  "traceparent": "00-4bf92f35...-00f067aa...-01"
}
```

Note what is absent: there is no field for the original value. `RedactionRef`
has no such slot, and `extra="forbid"` means one cannot be added.

### 4. Segment crosses the boundary (edge → ingest)

The transport re-validates the segment immediately before serialisation, then
sends it over a WebSocket with a bounded outbound queue.

Ingest **does not trust it**. It re-runs the same validation server-side,
because a segment can arrive from a stale build, a replayed capture, or
something that is not the edge agent at all. The check that matters: every
redaction span must actually contain its placeholder in the text. A client that
claims a redaction while leaving the raw value in place is rejected — otherwise
the portal would draw a redaction badge over readable PII.

Then dedupe on `(call_id, seq, final|partial)` via Redis — failing *open*, so a
cache outage does not drop live call content — and publish to Kafka,
partitioned by `call_id` so one conversation stays in order on one partition.

### 5. Segments become signals (worker)

The worker keeps a per-call sliding window of the last 6 final turns and
analyses every 2 new turns. Two paths run over each window:

**Fast path** — deterministic rules, ~0.3 ms. Prohibited phrases, missing
required disclosures, escalation patterns. On this call it finds nothing; on a
collections call it flags a mini-Miranda omission before any model is consulted.

**Agent path** — a bounded tool loop:

```
system: you analyse live calls... every flag must cite evidence_turns
user:   window turns 12-17 ... transcript ...
  ├─ lookup_policy("REC-001")        → full policy text
  ├─ search_transcript("recorded")   → turn 0, agent, 0:00-0:09
  └─ flag_risk(type=intent, label=fraud_report, evidence_turns=[3])
                                     → {"flagged": true, evidence_turns: [3]}
assistant: DONE
```

`flag_risk` **rejects** evidence that does not resolve to a real turn. An
unsourced claim becomes a tool error the model can correct, not a published
signal. That is why 100% of emitted signals carry a verifiable quote.

Signals dedupe across windows on *change* rather than repeating — an escalation
badge re-firing every two turns trains supervisors to ignore the panel.

### 6. Signals become analytics (sink → ClickHouse)

The sink batches rows and writes to three tables. The sort keys are chosen for
granule skipping, and that choice is verified rather than asserted: on a 4M-row
table, `(signal_type, emitted_date, agent_id, call_id, emitted_at)` narrows
**136 granules to 18**, while the natural-looking `(emitted_at, call_id)`
leaves the primary index unable to contribute at all. Captured plans:
[docs/clickhouse-explain.md](docs/clickhouse-explain.md).

### 7. Analytics become a screen (portal)

The supervisor sees the transcript with `<SSN_1>` styled as a redaction — not
hidden, because "a card number was spoken and removed" is different from "the
customer never said one", and there is nothing to reveal on click since the
value never left the operator's machine.

Each signal is a risk badge. **Hovering shows the transcript span that justifies
it** — speaker, turn, timestamp and the exact quote. An unsourced badge is an
accusation; this is how a supervisor checks before acting.

---

## Repository layout

```
packages/edgesense_core/   shared contracts (pydantic) + logging/tracing
services/edge-agent/       ASR, redaction, transport, egress tests
services/ingest/           Go WebSocket server, validation, Kafka
services/worker/           agent loop, rules engine, prompts (versioned)
services/sink/             ClickHouse writer + portal read API
portal/                    React + TypeScript
tools/corpus/              scenario templates, PII pools, policy catalog
tools/audio/               TTS synthesis with speech-form normalisation
eval/harness/              metrics, redaction/classification eval, regression
deploy/                    schema, queries, compose, otel, prometheus, grafana
scripts/                   seeding, load test, EXPLAIN report
```

## Key commands

| Command | What it does |
|---|---|
| `make test` | Egress suite (no PII crosses the wire) + Go contract tests |
| `make eval` | Full text-mode evaluation |
| `make eval-audio` | Full evaluation through real ASR |
| `make eval-adversarial` | Just the obfuscated-PII set |
| `make eval-compare BEFORE=v1 AFTER=v2` | Prompt regression with gates |
| `make loadtest` | Ramp concurrency until p95 breaches budget |
| `make explain` | Regenerate query plans from a live ClickHouse |
| `make demo` | Full stack + a real call streamed through it |

## Data note

Every value in the corpus is fabricated: public test card numbers, 555
fictional phone exchanges, `example.com` emails, and SSN area numbers the SSA
never issued. Nothing corresponds to a real person.

## Licence

MIT — see [LICENSE](LICENSE).
