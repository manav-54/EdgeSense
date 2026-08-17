# DESIGN.md — EdgeSense

## Problem

Contact centres want real-time intelligence on live calls: is this customer
about to escalate, did the agent give the required disclosure, what is this
call actually about. The useful version of that runs in the cloud, where the
models and the analytics live.

The obstacle is that call audio is the most sensitive data most businesses
handle. A single call routinely contains a card number, a date of birth, a
social security number and a home address, spoken aloud. Shipping that audio to
a cloud vendor drags the recording store, the transcript store, the model
provider and every log line in between into PCI-DSS and GDPR scope. The
conventional answer — upload everything, redact server-side — does not shrink
that scope at all. It just moves where the incident happens.

**EdgeSense inverts it.** Transcription and PII removal run on the operator's
machine. Only redacted text crosses the network. The cloud never receives a
sample of the customer's voice or a digit of their card, so it is not in scope
for either — not by policy, but because the data is not there.

## Goals

1. **No raw PII crosses the network boundary.** Not "usually", not "after
   filtering downstream". Enforced by types, tested at the byte level over a
   real socket, and measured over an adversarial corpus.
2. **Live signals inside 2 seconds** from a segment being spoken to a signal
   reaching a supervisor's screen (p95).
3. **Every claim is sourced.** No signal, no summary line, no compliance flag
   exists without the transcript span that justifies it.
4. **Measured, not asserted.** A hand-labelled corpus, per-type recall,
   adversarial obfuscation, and a regression harness that fails the build.
5. **Runs with no cloud account.** `docker compose up` produces a working
   pipeline and real eval numbers with zero credentials.

## Non-goals

Stated explicitly, because each one is a thing a reader might reasonably expect
and would otherwise assume was forgotten:

- **Speaker diarisation.** Speaker labels come from the audio manifest. A bad
  diariser corrupts every per-speaker metric downstream, and the corpus knows
  who spoke; guessing badly would be worse than not guessing.
- **Authentication and multi-tenancy.** The ingest WebSocket accepts any client
  that supplies a `call_id`. Real deployment needs mTLS or signed tokens per
  edge agent. This is a demonstrable gap, not an oversight.
- **Encryption at rest, key management, retention enforcement.** ClickHouse
  TTLs are set; nothing else is.
- **Durable worker state.** In-flight window state is in memory. A worker
  crash loses the current window for its calls.
- **Non-English speech.** `tiny.en` is English-only, and the spoken-number
  normaliser encodes English digit words.
- **Real telephony audio.** The corpus is clean 16 kHz TTS. Production audio is
  8 kHz μ-law with crosstalk and hold music.
- **Human-rated summary quality.** Citation validity is measured; whether the
  prose is *good* is not.

---

## Architecture

```
  operator's machine                    │              cloud
════════════════════════════════════════╪══════════════════════════════════════
                                        │
  WAV ──20 ms frames──▶ VAD segmenter   │
                            │           │
                            ▼           │
                     faster-whisper     │      ← audio never crosses this line
                            │           │
                            ▼           │
                    ┌───────────────┐   │
                    │   REDACTOR    │   │
                    │ regex+Luhn    │   │
                    │ digit-stream  │   │
                    │ NER · hold    │   │
                    └───────┬───────┘   │
                            │           │
                    ┌───────▼───────┐   │
                    │   PIIVault    │   │      ← raw values stop here, forever
                    │ in-memory     │   │
                    └───────────────┘   │
                            │           │
                   redacted segments    │
                            └───────────┼──▶ ingest (Go)
                                        │      validate · dedupe · backpressure
                                        │           │
                                        │           ▼
                                        │      Kafka  transcript.segments
                                        │           │
                                        │           ▼
                                        │      worker (Python, agentic)
                                        │       fast path ──┐
                                        │       agent loop ─┴─▶ Kafka call.insights
                                        │                          │
                                        │                          ▼
                                        │                    sink ──▶ ClickHouse
                                        │                          │
                                        │                          ▼
                                        │                    portal (React)
```

### Why the boundary is where it is

The redactor sits between ASR and the transport with nothing in between. There
is no code path from `Transcript.text` to `Transport.send` that skips it. That
is not a convention — it is the thing the egress test suite exists to keep
true as the code changes.

Three mechanisms make the guarantee structural rather than aspirational:

1. **The wire types cannot represent raw PII.** `RedactionRef` records a type, a
   placeholder and offsets. It has no field for an original value, and
   `extra="forbid"` means a caller cannot add one. You cannot leak through a
   field that does not exist.
2. **The vault never serialises.** `PIIVault.__repr__` is overridden, because
   the most plausible leak is a `log.debug("vault=%s", vault)` or a traceback
   rendering locals.
3. **Logging scrubs.** Both Python and Go log formatters strip digit- and
   email-shaped strings. Code should not log transcripts; backstops are what
   keep a guarantee alive through maintenance by someone who has not read this
   document.

### The hardest part: streaming

Single-shot redaction is easy. Streaming is not, and two failures found during
development show why.

**Failure 1 — the split value.** ASR emits "the card number, first part is
4000 0566" and the rest a second later. Redacting each segment independently
ships eight digits of a live card, and no downstream care undoes that: the
bytes are gone. The fix is a **hold buffer** — a trailing digit run that could
still be growing is withheld. It is released when the next segment completes
the value (halves join, Luhn passes, one `<CARD_1>` emitted) or, on expiry,
redacted rather than released.

**Failure 2 — punctuation is not proof.** faster-whisper punctuates
aggressively. A caller reading a card in four groups produces four segments
each rendered `"4242."`. The hold read the full stop as "the number ended" and
released all four. Only running real audio through the real model surfaced
this; authored transcripts never punctuate mid-number. See EVAL.md §2.

The bounded cost is one segment of extra latency on affected segments, and the
2 s budget is measured with the hold enabled.

### Why the agent loop, not one-shot prompting

One-shot prompting is less code: send the transcript, get JSON, parse it. It
was rejected because its failure mode is silent. A model asked for findings in
a single pass produces plausible ones — a policy id it half-remembers, a quote
it paraphrased into something nobody said — and the parser cannot distinguish
those from real findings. By the time the claim exists, the pressure is to
publish it.

The loop inverts that:

- `flag_risk` is a **tool**, not a parsed field. It rejects evidence that does
  not resolve to a real transcript turn, so an unsourced claim becomes a tool
  error the model can correct rather than a published signal.
- `lookup_policy` must return real text before a compliance claim can cite an
  id, so a hallucinated policy fails immediately.
- `search_transcript` makes the model locate the turn rather than recall it.

Result: 100% of emitted signals carry verifiable evidence, structurally.

### Why two paths in the live analyser

The **fast path** is deterministic rules, runs in ~0.3 ms, and cannot fail when
a provider throttles. The **agent path** handles judgement under a deadline
derived from the remaining budget.

This is what makes the 2 s p95 reachable. If a signal that a debt collector
just threatened a customer with court had to wait on a model round-trip, it
would arrive after the moment it mattered. Deterministic guardrails for
obligations you must never miss; model judgement for what rules cannot express.

---

## Three alternatives considered, and why each lost

### 1. Upload audio, redact server-side

**The shape:** stream raw audio to the cloud, run ASR and redaction there,
discard the audio afterwards.

**Why it is tempting:** far simpler. One deployment, GPU-class ASR, no CPU
budget on operator laptops, better transcription quality, and models can be
swapped without touching a client.

**Why it was rejected:** it does not solve the problem. The moment raw audio
reaches the cloud, the ingestion path, the transient buffers, the ASR service
and its logs are all in PCI-DSS and GDPR scope. "We delete it afterwards" is a
retention policy, not a scope reduction — it still has to be audited, and a bug
in the deletion path is a breach. The premise of this system is that the cloud
is *not in scope*, and that only holds if the data never arrives.

The real cost of rejecting it is quality: `tiny.en` on CPU is materially worse
than a large model on a GPU, and every downstream number inherits that. That is
the trade, taken deliberately.

### 2. Client-side format-preserving encryption instead of redaction

**The shape:** encrypt PII at the edge with FPE, ship ciphertext that keeps
shape, decrypt in a controlled enclave when a human genuinely needs the value.

**Why it is tempting:** strictly more capable. Reversible for legitimate
lookups, preserves referential integrity across calls, and analytics can still
join on a stable ciphertext.

**Why it was rejected:** it relocates the problem into key management, which is
harder than the problem. Now there is a key per tenant, rotation, an enclave
with decrypt authority, and an audit trail for every decrypt — and the cloud
holds ciphertext of live card numbers, which is still PCI-relevant data. The
threat model gets worse, not better: a redaction bug leaks one call, whereas a
key compromise leaks the archive.

Redaction with a local-only vault gets most of the benefit — stable
placeholders give cross-call linkage *within* a call — while keeping the
property that there is nothing in the cloud to decrypt.

### 3. One monolithic Python service

**The shape:** collapse ingest, worker and sink into one process. No Kafka, no
Go, no topics.

**Why it is tempting:** dramatically less machinery for this scale. 16
concurrent calls per host does not need a distributed log, and a single process
removes serialisation, partitioning and three deployments.

**Why it was rejected:** the three components have genuinely different scaling
and failure characteristics. Ingest is I/O-bound and must never drop a final
segment. The worker is latency-bound and may call a slow external model. The
sink is throughput-bound and wants large batches. In one process, a slow LLM
call blocks segment acceptance, and the natural fix — internal queues — is a
worse version of Kafka without the durability.

The buffer also has to exist somewhere. Without a broker, a ClickHouse outage
means either dropping insights or growing an unbounded in-memory queue. The
honest note: **for a single-tenant deployment under ~50 concurrent calls, the
monolith is probably the right call**, and this design would be over-built. It
is justified by the multi-tenant target, not by the demo.

---

## Failure modes

| Failure | Behaviour | Rationale |
|---|---|---|
| Redis down | Dedupe **fails open** — segments accepted, possibly duplicated | A duplicate double-counts one row. Dropping live call content because a cache is down loses data that cannot be recovered |
| Kafka slow | Partials shed immediately; finals block the reader, then TCP backpressure reaches the edge; finals shed only after timeout, with an explicit `throttle` message to the client | Pressure should land on the producer, not grow a buffer. A partial is superseded within a second; a final is content |
| Kafka down | Ingest returns throttle; edge retries with backoff | |
| LLM provider throttled | Agent path blows its deadline; **fast-path signals still publish** | Compliance detection must not depend on a vendor's capacity |
| LLM returns invalid JSON | Up to 2 repairs with the validator's own error text; then **publish nothing** | A partially-correct summary looks authoritative and corrupts every aggregate built on it |
| Worker crashes mid-call | In-flight window lost; call resumes with empty history. Partition-by-`call_id` bounds damage to that call | Accepted cost of in-memory windows. See 10,000× below |
| ClickHouse down | Sink retries with backoff, rows requeue; buffer sheds oldest at 50k with a counter | Unbounded growth kills the process and loses everything, not just the oldest |
| OTel collector down | No-op tracer; pipeline unaffected | A monitoring outage must not become a processing outage |
| Edge hold never resolves | Fragment **redacted**, not released | A withheld fragment was withheld because it looked like a secret |
| ASR mangles a card | Luhn fails; context tier still redacts; unclaimed 9+ digit runs redact as ACCOUNT | Recall bias. Costs precision, measured in EVAL.md |
| Redactor throws | **No fallback.** The segment is not emitted | Failing closed is the only safe direction |

### The failure this design does not handle

**A redaction bug is invisible from the cloud.** By construction, the server
side cannot tell the difference between "no PII in this call" and "the redactor
broke". There is no server-side leak detector, because a server-side detector
would need to see the raw value to know it leaked.

The mitigations are all indirect: the egress test suite in CI, the corpus eval
with hard gates, and a Prometheus alert (`RedactionRateCollapsed`) that fires
when segments flow but redaction counts hit zero. That alert watches the
*shadow* of the bug, not the bug. It is the weakest part of the design and the
first thing to strengthen — probably with a canary call, injected on a
schedule, containing known synthetic PII, asserted server-side to have arrived
redacted.

---

## What I would do differently at 10,000×

At ~160,000 concurrent calls (10,000× the measured 16 per host), most of this
design is wrong in specific, predictable ways.

**1. ASR stops being a per-host CPU problem and becomes the entire cost model.**
At 16 calls/host, 160k calls is 10,000 hosts. Nobody buys that. The edge agent
would move to a quantised streaming model with a real streaming architecture
(transducer rather than re-decoding a growing window), and where hardware
allows, onto the NPU that ships in modern laptops. Re-decoding the whole
utterance for every partial — acceptable here — is the first thing to go.

**2. The worker's in-memory window state becomes unacceptable.** Losing a
window per crash is fine at demo scale and a daily incident at 160k calls.
Window state moves to a keyed state store with changelog-backed recovery
(Flink/Kafka Streams semantics), so a rebalance restores state instead of
discarding it.

**3. The LLM becomes the budget, and the current design has no answer.** At
160k concurrent calls, an agent loop per window is financially absurd. I would:
cascade — a small classifier decides whether a window is *interesting* and only
those reach the large model; batch aggressively across calls; and cache
aggressively on near-duplicate windows, since contact-centre conversations are
extraordinarily repetitive. The `signals_hourly` rollup already suggests the
shape: most windows produce nothing new.

**4. One Kafka topic per stage stops working.** `transcript.segments`
partitioned by `call_id` is right, but at this volume it needs partition counts
in the thousands, tiered storage for the long tail, and a separate
high-priority topic for compliance-critical segments so a backlog of routine
traffic cannot delay a mini-Miranda violation.

**5. ClickHouse schema survives, mostly.** This is the part that scales best —
the sort keys are already chosen for granule skipping (see
`docs/clickhouse-explain.md`: 136 granules → 18). At 10,000× I would add
distributed tables with sharding on `call_id`, move the materialised views to
compute on a separate replica so ingest is not competing with dashboard
aggregation, and drop `segment_latency` retention from 30 days to about 3 —
it is the highest-volume, lowest-value table.

**6. The eval harness needs to become continuous.** A 48-call corpus run on
demand is right for one engineer. At scale it becomes a sampled shadow eval on
production traffic — with human review of a stratified sample — because the
corpus cannot anticipate the distribution shift that 160k calls of real
customers produce. The golden corpus stays as the regression gate; it stops
being the measurement of record.

**7. Redaction gets a second opinion.** A single redactor is a single point of
failure for the entire privacy claim. At this scale I would run two independent
implementations — the current regex/checksum path and a separately-trained
model — and redact the union. Disagreement between them is a high-value signal
for finding the cases neither corpus anticipated.

---

## Things I am not happy with

- **`VERIF-004` precision is 28.6%.** Phrase-matching cannot distinguish "did
  not verify" from "verified using words not in my list". It needs to be a
  model judgement, not a rule.
- **The offline provider makes citation validity look better than it is.** A
  rules engine quoting spans it selected cannot hallucinate. The metric is
  built for a model that can, and has not been tested against one.
- **Two eval measurement modes is one more than ideal.** It is honest — span
  metrics genuinely cannot survive ASR — but it means "recall" refers to
  different things in different tables, and I had to label that carefully
  rather than fix it.
- **The corpus is 30 scenarios wearing 48 hats.** Second fills add value for
  PII spans and none for judgement labels.
- **Speaker labels are cheated.** Honest about it, but every per-speaker number
  is downstream of a manifest rather than a diariser.
