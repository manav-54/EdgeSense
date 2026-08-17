# EVAL.md — what the numbers actually are

Every figure here was produced by `make eval` and `make eval-audio` on the
committed corpus, on the host described below. Nothing is estimated. Where the
system is bad, the number is here with the reason next to it.

**Reproduce:**

```bash
make corpus && make audio      # 48 labelled calls, 33 min of real audio
make eval                      # text mode  -> eval/reports/run-text.json
make eval-audio                # audio mode -> eval/reports/run-audio.json
make loadtest                  # capacity   -> eval/reports/loadtest-realtime.json
```

**Host:** Apple M5, 10 cores, 16 GB. **ASR:** faster-whisper `tiny.en`, int8, CPU.
**Provider:** deterministic offline provider (no Azure credentials were
available for this run — see [Provider caveat](#provider-caveat), which is the
single biggest limitation of everything below).

---

## 1. The golden corpus

48 calls, 446 turns, 110 labelled PII spans, from 30 hand-authored scenarios.

| Category | Calls | What it is for |
|---|---|---|
| `pii_heavy` | 10 | Every PII type, cleanly spoken |
| `compliance` | 14 | One specific policy breached per call |
| `adversarial` | 12 | Obfuscated PII — the hard set |
| `clean` | 4 | No PII, no violations. Catches over-firing |
| `escalation` | 4 | Including one that de-escalates |
| `ambiguous` | 4 | Sarcasm, third-party PII, near-miss numbers |

PII spans by type: CARD 26, DOB 18, ACCOUNT 14, PHONE 14, SSN 12, PERSON 11,
EMAIL 10, ADDRESS 4, plus one deliberate non-PII lookalike.

By surface form — how the value was *spoken*: `spaced` 32, `hyphenated` 30,
`plain` 16, `noisy` 14 (disfluencies between digit groups), `paired` 10
("forty two forty two"), `spelled` 8 ("four two four two").

### How the labels were made, and why that matters

Labels are **exact by construction, not annotated after the fact**. The
generator substitutes a fabricated value into an authored template and records
the character offsets at substitution time.

- **Strength** for PII spans: zero annotator drift. A span is right by
  construction.
- **Weakness** for judgement labels — intent, resolution, sentiment,
  escalation band. These are one author's opinion, written before seeing any
  system output. A careful human rater would disagree on some. Every
  classification number below inherits that, and the resolution numbers in
  particular should be read as "agrees with the author" rather than "correct".

The corpus is 30 distinct scenarios expanded to 48 calls by giving
PII-heavy/compliance/adversarial scenarios a second fill. Two calls from one
scenario share dialogue and differ only in the fabricated values, so the
effective diversity is 30, not 48.

---

## 2. Redaction — the number that matters

Two modes, measuring different things. Read both.

### Text mode — exact spans, no ASR noise

| Metric | Value |
|---|---|
| **Leaks** | **0 / 109** (0.000%) |
| Span recall | 98.17% |
| Precision | 95.54% |
| **F2** (recall-weighted) | **0.976** |
| F1 | 0.968 |
| Type accuracy | 72.9% |
| False positives | 5 |

Headline is F2, not F1. F1 treats a missed card and an over-redacted order
number as equally bad; one is a disclosure incident and the other is a support
ticket.

**By PII type** (n, recall, precision):

| Type | n | Recall | Precision |
|---|---|---|---|
| CARD | 26 | 100.0% | 96.3% |
| DOB | 18 | 100.0% | 100.0% |
| ACCOUNT | 14 | 100.0% | 93.3% |
| PHONE | 14 | 100.0% | 100.0% |
| EMAIL | 10 | 100.0% | 100.0% |
| PERSON | 11 | 100.0% | 84.6% |
| ADDRESS | 4 | 100.0% | 80.0% |
| **SSN** | 12 | **83.3%** | 100.0% |

**By surface form** — the adversarial result:

| Form | n | Recall |
|---|---|---|
| spelled ("four two four two") | 8 | 100.0% |
| paired ("forty two forty two") | 10 | 100.0% |
| noisy ("4242, uh, 4242") | 14 | 100.0% |
| plain / hyphenated | 46 | 100.0% |
| **spaced** | 31 | **93.5%** |

Obfuscation is not what breaks this system. Every fully-spelled, paired and
disfluent value was caught. The 6.5% gap in `spaced` is entirely the
split-across-turns SSN (below).

### Audio mode — the full path, real faster-whisper

Exact spans do not survive ASR, so this measures **catch rate** (did the
labelled value survive anywhere in the transmitted output) rather than span
overlap.

| Metric | Value |
|---|---|
| **Leaks** | **0 / 109** (0.000%) |
| Catch rate, every type | 100.0% |
| Catch rate, every surface form | 100.0% |

Audio mode does not report type accuracy or detector attribution — with no
labelled offsets there is nothing to attribute against.

### The bug audio mode found that text mode could not

Audio mode initially leaked **2 values (1.84%)**. The cause was worth the whole
exercise: faster-whisper punctuates aggressively, so a caller reading
`4242 4242 4242 4242` comes back as four separate segments, each rendered
`"4242."` with a full stop. The hold buffer treated the full stop as "the
speaker finished", released all four groups, and the card was reconstructible
from consecutive segments.

Authored transcripts never punctuate mid-number, so **no amount of text-mode
testing would have surfaced this**. It is the strongest argument in this repo
for running real audio through the real model.

Fixed by (a) not releasing a short unclaimed digit run on punctuation when an
identifier word is in scope, (b) allowing the hold to span four segments, and
(c) running the expiry path through the same orphan protection as the streaming
path — it previously released a four-digit tail raw.

### The second thing real audio found: partial segments

The egress suite originally exercised **finals only**. Running the pipeline
with partials enabled — the default, since the portal streams previews — leaked
two more classes immediately:

- **Six digits of a card in a preview.** The preview path built a scratch
  redactor that did not inherit the accumulated context, so `"It's 520808."`
  had no idea a card readback was in progress and looked like an ordinary
  number.
- **Truncated emails.** A partial cut mid-address (`"...email me at
  priya.nair@e"`) matches no email pattern, because the TLD has not arrived,
  so the local part and half the domain shipped in the clear.

Both are now held: a partial withholds any unclaimed trailing digit run or
in-progress email outright, skipping the release heuristics entirely. A partial
is superseded within about a second, so withholding costs nothing and releasing
is unrecoverable.

The suite now re-derives partials the way the pipeline does — against
progressively longer prefixes of every turn, which is what an ASR revising its
hypothesis actually produces — across all PII-heavy and adversarial calls.
13 tests, all passing. Re-verified on real audio: 50 segments (10 final, 40
partial), zero digit runs and zero email fragments surviving.

### Where redaction still fails

**SSN split across segments (2 of 12 SSN spans, text mode).** The corpus
scenario `adversarial_split_across_turns` breaks a 9-digit SSN into `"912 8"`
and `"8 7766"` across two turns with an agent turn between them. The value is
redacted — it does not leak — but as two `ACCOUNT` fragments rather than one
`SSN`. The hold catches the card in the same scenario because 16 digits is
unambiguous; 9 digits split 4/5 is not.

**Type confusion: 72.9%.** 27% of caught values get the wrong placeholder
label, mostly `SSN`/`DOB`/`ACCOUNT` collapsing into `ACCOUNT` when the context
word landed in a different segment. This is deliberate policy: the value is
gone either way, and counting mislabels as failures would push the system
toward fewer, more confident detections.

**The precision cost of the recall bias — all 5 false positives:**

| What was over-redacted | Why |
|---|---|
| `9999 8888 7777 6666` (order number) | 16 digits, Luhn-invalid. Redacted as CARD anyway |
| `one two three four five six` (confirmation code) | 6-digit run near "confirmation" |
| `the Fenwick Road` (branch name, ×2) | Street-suffix pattern with no house number |
| `Email` (capitalised mid-sentence) | Capitalised-run name heuristic |

The first is the recall bias working exactly as designed and is the reason
`ambiguous_near_miss_numbers` is in the corpus. The last is a genuine bug in
the capitalised-run heuristic.

**Not measured:** non-English speech, overlapping speakers, telephony codecs,
background noise. The corpus is clean TTS audio. Real contact-centre audio is
8 kHz μ-law with crosstalk, and every number above would be worse on it.

---

## 3. Classification and summary quality

Text mode, 48 calls.

| Metric | Value | Note |
|---|---|---|
| **Citation validity** | **100.0%** (138 quotes) | No hallucinated evidence |
| **Signals with evidence** | **100.0%** (106 signals) | Structurally enforced |
| Summary schema valid | 100.0% | 0 needed repair |
| Disclosure detection F1 | 1.000 | |
| Violation recall | 100.0% | |
| Violation precision | 80.0% | One policy is the whole gap |
| Sentiment direction | 87.5% | mean abs error 0.310 |
| Primary intent | 79.2% | |
| **Resolution** | **64.6%** | Weakest number here |
| **Escalation band** | **56.2%** | Weakest number here |
| Escalated flag F1 | 0.500 | P 100%, R 33% |
| Action item recall | 30.9% | |

### Compliance by policy

| Policy | n | Recall | Precision |
|---|---|---|---|
| REC-001 recording disclosure | 4 | 100% | 100% |
| PCI-002 full PAN readback | 4 | 100% | 100% |
| PROHIB-005 outcome guarantee | 4 | 100% | 100% |
| MINI-003 mini-Miranda | 2 | 100% | 100% |
| FDCPA-006 threat | 2 | 100% | 100% |
| RTC-007 right to cancel | 2 | 100% | 100% |
| **VERIF-004 identity verification** | 2 | 100% | **28.6%** |

VERIF-004 fires on five calls that did verify identity, because the check is
phrase-matching over agent turns and a call that verifies using wording outside
the phrase list looks identical to one that skipped it. It is the single
biggest false-positive source in the system.

### Where classification fails

**Resolution, 64.6%** — 12 of 17 errors are `resolved → unresolved`. The rules
classifier looks for closing markers in the last five turns; calls that end
with the customer satisfied but no explicit "that's all" get scored unresolved.
This is a lexical classifier doing a job that needs discourse understanding.

**Escalation band, 56.2%** — the band is a four-way ordinal
(none/low/medium/high) and most errors are off-by-one. The binary escalated
flag has perfect precision and 33% recall: it only fires on an explicit
supervisor request, so a call escalating toward a regulator complaint is missed.

**Sarcasm is not detected.** `ambiguous_sarcastic_satisfaction` — "Fantastic.
Truly a world class experience." after a six-day delay — scores positive. The
lexicon has no way to see it. It is in the corpus specifically so this failure
has a number attached rather than being a footnote.

**Action item recall, 30.9%** — pattern-matched commitments catch explicit
"I'll ..." constructions and miss commitments phrased as statements of fact
("A replacement card ships today").

---

## 4. Prompt regression suite

`make eval-compare BEFORE=v1 AFTER=v2` runs the full corpus twice and prints a
before/after table. Four metrics are **gates** that fail the build on
regression regardless of what improved: leak rate, redaction recall, citation
validity, evidence rate. There is no threshold at which more leaks is a
worthwhile trade.

### It detects real changes

Comparing NER backends (`--ner heuristic` vs `--ner spacy`), a genuine
behavioural difference:

```
metric                                 before        after        delta
--------------------------------------------------------------------------
PII leak rate (gate)                   0.000%       0.000%     +0.000pp     =
Redaction recall (gate)                98.17%       98.17%     +0.000pp     =
Redaction precision                    98.17%       95.54%     -2.630pp     ▼
Redaction F2 (recall-weighted)          0.982        0.976       -0.005     ▼
Citation validity (gate)              100.00%      100.00%     +0.000pp     =
...
0 improved, 2 regressed, 16 unchanged
```

**Finding: spaCy `en_core_web_sm` costs 2.63pp of precision for zero recall
gain on this corpus.** The dependency-free heuristic is strictly better here.
spaCy remains the default only because the heuristic is a capitalisation rule
that will not survive contact with real transcripts; on this corpus the model
earns nothing.

### What it cannot currently detect

Comparing `live_analysis` v1 → v2 produces **all zeros**, because the offline
provider is rule-driven and never reads the prompt. The machinery is exercised
end to end and the gates work; the *prompt* dimension is untested until an LLM
provider is configured. This is stated rather than papered over.

---

## 5. Latency and capacity

Measured by `make loadtest`, ramping concurrency until p95 segment-to-signal
breaches the 2 s budget. The clock starts when an utterance closes.

**Realtime-paced** (calls arrive at wall-clock speed, as in production):

| Concurrent calls | e2e p50 | e2e p95 | ASR p95 | Verdict |
|---|---|---|---|---|
| 8 | 161 ms | 331 ms | 470 ms | OK |
| **16** | **272 ms** | **1416 ms** | 1301 ms | **OK** |
| 24 | 2408 ms | 3145 ms | 3193 ms | BREACH |

**Max sustained: 16 concurrent calls per host**, p95 1416 ms against a 2000 ms
budget.

**Unpaced stress** (frames as fast as the CPU allows) breaches at 12, sustaining 8.

### Per-stage p95, from the live panel

| Stage | p50 | p95 | p99 |
|---|---|---|---|
| **End to end (SLO)** | 216 ms | **1334 ms** | 1506 ms |
| ASR (edge) | 220 ms | 1105 ms | 1430 ms |
| Redaction (edge) | 3.0 ms | 5.3 ms | 7.3 ms |
| Analysis (fast + agent) | 0.3 ms | 0.4 ms | 0.5 ms |

0.00% of segments breached the budget at 16 concurrent.

**ASR is the budget.** Redaction and analysis together account for under 0.5%
of it. Capacity is a whisper question, not a pipeline question — which is why
the fast/agent path split exists: the deterministic path adds nothing
measurable, so compliance signals are effectively free.

**Caveat:** the load test exercises the CPU-bound path (frames → ASR →
redaction → analysis) without Kafka, the Go ingest hop, or ClickHouse. Those
add network time this number does not include. It is an upper bound on
per-host capacity, not a full end-to-end SLO. It also runs the offline
provider — a real LLM call would add hundreds of milliseconds to the agent
path, which is precisely why that path has a deadline and the fast path does
not depend on it.

---

## 6. Provider caveat

**No Azure credentials were available, so every number above uses the
deterministic offline provider.** That provider walks the same tool-calling
loop, calls the same tools, and produces the same `Signal` objects — the
pipeline, the evidence enforcement, and the schema validation are all genuinely
exercised. What is *not* exercised is model judgement.

Concretely, this means:

- **Redaction numbers are unaffected.** Redaction is regex, checksum and NER at
  the edge. No LLM is involved, so those numbers stand as measured.
- **Citation validity of 100% is a weaker result than it looks.** A rules
  engine quoting spans it selected cannot hallucinate. The metric exists to
  catch a model that does; it has not yet had the chance to fail.
- **The classification numbers are a baseline, not a ceiling.** Resolution at
  64.6% and sarcasm at zero are exactly the failures an LLM should fix. They
  are the reason the harness exists.

To run against Azure OpenAI, set `AZURE_OPENAI_ENDPOINT`,
`AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_DEPLOYMENT` and re-run `make eval`.
The comparison against these numbers is one `make eval-compare` away.

---

## 7. Honest summary

**What is genuinely good:** zero leaks in both modes across 109 spans, including
every obfuscated form; 100% of signals carry verifiable transcript evidence;
compliance detection at 100% recall across all seven policies; 16 concurrent
calls per host inside a 2 s budget.

**What is genuinely weak:** resolution classification (64.6%), escalation band
(56.2%), action item extraction (30.9%), sarcasm (undetected), VERIF-004
precision (28.6%).

**What is untested:** LLM judgement quality, real telephony audio, non-English
speech, overlapping speakers, and prompt-level regression.

The gap between the first list and the second is not accidental. The first is
what deterministic engineering with a hard privacy invariant and an adversarial
corpus can deliver. The second is what needs a model, and the harness is built
so that adding one produces a diff table rather than an opinion.
