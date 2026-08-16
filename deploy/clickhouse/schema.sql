-- EdgeSense analytics schema.
--
-- Three fact tables, one per shape of thing the pipeline produces, plus
-- rollups. The ORDER BY keys are the only real design decision in a
-- MergeTree, because they determine the primary index, the compression, and
-- which queries can skip granules instead of scanning. Each choice is
-- justified below and the reasoning is checked against real EXPLAIN output in
-- queries.sql.
--
-- The rule applied throughout: order by low-cardinality filters first, then
-- time, then high-cardinality identifiers last. Putting call_id first would
-- give near-perfect compression on that one column and force a full scan for
-- every dashboard query, which is the opposite of what this workload does.

CREATE DATABASE IF NOT EXISTS edgesense;

-- ---------------------------------------------------------------------------
-- signals: one row per live signal.
-- ---------------------------------------------------------------------------
-- Query shapes this serves:
--   * "violation rate by policy over the last 7 days"      -> filters signal_type, ts
--   * "sentiment/escalation over time for one agent"       -> filters agent_id, ts
--   * "everything that fired on this call"                 -> filters call_id
--
-- ORDER BY (signal_type, emitted_date, agent_id, call_id, emitted_at):
--   signal_type first because every dashboard query filters on it and it has
--   ~4 distinct values, so one equality predicate eliminates ~75% of granules
--   immediately. Date next because every query is time-bounded. agent_id
--   before call_id because per-agent rollups are a first-class view and
--   per-call lookup is a rare, cheap point query.
--
-- Rejected: (emitted_at, call_id). Time-first looks natural and is wrong here
-- -- it sorts by a near-unique value, so the primary index degenerates to one
-- mark per granule and no dashboard filter can skip anything.
CREATE TABLE IF NOT EXISTS edgesense.signals
(
    signal_id           String,
    call_id             String,
    agent_id            LowCardinality(String),
    signal_type         LowCardinality(String),
    label               LowCardinality(String),
    severity            LowCardinality(String),
    policy_id           LowCardinality(String),
    confidence          Float32,
    rationale           String,

    emitted_at          DateTime64(3, 'UTC'),
    emitted_date        Date MATERIALIZED toDate(emitted_at),
    window_start_ms     UInt32,
    window_end_ms       UInt32,

    -- Evidence is stored as parallel arrays rather than a nested table: it is
    -- always read whole (to render a hover card) and never filtered on, so
    -- arrays avoid a join for the only access pattern that exists.
    evidence_seq        Array(UInt32),
    evidence_start_ms   Array(UInt32),
    evidence_end_ms     Array(UInt32),
    evidence_speaker    Array(LowCardinality(String)),
    evidence_quote      Array(String),

    -- Per-stage latency, so the portal's latency panel is served from the
    -- same rows as the signals rather than a separate metrics store.
    latency_asr_ms              Nullable(Float32),
    latency_redact_ms           Nullable(Float32),
    latency_ingest_ms           Nullable(Float32),
    latency_queue_ms            Nullable(Float32),
    latency_analyze_ms          Nullable(Float32),
    latency_llm_ms              Nullable(Float32),
    latency_segment_to_signal_ms Nullable(Float32),

    model_name          LowCardinality(String),
    prompt_version      LowCardinality(String),
    trace_id            String,
    ingested_at         DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(emitted_at)
ORDER BY (signal_type, emitted_date, agent_id, call_id, emitted_at)
-- signal_id is not in the sort key, so a replayed message would duplicate.
-- Dedupe is handled upstream in Redis; this index exists so the sink can
-- cheaply check for a specific signal during incident triage.
TTL toDateTime(emitted_at) + INTERVAL 400 DAY
SETTINGS index_granularity = 8192;

ALTER TABLE edgesense.signals
    ADD INDEX IF NOT EXISTS idx_call_id call_id TYPE bloom_filter(0.01) GRANULARITY 4;
-- A bloom filter on call_id makes "show me this call" a granule-skipping
-- lookup even though call_id is fourth in the sort key. Cheap to maintain,
-- and it is the query a supervisor runs when something looks wrong.

-- ---------------------------------------------------------------------------
-- call_summaries: one row per completed call.
-- ---------------------------------------------------------------------------
-- ORDER BY (ended_date, agent_id, primary_intent, call_id):
--   Date first here, not signal_type, because there is no equivalent
--   low-cardinality discriminator and essentially every query over this table
--   is "for this period". agent_id then intent matches the two rollups the
--   supervisor dashboard draws.
--
-- ReplacingMergeTree on generated_at: a post-call summary can legitimately be
-- rewritten (a worker restart re-finalises a call, or a prompt is re-run over
-- history). Replacing keeps the newest by version instead of accumulating
-- contradictory rows that would double every aggregate.
CREATE TABLE IF NOT EXISTS edgesense.call_summaries
(
    call_id             String,
    agent_id            LowCardinality(String),
    primary_intent      LowCardinality(String),
    secondary_intents   Array(LowCardinality(String)),
    resolution          LowCardinality(String),
    escalated           UInt8,

    summary             String,
    sentiment_start     Float32,
    sentiment_end       Float32,
    sentiment_delta     Float32 MATERIALIZED sentiment_end - sentiment_start,

    compliance_violations Array(LowCardinality(String)),
    disclosures_given     Array(LowCardinality(String)),
    violation_count       UInt8 MATERIALIZED length(compliance_violations),

    action_items        Array(String),
    action_item_owners  Array(LowCardinality(String)),

    turn_count          UInt16,
    duration_ms         UInt32,
    redaction_count     UInt16,

    started_at          DateTime64(3, 'UTC'),
    ended_at            DateTime64(3, 'UTC'),
    ended_date          Date MATERIALIZED toDate(ended_at),

    model_name          LowCardinality(String),
    prompt_version      LowCardinality(String),
    trace_id            String,
    generated_at        DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(generated_at)
PARTITION BY toYYYYMM(ended_at)
ORDER BY (ended_date, agent_id, primary_intent, call_id)
TTL toDateTime(ended_at) + INTERVAL 400 DAY
SETTINGS index_granularity = 8192;

-- ---------------------------------------------------------------------------
-- segment_latency: one row per segment, for the latency panel.
-- ---------------------------------------------------------------------------
-- Kept separate from signals because the cardinalities differ by an order of
-- magnitude (every segment vs. only those that produced a signal) and because
-- latency has a much shorter useful life than an insight. Mixing them would
-- force one TTL and one partitioning scheme onto two unrelated workloads.
CREATE TABLE IF NOT EXISTS edgesense.segment_latency
(
    call_id             String,
    seq                 UInt32,
    agent_id            LowCardinality(String),
    stage               LowCardinality(String),  -- asr | redact | ingest | queue | analyze | llm | e2e
    duration_ms         Float32,
    is_final            UInt8,
    emitted_at          DateTime64(3, 'UTC'),
    emitted_date        Date MATERIALIZED toDate(emitted_at),
    ingested_at         DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(emitted_at)
-- Daily partitions, not monthly: this table is written on every segment and
-- read almost exclusively for "the last hour". Dropping a whole day's part is
-- also how the short TTL stays cheap.
ORDER BY (stage, emitted_date, agent_id, emitted_at)
TTL toDateTime(emitted_at) + INTERVAL 30 DAY
SETTINGS index_granularity = 8192;

-- ---------------------------------------------------------------------------
-- Rollups.
-- ---------------------------------------------------------------------------
-- The dashboard reads hourly buckets over weeks. Aggregating raw rows every
-- refresh is wasteful when the inputs are immutable once the hour has passed,
-- so the aggregation happens once at insert time.
--
-- AggregatingMergeTree with state functions, not SummingMergeTree: quantiles
-- and uniq cannot be summed, and storing pre-computed averages would make
-- re-bucketing to a different window silently wrong.

CREATE TABLE IF NOT EXISTS edgesense.signals_hourly
(
    hour                DateTime('UTC'),
    agent_id            LowCardinality(String),
    signal_type         LowCardinality(String),
    label               LowCardinality(String),
    severity            LowCardinality(String),
    policy_id           LowCardinality(String),
    signal_count        SimpleAggregateFunction(sum, UInt64),
    calls               AggregateFunction(uniq, String),
    avg_confidence      AggregateFunction(avg, Float32),
    p95_latency_ms      AggregateFunction(quantile(0.95), Float32)
)
ENGINE = AggregatingMergeTree
PARTITION BY toYYYYMM(hour)
ORDER BY (signal_type, hour, agent_id, label, severity, policy_id);

CREATE MATERIALIZED VIEW IF NOT EXISTS edgesense.signals_hourly_mv
TO edgesense.signals_hourly
AS SELECT
    toStartOfHour(emitted_at)                       AS hour,
    agent_id,
    signal_type,
    label,
    severity,
    policy_id,
    count()                                         AS signal_count,
    uniqState(call_id)                              AS calls,
    avgState(confidence)                            AS avg_confidence,
    quantileState(0.95)(ifNull(latency_segment_to_signal_ms, 0)) AS p95_latency_ms
FROM edgesense.signals
GROUP BY hour, agent_id, signal_type, label, severity, policy_id;

CREATE TABLE IF NOT EXISTS edgesense.agent_daily
(
    day                 Date,
    agent_id            LowCardinality(String),
    calls               AggregateFunction(uniq, String),
    escalations         SimpleAggregateFunction(sum, UInt64),
    violations          SimpleAggregateFunction(sum, UInt64),
    resolved            SimpleAggregateFunction(sum, UInt64),
    avg_sentiment_delta AggregateFunction(avg, Float32),
    avg_turns           AggregateFunction(avg, UInt16)
)
ENGINE = AggregatingMergeTree
PARTITION BY toYYYYMM(day)
ORDER BY (day, agent_id);

CREATE MATERIALIZED VIEW IF NOT EXISTS edgesense.agent_daily_mv
TO edgesense.agent_daily
AS SELECT
    toDate(ended_at)                    AS day,
    agent_id,
    uniqState(call_id)                  AS calls,
    sum(escalated)                      AS escalations,
    sum(length(compliance_violations))  AS violations,
    sum(resolution = 'resolved')        AS resolved,
    -- Cast back to Float32 explicitly: ClickHouse promotes Float32 - Float32
    -- to Float64, which makes the aggregate state type Float64 and no longer
    -- assignable to the column declared above.
    avgState(toFloat32(sentiment_end - sentiment_start)) AS avg_sentiment_delta,
    avgState(turn_count)                AS avg_turns
FROM edgesense.call_summaries
GROUP BY day, agent_id;

-- ---------------------------------------------------------------------------
-- Convenience views for the portal.
-- ---------------------------------------------------------------------------

CREATE VIEW IF NOT EXISTS edgesense.v_latency_percentiles AS
SELECT
    stage,
    count()                                 AS samples,
    round(quantile(0.50)(duration_ms), 1)   AS p50_ms,
    round(quantile(0.95)(duration_ms), 1)   AS p95_ms,
    round(quantile(0.99)(duration_ms), 1)   AS p99_ms,
    round(max(duration_ms), 1)              AS max_ms
FROM edgesense.segment_latency
GROUP BY stage
ORDER BY p95_ms DESC;

CREATE VIEW IF NOT EXISTS edgesense.v_policy_violation_rates AS
SELECT
    policy_id,
    uniq(call_id)                                       AS calls_with_violation,
    count()                                             AS violation_events,
    round(avg(confidence), 3)                           AS avg_confidence,
    countIf(severity IN ('critical', 'high'))           AS severe_events
FROM edgesense.signals
WHERE signal_type = 'compliance_violation' AND policy_id != ''
GROUP BY policy_id
ORDER BY calls_with_violation DESC;
