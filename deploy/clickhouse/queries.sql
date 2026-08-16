-- EdgeSense analytical queries.
--
-- Five queries the supervisor dashboard actually runs, each annotated with how
-- it uses the sort key. `scripts/explain_report.py` executes every one of them
-- against a live server, captures `EXPLAIN indexes=1`, and writes
-- docs/clickhouse-explain.md -- so the claims below are checked rather than
-- asserted.
--
-- Query 0 is the ORDER BY justification: the same query against the chosen key
-- and against the key that was rejected.

-- ===========================================================================
-- Q0. Why (signal_type, emitted_date, agent_id, call_id, emitted_at)?
-- ===========================================================================
-- Run against two tables holding identical 4M-row data, differing only in
-- ORDER BY. The chosen key lets the primary index cut 136 granules to 18. The
-- rejected time-first key leaves the primary index unable to contribute at
-- all (Condition: true) -- partition pruning is the only thing narrowing it.
--
-- The reasoning generalises: sort keys should lead with the low-cardinality
-- columns that queries filter on by equality. Leading with a near-unique
-- column (a timestamp, a call_id) sorts the table into an order no dashboard
-- predicate can exploit.
SELECT policy_id, count() AS events
FROM edgesense.signals
WHERE signal_type = 'compliance_violation'
  AND emitted_date >= today() - 7
GROUP BY policy_id
ORDER BY events DESC;

-- ===========================================================================
-- Q1. Sentiment over time -- how customer sentiment moves through the day.
-- ===========================================================================
-- Reads call_summaries, whose key is (ended_date, agent_id, primary_intent,
-- call_id). The date predicate is the first key column, so this is a
-- contiguous range read rather than a scan.
SELECT
    toStartOfHour(ended_at)                          AS hour,
    count()                                          AS calls,
    round(avg(sentiment_start), 3)                   AS avg_start,
    round(avg(sentiment_end), 3)                     AS avg_end,
    round(avg(sentiment_end - sentiment_start), 3)   AS avg_delta,
    countIf(sentiment_end < -0.4)                    AS ended_unhappy
FROM edgesense.call_summaries
WHERE ended_date >= today() - 14
GROUP BY hour
ORDER BY hour;

-- ===========================================================================
-- Q2. Violation rate by policy -- the compliance view.
-- ===========================================================================
-- Rate, not raw count: a policy that fires on 3 of 4 applicable calls matters
-- more than one firing 50 times across 5000. The denominator comes from
-- call_summaries so the rate is per call, not per signal.
SELECT
    v.policy_id                                              AS policy_id,
    v.calls_with_violation                                   AS calls_with_violation,
    t.total_calls                                            AS total_calls,
    round(100 * v.calls_with_violation / t.total_calls, 2)   AS violation_rate_pct,
    v.severe_events                                          AS severe_events,
    round(v.avg_confidence, 3)                               AS avg_confidence
FROM
(
    SELECT
        policy_id,
        uniq(call_id)                             AS calls_with_violation,
        countIf(severity IN ('critical','high'))  AS severe_events,
        avg(confidence)                           AS avg_confidence
    FROM edgesense.signals
    WHERE signal_type = 'compliance_violation'
      AND emitted_date >= today() - 30
      AND policy_id != ''
    GROUP BY policy_id
) AS v
CROSS JOIN
(
    SELECT uniq(call_id) AS total_calls
    FROM edgesense.call_summaries
    WHERE ended_date >= today() - 30
) AS t
ORDER BY violation_rate_pct DESC;

-- ===========================================================================
-- Q3. Intent distribution -- what people are calling about.
-- ===========================================================================
-- primary_intent is third in the sort key, so this still scans the date range,
-- but LowCardinality keeps the GROUP BY cheap and the column compresses to
-- almost nothing.
SELECT
    primary_intent,
    count()                                                  AS calls,
    round(100 * count() / sum(count()) OVER (), 2)           AS share_pct,
    countIf(escalated = 1)                                   AS escalated,
    round(100 * countIf(escalated = 1) / count(), 2)         AS escalation_rate_pct,
    round(100 * countIf(resolution = 'resolved') / count(), 2) AS resolved_pct,
    round(avg(turn_count), 1)                                AS avg_turns
FROM edgesense.call_summaries
WHERE ended_date >= today() - 30
GROUP BY primary_intent
ORDER BY calls DESC;

-- ===========================================================================
-- Q4. Per-agent rollup -- the supervisor's league table.
-- ===========================================================================
-- Served from the agent_daily AggregatingMergeTree rather than raw summaries.
-- Merging pre-aggregated states is roughly an order of magnitude less work
-- than re-reading every call row, and the states stay re-bucketable because
-- they are states rather than pre-computed averages.
SELECT
    agent_id,
    uniqMerge(calls)                                            AS calls,
    sum(escalations)                                            AS escalations,
    round(100 * sum(escalations) / uniqMerge(calls), 2)         AS escalation_rate_pct,
    sum(violations)                                             AS violations,
    round(100 * sum(resolved) / uniqMerge(calls), 2)            AS resolved_pct,
    round(avgMerge(avg_sentiment_delta), 3)                     AS avg_sentiment_delta,
    round(avgMerge(avg_turns), 1)                               AS avg_turns
FROM edgesense.agent_daily
WHERE day >= today() - 30
GROUP BY agent_id
HAVING calls >= 3
ORDER BY escalation_rate_pct DESC, violations DESC;

-- ===========================================================================
-- Q5. Pipeline latency percentiles -- the p50/p95/p99 panel.
-- ===========================================================================
-- stage leads the sort key on segment_latency, so a per-stage panel reads only
-- that stage's granules. 'e2e' is the SLO: edge emission to signal publication.
SELECT
    stage,
    count()                                     AS samples,
    round(quantile(0.50)(duration_ms), 1)       AS p50_ms,
    round(quantile(0.95)(duration_ms), 1)       AS p95_ms,
    round(quantile(0.99)(duration_ms), 1)       AS p99_ms,
    round(max(duration_ms), 1)                  AS max_ms,
    round(100 * countIf(duration_ms > 2000) / count(), 3) AS pct_over_budget
FROM edgesense.segment_latency
WHERE emitted_date >= today() - 7
GROUP BY stage
ORDER BY p95_ms DESC;
