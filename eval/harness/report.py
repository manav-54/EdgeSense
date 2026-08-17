"""Report formatting: human tables for the terminal, JSON for the diff tool."""

from __future__ import annotations

from harness.runner import RunResult


def _bar(value: float, width: int = 18) -> str:
    filled = int(round(value * width))
    return "█" * filled + "·" * (width - filled)


def format_run(result: RunResult) -> str:
    out: list[str] = []
    red = result.redaction.as_dict()
    cls = result.classification.as_dict()
    config = result.config

    out.append("=" * 78)
    out.append(f"EdgeSense evaluation — {result.started_at}")
    out.append(
        f"mode={config['mode']}  provider={config['provider']}  model={config['model']}"
    )
    out.append(
        f"prompts: live={config['live_prompt']} post={config['post_prompt']}  "
        f"ner={config['ner_backend']}  agent_path={config['use_agent']}"
    )
    out.append(f"calls={config['calls']}  duration={result.duration_s:.1f}s")
    out.append("=" * 78)

    # -- redaction ---------------------------------------------------------
    overall = red["overall"]
    out.append("")
    out.append("REDACTION")
    out.append(f"  spans evaluated : {red['spans_total']}")
    out.append(
        f"  LEAKS           : {red['leak_count']}  "
        f"(leak rate {red['leak_rate'] * 100:.3f}%)   <- the safety metric"
    )
    out.append(
        f"  recall          : {overall['recall'] * 100:6.2f}%   "
        f"precision {overall['precision'] * 100:6.2f}%   "
        f"F2 {overall['f2']:.3f} (recall-weighted)   F1 {overall['f1']:.3f}"
    )
    out.append(
        f"  type accuracy   : {red['type_accuracy'] * 100:6.2f}%  "
        f"({red['type_confused']} caught but mislabelled)"
    )
    out.append(f"  false positives : {red['false_positives']}")

    out.append("")
    out.append("  by PII type")
    out.append(f"    {'type':<10} {'n':>4} {'recall':>8} {'prec':>8}  {'':<18}")
    for pii_type, counts in red["by_type"].items():
        if counts["support"] == 0:
            continue
        out.append(
            f"    {pii_type:<10} {counts['support']:>4} "
            f"{counts['recall'] * 100:>7.1f}% {counts['precision'] * 100:>7.1f}%  "
            f"{_bar(counts['recall'])}"
        )

    out.append("")
    out.append("  by surface form  (how the value was spoken)")
    for surface, counts in red["by_surface_form"].items():
        if counts["support"] == 0:
            continue
        out.append(
            f"    {surface:<12} {counts['support']:>4} "
            f"{counts['recall'] * 100:>7.1f}%  {_bar(counts['recall'])}"
        )

    out.append("")
    out.append("  by corpus category")
    for category, counts in red["by_category"].items():
        if counts["support"] == 0:
            continue
        out.append(
            f"    {category:<14} {counts['support']:>4} "
            f"{counts['recall'] * 100:>7.1f}%  {_bar(counts['recall'])}"
        )

    if red["by_detector"]:
        out.append("")
        out.append("  catches by detector")
        for detector, n in sorted(red["by_detector"].items(), key=lambda kv: -kv[1]):
            out.append(f"    {detector:<16} {n:>4}")

    if red["leaks"]:
        out.append("")
        out.append("  LEAKED VALUES (first 10)")
        for leak in red["leaks"][:10]:
            out.append(
                f"    {leak['call_id'][:46]:<46} turn {leak['turn']:>2} "
                f"{leak['type']:<8} {leak['surface_form']:<10} via {leak['space']}"
            )

    if red["fp_examples"]:
        out.append("")
        out.append("  OVER-REDACTIONS (first 6) — the cost of the recall bias")
        for example in red["fp_examples"][:6]:
            out.append(f"    {example}")

    # -- classification ----------------------------------------------------
    out.append("")
    out.append("CLASSIFICATION")
    out.append(
        f"  primary intent  : {cls['intent']['accuracy'] * 100:6.2f}%  "
        f"({cls['intent']['correct']}/{cls['intent']['total']})"
    )
    out.append(
        f"  resolution      : {cls['resolution']['accuracy'] * 100:6.2f}%  "
        f"({cls['resolution']['correct']}/{cls['resolution']['total']})"
    )
    out.append(
        f"  escalation band : {cls['escalation_risk_band']['accuracy'] * 100:6.2f}%  "
        f"({cls['escalation_risk_band']['correct']}/{cls['escalation_risk_band']['total']})"
    )
    esc = cls["escalated_flag"]
    out.append(
        f"  escalated flag  : P {esc['precision'] * 100:5.1f}%  R {esc['recall'] * 100:5.1f}%  "
        f"F1 {esc['f1']:.3f}  (support {esc['support']})"
    )
    vio = cls["compliance_violations"]
    out.append(
        f"  violations      : P {vio['precision'] * 100:5.1f}%  R {vio['recall'] * 100:5.1f}%  "
        f"F1 {vio['f1']:.3f}  (support {vio['support']})"
    )
    dis = cls["disclosures_given"]
    out.append(
        f"  disclosures     : P {dis['precision'] * 100:5.1f}%  R {dis['recall'] * 100:5.1f}%  "
        f"F1 {dis['f1']:.3f}  (support {dis['support']})"
    )
    sen = cls["sentiment"]
    out.append(
        f"  sentiment dir   : {sen['direction_accuracy'] * 100:6.2f}%  "
        f"(mean abs error {sen['mean_abs_error']:.3f})"
    )

    if cls["violations_by_policy"]:
        out.append("")
        out.append("  violations by policy")
        out.append(f"    {'policy':<12} {'n':>3} {'recall':>8} {'prec':>8}")
        for policy, counts in cls["violations_by_policy"].items():
            out.append(
                f"    {policy:<12} {counts['support']:>3} "
                f"{counts['recall'] * 100:>7.1f}% {counts['precision'] * 100:>7.1f}%"
            )

    if cls["intent"]["top_confusions"]:
        out.append("")
        out.append("  intent confusions")
        for confusion, n in cls["intent"]["top_confusions"].items():
            out.append(f"    {confusion:<52} {n:>3}")

    if cls["resolution"]["top_confusions"]:
        out.append("")
        out.append("  resolution confusions")
        for confusion, n in cls["resolution"]["top_confusions"].items():
            out.append(f"    {confusion:<52} {n:>3}")

    # -- summary quality ---------------------------------------------------
    quality = cls["summary_quality"]
    out.append("")
    out.append("SUMMARY QUALITY")
    out.append(
        f"  schema valid    : {quality['schema_valid_rate'] * 100:6.2f}%  "
        f"({quality['valid']}/{quality['attempted']}, {quality['needed_repair']} needed repair)"
    )
    out.append(
        f"  citation valid  : {quality['citation_validity'] * 100:6.2f}%  "
        f"({quality['citations_checked']} quotes checked against the transcript)"
    )
    out.append(
        f"  action items    : {quality['action_items_produced']} produced, "
        f"{quality['action_items_grounded']} with evidence; "
        f"recall vs labels {quality['action_item_recall_vs_labels'] * 100:.1f}%"
    )
    signals = cls["signals"]
    out.append(
        f"  signals         : {signals['total']} emitted, "
        f"{signals['without_evidence']} without evidence "
        f"({signals['evidence_rate'] * 100:.2f}% sourced)"
    )
    if quality["bad_citation_examples"]:
        out.append("  unverifiable citations (first 5)")
        for example in quality["bad_citation_examples"][:5]:
            out.append(f"    {example}")

    # -- timings -----------------------------------------------------------
    timings = result.timings
    out.append("")
    out.append("TIMINGS (per operation, ms)")
    out.append(
        f"  redact       mean {timings['redact_ms_mean']:>8.3f}   "
        f"p95 {timings['redact_ms_p95']:>8.3f}"
    )
    out.append(
        f"  live window  mean {timings['analysis_ms_mean']:>8.3f}   "
        f"p95 {timings['analysis_ms_p95']:>8.3f}"
    )
    out.append(
        f"  post-call    mean {timings['post_call_ms_mean']:>8.3f}   "
        f"p95 {timings['post_call_ms_p95']:>8.3f}"
    )
    out.append("")
    return "\n".join(out)
