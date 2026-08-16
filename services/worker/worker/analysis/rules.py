"""Deterministic conversation analysis.

This module is doing real work, not standing in for it. It serves three
callers:

* the **fast path**, which runs on every segment and emits compliance and
  escalation signals in microseconds -- fast enough that the p95 latency
  budget is met even when the LLM is slow or unavailable;
* the **offline provider**, which wraps these functions in the tool-calling
  protocol so the whole pipeline runs with no cloud credentials;
* the **eval baseline**, so an Azure run is compared against a real reference
  rather than against nothing.

Every function returns evidence: the turn index and the exact substring that
triggered the finding. A rule that cannot say why it fired is not allowed to
fire, because the same requirement is imposed on the model.

Known limits, measured rather than hidden (see EVAL.md):

* Sentiment is lexicon-based with negation handling. It scores sarcasm
  positively -- "Fantastic. Truly a world class experience." reads as happy.
  The corpus contains that case on purpose.
* Intent is weighted keyword scoring, so a call that changes topic mid-way
  gets the dominant topic rather than the caller's actual goal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------

NEGATIVE_LEXICON: dict[str, float] = {
    "angry": -0.8, "furious": -0.9, "unacceptable": -0.8, "ridiculous": -0.7,
    "absurd": -0.7, "terrible": -0.7, "awful": -0.7, "horrible": -0.8,
    "useless": -0.7, "frustrated": -0.6, "frustrating": -0.6, "annoyed": -0.5,
    "upset": -0.6, "disappointed": -0.5, "disgusted": -0.8, "harassment": -0.9,
    "lied": -0.8, "lying": -0.8, "scam": -0.8, "fraud": -0.4, "stolen": -0.6,
    "humiliating": -0.8, "humiliated": -0.8, "embarrassing": -0.6,
    "wasted": -0.6, "waste": -0.5, "again": -0.15, "still": -0.15,
    "never": -0.3, "nobody": -0.4, "no one": -0.4, "refuse": -0.5,
    "complaint": -0.5, "complain": -0.5, "lawyer": -0.7, "legal": -0.4,
    "regulator": -0.6, "sue": -0.7, "court": -0.5, "bounced": -0.5,
    "declined": -0.4, "denied": -0.5, "wrong": -0.4, "broken": -0.5,
    "crashes": -0.5, "failed": -0.5, "locked out": -0.4, "frozen": -0.4,
    "sorry": -0.1, "unhappy": -0.7, "worse": -0.5, "problem": -0.3,
    "issue": -0.25, "dispute": -0.3, "cannot": -0.2, "can't": -0.2,
}

POSITIVE_LEXICON: dict[str, float] = {
    "thanks": 0.4, "thank you": 0.5, "appreciate": 0.6, "great": 0.5,
    "perfect": 0.6, "excellent": 0.7, "wonderful": 0.6, "fantastic": 0.6,
    "amazing": 0.6, "good": 0.35, "helpful": 0.6, "happy": 0.6,
    "pleased": 0.5, "relief": 0.6, "sorted": 0.5, "resolved": 0.5,
    "works": 0.4, "worked": 0.45, "fixed": 0.5, "glad": 0.5, "fine": 0.25,
    "okay": 0.15, "ok": 0.15, "sure": 0.15, "please": 0.1, "yes": 0.1,
}

NEGATIONS = frozenset({"not", "no", "never", "don't", "doesn't", "didn't",
                       "isn't", "wasn't", "won't", "can't", "cannot", "nothing"})
INTENSIFIERS = {"very": 1.4, "extremely": 1.7, "really": 1.3, "so": 1.2,
                "incredibly": 1.6, "absolutely": 1.5, "completely": 1.4}

#: Negation flips sentiment for this many following tokens. Three is the
#: usual empirical window: "not happy at all" flips, but "not the fee, the
#: service was great" does not have its "great" wrongly inverted.
NEGATION_WINDOW = 3

WORD = re.compile(r"[a-z']+")


def sentiment(text: str) -> float:
    """Score text in [-1, 1]. Positive is satisfied, negative is not."""
    lowered = text.lower()
    tokens = WORD.findall(lowered)
    if not tokens:
        return 0.0

    score = 0.0
    hits = 0
    negate_until = -1
    multiplier = 1.0

    for i, token in enumerate(tokens):
        if token in NEGATIONS:
            negate_until = i + NEGATION_WINDOW
            continue
        if token in INTENSIFIERS:
            multiplier = INTENSIFIERS[token]
            continue

        weight = NEGATIVE_LEXICON.get(token) or POSITIVE_LEXICON.get(token)
        if weight is None:
            continue

        value = weight * multiplier
        if i <= negate_until:
            value = -value * 0.8  # negated sentiment is weaker than asserted
        score += value
        hits += 1
        multiplier = 1.0

    # Multi-word phrases the token loop cannot see.
    for phrase, weight in (("thank you", 0.5), ("no one", -0.4),
                           ("locked out", -0.4), ("master card", 0.0)):
        if phrase in lowered:
            score += weight
            hits += 1

    if hits == 0:
        return 0.0
    # Average, then squash. Averaging stops a long turn from scoring extreme
    # simply for containing more words.
    avg = score / max(hits, 1)
    return max(-1.0, min(1.0, avg * 1.35))


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------

ESCALATION_PATTERNS: tuple[tuple[str, float, str], ...] = (
    (r"\b(?:get|want|need|speak to|talk to)\s+(?:me\s+)?(?:a\s+)?(?:supervisor|manager)\b", 0.9, "supervisor_request"),
    (r"\bsupervisor\b", 0.6, "supervisor_mention"),
    (r"\bmanager\b", 0.6, "manager_mention"),
    (r"\b(?:speaking|talking|talk)\s+to\s+(?:a\s+)?lawyer\b", 0.85, "legal_threat"),
    (r"\blawyer\b|\battorney\b|\bsolicitor\b", 0.7, "legal_threat"),
    (r"\b(?:file|filing|report)\s+a\s+(?:complaint|grievance)\b", 0.75, "complaint_threat"),
    (r"\bregulator\b|\bombudsman\b|\bconsumer protection\b", 0.8, "regulator_threat"),
    (r"\bcancel (?:my|the) account\b|\bclose (?:my|the) account\b", 0.5, "churn_threat"),
    (r"\b(?:third|fourth|fifth|\d+(?:st|nd|rd|th))\s+time\s+(?:i've|i have|calling)\b", 0.7, "repeat_contact"),
    (r"\bthis is (?:harassment|ridiculous|unacceptable)\b", 0.7, "hostility"),
    (r"\bwasted?\s+(?:\w+\s+){0,2}hours\b", 0.6, "effort_complaint"),
    (r"\bnobody (?:will|can|has)\b|\bno one (?:will|can|has)\b", 0.55, "helplessness"),
    (r"\bin writing\b", 0.4, "documentation_demand"),
)

_COMPILED_ESCALATION = tuple(
    (re.compile(p, re.IGNORECASE), w, label) for p, w, label in ESCALATION_PATTERNS
)

RISK_BANDS = ((0.75, "high"), (0.45, "medium"), (0.2, "low"))


@dataclass
class Finding:
    """A rule hit, with the evidence that justifies it."""

    kind: str
    label: str
    score: float
    turn_idx: int
    quote: str
    policy_id: str | None = None
    detail: str = ""


def escalation_findings(turns: Iterable[dict]) -> list[Finding]:
    """Escalation signals, one per matched pattern, with the matched text."""
    out: list[Finding] = []
    for turn in turns:
        if turn.get("speaker") == "agent":
            continue  # escalation risk is about what the customer says
        text = turn.get("text", "")
        for pattern, weight, label in _COMPILED_ESCALATION:
            m = pattern.search(text)
            if m:
                out.append(
                    Finding(kind="escalation", label=label, score=weight,
                            turn_idx=turn.get("idx", turn.get("seq", 0)),
                            quote=_quote_around(text, m.start(), m.end()))
                )
    return out


def escalation_risk(turns: list[dict]) -> tuple[str, float, list[Finding]]:
    """Overall risk band for a call or window.

    Combines pattern hits with sustained negative sentiment. Sentiment alone
    is not enough: a caller can be furious and still not be an escalation
    risk if the agent is fixing the problem, which is exactly the
    ``escalation_deescalated_successfully`` case in the corpus.
    """
    findings = escalation_findings(turns)
    peak = max((f.score for f in findings), default=0.0)

    customer = [t for t in turns if t.get("speaker") == "customer"]
    if customer:
        recent = customer[-3:]
        avg_recent = sum(sentiment(t.get("text", "")) for t in recent) / len(recent)
        if avg_recent < -0.4:
            peak = max(peak, 0.45 + min(0.3, abs(avg_recent) - 0.4))
        # A conversation that ends well is not escalating, whatever was said
        # earlier. Without this, every resolved complaint reads as high risk.
        elif avg_recent > 0.25:
            peak *= 0.55

    band = "none"
    for threshold, name in RISK_BANDS:
        if peak >= threshold:
            band = name
            break
    return band, round(min(peak, 1.0), 3), findings


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------


def compliance_findings(turns: list[dict], policies: dict) -> list[Finding]:
    """Prohibited phrases and missing required disclosures, with evidence.

    A prohibited phrase cites the turn that contains it. A *missing* disclosure
    has no such turn, so it cites the window where the disclosure should have
    appeared -- an absence still has to point at where it should have been, or
    a reviewer cannot check it.
    """
    out: list[Finding] = []
    agent_turns = [t for t in turns if t.get("speaker") == "agent"]

    for policy in policies.get("policies", []):
        pid = policy["id"]

        for phrase in policy.get("prohibited_phrases", []) or []:
            for turn in agent_turns:
                text = turn.get("text", "")
                idx = text.lower().find(phrase.lower())
                if idx >= 0:
                    out.append(
                        Finding(kind="compliance", label=policy["title"],
                                score=_severity_score(policy.get("severity", "medium")),
                                turn_idx=turn.get("idx", turn.get("seq", 0)),
                                quote=_quote_around(text, idx, idx + len(phrase)),
                                policy_id=pid,
                                detail=f"prohibited phrase: {phrase!r}")
                    )
                    break

    return out


def disclosure_status(
    turns: list[dict], policies: dict, required: Iterable[str]
) -> tuple[list[str], list[Finding]]:
    """Which required disclosures were given, and findings for the ones missing."""
    by_id = {p["id"]: p for p in policies.get("policies", [])}
    agent_turns = [t for t in turns if t.get("speaker") == "agent"]
    given: list[str] = []
    missing: list[Finding] = []

    for pid in required:
        policy = by_id.get(pid)
        if not policy:
            continue
        phrases = policy.get("satisfied_by_phrases", []) or []
        window = _window_for(policy.get("window"), agent_turns)

        hit = None
        for turn in window:
            text = turn.get("text", "").lower()
            for phrase in phrases:
                idx = text.find(phrase.lower())
                if idx >= 0:
                    hit = (turn, idx, phrase)
                    break
            if hit:
                break

        if hit:
            given.append(pid)
            continue

        # Missing. Cite the window that should have contained it.
        anchor = window[0] if window else (agent_turns[0] if agent_turns else None)
        if anchor is None:
            continue
        missing.append(
            Finding(
                kind="compliance", label=policy["title"],
                score=_severity_score(policy.get("severity", "medium")),
                turn_idx=anchor.get("idx", anchor.get("seq", 0)),
                quote=_truncate(anchor.get("text", ""), 160),
                policy_id=pid,
                detail=(
                    f"required disclosure absent from {policy.get('window', 'the call')}"
                ),
            )
        )
    return given, missing


def _window_for(window: str | None, agent_turns: list[dict]) -> list[dict]:
    if window == "first_3_agent_turns":
        return agent_turns[:3]
    if window == "first_2_agent_turns":
        return agent_turns[:2]
    return agent_turns


def _severity_score(severity: str) -> float:
    return {"critical": 1.0, "high": 0.8, "medium": 0.55, "low": 0.3}.get(severity, 0.5)


# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------

INTENT_KEYWORDS: dict[str, dict[str, float]] = {
    "billing_dispute": {
        "dispute": 3.0, "charge": 2.0, "charged": 2.2, "double charged": 3.5,
        "didn't make": 2.0, "don't recognise": 2.5, "don't recognize": 2.5,
        "fee": 2.0, "overcharged": 3.0, "wrong amount": 2.5, "service charge": 2.0,
    },
    "payment_arrangement": {
        "payment plan": 3.5, "payment arrangement": 3.5, "make a payment": 3.0,
        "pay my balance": 3.0, "can't pay": 2.5, "instalment": 3.0,
        "installment": 3.0, "pay off": 2.0, "monthly": 1.2, "pay this off": 2.5,
    },
    "account_closure": {
        "close my account": 4.0, "close the account": 3.5, "cancel my account": 3.5,
        "closure": 3.0, "closing": 1.5,
    },
    "technical_support": {
        "app": 2.0, "crashes": 3.0, "not working": 2.5, "error": 2.0,
        "locked out": 2.5, "password": 1.5, "website": 2.0, "declined": 2.0,
        "pending": 2.0, "transfer": 1.5, "login": 2.0, "update": 1.0,
    },
    "fraud_report": {
        "fraud": 3.5, "didn't make": 2.5, "unauthorised": 3.5, "unauthorized": 3.5,
        "stolen": 3.0, "someone else": 2.0, "fraudulent": 3.5, "security hold": 2.0,
    },
    "address_change": {
        "change my address": 4.0, "update my address": 4.0, "moved": 2.5,
        "new address": 3.0, "mailing address": 3.0,
    },
    "refund_request": {
        "refund": 3.5, "money back": 3.0, "reimburse": 3.0, "credit back": 2.5,
        "refunded": 3.5,
    },
    "plan_upgrade": {
        "upgrade": 3.5, "premium": 3.0, "sign me up": 3.0, "enrol": 2.5,
        "enroll": 2.5, "new plan": 3.0, "tier": 2.5,
    },
    "balance_inquiry": {
        "balance": 3.0, "how much": 2.0, "available balance": 4.0,
        "check the balance": 4.0,
    },
    "collections": {
        "past due": 3.5, "collect a debt": 4.0, "debt collector": 4.0,
        "overdue": 3.0, "recovery": 2.5, "arrears": 3.0, "behind on": 2.5,
    },
    "card_replacement": {
        "replacement": 3.5, "new card": 3.0, "damaged": 2.5, "replace my card": 4.0,
        "lost my card": 3.5,
    },
    "password_reset": {
        "password": 3.0, "reset": 2.5, "locked out": 2.5, "can't log in": 3.0,
        "sign in": 2.0,
    },
    "general_inquiry": {
        "question": 1.5, "what time": 2.0, "hours": 2.0, "wondering": 1.5,
        "branch": 1.5, "order number": 2.0, "power of attorney": 2.0,
    },
}


def classify_intent(turns: list[dict]) -> tuple[str, list[str], dict[str, float]]:
    """Score every intent; return the best, runners-up, and the full scores.

    Customer turns are weighted more than agent turns: the agent's words
    describe the resolution, the customer's describe the reason for calling.
    Early turns are weighted more than late ones for the same reason.
    """
    scores: dict[str, float] = {intent: 0.0 for intent in INTENT_KEYWORDS}
    total_turns = max(len(turns), 1)

    for turn in turns:
        text = turn.get("text", "").lower()
        speaker_weight = 1.0 if turn.get("speaker") == "customer" else 0.45
        position = turn.get("idx", turn.get("seq", 0))
        # Linear decay to 0.5 by the end of the call.
        recency_weight = 1.0 - 0.5 * (position / total_turns)

        for intent, keywords in INTENT_KEYWORDS.items():
            for phrase, weight in keywords.items():
                if phrase in text:
                    scores[intent] += weight * speaker_weight * recency_weight

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked or ranked[0][1] <= 0:
        return "general_inquiry", [], scores

    primary = ranked[0][0]
    # A runner-up only counts as a secondary intent if it is a substantial
    # fraction of the winner; otherwise every call collects noise intents.
    secondary = [name for name, score in ranked[1:4] if score >= ranked[0][1] * 0.45]
    return primary, secondary, scores


def intent_evidence(turns: list[dict], intent: str) -> tuple[int, str] | None:
    """The turn that most supports ``intent``, so the signal can cite it."""
    keywords = INTENT_KEYWORDS.get(intent, {})
    best: tuple[float, int, str] | None = None
    for turn in turns:
        text = turn.get("text", "")
        lowered = text.lower()
        for phrase, weight in keywords.items():
            idx = lowered.find(phrase)
            if idx < 0:
                continue
            w = weight * (1.0 if turn.get("speaker") == "customer" else 0.45)
            if best is None or w > best[0]:
                best = (w, turn.get("idx", turn.get("seq", 0)),
                        _quote_around(text, idx, idx + len(phrase)))
    if best is None:
        return None
    return best[1], best[2]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

RESOLVED_MARKERS = (
    "all set", "that's everything", "that worked", "went through", "sorted",
    "thank you", "that's all", "resolved", "confirmed", "you're all set",
)
ESCALATED_MARKERS = ("supervisor", "escalat", "manager", "transfer you")
FOLLOWUP_MARKERS = (
    "business days", "follow up", "we'll be in touch", "you'll hear back",
    "within twenty-four hours", "next month", "call you back", "mail the",
    "send you", "provisional credit",
)


def classify_resolution(turns: list[dict], escalated: bool) -> tuple[str, int, str]:
    """Resolution status plus the turn that justifies it."""
    if escalated:
        for turn in reversed(turns):
            lowered = turn.get("text", "").lower()
            for marker in ESCALATED_MARKERS:
                idx = lowered.find(marker)
                if idx >= 0:
                    return ("escalated", turn.get("idx", turn.get("seq", 0)),
                            _quote_around(turn.get("text", ""), idx, idx + len(marker)))
        last = turns[-1] if turns else {}
        return "escalated", last.get("idx", 0), _truncate(last.get("text", ""), 160)

    tail = turns[-5:]
    for turn in reversed(tail):
        lowered = turn.get("text", "").lower()
        for marker in FOLLOWUP_MARKERS:
            idx = lowered.find(marker)
            if idx >= 0:
                return ("follow_up_required", turn.get("idx", turn.get("seq", 0)),
                        _quote_around(turn.get("text", ""), idx, idx + len(marker)))

    for turn in reversed(tail):
        lowered = turn.get("text", "").lower()
        for marker in RESOLVED_MARKERS:
            idx = lowered.find(marker)
            if idx >= 0:
                return ("resolved", turn.get("idx", turn.get("seq", 0)),
                        _quote_around(turn.get("text", ""), idx, idx + len(marker)))

    last = turns[-1] if turns else {}
    return "unresolved", last.get("idx", 0), _truncate(last.get("text", ""), 160)


# ---------------------------------------------------------------------------
# Action items
# ---------------------------------------------------------------------------

ACTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bi'?(?:ll|ve| will| have)\s+(?:go ahead and\s+)?(\w+(?:\s+\w+){1,7})", "agent"),
    (r"\bi'?m\s+(?:going to\s+)?(escalating|requesting|crediting|freezing|sending|mailing|releasing)\s+(\w+(?:\s+\w+){0,6})", "agent"),
    (r"\bwe'?ll\s+(\w+(?:\s+\w+){1,7})", "agent"),
    (r"\byou'?ll (?:need to|have to)\s+(\w+(?:\s+\w+){1,7})", "customer"),
    (r"\b(?:i can|let me)\s+(request|reopen|cancel|order|submit|send|mail|credit)\s+(\w+(?:\s+\w+){0,6})", "agent"),
)

_COMPILED_ACTIONS = tuple((re.compile(p, re.IGNORECASE), owner) for p, owner in ACTION_PATTERNS)

#: Verb phrases that are conversational filler rather than commitments.
ACTION_NOISE = re.compile(
    r"^(?:got|have|see|understand|hear|know|think|need|got it|look|check that)\b",
    re.IGNORECASE,
)


def action_items(turns: list[dict]) -> list[Finding]:
    """Commitments made during the call, each citing the turn that made it."""
    out: list[Finding] = []
    seen: set[str] = set()
    for turn in turns:
        if turn.get("speaker") != "agent":
            continue
        text = turn.get("text", "")
        for pattern, owner in _COMPILED_ACTIONS:
            for m in pattern.finditer(text):
                phrase = " ".join(g for g in m.groups() if g).strip(" .,")
                if len(phrase) < 6 or ACTION_NOISE.match(phrase):
                    continue
                key = phrase.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    Finding(kind="action_item", label=phrase, score=0.6,
                            turn_idx=turn.get("idx", turn.get("seq", 0)),
                            quote=_quote_around(text, m.start(), m.end()),
                            detail=owner)
                )
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _quote_around(text: str, start: int, end: int, pad: int = 45) -> str:
    lo = max(0, start - pad)
    hi = min(len(text), end + pad)
    quote = text[lo:hi].strip()
    if lo > 0:
        quote = "..." + quote
    if hi < len(text):
        quote = quote + "..."
    return quote


def _truncate(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else text[: n - 3] + "..."
