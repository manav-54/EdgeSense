"""The tool-using agent loop.

One-shot prompting would have been less code: send the transcript, get JSON
back, parse it. It was rejected because the failure mode is silent. A model
asked to produce findings in a single pass will produce plausible ones -- a
policy id it half-remembers, a quote it paraphrased into something nobody
said -- and the parser has no way to tell those from real findings.

The loop makes that structurally harder:

* to cite a policy, the agent must call ``lookup_policy`` and get the real
  text back, so a hallucinated id fails immediately as a tool error;
* to cite evidence, it must give turn indices that resolve against the actual
  transcript, so a fabricated quote cannot become a Signal;
* a tool error goes back into the conversation, giving the agent a chance to
  correct itself rather than the pipeline discarding the whole turn.

The loop is bounded by ``max_steps`` and by a wall-clock deadline, because the
live path has a p95 budget of 2 seconds and an agent that wants a fifth tool
round-trip has already lost.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from edgesense_core.timeutil import monotonic_ms

from worker.llm.base import (
    Completion,
    LLMProvider,
    Message,
    ProviderError,
    Role,
    ToolSpec,
    TransientProviderError,
    Usage,
)
from worker.obs import (
    AGENT_STEPS,
    SCHEMA_RETRIES,
    LLM_ERRORS,
    LLM_LATENCY,
    LLM_TOKENS,
    METRICS_AVAILABLE,
    TOOL_CALLS,
    get_logger,
)
from worker.prompts import registry
from worker.tools import TOOL_SPECS, FlaggedRisk, ToolContext, invoke

log = get_logger(__name__)

#: Returns (is_valid, human-readable detail). The detail is fed back to the
#: model verbatim on a repair attempt.
Validator = Callable[[str], tuple[bool, str]]

JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class AgentResult:
    """Everything one agent run produced, including how it got there."""

    flagged: list[FlaggedRisk] = field(default_factory=list)
    content: str = ""
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    tool_calls: list[str] = field(default_factory=list)
    tool_errors: list[str] = field(default_factory=list)
    llm_ms: float = 0.0
    total_ms: float = 0.0
    truncated: bool = False
    repairs: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def json_payload(self) -> dict | None:
        """Parse the final message as JSON, tolerating a wrapping code fence."""
        if not self.content:
            return None
        text = self.content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.split("\n", 1)[-1] if "\n" in text else text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Models sometimes wrap JSON in a sentence. Take the outermost object
        # rather than failing the whole call over a stray prefix.
        m = JSON_BLOCK.search(text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


class Agent:
    """Runs the tool loop against a provider."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: list[ToolSpec] | None = None,
        *,
        deadline_ms: float | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools if tools is not None else TOOL_SPECS
        self.deadline_ms = deadline_ms

    def run(
        self,
        prompt_id: str,
        ctx: ToolContext,
        *,
        version: str = "latest",
        json_mode: bool = False,
        max_steps: int | None = None,
        validator: Validator | None = None,
        max_repairs: int = 2,
        **template_vars: object,
    ) -> AgentResult:
        prompt = registry.get(prompt_id, version)
        steps_allowed = max_steps if max_steps is not None else prompt.max_steps

        messages = [
            Message(role=Role.SYSTEM, content=prompt.system),
            Message(role=Role.USER, content=prompt.render_user(**template_vars)),
        ]

        # The offline provider reads call state directly rather than parsing it
        # back out of the prompt it was just handed.
        if getattr(self.provider, "needs_context", False):
            self.provider.set_context(ctx, list(template_vars.get("_required_disclosures", []) or []))

        result = AgentResult()
        started = monotonic_ms()
        repairs_used = 0
        step = 0

        while step < steps_allowed + repairs_used:
            if self._out_of_time(started):
                result.truncated = True
                log.warning("agent loop hit its deadline",
                            call_id=ctx.state.call_id, prompt=prompt.ref, step=step)
                break
            step += 1

            try:
                completion = self._complete(messages, prompt, json_mode)
            except TransientProviderError as exc:
                # The caller decides whether to retry the whole analysis; a
                # partial loop result is still worth returning, because tool
                # calls from earlier steps have already produced findings.
                result.error = f"provider unavailable: {exc}"
                if METRICS_AVAILABLE:
                    LLM_ERRORS.labels(provider=self.provider.name, kind="transient").inc()
                break
            except ProviderError as exc:
                result.error = str(exc)
                if METRICS_AVAILABLE:
                    LLM_ERRORS.labels(provider=self.provider.name, kind="permanent").inc()
                break

            result.steps = step
            result.usage = result.usage + completion.usage
            result.llm_ms += completion.latency_ms

            if not completion.wants_tools:
                # Schema repair. The model gets the validator's own error text,
                # which is far more actionable than "invalid JSON" -- pydantic
                # names the field and says what was wrong with it, and models
                # correct reliably when told precisely what to fix.
                if validator is not None:
                    ok, detail = validator(completion.content)
                    if not ok and repairs_used < max_repairs:
                        repairs_used += 1
                        result.repairs = repairs_used
                        if METRICS_AVAILABLE:
                            SCHEMA_RETRIES.inc()
                        log.info("response failed validation; asking for a repair",
                                 call_id=ctx.state.call_id, prompt=prompt.ref,
                                 attempt=repairs_used, detail=detail[:200])
                        messages.append(
                            Message(role=Role.ASSISTANT, content=completion.content)
                        )
                        messages.append(Message(
                            role=Role.USER,
                            content=(
                                "Your previous response was rejected by schema "
                                f"validation:\n\n{detail}\n\n"
                                "Reply with the corrected JSON object only. No prose, "
                                "no markdown fence."
                            ),
                        ))
                        continue
                    if not ok:
                        result.error = f"schema validation failed after {repairs_used} repairs: {detail}"
                result.content = completion.content
                break

            messages.append(
                Message(role=Role.ASSISTANT, content=completion.content,
                        tool_calls=completion.tool_calls)
            )
            for call in completion.tool_calls:
                before = len(ctx.errors)
                payload = invoke(ctx, call.name, call.arguments)
                failed = len(ctx.errors) > before
                result.tool_calls.append(call.name)
                if METRICS_AVAILABLE:
                    TOOL_CALLS.labels(
                        tool=call.name, outcome="error" if failed else "ok"
                    ).inc()
                messages.append(
                    Message(role=Role.TOOL, content=payload,
                            tool_call_id=call.id, name=call.name)
                )
        if not result.content and result.error is None:
            # Loop exhausted without the model signalling completion.
            result.truncated = True

        result.flagged = list(ctx.flagged)
        result.tool_errors = list(ctx.errors)
        result.total_ms = monotonic_ms() - started

        if METRICS_AVAILABLE:
            AGENT_STEPS.observe(result.steps)
            LLM_TOKENS.labels(provider=self.provider.name, direction="prompt").inc(
                result.usage.prompt_tokens
            )
            LLM_TOKENS.labels(provider=self.provider.name, direction="completion").inc(
                result.usage.completion_tokens
            )
        return result

    # -- internals ---------------------------------------------------------

    def _complete(self, messages: list[Message], prompt, json_mode: bool) -> Completion:
        t0 = monotonic_ms()
        completion = self.provider.complete(
            messages, self.tools, temperature=prompt.temperature, json_mode=json_mode
        )
        if METRICS_AVAILABLE:
            LLM_LATENCY.labels(provider=self.provider.name).observe(
                (monotonic_ms() - t0) / 1000.0
            )
        return completion

    def _out_of_time(self, started: float) -> bool:
        return (
            self.deadline_ms is not None
            and (monotonic_ms() - started) >= self.deadline_ms
        )
