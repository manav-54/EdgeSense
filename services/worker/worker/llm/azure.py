"""Azure OpenAI chat-completions provider.

Deliberately implemented against ``httpx`` rather than the vendor SDK: the
surface used here is one POST with a documented body, and taking the SDK would
add a large transitive dependency tree to a latency-sensitive worker for no
capability we need.

Retry policy is narrow on purpose. 429 and 5xx are retried with exponential
backoff and honour ``Retry-After``; 400-class errors are not, because a
malformed request will be malformed on the second attempt too and retrying it
just burns the latency budget.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass

from edgesense_core.timeutil import monotonic_ms

from worker.llm.base import (
    Completion,
    Message,
    ProviderError,
    ToolCall,
    ToolSpec,
    TransientProviderError,
    Usage,
)
from worker.obs import get_logger

log = get_logger(__name__)

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


@dataclass
class AzureConfig:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str = "2024-10-21"
    timeout_s: float = 20.0
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> AzureConfig | None:
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
        if not (endpoint and api_key and deployment):
            return None
        return cls(
            endpoint=endpoint.rstrip("/"),
            api_key=api_key,
            deployment=deployment,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            timeout_s=float(os.environ.get("AZURE_OPENAI_TIMEOUT_S", "20")),
        )


class AzureOpenAIProvider:
    name = "azure-openai"

    def __init__(self, config: AzureConfig) -> None:
        import httpx

        self.config = config
        self.model = config.deployment
        self._url = (
            f"{config.endpoint}/openai/deployments/{config.deployment}"
            f"/chat/completions?api-version={config.api_version}"
        )
        # One pooled client for the process. Re-establishing TLS per request
        # would add tens of milliseconds to every call inside a 2s budget.
        self._client = httpx.Client(
            timeout=httpx.Timeout(config.timeout_s, connect=5.0),
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            headers={"api-key": config.api_key, "content-type": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> Completion:
        body: dict = {
            "messages": [m.to_wire() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = [t.to_wire() for t in tools]
            body["tool_choice"] = "auto"
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        t0 = monotonic_ms()
        payload = self._post_with_retries(body)
        latency = monotonic_ms() - t0

        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}

        tool_calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                # A model that emits unparseable arguments has failed this
                # tool call, but the rest of the turn may still be usable.
                log.warning("tool call arguments were not valid JSON",
                            tool=fn.get("name", "?"))
                continue
            tool_calls.append(
                ToolCall(id=raw.get("id", ""), name=fn.get("name", ""), arguments=args)
            )

        usage_raw = payload.get("usage") or {}
        return Completion(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            usage=Usage(
                prompt_tokens=usage_raw.get("prompt_tokens", 0),
                completion_tokens=usage_raw.get("completion_tokens", 0),
            ),
            model=payload.get("model", self.model),
            latency_ms=latency,
            finish_reason=choice.get("finish_reason", "stop"),
        )

    def _post_with_retries(self, body: dict) -> dict:
        import httpx

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._client.post(self._url, json=body)
            except httpx.TimeoutException as exc:
                last_error = TransientProviderError(f"azure request timed out: {exc}")
            except httpx.HTTPError as exc:
                last_error = TransientProviderError(f"azure request failed: {exc}")
            else:
                if response.status_code < 300:
                    return response.json()
                detail = response.text[:400]
                if response.status_code in RETRYABLE_STATUS:
                    last_error = TransientProviderError(
                        f"azure returned {response.status_code}: {detail}"
                    )
                    self._sleep_for(response, attempt)
                    continue
                # 4xx that retrying cannot fix: bad deployment name, content
                # filter, malformed body. Fail fast and loudly.
                raise ProviderError(f"azure returned {response.status_code}: {detail}")

            if attempt < self.config.max_retries:
                self._backoff(attempt)

        raise last_error or ProviderError("azure request failed with no detail")

    def _sleep_for(self, response, attempt: int) -> None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                # Honour the server's own pacing rather than guessing; Azure
                # returns this on 429 and ignoring it prolongs the throttle.
                time.sleep(min(float(retry_after), 10.0))
                return
            except ValueError:
                pass
        self._backoff(attempt)

    @staticmethod
    def _backoff(attempt: int) -> None:
        # Full jitter: synchronised retries from a fleet of workers are how a
        # transient throttle becomes a sustained one.
        time.sleep(random.uniform(0, min(2**attempt * 0.25, 4.0)))
