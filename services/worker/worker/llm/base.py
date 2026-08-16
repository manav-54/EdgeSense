"""Provider-agnostic chat + tool-calling interface.

Two implementations sit behind this: Azure OpenAI (generation) and a
deterministic offline provider. The offline one is not a mock in the usual
sense -- it implements the same tool-calling protocol, calls the same tools,
and produces the same ``Signal`` objects. That matters for three reasons:

* ``docker compose up`` works with no cloud credentials at all;
* the eval harness produces real, reproducible numbers today, which become the
  baseline that the Azure numbers are compared against;
* a provider outage degrades the pipeline to deterministic rules instead of
  taking live compliance detection offline entirely.

Nothing here is Azure-specific, so a third provider is a new file rather than
a refactor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """OpenAI/Azure chat-completions message shape."""
        if self.role is Role.TOOL:
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": self.content,
            }
        out: dict[str, Any] = {"role": self.role.value, "content": self.content or None}
        if self.tool_calls:
            import json

            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in self.tool_calls
            ]
        return out


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


@dataclass
class Completion:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    latency_ms: float = 0.0
    finish_reason: str = "stop"

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMProvider(Protocol):
    """Minimal surface the agent loop needs."""

    name: str
    model: str

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> Completion: ...


class ProviderError(RuntimeError):
    """Raised when a provider fails in a way retrying will not fix."""


class TransientProviderError(ProviderError):
    """Raised when a retry is worth attempting (429, 5xx, timeout)."""
