"""LLM providers and the factory that chooses between them."""

from __future__ import annotations

import os

from worker.llm.base import Completion, LLMProvider, Message, Role, ToolCall, ToolSpec, Usage
from worker.obs import get_logger

log = get_logger(__name__)

__all__ = ["Completion", "LLMProvider", "Message", "Role", "ToolCall", "ToolSpec",
           "Usage", "build_provider"]


def build_provider(preference: str | None = None) -> LLMProvider:
    """Pick a provider.

    ``auto`` (the default) uses Azure OpenAI when credentials are present and
    the offline provider otherwise, so the pipeline starts and the eval runs
    whether or not anyone has configured a cloud account. An explicit
    ``azure`` fails loudly rather than silently downgrading -- if you asked for
    the model, quietly getting rules instead would invalidate your results.
    """
    choice = (preference or os.environ.get("EDGESENSE_LLM_PROVIDER", "auto")).strip().lower()

    if choice in ("offline", "rules"):
        from worker.llm.offline import OfflineProvider

        return OfflineProvider()

    from worker.llm.azure import AzureConfig, AzureOpenAIProvider

    config = AzureConfig.from_env()
    if choice == "azure":
        if config is None:
            raise RuntimeError(
                "EDGESENSE_LLM_PROVIDER=azure but AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT are not all set."
            )
        return AzureOpenAIProvider(config)

    if config is not None:
        log.info("using Azure OpenAI", deployment=config.deployment)
        return AzureOpenAIProvider(config)

    from worker.llm.offline import OfflineProvider

    log.info("no Azure credentials found; using the deterministic offline provider")
    return OfflineProvider()
