"""Policy retrieval backing the ``lookup_policy`` tool.

Two stores behind one interface:

* ``LocalPolicyStore`` reads the YAML catalog that the golden corpus labels
  cite. It is the default and the one the eval runs against, so a policy edit
  and a label edit cannot desync.
* ``AzureSearchPolicyStore`` queries Azure AI Search, which is what a real
  deployment wants once the catalog is larger than a file: the agent can find
  the relevant policy from a description without already knowing its id.

The local store still implements ``search``, using token overlap scoring. It
is not semantic and does not pretend to be -- but it means the agent's
retrieval path is exercised offline instead of being dead code until
credentials appear.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from worker.obs import get_logger

log = get_logger(__name__)

DEFAULT_CATALOG = Path(
    os.environ.get("POLICY_CATALOG", "tools/corpus/policies.yaml")
)

STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "is", "be",
    "must", "not", "that", "this", "it", "with", "as", "at", "by", "from",
})


@dataclass(frozen=True)
class Policy:
    id: str
    title: str
    kind: str
    severity: str
    summary: str
    rationale: str = ""
    window: str | None = None
    applies_when: str | None = None
    satisfied_by_phrases: tuple[str, ...] = ()
    prohibited_phrases: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "severity": self.severity,
            "summary": self.summary.strip(),
            "rationale": self.rationale.strip(),
            "window": self.window,
            "applies_when": self.applies_when,
            "satisfied_by_phrases": list(self.satisfied_by_phrases),
            "prohibited_phrases": list(self.prohibited_phrases),
        }


class PolicyStore(Protocol):
    name: str

    def get(self, policy_id: str) -> Policy | None: ...
    def search(self, query: str, top: int = 3) -> list[Policy]: ...
    def all(self) -> list[Policy]: ...
    def raw(self) -> dict: ...


def _to_policy(raw: dict) -> Policy:
    return Policy(
        id=raw["id"],
        title=raw.get("title", ""),
        kind=raw.get("kind", ""),
        severity=raw.get("severity", "medium"),
        summary=raw.get("summary", ""),
        rationale=raw.get("rationale", ""),
        window=raw.get("window"),
        applies_when=raw.get("applies_when"),
        satisfied_by_phrases=tuple(raw.get("satisfied_by_phrases") or ()),
        prohibited_phrases=tuple(raw.get("prohibited_phrases") or ()),
    )


class LocalPolicyStore:
    name = "local-yaml"

    def __init__(self, path: Path | None = None) -> None:
        import yaml

        self.path = path or DEFAULT_CATALOG
        if not self.path.exists():
            raise FileNotFoundError(
                f"policy catalog not found at {self.path}; set POLICY_CATALOG"
            )
        self._raw = yaml.safe_load(self.path.read_text())
        self._policies = [_to_policy(p) for p in self._raw.get("policies", [])]
        self._by_id = {p.id: p for p in self._policies}
        log.info("policy catalog loaded", path=str(self.path), count=len(self._policies),
                 version=self._raw.get("version", "?"))

    def get(self, policy_id: str) -> Policy | None:
        return self._by_id.get(policy_id.strip().upper())

    def search(self, query: str, top: int = 3) -> list[Policy]:
        terms = {t for t in re.findall(r"[a-z]+", query.lower()) if t not in STOPWORDS}
        if not terms:
            return []
        scored: list[tuple[float, Policy]] = []
        for policy in self._policies:
            haystack = " ".join([
                policy.title, policy.summary, policy.kind,
                " ".join(policy.prohibited_phrases),
                " ".join(policy.satisfied_by_phrases),
            ]).lower()
            tokens = set(re.findall(r"[a-z]+", haystack))
            overlap = len(terms & tokens)
            if overlap:
                # Normalise by query length so a long query is not
                # automatically a better match for every policy.
                scored.append((overlap / len(terms), policy))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [p for _, p in scored[:top]]

    def all(self) -> list[Policy]:
        return list(self._policies)

    def raw(self) -> dict:
        return self._raw


class AzureSearchPolicyStore:
    """Azure AI Search backed retrieval, with the local catalog as a fallback.

    Retrieval failures fall back rather than propagate. A search outage should
    degrade the agent's context, not stop compliance analysis: the fast-path
    rules still fire from the local catalog either way.
    """

    name = "azure-ai-search"

    def __init__(self, endpoint: str, api_key: str, index: str,
                 fallback: LocalPolicyStore | None = None) -> None:
        import httpx

        self.endpoint = endpoint.rstrip("/")
        self.index = index
        self.api_version = os.environ.get("AZURE_SEARCH_API_VERSION", "2024-07-01")
        self._fallback = fallback
        self._client = httpx.Client(
            timeout=httpx.Timeout(5.0, connect=3.0),
            headers={"api-key": api_key, "content-type": "application/json"},
        )

    def _post_search(self, body: dict) -> list[dict]:
        url = f"{self.endpoint}/indexes/{self.index}/docs/search?api-version={self.api_version}"
        response = self._client.post(url, json=body)
        response.raise_for_status()
        return response.json().get("value", [])

    def get(self, policy_id: str) -> Policy | None:
        try:
            docs = self._post_search({
                "search": "*",
                "filter": f"id eq '{policy_id.strip().upper()}'",
                "top": 1,
            })
            if docs:
                return _to_policy(docs[0])
        except Exception as exc:
            log.warning("azure ai search lookup failed; using local catalog",
                        policy_id=policy_id, error=str(exc))
        return self._fallback.get(policy_id) if self._fallback else None

    def search(self, query: str, top: int = 3) -> list[Policy]:
        try:
            docs = self._post_search({
                "search": query,
                "queryType": "semantic",
                "semanticConfiguration": os.environ.get(
                    "AZURE_SEARCH_SEMANTIC_CONFIG", "default"
                ),
                "top": top,
            })
            return [_to_policy(d) for d in docs]
        except Exception as exc:
            log.warning("azure ai search query failed; using local catalog",
                        error=str(exc))
        return self._fallback.search(query, top) if self._fallback else []

    def all(self) -> list[Policy]:
        return self._fallback.all() if self._fallback else []

    def raw(self) -> dict:
        return self._fallback.raw() if self._fallback else {"policies": []}


@lru_cache(maxsize=1)
def load_policy_store() -> PolicyStore:
    """Build the configured store once per process."""
    local = LocalPolicyStore()
    endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT", "").strip()
    api_key = os.environ.get("AZURE_SEARCH_API_KEY", "").strip()
    index = os.environ.get("AZURE_SEARCH_INDEX", "edgesense-policies").strip()

    if endpoint and api_key:
        log.info("using Azure AI Search for policy retrieval", index=index)
        return AzureSearchPolicyStore(endpoint, api_key, index, fallback=local)
    return local
