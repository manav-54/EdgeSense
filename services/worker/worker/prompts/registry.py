"""Versioned prompt loading.

Prompts live on disk as ``prompts/<id>/<version>.yaml`` rather than in string
literals, for one reason: a prompt change is a behaviour change, and it should
be reviewable and revertable like any other. Adding a version means adding a
file, so the regression suite can run v1 and v2 over the same corpus and print
a before/after table without anyone editing the harness.

Each loaded prompt carries a content hash. The hash goes onto every Signal and
CallSummary, so a result in ClickHouse can be traced back to the exact prompt
text that produced it -- including the case where someone edited a file in
place without bumping the version.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Note: `Path("") or default` does not work -- Path("") is PosixPath('.'),
# which is truthy, so an unset PROMPT_DIR would silently resolve to the
# process's working directory.
_PROMPT_DIR_ENV = os.environ.get("PROMPT_DIR", "").strip()
PROMPT_DIR = Path(_PROMPT_DIR_ENV) if _PROMPT_DIR_ENV else Path(__file__).parent

VERSION_RE = re.compile(r"^v(\d+)$")


@dataclass(frozen=True)
class Prompt:
    id: str
    version: str
    description: str
    system: str
    user_template: str
    content_hash: str
    max_steps: int = 4
    temperature: float = 0.0

    @property
    def ref(self) -> str:
        """Stable identifier for provenance: ``live_analysis@v2:9f31c0``."""
        return f"{self.id}@{self.version}:{self.content_hash[:6]}"

    def render_user(self, **kwargs: object) -> str:
        """Fill the template. Missing keys fail loudly rather than silently.

        ``str.format`` would raise KeyError on a missing field but also choke
        on literal braces in transcript text, so substitution is explicit.
        """
        out = self.user_template
        for key, value in kwargs.items():
            out = out.replace("{{" + key + "}}", str(value))
        leftover = re.findall(r"\{\{(\w+)\}\}", out)
        if leftover:
            raise KeyError(
                f"prompt {self.ref} has unfilled placeholders: {sorted(set(leftover))}"
            )
        return out


def _load_file(path: Path, prompt_id: str, version: str) -> Prompt:
    import yaml

    raw = yaml.safe_load(path.read_text()) or {}
    system = raw.get("system", "").strip()
    user_template = raw.get("user_template", "").strip()
    if not system or not user_template:
        raise ValueError(f"prompt {path} must define both system and user_template")

    digest = hashlib.sha256(
        (system + "\x00" + user_template).encode("utf-8")
    ).hexdigest()

    return Prompt(
        id=prompt_id,
        version=version,
        description=raw.get("description", "").strip(),
        system=system,
        user_template=user_template,
        content_hash=digest,
        max_steps=int(raw.get("max_steps", 4)),
        temperature=float(raw.get("temperature", 0.0)),
    )


@lru_cache(maxsize=64)
def get(prompt_id: str, version: str = "latest") -> Prompt:
    """Load a prompt. ``latest`` resolves to the highest numeric version."""
    directory = PROMPT_DIR / prompt_id
    if not directory.is_dir():
        raise FileNotFoundError(f"no prompt directory for {prompt_id!r} at {directory}")

    if version == "latest":
        version = max(versions(prompt_id), key=_version_sort_key)

    path = directory / f"{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"prompt {prompt_id}@{version} not found; have {versions(prompt_id)}"
        )
    return _load_file(path, prompt_id, version)


def versions(prompt_id: str) -> list[str]:
    directory = PROMPT_DIR / prompt_id
    if not directory.is_dir():
        return []
    found = sorted(
        (p.stem for p in directory.glob("v*.yaml")), key=_version_sort_key
    )
    if not found:
        raise FileNotFoundError(f"prompt directory {directory} contains no versions")
    return found


def prompt_ids() -> list[str]:
    """Directories that actually contain versioned prompts."""
    return sorted(
        p.name
        for p in PROMPT_DIR.iterdir()
        if p.is_dir() and not p.name.startswith((".", "_")) and any(p.glob("v*.yaml"))
    )


def _version_sort_key(version: str) -> tuple[int, str]:
    m = VERSION_RE.match(version)
    return (int(m.group(1)), version) if m else (10_000, version)
