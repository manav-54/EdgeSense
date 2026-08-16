from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = REPO_ROOT / "services" / "edge-agent"

for p in (str(SERVICE_ROOT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

GOLDEN_DIR = REPO_ROOT / "eval" / "golden" / "calls"
AUDIO_DIR = REPO_ROOT / "data" / "audio"


@pytest.fixture(scope="session")
def golden_calls() -> list[dict]:
    if not GOLDEN_DIR.exists():
        pytest.skip("golden corpus missing; run `python -m tools.corpus.generate`")
    calls = [json.loads(p.read_text()) for p in sorted(GOLDEN_DIR.glob("*.json"))]
    if not calls:
        pytest.skip("golden corpus is empty")
    return calls


@pytest.fixture(scope="session")
def audio_dir() -> Path:
    return AUDIO_DIR
