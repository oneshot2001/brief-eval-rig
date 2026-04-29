"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_JSON_PATH = REPO_ROOT / "corpus" / "sample" / "sample-001.json"


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    """Fresh dict copy of the canonical sample sidecar.

    Tests freely mutate the returned dict; the fixture is recomputed per test
    so mutations don't bleed across tests.
    """
    return json.loads(SAMPLE_JSON_PATH.read_text(encoding="utf-8"))
