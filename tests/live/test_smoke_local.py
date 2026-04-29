"""Live integration test for the local Ollama lineup — opt-in via `pytest -m live`.

Skips entirely if Ollama is not reachable. Per-adapter failures are tolerated
(the most common cause is a model not being pulled on the host); the test
asserts that at least one local adapter produced a non-empty summary.

Set ``OLLAMA_BASE_URL=http://<vast-ip>:11434`` to point the suite at a remote
Ollama (e.g., a Vast.ai A100 instance).
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from brief_eval_rig.adapters.registry import all_local_adapters
from brief_eval_rig.runner.orchestrator import VIDEO_EXTENSIONS, run_smoke

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_JSON = REPO_ROOT / "corpus" / "sample" / "sample-001.json"


def _ollama_base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def _ollama_reachable(url: str) -> bool:
    try:
        r = httpx.get(f"{url}/api/version", timeout=5.0)
        return r.status_code == 200
    except (httpx.NetworkError, httpx.TimeoutException):
        return False


def _find_real_video() -> Path | None:
    parent = SAMPLE_JSON.parent
    for ext in VIDEO_EXTENSIONS:
        candidate = parent / f"sample-001{ext}"
        if candidate.is_file() and candidate.stat().st_size > 1024:
            return candidate
    return None


@pytest.mark.live
def test_smoke_against_local_adapters(tmp_path: Path) -> None:
    base_url = _ollama_base_url()
    if not _ollama_reachable(base_url):
        pytest.skip(
            f"Ollama not reachable at {base_url}. "
            "Install + start Ollama locally, or set OLLAMA_BASE_URL to a remote instance."
        )

    video = _find_real_video()
    if video is None:
        pytest.skip("No real sample video at corpus/sample/sample-001.{mp4,mov,...}")

    report = run_smoke(
        SAMPLE_JSON,
        output_root=tmp_path / "outputs",
        adapters=all_local_adapters(base_url=base_url),
    )

    # Every local-adapter run must report $0 cost.
    assert report.total_cost_usd == 0.0
    assert all(s.cost_usd == 0.0 for s in report.per_adapter)

    # Tolerate per-model failures (model not pulled is the common case);
    # require at least one adapter to have produced output for all 3 prompts.
    successful = sum(1 for s in report.per_adapter if s.error_count == 0)
    error_lines = "\n".join(f"  - {e.adapter}/{e.prompt_kind}: {e.message}" for e in report.errors)
    assert successful >= 1, f"no local adapters succeeded; errors:\n{error_lines}"
