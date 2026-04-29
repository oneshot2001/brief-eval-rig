"""Live integration test — opt-in only via `pytest -m live`.

Hits real provider APIs and consumes credits. Run when you want to verify
the cloud adapters actually work end-to-end against a real clip.

Requires:
- `.env` populated with all four API keys
- A real video file alongside ``corpus/sample/sample-001.json``
  (any short mp4/mov/webm; if absent, the test fails early)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brief_eval_rig.runner.orchestrator import VIDEO_EXTENSIONS, run_smoke

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_JSON = REPO_ROOT / "corpus" / "sample" / "sample-001.json"


def _find_real_video() -> Path | None:
    parent = SAMPLE_JSON.parent
    for ext in VIDEO_EXTENSIONS:
        candidate = parent / f"sample-001{ext}"
        if candidate.is_file() and candidate.stat().st_size > 1024:
            return candidate
    return None


@pytest.mark.live
def test_smoke_against_all_five_cloud_adapters(tmp_path: Path) -> None:
    video = _find_real_video()
    if video is None:
        pytest.skip(
            "No real sample video found at corpus/sample/sample-001.{mp4,mov,...}; "
            "drop a short clip there to enable the live smoke test."
        )

    report = run_smoke(SAMPLE_JSON, output_root=tmp_path / "outputs")

    assert report.adapters_run == 5
    assert len(report.output_paths) == 5
    # Total cost for one clip × 3 prompts × 5 adapters should be well under $0.50.
    assert report.total_cost_usd < 0.50, f"unexpectedly high cost: ${report.total_cost_usd}"
    # We tolerate up to one adapter failing (free tier rate limits, preview tag churn);
    # but at least 4 of 5 must succeed for the smoke to count.
    successful_adapters = sum(1 for s in report.per_adapter if s.error_count == 0)
    assert (
        successful_adapters >= 4
    ), f"only {successful_adapters}/5 adapters succeeded; errors: {report.errors}"
