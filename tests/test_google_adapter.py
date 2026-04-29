"""Google Gemini adapter unit tests with the genai client mocked."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from brief_eval_rig.adapters.base import Clip
from brief_eval_rig.adapters.google import GoogleAdapter
from brief_eval_rig.adapters.pricing import GEMINI_3_FLASH_PREVIEW


@pytest.fixture
def gemini() -> GoogleAdapter:
    return GoogleAdapter(
        model_id="gemini-3-flash-preview",
        lineup_name="gemini-3-flash-preview",
        pricing=GEMINI_3_FLASH_PREVIEW,
        api_key="test-key",
    )


def _make_clip(video: Path) -> Clip:
    return Clip(clip_id="smp-001", video_path=video, duration_seconds=3.0, metadata={})


def _active_file(name: str = "files/abc") -> Any:
    f = MagicMock()
    f.name = name
    state = MagicMock()
    state.name = "ACTIVE"
    f.state = state
    return f


def _fake_response(text: str = "ok", in_tok: int = 500, out_tok: int = 100) -> Any:
    response = MagicMock()
    response.text = text
    response.usage_metadata = MagicMock(prompt_token_count=in_tok, candidates_token_count=out_tok)
    response.model_dump.return_value = {"id": "resp_1", "text": text}
    return response


def test_metadata(gemini: GoogleAdapter) -> None:
    assert gemini.name() == "gemini-3-flash-preview"
    assert gemini.lineage() == "google"
    assert gemini.deployment() == "cloud"
    assert gemini.supports_native_video() is True


def test_estimate_cost_positive(gemini: GoogleAdapter, synthetic_clip_path: Path) -> None:
    clip = _make_clip(synthetic_clip_path)
    assert gemini.estimate_cost(clip) > 0


def test_analyze_success(gemini: GoogleAdapter, synthetic_clip_path: Path) -> None:
    gemini._client.files.upload = MagicMock(return_value=_active_file())  # type: ignore[method-assign]
    gemini._client.models.generate_content = MagicMock(  # type: ignore[method-assign]
        return_value=_fake_response(text="Gemini reply.", in_tok=1000, out_tok=200)
    )
    clip = _make_clip(synthetic_clip_path)

    result = gemini.analyze(clip, "describe this clip")

    assert result.error is None
    assert result.summary == "Gemini reply."
    # cost = 1000/1M * $0.5 + 200/1M * $3 = $0.0005 + $0.0006 = $0.0011
    assert result.cost_usd == pytest.approx(0.0011, rel=1e-6)


def test_analyze_error_returns_result(gemini: GoogleAdapter, synthetic_clip_path: Path) -> None:
    def _boom(*a: Any, **kw: Any) -> Any:
        raise RuntimeError("upload failed")

    gemini._client.files.upload = MagicMock(side_effect=_boom)  # type: ignore[method-assign]
    clip = _make_clip(synthetic_clip_path)

    result = gemini.analyze(clip, "prompt")

    assert result.error is not None
    assert "upload failed" in result.error
    assert result.summary == ""
    assert result.cost_usd == 0.0


def test_wait_for_active_failure(gemini: GoogleAdapter, synthetic_clip_path: Path) -> None:
    failed = MagicMock()
    failed.name = "files/x"
    failed_state = MagicMock()
    failed_state.name = "FAILED"
    failed.state = failed_state

    gemini._client.files.upload = MagicMock(return_value=failed)  # type: ignore[method-assign]
    clip = _make_clip(synthetic_clip_path)

    result = gemini.analyze(clip, "prompt")
    assert result.error is not None
    assert "FAILED" in result.error or "processing failed" in result.error
