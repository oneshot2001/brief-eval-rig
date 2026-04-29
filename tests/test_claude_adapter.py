"""Claude adapter unit tests with the Anthropic client mocked."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from brief_eval_rig.adapters.base import Clip
from brief_eval_rig.adapters.claude import ClaudeAdapter


@pytest.fixture
def claude(synthetic_clip_path: Path) -> ClaudeAdapter:
    return ClaudeAdapter(api_key="test-key")


def _make_clip(video: Path) -> Clip:
    return Clip(
        clip_id="smp-001",
        video_path=video,
        duration_seconds=3.0,
        metadata={},
    )


def _fake_anthropic_response(text: str = "ok", in_tok: int = 1500, out_tok: int = 200) -> Any:
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text

    response = MagicMock()
    response.content = [text_block]
    response.usage = MagicMock(input_tokens=in_tok, output_tokens=out_tok)
    response.model_dump.return_value = {"id": "msg_123", "content": [{"text": text}]}
    return response


def test_metadata(claude: ClaudeAdapter) -> None:
    assert claude.name() == "claude-sonnet-4-6"
    assert claude.lineage() == "anthropic"
    assert claude.deployment() == "cloud"
    assert claude.supports_native_video() is False


def test_estimate_cost_positive(claude: ClaudeAdapter, synthetic_clip_path: Path) -> None:
    clip = _make_clip(synthetic_clip_path)
    assert claude.estimate_cost(clip) > 0


def test_analyze_success(claude: ClaudeAdapter, synthetic_clip_path: Path) -> None:
    claude._client.messages.create = MagicMock(  # type: ignore[method-assign]
        return_value=_fake_anthropic_response(text="A summary.", in_tok=2000, out_tok=300)
    )
    clip = _make_clip(synthetic_clip_path)

    result = claude.analyze(clip, "describe this clip")

    assert result.error is None
    assert result.summary == "A summary."
    assert result.latency_ms >= 0
    # cost = 2000/1M * $3 + 300/1M * $15 = $0.006 + $0.0045 = $0.0105
    assert result.cost_usd == pytest.approx(0.0105, rel=1e-6)


def test_analyze_request_shape(claude: ClaudeAdapter, synthetic_clip_path: Path) -> None:
    captured: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _fake_anthropic_response()

    claude._client.messages.create = MagicMock(side_effect=_capture)  # type: ignore[method-assign]
    clip = _make_clip(synthetic_clip_path)

    claude.analyze(clip, "prompt-text")

    assert captured["model"] == "claude-sonnet-4-6"
    user_msg = captured["messages"][0]
    assert user_msg["role"] == "user"
    blocks = user_msg["content"]
    assert blocks[0] == {"type": "text", "text": "prompt-text"}
    image_blocks = [b for b in blocks if b.get("type") == "image"]
    assert len(image_blocks) >= 1
    for b in image_blocks:
        assert b["source"]["type"] == "base64"
        assert b["source"]["media_type"] == "image/jpeg"
        assert isinstance(b["source"]["data"], str)


def test_analyze_error_returns_result(claude: ClaudeAdapter, synthetic_clip_path: Path) -> None:
    def _boom(**kwargs: Any) -> Any:
        raise RuntimeError("api on fire")

    claude._client.messages.create = MagicMock(side_effect=_boom)  # type: ignore[method-assign]
    clip = _make_clip(synthetic_clip_path)

    result = claude.analyze(clip, "prompt")

    assert result.error is not None
    assert "api on fire" in result.error
    assert result.summary == ""
    assert result.cost_usd == 0.0


def test_constructor_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("brief_eval_rig.adapters.claude.load_dotenv", lambda: None)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ClaudeAdapter()
