"""OpenRouter adapter unit tests using respx to mock httpx."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from brief_eval_rig.adapters.base import Clip
from brief_eval_rig.adapters.openrouter import OpenRouterAdapter
from brief_eval_rig.adapters.pricing import NEMOTRON_3_NANO_OMNI_FREE, QWEN_3_6_FLASH

_URL = "https://openrouter.ai/api/v1/chat/completions"


@pytest.fixture
def qwen_adapter() -> OpenRouterAdapter:
    return OpenRouterAdapter(
        model_id="qwen/qwen3.6-flash",
        lineup_name="qwen3-6-flash",
        lineage="qwen",
        pricing=QWEN_3_6_FLASH,
        api_key="test-key",
        timeout_s=5.0,
    )


@pytest.fixture
def nemotron_adapter() -> OpenRouterAdapter:
    return OpenRouterAdapter(
        model_id="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        lineup_name="nemotron-3-nano-omni-cloud",
        lineage="nvidia",
        pricing=NEMOTRON_3_NANO_OMNI_FREE,
        api_key="test-key",
        timeout_s=5.0,
    )


def _make_clip(video: Path) -> Clip:
    return Clip(clip_id="smp-001", video_path=video, duration_seconds=3.0, metadata={})


def _success_payload(content: str = "ok", in_tok: int = 800, out_tok: int = 100) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok},
    }


def test_metadata(qwen_adapter: OpenRouterAdapter) -> None:
    assert qwen_adapter.name() == "qwen3-6-flash"
    assert qwen_adapter.lineage() == "qwen"
    assert qwen_adapter.deployment() == "cloud"
    assert qwen_adapter.supports_native_video() is True


@respx.mock
def test_analyze_success_qwen(qwen_adapter: OpenRouterAdapter, synthetic_clip_path: Path) -> None:
    route = respx.post(_URL).mock(
        return_value=httpx.Response(
            200, json=_success_payload(content="Qwen reply.", in_tok=2000, out_tok=300)
        )
    )
    clip = _make_clip(synthetic_clip_path)

    result = qwen_adapter.analyze(clip, "describe this")

    assert route.called
    assert result.error is None
    assert result.summary == "Qwen reply."
    # Qwen pricing: 2000/1M * 0.25 + 300/1M * 1.50 = $0.0005 + $0.00045 = $0.00095
    assert result.cost_usd == pytest.approx(0.00095, rel=1e-6)


@respx.mock
def test_request_payload_shape(qwen_adapter: OpenRouterAdapter, synthetic_clip_path: Path) -> None:
    route = respx.post(_URL).mock(return_value=httpx.Response(200, json=_success_payload()))
    clip = _make_clip(synthetic_clip_path)
    qwen_adapter.analyze(clip, "prompt-text")

    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "qwen/qwen3.6-flash"
    user_msg = sent["messages"][0]
    assert user_msg["role"] == "user"
    blocks = user_msg["content"]
    assert blocks[0] == {"type": "text", "text": "prompt-text"}
    image_block = next(b for b in blocks if b["type"] == "image_url")
    url = image_block["image_url"]["url"]
    assert url.startswith("data:video/mp4;base64,")
    assert len(url) > len("data:video/mp4;base64,") + 100


@respx.mock
def test_analyze_http_error_returns_result(
    qwen_adapter: OpenRouterAdapter, synthetic_clip_path: Path
) -> None:
    respx.post(_URL).mock(return_value=httpx.Response(503, json={"error": "boom"}))
    clip = _make_clip(synthetic_clip_path)

    result = qwen_adapter.analyze(clip, "prompt")

    assert result.error is not None
    assert result.summary == ""
    assert result.cost_usd == 0.0


@respx.mock
def test_nemotron_zero_cost(nemotron_adapter: OpenRouterAdapter, synthetic_clip_path: Path) -> None:
    respx.post(_URL).mock(
        return_value=httpx.Response(200, json=_success_payload(in_tok=5000, out_tok=500))
    )
    clip = _make_clip(synthetic_clip_path)

    result = nemotron_adapter.analyze(clip, "prompt")

    assert result.error is None
    assert result.cost_usd == 0.0


def test_constructor_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("brief_eval_rig.adapters.openrouter.load_dotenv", lambda: None)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterAdapter(
            model_id="qwen/qwen3.6-flash",
            lineup_name="qwen3-6-flash",
            lineage="qwen",
            pricing=QWEN_3_6_FLASH,
        )
