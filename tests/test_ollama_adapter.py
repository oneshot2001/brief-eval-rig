"""Ollama adapter unit tests using respx to mock httpx."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from brief_eval_rig.adapters.base import Clip
from brief_eval_rig.adapters.ollama import OllamaAdapter

_BASE = "http://localhost:11434"
_URL = f"{_BASE}/api/chat"


@pytest.fixture
def qwen_local() -> OllamaAdapter:
    return OllamaAdapter(
        model_id="qwen3.6:27b",
        lineup_name="qwen3-6-27b-local",
        lineage="qwen",
        timeout_s=5.0,
    )


@pytest.fixture
def nemotron_local() -> OllamaAdapter:
    return OllamaAdapter(
        model_id="nemotron3",
        lineup_name="nemotron-3-nano-omni-local",
        lineage="nvidia",
        max_output_tokens=4096,
        timeout_s=5.0,
    )


def _make_clip(video: Path) -> Clip:
    return Clip(clip_id="smp-001", video_path=video, duration_seconds=3.0, metadata={})


def _success_payload(content: str = "ok", in_tok: int = 1000, out_tok: int = 200) -> dict[str, Any]:
    return {
        "model": "qwen3.6:27b",
        "created_at": "2026-04-29T00:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": in_tok,
        "eval_count": out_tok,
        "total_duration": 9_000_000_000,
    }


def test_metadata(qwen_local: OllamaAdapter) -> None:
    assert qwen_local.name() == "qwen3-6-27b-local"
    assert qwen_local.lineage() == "qwen"
    assert qwen_local.deployment() == "local"
    assert qwen_local.supports_native_video() is False


def test_estimate_cost_zero(qwen_local: OllamaAdapter, synthetic_clip_path: Path) -> None:
    clip = _make_clip(synthetic_clip_path)
    assert qwen_local.estimate_cost(clip) == 0.0


@respx.mock
def test_analyze_success(qwen_local: OllamaAdapter, synthetic_clip_path: Path) -> None:
    respx.post(_URL).mock(
        return_value=httpx.Response(200, json=_success_payload(content="local reply"))
    )
    clip = _make_clip(synthetic_clip_path)
    result = qwen_local.analyze(clip, "describe this")
    assert result.error is None
    assert result.summary == "local reply"
    assert result.cost_usd == 0.0
    assert result.latency_ms >= 0


@respx.mock
def test_request_payload_shape(qwen_local: OllamaAdapter, synthetic_clip_path: Path) -> None:
    route = respx.post(_URL).mock(return_value=httpx.Response(200, json=_success_payload()))
    clip = _make_clip(synthetic_clip_path)
    qwen_local.analyze(clip, "prompt-text")

    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "qwen3.6:27b"
    assert sent["stream"] is False
    assert sent["options"]["num_predict"] == 1024
    user_msg = sent["messages"][0]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == "prompt-text"
    images = user_msg["images"]
    assert isinstance(images, list)
    assert len(images) >= 1
    for img in images:
        assert isinstance(img, str)
        assert len(img) > 100  # base64-encoded JPEG frame


@respx.mock
def test_falls_back_to_thinking_when_content_empty(
    nemotron_local: OllamaAdapter, synthetic_clip_path: Path
) -> None:
    payload = {
        "model": "nemotron3",
        "message": {"role": "assistant", "content": "", "thinking": "reasoning trace..."},
        "done": True,
        "prompt_eval_count": 500,
        "eval_count": 4000,
    }
    respx.post(_URL).mock(return_value=httpx.Response(200, json=payload))
    clip = _make_clip(synthetic_clip_path)
    result = nemotron_local.analyze(clip, "prompt")
    assert result.error is None
    assert result.summary == "reasoning trace..."


@respx.mock
def test_analyze_404_returns_result(qwen_local: OllamaAdapter, synthetic_clip_path: Path) -> None:
    respx.post(_URL).mock(return_value=httpx.Response(404, json={"error": "model not found"}))
    clip = _make_clip(synthetic_clip_path)
    result = qwen_local.analyze(clip, "prompt")
    assert result.error is not None
    assert result.summary == ""
    assert result.cost_usd == 0.0


def test_max_output_tokens_propagates(synthetic_clip_path: Path) -> None:
    """Nemotron-style large max_output_tokens must surface in the request payload."""
    adapter = OllamaAdapter(
        model_id="nemotron3",
        lineup_name="nemotron-local",
        lineage="nvidia",
        max_output_tokens=4096,
        timeout_s=5.0,
    )
    clip = _make_clip(synthetic_clip_path)
    with respx.mock:
        route = respx.post(_URL).mock(return_value=httpx.Response(200, json=_success_payload()))
        adapter.analyze(clip, "prompt")
    sent = json.loads(route.calls[0].request.content)
    assert sent["options"]["num_predict"] == 4096


def test_base_url_constructor_arg() -> None:
    adapter = OllamaAdapter(
        model_id="x",
        lineup_name="x",
        lineage="qwen",
        base_url="http://203.0.113.45:11434",
    )
    assert adapter._base_url == "http://203.0.113.45:11434"


def test_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://192.0.2.1:11434/")
    adapter = OllamaAdapter(model_id="x", lineup_name="x", lineage="qwen")
    assert adapter._base_url == "http://192.0.2.1:11434"  # trailing slash stripped


def test_base_url_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    adapter = OllamaAdapter(model_id="x", lineup_name="x", lineage="qwen")
    assert adapter._base_url == "http://localhost:11434"


def test_base_url_constructor_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://192.0.2.1:11434")
    adapter = OllamaAdapter(
        model_id="x",
        lineup_name="x",
        lineage="qwen",
        base_url="http://203.0.113.45:11434",
    )
    assert adapter._base_url == "http://203.0.113.45:11434"
