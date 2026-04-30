"""Registry / factory function tests — verify lineup composition without
hitting any network. API keys for cloud factories are stubbed via env-var
defaults (the .env file already populates them in the dev environment)."""

from __future__ import annotations

import pytest

from brief_eval_rig.adapters.base import VLMAdapter
from brief_eval_rig.adapters.registry import (
    all_adapters,
    all_cloud_adapters,
    all_local_adapters,
    gemma4_26b_local,
    nemotron_local,
    qwen3_6_27b_local,
)


def _names(adapters: list[VLMAdapter]) -> set[str]:
    return {a.name() for a in adapters}


def test_local_factories_instantiate() -> None:
    """Local factories don't need credentials; should always instantiate."""
    a = qwen3_6_27b_local()
    b = nemotron_local()
    c = gemma4_26b_local()
    assert a.deployment() == "local"
    assert b.deployment() == "local"
    assert c.deployment() == "local"
    assert a.lineage() == "qwen"
    assert b.lineage() == "nvidia"
    assert c.lineage() == "google"


def test_local_factories_respect_base_url_argument() -> None:
    a = qwen3_6_27b_local(base_url="http://203.0.113.45:11434")
    assert a._base_url == "http://203.0.113.45:11434"


def test_all_local_adapters_returns_three() -> None:
    adapters = all_local_adapters()
    assert len(adapters) == 3
    assert _names(adapters) == {
        "qwen3-6-27b-local",
        "nemotron-3-nano-omni-local",
        "gemma-4-26b-local",
    }


def test_all_local_adapters_propagates_base_url() -> None:
    adapters = all_local_adapters(base_url="http://203.0.113.45:11434")
    for a in adapters:
        assert a._base_url == "http://203.0.113.45:11434"  # type: ignore[attr-defined]


@pytest.mark.skipif("False")
def test_all_cloud_adapters_returns_five() -> None:
    """Requires .env with API keys; the dev environment already has them set."""
    adapters = all_cloud_adapters()
    assert len(adapters) == 5
    names = _names(adapters)
    assert "claude-sonnet-4-6" in names
    assert "gemini-3-1-pro-preview" in names
    assert "gemini-3-flash-preview" in names
    assert "qwen3-6-flash" in names
    assert "nemotron-3-nano-omni-cloud" in names


def test_all_adapters_returns_eight() -> None:
    """Five cloud + three local = eight contenders post-Phase-2."""
    adapters = all_adapters()
    assert len(adapters) == 8
    deployments = [a.deployment() for a in adapters]
    assert deployments.count("cloud") == 5
    assert deployments.count("local") == 3


def test_no_duplicate_names_across_lineup() -> None:
    """Every contender's `name()` must be unique — used as the JSON output filename."""
    adapters = all_adapters()
    names = [a.name() for a in adapters]
    assert len(names) == len(set(names)), f"duplicate adapter names: {names}"
