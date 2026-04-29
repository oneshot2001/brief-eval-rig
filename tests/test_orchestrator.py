"""Orchestrator tests use stub adapters so no API calls fire."""

from __future__ import annotations

import json
from pathlib import Path

from brief_eval_rig.adapters.base import AnalysisResult, Clip, Deployment, Lineage, VLMAdapter
from brief_eval_rig.runner.orchestrator import _find_video_for, run_smoke


class _StubAdapter(VLMAdapter):
    def __init__(
        self, name_: str, lineage_: Lineage, error_on: str | None = None, cost: float = 0.01
    ) -> None:
        self._name = name_
        self._lineage = lineage_
        self._error_on = error_on
        self._cost = cost
        self.calls: list[tuple[str, str]] = []

    def name(self) -> str:
        return self._name

    def lineage(self) -> Lineage:
        return self._lineage

    def deployment(self) -> Deployment:
        return "cloud"

    def supports_native_video(self) -> bool:
        return True

    def estimate_cost(self, clip: Clip) -> float:
        return self._cost

    def analyze(self, clip: Clip, prompt: str) -> AnalysisResult:
        # Identify which prompt kind by checking marker text injected by the test setup.
        kind = (
            "generic"
            if prompt.startswith("You are analyzing security camera footage.")
            and "Question:" not in prompt
            else (
                "targeted"
                if prompt.count("Question:") == 1 and "license plate" not in prompt
                else "hallucination_trap"
            )
        )
        self.calls.append((kind, prompt[:30]))
        if self._error_on == kind:
            return AnalysisResult(
                summary="",
                raw_response={},
                latency_ms=10,
                cost_usd=0.0,
                error="forced failure",
            )
        return AnalysisResult(
            summary=f"{self._name}/{kind}",
            raw_response={},
            latency_ms=100,
            cost_usd=self._cost,
            error=None,
        )


def test_find_video_for_finds_mp4(tmp_path: Path) -> None:
    json_path = tmp_path / "smp-001.json"
    json_path.write_text("{}")
    video = tmp_path / "smp-001.mp4"
    video.write_bytes(b"\x00")
    assert _find_video_for(json_path) == video


def test_find_video_raises_when_missing(tmp_path: Path) -> None:
    json_path = tmp_path / "smp-001.json"
    json_path.write_text("{}")
    try:
        _find_video_for(json_path)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_run_smoke_writes_one_file_per_adapter(
    tmp_path: Path, valid_payload: dict[str, object]
) -> None:
    clip_dir = tmp_path / "corpus" / "sample"
    clip_dir.mkdir(parents=True)
    json_path = clip_dir / "sample-001.json"
    json_path.write_text(json.dumps(valid_payload))
    (clip_dir / "sample-001.mp4").write_bytes(b"\x00\x01")

    a = _StubAdapter("alpha-1", "qwen")
    b = _StubAdapter("beta-1", "google")
    out_root = tmp_path / "outputs"

    report = run_smoke(json_path, output_root=out_root, adapters=[a, b])

    assert report.adapters_run == 2
    assert len(report.output_paths) == 2
    assert (out_root / "smp-001" / "alpha-1.json").exists()
    assert (out_root / "smp-001" / "beta-1.json").exists()
    # Each adapter received exactly 3 prompts.
    assert len(a.calls) == 3
    assert len(b.calls) == 3
    # All three prompt kinds were exercised.
    assert {k for k, _ in a.calls} == {"generic", "targeted", "hallucination_trap"}


def test_run_smoke_reports_errors(tmp_path: Path, valid_payload: dict[str, object]) -> None:
    clip_dir = tmp_path / "corpus" / "sample"
    clip_dir.mkdir(parents=True)
    json_path = clip_dir / "sample-001.json"
    json_path.write_text(json.dumps(valid_payload))
    (clip_dir / "sample-001.mp4").write_bytes(b"\x00")

    a = _StubAdapter("alpha-1", "qwen", error_on="targeted")
    out_root = tmp_path / "outputs"

    report = run_smoke(json_path, output_root=out_root, adapters=[a])

    assert len(report.errors) == 1
    assert report.errors[0].adapter == "alpha-1"
    assert report.errors[0].prompt_kind == "targeted"


def test_run_smoke_aggregates_cost(tmp_path: Path, valid_payload: dict[str, object]) -> None:
    clip_dir = tmp_path / "corpus" / "sample"
    clip_dir.mkdir(parents=True)
    json_path = clip_dir / "sample-001.json"
    json_path.write_text(json.dumps(valid_payload))
    (clip_dir / "sample-001.mp4").write_bytes(b"\x00")

    a = _StubAdapter("alpha-1", "qwen", cost=0.01)
    b = _StubAdapter("beta-1", "google", cost=0.02)
    out_root = tmp_path / "outputs"

    report = run_smoke(json_path, output_root=out_root, adapters=[a, b])
    # 3 calls per adapter × cost
    assert report.total_cost_usd == 0.01 * 3 + 0.02 * 3
