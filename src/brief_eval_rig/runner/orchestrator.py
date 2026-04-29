"""One-clip smoke test orchestrator and CLI entry point.

Loads a clip via the corpus loader, locates its sibling video file, runs every
configured adapter sequentially against all three canonical prompts, and writes
one JSON output file per adapter. Returns a SmokeReport with aggregate cost,
latency, and any per-call errors.

CLI invocation:
    python -m brief_eval_rig.runner.orchestrator <clip_json_path>
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from brief_eval_rig.adapters.base import AnalysisResult, Clip, VLMAdapter
from brief_eval_rig.adapters.registry import all_cloud_adapters
from brief_eval_rig.corpus.loader import load_clip
from brief_eval_rig.prompts.loader import (
    load_generic,
    load_hallucination_trap,
    load_targeted,
)
from brief_eval_rig.runner.output_writer import write_result

VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v")


class SmokeError(BaseModel):
    adapter: str
    prompt_kind: str
    message: str


class AdapterSummary(BaseModel):
    adapter: str
    latency_ms: int
    cost_usd: float
    error_count: int


class SmokeReport(BaseModel):
    clip_id: str
    adapters_run: int
    total_cost_usd: float
    total_latency_ms: int
    per_adapter: list[AdapterSummary]
    errors: list[SmokeError]
    output_paths: list[str]


def _find_video_for(clip_json_path: Path) -> Path:
    stem = clip_json_path.stem
    parent = clip_json_path.parent
    for ext in VIDEO_EXTENSIONS:
        candidate = parent / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No video file found alongside {clip_json_path} "
        f"(looked for {stem}{{{','.join(VIDEO_EXTENSIONS)}}})"
    )


def run_smoke(
    clip_json_path: Path,
    output_root: Path = Path("outputs"),
    adapters: list[VLMAdapter] | None = None,
) -> SmokeReport:
    """Run one clip × all configured adapters × 3 prompts and write results."""
    clip_meta = load_clip(clip_json_path)
    video_path = _find_video_for(clip_json_path)
    clip = Clip(
        clip_id=clip_meta.clip_id,
        video_path=video_path,
        duration_seconds=clip_meta.duration_seconds,
        metadata=clip_meta.model_dump(),
    )

    prompts = {
        "generic": load_generic(),
        "targeted": load_targeted(clip_meta.targeted_query),
        "hallucination_trap": load_hallucination_trap(clip_meta.hallucination_trap),
    }

    adapter_list = adapters if adapters is not None else all_cloud_adapters()
    per_adapter: list[AdapterSummary] = []
    errors: list[SmokeError] = []
    output_paths: list[str] = []
    total_cost = 0.0
    total_latency = 0

    for adapter in adapter_list:
        results: dict[str, AnalysisResult] = {}
        adapter_latency = 0
        adapter_cost = 0.0
        adapter_errors = 0
        for kind, prompt_text in prompts.items():
            result = adapter.analyze(clip, prompt_text)
            results[kind] = result
            adapter_latency += result.latency_ms
            adapter_cost += result.cost_usd
            if result.error:
                adapter_errors += 1
                errors.append(
                    SmokeError(adapter=adapter.name(), prompt_kind=kind, message=result.error)
                )
        out_path = write_result(output_root, clip.clip_id, adapter, results)
        output_paths.append(str(out_path))
        per_adapter.append(
            AdapterSummary(
                adapter=adapter.name(),
                latency_ms=adapter_latency,
                cost_usd=adapter_cost,
                error_count=adapter_errors,
            )
        )
        total_cost += adapter_cost
        total_latency += adapter_latency

    return SmokeReport(
        clip_id=clip.clip_id,
        adapters_run=len(adapter_list),
        total_cost_usd=total_cost,
        total_latency_ms=total_latency,
        per_adapter=per_adapter,
        errors=errors,
        output_paths=output_paths,
    )


def _print_report(report: SmokeReport) -> None:
    header = f"{'ADAPTER':<32} {'LATENCY':>10} {'COST':>10} {'ERRORS':>8}"
    print(header)
    print("-" * len(header))
    for s in report.per_adapter:
        print(
            f"{s.adapter:<32} {s.latency_ms / 1000:>9.1f}s {s.cost_usd:>9.4f}$ {s.error_count:>8}"
        )
    print("-" * len(header))
    print(
        f"{'TOTAL':<32} {report.total_latency_ms / 1000:>9.1f}s "
        f"{report.total_cost_usd:>9.4f}$ {len(report.errors):>8}"
    )
    if report.errors:
        print("\nErrors:")
        for e in report.errors:
            print(f"  - {e.adapter} / {e.prompt_kind}: {e.message}")
    print(f"\nOutputs written: {len(report.output_paths)} files")


def main(argv: list[str] | None = None) -> int:
    args: list[str] = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(
            "Usage: python -m brief_eval_rig.runner.orchestrator <clip_json_path>",
            file=sys.stderr,
        )
        return 2
    clip_json = Path(args[0])
    if not clip_json.is_file():
        print(f"Clip JSON not found: {clip_json}", file=sys.stderr)
        return 2
    report = run_smoke(clip_json)
    _print_report(report)
    return 1 if report.errors else 0


def _ignored_unused() -> Any:
    """Exists so mypy doesn't warn on Any import for typing only."""
    return None


if __name__ == "__main__":
    raise SystemExit(main())
