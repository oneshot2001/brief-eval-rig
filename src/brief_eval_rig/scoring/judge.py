"""CLI: walk outputs/, grade per-clip per-model JSONs, write to SQLite + JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from brief_eval_rig.corpus.loader import load_corpus
from brief_eval_rig.corpus.schema import ClipMetadata
from brief_eval_rig.runner.output_writer import update_with_judge
from brief_eval_rig.scoring.llm_judge import LLMJudge
from brief_eval_rig.scoring.storage import (
    DimensionScore,
    JudgeRun,
    connect,
    init_schema,
    upsert_judge_run,
)

_DRY_RUN_INPUT_TOKENS = 5000
_DRY_RUN_OUTPUT_TOKENS = 1500
_OPUS_INPUT_PRICE = 15.0 / 1_000_000
_OPUS_OUTPUT_PRICE = 75.0 / 1_000_000
_GPT5_INPUT_PRICE = 2.50 / 1_000_000
_GPT5_OUTPUT_PRICE = 10.0 / 1_000_000


def _discover_pairs(path: Path) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    if path.is_file() and path.suffix == ".json":
        clip_id = path.parent.name
        pairs.append((clip_id, path))
        return pairs
    if not path.is_dir():
        return pairs
    has_json_children = any(p.suffix == ".json" for p in path.iterdir() if p.is_file())
    if has_json_children:
        clip_id = path.name
        for p in sorted(path.iterdir()):
            if p.is_file() and p.suffix == ".json":
                pairs.append((clip_id, p))
        return pairs
    for sub in sorted(path.iterdir()):
        if sub.is_dir():
            pairs.extend(_discover_pairs(sub))
    return pairs


def _load_clip_index(corpus_dir: Path) -> dict[str, ClipMetadata]:
    return {clip.clip_id: clip for clip in load_corpus(corpus_dir)}


def _read_json(path: Path) -> dict[str, Any]:
    obj: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return obj


def _is_clean(model_output: dict[str, Any]) -> bool:
    prompts = model_output.get("prompts") or {}
    for kind in ("generic", "targeted", "hallucination_trap"):
        if (prompts.get(kind) or {}).get("error"):
            return False
        if kind not in prompts:
            return False
    return True


def _compute_lineup_minimums(clip_dir: Path) -> tuple[int, float]:
    min_latency: int | None = None
    costs: list[float] = []
    for p in sorted(clip_dir.iterdir()):
        if not (p.is_file() and p.suffix == ".json"):
            continue
        try:
            data = _read_json(p)
        except json.JSONDecodeError:
            continue
        if not _is_clean(data):
            continue
        prompts = data["prompts"]
        latency = int(prompts["generic"].get("latency_ms") or 0)
        cost = sum(
            float(prompts[k].get("cost_usd") or 0.0)
            for k in ("generic", "targeted", "hallucination_trap")
        )
        if min_latency is None or latency < min_latency:
            min_latency = latency
        costs.append(cost)
    if min_latency is None:
        min_latency = 0
    min_paid_cost = min((c for c in costs if c > 0.0), default=0.0)
    return min_latency, min_paid_cost


def _estimate_cost(lineage: str) -> float:
    if lineage == "anthropic":
        return (
            _DRY_RUN_INPUT_TOKENS * _GPT5_INPUT_PRICE + _DRY_RUN_OUTPUT_TOKENS * _GPT5_OUTPUT_PRICE
        )
    return _DRY_RUN_INPUT_TOKENS * _OPUS_INPUT_PRICE + _DRY_RUN_OUTPUT_TOKENS * _OPUS_OUTPUT_PRICE


def _format_pair_line(clip_id: str, model: str, run: JudgeRun) -> str:
    if run.error:
        return f"{clip_id}/{model}: ERROR: {run.error}"
    composite = f"{run.composite:.2f}" if run.composite is not None else "None"
    cost = run.judge_cost_usd if run.judge_cost_usd is not None else 0.0
    return f"{clip_id}/{model}: judge={run.judge_model} composite={composite} cost=${cost:.4f}"


def _build_judge(args: argparse.Namespace) -> LLMJudge:
    return LLMJudge()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brief_eval_rig.scoring.judge")
    parser.add_argument("path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--db", type=Path, default=Path("outputs/scores.db"))
    parser.add_argument("--corpus", type=Path, default=Path("corpus/"))
    args = parser.parse_args(argv)

    pairs = _discover_pairs(args.path)
    if not pairs:
        print(f"no model output JSONs found under {args.path}", file=sys.stderr)
        return 1

    clip_index = _load_clip_index(args.corpus)

    judge: LLMJudge | None = None
    conn = None
    if not args.dry_run:
        args.db.parent.mkdir(parents=True, exist_ok=True)
        conn = connect(args.db)
        init_schema(conn)
        judge = _build_judge(args)

    graded = 0
    skipped = 0
    errors = 0
    total_cost = 0.0

    lineup_cache: dict[Path, tuple[int, float]] = {}

    for clip_id, json_path in pairs:
        try:
            model_output = _read_json(json_path)
        except json.JSONDecodeError as e:
            print(f"{clip_id}/{json_path.name}: ERROR: invalid JSON ({e})")
            errors += 1
            continue
        model_name = str(model_output.get("model") or json_path.stem)

        if model_output.get("scores") is not None and not args.force:
            print(f"skip {clip_id}/{model_name}: already graded (use --force to re-grade)")
            skipped += 1
            continue

        clip_meta = clip_index.get(clip_id)
        if clip_meta is None:
            print(f"{clip_id}/{model_name}: ERROR: clip metadata not found in corpus")
            errors += 1
            continue

        clip_dir = json_path.parent
        if clip_dir not in lineup_cache:
            lineup_cache[clip_dir] = _compute_lineup_minimums(clip_dir)
        min_latency, min_paid_cost = lineup_cache[clip_dir]

        if args.dry_run:
            est = _estimate_cost(str(model_output.get("lineage") or ""))
            print(f"would grade {clip_id}/{model_name}: estimated cost ${est:.4f}")
            graded += 1
            total_cost += est
            continue

        assert judge is not None
        assert conn is not None
        run, dimension_scores = judge.judge_pair(
            clip_metadata=clip_meta,
            model_output=model_output,
            lineup_min_latency_ms=min_latency,
            lineup_min_paid_cost_usd=min_paid_cost,
        )
        upsert_judge_run(conn, run, dimension_scores)
        update_with_judge(json_path, run, dimension_scores)
        print(_format_pair_line(clip_id, model_name, run))
        if run.error:
            errors += 1
        else:
            graded += 1
        if run.judge_cost_usd:
            total_cost += run.judge_cost_usd

    if conn is not None:
        conn.close()

    print(
        f"graded {graded} pairs, skipped {skipped}, errors {errors}, "
        f"total cost ${total_cost:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DimensionScore",
    "JudgeRun",
    "main",
]
