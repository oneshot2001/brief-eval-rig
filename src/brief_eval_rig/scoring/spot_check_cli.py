"""Interactive spot-check CLI: stratified sampling, human override capture, session resume."""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from brief_eval_rig.corpus.loader import load_corpus
from brief_eval_rig.scoring.rubric import DIMENSIONS
from brief_eval_rig.scoring.storage import (
    Dimension,
    JudgeRun,
    SpotCheck,
    connect,
    init_schema,
    iter_judge_runs,
    iter_spot_checks,
    record_spot_check,
)

SESSION_FILENAME = ".spot_check_session.json"
SAMPLE_SEED = 42


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _session_path(outputs_dir: Path) -> Path:
    return outputs_dir / SESSION_FILENAME


def _load_session(outputs_dir: Path) -> dict[str, Any]:
    path = _session_path(outputs_dir)
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return raw


def _write_session(outputs_dir: Path, payload: dict[str, Any]) -> None:
    path = _session_path(outputs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _completed_judge_run_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT DISTINCT judge_run_id FROM spot_checks").fetchall()
    return {row["judge_run_id"] for row in rows}


def _vertical_map(corpus_dir: Path) -> dict[str, str]:
    clips = load_corpus(corpus_dir)
    return {c.clip_id: c.vertical for c in clips}


def _gradeable_runs(conn: sqlite3.Connection) -> list[JudgeRun]:
    out: list[JudgeRun] = []
    for run, _ in iter_judge_runs(conn):
        if run.composite is None or run.error is not None:
            continue
        out.append(run)
    return out


def stratified_sample(
    runs: list[JudgeRun],
    vertical_by_clip: dict[str, str],
    already_done: set[str],
    sample_size: int,
    *,
    seed: int = SAMPLE_SEED,
) -> list[str]:
    eligible = [r for r in runs if r.judge_run_id not in already_done]
    if not eligible:
        return []

    by_model: dict[str, list[JudgeRun]] = defaultdict(list)
    for r in eligible:
        by_model[r.model].append(r)

    n_models = len(by_model)
    if n_models == 0:
        return []
    per_model = max(1, sample_size // n_models)

    rng = random.Random(seed)
    picks: list[str] = []
    for model in sorted(by_model):
        model_runs = by_model[model]
        by_vertical: dict[str, list[JudgeRun]] = defaultdict(list)
        for r in model_runs:
            v = vertical_by_clip.get(r.clip_id, "_unknown")
            by_vertical[v].append(r)
        for v in by_vertical:
            rng.shuffle(by_vertical[v])
        verticals = sorted(by_vertical)
        chosen: list[JudgeRun] = []
        idx = 0
        while len(chosen) < per_model:
            progressed = False
            for v in verticals:
                if len(chosen) >= per_model:
                    break
                bucket = by_vertical[v]
                if idx < len(bucket):
                    chosen.append(bucket[idx])
                    progressed = True
            if not progressed:
                break
            idx += 1
        picks.extend(c.judge_run_id for c in chosen)

    rng.shuffle(picks)
    return picks[:sample_size]


def _load_payload(json_path: Path) -> dict[str, Any]:
    raw: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    return raw


def _write_payload(json_path: Path, payload: dict[str, Any]) -> None:
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _judge_score_for(run_id: str, dim: str, conn: sqlite3.Connection) -> tuple[int | None, str]:
    row = conn.execute(
        "SELECT score, justification FROM dimension_scores "
        "WHERE judge_run_id = ? AND dimension = ?",
        (run_id, dim),
    ).fetchone()
    if row is None:
        return None, ""
    score = row["score"]
    justification = row["justification"] or ""
    return (None if score is None else int(score)), justification


def _format_score_cell(score: int | None) -> str:
    return "N/A " if score is None else f"{score}/10"


def _print_pair_header(*, idx: int, total: int, run: JudgeRun, vertical: str) -> None:
    bar = "=" * 56
    print(bar)
    print(
        f"Spot check {idx} of {total}  |  clip={run.clip_id}  "
        f"model={run.model}  vertical={vertical}"
    )
    print(bar)
    print()


def _print_ground_truth(clip_id: str, corpus_dir: Path) -> None:
    clips = load_corpus(corpus_dir)
    by_id = {c.clip_id: c for c in clips}
    clip = by_id.get(clip_id)
    print("GROUND TRUTH")
    if clip is None:
        print(f"  (no metadata found for {clip_id})")
        print()
        return
    events = "; ".join(f"{e.time} {e.event}" for e in clip.expected_temporal_events)
    print(f"  Summary: {clip.ground_truth_summary}")
    print(f"  Objects: {', '.join(clip.ground_truth_objects)}")
    print(f"  OCR:     {', '.join(clip.ground_truth_ocr) if clip.ground_truth_ocr else '(none)'}")
    print(f"  Events:  {events}")
    print(f"  Targeted query: {clip.targeted_query}")
    print(f"  Hallucination trap: {clip.hallucination_trap}")
    print()


def _print_model_output(payload: dict[str, Any]) -> None:
    prompts = payload.get("prompts", {}) or {}
    print("MODEL OUTPUT")
    for key, label in (
        ("generic", "Generic"),
        ("targeted", "Targeted"),
        ("hallucination_trap", "Hallucination trap"),
    ):
        block = prompts.get(key) or {}
        response = block.get("response", "") or "(empty)"
        print(f"  {label:<19}: {response}")
    print()


def _print_judge_scores(
    run: JudgeRun, conn: sqlite3.Connection
) -> dict[str, tuple[int | None, str]]:
    composite_str = "—" if run.composite is None else f"{run.composite:.2f}"
    print(f"JUDGE SCORES (judge={run.judge_model}, composite={composite_str})")
    judge_data: dict[str, tuple[int | None, str]] = {}
    for dim in DIMENSIONS:
        score, justification = _judge_score_for(run.judge_run_id, dim, conn)
        judge_data[dim] = (score, justification)
        cell = _format_score_cell(score)
        just = justification or ""
        print(f"  {dim:<23}: {cell:<5} — {just}")
    print()
    return judge_data


def _prompt_dimension(dim: str, judge_score: int | None) -> str:
    label = "N/A" if judge_score is None else str(judge_score)
    return f"  {dim:<23} [judge={label}]: "


def _parse_human_input(
    raw: str, judge_score: int | None, dim: str
) -> tuple[str, int | None] | None:
    """Returns (action, human_score). action in {"recorded", "quit"}. None means re-prompt."""
    text = raw.strip().lower()
    if text == "q":
        return "quit", None
    if text == "":
        return "recorded", judge_score
    if text == "n":
        if dim == "ocr_fidelity" and judge_score is not None:
            print("    warning: N/A entered but judge gave a score")
        elif dim != "ocr_fidelity":
            print("    warning: N/A typically only valid for ocr_fidelity")
        return "recorded", None
    try:
        n = int(text)
    except ValueError:
        return None
    if n < 0 or n > 10:
        return None
    return "recorded", n


def _disagreement(judge_score: int | None, human_score: int | None) -> int | None:
    if judge_score is None or human_score is None:
        return None
    return abs(judge_score - human_score)


def _record_pair(
    *,
    run: JudgeRun,
    judge_data: dict[str, tuple[int | None, str]],
    human_scores: dict[str, int | None],
    conn: sqlite3.Connection,
    outputs_dir: Path,
) -> int:
    """Writes 7 spot_checks rows + mutates per-clip/model JSON. Returns disagreement count."""
    now = _utc_now()
    disagreements_above_threshold: list[str] = []
    total_disagreement = 0
    for dim in DIMENSIONS:
        judge_score, _ = judge_data[dim]
        human_score = human_scores[dim]
        delta = _disagreement(judge_score, human_score)
        sc = SpotCheck(
            spot_check_id=uuid.uuid4().hex,
            judge_run_id=run.judge_run_id,
            dimension=cast(Dimension, dim),
            judge_score=judge_score,
            human_score=human_score,
            disagreement=delta,
            notes=None,
            checked_at=now,
        )
        record_spot_check(conn, sc)
        if delta is not None and delta > 0:
            total_disagreement += delta
        if delta is not None and delta >= 2:
            disagreements_above_threshold.append(dim)

    json_path = outputs_dir / run.clip_id / f"{run.model}.json"
    if json_path.exists():
        payload = _load_payload(json_path)
        payload["human_spot_checked"] = True
        payload["human_spot_check_scores"] = {dim: human_scores[dim] for dim in DIMENSIONS}
        payload["spot_check_disagreements"] = disagreements_above_threshold
        _write_payload(json_path, payload)

    return total_disagreement


def _walk_pair(
    judge_data: dict[str, tuple[int | None, str]],
) -> tuple[bool, dict[str, int | None]]:
    """Returns (completed, human_scores). completed=False on quit."""
    print("For each dimension below, enter your score (0-10), 'n' for N/A, blank to ACCEPT")
    print("judge, or 'q' to quit and save progress.")
    print()
    human_scores: dict[str, int | None] = {}
    for dim in DIMENSIONS:
        judge_score, _ = judge_data[dim]
        while True:
            try:
                raw = input(_prompt_dimension(dim, judge_score))
            except EOFError:
                return False, human_scores
            parsed = _parse_human_input(raw, judge_score, dim)
            if parsed is None:
                print("    invalid input; enter 0-10, 'n', blank, or 'q'")
                continue
            action, value = parsed
            if action == "quit":
                return False, human_scores
            human_scores[dim] = value
            break
    return True, human_scores


def _summary(
    *,
    completed: int,
    total: int,
    total_disagreements: int,
    largest: tuple[str, str, str, int, int] | None,
) -> None:
    print("SUMMARY")
    print(f"  Pairs completed:    {completed} / {total}")
    dims_compared = completed * len(DIMENSIONS)
    print(
        f"  Total disagreements: {total_disagreements} "
        f"(across {dims_compared} dimension comparisons)"
    )
    if largest is not None:
        clip_id, model, dim, j, h = largest
        print(f"  Largest:            {clip_id}/{model} {dim}: judge={j} human={h}")
    print("  Run `python -m brief_eval_rig.scoring.agreement` to compute Cohen's kappa.")


def _interactive_walk(
    *,
    judge_run_ids: list[str],
    already_completed: list[str],
    conn: sqlite3.Connection,
    corpus_dir: Path,
    outputs_dir: Path,
) -> int:
    runs_by_id: dict[str, JudgeRun] = {r.judge_run_id: r for r, _ in iter_judge_runs(conn)}
    vmap = _vertical_map(corpus_dir)
    session = _load_session(outputs_dir)
    total = len(judge_run_ids)
    completed_set = set(already_completed)
    pending = [rid for rid in judge_run_ids if rid not in completed_set]

    pairs_completed = len(completed_set)
    total_disagreement = 0
    largest: tuple[str, str, str, int, int] | None = None
    largest_delta = -1

    for rid in pending:
        run = runs_by_id.get(rid)
        if run is None:
            print(f"warning: judge_run_id {rid} not found in DB; skipping")
            continue
        idx = pairs_completed + 1
        vertical = vmap.get(run.clip_id, "_unknown")
        json_path = outputs_dir / run.clip_id / f"{run.model}.json"
        payload = _load_payload(json_path) if json_path.exists() else {}
        _print_pair_header(idx=idx, total=total, run=run, vertical=vertical)
        _print_ground_truth(run.clip_id, corpus_dir)
        _print_model_output(payload)
        judge_data = _print_judge_scores(run, conn)

        completed_pair, human_scores = _walk_pair(judge_data)
        if not completed_pair:
            print()
            print("  -> quit; pair not recorded (incomplete dimensions)")
            break

        pair_disagreement = _record_pair(
            run=run,
            judge_data=judge_data,
            human_scores=human_scores,
            conn=conn,
            outputs_dir=outputs_dir,
        )
        total_disagreement += pair_disagreement
        for dim in DIMENSIONS:
            j, _ = judge_data[dim]
            h = human_scores[dim]
            if j is None or h is None:
                continue
            delta = abs(j - h)
            if delta > largest_delta:
                largest_delta = delta
                largest = (run.clip_id, run.model, dim, j, h)

        session["completed"] = list(session.get("completed", [])) + [rid]
        _write_session(outputs_dir, session)
        pairs_completed += 1
        print(f"  -> recorded; {len(DIMENSIONS)} dimensions logged")
        print()

    _summary(
        completed=pairs_completed,
        total=total,
        total_disagreements=total_disagreement,
        largest=largest,
    )
    return 0


def _cmd_sample(*, sample_size: int, db_path: Path, corpus_dir: Path, outputs_dir: Path) -> int:
    sess_path = _session_path(outputs_dir)
    if sess_path.exists():
        print(
            f"Existing spot-check session at {sess_path}. "
            "Use --resume to continue or delete the file to start over.",
            file=sys.stderr,
        )
        return 1
    if not db_path.exists():
        print(f"error: db file not found at {db_path}", file=sys.stderr)
        return 1
    conn = connect(db_path)
    try:
        init_schema(conn)
        runs = _gradeable_runs(conn)
        if not runs:
            print("no gradeable judge runs found in DB")
            return 1
        already_done = _completed_judge_run_ids(conn)
        vmap = _vertical_map(corpus_dir)
        picks = stratified_sample(runs, vmap, already_done, sample_size)
        if not picks:
            print("no eligible pairs to sample (all already spot-checked?)")
            return 1
        session = {
            "created_at": _utc_now(),
            "sample_size": sample_size,
            "judge_run_ids": picks,
            "completed": [],
        }
        outputs_dir.mkdir(parents=True, exist_ok=True)
        _write_session(outputs_dir, session)
        return _interactive_walk(
            judge_run_ids=picks,
            already_completed=[],
            conn=conn,
            corpus_dir=corpus_dir,
            outputs_dir=outputs_dir,
        )
    finally:
        conn.close()


def _cmd_resume(*, db_path: Path, corpus_dir: Path, outputs_dir: Path) -> int:
    sess_path = _session_path(outputs_dir)
    if not sess_path.exists():
        print(f"no session file at {sess_path}; run --sample N first", file=sys.stderr)
        return 1
    if not db_path.exists():
        print(f"error: db file not found at {db_path}", file=sys.stderr)
        return 1
    session = _load_session(outputs_dir)
    judge_run_ids = list(session.get("judge_run_ids", []))
    completed = list(session.get("completed", []))
    conn = connect(db_path)
    try:
        init_schema(conn)
        return _interactive_walk(
            judge_run_ids=judge_run_ids,
            already_completed=completed,
            conn=conn,
            corpus_dir=corpus_dir,
            outputs_dir=outputs_dir,
        )
    finally:
        conn.close()


def _cmd_status(*, db_path: Path, outputs_dir: Path) -> int:
    sess_path = _session_path(outputs_dir)
    if not sess_path.exists():
        print(f"no session file at {sess_path}")
        return 0
    session = _load_session(outputs_dir)
    sample_size = int(session.get("sample_size", 0))
    judge_run_ids = list(session.get("judge_run_ids", []))
    completed = list(session.get("completed", []))
    n_completed = len(completed)
    n_remaining = max(0, len(judge_run_ids) - n_completed)
    n_disagree = 0
    if db_path.exists():
        conn = connect(db_path)
        try:
            init_schema(conn)
            for sc in iter_spot_checks(conn):
                if sc.judge_run_id in completed and (sc.disagreement or 0) > 0:
                    n_disagree += 1
        finally:
            conn.close()
    print(f"Sample size: {sample_size}")
    print(f"Completed:   {n_completed}")
    print(f"Remaining:   {n_remaining}")
    print(f"Disagreements so far: {n_disagree}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m brief_eval_rig.scoring.spot_check",
        description="Interactive spot-check CLI for judge-vs-human review.",
    )
    parser.add_argument("--db", type=Path, default=Path("outputs/scores.db"))
    parser.add_argument("--corpus", type=Path, default=Path("corpus/"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs/"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample", type=int, metavar="N")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--status", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    db_path: Path = args.db
    corpus_dir: Path = args.corpus
    outputs_dir: Path = args.outputs

    if args.sample is not None:
        if args.sample <= 0:
            print("error: --sample N must be positive", file=sys.stderr)
            return 1
        return _cmd_sample(
            sample_size=int(args.sample),
            db_path=db_path,
            corpus_dir=corpus_dir,
            outputs_dir=outputs_dir,
        )
    if args.resume:
        return _cmd_resume(db_path=db_path, corpus_dir=corpus_dir, outputs_dir=outputs_dir)
    if args.status:
        return _cmd_status(db_path=db_path, outputs_dir=outputs_dir)
    parser.error("no mode selected")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
