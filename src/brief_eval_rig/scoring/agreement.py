"""Cohen's kappa inter-rater agreement (judge vs human spot-check) per dimension."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DIMENSIONS: tuple[str, ...] = (
    "description_accuracy",
    "ocr_fidelity",
    "spatial_temporal",
    "targeted_query",
    "hallucination_resistance",
    "court_grade_language",
    "speed_cost",
)


@dataclass(frozen=True)
class AgreementRow:
    dimension: str
    n: int
    kappa: float | None
    mean_disagreement: float | None


def _cohen_kappa(y1: list[int], y2: list[int]) -> float:
    """Unweighted Cohen's kappa for two equal-length integer label lists."""
    a1 = np.asarray(y1, dtype=np.int64)
    a2 = np.asarray(y2, dtype=np.int64)
    labels = np.unique(np.concatenate([a1, a2]))
    n_labels = int(labels.size)
    label_to_idx = {int(label): idx for idx, label in enumerate(labels)}
    cm = np.zeros((n_labels, n_labels), dtype=np.int64)
    for raw_a, raw_b in zip(a1, a2, strict=True):
        cm[label_to_idx[int(raw_a)], label_to_idx[int(raw_b)]] += 1
    n = int(cm.sum())
    sum_rows = cm.sum(axis=1)
    sum_cols = cm.sum(axis=0)
    expected = np.outer(sum_rows, sum_cols) / n
    w = np.ones((n_labels, n_labels), dtype=np.float64)
    np.fill_diagonal(w, 0.0)
    denom = float((w * expected).sum())
    if denom == 0.0:
        raise ZeroDivisionError("kappa undefined: zero expected disagreement")
    numer = float((w * cm).sum())
    return 1.0 - numer / denom


def load_paired_scores(conn: sqlite3.Connection, dimension: str) -> tuple[list[int], list[int]]:
    rows = conn.execute(
        """
        SELECT judge_score, human_score
        FROM spot_checks
        WHERE dimension = ?
          AND judge_score IS NOT NULL
          AND human_score IS NOT NULL
        """,
        (dimension,),
    ).fetchall()
    judge_scores: list[int] = []
    human_scores: list[int] = []
    for row in rows:
        judge_scores.append(int(row[0]))
        human_scores.append(int(row[1]))
    return judge_scores, human_scores


def compute_dimension_agreement(judge_scores: list[int], human_scores: list[int]) -> AgreementRow:
    if len(judge_scores) != len(human_scores):
        raise ValueError("judge_scores and human_scores must have equal length")
    n = len(judge_scores)
    if n == 0:
        return AgreementRow(dimension="", n=0, kappa=None, mean_disagreement=None)
    mean_disagreement = sum(abs(j - h) for j, h in zip(judge_scores, human_scores, strict=True)) / n
    kappa: float | None
    if n < 2 or len(set(judge_scores) | set(human_scores)) == 1:
        kappa = None
    else:
        try:
            kappa = _cohen_kappa(judge_scores, human_scores)
        except (ZeroDivisionError, ValueError, FloatingPointError):
            kappa = None
        else:
            if not np.isfinite(kappa):
                kappa = None
    return AgreementRow(dimension="", n=n, kappa=kappa, mean_disagreement=mean_disagreement)


def compute_all(conn: sqlite3.Connection) -> list[AgreementRow]:
    out: list[AgreementRow] = []
    for dim in DIMENSIONS:
        judge_scores, human_scores = load_paired_scores(conn, dim)
        row = compute_dimension_agreement(judge_scores, human_scores)
        out.append(
            AgreementRow(
                dimension=dim,
                n=row.n,
                kappa=row.kappa,
                mean_disagreement=row.mean_disagreement,
            )
        )
    return out


def format_markdown_table(rows: list[AgreementRow]) -> str:
    dim_width = max(len("Dimension"), *(len(r.dimension) for r in rows))
    header = (
        f"| {'Dimension'.ljust(dim_width)} "
        f"| {'n'.rjust(3)} "
        f"| {'κ (judge vs human)'.rjust(18)} "
        f"| {'mean disagreement'.rjust(17)} |"
    )
    separator = f"|{'-' * (dim_width + 2)}" f"|{'-' * 4}:" f"|{'-' * 19}:" f"|{'-' * 18}:|"
    lines = [header, separator]
    for r in rows:
        kappa_cell = "—" if r.kappa is None else f"{r.kappa:.2f}"
        mean_cell = "—" if r.mean_disagreement is None else f"{r.mean_disagreement:.2f}"
        lines.append(
            f"| {r.dimension.ljust(dim_width)} "
            f"| {str(r.n).rjust(3)} "
            f"| {kappa_cell.rjust(18)} "
            f"| {mean_cell.rjust(17)} |"
        )
    return "\n".join(lines) + "\n"


def _spot_checks_empty(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT COUNT(*) FROM spot_checks").fetchone()
    return int(row[0]) == 0


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m brief_eval_rig.scoring.agreement",
        description="Compute Cohen's kappa per dimension across recorded spot-checks.",
    )
    parser.add_argument("--db", type=Path, default=Path("outputs/scores.db"))
    parser.add_argument("--report", type=Path, default=Path("reports/inter_rater_agreement.md"))
    args = parser.parse_args(argv)

    db_path: Path = args.db
    report_path: Path = args.report

    if not db_path.exists():
        print(f"error: db file not found at {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db_path))
    try:
        if _spot_checks_empty(conn):
            print("no spot-checks recorded yet")
            return 0
        rows = compute_all(conn)
    finally:
        conn.close()

    table = format_markdown_table(rows)
    print(table, end="")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(table, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
