"""SQLite storage for judge runs, dimension scores, and human spot-checks."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Dimension = Literal[
    "description_accuracy",
    "ocr_fidelity",
    "spatial_temporal",
    "targeted_query",
    "hallucination_resistance",
    "court_grade_language",
    "speed_cost",
]


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS judge_runs (
    judge_run_id TEXT PRIMARY KEY,
    clip_id TEXT NOT NULL,
    model TEXT NOT NULL,
    judge_model TEXT NOT NULL,           -- "claude-opus-4-7" | "gpt-5"
    composite REAL,                      -- null when error is set
    error TEXT,
    judge_input_tokens INTEGER,
    judge_output_tokens INTEGER,
    judge_cost_usd REAL,
    raw_judge_response TEXT,             -- full raw response for replay/debug
    created_at TEXT NOT NULL,            -- ISO8601 UTC
    UNIQUE (clip_id, model)              -- idempotent overwrite
);

CREATE TABLE IF NOT EXISTS dimension_scores (
    judge_run_id TEXT NOT NULL,
    dimension TEXT NOT NULL,             -- 'description_accuracy' etc.
    score INTEGER,                       -- 0..10, null for N/A
    justification TEXT,
    PRIMARY KEY (judge_run_id, dimension),
    FOREIGN KEY (judge_run_id) REFERENCES judge_runs(judge_run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS spot_checks (
    spot_check_id TEXT PRIMARY KEY,
    judge_run_id TEXT NOT NULL,
    dimension TEXT NOT NULL,
    judge_score INTEGER,
    human_score INTEGER,
    disagreement INTEGER,                -- abs(human - judge), null if either side null
    notes TEXT,
    checked_at TEXT NOT NULL,
    FOREIGN KEY (judge_run_id) REFERENCES judge_runs(judge_run_id) ON DELETE CASCADE
);
"""


class JudgeRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    judge_run_id: str
    clip_id: str
    model: str
    judge_model: str
    composite: float | None = None
    error: str | None = None
    judge_input_tokens: int | None = None
    judge_output_tokens: int | None = None
    judge_cost_usd: float | None = None
    raw_judge_response: str | None = None
    created_at: str


class DimensionScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    judge_run_id: str
    dimension: Dimension
    score: int | None = Field(default=None, ge=0, le=10)
    justification: str | None = None


class SpotCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spot_check_id: str
    judge_run_id: str
    dimension: Dimension
    judge_score: int | None = Field(default=None, ge=0, le=10)
    human_score: int | None = Field(default=None, ge=0, le=10)
    disagreement: int | None = None
    notes: str | None = None
    checked_at: str


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def upsert_judge_run(
    conn: sqlite3.Connection,
    run: JudgeRun,
    dimension_scores: list[DimensionScore],
) -> None:
    with conn:
        prior = conn.execute(
            "SELECT judge_run_id FROM judge_runs WHERE clip_id = ? AND model = ?",
            (run.clip_id, run.model),
        ).fetchone()
        if prior is not None:
            conn.execute(
                "DELETE FROM judge_runs WHERE judge_run_id = ?",
                (prior["judge_run_id"],),
            )
        conn.execute(
            """
            INSERT INTO judge_runs (
                judge_run_id, clip_id, model, judge_model, composite, error,
                judge_input_tokens, judge_output_tokens, judge_cost_usd,
                raw_judge_response, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.judge_run_id,
                run.clip_id,
                run.model,
                run.judge_model,
                run.composite,
                run.error,
                run.judge_input_tokens,
                run.judge_output_tokens,
                run.judge_cost_usd,
                run.raw_judge_response,
                run.created_at,
            ),
        )
        for ds in dimension_scores:
            if ds.judge_run_id != run.judge_run_id:
                raise ValueError(
                    f"DimensionScore.judge_run_id {ds.judge_run_id!r} does not match "
                    f"JudgeRun.judge_run_id {run.judge_run_id!r}"
                )
            conn.execute(
                """
                INSERT INTO dimension_scores (
                    judge_run_id, dimension, score, justification
                ) VALUES (?, ?, ?, ?)
                """,
                (ds.judge_run_id, ds.dimension, ds.score, ds.justification),
            )


def record_spot_check(conn: sqlite3.Connection, sc: SpotCheck) -> None:
    with conn:
        conn.execute(
            """
            INSERT INTO spot_checks (
                spot_check_id, judge_run_id, dimension, judge_score,
                human_score, disagreement, notes, checked_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sc.spot_check_id,
                sc.judge_run_id,
                sc.dimension,
                sc.judge_score,
                sc.human_score,
                sc.disagreement,
                sc.notes,
                sc.checked_at,
            ),
        )


def iter_judge_runs(
    conn: sqlite3.Connection,
    *,
    clip_id: str | None = None,
    model: str | None = None,
) -> Iterator[tuple[JudgeRun, list[DimensionScore]]]:
    sql = "SELECT * FROM judge_runs"
    where: list[str] = []
    params: list[str] = []
    if clip_id is not None:
        where.append("clip_id = ?")
        params.append(clip_id)
    if model is not None:
        where.append("model = ?")
        params.append(model)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at"
    rows = conn.execute(sql, params).fetchall()
    for row in rows:
        run = JudgeRun(
            judge_run_id=row["judge_run_id"],
            clip_id=row["clip_id"],
            model=row["model"],
            judge_model=row["judge_model"],
            composite=row["composite"],
            error=row["error"],
            judge_input_tokens=row["judge_input_tokens"],
            judge_output_tokens=row["judge_output_tokens"],
            judge_cost_usd=row["judge_cost_usd"],
            raw_judge_response=row["raw_judge_response"],
            created_at=row["created_at"],
        )
        ds_rows = conn.execute(
            "SELECT * FROM dimension_scores WHERE judge_run_id = ? ORDER BY dimension",
            (run.judge_run_id,),
        ).fetchall()
        scores = [
            DimensionScore(
                judge_run_id=ds["judge_run_id"],
                dimension=ds["dimension"],
                score=ds["score"],
                justification=ds["justification"],
            )
            for ds in ds_rows
        ]
        yield run, scores


def iter_spot_checks(
    conn: sqlite3.Connection,
    *,
    dimension: str | None = None,
) -> Iterator[SpotCheck]:
    sql = "SELECT * FROM spot_checks"
    params: list[str] = []
    if dimension is not None:
        sql += " WHERE dimension = ?"
        params.append(dimension)
    sql += " ORDER BY checked_at"
    rows = conn.execute(sql, params).fetchall()
    for row in rows:
        yield SpotCheck(
            spot_check_id=row["spot_check_id"],
            judge_run_id=row["judge_run_id"],
            dimension=row["dimension"],
            judge_score=row["judge_score"],
            human_score=row["human_score"],
            disagreement=row["disagreement"],
            notes=row["notes"],
            checked_at=row["checked_at"],
        )
