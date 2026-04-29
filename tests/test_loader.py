"""Loader-level tests including business-rule enforcement at the loader chokepoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from brief_eval_rig.corpus.loader import load_clip, load_corpus
from brief_eval_rig.corpus.schema import CorpusValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO_ROOT / "corpus" / "sample"


def test_load_sample_clip() -> None:
    clip = load_clip(SAMPLE_DIR / "sample-001.json")
    assert clip.clip_id == "smp-001"
    assert clip.vertical == "retail"
    assert clip.duration_seconds == 28.5


def test_load_corpus_sample_dir() -> None:
    clips = load_corpus(SAMPLE_DIR)
    assert len(clips) == 1
    assert clips[0].clip_id == "smp-001"


def test_load_corpus_empty_dir(tmp_path: Path) -> None:
    assert load_corpus(tmp_path) == []


def test_load_corpus_skips_gitkeep(tmp_path: Path) -> None:
    (tmp_path / ".gitkeep").touch()
    assert load_corpus(tmp_path) == []


def test_load_clip_rejects_missing_consent(tmp_path: Path, valid_payload: dict[str, Any]) -> None:
    valid_payload["consent_documented"] = False
    bad = tmp_path / "bad-001.json"
    bad.write_text(json.dumps(valid_payload), encoding="utf-8")
    with pytest.raises(CorpusValidationError):
        load_clip(bad)


def test_load_clip_rejects_minor_present(tmp_path: Path, valid_payload: dict[str, Any]) -> None:
    valid_payload["minor_present"] = True
    bad = tmp_path / "bad-002.json"
    bad.write_text(json.dumps(valid_payload), encoding="utf-8")
    with pytest.raises(CorpusValidationError):
        load_clip(bad)


def test_load_clip_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(CorpusValidationError):
        load_clip(tmp_path / "does-not-exist.json")
