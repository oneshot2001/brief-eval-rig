"""Shared pytest fixtures."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_JSON_PATH = REPO_ROOT / "corpus" / "sample" / "sample-001.json"


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    """Fresh dict copy of the canonical sample sidecar.

    Tests freely mutate the returned dict; the fixture is recomputed per test
    so mutations don't bleed across tests.
    """
    return json.loads(SAMPLE_JSON_PATH.read_text(encoding="utf-8"))


def _write_synthetic_clip(path: Path, *, frames: int = 30, fps: int = 10) -> None:
    """Write a tiny deterministic 320x240 color-gradient mp4 at the given path."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open mp4v container")
    try:
        for i in range(frames):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            # Deterministic gradient that shifts per-frame so scene-change can fire.
            phase = i / max(frames - 1, 1)
            frame[:, :, 0] = int(255 * phase)
            frame[:, :, 1] = int(255 * (1 - phase))
            frame[:, :, 2] = (i * 7) % 256
            writer.write(frame)
    finally:
        writer.release()


@pytest.fixture(scope="session")
def synthetic_clip_path(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Generate a tiny mp4 once per test session for frame-sampler / adapter tests."""
    out_dir = tmp_path_factory.mktemp("video")
    out = out_dir / "synthetic.mp4"
    _write_synthetic_clip(out)
    yield out
