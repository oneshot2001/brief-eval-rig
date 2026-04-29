"""Frame sampler unit tests using a synthetic generated clip."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from brief_eval_rig.runner.frame_sampler import default_n_for_duration, sample_frames

JPEG_SOI = b"\xff\xd8"


def test_default_n_short() -> None:
    assert default_n_for_duration(30) == 8


def test_default_n_medium() -> None:
    assert default_n_for_duration(180) == 16


def test_default_n_long() -> None:
    assert default_n_for_duration(600) == 32


def test_default_n_boundaries() -> None:
    assert default_n_for_duration(60) == 8
    assert default_n_for_duration(60.5) == 16
    assert default_n_for_duration(300) == 16
    assert default_n_for_duration(300.1) == 32


def test_sample_frames_count(synthetic_clip_path: Path) -> None:
    frames = sample_frames(synthetic_clip_path, n_frames=4, include_keyframes=False)
    assert len(frames) == 4


def test_sample_frames_returns_jpeg_bytes(synthetic_clip_path: Path) -> None:
    frames = sample_frames(synthetic_clip_path, n_frames=4, include_keyframes=False)
    for f in frames:
        assert isinstance(f, bytes)
        assert len(f) > 0
        assert f[:2] == JPEG_SOI  # JPEG magic bytes


def test_sample_frames_deterministic(synthetic_clip_path: Path) -> None:
    a = sample_frames(synthetic_clip_path, n_frames=4, include_keyframes=False)
    b = sample_frames(synthetic_clip_path, n_frames=4, include_keyframes=False)
    assert a == b


def test_sample_frames_invalid_n(synthetic_clip_path: Path) -> None:
    with pytest.raises(ValueError):
        sample_frames(synthetic_clip_path, n_frames=0)


def test_sample_frames_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sample_frames(tmp_path / "nope.mp4", n_frames=4)


def test_sample_frames_keyframes_added(synthetic_clip_path: Path) -> None:
    """With include_keyframes=True, frame count is >= n_frames."""
    base = sample_frames(synthetic_clip_path, n_frames=4, include_keyframes=False)
    with_kf = sample_frames(synthetic_clip_path, n_frames=4, include_keyframes=True)
    assert len(with_kf) >= len(base)


def test_struct_unpack_jpeg_marker(synthetic_clip_path: Path) -> None:
    """Sanity: the second byte of each JPEG starts the marker FF Dx."""
    frames = sample_frames(synthetic_clip_path, n_frames=2, include_keyframes=False)
    for f in frames:
        marker = struct.unpack(">H", f[:2])[0]
        assert marker == 0xFFD8
