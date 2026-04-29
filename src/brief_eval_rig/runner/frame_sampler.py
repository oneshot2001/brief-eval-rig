"""Deterministic frame sampler used by adapters that don't ingest native video.

For Phase 1, only the Claude adapter needs this — Anthropic Messages does not
accept video input directly, so we extract N frames at uniform intervals,
optionally augmented with up to 2 scene-change keyframes detected via a
histogram-difference heuristic, and feed each as a base64 JPEG image.

Determinism guarantee: same input video + same N produces byte-identical
output across runs on the same machine. We do not interpolate, scale, or
randomize.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def default_n_for_duration(duration_seconds: float) -> int:
    """Choose a sensible frame count for a clip of the given duration.

    Per spec §7: 8 for short (<=60s), 16 for medium (1-5min), 32 for long.
    """
    if duration_seconds <= 60:
        return 8
    if duration_seconds <= 300:
        return 16
    return 32


def _read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def _encode_jpeg(frame: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("OpenCV failed to encode frame as JPEG")
    return bytes(buf.tobytes())


def _detect_scene_changes(
    cap: cv2.VideoCapture, total_frames: int, max_keyframes: int
) -> list[int]:
    """Return up to max_keyframes frame indices where the histogram diff spikes.

    Cheap heuristic: walk the video, compute color-histogram differences between
    consecutive sampled frames, return the top-K spike indices. Skips the very
    first frame (always a "change" by definition).
    """
    if max_keyframes <= 0 or total_frames < 4:
        return []

    sample_stride = max(1, total_frames // 64)
    prev_hist: np.ndarray | None = None
    diffs: list[tuple[float, int]] = []
    for idx in range(0, total_frames, sample_stride):
        frame = _read_frame_at(cap, idx)
        if frame is None:
            continue
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            diff = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            diffs.append((diff, idx))
        prev_hist = hist

    diffs.sort(reverse=True)
    return sorted({idx for _, idx in diffs[:max_keyframes]})


def sample_frames(
    video_path: Path,
    n_frames: int,
    *,
    include_keyframes: bool = True,
) -> list[bytes]:
    """Extract n_frames at uniform temporal intervals as JPEG bytes.

    If include_keyframes is True, append up to 2 additional scene-change frames
    detected via histogram diff. Returns frames in ascending temporal order.
    Deterministic: same inputs produce byte-identical output on the same
    OpenCV/codec stack.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            raise RuntimeError(f"Video reports zero frames: {video_path}")

        uniform_indices = [
            int(round(i * (total_frames - 1) / max(n_frames - 1, 1))) for i in range(n_frames)
        ]
        keyframe_indices = (
            _detect_scene_changes(cap, total_frames, max_keyframes=2) if include_keyframes else []
        )

        all_indices = sorted(set(uniform_indices + keyframe_indices))
        frames: list[bytes] = []
        for idx in all_indices:
            frame = _read_frame_at(cap, idx)
            if frame is None:
                continue
            frames.append(_encode_jpeg(frame))
        return frames
    finally:
        cap.release()
