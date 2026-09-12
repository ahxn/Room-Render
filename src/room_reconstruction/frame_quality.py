"""OpenCV-based blur scoring and selected-frame manifest generation."""

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .errors import FrameQualityError


@dataclass(frozen=True, slots=True)
class FrameScore:
    filename: str
    blur_score: float
    accepted: bool


@dataclass(frozen=True, slots=True)
class FrameQualityResult:
    extracted_count: int
    accepted_count: int
    rejected_count: int
    minimum_blur_score: float
    frames: tuple[FrameScore, ...]


def calculate_blur_score(path: Path) -> float:
    """Return the variance of the grayscale Laplacian for one image."""
    try:
        import cv2
    except ImportError as exc:
        raise FrameQualityError(
            "OpenCV is required for frame-quality filtering. Install a compatible "
            "opencv-python build or run without --filter-blurry-frames."
        ) from exc

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FrameQualityError(f"Could not load extracted frame: {path}")
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def filter_frames(
    frames_dir: Path,
    selected_dir: Path,
    manifest_path: Path,
    *,
    minimum_blur_score: float,
) -> FrameQualityResult:
    """Score extracted JPEGs, link accepted frames, and write a JSON manifest."""
    if minimum_blur_score < 0:
        raise FrameQualityError("Blur threshold must be zero or greater.")

    frame_paths = sorted(path for path in frames_dir.glob("frame_*.jpg") if path.is_file())
    if not frame_paths:
        raise FrameQualityError(f"No extracted frames found under {frames_dir}")

    selected_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in selected_dir.glob("frame_*.jpg"):
        if stale_path.is_file() or stale_path.is_symlink():
            stale_path.unlink()

    scores: list[FrameScore] = []
    for frame_path in frame_paths:
        score = calculate_blur_score(frame_path)
        accepted = score >= minimum_blur_score
        scores.append(FrameScore(frame_path.name, round(score, 4), accepted))
        if accepted:
            destination = selected_dir / frame_path.name
            try:
                os.link(frame_path, destination)
            except OSError:
                shutil.copy2(frame_path, destination)

    accepted_count = sum(score.accepted for score in scores)
    result = FrameQualityResult(
        extracted_count=len(scores),
        accepted_count=accepted_count,
        rejected_count=len(scores) - accepted_count,
        minimum_blur_score=minimum_blur_score,
        frames=tuple(scores),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)

    if accepted_count == 0:
        raise FrameQualityError(
            f"All {len(scores)} frames were rejected at blur threshold "
            f"{minimum_blur_score:g}. Lower --blur-threshold or capture a sharper video."
        )
    return result
