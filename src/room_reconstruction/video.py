"""Video inspection, validation, and frame-sampling helpers."""

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .commands import require_command
from .config import VideoConfig
from .errors import InputValidationError

SUPPORTED_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    duration_seconds: float
    width: int
    height: int
    frame_rate: float
    file_size_bytes: int
    codec: str

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def parse_frame_rate(value: str) -> float:
    if "/" not in value:
        return float(value)
    numerator, denominator = value.split("/", maxsplit=1)
    denominator_value = float(denominator)
    return float(numerator) / denominator_value if denominator_value else 0.0


def parse_ffprobe_output(payload: str, *, file_size_bytes: int) -> VideoMetadata:
    try:
        data = json.loads(payload)
        stream = next(item for item in data["streams"] if item.get("codec_type") == "video")
        format_data = data.get("format", {})
        duration = float(stream.get("duration") or format_data["duration"])
        return VideoMetadata(
            duration_seconds=duration,
            width=int(stream["width"]),
            height=int(stream["height"]),
            frame_rate=parse_frame_rate(stream.get("avg_frame_rate", "0/1")),
            file_size_bytes=file_size_bytes,
            codec=str(stream.get("codec_name", "unknown")),
        )
    except (KeyError, TypeError, ValueError, StopIteration, json.JSONDecodeError) as exc:
        raise InputValidationError("FFprobe did not return a valid video stream.") from exc


def probe_video(path: Path) -> VideoMetadata:
    require_command("ffprobe")
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown FFprobe error"
        raise InputValidationError(f"Could not inspect video: {detail}")
    return parse_ffprobe_output(completed.stdout, file_size_bytes=path.stat().st_size)


def validate_video(path: Path, config: VideoConfig) -> VideoMetadata:
    if not path.exists():
        raise InputValidationError(f"Input video does not exist: {path}")
    if not path.is_file():
        raise InputValidationError(f"Input path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise InputValidationError(f"Unsupported video format '{path.suffix}'. Use one of: {supported}")
    if path.stat().st_size > config.maximum_file_size_bytes:
        raise InputValidationError("Input video exceeds the configured maximum file size.")

    metadata = probe_video(path)
    if metadata.duration_seconds <= 0:
        raise InputValidationError("Video duration must be greater than zero.")
    if metadata.duration_seconds > config.maximum_duration_seconds:
        raise InputValidationError(
            f"Video is {metadata.duration_seconds:.1f}s; maximum is "
            f"{config.maximum_duration_seconds:.1f}s."
        )
    if metadata.width < config.minimum_width or metadata.height < config.minimum_height:
        raise InputValidationError(
            f"Video resolution is {metadata.width}x{metadata.height}; minimum is "
            f"{config.minimum_width}x{config.minimum_height}."
        )
    return metadata


def sampling_rate(duration_seconds: float, target_frame_count: int) -> float:
    if duration_seconds <= 0 or target_frame_count <= 0:
        raise ValueError("Duration and target frame count must be positive")
    return target_frame_count / duration_seconds
