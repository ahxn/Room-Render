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
    bit_rate_bits_per_second: int | None

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
        raw_bit_rate = stream.get("bit_rate") or format_data.get("bit_rate")
        bit_rate = int(raw_bit_rate) if raw_bit_rate not in (None, "N/A") else None
        return VideoMetadata(
            duration_seconds=duration,
            width=int(stream["width"]),
            height=int(stream["height"]),
            frame_rate=parse_frame_rate(stream.get("avg_frame_rate", "0/1")),
            file_size_bytes=file_size_bytes,
            codec=str(stream.get("codec_name", "unknown")),
            bit_rate_bits_per_second=bit_rate,
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


def resolution_meets_minimum(
    width: int,
    height: int,
    minimum_width: int,
    minimum_height: int,
) -> bool:
    """Return whether either landscape or portrait orientation meets the minimum."""
    actual_short, actual_long = sorted((width, height))
    required_short, required_long = sorted((minimum_width, minimum_height))
    return actual_short >= required_short and actual_long >= required_long


def source_quality_advisories(metadata: VideoMetadata) -> list[str]:
    """Return non-blocking warnings for sources likely to be compressed copies."""
    if (
        metadata.bit_rate_bits_per_second is not None
        and max(metadata.width, metadata.height) <= 1920
        and metadata.bit_rate_bits_per_second < 8_000_000
    ):
        return [
            (
                "Input video is 1080p-or-smaller at under 8 Mb/s and may be a compressed copy. "
                "For best pose recovery, use the original high-quality recording when available."
            )
        ]
    return []


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
    if not resolution_meets_minimum(
        metadata.width,
        metadata.height,
        config.minimum_width,
        config.minimum_height,
    ):
        raise InputValidationError(
            f"Video resolution is {metadata.width}x{metadata.height}; minimum dimensions are "
            f"{config.minimum_width}x{config.minimum_height} in either orientation."
        )
    return metadata


def sampling_rate(duration_seconds: float, target_frame_count: int) -> float:
    if duration_seconds <= 0 or target_frame_count <= 0:
        raise ValueError("Duration and target frame count must be positive")
    return target_frame_count / duration_seconds
