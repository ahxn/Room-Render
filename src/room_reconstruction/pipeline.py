"""End-to-end orchestration for video-to-Gaussian-splat reconstruction."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time

from .commands import run_command
from .config import PipelineConfig
from .errors import RegistrationError
from .metrics import count_registered_frames, registration_rate
from .video import VideoMetadata, sampling_rate, validate_video


@dataclass(frozen=True, slots=True)
class OutputPaths:
    root: Path
    logs: Path
    frames: Path
    processed: Path
    reconstruction: Path
    metadata: Path


def create_output_layout(root: Path) -> OutputPaths:
    paths = OutputPaths(
        root=root,
        logs=root / "logs",
        frames=root / "frames",
        processed=root / "processed",
        reconstruction=root / "reconstruction",
        metadata=root / "metadata.json",
    )
    for directory in (paths.root, paths.logs, paths.frames, paths.processed, paths.reconstruction):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def configure_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"room_reconstruction.{log_path.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def _write_metadata(path: Path, payload: dict[str, object]) -> None:
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _frame_count(frames_dir: Path) -> int:
    return sum(1 for path in frames_dir.glob("frame_*.jpg") if path.is_file())


def run_pipeline(
    input_path: Path,
    output_root: Path,
    *,
    config: PipelineConfig | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    config = config or PipelineConfig()
    paths = create_output_layout(output_root)
    logger = configure_logging(paths.logs / "processing.log")
    started_at = datetime.now(timezone.utc)
    start_time = time.monotonic()
    payload: dict[str, object] = {
        "status": "running",
        "started_at": started_at.isoformat(),
        "input_path": str(input_path.resolve()),
    }
    _write_metadata(paths.metadata, payload)

    try:
        logger.info("Validating input video")
        video: VideoMetadata = validate_video(input_path, config.video)
        payload["input"] = video.to_dict()

        fps = sampling_rate(video.duration_seconds, config.video.target_frame_count)
        logger.info("Extracting approximately %d frames", config.video.target_frame_count)
        run_command(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-i",
                str(input_path),
                "-vf",
                f"fps={fps:.8f}",
                "-q:v",
                "2",
                str(paths.frames / "frame_%05d.jpg"),
            ],
            logger=logger,
            dry_run=dry_run,
        )

        selected = config.video.target_frame_count if dry_run else _frame_count(paths.frames)
        if selected == 0:
            raise RegistrationError("Frame extraction produced no images.")

        logger.info("Estimating camera poses with Nerfstudio/COLMAP")
        run_command(
            [
                "ns-process-data",
                "images",
                "--data",
                str(paths.frames),
                "--output-dir",
                str(paths.processed),
            ],
            logger=logger,
            dry_run=dry_run,
        )

        transforms_path = paths.processed / "transforms.json"
        registered = selected if dry_run else count_registered_frames(transforms_path)
        rate = registration_rate(registered, selected)
        payload["frames"] = {
            "extracted": selected,
            "selected": selected,
            "registered": registered,
            "registration_rate": rate,
        }
        logger.info("Registered %d of %d frames (%.1f%%)", registered, selected, rate * 100)
        if rate < config.reconstruction.minimum_registration_rate:
            raise RegistrationError(
                f"Registration rate {rate:.1%} is below the configured minimum "
                f"of {config.reconstruction.minimum_registration_rate:.1%}. "
                "Capture again with more camera movement, overlap, and less blur."
            )

        logger.info("Training Gaussian-splat scene")
        run_command(
            [
                "ns-train",
                config.reconstruction.method,
                "--output-dir",
                str(paths.reconstruction),
                "--data",
                str(paths.processed),
            ],
            logger=logger,
            dry_run=dry_run,
        )

        payload["status"] = "dry_run" if dry_run else "completed"
        payload["result"] = {
            "reconstruction_path": str(paths.reconstruction.resolve()),
            "viewer_command": f"ns-viewer --load-config <config.yml under {paths.reconstruction}>",
        }
        return payload
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = str(exc)
        logger.exception("Reconstruction failed")
        raise
    finally:
        payload["finished_at"] = datetime.now(timezone.utc).isoformat()
        payload["processing_duration_seconds"] = round(time.monotonic() - start_time, 3)
        _write_metadata(paths.metadata, payload)


def find_training_config(output_root: Path) -> Path:
    candidates = sorted(output_root.rglob("config.yml"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No Nerfstudio config.yml found under {output_root}")
    return candidates[-1]

