"""End-to-end orchestration for video-to-Gaussian-splat reconstruction."""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .commands import run_command
from .config import PipelineConfig
from .errors import OutputExistsError, RegistrationError
from .frame_quality import filter_frames
from .metrics import count_registered_frames, registration_rate
from .video import VideoMetadata, sampling_rate, validate_video


@dataclass(frozen=True, slots=True)
class OutputPaths:
    root: Path
    logs: Path
    frames: Path
    selected_frames: Path
    quality_manifest: Path
    processed: Path
    reconstruction: Path
    metadata: Path


def create_output_layout(root: Path) -> OutputPaths:
    paths = OutputPaths(
        root=root,
        logs=root / "logs",
        frames=root / "frames",
        selected_frames=root / "selected_frames",
        quality_manifest=root / "frame_quality.json",
        processed=root / "processed",
        reconstruction=root / "reconstruction",
        metadata=root / "metadata.json",
    )
    for directory in (
        paths.root,
        paths.logs,
        paths.frames,
        paths.selected_frames,
        paths.processed,
        paths.reconstruction,
    ):
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


def _close_logging(logger: logging.Logger) -> None:
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def _write_metadata(path: Path, payload: dict[str, object]) -> None:
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _frame_count(frames_dir: Path) -> int:
    return sum(1 for path in frames_dir.glob("frame_*.jpg") if path.is_file())


def _quality_counts(manifest_path: Path) -> tuple[int, int] | None:
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return int(payload["accepted_count"]), int(payload["rejected_count"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _has_reconstruction_artifacts(paths: OutputPaths) -> bool:
    return bool(
        _frame_count(paths.frames)
        or _frame_count(paths.selected_frames)
        or paths.quality_manifest.is_file()
        or (paths.processed / "transforms.json").is_file()
        or any(paths.reconstruction.rglob("config.yml"))
    )


def _set_stage(payload: dict[str, object], name: str, status: str, **details: object) -> None:
    stages = payload.setdefault("stages", {})
    assert isinstance(stages, dict)
    stages[name] = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }


def _find_completed_training_config(output_root: Path) -> Path | None:
    candidates = sorted(output_root.rglob("config.yml"), key=lambda path: path.stat().st_mtime)
    for config_path in reversed(candidates):
        checkpoint_dir = config_path.parent / "nerfstudio_models"
        if any(checkpoint_dir.glob("*.ckpt")):
            return config_path
    return None


def run_pipeline(
    input_path: Path,
    output_root: Path,
    *,
    config: PipelineConfig | None = None,
    dry_run: bool = False,
    resume: bool = False,
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
        "resume": resume,
        "config": config.to_dict(),
        "stages": {},
    }
    _write_metadata(paths.metadata, payload)

    try:
        logger.info("Validating input video")
        video: VideoMetadata = validate_video(input_path, config.video)
        payload["input"] = video.to_dict()

        if not resume and _has_reconstruction_artifacts(paths):
            raise OutputExistsError(
                f"Output directory already contains reconstruction artifacts: {paths.root}. "
                "Use --resume or choose a new --output directory."
            )

        existing_frames = _frame_count(paths.frames)
        if resume and existing_frames:
            logger.info("Reusing %d previously extracted frames", existing_frames)
            _set_stage(payload, "frames", "reused", count=existing_frames)
        else:
            fps = sampling_rate(video.duration_seconds, config.video.target_frame_count)
            logger.info("Extracting approximately %d frames", config.video.target_frame_count)
            _set_stage(payload, "frames", "planned" if dry_run else "running")
            run_command(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "warning",
                    "-y",
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

        extracted = (
            existing_frames
            if resume and existing_frames
            else config.video.target_frame_count
            if dry_run
            else _frame_count(paths.frames)
        )
        if extracted == 0:
            raise RegistrationError("Frame extraction produced no images.")
        if not (resume and existing_frames):
            _set_stage(payload, "frames", "planned" if dry_run else "completed", count=extracted)

        transforms_path = paths.processed / "transforms.json"
        reusing_camera_poses = resume and transforms_path.is_file()
        selected = extracted
        rejected = 0
        processing_frames = paths.frames
        if reusing_camera_poses:
            logger.info("Skipping frame-quality changes because camera poses already exist")
            quality_counts = _quality_counts(paths.quality_manifest)
            if quality_counts is not None:
                selected, rejected = quality_counts
            _set_stage(payload, "frame_quality", "reused", selected=selected, rejected=rejected)
        elif config.frame_quality.enabled:
            logger.info(
                "Scoring frames with blur threshold %.2f",
                config.frame_quality.minimum_blur_score,
            )
            if dry_run:
                _set_stage(
                    payload,
                    "frame_quality",
                    "planned",
                    threshold=config.frame_quality.minimum_blur_score,
                )
            else:
                quality = filter_frames(
                    paths.frames,
                    paths.selected_frames,
                    paths.quality_manifest,
                    minimum_blur_score=config.frame_quality.minimum_blur_score,
                )
                selected = quality.accepted_count
                rejected = quality.rejected_count
                _set_stage(
                    payload,
                    "frame_quality",
                    "completed",
                    threshold=quality.minimum_blur_score,
                    manifest_path=str(paths.quality_manifest),
                    selected=selected,
                    rejected=rejected,
                )
            processing_frames = paths.selected_frames
        else:
            _set_stage(payload, "frame_quality", "disabled", selected=selected, rejected=rejected)

        if reusing_camera_poses:
            logger.info("Reusing previously estimated camera poses")
            _set_stage(payload, "camera_poses", "reused", transforms_path=str(transforms_path))
        else:
            logger.info("Estimating camera poses with Nerfstudio/COLMAP")
            _set_stage(payload, "camera_poses", "planned" if dry_run else "running")
            run_command(
                [
                    "ns-process-data",
                    "images",
                    "--data",
                    str(processing_frames),
                    "--output-dir",
                    str(paths.processed),
                    "--matching-method",
                    config.reconstruction.matching_method,
                ],
                logger=logger,
                dry_run=dry_run,
            )
            _set_stage(payload, "camera_poses", "planned" if dry_run else "completed")

        registered = selected if dry_run else count_registered_frames(transforms_path)
        rate = registration_rate(registered, selected)
        payload["frames"] = {
            "extracted": extracted,
            "selected": selected,
            "rejected": rejected,
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

        training_config = _find_completed_training_config(paths.reconstruction) if resume else None
        if training_config is not None:
            logger.info("Reusing completed training checkpoint: %s", training_config)
            _set_stage(payload, "training", "reused", config_path=str(training_config))
        else:
            logger.info("Training Gaussian-splat scene")
            _set_stage(payload, "training", "planned" if dry_run else "running")
            training_command = [
                "ns-train",
                config.reconstruction.method,
                "--output-dir",
                str(paths.reconstruction),
                "--data",
                str(paths.processed),
                "--viewer.quit-on-train-completion",
                "True",
            ]
            if config.reconstruction.max_num_iterations is not None:
                training_command.extend(
                    ["--max-num-iterations", str(config.reconstruction.max_num_iterations)]
                )
            if config.reconstruction.cache_images is not None:
                training_command.extend(
                    [
                        "--pipeline.datamanager.cache-images",
                        config.reconstruction.cache_images,
                    ]
                )
            if config.reconstruction.camera_res_scale_factor is not None:
                training_command.extend(
                    [
                        "--pipeline.datamanager.camera-res-scale-factor",
                        str(config.reconstruction.camera_res_scale_factor),
                    ]
                )
            run_command(
                training_command,
                logger=logger,
                dry_run=dry_run,
            )
            training_config = None if dry_run else _find_completed_training_config(paths.reconstruction)
            if not dry_run and training_config is None:
                raise RegistrationError("Training exited without producing a checkpoint.")
            _set_stage(
                payload,
                "training",
                "planned" if dry_run else "completed",
                config_path=str(training_config) if training_config else None,
            )

        payload["status"] = "dry_run" if dry_run else "completed"
        payload["result"] = {
            "reconstruction_path": str(paths.reconstruction.resolve()),
            "config_path": str(training_config) if training_config else None,
            "viewer_command": (
                f"ns-viewer --load-config {training_config}"
                if training_config
                else f"ns-viewer --load-config <config.yml under {paths.reconstruction}>"
            ),
        }
        return payload
    except KeyboardInterrupt:
        payload["status"] = "interrupted"
        payload["error"] = "Run interrupted by user"
        logger.warning("Reconstruction interrupted")
        raise
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = str(exc)
        logger.exception("Reconstruction failed")
        raise
    finally:
        payload["finished_at"] = datetime.now(timezone.utc).isoformat()
        payload["processing_duration_seconds"] = round(time.monotonic() - start_time, 3)
        _write_metadata(paths.metadata, payload)
        _close_logging(logger)


def find_training_config(output_root: Path) -> Path:
    config_path = _find_completed_training_config(output_root)
    if config_path is None:
        raise FileNotFoundError(f"No completed Nerfstudio checkpoint found under {output_root}")
    return config_path
