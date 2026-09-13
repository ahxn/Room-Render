"""Pipeline configuration values and YAML loading."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class VideoConfig:
    maximum_duration_seconds: float = 90.0
    target_frame_count: int = 250
    minimum_width: int = 1280
    minimum_height: int = 720
    maximum_file_size_bytes: int = 2_000_000_000


@dataclass(frozen=True, slots=True)
class ReconstructionConfig:
    method: str = "splatfacto"
    matching_method: str = "sequential"
    minimum_registration_rate: float = 0.5
    max_num_iterations: int | None = None
    cache_images: str | None = None
    camera_res_scale_factor: float | None = None


@dataclass(frozen=True, slots=True)
class FrameQualityConfig:
    enabled: bool = False
    minimum_blur_score: float = 2.5


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    video: VideoConfig = VideoConfig()
    frame_quality: FrameQualityConfig = FrameQualityConfig()
    reconstruction: ReconstructionConfig = ReconstructionConfig()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ConfigSection = TypeVar("ConfigSection", VideoConfig, FrameQualityConfig, ReconstructionConfig)


def _load_section(
    payload: dict[str, Any],
    name: str,
    section_type: type[ConfigSection],
) -> ConfigSection:
    section = payload.get(name, {})
    if not isinstance(section, dict):
        raise ConfigurationError(f"Configuration section '{name}' must be a mapping.")
    allowed = {field.name for field in fields(section_type)}
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ConfigurationError(
            f"Unknown option(s) in '{name}': {', '.join(unknown)}"
        )
    try:
        return section_type(**section)
    except TypeError as exc:
        raise ConfigurationError(f"Invalid values in configuration section '{name}'.") from exc


def validate_config(config: PipelineConfig) -> None:
    video = config.video
    if (
        not isinstance(video.target_frame_count, int)
        or isinstance(video.target_frame_count, bool)
        or video.target_frame_count <= 0
    ):
        raise ConfigurationError("video.target_frame_count must be greater than zero.")
    if (
        not isinstance(video.maximum_duration_seconds, (int, float))
        or isinstance(video.maximum_duration_seconds, bool)
        or video.maximum_duration_seconds <= 0
    ):
        raise ConfigurationError("video.maximum_duration_seconds must be greater than zero.")
    for name, value in (
        ("video.minimum_width", video.minimum_width),
        ("video.minimum_height", video.minimum_height),
        ("video.maximum_file_size_bytes", video.maximum_file_size_bytes),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigurationError(f"{name} must be a positive integer.")
    blur_score = config.frame_quality.minimum_blur_score
    if not isinstance(config.frame_quality.enabled, bool):
        raise ConfigurationError("frame_quality.enabled must be true or false.")
    if (
        not isinstance(blur_score, (int, float))
        or isinstance(blur_score, bool)
        or blur_score < 0
    ):
        raise ConfigurationError("frame_quality.minimum_blur_score must be zero or greater.")
    reconstruction = config.reconstruction
    if not isinstance(reconstruction.method, str) or not reconstruction.method.strip():
        raise ConfigurationError("reconstruction.method must be a non-empty string.")
    registration_rate = reconstruction.minimum_registration_rate
    if (
        not isinstance(registration_rate, (int, float))
        or isinstance(registration_rate, bool)
        or not 0 <= registration_rate <= 1
    ):
        raise ConfigurationError(
            "reconstruction.minimum_registration_rate must be between zero and one."
        )
    if not isinstance(reconstruction.matching_method, str) or reconstruction.matching_method not in {
        "exhaustive",
        "sequential",
        "vocab_tree",
    }:
        raise ConfigurationError(
            "reconstruction.matching_method must be exhaustive, sequential, or vocab_tree."
        )
    iterations = reconstruction.max_num_iterations
    if iterations is not None and (
        not isinstance(iterations, int) or isinstance(iterations, bool) or iterations <= 0
    ):
        raise ConfigurationError(
            "reconstruction.max_num_iterations must be a positive integer."
        )
    if not (
        reconstruction.cache_images is None
        or isinstance(reconstruction.cache_images, str)
        and reconstruction.cache_images in {"cpu", "gpu"}
    ):
        raise ConfigurationError("reconstruction.cache_images must be cpu or gpu.")
    scale = reconstruction.camera_res_scale_factor
    if scale is not None and (
        not isinstance(scale, (int, float))
        or isinstance(scale, bool)
        or not 0 < scale <= 1
    ):
        raise ConfigurationError(
            "reconstruction.camera_res_scale_factor must be greater than zero and at most one."
        )


def load_pipeline_config(path: Path) -> PipelineConfig:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigurationError(
            "PyYAML is required to load --config files. Install the project dependencies."
        ) from exc

    if not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Could not parse YAML configuration: {path}") from exc
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ConfigurationError("The YAML configuration root must be a mapping.")
    allowed_sections = {"video", "frame_quality", "reconstruction"}
    unknown_sections = sorted(set(payload) - allowed_sections)
    if unknown_sections:
        raise ConfigurationError(
            f"Unknown configuration section(s): {', '.join(unknown_sections)}"
        )

    config = PipelineConfig(
        video=_load_section(payload, "video", VideoConfig),
        frame_quality=_load_section(payload, "frame_quality", FrameQualityConfig),
        reconstruction=_load_section(payload, "reconstruction", ReconstructionConfig),
    )
    validate_config(config)
    return config
