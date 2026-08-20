"""Pipeline configuration values."""

from dataclasses import dataclass


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
    minimum_registration_rate: float = 0.5


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    video: VideoConfig = VideoConfig()
    reconstruction: ReconstructionConfig = ReconstructionConfig()

