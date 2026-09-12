"""Runtime checks for the external reconstruction environment."""

import importlib.metadata
import shutil
import sys
from dataclasses import dataclass

from .commands import REQUIRED_COMMANDS


@dataclass(frozen=True, slots=True)
class EnvironmentCheck:
    name: str
    available: bool
    detail: str


def check_environment() -> list[EnvironmentCheck]:
    """Exercise critical integrations instead of only checking command names."""
    checks = [
        EnvironmentCheck(
            f"command:{command}",
            shutil.which(command) is not None,
            shutil.which(command) or "not found on PATH",
        )
        for command in REQUIRED_COMMANDS
    ]
    checks.append(
        EnvironmentCheck(
            "python",
            sys.version_info >= (3, 10),
            ".".join(str(part) for part in sys.version_info[:3]),
        )
    )

    try:
        import cv2
        import numpy as np

        camera_matrix = np.eye(3, dtype=np.float32)
        distortion = np.zeros(8, dtype=np.float32)
        cv2.getOptimalNewCameraMatrix(camera_matrix, distortion, (100, 100), 0)
        checks.append(EnvironmentCheck("opencv", True, f"{cv2.__version__}; NumPy {np.__version__}"))
    except Exception as exc:  # noqa: BLE001 - a readiness check must report binary failures
        checks.append(EnvironmentCheck("opencv", False, f"compatibility test failed: {exc}"))

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        detail = torch.cuda.get_device_name(0) if cuda_available else "CUDA is unavailable"
        checks.append(EnvironmentCheck("cuda", cuda_available, f"PyTorch {torch.__version__}; {detail}"))
    except Exception as exc:  # noqa: BLE001 - a readiness check must report driver failures
        checks.append(EnvironmentCheck("cuda", False, f"PyTorch check failed: {exc}"))

    try:
        version = importlib.metadata.version("nerfstudio")
        checks.append(EnvironmentCheck("nerfstudio", True, version))
    except importlib.metadata.PackageNotFoundError:
        checks.append(EnvironmentCheck("nerfstudio", False, "Python package is not installed"))
    return checks
