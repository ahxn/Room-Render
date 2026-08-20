"""Run metrics and reconstruction output inspection."""

import json
from pathlib import Path

from .errors import RegistrationError


def count_registered_frames(transforms_path: Path) -> int:
    try:
        data = json.loads(transforms_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistrationError(f"Could not read camera transforms: {transforms_path}") from exc
    frames = data.get("frames")
    if not isinstance(frames, list):
        raise RegistrationError("Camera transforms do not contain a frame list.")
    return len(frames)


def registration_rate(registered: int, selected: int) -> float:
    if registered < 0 or selected <= 0 or registered > selected:
        raise ValueError("Registration counts are inconsistent")
    return registered / selected

