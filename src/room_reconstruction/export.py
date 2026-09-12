"""Portable Gaussian-splat export support."""

import json
from datetime import datetime, timezone
from pathlib import Path

from .commands import run_command
from .errors import ExportError
from .pipeline import _close_logging, _write_metadata, configure_logging, find_training_config


def _load_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def export_gaussian_splat(
    output_root: Path,
    *,
    export_dir: Path | None = None,
    filename: str = "splat.ply",
) -> Path:
    """Export the latest completed checkpoint to a portable PLY file."""
    filename_path = Path(filename)
    if filename_path.name != filename or filename_path.suffix.lower() != ".ply":
        raise ExportError("Export filename must be a plain filename ending in .ply")

    config_path = find_training_config(output_root)
    destination = export_dir or output_root / "exports"
    destination.mkdir(parents=True, exist_ok=True)
    exported_path = destination / filename
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(logs_dir / "export.log")
    try:
        run_command(
            [
                "ns-export",
                "gaussian-splat",
                "--load-config",
                str(config_path),
                "--output-dir",
                str(destination),
                "--output-filename",
                filename,
            ],
            logger=logger,
        )
        if not exported_path.is_file():
            raise ExportError(f"Nerfstudio finished without creating {exported_path}")

        metadata_path = output_root / "metadata.json"
        metadata = _load_metadata(metadata_path)
        exports = metadata.setdefault("exports", [])
        if not isinstance(exports, list):
            exports = []
            metadata["exports"] = exports
        exports.append(
            {
                "format": "gaussian-splat-ply",
                "path": str(exported_path.resolve()),
                "config_path": str(config_path.resolve()),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "size_bytes": exported_path.stat().st_size,
            }
        )
        _write_metadata(metadata_path, metadata)
        return exported_path
    finally:
        _close_logging(logger)
