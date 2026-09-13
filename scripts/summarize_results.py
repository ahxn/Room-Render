"""Print a Markdown summary table for reconstruction metadata files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def metadata_path(path: Path) -> Path:
    return path / "metadata.json" if path.is_dir() else path


def load_run(path: Path) -> dict[str, Any]:
    resolved = metadata_path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read metadata: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON metadata: {resolved}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Metadata root must be an object: {resolved}")
    payload["_metadata_path"] = resolved
    return payload


def _number(value: object, digits: int = 3) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def summarize(runs: list[dict[str, Any]]) -> str:
    lines = [
        "| Run | Status | Resumed | Extracted | Selected | Registered | Registration | Invocation (s) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        frames = run.get("frames") if isinstance(run.get("frames"), dict) else {}
        source = Path(str(run["_metadata_path"]))
        name = source.parent.name or source.stem
        rate = frames.get("registration_rate")
        formatted_rate = (
            f"{rate * 100:.2f}%"
            if isinstance(rate, (int, float)) and not isinstance(rate, bool)
            else "—"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    name.replace("|", "\\|"),
                    str(run.get("status", "—")),
                    "yes" if run.get("resume") is True else "no",
                    _number(frames.get("extracted"), 0),
                    _number(frames.get("selected"), 0),
                    _number(frames.get("registered"), 0),
                    formatted_rate,
                    _number(run.get("processing_duration_seconds")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Run directories or metadata.json files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(summarize([load_run(path) for path in args.runs]))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
