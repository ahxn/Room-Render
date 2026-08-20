"""Command-line interface."""

import argparse
import logging
from pathlib import Path
import subprocess
import sys

from .commands import command_availability, require_command
from .errors import ReconstructionError
from .pipeline import find_training_config, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a handheld room video into a Gaussian-splat reconstruction."
    )
    parser.add_argument("video", nargs="?", type=Path, help="Path to the input room video")
    parser.add_argument("--output", type=Path, help="Directory for reconstruction artifacts")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the video and print/log planned reconstruction commands",
    )
    parser.add_argument(
        "--check-environment",
        action="store_true",
        help="Report whether required external commands are available",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        dest="open_result",
        help="Open an existing result with ns-viewer instead of reconstructing",
    )
    return parser


def _check_environment() -> int:
    availability = command_availability()
    for command, available in availability.items():
        print(f"{'OK' if available else 'MISSING':7} {command}")
    return 0 if all(availability.values()) else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_environment:
        return _check_environment()
    if args.video is None or args.output is None:
        build_parser().error("video and --output are required unless --check-environment is used")

    try:
        if args.open_result:
            require_command("ns-viewer")
            config_path = find_training_config(args.output)
            return subprocess.run(["ns-viewer", "--load-config", str(config_path)], check=False).returncode
        run_pipeline(args.video, args.output, dry_run=args.dry_run)
        print(f"Run metadata: {args.output / 'metadata.json'}")
        return 0
    except (ReconstructionError, FileNotFoundError) as exc:
        logging.getLogger(__name__).debug("Command failed", exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        return 2

