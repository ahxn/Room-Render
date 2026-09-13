"""Command-line interface."""

import argparse
import logging
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from .commands import require_command
from .config import PipelineConfig, load_pipeline_config, validate_config
from .environment import check_environment
from .errors import ReconstructionError
from .export import export_gaussian_splat
from .pipeline import find_training_config, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a handheld room video into a Gaussian-splat reconstruction."
    )
    parser.add_argument("video", nargs="?", type=Path, help="Path to the input room video")
    parser.add_argument("--output", type=Path, help="Directory for reconstruction artifacts")
    parser.add_argument("--config", type=Path, help="YAML pipeline configuration file")
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
    result_action = parser.add_mutually_exclusive_group()
    result_action.add_argument(
        "--open",
        action="store_true",
        dest="open_result",
        help="Open an existing result with ns-viewer instead of reconstructing",
    )
    result_action.add_argument(
        "--export",
        action="store_true",
        dest="export_result",
        help="Export the latest completed checkpoint as a Gaussian-splat PLY",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="Export destination (default: <output>/exports)",
    )
    parser.add_argument(
        "--export-filename",
        default="splat.ply",
        help="PLY filename used with --export (default: splat.ply)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed frames, camera processing, and training artifacts",
    )
    parser.add_argument(
        "--filter-blurry-frames",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Score extracted frames with OpenCV and exclude frames below the blur threshold",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=None,
        help="Override the configured minimum Laplacian-variance score",
    )
    return parser


def _check_environment() -> int:
    checks = check_environment()
    for check in checks:
        print(f"{'OK' if check.available else 'FAILED':7} {check.name:25} {check.detail}")
    return 0 if all(check.available for check in checks) else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_environment:
        return _check_environment()
    if args.output is None:
        build_parser().error("--output is required unless --check-environment is used")
    if args.video is None and not (args.open_result or args.export_result):
        build_parser().error(
            "video is required unless --check-environment, --open, or --export is used"
        )

    try:
        if args.open_result:
            require_command("ns-viewer")
            config_path = find_training_config(args.output)
            return subprocess.run(["ns-viewer", "--load-config", str(config_path)], check=False).returncode
        if args.export_result:
            exported_path = export_gaussian_splat(
                args.output,
                export_dir=args.export_dir,
                filename=args.export_filename,
            )
            print(f"Exported Gaussian splat: {exported_path}")
            return 0
        config = load_pipeline_config(args.config) if args.config else PipelineConfig()
        frame_quality = config.frame_quality
        if args.filter_blurry_frames is not None:
            frame_quality = replace(frame_quality, enabled=args.filter_blurry_frames)
        if args.blur_threshold is not None:
            frame_quality = replace(frame_quality, minimum_blur_score=args.blur_threshold)
        config = replace(config, frame_quality=frame_quality)
        validate_config(config)
        run_pipeline(
            args.video,
            args.output,
            config=config,
            dry_run=args.dry_run,
            resume=args.resume,
        )
        print(f"Run metadata: {args.output / 'metadata.json'}")
        return 0
    except (ReconstructionError, FileNotFoundError) as exc:
        logging.getLogger(__name__).debug("Command failed", exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        return 2
