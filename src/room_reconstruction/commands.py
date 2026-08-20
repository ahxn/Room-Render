"""Safe, logged execution of external reconstruction tools."""

from collections.abc import Sequence
import logging
from pathlib import Path
import shutil
import subprocess

from .errors import CommandUnavailableError, ExternalCommandError


REQUIRED_COMMANDS = ("ffmpeg", "ffprobe", "ns-process-data", "ns-train", "ns-viewer")


def command_availability() -> dict[str, bool]:
    return {command: shutil.which(command) is not None for command in REQUIRED_COMMANDS}


def require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise CommandUnavailableError(
            f"Required command '{command}' was not found on PATH. "
            "Install it in the active environment and try again."
        )


def run_command(
    args: Sequence[str],
    *,
    logger: logging.Logger,
    cwd: Path | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    if not args:
        raise ValueError("Command cannot be empty")
    logger.info("Command: %s", subprocess.list2cmdline(list(args)))
    if dry_run:
        return None

    require_command(args[0])
    completed = subprocess.run(
        list(args),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.stdout:
        for line in completed.stdout.splitlines():
            logger.info("[%s] %s", args[0], line)
    if completed.returncode != 0:
        raise ExternalCommandError(
            f"'{args[0]}' failed with exit code {completed.returncode}. "
            "See the processing log for command output."
        )
    return completed

