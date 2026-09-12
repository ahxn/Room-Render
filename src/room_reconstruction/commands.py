"""Safe, logged execution of external reconstruction tools."""

import logging
import os
import shutil
import signal
import subprocess
from collections.abc import Sequence
from pathlib import Path

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
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        list(args),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        start_new_session=os.name != "nt",
        creationflags=creation_flags,
    )
    output: list[str] = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip("\r\n")
            output.append(line)
            logger.info("[%s] %s", args[0], line)
        return_code = process.wait()
    except KeyboardInterrupt:
        logger.warning("Interrupting '%s' and cleaning up its child process", args[0])
        if process.poll() is None:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise

    completed = subprocess.CompletedProcess(list(args), return_code, "\n".join(output))
    if return_code != 0:
        raise ExternalCommandError(
            f"'{args[0]}' failed with exit code {return_code}. "
            "See the processing log for command output."
        )
    return completed
