"""Repository-local entry point for the room reconstruction CLI."""

import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

main = import_module("room_reconstruction.cli").main


if __name__ == "__main__":
    raise SystemExit(main())
