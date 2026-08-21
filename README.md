# Room Reconstruction

Convert a handheld phone video into an interactive 3D Gaussian-splat scene by
orchestrating FFmpeg, COLMAP (through Nerfstudio), and Nerfstudio's `splatfacto`
pipeline.

## Current status

The first end-to-end CLI slice is in place: input validation, frame extraction,
camera processing, registration-rate checks, training, logs, and run metadata.
The external reconstruction commands require a CUDA-capable Nerfstudio environment.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python reconstruct.py room.mp4 --output results/living-room
```

Validate a video and inspect the planned commands without running GPU work:

```powershell
python reconstruct.py room.mp4 --output results/living-room --dry-run
```

Open a completed reconstruction:

```powershell
python reconstruct.py room.mp4 --output results/living-room --open
```

## Prerequisites

- Python 3.10+
- `ffmpeg` and `ffprobe` on `PATH`
- Nerfstudio with a compatible PyTorch/CUDA environment (`ns-process-data`,
  `ns-train`, and `ns-viewer` on `PATH`)

Run `python reconstruct.py --check-environment` to inspect command availability.

## Output layout

```text
results/living-room/
├── metadata.json
├── logs/
│   └── processing.log
├── frames/
├── processed/
└── reconstruction/
```

Raw videos, extracted frames, reconstructions, and large `.ply` files are ignored
by Git.

## Running tests

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

