# Room Reconstruction

Convert a handheld phone video into an interactive 3D Gaussian-splat scene by
orchestrating FFmpeg, COLMAP (through Nerfstudio), and Nerfstudio's `splatfacto`
pipeline.

## Current status

Version 1 works end to end on a real phone capture: input validation, frame
extraction, optional blur filtering, camera-pose recovery, registration checks,
Gaussian-splat optimization, logs, metadata, viewing, and PLY export. The verified
baseline registered all 250 frames and produced a 266,201-Gaussian export.

The remaining validation work is to repeat the experiment on additional rooms.
See the [evaluation report](docs/evaluation.md) for measured results and the
[limitations](#limitations) section for the current boundaries.

## How it works

```text
phone video
   -> FFprobe validation
   -> FFmpeg frame sampling
   -> optional OpenCV blur filtering
   -> COLMAP camera poses (through Nerfstudio)
   -> registration quality gate
   -> Nerfstudio splatfacto optimization
   -> interactive viewer / portable PLY export
```

More detail: [architecture](docs/architecture.md) ·
[capture guide](docs/capture-guide.md) ·
[evaluation methodology](docs/evaluation.md) ·
[demo runbook](docs/demo.md)

## Quick start

```bash
conda activate nerfstudio
python -m pip install -e ".[dev]"
python reconstruct.py room.mp4 --output /home/allen/results/living-room
```

Validate a video and inspect the planned commands without running GPU work:

```bash
python reconstruct.py room.mp4 --output /home/allen/results/living-room-dry-run --dry-run
```

Open a completed reconstruction:

```bash
python reconstruct.py --output /home/allen/results/living-room --open
```

Resume after a failed or interrupted stage. Existing frames, camera poses, and
completed checkpoints are detected and reused:

```bash
python reconstruct.py room.mp4 --output /home/allen/results/living-room --resume
```

Optionally score every extracted frame using the variance of its grayscale
Laplacian and exclude blurry frames before camera-pose estimation:

```bash
python reconstruct.py room.mp4 \
  --output results/living-room-filtered \
  --filter-blurry-frames \
  --blur-threshold 2.5
```

Filtering writes `frame_quality.json` with every score and acceptance decision,
and links accepted images into `selected_frames/`. The default is off so an
unfiltered run remains available as an evaluation baseline.

Use a checked-in YAML configuration for repeatable runs:

```bash
python reconstruct.py room.mp4 \
  --output results/living-room \
  --config configs/default.yml
```

For a faster preview with lower GPU-memory pressure, the low-memory preset uses
180 frames, 15,000 training iterations, half-resolution training cameras, and
CPU image caching:

```bash
python reconstruct.py room.mp4 \
  --output results/living-room-preview \
  --config configs/low-memory.yml
```

Command-line blur options override their YAML values. Use
`--filter-blurry-frames` or `--no-filter-blurry-frames` to explicitly enable or
disable filtering for a configured run.

Export the latest completed checkpoint to a portable Gaussian-splat PLY:

```bash
python reconstruct.py --output results/living-room --export
```

To make the export immediately accessible from Windows, choose a mounted Windows
directory:

```bash
python reconstruct.py \
  --output "/home/allen/results/living-room" \
  --export \
  --export-dir "/mnt/c/Users/allen/Downloads/living-room-export"
```

## Local browser interface

The optional local web UI wraps the same CLI and runs reconstruction on the
computer where it is started. It does not upload videos or require a hosted
service:

```bash
python -m pip install -e ".[web]"
room-reconstruct-web
```

Open `http://127.0.0.1:8000`, choose an original-quality `.mov` or `.mp4`, and
start a reconstruction. The page polls the local job and reports when the
result is ready. The completed scene can then be opened with the saved
Nerfstudio configuration:

```bash
python reconstruct.py --output /home/allen/results/web-<job-id> --open
```

The browser UI is intentionally a thin layer around the tested command-line
pipeline. The CLI remains the reproducible entry point for automation and
resume workflows.

## Prerequisites

- Python 3.10+
- `ffmpeg` and `ffprobe` on `PATH`
- Nerfstudio with a compatible PyTorch/CUDA environment (`ns-process-data`,
  `ns-train`, `ns-viewer`, and `ns-export` on `PATH`)

Run `python reconstruct.py --check-environment` to inspect command availability
and exercise the installed OpenCV, NumPy, PyTorch, CUDA, and Nerfstudio runtime.

## Verified Windows/WSL setup

Native Windows support is fragile, so the tested configuration uses Ubuntu 22.04
under WSL 2. From an Administrator PowerShell prompt, install WSL and restart if
requested:

```powershell
wsl --install -d Ubuntu-22.04
```

In Ubuntu, install the system packages and Miniforge:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg build-essential cmake ninja-build libopengl0 \
  python3-dev python3-venv python3-pip
curl -fL \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
  -o /tmp/Miniforge3-Linux-x86_64.sh
bash /tmp/Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init bash
```

Open a new Ubuntu shell, then create the reconstruction environment. The CUDA
architecture value `86` is correct for the verified RTX 3060 Ti; use the value
for your GPU when running different hardware.

```bash
conda create -y -n nerfstudio python=3.10 pip
conda install -y -n nerfstudio -c nvidia/label/cuda-11.8.0 \
  cuda-nvcc=11.8.89 cuda-cudart-dev=11.8.89
conda install -y -n nerfstudio colmap
conda activate nerfstudio

python -m pip install \
  torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
python -m pip install nerfstudio
python -m pip install numpy==1.26.4 scipy==1.11.4 \
  setuptools==80.9.0 ninja

export TCNN_CUDA_ARCHITECTURES=86
export MAX_JOBS=2
export LIBRARY_PATH=/usr/lib/wsl/lib
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
python -m pip install --no-build-isolation \
  "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

conda env config vars set -n nerfstudio \
  LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  LIBRARY_PATH=/usr/lib/wsl/lib \
  TCNN_CUDA_ARCHITECTURES=86 \
  MAX_JOBS=2
```

Open one more Ubuntu shell, activate the environment, and run the project from
its Windows-mounted path:

```bash
conda activate nerfstudio
cd "/mnt/c/Users/allen/Documents/ChatGPT/Room Reconstruction"
python reconstruct.py --check-environment
```

For a new or repaired environment, the repository also contains a repeatable
setup script. It preserves the known-good OpenCV 4.9 runtime required by this
WSL configuration:

```bash
cd "/mnt/c/Users/allen/Documents/ChatGPT/Room Reconstruction"
bash scripts/setup_wsl.sh
```

The verified environment reports all five commands as available and uses:

- Ubuntu 22.04.5 LTS on WSL 2
- NVIDIA GeForce RTX 3060 Ti (8 GB)
- Python 3.10
- FFmpeg/FFprobe 4.4.2
- COLMAP 3.11.1 with CUDA
- CUDA toolkit/compiler 11.8
- PyTorch 2.1.2+cu118
- Nerfstudio 1.1.5
- NumPy 1.26.4
- OpenCV 4.9.0.80

A 20-iteration `splatfacto` smoke test completed on the official D-NeRF sample,
saved a checkpoint, and reopened successfully in `ns-viewer` on port 7007.

## Output layout

```text
results/living-room/
├── metadata.json
├── logs/
│   └── processing.log
├── frames/
├── selected_frames/
├── frame_quality.json
├── processed/
├── reconstruction/
└── exports/
    └── splat.ply
```

Raw videos, extracted frames, reconstructions, and large `.ply` files are ignored
by Git.

Each run writes stage state and the exact completed `config.yml` path to
`metadata.json`. A normal run refuses to use a directory that already contains
reconstruction artifacts; pass `--resume` explicitly to reuse them.
Successful exports are appended to the metadata with their path, size, source
configuration, and creation time.

## Running tests

```bash
python -m pytest
python -m ruff check .
```

## Summarizing experiments

Generate a Markdown results table from one or more completed run directories:

```bash
python scripts/summarize_results.py \
  /home/allen/results/my-room \
  /home/allen/results/my-room-filtered-v2
```

Pass either a run directory or its `metadata.json`. The output can be pasted into
an issue, README, or evaluation notes. Metrics produced separately by
`ns-eval`—PSNR, SSIM, and LPIPS—should be added to the evaluation report with the
exact config used.

## Troubleshooting

**`Input video does not exist`** — replace example paths with the actual WSL path.
A Windows download such as `C:\Users\allen\Downloads\room.mp4` is available at
`/mnt/c/Users/allen/Downloads/room.mp4`.

**Ubuntu waits for OOBE** — finish the first-launch username and password prompts
before running project commands. If WSL was just enabled, restart Windows first.

**Environment check fails** — activate the `nerfstudio` Conda environment, then run
`python reconstruct.py --check-environment`. Re-run `scripts/setup_wsl.sh` only when
repairing or recreating the environment.

**OpenCV rejects a valid NumPy array** — verify the pinned known-good versions from
the setup section. Mixing OpenCV wheels or NumPy 2.x with this environment can cause
binary incompatibilities.

**Registration is low** — recapture with slower movement, more overlap, and more
sideways translation. Training longer cannot repair incorrect camera poses. See the
[capture guide](docs/capture-guide.md).

**CUDA runs out of memory** — try `configs/low-memory.yml`, close other GPU-heavy
applications, and stop any old viewer or training process before retrying.

**GPU or WSL memory remains allocated after Ctrl+C** — close the viewer tab and stop
remaining Nerfstudio processes. From PowerShell, `wsl --shutdown` fully releases the
WSL virtual machine when no Linux work needs to remain running.

## Limitations

- Each new room requires its own camera-pose solve and Gaussian-splat optimization;
  this is scene fitting, not inference from a reusable room model.
- A CUDA-capable NVIDIA GPU is required for practical training. A Mac can SSH into
  the Windows/WSL host and start or inspect a run, but the GPU work still executes
  on that host.
- Reflective surfaces, blank walls, motion blur, moving objects, changing exposure,
  and low-parallax captures can weaken camera registration or visual quality.
- The output is a viewable Gaussian splat, not a clean CAD model or measurement-grade
  mesh.
- Version 1 is a local CLI. Upload UI, job queues, cloud storage, and hosted viewing
  are intentionally deferred until multi-room reliability is measured.

## Resume-ready summary

Built a Python video-to-3D room reconstruction pipeline integrating OpenCV, FFmpeg,
COLMAP, PyTorch, and Nerfstudio; automated frame selection, camera-registration
quality gates, resumable Gaussian-splat optimization, structured run metadata,
interactive viewing, and PLY export, with 100% camera registration on the first
250-frame room capture.

