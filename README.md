# Room Reconstruction

Convert a handheld phone video into an interactive 3D Gaussian-splat scene by
orchestrating FFmpeg, COLMAP (through Nerfstudio), and Nerfstudio's `splatfacto`
pipeline.

## Current status

The first end-to-end CLI slice is in place: input validation, frame extraction,
camera processing, registration-rate checks, training, logs, and run metadata.
The external reconstruction commands require a CUDA-capable Nerfstudio environment.
Camera processing uses COLMAP sequential matching because the inputs are ordered
video frames.

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
python reconstruct.py --output results/living-room --open
```

Resume after a failed or interrupted stage. Existing frames, camera poses, and
completed checkpoints are detected and reused:

```powershell
python reconstruct.py room.mp4 --output results/living-room --resume
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

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

