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
sudo apt-get install -y ffmpeg build-essential cmake ninja-build \
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

The verified environment reports all five commands as available and uses:

- Ubuntu 22.04.5 LTS on WSL 2
- NVIDIA GeForce RTX 3060 Ti (8 GB)
- Python 3.10
- FFmpeg/FFprobe 4.4.2
- COLMAP 3.11.1 with CUDA
- CUDA toolkit/compiler 11.8
- PyTorch 2.1.2+cu118
- Nerfstudio 1.1.5

A 20-iteration `splatfacto` smoke test completed on the official D-NeRF sample,
saved a checkpoint, and reopened successfully in `ns-viewer` on port 7007.

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

Additional test for remote SSH

