# Room Render

Room Render converts a handheld room video into an interactive 3D
Gaussian-splat scene using FFmpeg, OpenCV, COLMAP, Nerfstudio, and PyTorch/CUDA.

## How it works

```text
video -> frame sampling -> camera-pose estimation -> Gaussian-splat training -> viewer/export
```

## Prerequisites

- Linux or Ubuntu under WSL2
- Python 3.10+
- FFmpeg/FFprobe and COLMAP
- Nerfstudio with a compatible PyTorch/CUDA environment
- CUDA-capable NVIDIA GPU for practical training

Check the environment with:

```bash
python reconstruct.py --check-environment
```

## Tested environment

The verified run used:

- Windows 11 with Ubuntu 22.04 under WSL2
- NVIDIA GeForce RTX 3060 Ti (8 GB)
- CUDA 11.8 and PyTorch 2.1.2+cu118
- CUDA compute capability `8.6` (`TCNN_CUDA_ARCHITECTURES=86` for Tiny-CUDA-NN)
- Nerfstudio 1.1.5 and COLMAP 3.11.1
- Python 3.10, NumPy 1.26.4, and OpenCV 4.9.0

Other CUDA-capable NVIDIA GPUs should work with compatible PyTorch and
Nerfstudio versions.

Create the tested WSL environment with:

```bash
bash scripts/setup_wsl.sh
```

The setup script targets CUDA architecture `86` by default. Set
`TCNN_CUDA_ARCHITECTURES` before running it when using a different GPU.

## CLI quick start

```bash
conda activate nerfstudio
python -m pip install -e ".[dev,web]"
python reconstruct.py room.mp4 \
  --output "$HOME/results/living-room" \
  --quality low
```

Inspect the planned commands without GPU work:

```bash
python reconstruct.py room.mp4 \
  --output "$HOME/results/living-room-dry-run" \
  --dry-run
```

Open a completed reconstruction:

```bash
python reconstruct.py --output "$HOME/results/living-room" --open
```

## Quality presets

Low quality uses 15,000 iterations with lower memory use:

```bash
python reconstruct.py room.mp4 \
  --output "$HOME/results/living-room-low" \
  --quality low
```

High quality uses 30,000 iterations and full-resolution training:

```bash
python reconstruct.py room.mp4 \
  --output "$HOME/results/living-room-high" \
  --quality high
```

## Local web interface

The optional browser UI runs locally and processes the video on your computer's
GPU. It is not a hosted service.

```bash
python -m pip install -e ".[web]"
room-render-web
```

Open `http://127.0.0.1:8000`, upload a video, and choose **Low quality** or
**High quality**. The resulting scene can be opened with the CLI command shown
by the page.

## Validation

The pipeline registered 250/250 frames on Room 1 and 180/180 frames on Room 2.
A third capture registered only 2/180 frames and was safely rejected before GPU
training, confirming that the registration quality gate works.

## Outputs

Each run writes metadata, logs, extracted frames, processed camera data, and a
trained checkpoint under its output directory. Export a portable Gaussian-splat
file with:

```bash
python reconstruct.py --output "$HOME/results/living-room" --export
```

## Tests

```bash
python -m pytest
python -m ruff check .
```

## Limitations

Each room requires its own camera-pose solve and training run. Reconstruction
quality depends on camera coverage, motion blur, scene texture, and lighting.
The output is a viewable Gaussian splat rather than a CAD or measurement-grade
mesh.

The public [`docs/`](docs/) directory contains the architecture and capture guide.
