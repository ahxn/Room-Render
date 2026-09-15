#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVIRONMENT_NAME="${NERFSTUDIO_ENVIRONMENT:-nerfstudio}"
CUDA_ARCHITECTURE="${TCNN_CUDA_ARCHITECTURES:-86}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda was not found. Install Miniforge and open a new Ubuntu shell." >&2
  exit 1
fi

if ! ldconfig -p 2>/dev/null | grep -q 'libOpenGL.so.0'; then
  echo "Warning: libOpenGL.so.0 is missing. Run: sudo apt-get install -y libopengl0" >&2
fi

if conda env list | awk '{print $1}' | grep -qx "$ENVIRONMENT_NAME"; then
  conda env update --name "$ENVIRONMENT_NAME" --file "$REPO_ROOT/environment.yml"
else
  conda env create --name "$ENVIRONMENT_NAME" --file "$REPO_ROOT/environment.yml"
fi

conda run --name "$ENVIRONMENT_NAME" python -m pip install --upgrade pip setuptools
conda run --name "$ENVIRONMENT_NAME" python -m pip install \
  torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
conda run --name "$ENVIRONMENT_NAME" python -m pip install \
  nerfstudio==1.1.5 numpy==1.26.4 scipy==1.11.4 ninja

# Nerfstudio 1.1.5 pins OpenCV 4.10, but that wheel rejects NumPy arrays in the
# verified WSL environment. Keep the known-good runtime without altering deps.
conda run --name "$ENVIRONMENT_NAME" python -m pip install \
  --force-reinstall --no-cache-dir --no-deps opencv-python-headless==4.9.0.80
conda run --name "$ENVIRONMENT_NAME" python -m pip install -e "${REPO_ROOT}[dev,web]"

export TCNN_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURE"
export MAX_JOBS="${MAX_JOBS:-2}"
export LIBRARY_PATH=/usr/lib/wsl/lib
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
conda run --name "$ENVIRONMENT_NAME" --no-capture-output python -m pip install \
  --no-build-isolation \
  "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

conda env config vars set --name "$ENVIRONMENT_NAME" \
  LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  LIBRARY_PATH=/usr/lib/wsl/lib \
  TCNN_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURE" \
  MAX_JOBS="$MAX_JOBS"

echo "Setup complete. Open a new shell, run 'conda activate $ENVIRONMENT_NAME', then run:"
echo "  python reconstruct.py --check-environment"
