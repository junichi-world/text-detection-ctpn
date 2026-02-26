#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKIP_PIP=0

if [[ "${1:-}" == "--skip-pip" ]]; then
  SKIP_PIP=1
fi

cd "${ROOT_DIR}"

if [[ "${SKIP_PIP}" -eq 0 ]]; then
  echo "[setup] Installing Python dependencies (keeping TensorFlow from the base image)"
  python -m pip install --upgrade pip
  python -m pip install --no-cache-dir "numpy<2"
  python -m pip install --no-cache-dir \
    easydict \
    opencv-python \
    Pillow \
    matplotlib \
    tqdm \
    PyYAML \
    scipy \
    pandas \
    Cython \
    setuptools
fi

echo "[setup] Building Cython extensions in lib/utils"
(
  cd lib/utils
  bash ./make.sh
)

echo "[setup] Done"
echo "[setup] If training, ensure these exist:"
echo "  - data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt"
echo "  - data/pretrain/VGG_imagenet.npy"

