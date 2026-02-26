#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt ]]; then
  echo "[train] Missing dataset index: data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt" >&2
  exit 1
fi

if [[ ! -f data/pretrain/VGG_imagenet.npy ]]; then
  echo "[train] Missing pretrained weights: data/pretrain/VGG_imagenet.npy" >&2
  exit 1
fi

python ./ctpn/train_net.py "$@"
