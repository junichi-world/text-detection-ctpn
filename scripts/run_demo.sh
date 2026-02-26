#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f ctpn/text.yml ]]; then
  echo "[demo] ctpn/text.yml not found" >&2
  exit 1
fi

python ./ctpn/demo.py "$@"

