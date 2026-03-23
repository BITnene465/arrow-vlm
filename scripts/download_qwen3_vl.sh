#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

MODEL_ID="${MODEL_ID:-${VLM_DET_REMOTE_MODEL_NAME_OR_PATH:-Qwen/Qwen3-VL-2B-Instruct}}"
LOCAL_DIR="${LOCAL_DIR:-$ROOT_DIR/${VLM_DET_MODEL_NAME_OR_PATH:-models/Qwen3-VL-2B-Instruct}}"

mkdir -p "$LOCAL_DIR"

if command -v hf >/dev/null 2>&1; then
  hf download "$MODEL_ID" --local-dir "$LOCAL_DIR"
else
  huggingface-cli download "$MODEL_ID" --local-dir "$LOCAL_DIR"
fi

echo "Model downloaded to: $LOCAL_DIR"
