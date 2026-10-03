#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-/home/szq/miniconda3/envs/moge310/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHONUNBUFFERED=1 OPENCV_IO_ENABLE_OPENEXR=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="${TMPDIR:-/data1/szq/moge310/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data1/szq/moge310/cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
CONFIG_PATH="${CONFIG_PATH:-$PWD/configs/train.json}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/data1/szq/moge310/weights/vitl-normal.pt}"
TRAIN_WORKSPACE="${TRAIN_WORKSPACE:-/data1/szq/moge2/权重/workspace/lora96-ft-$(date +%Y%m%d_%H%M%S)}"
exec "$PYTHON_BIN" -m accelerate.commands.launch \
  --multi_gpu --num_processes 2 --mixed_precision bf16 \
  --main_process_port "${MAIN_PROCESS_PORT:-29623}" \
  moge/scripts/trainall-lora96-192-1800.py \
  --enable_ema True --config "$CONFIG_PATH" --workspace "$TRAIN_WORKSPACE" \
  --gradient_accumulation_steps 4 --batch_size_forward 4 \
  --checkpoint "$BASE_CHECKPOINT" --enable_gradient_checkpointing True \
  --vis_every 50 --save_every 50 --enable_mlflow False --seed 42 \
  --num_iterations 1800 "$@"
