#!/bin/bash
#
# OFFICIAL NVIDIA BASELINE — Nemotron4 340B FP8 — 512 GPUs (64 nodes)
# Source: https://github.com/NVIDIA/dgxc-benchmarking/blob/main/nemotron4-340b/README.md
#
# B200 Config: TP=8, PP=4, VP=12, MBS=1, GBS=128 (512/4), FP8 recipe=cs
# Container: nvidia+nemo+25.07.01 (NeMo2 framework)
# CUDA graphs: enabled
#

set -euo pipefail

DGXC_REPO=/mnt/vast/johnson/dgxc-benchmarking

echo "=== Nemotron4 340B FP8 — 512 GPU — NVIDIA Official Baseline ==="
echo "Config: TP=8, PP=4, VP=12, MBS=1, GBS=128, CUDA_GRAPH=true, FP8_RECIPE=cs"
echo ""

cd "${DGXC_REPO}/nemotron4-340b"
LLMB_INSTALL=/mnt/vast/johnson/llmb \
  JOB_TOTAL_GPUS=512 \
  GPU_TYPE=b200 \
  DTYPE=fp8 \
  SBATCH_ACCOUNT=root \
  SBATCH_PARTITION=batch \
  bash launch.sh
