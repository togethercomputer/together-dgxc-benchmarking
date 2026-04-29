#!/bin/bash
#
# OFFICIAL NVIDIA BASELINE — Grok1 314B FP8 — 512 GPUs (64 nodes)
# Source: https://github.com/NVIDIA/dgxc-benchmarking/blob/main/grok1/README.md
#
# B200 Config: TP=4, PP=4, EP=8, VP=8, MBS=1, GBS=1024 (512*2), SeqLen=8192
# Container: nvidia+nemo+25.09.00 (NeMo2 framework)
# Requires llmb_venv for fiddle dependency
#

set -euo pipefail

DGXC_REPO=/mnt/vast/johnson/dgxc-benchmarking

echo "=== Grok1 314B FP8 — 512 GPU — NVIDIA Official Baseline ==="
echo "Config: TP=4, PP=4, EP=8, VP=8, MBS=1, GBS=1024"
echo ""

source /mnt/vast/johnson/llmb_venv/bin/activate

cd "${DGXC_REPO}/grok1"
LLMB_INSTALL=/mnt/vast/johnson/llmb \
  JOB_TOTAL_GPUS=512 \
  GPU_TYPE=b200 \
  DTYPE=fp8 \
  SBATCH_ACCOUNT=root \
  SBATCH_PARTITION=batch \
  bash launch.sh
