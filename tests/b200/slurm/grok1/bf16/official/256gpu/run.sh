#!/bin/bash
#
# OFFICIAL NVIDIA BASELINE — Grok1 314B BF16 — 256 GPUs (32 nodes)
# Source: https://github.com/NVIDIA/dgxc-benchmarking/blob/main/grok1/README.md
#
# B200 Config: TP=4, PP=4, EP=8, VP=8, MBS=1, GBS=512 (256*2), SeqLen=8192
# Container: nvidia+nemo+25.09.00 (NeMo2 framework)
# CUDA graphs: disabled (B200 default)
#

set -euo pipefail

DGXC_REPO=/mnt/vast/johnson/dgxc-benchmarking

echo "=== Grok1 314B BF16 — 256 GPU — NVIDIA Official Baseline ==="
echo "Config: TP=4, PP=4, EP=8, VP=8, MBS=1, GBS=512, CUDA_GRAPH=false"
echo ""

cd "${DGXC_REPO}/grok1"
LLMB_INSTALL=/mnt/vast/johnson/llmb \
  JOB_TOTAL_GPUS=256 \
  GPU_TYPE=b200 \
  DTYPE=bf16 \
  SBATCH_ACCOUNT=root \
  SBATCH_PARTITION=batch \
  bash launch.sh
