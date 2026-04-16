#!/bin/bash
#
# OFFICIAL NVIDIA BASELINE — Nemotron4 15B BF16 — 64 GPUs (8 nodes)
# Source: https://github.com/NVIDIA/dgxc-benchmarking/blob/main/nemotron4-15b/README.md
#
# B200 Config: TP=1, PP=1, CP=1, VP=1, MBS=2, GBS=256 (=64*4), SeqLen=4096
# CUDA graphs: enabled
# Container: nvidia+nemo+25.09.00 (NeMo2 framework)
#
# Your result with 26.02+compat_runner: Job 81054 = 1,571 TFLOP/s/GPU
# Tranche-1 target: 1,264 TFLOP/s/GPU (+24.2% above target)
#

set -euo pipefail

DGXC_REPO=/mnt/vast/johnson/dgxc-benchmarking

echo "=== Nemotron4 15B BF16 — 64 GPU — NVIDIA Official Baseline ==="
echo "Config: TP=1, PP=1, VP=1, MBS=2, GBS=256, CUDA_GRAPH=true"
echo ""

echo "To run:"
echo "  cd ${DGXC_REPO}/nemotron4-15b"
echo "  JOB_TOTAL_GPUS=64 GPU_TYPE=b200 DTYPE=bf16 bash launch.sh"
