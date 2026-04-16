#!/bin/bash
#
# OFFICIAL NVIDIA BASELINE — Nemotron4 15B FP8 — 256 GPUs (32 nodes)
# Source: https://github.com/NVIDIA/dgxc-benchmarking/blob/main/nemotron4-15b/README.md
#
# B200 Config: TP=1, PP=1, CP=1, VP=1, MBS=2, GBS=1024 (=256*4), SeqLen=4096
# FP8 recipe: cs (column-statistic)
# CUDA graphs: enabled
# Container: nvidia+nemo+25.09.00 (NeMo2 framework)
#
# IMPORTANT: Official GBS=1024 differs from your optimized GBS=2048.
#   - Official GBS=1024: ~1,800 TFLOP/s (Job 81060)
#   - Your GBS=2048:     ~1,904 TFLOP/s (Job 81072, +6%)
#   - Your GBS=2560:     ~1,923 TFLOP/s (Job 81071, +7%)
#   The GBS increase hides NCCL allreduce behind gradient accumulation.
#
# Tranche-1 target: 1,908 TFLOP/s/GPU
#

set -euo pipefail

DGXC_REPO=/mnt/vast/johnson/dgxc-benchmarking

echo "=== Nemotron4 15B FP8 — 256 GPU — NVIDIA Official Baseline ==="
echo "Config: TP=1, PP=1, VP=1, MBS=2, GBS=1024, CUDA_GRAPH=true, FP8_RECIPE=cs"
echo "Official GBS=1024 gives ~1,800 TFLOP/s (Job 81060)"
echo "Optimized GBS=2048 gives ~1,904 TFLOP/s (Job 81072)"
echo ""

# Official way via NVIDIA launch.sh:
# cd ${DGXC_REPO}/nemotron4-15b
# JOB_TOTAL_GPUS=256 GPU_TYPE=b200 DTYPE=fp8 bash launch.sh

echo "To run with official 25.09 container:"
echo "  cd ${DGXC_REPO}/nemotron4-15b"
echo "  JOB_TOTAL_GPUS=256 GPU_TYPE=b200 DTYPE=fp8 bash launch.sh"
echo ""
echo "To run with 26.02 container (cluster workaround):"
echo "  Use the sbatch.sh from ../256gpu/ with OVERRIDE_GBS=1024"
