#!/bin/bash
#
# OFFICIAL NVIDIA BASELINE — Nemotron4 15B FP8 — 64 GPUs (8 nodes)
# Source: https://github.com/NVIDIA/dgxc-benchmarking/blob/main/nemotron4-15b/README.md
#
# B200 Config: TP=1, PP=1, CP=1, VP=1, MBS=2, GBS=256 (=64*4), SeqLen=4096
# FP8 recipe: cs (column-statistic)
# CUDA graphs: enabled
# Container: nvidia+nemo+25.09.00 (NeMo2 framework)
#
# NOTE: On this cluster we use 26.02 container + compat_runner.py because
#       the 25.09 container has issues with the cluster's Pyxis setup.
#       Official config has CUDA graphs ON, but FP8 CUDA graphs have a
#       tensor copy bug (cudaErrorInvalidValue) in 26.02 — disable if needed.
#
# Your result with 26.02+compat_runner (CUDA graphs OFF): Job 81053 = 1,916 TFLOP/s/GPU
# Tranche-1 target: 1,908 TFLOP/s/GPU
#

set -euo pipefail

DGXC_REPO=/mnt/vast/johnson/dgxc-benchmarking

echo "=== Nemotron4 15B FP8 — 64 GPU — NVIDIA Official Baseline ==="
echo "Config: TP=1, PP=1, VP=1, MBS=2, GBS=256, CUDA_GRAPH=true, FP8_RECIPE=cs"
echo "Container: 25.09.00 (official) or 26.02.00+compat_runner.py (cluster workaround)"
echo ""

# Official way via NVIDIA launch.sh:
# cd ${DGXC_REPO}/nemotron4-15b
# JOB_TOTAL_GPUS=64 GPU_TYPE=b200 DTYPE=fp8 bash launch.sh

echo "To run with official 25.09 container:"
echo "  cd ${DGXC_REPO}/nemotron4-15b"
echo "  JOB_TOTAL_GPUS=64 GPU_TYPE=b200 DTYPE=fp8 bash launch.sh"
echo ""
echo "To run with 26.02 container (cluster workaround):"
echo "  Use the sbatch.sh from ../64gpu/ with compat_runner.py"
