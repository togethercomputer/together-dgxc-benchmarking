#!/bin/bash
#
# OFFICIAL NVIDIA BASELINE — Llama 3.1 405B NVFP4 — 256 GPUs (32 nodes)
# Source: https://github.com/NVIDIA/dgxc-benchmarking/blob/main/llama3.1/README.md
#
# B200 Config: TP=4, PP=16, CP=1, VP=8, MBS=1, GBS=1536
# Container: nvidia+nemo+26.02.00 (Megatron-Bridge)
# SeqLen=8192, Layers=126
#
# NOTE: This is the same config as your configA_pp16 (Job 80073: 1,564 TFLOP/s).
#       Your optimized configD_pp8 (PP=8) achieved 2,006 TFLOP/s (+28%).
#

set -euo pipefail

LLMB_DIR=/mnt/vast/johnson/llmb

echo "=== Llama 3.1 405B NVFP4 — 256 GPU — NVIDIA Official Baseline ==="
echo "Config: TP=4, PP=16, CP=1, VP=8, MBS=1, GBS=1536"
echo "Known result: Job 80073 = 1,564 TFLOP/s/GPU (this IS the official baseline)"
echo ""

cd "${LLMB_DIR}"
./llmb-run submit \
    -w pretrain_llama3.1 \
    -s 405b \
    -d nvfp4 \
    --scale 256 \
    "$@"
