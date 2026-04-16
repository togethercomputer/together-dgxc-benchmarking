#!/bin/bash
#
# OFFICIAL NVIDIA BASELINE — Llama 3.1 70B FP8 — 256 GPUs (32 nodes)
# Source: https://github.com/NVIDIA/dgxc-benchmarking/blob/main/llama3.1/README.md
#
# B200 Config (256+ GPUs): TP=2, PP=4, CP=1, VP=5, FSDP=False, MBS=1, GBS=1024
# FP8 recipe: mx (on B200)
# Container: nvidia+nemo+26.02.00 (Megatron-Bridge)
# SeqLen=8192, Layers=80
#
# NOTE: Your previous runs used TP=1, PP=1 at 256 GPUs. The official config
#       switches to TP=2, PP=4, VP=5 at 256+ GPUs. This is a different config
#       and may yield different results.
#

set -euo pipefail

LLMB_DIR=/mnt/vast/johnson/llmb

echo "=== Llama 3.1 70B FP8 — 256 GPU — NVIDIA Official Baseline ==="
echo "Config: TP=2, PP=4, CP=1, VP=5, MBS=1, GBS=1024, FP8_RECIPE=mx"
echo ""

cd "${LLMB_DIR}"
./llmb-run submit \
    -w pretrain_llama3.1 \
    -s 70b \
    -d fp8 \
    --scale 256 \
    "$@"
