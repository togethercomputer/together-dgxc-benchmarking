#!/bin/bash
#
# OFFICIAL NVIDIA BASELINE — Llama 3.1 70B FP8 — 64 GPUs (8 nodes)
# Source: https://github.com/NVIDIA/dgxc-benchmarking/blob/main/llama3.1/README.md
#
# B200 Config (64-128 GPUs): TP=1, PP=1, CP=1, FSDP=True, MBS=1, GBS=256
# FP8 recipe: mx (on B200)
# Container: nvidia+nemo+26.02.00 (Megatron-Bridge)
# SeqLen=8192, Layers=80
#
# Expected: compare against your optimized results to establish baseline
#

set -euo pipefail

LLMB_DIR=/mnt/vast/johnson/llmb

echo "=== Llama 3.1 70B FP8 — 64 GPU — NVIDIA Official Baseline ==="
echo "Config: TP=1, PP=1, CP=1, FSDP=True, MBS=1, GBS=256, FP8_RECIPE=mx"
echo ""

cd "${LLMB_DIR}"
./llmb-run submit \
    -w pretrain_llama3.1 \
    -s 70b \
    -d fp8 \
    --scale 64 \
    "$@"
