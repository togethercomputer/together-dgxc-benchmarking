#!/bin/bash
#
# OFFICIAL NVIDIA BASELINE — Llama 3.1 405B FP8 — 512 GPUs (64 nodes)
# Source: https://github.com/NVIDIA/dgxc-benchmarking/blob/main/llama3.1/README.md
#
# B200 Config: TP=4, PP=16, CP=1, VP=8, MBS=1, GBS=1536
# Container: nvidia+nemo+26.02.00 (Megatron-Bridge)
# SeqLen=8192, Layers=126
#

set -euo pipefail

LLMB_DIR=/mnt/vast/johnson/llmb

echo "=== Llama 3.1 405B FP8 — 512 GPU — NVIDIA Official Baseline ==="
echo "Config: TP=4, PP=16, CP=1, VP=8, MBS=1, GBS=1536"
echo ""

cd "${LLMB_DIR}"
./llmb-run submit \
    -w pretrain_llama3.1 \
    -s 405b \
    -d fp8 \
    --scale 512 \
    "$@"
