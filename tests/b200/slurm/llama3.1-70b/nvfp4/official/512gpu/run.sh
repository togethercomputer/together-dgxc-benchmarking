#!/bin/bash
#
# OFFICIAL NVIDIA BASELINE — Llama 3.1 70B NVFP4 — 512 GPUs (64 nodes)
# Source: https://github.com/NVIDIA/dgxc-benchmarking/blob/main/llama3.1/README.md
#
# B200 Config: TP=2, PP=4, CP=1, VP=5, FSDP=False, MBS=1, GBS=2048
# Container: nvidia+nemo+26.02.00 (Megatron-Bridge)
# SeqLen=8192, Layers=80
#

set -euo pipefail

LLMB_DIR=/mnt/vast/johnson/llmb

echo "=== Llama 3.1 70B NVFP4 — 512 GPU — NVIDIA Official Baseline ==="
echo "Config: TP=2, PP=4, CP=1, VP=5, MBS=1, GBS=2048"
echo ""

cd "${LLMB_DIR}"
./llmb-run submit \
    -w pretrain_llama3.1 \
    -s 70b \
    -d nvfp4 \
    --scale 512 \
    "$@"
