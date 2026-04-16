#!/bin/bash
#
# OFFICIAL NVIDIA BASELINE — Qwen3 30B BF16 — 64 GPUs (8 nodes)
# Source: https://github.com/NVIDIA/dgxc-benchmarking/blob/main/qwen3/README.md
#
# B200 Config: TP=1, PP=1, CP=1, EP=8, ETP=1, VP=1, MBS=1, GBS=4096
# Container: nvidia+nemo+26.02.00 (Megatron-Bridge)
# SeqLen=4096, Layers=48, MoE 3B active
#
# NOTE: Only BF16 is available in official configs for Qwen3 on B200.
#       Your FP8 MX run used a different recipe (fp8_mx).
#

set -euo pipefail

LLMB_DIR=/mnt/vast/johnson/llmb

echo "=== Qwen3 30B BF16 — 64 GPU — NVIDIA Official Baseline ==="
echo "Config: TP=1, PP=1, EP=8, VP=1, MBS=1, GBS=4096"
echo ""

cd "${LLMB_DIR}"
./llmb-run submit \
    -w pretrain_qwen3 \
    -s 30b \
    -d bf16 \
    --scale 64 \
    "$@"
