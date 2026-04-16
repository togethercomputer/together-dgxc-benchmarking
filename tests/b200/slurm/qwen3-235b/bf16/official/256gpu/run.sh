#!/bin/bash
#
# OFFICIAL NVIDIA BASELINE — Qwen3 235B BF16 — 256 GPUs (32 nodes)
# Source: https://github.com/NVIDIA/dgxc-benchmarking/blob/main/qwen3/README.md
#
# B200 Config: TP=1, PP=8, CP=1, EP=8, ETP=1, VP=4, MBS=1, GBS=8192
# Container: nvidia+nemo+26.02.00 (Megatron-Bridge)
# SeqLen=4096, Layers=94, 128 experts, 22B active
#
# NOTE: Your baseline (Job 80077: 514 TFLOP/s) already matches this official config exactly.
#       This is confirmed as the official NVIDIA baseline.
#

set -euo pipefail

LLMB_DIR=/mnt/vast/johnson/llmb

echo "=== Qwen3 235B BF16 — 256 GPU — NVIDIA Official Baseline ==="
echo "Config: TP=1, PP=8, CP=1, EP=8, VP=4, MBS=1, GBS=8192"
echo "Known result: Job 80077 = 514 TFLOP/s/GPU (matches official baseline)"
echo ""

cd "${LLMB_DIR}"
./llmb-run submit \
    -w pretrain_qwen3 \
    -s 235b \
    -d bf16 \
    --scale 256 \
    "$@"
