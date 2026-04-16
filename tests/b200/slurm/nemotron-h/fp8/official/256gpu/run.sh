#!/bin/bash
#
# OFFICIAL NVIDIA BASELINE — Nemotron-H 56B FP8 — 256 GPUs (32 nodes)
# Source: https://github.com/NVIDIA/dgxc-benchmarking/blob/main/nemotron-h/README.md
#
# B200 Config: TP=2, MBS=1, GBS=768 (256*3), FP8 recipe=cs
# Container: nvidia+nemo+26.02.00 (Megatron-Bridge)
# SeqLen=8192, Layers=118, 56B parameters
#

set -euo pipefail

LLMB_DIR=/mnt/vast/johnson/llmb

echo "=== Nemotron-H 56B FP8 — 256 GPU — NVIDIA Official Baseline ==="
echo "Config: TP=2, MBS=1, GBS=768, FP8_RECIPE=cs"
echo ""

cd "${LLMB_DIR}"
./llmb-run submit \
    -w pretrain_nemotron-h \
    -s 56b \
    -d fp8 \
    --scale 256 \
    "$@"
