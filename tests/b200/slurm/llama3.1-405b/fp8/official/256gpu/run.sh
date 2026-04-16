#!/bin/bash
#
# OFFICIAL NVIDIA BASELINE — Llama 3.1 405B FP8 — 256 GPUs (32 nodes)
# Source: https://github.com/NVIDIA/dgxc-benchmarking/blob/main/llama3.1/README.md
#
# B200 Config: TP=4, PP=8, CP=2, VP=8, MBS=1, GBS=1536
# FP8 recipe: mx (on B200)
# Container: nvidia+nemo+26.02.00 (Megatron-Bridge)
# SeqLen=8192, Layers=126
#
# NOTE: The official FP8 config already uses PP=8 (not PP=16 like NVFP4).
#       It also adds CP=2 (context parallelism). This has NOT been run yet.
#

set -euo pipefail

LLMB_DIR=/mnt/vast/johnson/llmb

echo "=== Llama 3.1 405B FP8 — 256 GPU — NVIDIA Official Baseline ==="
echo "Config: TP=4, PP=8, CP=2, VP=8, MBS=1, GBS=1536, FP8_RECIPE=mx"
echo "Status: NOT YET RUN"
echo ""

cd "${LLMB_DIR}"
./llmb-run submit \
    -w pretrain_llama3.1 \
    -s 405b \
    -d fp8 \
    --scale 256 \
    "$@"
