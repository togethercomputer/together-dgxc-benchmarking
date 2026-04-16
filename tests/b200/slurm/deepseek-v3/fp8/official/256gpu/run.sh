#!/bin/bash
#
# OFFICIAL NVIDIA BASELINE — DeepSeek-V3 671B FP8 — 256 GPUs (32 nodes)
# Source: https://github.com/NVIDIA/dgxc-benchmarking/blob/main/deepseek_v3/README.md
#
# Container: nvidia+nemo+26.02.00 (Megatron-Bridge)
# SeqLen=4096, Layers=61, MoE 671B
#
# NOTE: BF16 at 512 GPUs failed with NCCL collective timeouts.
#       FP8 at 256 GPUs may avoid the issue due to smaller scale.
#

set -euo pipefail

LLMB_DIR=/mnt/vast/johnson/llmb

echo "=== DeepSeek-V3 671B FP8 — 256 GPU — NVIDIA Official Baseline ==="
echo ""

cd "${LLMB_DIR}"
./llmb-run submit \
    -w pretrain_deepseek-v3 \
    -d fp8 \
    --scale 256 \
    "$@"
