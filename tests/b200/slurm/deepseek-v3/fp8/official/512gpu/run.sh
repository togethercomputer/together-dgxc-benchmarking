#!/bin/bash
#
# OFFICIAL NVIDIA BASELINE — DeepSeek-V3 671B FP8 — 512 GPUs (64 nodes)
# Source: https://github.com/NVIDIA/dgxc-benchmarking/blob/main/deepseek_v3/README.md
#
# Container: nvidia+nemo+26.02.00 (Megatron-Bridge)
# SeqLen=4096, Layers=61, MoE 671B
#
# WARNING: BF16 at 512 GPUs failed with NCCL collective timeouts (~640s).
#          FP8 may hit the same issue at this scale.
#

set -euo pipefail

LLMB_DIR=/mnt/vast/johnson/llmb

echo "=== DeepSeek-V3 671B FP8 — 512 GPU — NVIDIA Official Baseline ==="
echo "WARNING: BF16 at 512 GPUs failed NCCL timeout — FP8 may also fail"
echo ""

cd "${LLMB_DIR}"
./llmb-run submit \
    -w pretrain_deepseek-v3 \
    -d fp8 \
    --scale 512 \
    "$@"
