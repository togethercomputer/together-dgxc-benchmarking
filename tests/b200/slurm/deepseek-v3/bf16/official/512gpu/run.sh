#!/bin/bash
#
# OFFICIAL NVIDIA BASELINE — DeepSeek-V3 671B BF16 — 512 GPUs (64 nodes)
# Source: https://github.com/NVIDIA/dgxc-benchmarking/blob/main/deepseek_v3/README.md
#
# B200 Config: TP=1, PP=16, CP=1, EP=8, ETP=1, VP=None, MBS=1, GBS=4096
# Container: nvidia+nemo+26.02.00 (Megatron-Bridge)
# SeqLen=4096, Layers=61
#
# NOTE: Your runs (Jobs 81073, 81076) used this exact config but both failed
#       with NCCL collective timeouts (~640s). This is a known issue on this cluster
#       at 512-GPU scale with PP=16 cross-node P2P.
#

set -euo pipefail

LLMB_DIR=/mnt/vast/johnson/llmb

echo "=== DeepSeek-V3 671B BF16 — 512 GPU — NVIDIA Official Baseline ==="
echo "Config: TP=1, PP=16, EP=8, VP=None, MBS=1, GBS=4096"
echo "Known issue: NCCL timeout at 512 GPUs (Jobs 81073, 81076 both failed)"
echo ""

cd "${LLMB_DIR}"
./llmb-run submit \
    -w pretrain_deepseek-v3 \
    -d bf16 \
    --scale 512 \
    "$@"
