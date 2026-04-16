# Llama 3.1 405B NVFP4 — B200 Benchmark Results

**Cluster:** Together AI B200 (8x B200/node, ConnectX-7 IB 400Gb/s)
**Container:** nvidia+nemo+26.02.00 (Megatron-Bridge 6b3b5ba7e)
**Date:** 2026-04-09 to 2026-04-11

## 256 GPU Results (32 nodes)

| Config | TP | PP | VP | CP | DP | GBS | Job ID | TFLOP/s/GPU | Step Time (s) | Status |
|--------|----|----|----|----|-----|-----|--------|-------------|---------------|--------|
| A (baseline) | 4 | 16 | 8 | 1 | 4 | 1536 | 80073 | 1,564 | 79.3 | OK |
| A (old mbridge) | 4 | 16 | 8 | 1 | 4 | 1536 | 79866 | 1,352 | 91.8 | OK |
| A+mbridge | 4 | 16 | 8 | 1 | 4 | 1536 | 80073 | 1,568 | 79.3 | OK (+7% mbridge upgrade) |
| B (mbridge default) | 4 | 8 | 4 | 2 | 4 | 128 | — | 1,491 | — | OK |
| C | 8 | 8 | — | — | — | — | — | — | — | FAILED (NCCL hang) |
| **D (winner)** | **4** | **8** | **4** | **1** | **8** | **1536** | **80076** | **2,006** | **61.8** | **OK** |
| E | 4 | 8 | — | 2 | — | — | — | — | — | tested |
| F | 4 | 8 | 8 | — | — | — | — | — | — | tested |

## 512 GPU Results (64 nodes)

| Config | Job ID | TFLOP/s/GPU | Step Time (s) | Status |
|--------|--------|-------------|---------------|--------|
| A (PP=16) | 79952 | 1,384 | — | OK |
| D (PP=8) | 79953 | 1,818 | — | OK |

## Key Findings

- **PP=16→PP=8 = +38% at 256 GPUs** (pipeline bubble: 43% → 15.5%)
- PP=8 advantage narrows at 512 GPUs: 38% → 31% (DP gradient sync cost grows)
- TP=8 hangs — not viable on this cluster, stick with TP=4
- Megatron-Bridge upgrade (4df8c9739 → 6b3b5ba7e) gave +7% on PP=16
- SHARP (NCCL_COLLNET_ENABLE=1) causes hangs — no SHARP AM reservations on cluster
- P2P bandwidth degrades from 42 GB/s (2-4 nodes) to 14.5 GB/s (32+ nodes)

## Profiler Analysis

- Nsys traces: `/home/johnson/johnson/traces/nsys/profile_79846_*`
- Bottleneck report: `~/johnson/worklog/profiler_reports/405b_nvfp4_nsys_bottleneck.md`

## Log Locations

- `/home/johnson/johnson/scripts/llama31_405b_nvfp4/*/logs/`
- `/mnt/vast/johnson/llmb/workloads/pretrain_llama3.1/experiments/`
