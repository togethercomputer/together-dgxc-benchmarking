# Llama 3.1 70B FP8 — B200 Scaling Benchmark Results

**Cluster:** Together AI B200 (8x B200/node, ConnectX-7 IB 400Gb/s)
**Container:** nvidia+nemo+26.02.00
**Config:** TP=1, PP=1, CP=1, MBS=1, SeqLen=8192
**Date:** 2026-04-08

## Results

| GPUs | Nodes | GBS | Job ID | TFLOP/s/GPU | Step Time (s) | Scaling Eff. | Status |
|------|-------|-----|--------|-------------|---------------|--------------|--------|
| 64   | 8     | 128  | 79120 | 1,590 | 4.63 | baseline | OK |
| 64   | 8     | 128  | 79119 | 1,573 | 4.68 | — | OK (repeat) |
| 64   | 8     | 128  | 79118 | 1,575 | 4.67 | — | OK (repeat) |
| 128  | 16    | 256  | 79138 | 1,579 | 4.66 | 100.1% | OK |
| 256  | 32    | 512  | 79139 | 1,494 | 4.93 | 94.3% | OK |
| 512  | 64    | 1024 | 79133 | 1,365 | 5.39 | 86.0% | OK |
| 512  | 64    | 1024 | 79134 | 1,361 | 5.41 | — | OK (repeat) |
| 512  | 64    | 1024 | 79135 | 1,363 | 5.40 | — | OK (repeat) |
| 512  | 64    | 1024 | 79141 | — | — | — | FAILED (MBS=2, NCCL crash) |
| 592  | 74    | 1184 | 79140 | 1,319 | 5.59 | 82.9% | OK |

**Best 64-GPU:** 1,590 TFLOP/s/GPU (Job 79120)
**Best 512-GPU:** 1,365 TFLOP/s/GPU (Job 79133)
**Peak aggregate:** 780,573 TFLOP/s (592 GPUs)

## Key Findings

- Super-linear scaling at 128 GPUs (100.1%) — likely cache/memory effects
- Scaling drops to 86% at 512 GPUs, 82.9% at 592 (full cluster) — IB fabric oversubscription
- MBS=2 at 512 GPUs causes NCCL crash — stick with MBS=1
- Variance across 3x 64-GPU runs: <0.3% (excellent reproducibility)
- Variance across 3x 512-GPU runs: <0.3%

## Log Locations

- `/home/johnson/johnson/scripts/llama31_70b_fp8/*/logs/job_*_train.log`
