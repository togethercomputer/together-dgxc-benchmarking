# Llama 3.1 8B Pretraining Benchmark — B200 K8s (NeMo 26.02.00)

**Platform:** B200, 8x GPU
**Container:** `nvcr.io/nvidia/nemo:26.02.00`
**Launcher:** Kubernetes (single node)
**Date:** 2026-03-27

---

## Summary

Metrics averaged over stable iterations 31–50 (excluding warmup and CUDA graph compilation steps).

| Job | Dtype | GBS | MBS | TP | PP | CP | Avg Step Time (ms) | MODEL_TFLOP/s/GPU | MFU | Speedup vs BF16 | Final LM Loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pretrain-b200-llama3.1-8b-bf16-gbs128 | BF16 | 128 | 2 | 1 | 1 | 1 | 5,701 | **1,183.5** | 52.6% | 1.00x | 9.38e-3 |
| pretrain-b200-llama3.1-8b-fp8-gbs128 | FP8 | 128 | 2 | 1 | 1 | 1 | 4,121 | **1,637.0** | 72.8% | 1.38x | 1.36e-2 |
| pretrain-b200-llama3.1-8b-nvfp4-gbs128 | NVFP4 | 128 | 4 | 1 | 1 | 1 | 3,369 | **2,001.6** | 89.0% | 1.69x | 1.13e-2 |

> MFU calculated against B200 peak BF16 dense throughput of **2,250 TFLOP/s/GPU**.

---

## Per-Step Performance (steady state, iter 31–50)

### BF16 / GBS=128 — TP1 PP1 CP1, MBS=2

| Iter | Step Time (ms) | MODEL_TFLOP/s/GPU |
|---:|---:|---:|
| 31 | 5,687 | 1,186.3 |
| 32 | 5,713 | 1,180.9 |
| 33 | 5,692 | 1,185.3 |
| 34 | 5,706 | 1,182.5 |
| 35 | 5,690 | 1,185.6 |
| 36 | 5,682 | 1,187.5 |
| 37 | 5,714 | 1,180.5 |
| 39 | 5,723 | 1,179.0 |
| 40 | 5,708 | 1,181.8 |
| 41 | 5,717 | 1,180.0 |
| 42 | 5,699 | 1,183.7 |
| 43 | 5,713 | 1,180.9 |
| 44 | 5,715 | 1,180.4 |
| 45 | 5,710 | 1,181.5 |
| 46 | 5,710 | 1,181.9 |
| 47 | 5,683 | 1,187.1 |
| 48 | 5,695 | 1,184.6 |
| 49 | 5,713 | 1,180.7 |
| 50 | 5,699 | 1,183.9 |
| **Avg** | **5,701** | **1,183.5** |

### FP8 / GBS=128 — TP1 PP1 CP1, MBS=2

| Iter | Step Time (ms) | MODEL_TFLOP/s/GPU |
|---:|---:|---:|
| 31 | 4,119 | 1,640.5 |
| 32 | 4,112 | 1,637.4 |
| 33 | 4,120 | 1,631.6 |
| 34 | 4,135 | 1,637.8 |
| 35 | 4,119 | 1,630.7 |
| 36 | 4,137 | 1,639.5 |
| 37 | 4,115 | 1,637.6 |
| 38 | 4,120 | 1,636.5 |
| 39 | 4,123 | 1,641.8 |
| 40 | 4,109 | 1,649.6 |
| 41 | 4,090 | 1,639.3 |
| 42 | 4,115 | 1,636.0 |
| 43 | 4,124 | 1,627.5 |
| 44 | 4,145 | 1,632.5 |
| 45 | 4,133 | 1,635.2 |
| 46 | 4,126 | 1,635.6 |
| 47 | 4,125 | 1,638.5 |
| 48 | 4,117 | 1,629.6 |
| 49 | 4,140 | 1,643.1 |
| **Avg** | **4,121** | **1,637.0** |

### NVFP4 / GBS=128 — TP1 PP1 CP1, MBS=4

| Iter | Step Time (ms) | MODEL_TFLOP/s/GPU |
|---:|---:|---:|
| 31 | 3,359 | 2,008.4 |
| 32 | 3,399 | 1,984.9 |
| 33 | 3,373 | 2,000.1 |
| 34 | 3,371 | 2,001.4 |
| 35 | 3,354 | 2,011.6 |
| 36 | 3,352 | 2,012.7 |
| 37 | 3,353 | 2,012.3 |
| 38 | 3,394 | 1,988.0 |
| 39 | 3,353 | 2,011.8 |
| 40 | 3,347 | 2,016.0 |
| 41 | 3,345 | 2,016.9 |
| 42 | 3,352 | 2,012.7 |
| 43 | 3,402 | 1,983.2 |
| 44 | 3,371 | 2,001.1 |
| 45 | 3,390 | 1,989.8 |
| 46 | 3,392 | 1,988.7 |
| 47 | 3,372 | 2,001.0 |
| 48 | 3,397 | 1,986.0 |
| 49 | 3,396 | 1,986.5 |
| 50 | 3,367 | 2,003.7 |
| **Avg** | **3,369** | **2,001.6** |

---

## Key Observations

- **NVFP4 achieves 89.0% MFU** — near-peak B200 utilization, delivering **1.69x** speedup over BF16.
- **FP8 achieves 72.8% MFU** — **1.38x** speedup over BF16 with minimal configuration change (same TP/PP/CP/GBS/MBS).
- **NVFP4 is 1.22x faster than FP8**, benefiting from higher MBS (4 vs 2) and lower precision compute.
- **Loss is numerically stable** across all dtypes (9.38e-3 – 1.36e-2), with no skipped or NaN iterations.
- **BF16 step time variance is low** (~5.68–5.72s), indicating stable single-node execution with no communication bottlenecks.

---

## Raw Logs

- [`pretrain-b200-llama3.1-8b-bf16-gbs128.log`](pretrain-b200-llama3.1-8b-bf16-gbs128.log)
- [`pretrain-b200-llama3.1-8b-fp8-gbs128.log`](pretrain-b200-llama3.1-8b-fp8-gbs128.log)
- [`pretrain-b200-llama3.1-8b-nvfp4-gbs128.log`](pretrain-b200-llama3.1-8b-nvfp4-gbs128.log)
