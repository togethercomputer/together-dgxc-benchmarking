# Qwen3-30B-A3B Pretraining Benchmark — B200 K8s (NeMo 26.02.00)
**Platform:** B200, 8x GPU
**Container:** `nvcr.io/nvidia/nemo:26.02.00`
**Launcher:** Kubernetes (single node)
**Date:** 2026-03-27

---

## Summary

Metrics averaged over stable iterations 31–50 (profile job: iter 31–45, excluding profiling overhead).

| Job | Dtype | GBS | Avg Step Time (ms) | Throughput (samples/s) | MODEL_TFLOP/s/GPU | Final LM Loss |
|---|---|---:|---:|---:|---:|---:|
| pretrain-qwen3-30b-bf16 | BF16 | 64 | 3,479 | 18.4 | **216.7** | 8.174 |
| pretrain-qwen3-30b-bf16-profile | BF16 + Profiling | 64 | 3,643 | 17.6 | 206.9 | 8.175 |
| pretrain-b200-qwen3-30b-fp8-gbs512 | FP8 | 512 | 41,059 | 12.5 | 146.9 | 8.114 |
| pretrain-b200-qwen3-30b-fp8mx-gbs512 | FP8-MX | 512 | 39,487 | 13.0 | **152.7** | 8.114 |

---

## Per-Step Performance (steady state, iter 31–50)

### BF16 / GBS=64 — TP1 PP1 CP1 EP8 ETP1, MBS=1
| Iter | Step Time (ms) | MODEL_TFLOP/s/GPU |
|---:|---:|---:|
| 31 | 3,460 | 217.9 |
| 35 | 3,479 | 216.7 |
| 40 | 3,450 | 218.5 |
| 45 | 3,462 | 217.8 |
| 50 | 3,503 | 215.2 |
| **Avg** | **3,479** | **216.7** |

### FP8 / GBS=512
| Iter | Step Time (ms) | MODEL_TFLOP/s/GPU |
|---:|---:|---:|
| 31 | 41,108 | 146.7 |
| 35 | 41,559 | 145.1 |
| 40 | 41,149 | 146.6 |
| 45 | 41,238 | 146.2 |
| 50 | 40,797 | 147.8 |
| **Avg** | **41,059** | **146.9** |

### FP8-MX / GBS=512
| Iter | Step Time (ms) | MODEL_TFLOP/s/GPU |
|---:|---:|---:|
| 31 | 39,772 | 151.6 |
| 35 | 39,302 | 153.5 |
| 40 | 39,745 | 151.7 |
| 45 | 39,350 | 153.3 |
| 50 | 39,803 | 151.5 |
| **Avg** | **39,487** | **152.7** |

---

## Key Observations

- **FP8-MX outperforms FP8** by ~3.9% in step time and TFLOP/s (152.7 vs 146.9).
- **Profiling overhead** costs ~4.7% in TFLOP/s (206.9 vs 216.7 for BF16/GBS=64).
- **Loss is consistent** across all dtypes (~8.11–8.18), confirming numerical stability.
- GBS=512 jobs have lower TFLOP/s/GPU than GBS=64 due to larger per-GPU batch (64 vs 8 samples/GPU) increasing memory pressure; a BF16/GBS=512 run would be needed for a direct dtype comparison.

---

## Raw Logs

- [`pretrain-qwen3-30b-bf16.log`](pretrain-qwen3-30b-bf16.log)
- [`pretrain-qwen3-30b-bf16-profile.log`](pretrain-qwen3-30b-bf16-profile.log)
- [`pretrain-b200-qwen3-30b-fp8-gbs512.log`](pretrain-b200-qwen3-30b-fp8-gbs512.log)
- [`pretrain-b200-qwen3-30b-fp8mx-gbs512.log`](pretrain-b200-qwen3-30b-fp8mx-gbs512.log)
