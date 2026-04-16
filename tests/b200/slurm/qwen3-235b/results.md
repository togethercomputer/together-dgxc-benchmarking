# Qwen3 235B (22B active MoE, 128 experts) — B200 Benchmark Results

**Cluster:** Together AI B200 (8x B200/node, ConnectX-7 IB 400Gb/s)
**Container:** nvidia+nemo+26.02.00
**Base config:** TP=1, PP=8, EP=8, ETP=1, MBS=1, 256 GPUs (32 nodes)
**Date:** 2026-04-09 to 2026-04-11

## BF16 Results

| Variant | VP | GBS | Job ID | TFLOP/s/GPU | Step Time (s) | vs Baseline | Status |
|---------|-----|-----|--------|-------------|---------------|-------------|--------|
| **Baseline (optimized)** | **4** | **8192** | **80077** | **514** | **37.8** | — | **OK** |
| PP=4 | — | 8192 | 80014 | ~509 | 55.2 | -1.3% | marginal |
| NVLS | — | 8192 | 80015 | ~507 | 56.2 | +0.5% worse | no help |
| PP=4+NVLS | — | 8192 | 80016 | ~510 | 55.7 | -0.4% | no help |
| PP=4+EP=4 | — | 8192 | 80017 | — | — | — | FAILED (OOM) |
| SHARP | — | 8192 | 80018 | ~506 | 56.5 | — | fallback to Ring (no SHARP) |

**Tranche-1 Target:** 557 TFLOP/s/GPU | **Gap:** -8%

## FP8 MX Results

| Variant | VP | Job ID | TFLOP/s/GPU | Step Time (s) | vs FP8 Baseline | Status |
|---------|-----|--------|-------------|---------------|----------------|--------|
| FP8 baseline | None | 80078 | 383 | 50.7 | — | OK |
| FP8 + VP=4 | 4 | 80079 | 387 | 50.2 | +1% | negligible |
| **FP8 + CUDA Graphs** | **None** | **80084** | **426** | **45.6** | **+10%** | **OK (best FP8)** |
| FP8 + CG + HybridEP | None | 80085 | 426 | 45.5 | +10% | HybridEP adds nothing |
| FP8 + CG + A2A | 4 | 80088 | 300 | 64.8 | -42% | **regression** |
| FP8 + CG + A2A + ParamOverlap | 4 | 80089 | 300 | 64.8 | -42% | **regression** |

**Tranche-1 Target:** 436 TFLOP/s/GPU | **Gap:** -3%

## Key Findings

- **BF16 baseline at ~43% MFU is the ceiling** for 22B-active MoE with 128 experts
- PP=8 is entirely intra-node (NVLink) — reducing PP gives no benefit unlike 405B
- NVLS doesn't help: dominant comm is SendRecv (point-to-point), not AllReduce
- CUDA Graphs is the only effective FP8 optimization (+10%)
- A2A overlap + delay_wgrad_compute causes massive -42% regression
- Remaining 17% FP8-to-BF16 gap is hardware-intrinsic (MXFP8 GEMM + quantization overhead)

## Profiler Analysis

- BF16 profiler: `~/johnson/worklog/profiler_reports/qwen3_235b_bf16_256gpu_profiler.md`
- FP8 profiler: `~/johnson/worklog/profiler_reports/qwen3_235b_fp8mx_256gpu_profiler.md`
- Nsys traces: `/mnt/vast/llmb_/workloads/pretrain_qwen3/profiler_traces/`

## Log Locations

- `/home/johnson/johnson/scripts/qwen3_235b/*/logs/`
- `/mnt/vast/johnson/scripts/qwen3_235b_fp8mx_*/log_*.out`
- `/mnt/vast/johnson/llmb/workloads/pretrain_qwen3/experiments/`
