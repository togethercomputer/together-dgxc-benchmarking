# Nemotron4 15B BF16 Benchmark Report

**Date:** 2026-04-13
**Cluster:** Together AI B200 (Slurm)
**Container:** nvidia+nemo+26.02.00

## Summary

Both 64-GPU and 256-GPU BF16 benchmarks completed successfully, exceeding the Best Tranche-1 target of **1,264 TFLOPS/GPU** by a significant margin.

| Scale | TFLOPS/GPU (avg) | vs Target (1,264) |
|-------|-----------------|-------------------|
| 64 GPUs | **1,570** | **+24.2%** |
| 256 GPUs | **1,439** | **+13.8%** |

## Configuration

| Parameter | 64 GPUs | 256 GPUs |
|-----------|---------|----------|
| Nodes | 8 | 32 |
| GPUs per node | 8 | 8 |
| TP / PP / CP / VP | 1 / 1 / 1 / 1 | 1 / 1 / 1 / 1 |
| MBS | 2 | 2 |
| GBS | 256 | 1024 |
| Seq Length | 4096 | 4096 |
| Max Steps | 50 | 50 |
| Precision | bf16-mixed | bf16-mixed |
| CUDA Graphs | Enabled | Enabled |
| Optimizer | Distributed Adam | Distributed Adam |

## Results (Steady-State, Steps 3–49)

### 64 GPUs (8 nodes) — Job 81054

| Metric | Value |
|--------|-------|
| Avg TFLOPS/GPU | **1,570.4** |
| Min / Max TFLOPS/GPU | 1,566 / 1,572 |
| Avg step time | 0.9302s |
| Min / Max step time | 0.9294s / 0.9327s |
| Peak memory reserved | 118.9 GB |
| Peak memory allocated | 92.5 GB |

### 256 GPUs (32 nodes) — Job 81056

| Metric | Value |
|--------|-------|
| Avg TFLOPS/GPU | **1,439.4** |
| Min / Max TFLOPS/GPU | 1,437 / 1,441 |
| Avg step time | 1.0149s |
| Min / Max step time | 1.0140s / 1.0160s |
| Peak memory reserved | 117.2 GB |
| Peak memory allocated | 92.5 GB |

### Scaling Efficiency

| Metric | Value |
|--------|-------|
| 64 → 256 GPU scaling efficiency | **91.7%** |
| Per-GPU TFLOPS drop | -131 (8.3%) |
| Step time increase | +0.085s (9.1%) |

The 8.3% per-GPU throughput drop at 4x scale is expected from increased inter-node allreduce communication over InfiniBand.

## Warmup Steps (0–2)

| Step | 64 GPUs | 256 GPUs |
|------|---------|----------|
| 0 | 104.2 TFLOPS (14.02s) | 99.5 TFLOPS (14.68s) |
| 1 | 1,539 TFLOPS (0.95s) | 1,427 TFLOPS (1.02s) |
| 2 | 1,564 TFLOPS (0.93s) | 1,439 TFLOPS (1.02s) |

Both runs stabilize by step 2–3. Step 0 includes JIT compilation and CUDA graph capture overhead.

## Key Files

- **64-GPU sbatch:** `~/johnson/scripts/nemotron4_15b/64gpus_bf16/sbatch.sh`
- **64-GPU log:** `.../log-nemotron4_15b_bf16_64gpu_81054_0.out`
- **256-GPU sbatch:** `~/johnson/scripts/nemotron4_15b/256gpus_bf16/sbatch.sh`
- **256-GPU log:** `.../log-nemotron4_15b_bf16_256gpu_81056_0.out`
- **Compat wrapper:** `compat_runner.py` (tensorstore stub, cuda_graph compat, optimizer kwarg fix)
