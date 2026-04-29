# Nsys Profiling Bottleneck Report — Llama 3.1 405B NVFP4

**Date:** 2026-04-10
**Job:** 79846 (32 nodes, 256 GPUs)
**Config:** TP=4, PP=16, CP=1, VP=8, MBS=1, GBS=1536, recompute_num_layers=8
**Container:** nvidia+nemo+26.02.00 (nsys 2025.5.1)
**Profiled Steps:** 3–6 (3 active nsys steps)
**Profiled Ranks:** 0 (PP stage 0), 32 (PP stage 8), 60 (PP stage 15)

## Trace Files

| File | Size | Rank | PP Stage |
|------|------|------|----------|
| `profile_79846_node0_gpu0.nsys-rep` | 578 MB | 0 | 0 (first) |
| `profile_79846_node4_gpu0.nsys-rep` | 614 MB | 32 | 8 (middle) |
| `profile_79846_node7_gpu4.nsys-rep` | 617 MB | 60 | 15 (last) |

Location: `/mnt/vast/llmb_/workloads/pretrain_llama3.1/profiler_traces/405b_nvfp4_256gpu_nsys/`

## Step Timing (all 10 steps)

| Step | Time (s) | TFLOP/s/GPU | Notes |
|------|----------|-------------|-------|
| 1 | 543.3 | 228.4 | warmup |
| 2 | 88.6 | 1400.5 | baseline |
| 3 | 88.3 | 1405.2 | nsys cudaProfilerStart |
| 4 | 119.0 | 1042.4 | nsys recording (first active step overhead) |
| 5 | 89.4 | 1388.3 | nsys recording |
| 6 | 89.7 | 1383.0 | nsys cudaProfilerStop |
| 7 | 252.0 | 492.0 | nsys .nsys-rep file write |
| 8 | 88.8 | 1397.0 | post-profiling |
| 9 | 87.8 | 1413.0 | post-profiling |
| 10 | 88.5 | 1402.0 | post-profiling |

**Steady-state throughput:** ~1400 TFLOP/s/GPU @ 88–89s/step

---

## Executive Summary

The workload is **heavily communication-bound**. Approximately 67% of GPU time is spent on NCCL communication, with only ~20% on GEMM compute and ~5% on attention. The dominant bottleneck is **pipeline parallelism send/recv**, consuming 38–43% of GPU time across all ranks.

---

## 1. GPU Kernel Time Breakdown

### Rank 0 — PP Stage 0 (first)

```
 Time (%)  Total Time (s)  Instances  Avg (ms)  Max (ms)   Kernel
 --------  --------------  ---------  --------  --------   ------
    42.9%         153.3       34128     4.49     155.1     ncclDevKernel_SendRecv
    19.3%          69.1      127873     0.54       1.4     cutlass3x GEMM (NVFP4 block-scaled)
    14.5%          51.8       47896     1.08      38.1     ncclAllReduce_Sum_f32_RING_LL
     5.6%          20.0       49041     0.41      13.8     ncclReduceScatter_Sum_bf16_RING_LL
     4.2%          15.0       65105     0.23      25.3     ncclAllGather_RING_LL
     2.8%           9.9        8020     1.24       1.3     flash_attn_bwd (cuDNN SM100)
     1.8%           6.3       15928     0.40       0.4     flash_attn_fwd (cuDNN SM100)
     1.0%           3.6       31856     0.11       0.1     rmsnorm_fwd
     1.0%           3.5       64158     0.05       0.4     rht_gemm_device
     0.8%           2.7        8020     0.34       0.3     triton fused_add_cat_mul_rsub_sigmoid_silu_split
     0.7%           2.6       95793     0.03       0.1     quantize_transpose_nvfp4
     0.7%           2.4       16040     0.15       0.2     rmsnorm_bwd
     0.7%           2.3      103813     0.02       0.1     HadamardAmaxTmaKernel
     0.6%           2.3      255746     0.01       0.0     swizzle_row_scaling
     0.5%           1.6       15928     0.10       0.1     triton fused_mul_silu_split
```

### Rank 32 — PP Stage 8 (middle)

```
 Time (%)  Total Time (s)  Instances  Avg (ms)  Max (ms)   Kernel
 --------  --------------  ---------  --------  --------   ------
    41.1%         167.0       36332     4.60     200.4     ncclDevKernel_SendRecv
    19.6%          79.6      145887     0.55       1.4     cutlass3x GEMM (NVFP4 block-scaled)
    17.8%          72.3       54637     1.32      39.3     ncclAllReduce_Sum_f32_RING_LL
     5.5%          22.5       54662     0.41      13.8     ncclReduceScatter_Sum_bf16_RING_LL
     3.2%          12.9       72960     0.18      12.9     ncclAllGather_RING_LL
     2.8%          11.5        9153     1.25       1.3     flash_attn_bwd
     1.8%           7.2       18166     0.40       0.4     flash_attn_fwd
     1.0%           4.2       73220     0.06       0.3     rht_gemm_device
     1.0%           4.1       36332     0.11       0.1     rmsnorm_fwd
     0.8%           3.2        9153     0.35       0.4     triton fused_add_cat_mul_rsub_sigmoid_silu_split
     0.7%           3.0      109275     0.03       0.1     quantize_transpose_nvfp4
     0.7%           2.8       18306     0.15       0.2     rmsnorm_bwd
     0.7%           2.7      118428     0.02       0.1     HadamardAmaxTmaKernel
```

### Rank 60 — PP Stage 15 (last)

```
 Time (%)  Total Time (s)  Instances  Avg (ms)   Max (ms)   Kernel
 --------  --------------  ---------  --------   --------   ------
    38.6%         157.2       36584     4.30      161.2     ncclDevKernel_SendRecv
    19.5%          79.4      146880     0.54        1.4     cutlass3x GEMM (NVFP4 block-scaled)
    16.8%          68.6       55012     1.25       52.5     ncclAllReduce_Sum_f32_RING_LL
     9.0%          36.5       55036     0.66     4374.9     ncclReduceScatter_Sum_bf16_RING_LL  ← outlier
     3.2%          12.9       73457     0.18       50.6     ncclAllGather_RING_LL
     2.8%          11.4        9214     1.23        1.3     flash_attn_bwd
     1.8%           7.3       18292     0.40        0.5     flash_attn_fwd
     1.0%           4.3       36584     0.12        0.1     rmsnorm_fwd
     1.0%           4.2       73712     0.06        0.4     rht_gemm_device
     0.8%           3.1        9214     0.34        0.4     triton fused_add_cat_mul_rsub_sigmoid_silu_split
     0.7%           3.0      110024     0.03        0.1     quantize_transpose_nvfp4
     0.7%           2.9       18428     0.16        0.2     rmsnorm_bwd
     0.7%           2.6      119238     0.02        0.1     HadamardAmaxTmaKernel
```

---

## 2. Communication vs Compute Split

| Category | Rank 0 (PP0) | Rank 32 (PP8) | Rank 60 (PP15) |
|---|---|---|---|
| **PP Send/Recv** | 42.9% (153s) | 41.1% (167s) | 38.6% (157s) |
| **AllReduce f32** (grad sync) | 14.5% (52s) | 17.8% (72s) | 16.8% (69s) |
| **ReduceScatter bf16** | 5.6% (20s) | 5.5% (22s) | **9.0% (36s)** |
| **AllGather** | 4.2% (15s) | 3.2% (13s) | 3.2% (13s) |
| **Total NCCL** | **67.2%** | **67.6%** | **67.6%** |
| GEMM (cutlass NVFP4) | 19.3% (69s) | 19.6% (80s) | 19.5% (79s) |
| Flash Attention BWD | 2.8% (10s) | 2.8% (11s) | 2.8% (11s) |
| Flash Attention FWD | 1.8% (6s) | 1.8% (7s) | 1.8% (7s) |
| RMSNorm FWD+BWD | 1.7% (6s) | 1.7% (7s) | 1.7% (7s) |

**The GPU spends ~67% of its time on communication and only ~33% on compute.**

---

## 3. Bottleneck Analysis

### Bottleneck #1: Pipeline Parallelism Bubble (38–43% of GPU time)

`ncclDevKernel_SendRecv` is the single largest cost across all ranks. With PP=16 and VP=8, the pipeline bubble dominates.

**Latency distribution (SendRecv):**

| Rank | Avg (ms) | Median (ms) | Max (ms) | StdDev (ms) |
|------|----------|-------------|----------|-------------|
| 0 | 4.49 | 3.05 | 155.1 | 4.30 |
| 32 | 4.60 | 3.10 | 200.4 | 4.27 |
| 60 | 4.30 | 3.21 | 161.2 | 3.25 |

The max latency is 30–45x the average, indicating severe pipeline stalls during bubble phases. Rank 32 (middle PP stage) has the worst max latency at 200ms — the middle of the pipeline is where the bubble impact is highest in interleaved schedules.

### Bottleneck #2: Gradient AllReduce f32 (14–18% of GPU time)

The data-parallel gradient all-reduce uses **f32 precision** via RING_LL algorithm.

| Rank | Total (s) | % GPU Time | Count | Avg (ms) | Max (ms) |
|------|-----------|------------|-------|----------|----------|
| 0 | 51.8 | 14.5% | 47896 | 1.08 | 38.1 |
| 32 | 72.3 | 17.8% | 54637 | 1.32 | 39.3 |
| 60 | 68.6 | 16.8% | 55012 | 1.25 | 52.5 |

Rank 32 spends the most time in AllReduce (17.8%). The median is only ~46 μs per call, but max reaches 39–52ms — extreme tail latency suggests contention with pipeline send/recv traffic on the same network.

### Bottleneck #3: ReduceScatter Imbalance on Last PP Stage

Rank 60 (PP stage 15) has a disproportionately high ReduceScatter cost:

| Rank | % GPU Time | Total (s) | Max (ms) |
|------|------------|-----------|----------|
| 0 | 5.6% | 20.0 | 13.8 |
| 32 | 5.5% | 22.5 | 13.8 |
| **60** | **9.0%** | **36.5** | **4374.9** |

Rank 60 has a single ReduceScatter call that took **4.37 seconds** — 300x the average. This extreme outlier suggests the last PP stage experiences a scheduling conflict or network congestion during certain pipeline phases.

---

## 4. NVTX GPU Projection (Rank 0)

Top operations by projected GPU busy time:

| Operation | Proj GPU Time (s) | GPU Ops | Notes |
|---|---|---|---|
| NCCL:ncclGroupEnd | 217.0 | 145989 | PP send/recv batched |
| nvte_cublas_gemm_v2 | 69.7 | 255720 | GEMM (2 kernels per op) |
| nvte_rmsnorm_fwd | 48.6 | 63708 | Large projection due to AllReduce overlap |
| NCCL:ncclReduceScatter | 19.7 | 49012 | |
| nvte_flash_attn_bwd | 10.8 | 40095 | 5 kernels per op |
| nvte_flash_attn_fwd | 6.3 | 15927 | |
| hadamard_transform_cast | 3.5 | 64152 | NVFP4 quantization |
| NCCL:ncclAllGather | 3.5 | 1152 | |
| nvte_quantize_v2 | 2.6 | 95840 | NVFP4 quantization |
| nvte_rmsnorm_bwd | 2.6 | 48114 | |

---

## 5. Memory Operations

Memory operations are **negligible** — only 351ms total on rank 0:

| Operation | Total (ms) | Count |
|---|---|---|
| H2D memcpy | 131.0 | 18067 |
| D2D memcpy | 123.0 | 3490 |
| memset | 96.7 | 59416 |
| D2H memcpy | 0.08 | 21 |

---

## 6. Compute Kernel Efficiency

The GEMM kernel (`cutlass3x_sm100_bstensorop_s256x256x64gemm_block_scaled`) is well optimized for B200 SM100:
- Uses NVFP4 block-scaled tensor ops (ue4m3xf4)
- Tile size 256x256x256
- Average 540 μs per GEMM call, low variance (StdDev ~400 μs)
- This kernel achieves good utilization *when it runs* — the issue is it only runs 20% of the time

Flash attention uses cuDNN SM100-optimized kernels:
- Forward: 397 μs avg, 128x128x128 tiles
- Backward: 1.24 ms avg (3.1x forward) — expected ratio for attention backward

---

## 7. Recommendations

### High Impact

1. **Reduce PP depth (PP=16 → PP=8):** The pipeline bubble (40% of GPU time) is the #1 bottleneck. With PP=8:
   - Bubble fraction decreases significantly
   - SendRecv volume halves (8 stages instead of 16)
   - Trade-off: requires TP=8 to maintain the same model split, doubling TP allreduce cost, but TP allreduce is intra-node (NVLink) and much cheaper than inter-node PP send/recv

2. **AllReduce precision (f32 → bf16):** Gradient AllReduce uses f32 (14–18% of GPU time). If loss scaling and training stability allow, bf16 AllReduce would halve the bandwidth requirement and could cut this cost by ~40–50%.

### Medium Impact

3. **Overlap investigation:** With `CUDA_DEVICE_MAX_CONNECTIONS=32`, there should be opportunity for compute-communication overlap. The large gap between median (3ms) and max (200ms) SendRecv latency suggests communication is not fully overlapped during pipeline fill/drain phases. Consider:
   - Increasing NCCL buffer sizes
   - Verifying interleaved 1F1B schedule is active with VP=8

4. **ReduceScatter outlier on rank 60:** The 4.37s ReduceScatter on the last PP stage needs investigation. Check if this correlates with a pipeline drain phase where all stages flush simultaneously.

### Lower Impact

5. **Context Parallelism (CP=1 → CP=2):** Would allow halving PP to PP=8 while keeping TP=4. Requires sequence length > 4096 to be beneficial.

6. **Recompute layers:** Currently at 8 recompute layers. Increasing this trades compute for memory but doesn't address the communication bottleneck.

---

## Appendix: Commands Used

```bash
# Profiling job submission
sbatch sbatch_nsys.sh  # Job 79846

# Nsys capture config (in train_nsys.sh)
nsys profile -s none -t nvtx,cuda \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o <output_path> --force-overwrite true

# Profiled ranks: 0, 32, 60 (conditional wrapping in train_nsys.sh)
# profiling.profile_step_start=3, profiling.profile_step_end=6

# Analysis (requires container nsys 2025.5.1)
srun --container-image nvidia+nemo+26.02.00.sqsh \
  nsys stats -r cuda_gpu_kern_sum:base -f column <trace.nsys-rep>
srun --container-image nvidia+nemo+26.02.00.sqsh \
  nsys stats -r nvtx_pushpop_sum -f column <trace.nsys-rep>
srun --container-image nvidia+nemo+26.02.00.sqsh \
  nsys stats -r nvtx_gpu_proj_sum -f column <trace.nsys-rep>
srun --container-image nvidia+nemo+26.02.00.sqsh \
  nsys stats -r cuda_gpu_mem_time_sum -f column <trace.nsys-rep>
```
