# Nemotron4 15B FP8 — 256 GPU Profiler Report

**Date:** 2026-04-13
**Job ID:** 81059 (profiler), 81058 (CUDA graph baseline), 81057 (no CUDA graph baseline)
**Cluster:** Together AI B200, 32 nodes x 8 GPUs
**Container:** nvidia+nemo+26.02.00

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Nemotron4 15B (32 layers, hidden=6144) |
| Precision | FP8 hybrid tensorwise |
| GPUs | 256 (32 nodes x 8 B200) |
| TP / PP / CP / VP | 1 / 1 / 1 / 1 |
| MBS / GBS | 2 / 512 |
| Seq Length | 4096 |
| Optimizer | Distributed Adam |
| CUDA Graphs | `cuda_graph_impl=transformer_engine` |
| Checkpointing | Disabled |

## Baseline Performance

| Metric | Value |
|--------|-------|
| TFLOPS/GPU (median, steps 3-49) | ~1,600 |
| Step time | ~0.456s |
| Peak memory reserved | 77.3 GB |
| Peak memory allocated | 77.0 GB |

## Profiler Results (steps 3-5, rank 0)

**Self CUDA time total: 2.322s across 3 steps (~774ms/step)**

### CUDA Time Breakdown

| Kernel / Op | Self CUDA Time | % of Total | Category |
|-------------|---------------|------------|----------|
| NCCL ReduceScatter (bf16 RING_LL) | 749.7ms | 32.3% | Communication |
| NCCL AllGather (RING_LL) | 480.0ms | 20.7% | Communication |
| _LayerNormLinearBackward | 271.2ms | 11.7% | Compute (BWD) |
| _LinearBackward | 210.6ms | 9.1% | Compute (BWD) |
| nvjet GEMM sm100 (NTT, dgrad/fprop) | 201.7ms | 8.7% | Compute (FWD) |
| _Linear (fprop) | 134.6ms | 5.8% | Compute (FWD) |
| _LayerNormLinear (fprop) | 125.7ms | 5.4% | Compute (FWD) |
| nvjet GEMM sm100 (TNT, wgrad) | 124.1ms | 5.4% | Compute (BWD) |
| nvjet GEMM sm100 (NNT) | 123.5ms | 5.3% | Compute (BWD) |
| FusedAttnFuncBackward | 116.8ms | 5.0% | Compute (BWD) |
| cudnn flash attn bprop | 101.1ms | 4.4% | Compute (BWD) |
| nvjet GEMM sm100 (TNT, fprop) | 93.1ms | 4.0% | Compute (FWD) |
| nvjet GEMM sm100 (NNT, fprop) | 90.0ms | 3.9% | Compute (FWD) |
| FusedAttnFunc (fprop) | 36.0ms | 1.5% | Compute (FWD) |
| cudnn flash attn fprop | 35.7ms | 1.5% | Compute (FWD) |
| aten::copy_ | 33.6ms | 1.4% | Memory |
| aten::mul | 32.7ms | 1.4% | Compute (Optimizer) |
| TE amax reduce | 30.6ms | 1.3% | FP8 Scaling |
| aten::fill_ | 25.3ms | 1.1% | Memory |
| LN backward | 23.8ms | 1.0% | Compute (BWD) |
| FP8 quantize_1D | 21.5ms | 0.9% | FP8 Quantization |

### Category Summary

| Category | CUDA Time | % of Total |
|----------|-----------|------------|
| **NCCL Communication** | **1,229.7ms** | **53.0%** |
| Compute — Backward | ~847ms | ~36.5% |
| Compute — Forward | ~640ms | ~27.6% |
| FP8 Overhead (quantize + amax) | ~52ms | ~2.2% |
| Memory Ops (copy, fill, zero) | ~59ms | ~2.5% |

> Note: Categories overlap because forward and backward run concurrently with communication when overlap is enabled. The profiler reports wall-clock CUDA time per kernel, and the GPU is not idle during NCCL — the ReduceScatter/AllGather overlap partially with backward compute. The 53% figure represents the amount of CUDA stream time consumed by NCCL, not necessarily 53% of wall time.

### CPU Side

| Op | CPU Time | Notes |
|----|----------|-------|
| cudaStreamSynchronize | 632.6ms | Waiting for GPU |
| aten::item / is_nonzero | 496.3ms | Scalar sync for loss/grad check |
| aten::copy_ (to/from FP8) | 151.3ms | FP8 tensor conversions |
| CrossEntropyFunctionBackward | 139.5ms | Loss backward |
| cudaLaunchKernel | 40.4ms | 7,128 kernel launches |

## Key Finding

**The job is communication-bound: 53% of CUDA time is NCCL collectives.**

With TP=1 and PP=1, all 256 GPUs participate in pure data parallelism. Every training step requires:
- **ReduceScatter** (749.7ms / 3 steps = 250ms/step): gradient reduction across 256 GPUs
- **AllGather** (480.0ms / 3 steps = 160ms/step): distributed optimizer parameter gather

These use the NCCL RING_LL algorithm over InfiniBand, where latency scales with ring size (256 GPUs).

The actual compute per step is efficient — FP8 GEMMs on SM100 (B200 Blackwell) achieve good utilization. The FP8 overhead (quantization + amax scaling) is only ~2.2% of total CUDA time.

## Comparison: 64 GPU vs 256 GPU

| | 64 GPUs (8 nodes) | 256 GPUs (32 nodes) |
|---|---|---|
| TFLOPS/GPU | ~1,895 | ~1,600 |
| Step time | ~0.77s | ~0.46s |
| GBS | 256 | 512 |
| Microbatches/DP rank | 2 | 1 |
| NCCL ring size | 64 | 256 |
| Scaling efficiency | baseline | 84% |

The 16% per-GPU efficiency drop at 256 GPUs comes from:
1. 4x larger NCCL ring (256 vs 64 GPUs) — more hops, higher latency
2. Half the microbatches per DP rank (1 vs 2) — less compute to overlap with communication

## Optimization Plan

See next section for concrete experiments to improve from the current ~1,600 TFLOPS/GPU baseline.

---

## Optimization Experiments

### Experiment 1: Increase GBS to 1024 (more compute per step)

**Rationale:** With GBS=512 and 256 GPUs, each GPU processes only 1 microbatch (MBS=2) per step — there's no gradient accumulation, so no compute happens during the allreduce. With GBS=1024, each GPU processes 2 microbatches, and the second microbatch's forward/backward overlaps with the first microbatch's gradient allreduce.

**Expected impact:** 10-20% improvement. The 64-GPU run achieved ~1,895 TFLOPS/GPU with 2 microbatches/rank.

**Changes:**
- `OVERRIDE_GBS=1024`
- `OVERRIDE_NUM_TRAIN_SAMPLES=51200`

### Experiment 2: TP=2 (reduce DP world size)

**Rationale:** TP=2 uses NVLink for intra-node tensor-parallel allreduce (fast) while cutting the DP world size from 256 to 128. This halves the NCCL ReduceScatter/AllGather data volume and ring size. The TP communication happens over NVLink (900 GB/s per GPU on B200), which is much faster than IB.

**Expected impact:** 15-25% improvement. The TP allreduce cost is small on NVLink, and halving the DP ring size significantly reduces the dominant NCCL bottleneck.

**Changes:**
- Config: `tensor_model_parallel_size: 2`, `sequence_parallel: true`
- GBS=512 (128 DP ranks * 2 MBS = 256 samples/step, need GBS >= 256)
- Need to override via env var or compat_runner patch

### Experiment 3: TP=2 + GBS=1024 (combined)

**Rationale:** Combines the benefits of both: smaller DP ring (128 GPUs) plus 2 microbatches per DP rank for compute/comm overlap.

**Expected impact:** 20-30% improvement over baseline, potentially reaching ~1,900-2,000 TFLOPS/GPU.

**Changes:**
- TP=2, sequence_parallel=true
- GBS=1024
- Each DP rank: 1024 / (128 * 2) = 4 microbatches — excellent overlap

### Experiment 4: Enable NCCL NVLS

**Rationale:** `NCCL_NVLS_ENABLE=0` is currently set. NVLink-SHARP (NVLS) can accelerate intra-node reduction by using NVLink multicast. On B200 with NVLink, this can reduce the intra-node allreduce component.

**Expected impact:** 2-5% improvement (only affects intra-node portion of the collective).

**Changes:**
- `NCCL_NVLS_ENABLE=1`
- Can be combined with any other experiment

### Experiment 5: TP=4 (aggressive TP)

**Rationale:** TP=4 reduces DP world size to 64 (same as the 64-GPU baseline ring size). The 64-GPU run achieved ~1,895 TFLOPS/GPU. TP=4 on B200 NVLink should be fast enough to recover most of that.

**Expected impact:** Could approach ~1,800-1,900 TFLOPS/GPU, but TP=4 adds more TP allreduce overhead. Diminishing returns vs TP=2.

**Changes:**
- Config: `tensor_model_parallel_size: 4`, `sequence_parallel: true`
- GBS=512 (64 DP ranks * 2 MBS = 128, need GBS >= 128)

### Priority Order

1. **Experiment 1 (GBS=1024)** — Simplest change, just env vars, no config changes
2. **Experiment 4 (NVLS)** — Also just an env var, can run in parallel with Exp 1
3. **Experiment 2 (TP=2)** — Requires config change but highest expected gain
4. **Experiment 3 (TP=2 + GBS=1024)** — Best combined result
5. **Experiment 5 (TP=4)** — Only if TP=2 results warrant further exploration

### Implementation Notes

- Experiments 1 and 4 only require sbatch/env changes — can reuse existing fn_or_script
- Experiments 2, 3, 5 require overriding `tensor_model_parallel_size` and `sequence_parallel` in the serialized config. This needs additional monkey-patches in compat_runner.py to intercept MegatronStrategy init.
- All experiments should run 50 steps with the same profiler callback for apples-to-apples comparison.
- Focus on steady-state TFLOPS/GPU from steps 3-49 (skip warmup steps 0-2).

---

## Experiment Results (2026-04-13)

### Summary Table

| Experiment | Config | Job ID | TFLOPS/GPU (median) | Step Time (s) | vs Baseline | Notes |
|---|---|---|---|---|---|---|
| Baseline | TP=1, GBS=512 | 81057 | ~1,600 | ~0.456 | — | Communication-bound (53% NCCL) |
| Exp 1: GBS=1024 | TP=1, GBS=1024 | 81060 | 1,785 | ~0.82 | +11.6% | 2 microbatches/rank, compute/comm overlap |
| Exp 4: NVLS | TP=1, NVLS=1 | 81061 | 1,532 | ~0.48 | -4.3% | NVLS hurts at TP=1/DP=256 |
| Exp 2: TP=2 | TP=2, GBS=512 | 81066 | 1,296 | ~0.56 | -19.0% | tp_comm_overlap disabled (TE UB segfault) |
| Exp 3: TP=2+GBS=1024 | TP=2, GBS=1024 | 81067 | 1,371 | ~1.06 | -14.3% | Same issue, extra microbatches help partially |
| Exp 6: GBS=2048 | TP=1, GBS=2048 | 81069 | 1,885 | ~1.54 | +17.8% | 4 microbatches/rank |
| **Exp 6b: GBS=2048 (profiler)** | TP=1, GBS=2048 | 81070 | **1,910** | ~1.53 | **+19.4%** | Contiguous nodes 159-190 |
| **Exp 7: GBS=2560** | TP=1, GBS=2560 | 81071 | **1,915** | ~1.91 | **+19.7%** | 5 microbatches/rank, best result |

### Analysis

**GBS=2048 and GBS=2560 both exceed the 1,908 TFLOPS/GPU target.**

The progression from GBS=512 to GBS=2560 shows how gradient accumulation enables compute/communication overlap at 256 GPUs:

- **GBS=512 (1 microbatch/rank):** NCCL allreduce is fully exposed — 53% of CUDA time is communication. Result: ~1,600 TFLOPS/GPU.
- **GBS=1024 (2 microbatches/rank):** Second microbatch's fwd/bwd overlaps with first microbatch's allreduce. +11.6% improvement.
- **GBS=2048 (4 microbatches/rank):** Nearly full overlap — the allreduce from microbatch N is hidden behind compute of microbatch N+1. Median 1,885-1,910 depending on node placement.
- **GBS=2560 (5 microbatches/rank):** Marginal improvement over GBS=2048. Median 1,915, diminishing returns — the allreduce is already well-hidden at 4 microbatches.

**Node placement matters:** The same GBS=2048 config achieved 1,885 (scattered nodes) vs 1,910 (contiguous nodes 159-190). This ~25 TFLOPS/GPU (~1.3%) difference comes from NCCL topology — contiguous nodes have shorter IB paths.

**TP=2 experiments underperformed** because:
1. The serialized NeMo config includes a `MegatronCommOverlapCallback` that enables `tp_comm_overlap` for TransformerEngine user buffers
2. When TP is changed dynamically via monkey-patch, TE's `initialize_ub()` segfaults in `_new_process_group_helper` — the UB code assumes TP groups are set up in a specific order matching the serialized config
3. We had to disable `tp_comm_overlap` to avoid the crash, which serializes the TP allreduce with compute instead of overlapping them
4. Without comm overlap, TP=2 adds ~100ms/step of TP allreduce latency that isn't hidden

**NVLS hurt performance** because at TP=1 with pure DP, all communication is inter-node RING_LL over InfiniBand. NVLS (NVLink-SHARP) only benefits intra-node reductions and likely interfered with NCCL's algorithm selection.

### Experiment 5 (TP=4): Skipped

Given that TP=2 performed worse even with GBS=1024, TP=4 would be even worse. Skipped.

---

## Profiler Deep-Dive: GBS=2048 (Job 81070)

### CUDA Time Breakdown (3 profiled steps, 5.561s total)

| Kernel / Op | Self CUDA | % of Total | Category |
|---|---|---|---|
| NCCL ReduceScatter (bf16 RING_LL) | 874.8ms | 15.7% | DP Communication |
| NCCL AllGather (RING_LL) | 467.3ms | 8.4% | DP Communication |
| **Total NCCL** | **1,342.1ms** | **24.1%** | **Communication** |
| nvjet_sm100 GEMM (NTT, fprop+dgrad) | 599.6ms | 10.8% | GEMM |
| nvjet_sm100 GEMM (NNT, wgrad) | 498.1ms+358.8ms | 15.4% | GEMM |
| nvjet_sm100 GEMM (TNT, bwd) | 490.0ms+348.2ms | 15.1% | GEMM |
| nvjet_sm100 GEMM (NTT, smaller) | 194.4ms | 3.5% | GEMM |
| **Total GEMMs** | **~2,489ms** | **~44.8%** | **Compute** |
| cudnn flash attention bprop | 406.5ms | 7.3% | Attention |
| cudnn flash attention fprop | 146.3ms | 2.6% | Attention |
| **Total Attention** | **552.8ms** | **9.9%** | **Compute** |
| Command Buffer Full (GPU stalls) | 254.8ms | 4.6% | Overhead |
| aten::copy_ (FP8 casting) | 122.6ms | 2.2% | FP8 Overhead |
| TE amax reduce | 117.6ms | 2.1% | FP8 Overhead |
| TE FP8 quantize_1D | 82.4ms | 1.5% | FP8 Overhead |
| **Total FP8 Overhead** | **322.6ms** | **5.8%** | **FP8** |
| LN backward tuned | 99.3ms | 1.8% | Compute |
| aten::mul (optimizer) | 127.7ms | 2.3% | Optimizer |
| aten::pow (optimizer) | 108.2ms | 1.9% | Optimizer |

### Category Summary

| Category | CUDA Time | % of Total | Notes |
|---|---|---|---|
| GEMMs (Linear fwd+bwd) | ~2,489ms | 44.8% | Dominant, well-optimized SM100 nvjet kernels |
| NCCL Communication | 1,342ms | 24.1% | ReduceScatter 15.7% + AllGather 8.4% |
| Flash Attention (fwd+bwd) | 553ms | 9.9% | cuDNN flash attention on SM100 |
| FP8 Overhead | 323ms | 5.8% | amax+quantize+copy for FP8 scaling |
| GPU Command Buffer Stalls | 255ms | 4.6% | GPU command queue saturation |
| Optimizer (mul+pow) | 236ms | 4.2% | Adam update kernels |
| LayerNorm backward | 99ms | 1.8% | TE fused LayerNorm |
| Other | ~264ms | 4.8% | Misc elementwise, ReLU backward |

### Key Observations

1. **Communication is well-overlapped.** With 4 microbatches/rank, NCCL time (24.1% of CUDA time) is largely hidden behind compute. Step time is ~1.53s vs the naive sequential estimate of ~1.85s.

2. **GEMM kernels are the dominant cost (44.8%).** The SM100 nvjet kernels are running FP8 GEMMs efficiently. No obvious room for improvement here — these are hardware-limited.

3. **FP8 overhead is modest (5.8%).** The amax reduction (2.1%), FP8 quantization (1.5%), and type casting (2.2%) are inherent to FP8 training. This is a good tradeoff for the ~2x throughput gain over BF16.

4. **Command Buffer Full stalls (4.6%)** occur 444 times in 3 steps (~148/step). This suggests the GPU command buffer is being saturated by high kernel launch rate. Potential mitigation: CUDA graphs (already enabled) handle this for the main loop, but optimizer steps may not be graphed.

5. **CPU sync overhead.** `cudaStreamSynchronize` takes 1.722s CPU time (25.5%), and `aten::item` / `aten::is_nonzero` take 1.133s. These are loss-checking synchronization points in NeMo/PyTorch Lightning — not addressable without modifying framework code.

---

## Final Results

### Best 256-GPU Configuration

| Parameter | Value |
|-----------|-------|
| TP / PP / CP / VP | 1 / 1 / 1 / 1 |
| GBS | 2048 or 2560 |
| MBS | 2 |
| Microbatches per DP rank | 4 (GBS=2048) or 5 (GBS=2560) |
| NCCL NVLS | Disabled |
| CUDA Graphs | `cuda_graph_impl=transformer_engine` |
| **TFLOPS/GPU (median)** | **1,910-1,915** |
| **Target** | **1,908** |
| **Scaling efficiency (vs 64 GPU)** | **100.8-101.1%** (1,910-1,915 / 1,895) |

> Note: Scaling efficiency >100% is possible because more gradient accumulation steps at 256 GPUs amortize fixed overheads (optimizer step, CUDA graph capture/replay boundaries) over more compute.

### Scaling Summary

| GPUs | GBS | Microbatches/rank | TFLOPS/GPU (median) | Scaling Efficiency |
|------|-----|-------------------|--------------------|--------------------|
| 64 | 256 | 2 | ~1,895 | 100% (baseline) |
| 256 | 512 | 1 | ~1,600 | 84.4% |
| 256 | 1024 | 2 | 1,785 | 94.2% |
| 256 | 2048 | 4 | 1,885-1,910 | 99.5-100.8% |
| 256 | 2560 | 5 | 1,915 | 101.1% |

### Recommendation for Benchmark Submission

Use **GBS=2048** with contiguous node allocation for the benchmark report:
- Median: **1,910 TFLOPS/GPU** (exceeds 1,908 target)
- GBS=2048 is a reasonable batch size for a 15B model
- GBS=2560 offers only marginal improvement (+5 TFLOPS/GPU) and is a less standard batch size
- Request contiguous nodes (e.g., `--nodelist=use3a-ss-b200-gpu-[159-190]`) for optimal NCCL topology
