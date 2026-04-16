# Qwen3 235B BF16 256-GPU Profiler Bottleneck Report

**Model:** Qwen3 235B A22B (MoE, 22B active params)
**Precision:** BF16
**Scale:** 256 GPUs / 32 nodes (B200)
**Parallelism:** TP=1, PP=8, CP=1, EP=8, VP=4, MBS=1, GBS=8192, DP=32
**Topology:** PP intra-node (NVLink), EP and DP inter-node (IB)
**Profiler:** PyTorch profiler (CPU + CUDA), rank 0 (PP stage 0), step 3-4
**Baseline step time:** 57.5s (337.7 TFLOP/s/GPU, iteration 2 steady state)
**Trace file:** `qwen3_235b_bf16_baseline_cuda_rank0.pt.trace.json.gz` (738MB, job 79973)
**Date:** 2026-04-10

---

## 1. High-Level Time Breakdown

| Metric                   | Value   |
|--------------------------|---------|
| GPU Compute Kernels      | 5.29s   |
| GPU Total Span           | 31.07s  |
| GPU Idle (gaps >1ms)     | 16.90s  |
| **GPU Bubble Fraction**  | **54.4%** |
| NCCL Comm (CPU-side)     | 28.93s  |
| CUDA Runtime             | 1.18s   |
| aten::item sync stalls   | 0.05s (negligible) |

> Note: GPU kernel data is partial — CUDA events appeared in the last ~8M lines of the 238M-line trace (gzip truncated at 738MB). Relative proportions and bubble fraction are representative of a partial step.

---

## 2. NCCL Communication Breakdown

| Type              | Total (s) | Count    | Avg (ms) | % of Comm | Role                        |
|-------------------|-----------|----------|----------|-----------|-----------------------------|
| **AllToAll (MoE)**| **22.97** | 109,824  | 0.21     | **79.4%** | Expert routing (inter-node IB) |
| Send/Recv (PP)    | 4.31      | 12,319   | 0.35     | 14.9%     | Pipeline P2P (intra-node NVLink) |
| AllGather         | 1.65      | 12,505   | 0.13     | 5.7%      | EP/DP weight gather         |
| Recv (PP)         | 0.21      | 3,584    | 0.06     | 0.7%      | Pipeline receive            |
| ReduceScatter     | ~0        | 32       | 0.12     | 0.0%      | DP gradient sync            |
| AllReduce         | ~0        | 20       | 0.08     | 0.0%      | -                           |

**Key finding:** MoE AllToAll expert routing dominates communication at **79.4%**. Pipeline P2P is secondary at 14.9%. DP gradient sync (ReduceScatter/AllReduce) is negligible.

---

## 3. Top GPU Kernels

| Rank | Kernel                                            | Total (ms) | Count  | Avg (ms) | % of GPU |
|------|---------------------------------------------------|-----------|--------|----------|----------|
| 1    | nvjet_sm100 GEMM (128x256, NTT)                  | 810.0     | 17,625 | 0.05     | 15.3%    |
| 2    | nvjet_sm100 GEMM (176x128, TNN)                  | 706.5     | 9,410  | 0.08     | 13.4%    |
| 3    | nvjet_sm100 GEMM (128x192, TNT)                  | 478.5     | 8,109  | 0.06     | 9.1%     |
| 4    | nvjet_sm100 GEMM (128x192, NTT)                  | 397.7     | 17,632 | 0.02     | 7.5%     |
| 5    | cuDNN flash attention SDPA (fprop)                | 264.8     | 1,203  | 0.22     | 5.0%     |
| 6-7  | nvjet_sm100 GEMM (128x256, various)              | 430.6     | 2,813  | 0.15     | 8.1%     |
| 8    | _sort_chunks_by_map_kernel (MoE routing)          | 198.4     | 2,454  | 0.08     | 3.8%     |
| 9    | nvjet_sm100 GEMM (128x232, TNT)                  | 191.1     | 5,738  | 0.03     | 3.6%     |
| 10   | nvjet_sm100 GEMM (256x128, TNT)                  | 175.1     | 6,082  | 0.03     | 3.3%     |

**Summary by category:**
- Grouped GEMM (MoE experts): ~70% of GPU compute
- Flash attention (cuDNN SDPA): ~5.3%
- MoE token routing/permutation: ~9% (_sort_chunks, _permute, _unpermute, _make_chunk_sort_map)
- Normalization (RMSNorm): ~1.8%
- Other (RoPE, SiLU, elementwise): ~14%

---

## 4. GPU Idle Gap Analysis (Bubble Detection)

| Metric                       | Value     |
|------------------------------|-----------|
| Total gaps >1ms              | 4,985     |
| Total idle time              | 16.90s    |
| Bubble fraction              | **54.4%** |
| Largest single gap           | 111.4ms   |

### Gap Distribution

| Bucket     | Count | Total (s) | % of idle |
|------------|-------|-----------|-----------|
| 1-10ms     | 4,982 | 16.77     | 99.2%     |
| 10-100ms   | 2     | 0.02      | 0.1%      |
| 100ms-1s   | 1     | 0.11      | 0.7%      |
| >1s        | 0     | 0.00      | 0.0%      |

### Dominant Gap Pattern

The majority of idle gaps follow the pattern:

```
_permute_kernel (MoE token dispatch) → [gap 7-10ms] → nvjet GEMM (expert compute)
```

This indicates the GPU is waiting for MoE token permutation/AllToAll dispatch to complete before expert GEMM execution can begin. The AllToAll goes over inter-node IB, introducing latency between the permutation and the grouped GEMM.

The largest gap (111.4ms) coincides with `ncclDevKernel_SendRecv` max duration (111.2ms), indicating a pipeline P2P stall.

---

## 5. MoE-Specific Operations

| Operation                    | Total (ms) | Count  | Avg (ms) |
|------------------------------|-----------|--------|----------|
| MoE layer forward            | 9,984.9   | 2,816  | 3.55     |
| MoE token permutation fwd    | 9,899.3   | 2,816  | 3.52     |
| Grouped GEMM (TE)            | 7,638.0   | 16,896 | 0.45     |
| MoE expert forward           | 7,215.6   | 2,816  | 2.56     |
| Grouped GEMM (native)        | 7,155.1   | 16,896 | 0.42     |
| sort_chunks (Triton)         | 2,420.6   | 11,264 | 0.21     |
| _moe_chunk_sort              | 2,004.0   | 5,632  | 0.36     |
| _moe_chunk_sortBackward      | 1,698.0   | 5,632  | 0.30     |

---

## 6. Comparison with Llama 3.1 405B NVFP4

|                        | 405B NVFP4 (PP=16)   | Qwen3 235B BF16      |
|------------------------|----------------------|----------------------|
| Model type             | Dense                | MoE (235B/22B active)|
| GPU bubble fraction    | 43%                  | **54.4%**            |
| Dominant bottleneck    | PP bubble (inter-node P2P) | MoE AllToAll (inter-node IB) |
| PP communication share | Dominant             | 14.9% of comm        |
| MoE AllToAll share     | N/A                  | 79.4% of comm        |
| DP gradient sync       | Significant          | Negligible (0.0%)    |
| Optimization applied   | PP=16→PP=8 (+38%)    | TBD                  |

The Qwen3 235B bottleneck profile is fundamentally different from the dense 405B model. The 405B was PP-bubble-dominated and fixed by reducing PP stages. The Qwen3 is MoE-AllToAll-dominated with many small (1-10ms) gaps caused by inter-node expert token dispatch.

---

## 7. Recommended Optimization Experiments

### Priority 1: Reduce MoE AllToAll overhead

1. **Communication-compute overlap for AllToAll** — Overlap expert token dispatch with attention or other compute. Check if Megatron-Bridge has `--moe-alltoall-overlap` or similar flag.

2. **EP tuning (EP=4)** — Reduce EP from 8 to 4 to halve the number of ranks participating in AllToAll. This reduces IB traffic at the cost of increased per-expert compute. With 128 experts / EP=4 = 32 experts/rank.

3. **FP8 quantization** — Use FP8 for MoE expert compute and AllToAll payloads to halve communication volume. Scripts already prepared at `~/johnson/scripts/qwen3_235b/fp8_*/`.

### Priority 2: Reduce pipeline bubble

4. **PP reduction (PP=4)** — Reduce PP from 8 to 4. With TP=1, PP=4, EP=8: DP = 256/(1x4x8) = 8. Halves pipeline bubble contribution (currently 14.9% of comm).

### Priority 3: Communication acceleration

5. **NVLS enable** — `NCCL_NVLS_ENABLE=1` for intra-node collective acceleration. May help intra-node AllToAll component.

6. **Hybrid EP (EP + TP)** — Use TP=2 + EP=4 to keep AllToAll within fewer nodes while leveraging NVLink for TP. Scripts at `~/johnson/scripts/qwen3_235b/fp8_hybridep/`.

---

## 8. Methodology Notes

- PyTorch profiler with `ProfilerActivity.CPU` and `ProfilerActivity.CUDA` enabled
- GPU kernel events appeared in the last ~8M lines of a 238M-line trace (Chrome trace format stores CPU events first, GPU events last)
- Trace gzip was truncated at 738MB due to NCCL timeout during `export_chrome_trace` — GPU data is partial but representative
- `profiling.py` was patched to add CUDA activity (original Megatron-Bridge code omitted `activities` parameter)
- Profiler overhead: step 3 (warmup) ran at 74.6s vs 57.5s steady state (~30% overhead)
