# Qwen3 235B FP8 MX — Profiler Bottleneck Analysis & Optimization Report

**Date:** 2026-04-11
**Cluster:** Together AI B200 (use3a), 32 nodes x 8 GPUs (256 GPUs total)
**Container:** nvidia+nemo+26.02.00

---

## Executive Summary

FP8 MX baseline achieves **383 TFLOP/s/GPU** (~50.6s/step), **25% slower** than BF16 at
**514 TFLOP/s/GPU** (~37.8s/step). After systematic profiling and 6 optimization
experiments, the **best FP8 MX configuration** uses **CUDA graphs only**, achieving
**425 TFLOP/s/GPU** (~45.6s/step) — a **+10% improvement** over the FP8 baseline.

A **17% gap to BF16 remains** and is largely intrinsic to MXFP8 on B200.

### Profiler-Identified Bottlenecks

1. **Pipeline P2P stalls (largest contributor):** 215 idle gaps averaging 117ms each,
   accounting for 94% of GPU idle time. PP stage 0 finishes compute and waits for
   activations from downstream stages.
2. **MXFP8 quantization overhead (~2.5% of step):** 107K quantize_mx kernel launches
   per step add 858ms of pure overhead not present in BF16.
3. **MXFP8 GEMM throughput penalty:** Block-scaled FP8 GEMMs (`nvjet_sm100_qqtst`)
   are inherently slower than BF16 native GEMMs for the small per-expert MoE matrix
   sizes in Qwen3-235B.

### Optimization Results

| Optimization | Result | Verdict |
|---|---|---|
| **CUDA Graphs** | 50.6s → **45.6s** (+10%) | **Use this** |
| VP=4 | +1% | Not worthwhile |
| HybridEP | +0.2% over CG alone | No benefit |
| MoE A2A Overlap + VP=4 | **-42% (regression)** | Do not use |
| Param Gather Overlap | No benefit | Do not use |

**Recommended FP8 config:** CUDA graphs (`--cuda_graph_impl transformer_engine
--cuda_graph_scope moe_router,moe_preprocess`) with default VP=None and EP=8.

---

## 1. Setup

| Parameter | FP8 MX (profiled) | BF16 (reference) |
|-----------|-------------------|------------------|
| Precision | FP8 MX (MXFP8 e4m3, block scaling) | BF16 |
| Parallelism | TP=1, PP=8, CP=1, EP=8, VP=None | TP=1, PP=8, CP=1, EP=8, VP=4 |
| Batch | MBS=1, GBS=8192, DP=32 | MBS=1, GBS=8192, DP=32 |
| Topology | PP intra-node (NVLink), EP/DP inter-node (IB) | Same |
| Profiler | PyTorch (CPU + CUDA), rank 0, step 3-4 | Same |
| Steady-state step time | ~50.6s (383 TFLOP/s/GPU) | ~37.8s (514 TFLOP/s/GPU) |
| Profiled step (warmup) | 71.1s (job 80083) | 74.6s (job 79973) |
| Trace size | 766MB gz / 9.5GB decompressed | 738MB gz |
| moe_a2a_overlap | False | False |

> Both traces are partial (gzip truncated during export due to NCCL timeout). Kernel
> event coverage: FP8 captured 391K kernel + 2.4M cpu_op events; BF16 captured ~8M
> lines of GPU events. Proportional breakdowns are representative; absolute timings
> should not be compared directly between the two traces.

---

## 2. Bottleneck #1 — Pipeline P2P Stalls

**The single largest bottleneck.** PP stage 0 (rank 0) repeatedly idles waiting for
activation data from downstream pipeline stages.

### Evidence

| Metric | FP8 MX | BF16 |
|--------|--------|------|
| Gaps >1ms (count) | 1,099 | 4,985 |
| Total idle (gaps >1ms) | 26.80s | 16.90s |
| Dominant gap size | 100ms-1s (**94.4%** of idle) | 1-10ms (99.2% of idle) |
| Dominant pattern | `SendRecv → [117ms avg] → Memcpy HtoD` | `_permute → [3.4ms avg] → GEMM` |
| Largest single gap | 124.1ms | 111.4ms |

### Gap Distribution

| Bucket | FP8 Count | FP8 Time (s) | FP8 % of idle | BF16 Count | BF16 % of idle |
|--------|-----------|-------------|---------------|------------|----------------|
| 1-10ms | 880 | 1.20 | 4.5% | 4,982 | 99.2% |
| 10-100ms | 4 | 0.31 | 1.2% | 2 | 0.1% |
| 100ms-1s | **215** | **25.29** | **94.4%** | 1 | 0.7% |

### Interpretation

BF16 and FP8 have **fundamentally different bubble patterns**:

- **BF16:** Thousands of small 1-10ms gaps caused by MoE AllToAll dispatch latency
  over IB. The GPU waits briefly after each expert permutation for AllToAll to complete
  before starting the next grouped GEMM.

- **FP8 MX:** 215 large ~117ms gaps caused by pipeline stage imbalance. PP stage 0
  completes its compute faster (FP8 GEMMs are individually faster at lower precision),
  then sits idle while downstream stages — which carry the same MoE AllToAll overhead
  plus FP8 quantization overhead — catch up.

VP=4 was tested (job 80079) and gave only **+1%** improvement, confirming the bubble
is not a scheduling issue but a stage workload imbalance. With PP=8 intra-node, each
stage has different numbers of MoE vs dense layers, and the FP8 overhead amplifies this
imbalance.

---

## 3. Bottleneck #2 — MXFP8 Quantization Overhead

Every GEMM input must be block-quantized to FP8 e4m3 with per-32-element scaling factors.
This is pure additive overhead that does not exist in BF16.

### FP8-Specific Kernels

| Kernel | Total (ms) | Count | Avg (us) | Role |
|--------|-----------|-------|----------|------|
| `mxfp8::quantize_mx` (primary) | 857.7 | 107,258 | 8.0 | Block-scale quantization for every GEMM input |
| `elementwise_kernel` (cast/copy) | 275.7 | 8,447 | 32.6 | FP8 tensor element-wise operations |
| `mxfp8::quantize_mx` (variant 2) | 44.1 | 2,816 | 15.7 | Additional quantization path |
| `unrolled_elementwise` (copy) | 22.6 | 5,632 | 4.0 | FP8 data movement |
| Other FP8 support ops | 38.1 | — | — | Small cast/scale kernels |
| **Total** | **1,238.2** | — | — | **8.4% of GPU compute, ~2.5% of step** |

### Key Observations

- The primary `quantize_mx` kernel is invoked **107,258 times per step** at just 8us
  each. The sheer volume of launches (not per-kernel cost) drives overhead.
- **CUDA graphs** could amortize these launches by capturing the quantization + GEMM
  sequence as a single graph replay, eliminating per-kernel launch overhead.
- The existing `fp8_overlap/train.sh` already uses `--cuda_graph_scope=moe_router,moe_preprocess`
  to address this.

---

## 4. Bottleneck #3 — MXFP8 GEMM Throughput

FP8 uses specialized MXFP8 GEMM kernels that are inherently slower than BF16 native
GEMMs for the MoE expert matrix sizes in Qwen3-235B.

### GEMM Kernel Comparison

| FP8 MX GEMM | Total (ms) | Count | Avg (ms) |
|-------------|-----------|-------|----------|
| `nvjet_sm100_qqtst` 128x128 | 797.7 | 22,394 | 0.036 |
| `nvjet_sm100_qqtst` 128x256 (v1) | 791.3 | 25,478 | 0.031 |
| `nvjet_sm100_qqtst` 128x256 (v2) | 705.6 | 25,210 | 0.028 |
| `nvjet_sm100_qqtst` 256x128 | 697.5 | 22,662 | 0.031 |
| All FP8 GEMMs | **4,854.8** | — | — |

| BF16 GEMM | Total (ms) | Count | Avg (ms) |
|-----------|-----------|-------|----------|
| `nvjet_sm100` 128x256 NTT | 810.0 | 17,625 | 0.046 |
| `nvjet_sm100` 176x128 TNN | 706.5 | 9,410 | 0.075 |
| `nvjet_sm100` 128x192 TNT | 478.5 | 8,109 | 0.059 |
| `nvjet_sm100` 128x192 NTT | 397.7 | 17,632 | 0.023 |
| All BF16 GEMMs (partial) | ~3,700 | — | — |

### GEMM % of GPU Time

| | FP8 MX | BF16 |
|---|---|---|
| GEMM as % of total GPU | 32.9% | ~70% |
| GEMM as % of non-NCCL GPU | ~51% | ~70% |

### Interpretation

- FP8 GEMM kernel names (`qqtst`, `Avec32UE8M0_Bvec32UE8M0`) indicate both inputs are
  quantized with 32-element block scaling using unsigned E8M0 scale factors. These kernels
  must dequantize on-the-fly during the matrix multiply.
- MXFP8 on B200 is relatively new. The `nvjet_sm100_qqtst` kernels may not yet be as
  optimized as the mature BF16 `nvjet_sm100` kernels for these specific tile sizes.
- MoE expert GEMMs have relatively small per-expert matrices (235B total / 128 experts),
  which may not fully utilize FP8 tensor cores.

---

## 5. Full GPU Time Breakdown

### FP8 MX GPU Kernel Categories

| Category | Time (ms) | % of GPU | BF16 ref % | Notes |
|----------|----------|----------|------------|-------|
| **NCCL SendRecv (PP P2P)** | 4,864 | 32.9% | — | Pipeline point-to-point |
| **GEMM (MXFP8)** | 4,855 | 32.9% | ~70% | Expert + attention GEMMs |
| MoE routing (sort/permute) | 1,189 | 8.1% | 9.0% | Token dispatch/collection |
| **MXFP8 quantize/cast** | 916 | 6.2% | 0% | Pure FP8 overhead |
| Flash Attention (SDPA) | 946 | 6.4% | 5.3% | Forward + backward |
| TE padding/unpadding | 795 | 5.4% | — | Transformer Engine |
| RMSNorm | 333 | 2.3% | 1.8% | Layer normalization |
| Other | 866 | 5.9% | 14% | RoPE, SiLU, elementwise, etc. |

### NCCL Communication on GPU

| Kernel | Time (ms) | % of NCCL | Notes |
|--------|----------|-----------|-------|
| SendRecv (PP P2P) | 4,864 | 94.0% | Pipeline activations (intra-node NVLink) |
| AllGather RING_LL | 167 | 3.2% | Parameter gather (EP/DP) |
| ReduceScatter bf16 | 142 | 2.8% | Gradient sync (DP) |

### CPU-side Communication

| Op | Time (s) | Count | Avg (ms) | % of Comm |
|---|---------|-------|----------|-----------|
| AllToAll (MoE forward) | 3.72 | 16,896 | 0.22 | 35.0% |
| AllToAll (MoE backward) | 2.13 | 8,447 | 0.25 | 20.0% |
| alltoall_base_ | 2.13 | 16,896 | 0.13 | 20.1% |
| allgather_base_ | 0.42 | 2,816 | 0.15 | 3.9% |
| send (PP) | 0.03 | 256 | 0.10 | 0.2% |

---

## 6. Optimization Opportunities

### 6.1 CUDA Graphs for MoE (Priority 1 — Low Risk)

**Target:** Bottleneck #2 (107K quantization kernel launches) + general launch overhead.

The existing `fp8_overlap/train.sh` already demonstrates this:
```
--cuda_graph_impl=transformer_engine
--cuda_graph_scope=moe_router,moe_preprocess
```

CUDA graphs capture a sequence of kernel launches and replay them as a single GPU
operation, eliminating per-kernel CPU launch overhead. With 107K quantize_mx invocations
at 8us each, the cumulative launch overhead is substantial.

**Expected impact:** Reduce FP8 quantization overhead and MoE routing kernel launch
latency. Estimated 3-5% step time improvement.

### 6.2 HybridEP Dispatcher (Priority 2 — Medium Risk)

**Target:** Bottleneck #1 (pipeline stalls due to AllToAll blocking expert compute).

The existing `fp8_overlap/train.sh` uses:
```
model.moe_flex_dispatcher_backend=hybridep
model.moe_hybridep_num_sms=32
```

HybridEP dedicates 32 SMs to AllToAll communication while the remaining 116 SMs (of
148 total on B200) run expert GEMMs in parallel. This overlaps the AllToAll with compute
instead of running them sequentially.

**Expected impact:** Reduce the AllToAll-induced component of pipeline stalls. The CPU-side
AllToAll is 5.85s (fwd+bwd) and contributes to the stage imbalance that causes the 117ms
P2P gaps. Estimated 5-10% step time improvement.

### 6.3 MoE A2A Overlap (Priority 3 — Medium Risk)

**Target:** AllToAll communication overlap with compute.

The `moe_a2a_overlap` flag enables:
```python
comm_overlap.overlap_moe_expert_parallel_comm = True
comm_overlap.delay_wgrad_compute = True
model.moe_shared_expert_overlap = False
```

Currently `False` for B200 V2 configs, but `True` for B300, GB200, and H100 configs.
This overlaps MoE expert-parallel communication with weight gradient computation.

**Note:** This is a different mechanism than HybridEP. HybridEP overlaps AllToAll with
expert forward/backward compute at the SM level. `moe_a2a_overlap` overlaps the AllToAll
with weight gradient computation at the scheduling level. They may be complementary.

**Expected impact:** 2-5% improvement. May interact with HybridEP — test independently.

### 6.4 Param Gather Overlap (Priority 4 — High Risk)

**Target:** Overlap parameter AllGather with optimizer step.

```
comm_overlap.overlap_param_gather=true
ddp.overlap_param_gather=true
optimizer.overlap_param_gather=true
```

Currently disabled for MXFP8 due to a known **NaN grad norm bug** (see
`overrides.py:430-434`). The `fp8_overlap/train.sh` enables it, suggesting it may work
with newer container versions or specific configurations.

**Expected impact:** 2-3% improvement, but requires validating no NaN grad norms.

### 6.5 Newer Container (Priority 5 — Low Risk)

**Target:** Bottleneck #3 (MXFP8 GEMM throughput).

MXFP8 support on B200 is new. The `nvjet_sm100_qqtst` GEMM kernels may be better
optimized in newer CUDA/cuBLAS releases. The `fp8_overlap/train.sh` references the
25.11.01 container; our setup uses 26.02.00. Even newer releases (26.03+) may have
further improvements.

### Experiment Results

| # | Optimization | Job | Avg Step (5-16) | TFLOP/s/GPU | vs Baseline | Result |
|---|---|---|---|---|---|---|
| 1 | CUDA Graphs | 80084 | **45.62s** | **425** | **+9.9%** | **Best config** |
| 2 | CUDA Graphs + HybridEP | 80085 | **45.53s** | **426** | **+10.1%** | HybridEP adds nothing |
| 3a | CG + A2A Overlap + VP=4 | 80088 | 64.81s | 299 | -28.0% | **REGRESSION** |
| 3b | CG + A2A + VP=4 + Param Overlap | 80089 | 64.80s | 299 | -28.0% | **REGRESSION** |

**CUDA graphs account for the full ~10% gain.** They eliminate kernel launch overhead
for the 107K quantize_mx invocations and MoE routing kernels by replaying captured GPU
command sequences. The graph capture happens on iteration 4 (~83s spike), after which
steady state drops from 50.6s to 45.6s.

**HybridEP adds no measurable benefit** (<0.2%). The SM-level AllToAll/compute overlap
does not help because the dominant bottleneck is pipeline P2P stalls, not AllToAll
latency. The 32 SMs dedicated to AllToAll are wasted.

### Remaining Gap Analysis

After CUDA graphs, **17% gap** to BF16 remains (45.6s vs 37.8s = 7.8s):

| Source | Estimated Contribution | Addressable? |
|---|---|---|
| MXFP8 GEMM throughput penalty | ~5-6s | No (hardware/kernel maturity) |
| Residual FP8 quantization overhead | ~1-1.5s | Partially (newer kernels) |
| Pipeline P2P stall difference | ~1-2s | Maybe (PP reduction, stage balancing) |

This gap is **largely intrinsic to MXFP8 on B200** — block-scaled FP8 GEMMs
(`nvjet_sm100_qqtst`) are fundamentally slower than BF16 native GEMMs for the small
per-expert MoE matrix sizes. Further improvement requires either NVIDIA kernel
optimization (newer containers) or reducing PP stages.

### Summary Table

| # | Optimization | Mechanism | Expected | Actual | Status |
|---|---|---|---|---|---|
| 1 | **CUDA Graphs** | Reduce kernel launch overhead | 3-5% | **+9.9%** | **Done — use this** |
| 2 | HybridEP | Overlap AllToAll with expert compute | 5-10% | +0.2% | Tested — no benefit |
| 3a | MoE A2A Overlap + VP=4 | Overlap AllToAll with wgrad | 2-5% | **-42%** | **REGRESSION** |
| 3b | A2A Overlap + VP=4 + Param Overlap | Combine A2A + param gather overlap | 2-5% | **-42%** | **REGRESSION** |
| 4 | Newer Container | Better MXFP8 GEMM kernels | Unknown | — | Untested |

### Experiment 3 Details: MoE A2A Overlap (REGRESSION)

Jobs 80088 and 80089 tested `moe_a2a_overlap=true` with VP=4 (required when PP>1).
Both showed a catastrophic **42% regression** vs the CUDA-graphs-only baseline:

| Job | Config | Avg Step (5-16) | TFLOP/s/GPU | vs CG baseline |
|-----|--------|-----------------|-------------|----------------|
| 80088 | CG + A2A Overlap + VP=4 | **64.81s** | **299** | **-42.1%** |
| 80089 | CG + A2A + VP=4 + Param Overlap | **64.80s** | **299** | **-42.1%** |
| 80084 | CG only (reference) | 45.62s | 425 | — |

**Root cause:** The `moe_a2a_overlap` flag enables `delay_wgrad_compute=True` in
Megatron, which serializes weight gradient computation that was previously overlapped
with other operations. Combined with VP=4 (which adds pipeline microbatch interleaving
overhead without reducing the bubble for FP8), this creates a net slowdown of ~19s/step.

**Param gather overlap (job 80089)** adds no measurable difference when combined with
A2A overlap — the dominant regression is from `delay_wgrad_compute`. No NaN grad norms
were observed in either job, so the param overlap itself is safe with this config.

### Not Recommended

- **VP=4**: Tested (job 80079), only +1% improvement
- **HybridEP**: Tested (job 80085), no benefit over CUDA graphs alone
- **MoE A2A Overlap**: Tested (jobs 80088/80089), causes -42% regression due to `delay_wgrad_compute`
- **Param Gather Overlap**: Tested (job 80089), no benefit when combined with A2A overlap
- **SHARP**: Not available on this cluster (sharpd not running, needs admin)
- **NVLS**: Previously tested on BF16, showed no improvement
- **EP tuning**: EP=8 is already the standard config for 256 GPUs; EP=4 would increase
  per-expert memory

---

## 7. Experiment Log

| Job | Config | Status | Step Time | TFLOP/s/GPU | Delta | Notes |
|-----|--------|--------|-----------|-------------|-------|-------|
| 80077 | BF16 VP=4 baseline | COMPLETED | 37.79s | 514 | — | Reference |
| 80078 | FP8 MX VP=None baseline | COMPLETED | 50.65s | 383 | -25.4% | FP8 baseline |
| 80079 | FP8 MX VP=4 override | COMPLETED | 50.14s | 387 | +1.0% | VP=4 negligible |
| 80083 | FP8 MX profiler | COMPLETED | — | — | — | 766MB trace |
| 79973 | BF16 profiler | COMPLETED | — | — | — | 738MB trace (ref) |
| **80084** | **FP8 MX + CUDA graphs** | **COMPLETED** | **45.62s** | **425** | **+9.9%** | **Best FP8 config** |
| 80085 | FP8 MX + CG + HybridEP | COMPLETED | 45.53s | 426 | +10.1% | HybridEP adds nothing |
| 80086 | FP8 MX + CG + A2A (no VP) | FAILED | — | — | — | Assert: VP required with A2A+PP |
| 80087 | FP8 MX + CG + A2A + PO (no VP) | FAILED | — | — | — | Assert: VP required with A2A+PP |
| 80088 | FP8 MX + CG + A2A + VP=4 | COMPLETED | 64.81s | 299 | -28.0% | delay_wgrad regression |
| 80089 | FP8 MX + CG + A2A + VP=4 + PO | COMPLETED | 64.80s | 299 | -28.0% | Same regression |

---

## 8. Methodology

- **Profiler:** PyTorch profiler with `ProfilerActivity.CPU` and `ProfilerActivity.CUDA`
- **Patched profiling.py:** Added CUDA activities + custom chrome trace handler (container
  stock version omits `activities` parameter, capturing only CPU events)
- **Trace truncation:** Both traces gzip-truncated during `export_chrome_trace` due to
  NCCL timeout. GPU kernel events are in the tail of the trace; FP8 captured 391K kernel
  events, BF16 captured similar coverage. Proportions are representative.
- **Rank 0 only:** All data is from PP stage 0. Other stages may have different profiles
  (especially MoE-heavy stages), which could explain the pipeline imbalance.
- **Analysis tool:** `/tmp/analyze_fp8_trace_v2.py` — streaming regex extraction from
  multi-line Chrome trace JSON, processes 9.5GB in ~80s.

---

## Key Paths

| What | Path |
|------|------|
| FP8 trace | `.../profiler_traces/qwen3_235b_fp8mx_baseline/use3a-ss-b200-gpu-159...rank0.pt.trace.json.gz` |
| BF16 trace | `.../profiler_traces/qwen3_235b_bf16_baseline_cuda_rank0.pt.trace.json.gz` |
| BF16 profiler report | `~/johnson/scripts/qwen3_235b/qwen3_235b_bf16_256gpu_profiler_report.md` |
| FP8 profiler scripts | `~/johnson/scripts/qwen3_235b/256gpus_fp8mx_profiler/` |
| Existing FP8 optimization scripts | `~/johnson/scripts/qwen3_235b/fp8_overlap/` |
| Patched profiling.py | `.../profiler_traces/profiling.py` |
| Megatron-Bridge overrides | `.../Megatron-Bridge/scripts/performance/utils/overrides.py` |
| Qwen3 base configs | `.../Megatron-Bridge/scripts/performance/configs/qwen/qwen3_workload_base_configs.py` |
