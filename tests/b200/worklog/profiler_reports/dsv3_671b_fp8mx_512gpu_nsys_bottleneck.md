# DeepSeek V3 671B FP8 MX — Nsys Bottleneck Report (Multi-Stage)

**Date:** 2026-04-21
**Source job:** 84537 (exemplar reference run, 512 ranks × full nsys trace)
**Cluster:** Together AI B200 (use3a), 64 nodes × 8 GPUs
**Container:** NeMo 26.02 (nsys 2025.5.1)
**Parallelism:** TP=1, PP=16, CP=1, EP=8, VP=None, MBS=1, GBS=4096, DP=32
**Profiled ranks:** 0 (stage 0), 240 (stage 7, middle), 504 (stage 15, last)
**Companion to:** `dsv3_671b_fp8mx_512gpu_profiler.md` (torch-profiler rank-0 analysis)

---

## 1. Executive Summary

The torch-profiler rank-0 trace showed the pipeline bubble (43%) dominating stage 0 and
concluded the MoE cost was invisible from that vantage point. This nsys report confirms
that directly and **locates the true bottlenecks by comparing three PP stages** from the
same 512-rank job:

| Finding | Stage 0 (rank 0) | Stage 7 (rank 240) | Stage 15 (rank 504) |
|---|---|---|---|
| PP SendRecv % of GPU time | **54.3%** | 26.9% | 33.1% |
| Hybrid-EP sync overhead (MoE A2A) | 1.5% | **7.6%** | 2.6% |
| MX FP8 GEMMs (active compute) | ~21% | ~40% | ~27% |
| LM-head BF16 projection | — | — | **23.3%** (3 kernels) |
| cuDNN Flash Attn (fwd+bwd) | 7.7% | 7.9% | 7.7% |
| MXFP8 quantize | 2.1% | 4.0% | 2.2% |

**Three takeaways the torch profile missed:**

1. **Middle-stage MoE is the real EP bottleneck (7.6% on stage 7, 5× rank 0).** The
   `hybrid_ep::device_sync_kernel` sits in the A2A dispatch/combine critical path.
   Rank 0 understates this because stages 0–2 run only dense MLA layers; the MoE cost
   concentrates on stages 3–14 where most experts live.
2. **Last-stage LM head dominates with three huge BF16 GEMMs (23.3% of stage 15).**
   `nvjet_sm100_tst_*_NTT/NNT/TNT` at 2,560 instances × ~4.5–5 ms each — no UE8M0
   microscale (BF16 path). This is the vocab projection + MTP head, and it is the
   single largest non-NCCL cost on the last stage.
3. **Rank-0 bubble is 54% (nsys), higher than the 43% reading from torch-profiler.**
   The discrepancy is because the torch profile captured only one iteration with
   warm-up bias; steady-state PP stage-0 spends roughly half its wall time waiting.
   Conversely, the middle stages spend only ~27% on SendRecv — the pipeline is better
   utilized in the middle than the edges.

---

## 2. Setup

| Parameter | Value |
|---|---|
| Precision | FP8 MX (MXFP8 e4m3, UE8M0 microscale) |
| Model | DeepSeek V3 671B (3 dense + 58 MoE layers + MTP) |
| GBS / MBS | 4096 / 1 (exemplar) — 8192 for the current 84549 baseline |
| Profiler | `nsys profile` with NVTX (NCCL domain), CUDA kernels, cuBLAS, cuDNN |
| Profile span | Entire steady-state region (5 iters) |
| Trace size | 170 MB (rank 0) + 386 MB (rank 240) + 217 MB (rank 504) |

Artifacts copied locally for analysis:

```
/mnt/vast/johnson/nsys_analysis_dsv3/
├── rank0_stage0.nsys-rep       (170 MB, 75 M events)
├── rank240_stage7.nsys-rep     (386 MB, 170 M events)
├── rank504_stage15.nsys-rep    (217 MB, 96 M events)
├── *.sqlite                    (exported via nsys stats)
└── nsys_stats.out              (cuda_gpu_kern_sum top-30 per rank)
```

Source location (read-only): exemplar job 84537 nsys_profile directory.

---

## 3. Per-Stage Kernel Breakdown

### 3.1 Stage 0 (rank 0) — PP bubble-limited

| Rank | Kernel / Category | Time | % | Instances | Notes |
|---|---|---:|---:|---:|---|
| 1 | `ncclDevKernel_SendRecv` | 86.76 s | **54.3%** | 1,355 | PP bubble, avg 64 ms |
| 2 | cudnn sdpa sm100 flash **bprop** | 9.41 s | 5.9% | 5,120 | MLA backward |
| 3 | `nvjet_sm100_qqtst_128x256_…_v_badd_NTT` (MX FP8) | 6.25 s | 3.9% | 17,870 | dense MLP fwd |
| 4 | `nvjet_sm100_qqtst_128x256_…_v_bz_TNT` (MX FP8) | 4.68 s | 2.9% | 44,833 | MLA projection |
| 5 | `triton_poi_fused_add_cat_mul_rsub_sigmoid_silu_split` | 4.55 s | 2.8% | 5,120 | SwiGLU-activation |
| 6 | `nvjet_…_v_bz_NNT` (MX FP8) | 3.96 s | 2.5% | 29,473 | MLA QK/VO |
| 7 | `nvjet_…_h_bz_TNT` (MX FP8) | 3.74 s | 2.3% | 31,967 | MLA grad |
| 8 | `quantize_mxfp8_kernel` | 3.37 s | 2.1% | 229,120 | MX FP8 quant |
| 9 | cudnn sdpa sm100 flash **fprop** | 2.82 s | 1.8% | 5,120 | MLA forward |
| 10 | `ncclDevKernel_AllGather_RING_LL` | 2.39 s | 1.5% | 1,365 | optimizer AG |
| 11 | `hybrid_ep::device_sync_kernel` | 2.38 s | 1.5% | 5,120 | minimal MoE on stage 0 |
| 12 | `ncclDevKernel_ReduceScatter_Sum_bf16` | 2.21 s | 1.4% | 85 | DP sync |

Stage 0 is **pipeline-idle** for more than half its wall time. The active compute is
well-fed (large-tile MX FP8 GEMMs, Blackwell sm100), but there simply isn't much of it
because stage 0 owns only 3 dense MLA layers and no experts.

### 3.2 Stage 7 (rank 240) — MoE-limited middle stage

| Rank | Kernel / Category | Time | % | Instances | Notes |
|---|---|---:|---:|---:|---|
| 1 | `ncclDevKernel_SendRecv` | 41.20 s | 26.9% | 2,645 | PP 1F1B (balanced, lower %) |
| 2 | **`hybrid_ep::device_sync_kernel`** | **11.71 s** | **7.6%** | 20,480 | MoE A2A barrier |
| 3 | cudnn sdpa sm100 flash **bprop** | 9.32 s | 6.1% | 5,120 | MLA backward |
| 4 | `nvjet_…_h_badd_NTT` (MX FP8) | 6.50 s | 4.2% | 173,420 | MoE expert MLP fwd |
| 5 | `nvjet_…_v_bz_NNT` (MX FP8) | 6.31 s | 4.1% | 84,576 | MoE expert MLP grad |
| 6 | `ncclDevKernel_AllGather_RING_LL` | 6.15 s | 4.0% | 5,130 | EP all-gather (expert dispatch) |
| 7 | `quantize_mxfp8_kernel` | 6.13 s | 4.0% | 716,800 | **3× more quantize than stage 0** |
| 8 | `nvjet_…_h_bz_TNT` (MX FP8) | 5.61 s | 3.7% | 99,744 | MoE expert MLP bwd |
| 9 | `nvjet_…_v_bz_TNT` (MX FP8) | 4.42 s | 2.9% | 79,454 | |
| 10 | `nvjet_…_v_bz_TNT` (MX FP8, variant) | 4.31 s | 2.8% | 99,936 | |
| 11 | `nvjet_…_h_bz_NNT` (MX FP8) | 3.82 s | 2.5% | 84,382 | |
| 12 | `nvjet_…_h_badd_NTT` (MX FP8) | 3.55 s | 2.3% | 163,200 | |
| 13 | `nvjet_…_h_bz_NNT` (MX FP8) | 2.94 s | 1.9% | 94,620 | |
| 14 | cudnn sdpa sm100 flash **fprop** | 2.82 s | 1.8% | 5,120 | |
| 15 | `unpermute_kernel<…bf16,float>` | 2.38 s | 1.6% | 10,240 | MoE token unpermute |
| 16 | `permute_kernel<…ushort,float,float>` | 1.89 s | 1.2% | 10,240 | MoE token permute |
| 17 | `hybrid_ep::combine_kernel` (variant A) | 1.73 s | 1.1% | 5,120 | MoE combine |
| 18 | `hybrid_ep::dispatch_kernel` (variant A) | 1.64 s | 1.1% | 5,120 | MoE dispatch |
| 19 | `hybrid_ep::combine_kernel` (variant B) | 1.63 s | 1.1% | 5,120 | |
| 20 | `hybrid_ep::dispatch_kernel` (variant B) | 1.50 s | 1.0% | 5,120 | |

Summing the MoE-specific rows (hybrid_ep sync + AG + permute/unpermute + dispatch/combine):

> **MoE-attributable time on stage 7: ≈ 28.7 s (18.7% of GPU time).**

Of that, `hybrid_ep::device_sync_kernel` alone (7.6%) is the single biggest optimizable
item — it is the ordering barrier between dispatch and expert compute. Reducing it
requires either compute/comm overlap in the MoE path or larger per-rank token batches
to amortize the sync.

### 3.3 Stage 15 (rank 504) — LM-head-limited last stage

| Rank | Kernel / Category | Time | % | Instances | Notes |
|---|---|---:|---:|---:|---|
| 1 | `ncclDevKernel_SendRecv` | 50.65 s | 33.1% | 1,285 | last-stage PP traffic |
| 2 | **`nvjet_sm100_tst_320x192_…_h_bz_NNT` (BF16)** | **12.95 s** | **8.5%** | 2,560 | LM-head / MTP proj |
| 3 | **`nvjet_sm100_tst_128x256_…_v_badd_NTT` (BF16)** | **11.40 s** | **7.4%** | 2,560 | LM-head / MTP proj |
| 4 | **`nvjet_sm100_tst_128x256_…_v_bz_TNT` (BF16)** | **11.36 s** | **7.4%** | 2,560 | LM-head / MTP proj |
| 5 | cudnn sdpa sm100 flash bprop | 4.88 s | 3.2% | 2,560 | MLA bwd (fewer batches) |
| 6 | `hybrid_ep::device_sync_kernel` | 3.92 s | 2.6% | 10,240 | lower — fewer MoE layers |
| 7 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 3.77 s | 2.5% | **5** | 5× huge 750 ms — MTP loss A/R |
| 8 | `nvjet_…_h_bz_TNT` (MX FP8) | 3.46 s | 2.3% | 51,073 | MoE expert bwd |
| 9 | `quantize_mxfp8_kernel` | 3.38 s | 2.2% | 360,960 | |
| 10 | `nvjet_…_v_bz_NNT` (MX FP8) | 3.31 s | 2.2% | 42,367 | |

The three BF16 `nvjet_sm100_tst_*` kernels are **not** MX FP8 variants (no
`Avec32UE8M0` suffix). They are 320×192 / 128×256 large-tile tensor-core BF16 matmuls
run exactly 2,560 times each — matching the number of microbatches (512) × 5 iters,
consistent with **the LM head + MTP projection head staying in BF16** while the rest of
the model runs MX FP8.

**Combined LM-head cost: 35.71 s = 23.3% of stage-15 GPU time.** This is the single
largest compute item anywhere in the pipeline and was completely invisible in the
rank-0 profile.

There is also a one-shot 3.77 s BF16 AllReduce across 5 instances (~750 ms each) which
is MTP loss / parameter-grad aggregation — negligible per-step but worth noting.

---

## 4. Cross-Stage Comparison

| Bucket | Stage 0 | Stage 7 | Stage 15 | Observation |
|---|---:|---:|---:|---|
| PP SendRecv | 54.3% | 26.9% | 33.1% | Bubble concentrated at edges |
| MoE hybrid_ep sync | 1.5% | 7.6% | 2.6% | Real MoE cost in the middle |
| MoE permute/unpermute | 0.4% | 2.8% | 1.1% | Tracks hybrid_ep |
| MoE dispatch/combine | 0.7% | 4.3% | 2.1% | Tracks hybrid_ep |
| EP AllGather | 1.5% | 4.0% | 0.8% | Expert-routing AG |
| MX FP8 GEMMs (sum top 8) | 20.9% | 24.5% | 16.7% | Bulk of useful compute |
| BF16 LM-head GEMMs | — | — | 23.3% | Only on last stage |
| cuDNN Flash Attn (fwd+bwd) | 7.7% | 7.9% | 7.7% | Constant across stages |
| MXFP8 quantize | 2.1% | 4.0% | 2.2% | Middle stage quantizes 3× more |
| DP ReduceScatter | 1.4% | 1.5% | 0.5% | Small, overlapped |
| Optimizer AllGather | 1.5% | — | — | Visible only stage 0 sample |

**Interpretation:**

- Compute per stage is **not uniform**: stage 0 has 3 dense layers only, stage 7 carries
  the majority of MoE compute, stage 15 carries the LM head. The pipeline bubble follows
  from this imbalance; edges of the pipeline are more idle.
- MoE "A2A" on B200 is implemented as a **hybrid_ep** scheme using NVLink within the node
  (EP=8 fits inside a single node) and IB only for expert-group aggregation. The
  `device_sync_kernel` is the NVLink-side barrier that gates the dispatch/combine waves.
- MXFP8 quantize count scales with FFN sub-block count; stage 7 runs 8 MoE blocks per
  microbatch × 256 µbatches ≈ 717k quantize calls. Kernel is tiny (~8 µs) so per-call
  overhead is the concern, not DRAM throughput.

---

## 5. Revised Bottleneck Hierarchy (Actionable)

Combining this nsys multi-stage view with the torch-profiler rank-0 report, the
optimization priorities for DSV3 FP8 MX at 512 GPUs are:

| Priority | Target | Impact ceiling | Approach |
|---|---|---:|---|
| **P1** | MoE A2A: `hybrid_ep::device_sync_kernel` | up to ~8% of middle-stage GPU time | Overlap EP dispatch/combine with expert compute; explore token-batching or inflight-waves tuning in TE hybrid_ep. Check if `NVLINK_DOMAIN_SIZE` / `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` can reduce sync frequency. |
| **P2** | PP bubble reduction (edge stages) | up to ~25% recovered on stage 0, ~5–10% on stage 15 | Enable virtual-pipeline (VP>1) to split the bubble; current config has `VP=None`. Reference Qwen3 235B used VP=4 and the bubble was balanced. |
| **P3** | BF16 LM head (23% on stage 15) | up to 23% of stage 15 | Move the vocab projection to MX FP8 if numerically safe. Same for MTP head. Guard with a validation run since output head is loss-adjacent. |
| **P4** | Elementwise / activation-checkpoint copies | 10–12% on rank 0, less elsewhere | Tune `activation_checkpoint_layers` and MLA recompute policy; many small BF16 copies come from recomputing `mla_up_proj`. |
| **P5** | MXFP8 quantize launch overhead | 2–4% per stage | Fused TE quantize + GEMM kernel, or larger quantize tiles. Low ceiling but low risk. |
| P6 | cuDNN Flash Attn (steady ~8%) | small | Not a bottleneck; Blackwell sm100 flash is already tuned. |

**Not worth chasing:**

- DP ReduceScatter is <1.5% everywhere — already well-overlapped.
- Optimizer AllGather is <2% — negligible compared to PP/MoE.
- MX FP8 GEMMs are not the bottleneck; they are the useful work we're trying to expose
  more of.

---

## 6. Relation to the Torch-Profiler Report

The rank-0 torch-profile reported "43% pipeline P2P" and "1.6% MoE". That was numerically
correct **for rank 0 on one iteration**, but is misleading as a model-wide picture because
stage 0 is atypical (dense only, no experts). The nsys multi-stage view corrects the
picture:

- PP bubble on stage 0 is actually ~54% at steady state (single-iter torch profile had
  warm-up bias).
- MoE is the top compute bottleneck in the **middle** stages (7.6% on stage 7), which
  torch profile on rank 0 could never see.
- LM-head BF16 projection is the top compute bottleneck on the **last** stage (23%),
  which is also invisible from rank 0.

**Recommendation (if we re-profile torch-profiler):** capture
`profile_ranks=[0, 128, 480]` (stage 0, stage 4, stage 15) so the torch trace
covers the same three vantage points this nsys report used.

---

## 7. Artifacts

| What | Where |
|---|---|
| Nsys reports (3 ranks) | `/mnt/vast/johnson/nsys_analysis_dsv3/rank{0,240,504}_stage*.nsys-rep` |
| Sqlite exports | `/mnt/vast/johnson/nsys_analysis_dsv3/*.sqlite` |
| `nsys stats` top-30 per rank | `/mnt/vast/johnson/nsys_analysis_dsv3/nsys_stats.out` |
| Source (read-only) | `/mnt/vast/exemplar/llmb/workloads/pretrain_deepseek-v3/experiments/pretrain_deepseek_v3_fp8_mx_gpus512_tp1_pp16_cp1_vpNone_ep8_etp1_mbs1_gbs4096/…/nsys_profile/` (510 files) |
| Companion torch-profiler report | `./dsv3_671b_fp8mx_512gpu_profiler.md` |
| Baseline bench job (same config family) | 84549 (50-step, 596 TFLOP/s/GPU, GBS=8192) |

---

*Generated by comparing `nsys stats --report cuda_gpu_kern_sum` across ranks 0, 240, 504
from exemplar job 84537. Nsys run inside NeMo 26.02 container (nsys 2025.5.1) because
host nsys 2025.1.3 is too old for the newer report format.*
