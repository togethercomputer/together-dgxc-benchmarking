# DeepSeek V3 671B FP8 MX — Profiler Bottleneck Analysis (Rank 0, Initial Pass)

**Date:** 2026-04-21
**Cluster:** Together AI B200 (use3a), 64 nodes × 8 GPUs (512 GPUs total)
**Container:** nvidia+nemo+26.02.00
**Job:** 84552 (profiler run, 8 steps), baseline job: 84549 (50-step benchmark)

---

## Executive Summary

Profiled rank 0 of DeepSeek V3 671B FP8 MX on 512 GPUs. At steady state the benchmark
hits **~596 TFLOP/s/GPU** (28.58–28.72 s/step, GBS=8192). Torch profiler captured 1
full training iteration on rank 0, producing a 3.3 GB JSON trace (253 MB gzipped,
~12 M events).

**Headline finding:** Rank 0 (PP stage 0) spends **43% of its iteration idle on
pipeline P2P** — this is the expected 1F1B bubble for the first stage at PP=16. Actual
GPU compute is well-utilized when active. **This trace alone cannot show the MoE
bottleneck** because stage 0 has no MoE experts (first 3 DSV3 layers are dense MLA).
A follow-up multi-rank profile is required for a complete picture.

### Identified bottlenecks on rank 0

1. **Pipeline P2P wait (42.9%):** 271 `ncclDevKernel_SendRecv` events averaging 41.6 ms
   each; 256 microbatches × 1F1B warmup/cooldown waiting for activations and gradients.
   Intrinsic to PP=16 stage-0 — **not** a GPU inefficiency.
2. **Elementwise ops (11.8%, 3.11 s):** Largest optimizable overhead after GEMM. Drivers:
   activation-checkpoint tensor copies (`mla_up_proj` recompute policy), BF16→FP8 input
   staging, binary ops, fills.
3. **MXFP8 quantize overhead (2.9%, 0.77 s):** 50,524 launches of
   `quantize_mxfp8_kernel` @ ~15 µs each. Modest; FP8 recipe is not the bottleneck.

### Non-bottlenecks on rank 0 (this trace)

- **MoE/EP traffic negligible (1.6%, 0.41 s).** Rank 0 barely runs experts. Real MoE
  cost lives on middle PP stages — needs a separate profile.
- **DP collectives trivial (1.3%, 0.35 s):** optimizer reduce-scatter/all-gather +
  grad-norm all-reduce are small.

---

## 1. Setup

| Parameter | Value |
|---|---|
| Precision | FP8 MX (MXFP8 e4m3, block scaling, UE8M0 microscale) |
| Parallelism | TP=1, PP=16, CP=1, EP=8, VP=None |
| Batch | MBS=1, GBS=8192, DP=32 (seq len 4096) |
| Topology | PP intra-cluster (IB), EP intra-node (NVLink hybrid_ep), DP inter-node (IB) |
| Profiler | PyTorch profiler (CPU+CUDA), rank 0 only, steps 6-7 |
| Steady-state step time | 28.58–28.72 s (**~596 TFLOP/s/GPU**) |
| Profiled step (7) wall time | 228.3 s (includes trace-dump overhead at end of step) |
| Trace size | 253 MB gz / 3.3 GB decompressed |
| Trace events | 11,916,186 total; 219,862 GPU kernels |

### How it was run

Generated from `/mnt/vast/johnson/scripts/dsv3_512gpus_fp8mx_profiler/sbatch.sh`.
Key ingredients:

- Excluded nodes with known issues: `[190,199,201,202,204,211,228,230,233,239]`
  (204 had IB HCA fault in 84541; others flaky in 84544/84547).
- `NCCL_SOCKET_IFNAME=bond0` + `GLOO_SOCKET_IFNAME=bond0` to pin bootstrap to compute
  fabric (storage-fabric misrouting caused 84547 to hang; see
  `~/johnson/worklog/2026-04-21_dsv3_fp8_mx_512gpu_issues.md`).
- Bind-mounted patched `profiling.py` (adds `ProfilerActivity.CUDA`) over container's
  stock copy.
- Hydra overrides: `profiling.use_pytorch_profiler=true profile_step_start=6 end=7
  profile_ranks=[0] record_shapes=true`.

---

## 2. GPU-time category breakdown (rank 0, 1 iter)

Sum of GPU kernel durations = 26.31 s across 219,862 kernel events.
Per-stream merged busy time = 25.36 s / 28.68 s wall = **88.4% GPU utilization**.

| Category | Time (s) | % GPU | Count | What |
|---|---:|---:|---:|---|
| NCCL P2P (PP send/recv) | **11.282** | **42.9%** | 271 | `ncclDevKernel_SendRecv`, avg 41.6 ms |
| GEMM (MX FP8, nvjet sm100 UE8M0) | **6.115** | **23.2%** | 72,704 | MLA projections, grouped-linear MLP/expert |
| Elementwise (casts, copies, bin-ops) | 3.113 | 11.8% | 41,509 | Input staging, checkpoint saves |
| Flash Attention (cudnn sm100) | 2.703 | 10.3% | 4,096 | MLA fwd (0.586 s) + bwd (1.916 s) |
| MXFP8 quantize | 0.774 | 2.9% | 50,524 | `quantize_mxfp8_kernel` @ 15 µs avg |
| RMSNorm fwd/bwd | 0.606 | 2.3% | 14,336 | `rmsnorm_*_general_kernel` |
| RoPE (Q, KV, MTP) | 0.492 | 1.9% | 6,144 | `rotary_fwd/bwd_{q,kv}_kernel` |
| MoE permute / combine / dispatch | 0.410 | 1.6% | 1,792 | `permute_kernel`, `hybrid_ep::{dispatch,combine}_kernel` |
| Other | 0.407 | 1.5% | 25,856 | — |
| NCCL AllReduce | 0.146 | 0.6% | 11 | Grad norm, small-tensor sums |
| NCCL ReduceScatter | 0.121 | 0.5% | 17 | DP distributed-optimizer grads |
| NCCL AllGather | 0.082 | 0.3% | 273 | DP distributed-optimizer params |
| GEMM BF16 | 0.027 | 0.1% | 768 | Small residual BF16 paths |
| Reduce/optimizer/other | 0.026 | 0.1% | 1,561 | Adam, clip-grad |
| **TOTAL** | **26.305** | **100%** | **219,862** | |

### Interpretation

- **Compute share** (GEMM + attention + norm + rope) = 9.92 s = **37.7%** of GPU time.
- **Pipeline wait** = 11.28 s = 42.9%.
- **Memory-bound overhead** (elementwise + quantize) = 3.89 s = **14.8%**.
- **MoE + DP collectives** = 0.76 s = 2.9% — trivial on this rank.
- Remaining ≈ 0.50 s = other.

When we remove PP bubble from the denominator, rank-0 on-compute mix is:
GEMM 41% / attention 18% / elementwise 21% / quantize+norm+rope 13% / MoE+DP 5% / other 2%.

---

## 3. Top 15 GPU kernels

| Time (s) | % | Count | Avg (µs) | Kernel |
|---:|---:|---:|---:|---|
| 11.282 | 42.9% | 271 | 41,630 | `ncclDevKernel_SendRecv` (PP) |
| 1.916 | 7.3% | 1,024 | 1,871 | `cudnn sdpa_sm100_flash_bprop` (MLA bwd) |
| 1.243 | 4.7% | 3,574 | 348 | `nvjet_sm100_qqtst 128x256_128x6_2x1_2cta_v_badd UE8M0 NTT` |
| 0.991 | 3.8% | 8,817 | 112 | `nvjet_sm100_qqtst 128x256_128x6_2x1_2cta_v_bz UE8M0 TNT` |
| 0.800 | 3.0% | 5,745 | 139 | `nvjet_sm100_qqtst 128x256_128x6_2x1_2cta_v_bz UE8M0 NNT` |
| 0.792 | 3.0% | 6,543 | 121 | `nvjet_sm100_qqtst 128x256_128x6_2x2_2cta_h_bz UE8M0 TNT` |
| 0.687 | 2.6% | 45,824 | 15 | `te::dispatch::mxfp8::quantize_mxfp8_kernel` |
| 0.631 | 2.4% | 2,304 | 274 | `nvjet_sm100_qqtst 128x256_128x6_2x2_2cta_h_bz UE8M0 NNT` |
| 0.612 | 2.3% | 3,840 | 159 | `at::elementwise BinaryFunctor` |
| 0.586 | 2.2% | 1,024 | 573 | `cudnn sdpa_sm100_flash_fprop` (MLA fwd) |
| 0.462 | 1.8% | 2,816 | 164 | `at::elementwise` |
| 0.422 | 1.6% | 2,560 | 165 | `at::elementwise sigmoid` |
| 0.374 | 1.4% | 10,496 | 36 | `at::vectorized_elementwise_kernel BFloat16 add` |
| 0.318 | 1.2% | 2,048 | 155 | `rmsnorm_bwd_general_kernel` |
| 0.238 | 0.9% | 1,024 | 233 | `at::elementwise` (float broadcast) |

---

## 4. Why rank 0 can't tell the whole story

DSV3's architecture has **dense MLA for the first 3 layers**, then **MoE starting at
layer 4** (60+ MoE layers). With PP=16 (~4 layers per stage), **rank 0 (stage 0) holds
the 3 dense layers + 1 MoE layer** — a much lighter MoE footprint than middle stages,
which run ~4 MoE layers each.

What rank 0 *does* show:
- Pipeline bubble characteristic of 1F1B stage 0 (large)
- Embedding forward + logit-adjacent tail work (MTP head if applicable)
- Dense MLA fwd/bwd at full compute intensity

What rank 0 *does not* show well:
- MoE alltoall traffic / routing cost (EP=8 intra-node hybrid)
- Middle-stage steady-state compute-to-comm balance
- Last-stage output/loss + MTP overhead

**Recommendation:** Re-profile with `profile_ranks=[0,128,480]` to capture stage 0, a
middle stage (PP stage 4 ≈ rank 128), and the last stage (PP stage 15 ≈ rank 480+).
Expected trace total: ~750 MB gz.

---

## 5. Preliminary optimization targets

Ranked by expected impact on rank 0 (with caveat that middle-stage profile may shift
priorities):

1. **Pipeline bubble reduction (43% on rank 0):** Not actionable without architectural
   changes — VP=None already enforced; interleaved 1F1B would shuffle bubble but not
   eliminate it. DSV3 recipe specifies `pipeline_model_parallel_layout` tuned for
   stage balancing; changes may regress other stages.
2. **Elementwise traffic (11.8%):** Look for redundant casts around FP8 boundaries and
   activation checkpoint saves. The `mla_up_proj` recompute policy writes and reads
   activation tensors — costly in memory bandwidth on B200.
3. **Try CUDA Graphs:** Qwen3 235B FP8 MX gained **+10%** from CUDA graphs; could
   similarly help DSV3 by eliminating per-kernel launch gaps (launch overhead is
   ~200 ns × 220 k kernels ≈ 44 ms, 0.15% — not the dominant effect, but graph capture
   also collapses elementwise fusions).
4. **MXFP8 quantize (2.9%):** Small win available if TE upgrades fuse quantization
   with preceding cast/layout ops.

---

## 6. Artifacts

| File | Path |
|---|---|
| Trace (rank 0) | `/mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3/profiler_traces/dsv3_fp8mx_baseline/use3a-ss-b200-gpu-130.cloud.together.ai_rank0.pt.trace.json.gz` |
| TensorBoard events | Same dir, `events.out.tfevents.*.use3a-ss-b200-gpu-256.*` |
| sbatch script | `/mnt/vast/johnson/scripts/dsv3_512gpus_fp8mx_profiler/sbatch.sh` |
| Inner run script | `/mnt/vast/johnson/scripts/dsv3_512gpus_fp8mx_profiler/run_profiler.sh` |
| Patched `profiling.py` (CUDA activities) | `/mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3/profiler_traces/profiling.py` (copy of `tests/b200/tools/profiling.py`) |
| Analysis script | `/tmp/analyze_trace.py` |
| Raw category output | `/tmp/trace_analysis.txt` |

### How to view the trace

```
# Copy to workstation
scp <johnson@cluster>:/mnt/vast/johnson/.../dsv3_fp8mx_baseline/use3a*rank0.pt.trace.json.gz .

# Option A: perfetto.dev/#!/viewer — drag-drop .gz file
# Option B: chrome://tracing — Load, select .gz (slower)
# Option C: TensorBoard profiler plugin (if desired):
#   tensorboard --logdir /mnt/vast/.../dsv3_fp8mx_baseline/
```

---

## 7. Open questions / follow-ups

- [ ] Multi-rank profile (stage 0 / 4 / 15) to quantify MoE alltoall and middle-stage
      balance
- [ ] Try CUDA Graphs for DSV3 FP8 MX (follow Qwen3's `--cuda_graph_impl
      transformer_engine --cuda_graph_scope moe_router,moe_preprocess`)
- [ ] Investigate elementwise 12% — breakdown by op class, determine if checkpoint
      policy change helps
- [ ] Compare FP8 MX vs BF16 rank-0 trace once a BF16 DSV3 512-GPU profile exists
