# Qwen3 235B A22B BF16 — Nsys Bottleneck Report

**Job**: 79956 | **Nodes**: 32 (256 B200 GPUs) | **Container**: NeMo 26.02  
**Parallelism**: TP=1, PP=8, CP=1, EP=8, VP=4, MBS=1, GBS=8192, DP=32  
**Profiled steps**: 5–7 (3 iterations, 4 warmup steps)  
**Profiled ranks**: 0 (PP0), 4 (PP4), 7 (PP7) — all on node 0  
**Topology**: PP intra-node (NVLink), EP and DP inter-node (InfiniBand)

---

## 1. Executive Summary

Communication dominates GPU time on ranks 0 and 4 (~73%), driven primarily by **PP SendRecv** (57%) and **EP AllGather** (16%). Rank 7 (last PP stage) is significantly less communication-bound (42%) due to asymmetric PP traffic. The overall communication-to-compute ratio (~73% comm / 27% compute on early PP stages) is **worse than the 405B NVFP4 baseline** (67% comm / 33% compute), likely because the MoE architecture adds EP all-to-all communication on top of PP overhead.

---

## 2. Compute vs Communication Split

| Metric | Rank 0 (PP0) | Rank 4 (PP4) | Rank 7 (PP7) |
|--------|-------------|-------------|-------------|
| Total GPU kernel time | 282.2 s | 271.3 s | 118.0 s |
| Compute kernels | 74.7 s (26.5%) | 73.8 s (27.2%) | 69.0 s (58.5%) |
| NCCL communication | 207.4 s (73.5%) | 197.5 s (72.8%) | 49.0 s (41.5%) |

**Key takeaway**: Ranks 0 and 4 spend nearly 3x more time in communication than compute. Rank 7 is more balanced but still 42% communication.

---

## 3. NCCL Communication Breakdown

### 3.1 Per-Collective Time

| Collective | Rank 0 | Rank 4 | Rank 7 | Purpose |
|-----------|--------|--------|--------|---------|
| **SendRecv** | 160.6 s (56.9%) | 154.4 s (56.9%) | 45.8 s (38.8%) | PP pipeline communication |
| **AllGather (RING_LL)** | 46.4 s (16.4%) | 42.6 s (15.7%) | 2.7 s (2.3%) | EP expert routing (all-to-all) |
| **ReduceScatter** | 0.45 s (0.2%) | 0.45 s (0.2%) | 0.41 s (0.4%) | DP gradient sync |
| **AllReduce (f32)** | 0.08 s | 0.08 s | 0.07 s | Grad norm / loss reduction |
| **AllReduce (u32)** | ~0 s | ~0 s | ~0 s | Token counting |

### 3.2 PP SendRecv Deep Dive

- **61,440 instances per rank** across 3 profiled iterations
- Rank 0/4: avg=2.5–2.6 ms, median=1.8 ms (NVLink P2P)
- Rank 7: avg=0.7 ms, median=0.4 ms — **3.5x faster** because PP7 only sends backwards (no forward send beyond the pipeline)
- SendRecv alone accounts for **57% of all GPU kernel time** on ranks 0 and 4

### 3.3 EP AllGather (Expert Routing)

This is the MoE-specific bottleneck — inter-node AllGather over InfiniBand for expert dispatch/combine.

- Rank 0: 46.4 s (8,472 instances, avg=5.5 ms)
- Rank 4: 42.6 s (8,472 instances, avg=5.0 ms)
- Rank 7: 2.7 s (8,472 instances, avg=0.3 ms) — **17x less time**

The dramatic difference for rank 7 suggests the last PP stage handles fewer MoE layers (or layers with fewer expert dispatches). With VP=4, the virtual pipeline stage assignment is non-uniform across PP stages.

### 3.4 DP Gradient Sync

ReduceScatter is only 0.4–0.5 s across all ranks — **negligible**. The DP gradient sync is well-overlapped with backward computation, confirmed by the NVTX data showing ReduceScatter as only 24 instances (8 per profiled step) with avg ~19 ms each.

---

## 4. PP Bubble Analysis

The NVTX layer ranges show interleaved forward-backward scheduling (1F1B pattern with VP=4):

| Range | Rank 0 (avg) | Rank 4 (avg) | Rank 7 (avg) |
|-------|-------------|-------------|-------------|
| `layer_0f-layer_2b` | 23.2 ms | 23.4 ms | 22.8 ms |
| `layer_1f-layer_1b` | 21.2 ms | 21.4 ms | 21.8 ms |
| `layer_2f-layer_0b` | 21.9 ms | 22.0 ms | 21.9 ms |
| `layer_0f-layer_1b` | 26.7 ms | 26.2 ms | 25.1 ms |
| `layer_1f-layer_0b` | 20.1 ms | 20.6 ms | 21.5 ms |

The PP stages are reasonably balanced (21–27 ms range per interleaved chunk). The variation is ~30%, primarily in `layer_0f-layer_1b` which includes warmup/startup overhead.

PP bubble is intra-node over NVLink, which gives low-latency SendRecv. However, the sheer volume (61,440 instances per rank) means even 1.8 ms average adds up to 160+ seconds of GPU time.

---

## 5. Per-PP-Stage Compute Imbalance

| Component | Rank 0 (PP0) | Rank 4 (PP4) | Rank 7 (PP7) | Delta (0 vs 7) |
|-----------|-------------|-------------|-------------|----------------|
| Compute kernels | 74.7 s | 73.8 s | 69.0 s | +8.3% |
| attn forward | 35.3 s | 35.0 s | 38.9 s | -9.3% |
| attn backward | 28.8 s | 31.7 s | 31.8 s | -9.4% |
| mlp forward | 23.7 s | 23.6 s | 24.8 s | -4.5% |
| mlp backward | 28.3 s | 32.3 s | 31.9 s | -11.3% |
| moe_dispatch fwd | 27.0 s | 18.1 s | 9.1 s | +197% |
| moe_combine fwd | 5.4 s | 5.4 s | 6.0 s | -10.0% |

**Key finding**: `moe_dispatch forward` is 3x more expensive on rank 0 (27.0 s) than rank 7 (9.1 s). This is because moe_dispatch includes the EP all-to-all communication that routes tokens to expert ranks — rank 0/4 handle more MoE layers per VP assignment. Pure compute (attn, mlp) is within 10% across stages, showing good compute balance.

---

## 6. Top Compute Kernels

The compute portion is dominated by GEMM kernels (nvjet) and attention (cuDNN Flash Attention):

| Kernel | Time (Rank 0) | % of Total | Type |
|--------|-------------|-----------|------|
| `nvjet_sm100_tst_128x256_64x6_2x2_2cta_h_badd_NTT` | 6.4 s | 2.3% | GEMM (MoE FFN) |
| `cudnn_flash_bprop` | 6.2 s | 2.2% | Attention backward |
| `nvjet_sm100_tst_176x128_64x8_1x2_2cta_h_bz_TNN` | 4.9 s | 1.7% | GEMM |
| `nvjet_sm100_tst_128x192_64x6_2x2_2cta_h_badd_NTT` | 4.0 s | 1.4% | GEMM (MoE FFN) |
| `_sort_chunks_by_map_kernel` | 2.7 s | 1.0% | MoE token routing |
| `cudnn_flash_fprop` | 2.0 s | 0.7% | Attention forward |
| `triton_fused_add_cat_mul_rsub_sigmoid_silu_split` | 1.9 s | 0.7% | MoE gating |

The `_sort_chunks_by_map_kernel` and `_permute/_unpermute_kernel` are MoE-specific overhead for token routing (sort + permute), totaling ~4.7 s (1.7%) on rank 0.

---

## 7. Memory Operations

Memory copy/set time is negligible across all ranks (~0.4 s total per rank):

| Operation | Rank 0 | Rank 4 | Rank 7 |
|-----------|--------|--------|--------|
| D2D memcpy | 161 ms | 150 ms | 140 ms |
| D2H memcpy | 154 ms | 147 ms | 150 ms |
| H2D memcpy | 45 ms | 44 ms | 44 ms |
| memset | 44 ms | 43 ms | 51 ms |

Not a bottleneck.

---

## 8. Comparison with 405B NVFP4

| Metric | Qwen3 235B BF16 (this) | Llama 405B NVFP4 |
|--------|----------------------|------------------|
| Communication % | 73% (rank 0/4) | 67% |
| Dominant collective | SendRecv (57%) | SendRecv |
| EP all-to-all | 16% (MoE-specific) | N/A (dense) |
| DP gradient sync | Negligible (0.2%) | Negligible |
| PP topology | Intra-node (NVLink) | Inter-node (IB) |

The Qwen3 MoE model has ~6% more communication overhead than the dense 405B, entirely attributable to EP all-to-all for expert routing. However, PP is intra-node (NVLink) vs inter-node for 405B, which keeps per-SendRecv latency low.

---

## 9. Optimization Recommendations

1. **Reduce EP all-to-all overhead (16% of GPU time)**: Consider EP=4 with larger expert groups to reduce inter-node traffic, or overlap EP communication with compute using pipelined expert dispatch.

2. **PP SendRecv dominance (57%)**: With VP=4 creating many micro-batches, each requiring PP communication, the high instance count (61K per rank) drives the total. Reducing VP or increasing MBS could help amortize PP overhead.

3. **MoE routing overhead**: `_sort_chunks_by_map_kernel` + `_permute/_unpermute` consume 1.7% — minor but could benefit from fused kernels.

4. **PP stage imbalance for MoE dispatch**: Rank 0 spends 27 s in moe_dispatch vs 9 s for rank 7. Rebalancing which virtual stages map to which PP ranks could equalize this.

---

*Generated from nsys traces: profile_79956_node0_gpu{0,4,7}.nsys-rep*  
*Report date: 2026-04-10*
