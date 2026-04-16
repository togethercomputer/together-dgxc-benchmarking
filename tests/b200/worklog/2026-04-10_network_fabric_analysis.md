# B200 Cluster Network Fabric Analysis

**Date:** 2026-04-10
**Cluster:** Together AI B200 (75 nodes, 8x B200 GPUs/node, InfiniBand NDR)
**Workload:** Llama 3.1 405B pretraining with NVFP4 quantization, 256 GPUs (32 nodes)

## Executive Summary

Training throughput for Llama 3.1 405B NVFP4 on this cluster reaches **1,868 TFLOP/s/GPU** — approximately 6% below the 1,990 TFLOP/s/GPU achieved on a reference cluster with the same model and GPU count. PyTorch profiler analysis and NCCL point-to-point benchmarks identify **inter-node communication bandwidth degradation at scale** as the primary bottleneck. P2P bandwidth drops from ~42 GB/s at 2-4 nodes to ~14.5 GB/s at 16-32 nodes — a 66% reduction that directly impacts pipeline-parallel and data-parallel communication.

## Evidence

### 1. NCCL SendRecv (Point-to-Point) Benchmark

All-pairs SendRecv benchmark using `nccl-tests`, message size 512MB, measured across multiple node counts:

| Nodes | Avg busBW (GB/s) | Scaling Efficiency | Step Drop |
|-------|-------------------|-------------------|-----------|
| 2 | 42.50 | 100% (baseline) | — |
| 4 | 44.16 | 104% | — |
| 8 | 26.13 | 61.5% | -40.8% from 4N |
| 16 | 14.47 | 34.0% | -44.6% from 8N |
| 32 | 14.50 | 34.1% | +0.2% from 16N |

**Observations:**
- 2-4 nodes: Full bandwidth (~42-44 GB/s). IB links are not contended.
- 4→8 nodes: Sharp cliff — 40.8% bandwidth drop. This is the critical transition.
- 8→16 nodes: Another 44.6% drop. Cumulative 66% loss from baseline.
- 16→32 nodes: Bandwidth plateaus at ~14.5 GB/s. The fabric is saturated.

### 2. All-Reduce Benchmark (Observed Separately)

All-reduce bandwidth also showed significant degradation when scaling beyond 4 nodes. This impacts data-parallel gradient synchronization.

### 3. Training Profiler Traces — PP=16 (32-node pipeline)

PyTorch profiler on Llama 3.1 405B NVFP4 with TP=4, PP=16, CP=1, VP=8, GBS=1536 (job 79847):

- **P2P SendRecv time: ~56s per training step** (dominant communication cost)
- **Pipeline bubble fraction: 43%**
- **Throughput: 1,352 TFLOP/s/GPU** (32% below reference)
- Pipeline spans all 32 nodes, operating in the degraded bandwidth regime (~14.5 GB/s)

### 4. Training Profiler Traces — PP=8 (fewer nodes per pipeline)

After reducing pipeline parallelism from PP=16 to PP=8 (TP=4, PP=8, CP=1, VP=4, GBS=1536, job 79904):

- **P2P SendRecv time: ~9.1s per training step** (84% reduction from PP=16)
- **Pipeline bubble fraction: 15.5%**
- **DP gradient sync (ReduceScatter + AllGather): ~8.6s per step**
- **Throughput: 1,868 TFLOP/s/GPU** (+38% over PP=16)
- Pipeline operates with fewer cross-node hops, partially avoiding the worst of the bandwidth degradation

### 5. Correlation Between Benchmark and Training Results

| Configuration | Network Regime | Effective P2P BW | Training Throughput |
|---------------|---------------|-------------------|-------------------|
| PP=16, 32 nodes | Saturated (~14.5 GB/s) | Low | 1,352 TFLOP/s/GPU |
| PP=8, 32 nodes* | Mixed (~26 GB/s for PP, degraded for DP) | Medium | 1,868 TFLOP/s/GPU |
| Reference cluster | Healthy (assumed full BW) | High | 1,990 TFLOP/s/GPU |

*PP=8 with DP=8: pipeline P2P stays within smaller node groups, but DP all-reduce still spans all 32 nodes.

### 6. Additional Failed Configurations

| Config | Result | Network Implication |
|--------|--------|-------------------|
| TP=8, PP=8 | Hung during NCCL init | TP=8 requires cross-node tensor parallelism — all-reduce on degraded fabric |
| CP=2, PP=8 | 119.5s/step, 1,038 TFLOP/s/GPU (-44% vs PP=8) | Context parallelism adds cross-node ring-attention communication |

Both failures are consistent with cross-node communication being the limiting factor.

## Remaining Performance Gap

With the best configuration (PP=8), the per-step time breakdown shows:

| Component | Time/Step | Notes |
|-----------|-----------|-------|
| Compute | ~21.7s | NVFP4 GEMM, attention, etc. — not network-bound |
| P2P pipeline comm | ~9.1s | Would be ~5.6s at healthy bandwidth (42 GB/s) |
| DP gradient sync | ~8.6s | Would be ~5.3s at healthy bandwidth |
| GPU idle/bubble | ~10.4s | Partially caused by comm latency |
| Overlap (hidden) | ~26.6s | Compute-comm overlap is working |
| **Total** | **~66.5s** | |

Estimated savings with healthy fabric bandwidth: ~6.8s/step → ~59.7s/step → ~2,080 TFLOP/s/GPU. This would close or exceed the gap to the 1,990 reference.

## Diagnostic Questions for Cluster Team

1. **Fabric topology:** What is the oversubscription ratio at each switch tier (leaf-to-spine, spine-to-core)? The 4→8 node bandwidth cliff suggests significant oversubscription above the leaf level.

2. **IB routing:** Is adaptive routing enabled? Are there known hotspots or congested links?

3. **SHARP:** The SHARP Aggregation Manager currently has no reservations for training workloads (`SHARP Job init error: No reservation`). Can SHARP AM be configured for the batch partition? This would directly accelerate ReduceScatter and AllGather operations.

4. **Node placement:** Is there a way to request topology-aware allocation (e.g., Slurm topology plugin) to keep 32-node jobs within the same switch subtree?

5. **Multi-rail configuration:** Nodes have 8 IB HCAs (`mlx5_0,1,4,5,6,11,14,15`). Are these evenly distributed across switch planes?

## Appendix: Experiment Details

All experiments used:
- Container: nvidia+nemo+26.02.00
- Model: Llama 3.1 405B, NVFP4 quantization, seq_length=8192
- 32 nodes (256 GPUs), MBS=1, GBS=1536
- NCCL settings: NCCL_NVLS_ENABLE=0, NCCL_TIMEOUT=1800, TORCH_NCCL_HIGH_PRIORITY=1

Full training results and profiler traces are available at:
- Training report: `/home/johnson/johnson/405b_nvfp4_256gpu_results_report.md`
- Profiler trace: `/mnt/vast/llmb_/workloads/pretrain_llama3.1/profiler_traces/405b_nvfp4_tp4_pp8_configD/`
- P2P benchmark logs: NCCL SendRecv test jobs 79867-79891
