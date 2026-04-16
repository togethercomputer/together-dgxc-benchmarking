**Llama 3.1 405B NVFP4 — 512 GPU (64 Node) Benchmark Report**

Cluster: Together AI B200 (use3a), 64 nodes x 8 B200 GPUs
Container: nvidia+nemo+26.02.00
Date: 2026-04-10

---

## Objective

Scale the 405B NVFP4 benchmark from 256 GPUs (32 nodes) to 512 GPUs (64 nodes). Compare Config A (NVIDIA reference, PP=16) and Config D (optimized, PP=8) at the larger scale to confirm the PP=8 advantage holds and measure the impact of increased cross-node communication.

## Configurations

| Parameter | Config A (baseline) | Config D (optimized) |
|-----------|--------------------|--------------------|
| TP | 4 | 4 |
| PP | 16 | 8 |
| CP | 1 | 1 |
| VP | 8 | 4 |
| MBS | 1 | 1 |
| GBS | 1536 | 1536 |
| DP | 8 | 16 |
| Nodes | 64 | 64 |
| GPUs | 512 | 512 |
| Job ID | 79952 | 79953 |

## Results

### Config A — TP=4, PP=16, VP=8 (Job 79952)

| Iteration | Step Time (s) | TFLOP/s/GPU |
|-----------|--------------|-------------|
| 1 (warmup) | 502.13 | 123.5 |
| 2 | 44.57 | 1,391.6 |
| 3 | 44.66 | 1,388.8 |
| 4 | 44.74 | 1,386.4 |
| 5 | 44.66 | 1,388.8 |
| 6 | 44.77 | 1,385.5 |
| 7 | 44.75 | 1,386.3 |
| 8 | 45.28 | 1,369.9 |
| 9 | 44.63 | 1,389.7 |
| 10 | 45.27 | 1,370.2 |
| **Avg (2-10)** | **44.8** | **1,384** |

### Config D — TP=4, PP=8, VP=4 (Job 79953)

| Iteration | Step Time (s) | TFLOP/s/GPU |
|-----------|--------------|-------------|
| 1 (warmup) | 292.47 | 212.1 |
| 2 | 34.10 | 1,819.2 |
| 3 | 34.15 | 1,816.2 |
| 4 | 34.10 | 1,819.3 |
| 5 | 34.16 | 1,815.9 |
| 6 | 34.22 | 1,812.7 |
| 7 | 34.20 | 1,813.8 |
| 8 | 34.16 | 1,815.8 |
| 9 | 34.21 | 1,812.9 |
| 10 | 33.84 | 1,833.0 |
| **Avg (2-10)** | **34.1** | **1,818** |

## 256 vs 512 GPU Comparison

| Config | 256 GPU TFLOP/s/GPU | 512 GPU TFLOP/s/GPU | Delta |
|--------|--------------------|--------------------|-------|
| A (PP=16) | 1,352 | 1,384 | +2.4% |
| D (PP=8) | 1,868 | 1,818 | -2.7% |
| **D vs A gain** | **+38.1%** | **+31.3%** | |

Reference target (healthy fabric): ~1,990 TFLOP/s/GPU

## Analysis

**Config D (PP=8) outperforms Config A (PP=16) by 31% at 512 GPUs**, consistent with the 38% gain observed at 256 GPUs.

**Config A improved slightly at scale (+2.4%).** Doubling GPUs doubled DP from 4 to 8, halving the number of micro-batches per pipeline stage per step (from 1536/4=384 to 1536/8=192). This reduced per-step wall time from 91.75s to 44.8s. The slight per-GPU throughput increase suggests reduced pipeline bubble fraction from the shorter pipeline fill/drain relative to compute.

**Config D dropped slightly at scale (-2.7%).** DP increased from 8 to 16, meaning the DP gradient all-reduce now spans more nodes. With 16-way DP across 64 nodes, more of the all-reduce traffic hits the degraded spine bandwidth (14.5 GB/s at 32+ nodes per P2P benchmarks). This added ~0.5s of overhead per step compared to 256 GPUs.

**The PP=8 advantage narrows from 38% to 31% at scale.** This is expected: Config A benefits from more DP (shorter pipeline), while Config D's DP gradient sync cost grows with more participants across the degraded fabric.

**First-step warmup was significantly longer at 512 GPUs.** Config A: 502s (vs ~170s at 256 GPU). Config D: 292s. The increase is from lazy NCCL P2P communicator creation across 64 nodes — PP=16 requires more communicators than PP=8, hence the larger warmup.

## Network Bottleneck Confirmation

Both configs remain well below the reference target of ~1,990 TFLOP/s/GPU. The results at 512 GPUs reinforce the earlier finding: the IB fabric oversubscription at the spine layer is the primary throughput limiter. P2P benchmarks showed bandwidth dropping from ~42 GB/s (2-4 nodes) to ~14.5 GB/s (16-32 nodes) — a 66% degradation that directly impacts both pipeline P2P and DP all-reduce communication.

With full fabric bandwidth, estimated throughput at 512 GPUs would reach ~2,000+ TFLOP/s/GPU for Config D.
