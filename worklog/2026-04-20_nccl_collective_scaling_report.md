# NCCL Collective Scaling Benchmark Report — 2026-04-20

NCCL collective performance across 1–64 node scale (8–512 GPUs) on the B200 DGXC cluster, testing all_reduce, all_gather, and sendrecv P2P across five algorithm configurations and three message sizes. Compared against the 2026-04-17 baseline (`20260417_173551`) to confirm cluster stability.

## Run Notes

- **Submitted**: 71 jobs (IDs 84286–84356) via `submit_all.sh`; `MAX_BYTES` bumped from 8G→16G for this run (raw data captured, parser tables still show 2/4/8 GB).
- **Teardown hangs**: 6 jobs completed the benchmark but hung in MPI/UCX `finalize` for 10+ min — `scancel`'d (data preserved). Pattern: 4n/8n ring/nvls/collnet_sharp — transient UCX connection-reset during shutdown.
- **Resubmit round 1** (`resubmit_failed.sh`, IDs 84357–84363): 3/7 succeeded. 4 `all_reduce` jobs failed with `NCCL WARN Unrecognized element token "Ring$nexport"` due to a bash-quoting bug in the resubmit script's env emission (`${algo_env//;/$'\n'}` inside an unquoted heredoc).
- **Resubmit round 2** (IDs 84364–84367) after `printf`-based env emission: all 4 succeeded.
- **Final**: 59 OK / 12 expected failures (AllGather doesn't support tree/nvls_tree) / 0 unexpected failures.
- **Excluded nodes today**: `use3a-ss-b200-gpu-[130,190,197,199,201,211,233,239]` (node 130 lacks `/opt/hpcx`).

## Headline

**Cluster is performance-stable** — all deltas vs 2026-04-17 are within ±5%. Largest regression: `all_gather ring 8n` at -3.8%. Largest improvement: `all_reduce nvls 8n` at +3.1%. Headline rankings unchanged: CollNet SHARP best at 4+ nodes (~384 GB/s all_reduce), NVLS best intra-node (834 GB/s), tree consistently worst (~192 GB/s ceiling).

## System Configuration

| Component | Details |
|-----------|---------|
| GPU | NVIDIA B200, 8 per node |
| Nodes | 1–64 (67 available in batch) |
| Total GPUs | 8–512 |
| Interconnect | Mellanox ConnectX-7 (MT4129), NDR 400 Gb/s InfiniBand |
| NICs per node | 8 HCAs (mlx5_0, mlx5_1, mlx5_4, mlx5_5, mlx5_6, mlx5_11, mlx5_14, mlx5_15) |
| NVLink | 5th gen, ~900 GB/s bidirectional intra-node |
| SHARP | Not available (sharpd not running, no reservations) |
| Excluded | use3a-ss-b200-gpu-[190,197,199,201,211,233,239] |

## Software Configuration

| Component | Details |
|-----------|---------|
| NCCL | 2.29.7+cuda12.9 |
| NCCL Library | `/mnt/vast/dgxc-benchmarking-auto/nccl-libs/usr/lib/x86_64-linux-gnu` |
| NCCL Test Binaries | `/mnt/vast/dgxc-benchmarking-auto/nccl-tests/build` |
| MPI | PMIx (`srun --mpi=pmix`) |
| HPC-X | `/opt/hpcx/hpcx-init.sh` |
| Scheduler | Slurm, partition=batch, exclusive allocation |

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Message sizes | 2 GB, 4 GB, 8 GB (2x step factor) |
| Iterations | 5 warmup + 20 measured |
| Data type | float32 |
| Reduction op | sum (all_reduce) / none (all_gather, sendrecv) |
| Validation | enabled (Out of bounds check) |
| GPUs per rank | 1 (`-g 1`) |

## Test Matrix

71 total jobs — 3 operations × 5 algorithms × 7 node counts, minus 1-node inter-node-only configs:

| Config Label | NCCL_ALGO | NCCL_NVLS_ENABLE | NCCL_COLLNET_ENABLE | Nodes |
|--------------|-----------|------------------|---------------------|-------|
| ring | Ring | 0 | 0 | 1–64 |
| nvls | (auto) | 1 | 0 | 1–64 |
| tree | Tree | 0 | 0 | 2–64 |
| nvls_tree | NVLSTree | 1 | 0 | 2–64 |
| collnet_sharp | (auto) | 0 | 1 | 2–64 |

**Expected failures:** tree and nvls_tree are not supported by NCCL for AllGather (12 jobs failed as expected).

**Job results:** 59 OK, 12 expected failures, 0 unexpected failures.

---

## Results — all_reduce Peak busBW (GB/s)

### all_reduce @ 2 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree | Best |
|--------------|------|------|---------------|----------|------|------|
| 1n (8) | 666.8 | **816.4** | -- | -- | -- | NVLS |
| 2n (16) | 342.4 | 686.3 | 376.9 | **687.3** | 330.1 | NVLSTree |
| 4n (32) | 329.0 | 312.5 | **379.0** | 257.8 | 184.8 | CollNet SHARP |
| 8n (64) | 311.6 | 320.7 | **382.5** | 255.7 | 189.7 | CollNet SHARP |
| 16n (128) | 287.4 | 251.9 | **359.6** | 253.2 | 187.4 | CollNet SHARP |
| 32n (256) | 269.4 | 251.2 | **336.2** | 250.8 | 187.1 | CollNet SHARP |
| 64n (512) | 223.8 | 247.8 | **274.9** | 247.9 | 186.6 | CollNet SHARP |

### all_reduce @ 4 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree | Best |
|--------------|------|------|---------------|----------|------|------|
| 1n (8) | 672.7 | **831.2** | -- | -- | -- | NVLS |
| 2n (16) | 343.7 | **705.7** | 372.5 | 705.6 | 335.3 | NVLS |
| 4n (32) | 332.2 | 317.0 | **384.2** | 266.6 | 186.8 | CollNet SHARP |
| 8n (64) | 309.7 | 321.4 | **378.4** | 266.0 | 192.3 | CollNet SHARP |
| 16n (128) | 307.6 | 312.5 | **384.4** | 263.9 | 190.4 | CollNet SHARP |
| 32n (256) | 289.8 | 265.0 | **361.0** | 264.5 | 191.0 | CollNet SHARP |
| 64n (512) | 269.3 | 263.7 | **336.5** | 263.5 | 190.5 | CollNet SHARP |

### all_reduce @ 8 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree | Best |
|--------------|------|------|---------------|----------|------|------|
| 1n (8) | 674.6 | **834.4** | -- | -- | -- | NVLS |
| 2n (16) | 344.4 | **713.2** | 378.1 | 713.1 | 338.1 | NVLS |
| 4n (32) | 333.0 | 319.3 | **384.4** | 267.2 | 188.1 | CollNet SHARP |
| 8n (64) | 308.4 | 322.2 | **385.2** | 266.4 | 193.8 | CollNet SHARP |
| 16n (128) | 307.9 | 308.1 | **379.9** | 265.4 | 192.0 | CollNet SHARP |
| 32n (256) | 310.0 | 306.8 | **385.2** | 266.7 | 192.9 | CollNet SHARP |
| 64n (512) | 288.3 | 264.1 | **363.6** | 266.6 | 192.6 | CollNet SHARP |

---

## Results — all_gather Peak busBW (GB/s)

tree and nvls_tree are not supported by NCCL for AllGather.

### all_gather @ 2 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |
|--------------|------|------|---------------|------|
| 1n (8) | 636.4 | **640.4** | -- | NVLS |
| 2n (16) | 338.7 | 338.9 | **357.5** | CollNet SHARP |
| 4n (32) | 325.7 | 326.0 | **371.1** | CollNet SHARP |
| 8n (64) | 313.2 | 311.7 | **378.2** | CollNet SHARP |
| 16n (128) | 288.0 | 287.3 | **354.6** | CollNet SHARP |
| 32n (256) | 237.0 | 236.7 | **284.3** | CollNet SHARP |
| 64n (512) | 224.7 | 224.9 | **277.0** | CollNet SHARP |

### all_gather @ 4 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |
|--------------|------|------|---------------|------|
| 1n (8) | 648.2 | **654.4** | -- | NVLS |
| 2n (16) | 341.6 | 341.7 | **367.6** | CollNet SHARP |
| 4n (32) | 328.8 | 330.3 | **381.9** | CollNet SHARP |
| 8n (64) | 316.0 | 314.0 | **378.3** | CollNet SHARP |
| 16n (128) | 317.2 | 316.3 | **382.3** | CollNet SHARP |
| 32n (256) | 286.2 | 286.9 | **357.8** | CollNet SHARP |
| 64n (512) | 234.1 | 236.9 | **282.2** | CollNet SHARP |

### all_gather @ 8 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |
|--------------|------|------|---------------|------|
| 1n (8) | 654.2 | **660.8** | -- | NVLS |
| 2n (16) | 343.1 | 343.1 | **362.8** | CollNet SHARP |
| 4n (32) | 334.2 | 332.7 | **382.8** | CollNet SHARP |
| 8n (64) | 312.7 | 315.5 | **384.1** | CollNet SHARP |
| 16n (128) | 310.9 | 313.1 | **378.8** | CollNet SHARP |
| 32n (256) | 317.2 | 312.5 | **384.1** | CollNet SHARP |
| 64n (512) | 286.2 | 286.1 | **362.2** | CollNet SHARP |

---

## Results — sendrecv P2P Peak busBW (GB/s)

| Nodes (GPUs) | 2 GB | 4 GB | 8 GB |
|--------------|------|------|------|
| 1n (8) | 646.2 | 651.7 | **655.5** |
| 2n (16) | 43.4 | 43.4 | **43.4** |
| 4n (32) | 43.0 | 43.0 | **43.0** |
| 8n (64) | 25.0 | 25.0 | **25.0** |
| 16n (128) | **15.3** | **15.3** | **15.3** |
| 32n (256) | **15.3** | **15.3** | **15.3** |
| 64n (512) | 15.7 | **15.7** | **15.7** |

---

## Scaling Efficiency — all_reduce @ 8 GB

Efficiency = busBW(N nodes) / busBW(baseline) × 100%. Baseline: 1-node for ring/nvls; 2-node for others.

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|--------------|------|------|---------------|----------|------|
| 1n (8) | **100.0%** | 100.0% | -- | -- | -- |
| 2n (16) | 51.1% | 85.5% | **100.0%** | 100.0% | 100.0% |
| 4n (32) | 49.4% | 38.3% | **101.7%** | 37.5% | 55.6% |
| 8n (64) | 45.7% | 38.6% | **101.9%** | 37.4% | 57.3% |
| 16n (128) | 45.6% | 36.9% | **100.5%** | 37.2% | 56.8% |
| 32n (256) | 46.0% | 36.8% | **101.9%** | 37.4% | 57.1% |
| 64n (512) | 42.7% | 31.7% | **96.2%** | 37.4% | 57.0% |

## Scaling Efficiency — all_gather @ 8 GB

Efficiency = busBW(N nodes) / busBW(baseline) × 100%. Baseline: 1-node for ring/nvls; 2-node for others.

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP |
|--------------|------|------|---------------|
| 1n (8) | **100.0%** | 100.0% | -- |
| 2n (16) | 52.4% | 51.9% | **100.0%** |
| 4n (32) | 51.1% | 50.3% | **105.5%** |
| 8n (64) | 47.8% | 47.7% | **105.9%** |
| 16n (128) | 47.5% | 47.4% | **104.4%** |
| 32n (256) | 48.5% | 47.3% | **105.9%** |
| 64n (512) | 43.7% | 43.3% | **99.8%** |

## Scaling Efficiency — sendrecv P2P @ 8 GB

| Nodes (GPUs) | Efficiency |
|--------------|-----------|
| 1n (8) | 100.0% |
| 2n (16) | 6.6% |
| 4n (32) | 6.6% |
| 8n (64) | 3.8% |
| 16n (128) | 2.3% |
| 32n (256) | 2.3% |
| 64n (512) | 2.4% |

---

## Step-by-Step Bandwidth Drop — all_reduce @ 8 GB

Identifies scaling cliffs between consecutive node counts.

| Transition | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|------------|------|------|---------------|----------|------|
| 1n → 2n | **-48.9%** | -14.5% | -- | -- | -- |
| 2n → 4n | -3.3% | **-55.2%** | +1.7% | **-62.5%** | **-44.4%** |
| 4n → 8n | -7.4% | +0.9% | +0.2% | -0.3% | +3.0% |
| 8n → 16n | -0.1% | -4.4% | -1.4% | -0.4% | -0.9% |
| 16n → 32n | +0.7% | -0.4% | +1.4% | +0.5% | +0.5% |
| 32n → 64n | -7.0% | -13.9% | -5.6% | -0.0% | -0.2% |

---

## Message Size Sensitivity — all_reduce 8G/2G Ratio

How much 8 GB messages outperform 2 GB, revealing bandwidth saturation effects at scale.

| Nodes | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|-------|------|------|---------------|----------|------|
| 1n | 1.01x | 1.02x | -- | -- | -- |
| 2n | 1.01x | 1.04x | 1.00x | 1.04x | 1.02x |
| 4n | 1.01x | 1.02x | 1.01x | 1.04x | 1.02x |
| 8n | 0.99x | 1.00x | 1.01x | 1.04x | 1.02x |
| 16n | 1.07x | 1.22x | 1.06x | 1.05x | 1.02x |
| 32n | 1.15x | 1.22x | 1.15x | 1.06x | 1.03x |
| 64n | 1.29x | 1.07x | **1.32x** | 1.08x | 1.03x |

## Message Size Sensitivity — all_gather 8G/2G Ratio

How much 8 GB messages outperform 2 GB, revealing bandwidth saturation effects at scale.

| Nodes | Ring | NVLS | CollNet SHARP |
|-------|------|------|---------------|
| 1n | 1.03x | 1.03x | -- |
| 2n | 1.01x | 1.01x | 1.01x |
| 4n | 1.03x | 1.02x | 1.03x |
| 8n | 1.00x | 1.01x | 1.02x |
| 16n | 1.08x | 1.09x | 1.07x |
| 32n | **1.34x** | **1.32x** | **1.35x** |
| 64n | 1.27x | 1.27x | **1.31x** |

---

## Algorithm Head-to-Head @ 8 GB

### all_reduce — Best Algorithm per Scale

| Nodes (GPUs) | Best Algorithm | busBW (GB/s) | 2nd Best | busBW (GB/s) | Gap |
|--------------|---------------|-------------|----------|-------------|-----|
| 1n (8) | **NVLS** | 834.4 | Ring | 674.6 | +23.7% |
| 2n (16) | **NVLS** | 713.2 | NVLSTree | 713.1 | +0.0% |
| 4n (32) | **CollNet SHARP** | 384.4 | Ring | 333.0 | +15.4% |
| 8n (64) | **CollNet SHARP** | 385.2 | NVLS | 322.2 | +19.6% |
| 16n (128) | **CollNet SHARP** | 379.9 | NVLS | 308.1 | +23.3% |
| 32n (256) | **CollNet SHARP** | 385.2 | Ring | 310.0 | +24.3% |
| 64n (512) | **CollNet SHARP** | 363.6 | Ring | 288.3 | +26.1% |

### all_gather — Best Algorithm per Scale

| Nodes (GPUs) | Best Algorithm | busBW (GB/s) | 2nd Best | busBW (GB/s) | Gap |
|--------------|---------------|-------------|----------|-------------|-----|
| 1n (8) | **NVLS** | 660.8 | Ring | 654.2 | +1.0% |
| 2n (16) | **CollNet SHARP** | 362.8 | Ring | 343.1 | +5.7% |
| 4n (32) | **CollNet SHARP** | 382.8 | Ring | 334.2 | +14.5% |
| 8n (64) | **CollNet SHARP** | 384.1 | NVLS | 315.5 | +21.7% |
| 16n (128) | **CollNet SHARP** | 378.8 | NVLS | 313.1 | +21.0% |
| 32n (256) | **CollNet SHARP** | 384.1 | Ring | 317.2 | +21.1% |
| 64n (512) | **CollNet SHARP** | 362.2 | Ring | 286.2 | +26.6% |

---

## Interconnect Topology Analysis

### NVLink vs InfiniBand Boundary

| Metric | NVLink 1n (GB/s) | IB 2n (GB/s) | IB 64n (GB/s) | 1n→2n Drop |
|--------|-----------------|-------------|--------------|-----------|
| all_reduce (NVLS) | 834.4 | 713.2 | 264.1 | **-14.5%** |
| all_reduce (Ring) | 674.6 | 344.4 | 288.3 | **-48.9%** |
| sendrecv P2P | 655.5 | 43.4 | 15.7 | **-93.4%** |

### Bandwidth Utilization vs Theoretical @ 2 Nodes, 8 GB

| Collective | Best Algorithm | busBW (GB/s) | IB Theoretical | Utilization |
|------------|---------------|-------------|----------------|-------------|
| all_reduce | NVLS | 713.2 | 400 GB/s (8 HCAs) | 178.3% |
| all_gather | CollNet SHARP | 362.8 | 400 GB/s (8 HCAs) | 90.7% |
| sendrecv | P2P | 43.4 | 50 GB/s (1 HCA) | 86.8% |

---

## Delta vs Previous Run

### all_reduce @ 8 GB — Delta (%)

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|--------------|------|------|---------------|----------|------|
| 1n (8) | -0.3% | -0.3% | -- | -- | -- |
| 2n (16) | +0.1% | -0.0% | +0.7% | -0.0% | +0.9% |
| 4n (32) | +0.0% | +2.5% | +0.0% | +1.4% | +0.2% |
| 8n (64) | +0.3% | +3.1% | -0.0% | +0.7% | +0.6% |
| 16n (128) | +0.9% | +0.3% | +0.1% | +0.0% | -0.2% |
| 32n (256) | +0.1% | -1.4% | +0.0% | -0.3% | +0.5% |
| 64n (512) | +0.1% | -1.0% | -0.2% | -0.3% | +0.3% |

### all_gather @ 8 GB — Delta (%)

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP |
|--------------|------|------|---------------|
| 1n (8) | -0.9% | -0.0% | -- |
| 2n (16) | -0.1% | +0.0% | -0.2% |
| 4n (32) | +0.3% | +0.0% | +0.0% |
| 8n (64) | -3.8% | -1.0% | +0.0% |
| 16n (128) | -0.4% | -0.5% | +0.1% |
| 32n (256) | +0.4% | -0.3% | -0.0% |
| 64n (512) | -0.0% | +0.0% | -0.0% |

### sendrecv P2P @ 8 GB — Delta (%)

| Nodes (GPUs) | Delta |
|--------------|-------|
| 1n (8) | -0.0% |
| 2n (16) | +1.4% |
| 4n (32) | -1.0% |
| 8n (64) | +3.6% |
| 16n (128) | +3.3% |
| 32n (256) | +0.2% |
| 64n (512) | +0.2% |

All results within ±5% of previous run.

---

## Observations

1. **CollNet SHARP is the clear winner at 4+ nodes** across both all_reduce (~384 GB/s) and all_gather, despite SHARP hardware not being operational. NCCL's CollNet fallback outperforms ring by 24% at 32 nodes. All CollNet jobs logged `SHARP coll init error: Cannot create SHARP job` — actual SHARP offload would likely push numbers higher.

2. **NVLS dominates intra-node and 2-node scale** with 834 GB/s all_reduce at 1 node (+24% over ring) and 713 GB/s at 2 nodes. NVLS leverages NVLink multicast for efficient intra-node reduction.

3. **NVLS collapses at 64 nodes**: all_reduce NVLS drops to 264 GB/s — worse than ring (288 GB/s). Don't force `NCCL_NVLS_ENABLE=1` for large-scale training.

4. **CollNet scaling is remarkably flat**: all_reduce CollNet holds ~384 GB/s from 4 to 32 nodes (0% variation), only dropping 6% at 64 nodes.

5. **The NVLink→IB cliff is 15x for P2P**: sendrecv drops from 656 GB/s intra-node to 43.4 GB/s at 2 nodes. P2P is single-HCA limited (43.4 / 50 = 87% utilization). This makes pipeline parallelism (PP) the inter-node bottleneck.

6. **Larger messages improve bandwidth at scale**: ring 64n gains 29% from 2G→8G (224→288 GB/s). This validates GBS tuning (more gradient accumulation) as a communication optimization strategy.

7. **Tree algorithm is consistently worst** for large messages: ~192 GB/s ceiling regardless of node count at 4–64 nodes. Never force `NCCL_ALGO=Tree` for LLM training.

8. **Sendrecv P2P is message-size invariant at multi-node**: 43.4 GB/s at 2 nodes regardless of 2G/4G/8G, confirming pure link-bandwidth limitation. Bandwidth continues dropping with node count (43→16 GB/s) due to bisection bandwidth scaling.

---

## Recommendations for LLM Training

| Scale | Recommended NCCL Settings | Rationale |
|-------|--------------------------|-----------|
| 1 node (8 GPU) | `NCCL_NVLS_ENABLE=1`, default algo | NVLS gives best intra-node performance |
| 2 nodes (16 GPU) | `NCCL_NVLS_ENABLE=1`, default algo | NVLS still dominant at 2-node scale |
| 4–32 nodes (32–256 GPU) | `NCCL_COLLNET_ENABLE=1` | CollNet fallback outperforms ring at scale |
| 64 nodes (512 GPU) | `NCCL_COLLNET_ENABLE=1` | CollNet maintains best scaling efficiency |
| All scales | Use largest feasible GBS | Larger messages improve BW utilization |

### Optimization Opportunities

1. **Enable SHARP:** Admin action needed — start `sharp_am` on management node, `sharpd` on compute nodes, create SHARP reservations. Could boost CollNet beyond current fallback numbers.
2. **Pipeline parallelism bottleneck:** PP SendRecv at inter-node speeds is the scaling wall. Minimize PP stages or keep PP intra-node (NVLink) where possible.
3. **GBS tuning:** Maximize gradient accumulation steps to send larger messages and amortize latency overhead, especially at 32+ nodes.

---

## Raw Results

Results directory: `/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/collective-scaling/20260420_124543/`

Slurm jobs: 84286–84356

Submit script: `~/together-nccl-tests/benchmarks/B200/collective-scaling/submit_all.sh`

Manifest: `manifest.txt` in results directory
