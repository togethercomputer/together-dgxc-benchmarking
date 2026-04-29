# NCCL Collective Scaling Benchmark Report — 2026-04-25 (full re-run with local hpcx)

NCCL collective performance across 1–64 node scale (8–512 GPUs) on the B200 DGXC cluster, testing all_reduce, all_gather, and sendrecv P2P across five algorithm configurations and three message sizes.

---

## TL;DR

**59 OK / 12 expected failures / 0 hpcx failures** — perfect parity with 2026-04-20 baseline counts. Compare dir: `20260420_124543`.

**Cluster is performance-stable.** Across ~177 data points, nearly all deltas vs 4/20 are within ±5%:

- Largest regression: `all_reduce ring 4n` -6.8% at 8 GB (also seen morning run; consistent, ~real but small).
- Largest improvement: `all_reduce nvls_tree 4-8n` **+7-8%**, `all_reduce nvls 64n` +5.1%.

**Workaround that made this possible: local-disk hpcx staging.**

1. `rsync /opt/hpcx/` → `/mnt/vast/dgxc-benchmarking-auto/hpcx-2.18/` (1.2 GB shared).
2. `sbatch` broadcast-rsync to `/var/tmp/hpcx-2.18/` on each node (job 84938, 19s wall, **64/64 OK**).
3. `submit_all.sh` `HPCX_INIT` re-pointed to `/var/tmp/hpcx-2.18/hpcx-init.sh` with `/mnt/vast` and `/opt/hpcx` fallbacks.
4. Re-ran full 71-job suite with **no node exclude** (jobs 84939–85009).

**Result:** 0 jobs failed at startup — vs **27/71** in the morning's no-exclude bare-host run. The host `/opt/hpcx` gap is now invisible to bare-host NCCL benchmarks.

**Total wall:** ~34 min (vs 17 min for morning's run; more jobs reached the post-srun hang stage and watchdog needed to scancel — 44 watchdog cancellations in this run).

**Reproduction:**
- Stage: `sbatch ~/johnson/scripts/stage-hpcx-to-vartmp.sh`.
- Submit: `cd ~/together-nccl-tests/benchmarks/B200/collective-scaling && bash submit_all.sh --exclude ""`.
- Watchdog: `nohup /tmp/nccl-watchdog.sh > /tmp/nccl-watchdog.log 2>&1 &`.

**Action items:**
- Cluster ops ticket to reinstall `/opt/hpcx` on missing nodes — still the proper long-term fix.
- Re-stage hpcx after cluster reboots or when new nodes are added.

---

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
| Excluded |  |

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
| 1n (8) | 668.8 | **821.1** | -- | -- | -- | NVLS |
| 2n (16) | 341.8 | 687.0 | 378.3 | **688.1** | 326.5 | NVLSTree |
| 4n (32) | 311.9 | 329.4 | **378.5** | 269.2 | 185.9 | CollNet SHARP |
| 8n (64) | 304.6 | 308.0 | **382.4** | 267.1 | 187.9 | CollNet SHARP |
| 16n (128) | 287.3 | 266.4 | **357.8** | 261.5 | 187.8 | CollNet SHARP |
| 32n (256) | 269.0 | 256.5 | **337.3** | 261.0 | 187.1 | CollNet SHARP |
| 64n (512) | 222.9 | 257.6 | **274.4** | 257.7 | 186.1 | CollNet SHARP |

### all_reduce @ 4 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree | Best |
|--------------|------|------|---------------|----------|------|------|
| 1n (8) | 680.2 | **833.7** | -- | -- | -- | NVLS |
| 2n (16) | 343.2 | 705.6 | 381.9 | **705.7** | 331.8 | NVLSTree |
| 4n (32) | 311.4 | 331.4 | **384.2** | 285.3 | 187.9 | CollNet SHARP |
| 8n (64) | 303.7 | 308.4 | **380.8** | 286.7 | 190.4 | CollNet SHARP |
| 16n (128) | 310.5 | 311.0 | **384.3** | 274.8 | 190.9 | CollNet SHARP |
| 32n (256) | 287.5 | 269.2 | **361.2** | 277.5 | 191.0 | CollNet SHARP |
| 64n (512) | 269.2 | 275.2 | **337.0** | 273.0 | 190.1 | CollNet SHARP |

### all_reduce @ 8 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree | Best |
|--------------|------|------|---------------|----------|------|------|
| 1n (8) | 687.0 | **836.6** | -- | -- | -- | NVLS |
| 2n (16) | 343.8 | **713.6** | 376.7 | 713.5 | 334.2 | NVLS |
| 4n (32) | 310.4 | 333.0 | **384.4** | 289.8 | 189.0 | CollNet SHARP |
| 8n (64) | 303.9 | 309.1 | **385.2** | 287.7 | 191.8 | CollNet SHARP |
| 16n (128) | 309.3 | 310.9 | **379.8** | 276.9 | 192.5 | CollNet SHARP |
| 32n (256) | 307.4 | 304.2 | **385.2** | 279.1 | 192.9 | CollNet SHARP |
| 64n (512) | 288.1 | 277.7 | **363.4** | 274.4 | 192.3 | CollNet SHARP |

---

## Results — all_gather Peak busBW (GB/s)

tree and nvls_tree are not supported by NCCL for AllGather.

### all_gather @ 2 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |
|--------------|------|------|---------------|------|
| 1n (8) | 640.2 | **644.2** | -- | NVLS |
| 2n (16) | 336.9 | 337.2 | **356.9** | CollNet SHARP |
| 4n (32) | 324.8 | 325.5 | **372.5** | CollNet SHARP |
| 8n (64) | 312.5 | 312.7 | **378.2** | CollNet SHARP |
| 16n (128) | 286.2 | 286.0 | **353.4** | CollNet SHARP |
| 32n (256) | 233.5 | 234.2 | **283.7** | CollNet SHARP |
| 64n (512) | 223.9 | 224.9 | **274.8** | CollNet SHARP |

### all_gather @ 4 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |
|--------------|------|------|---------------|------|
| 1n (8) | 654.2 | **656.4** | -- | NVLS |
| 2n (16) | 339.7 | 340.3 | **367.9** | CollNet SHARP |
| 4n (32) | 330.1 | 329.9 | **381.9** | CollNet SHARP |
| 8n (64) | 314.6 | 312.6 | **375.7** | CollNet SHARP |
| 16n (128) | 315.4 | 314.2 | **382.2** | CollNet SHARP |
| 32n (256) | 286.2 | 286.3 | **358.6** | CollNet SHARP |
| 64n (512) | 231.5 | 231.2 | **280.8** | CollNet SHARP |

### all_gather @ 8 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |
|--------------|------|------|---------------|------|
| 1n (8) | 660.5 | **663.3** | -- | NVLS |
| 2n (16) | 341.1 | 341.6 | **363.6** | CollNet SHARP |
| 4n (32) | 332.4 | 333.2 | **382.9** | CollNet SHARP |
| 8n (64) | 315.9 | 315.7 | **384.0** | CollNet SHARP |
| 16n (128) | 309.9 | 316.4 | **378.8** | CollNet SHARP |
| 32n (256) | 309.6 | 311.5 | **384.2** | CollNet SHARP |
| 64n (512) | 286.0 | 286.0 | **361.7** | CollNet SHARP |

---

## Results — sendrecv P2P Peak busBW (GB/s)

| Nodes (GPUs) | 2 GB | 4 GB | 8 GB |
|--------------|------|------|------|
| 1n (8) | 647.2 | 652.1 | **654.9** |
| 2n (16) | 43.3 | **43.3** | 43.3 |
| 4n (32) | 44.0 | **44.0** | 44.0 |
| 8n (64) | 25.5 | **25.5** | **25.5** |
| 16n (128) | **14.8** | **14.8** | 14.8 |
| 32n (256) | **15.3** | **15.3** | **15.3** |
| 64n (512) | **15.7** | **15.7** | **15.7** |

---

## Scaling Efficiency — all_reduce @ 8 GB

Efficiency = busBW(N nodes) / busBW(baseline) × 100%. Baseline: 1-node for ring/nvls; 2-node for others.

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|--------------|------|------|---------------|----------|------|
| 1n (8) | **100.0%** | 100.0% | -- | -- | -- |
| 2n (16) | 50.0% | 85.3% | **100.0%** | 100.0% | 100.0% |
| 4n (32) | 45.2% | 39.8% | **102.1%** | 40.6% | 56.6% |
| 8n (64) | 44.2% | 36.9% | **102.3%** | 40.3% | 57.4% |
| 16n (128) | 45.0% | 37.2% | **100.8%** | 38.8% | 57.6% |
| 32n (256) | 44.7% | 36.4% | **102.3%** | 39.1% | 57.7% |
| 64n (512) | 41.9% | 33.2% | **96.5%** | 38.5% | 57.5% |

## Scaling Efficiency — all_gather @ 8 GB

Efficiency = busBW(N nodes) / busBW(baseline) × 100%. Baseline: 1-node for ring/nvls; 2-node for others.

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP |
|--------------|------|------|---------------|
| 1n (8) | **100.0%** | 100.0% | -- |
| 2n (16) | 51.7% | 51.5% | **100.0%** |
| 4n (32) | 50.3% | 50.2% | **105.3%** |
| 8n (64) | 47.8% | 47.6% | **105.6%** |
| 16n (128) | 46.9% | 47.7% | **104.2%** |
| 32n (256) | 46.9% | 47.0% | **105.7%** |
| 64n (512) | 43.3% | 43.1% | **99.5%** |

## Scaling Efficiency — sendrecv P2P @ 8 GB

| Nodes (GPUs) | Efficiency |
|--------------|-----------|
| 1n (8) | 100.0% |
| 2n (16) | 6.6% |
| 4n (32) | 6.7% |
| 8n (64) | 3.9% |
| 16n (128) | 2.3% |
| 32n (256) | 2.3% |
| 64n (512) | 2.4% |

---

## Step-by-Step Bandwidth Drop — all_reduce @ 8 GB

Identifies scaling cliffs between consecutive node counts.

| Transition | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|------------|------|------|---------------|----------|------|
| 1n → 2n | **-50.0%** | -14.7% | -- | -- | -- |
| 2n → 4n | -9.7% | **-53.3%** | +2.1% | **-59.4%** | **-43.4%** |
| 4n → 8n | -2.1% | -7.2% | +0.2% | -0.7% | +1.5% |
| 8n → 16n | +1.8% | +0.6% | -1.4% | -3.8% | +0.3% |
| 16n → 32n | -0.6% | -2.1% | +1.4% | +0.8% | +0.2% |
| 32n → 64n | -6.3% | -8.7% | -5.7% | -1.7% | -0.3% |

---

## Message Size Sensitivity — all_reduce 8G/2G Ratio

How much 8 GB messages outperform 2 GB, revealing bandwidth saturation effects at scale.

| Nodes | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|-------|------|------|---------------|----------|------|
| 1n | 1.03x | 1.02x | -- | -- | -- |
| 2n | 1.01x | 1.04x | 1.00x | 1.04x | 1.02x |
| 4n | 1.00x | 1.01x | 1.02x | 1.08x | 1.02x |
| 8n | 1.00x | 1.00x | 1.01x | 1.08x | 1.02x |
| 16n | 1.08x | 1.17x | 1.06x | 1.06x | 1.02x |
| 32n | 1.14x | 1.19x | 1.14x | 1.07x | 1.03x |
| 64n | 1.29x | 1.08x | **1.32x** | 1.06x | 1.03x |

## Message Size Sensitivity — all_gather 8G/2G Ratio

How much 8 GB messages outperform 2 GB, revealing bandwidth saturation effects at scale.

| Nodes | Ring | NVLS | CollNet SHARP |
|-------|------|------|---------------|
| 1n | 1.03x | 1.03x | -- |
| 2n | 1.01x | 1.01x | 1.02x |
| 4n | 1.02x | 1.02x | 1.03x |
| 8n | 1.01x | 1.01x | 1.02x |
| 16n | 1.08x | 1.11x | 1.07x |
| 32n | **1.33x** | **1.33x** | **1.35x** |
| 64n | 1.28x | 1.27x | **1.32x** |

---

## Algorithm Head-to-Head @ 8 GB

### all_reduce — Best Algorithm per Scale

| Nodes (GPUs) | Best Algorithm | busBW (GB/s) | 2nd Best | busBW (GB/s) | Gap |
|--------------|---------------|-------------|----------|-------------|-----|
| 1n (8) | **NVLS** | 836.6 | Ring | 687.0 | +21.8% |
| 2n (16) | **NVLS** | 713.6 | NVLSTree | 713.5 | +0.0% |
| 4n (32) | **CollNet SHARP** | 384.4 | NVLS | 333.0 | +15.4% |
| 8n (64) | **CollNet SHARP** | 385.2 | NVLS | 309.1 | +24.6% |
| 16n (128) | **CollNet SHARP** | 379.8 | NVLS | 310.9 | +22.2% |
| 32n (256) | **CollNet SHARP** | 385.2 | Ring | 307.4 | +25.3% |
| 64n (512) | **CollNet SHARP** | 363.4 | Ring | 288.1 | +26.1% |

### all_gather — Best Algorithm per Scale

| Nodes (GPUs) | Best Algorithm | busBW (GB/s) | 2nd Best | busBW (GB/s) | Gap |
|--------------|---------------|-------------|----------|-------------|-----|
| 1n (8) | **NVLS** | 663.3 | Ring | 660.5 | +0.4% |
| 2n (16) | **CollNet SHARP** | 363.6 | NVLS | 341.6 | +6.4% |
| 4n (32) | **CollNet SHARP** | 382.9 | NVLS | 333.2 | +14.9% |
| 8n (64) | **CollNet SHARP** | 384.0 | Ring | 315.9 | +21.5% |
| 16n (128) | **CollNet SHARP** | 378.8 | NVLS | 316.4 | +19.7% |
| 32n (256) | **CollNet SHARP** | 384.2 | NVLS | 311.5 | +23.3% |
| 64n (512) | **CollNet SHARP** | 361.7 | Ring | 286.0 | +26.4% |

---

## Interconnect Topology Analysis

### NVLink vs InfiniBand Boundary

| Metric | NVLink 1n (GB/s) | IB 2n (GB/s) | IB 64n (GB/s) | 1n→2n Drop |
|--------|-----------------|-------------|--------------|-----------|
| all_reduce (NVLS) | 836.6 | 713.6 | 277.7 | **-14.7%** |
| all_reduce (Ring) | 687.0 | 343.8 | 288.1 | **-50.0%** |
| sendrecv P2P | 654.9 | 43.3 | 15.7 | **-93.4%** |

### Bandwidth Utilization vs Theoretical @ 2 Nodes, 8 GB

| Collective | Best Algorithm | busBW (GB/s) | IB Theoretical | Utilization |
|------------|---------------|-------------|----------------|-------------|
| all_reduce | NVLS | 713.6 | 400 GB/s (8 HCAs) | 178.4% |
| all_gather | CollNet SHARP | 363.6 | 400 GB/s (8 HCAs) | 90.9% |
| sendrecv | P2P | 43.3 | 50 GB/s (1 HCA) | 86.6% |

---

## Delta vs Previous Run

### all_reduce @ 8 GB — Delta (%)

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|--------------|------|------|---------------|----------|------|
| 1n (8) | +1.8% | +0.3% | -- | -- | -- |
| 2n (16) | -0.2% | +0.1% | -0.4% | +0.1% | -1.1% |
| 4n (32) | **-6.8%** | +4.3% | -0.0% | **+8.5%** | +0.5% |
| 8n (64) | -1.5% | -4.1% | +0.0% | **+8.0%** | -1.0% |
| 16n (128) | +0.4% | +0.9% | -0.1% | +4.3% | +0.3% |
| 32n (256) | -0.8% | -0.8% | +0.0% | +4.6% | -0.0% |
| 64n (512) | -0.1% | **+5.1%** | -0.1% | +2.9% | -0.2% |

### all_gather @ 8 GB — Delta (%)

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP |
|--------------|------|------|---------------|
| 1n (8) | +0.9% | +0.4% | -- |
| 2n (16) | -0.6% | -0.4% | +0.2% |
| 4n (32) | -0.5% | +0.2% | +0.0% |
| 8n (64) | +1.0% | +0.1% | -0.0% |
| 16n (128) | -0.3% | +1.0% | +0.0% |
| 32n (256) | -2.4% | -0.3% | +0.0% |
| 64n (512) | -0.0% | -0.0% | -0.1% |

### sendrecv P2P @ 8 GB — Delta (%)

| Nodes (GPUs) | Delta |
|--------------|-------|
| 1n (8) | -0.1% |
| 2n (16) | -0.2% |
| 4n (32) | +2.3% |
| 8n (64) | +2.0% |
| 16n (128) | -3.8% |
| 32n (256) | -0.5% |
| 64n (512) | -0.1% |

### Significant Regressions (>5% drop)

- all_reduce ring 4n @ 8 GB: 333.0 → 310.4 GB/s (-6.8%)
- all_reduce ring 4n @ 4 GB: 332.2 → 311.4 GB/s (-6.3%)
- all_reduce ring 4n @ 2 GB: 329.0 → 311.9 GB/s (-5.2%)

### Significant Improvements (>5% gain)

- all_reduce nvls_tree 4n @ 8 GB: 267.2 → 289.8 GB/s (+8.5%)
- all_reduce nvls_tree 8n @ 8 GB: 266.4 → 287.7 GB/s (+8.0%)
- all_reduce nvls_tree 8n @ 4 GB: 266.0 → 286.7 GB/s (+7.8%)
- all_reduce nvls_tree 4n @ 4 GB: 266.6 → 285.3 GB/s (+7.0%)
- all_reduce nvls 16n @ 2 GB: 251.9 → 266.4 GB/s (+5.8%)
- all_reduce nvls 4n @ 2 GB: 312.5 → 329.4 GB/s (+5.4%)
- all_reduce nvls 64n @ 8 GB: 264.1 → 277.7 GB/s (+5.1%)

---

## Observations

1. **CollNet SHARP is the clear winner at 4+ nodes** across both all_reduce (~384 GB/s) and all_gather, despite SHARP hardware not being operational. NCCL's CollNet fallback outperforms ring by 25% at 32 nodes. All CollNet jobs logged `SHARP coll init error: Cannot create SHARP job` — actual SHARP offload would likely push numbers higher.

2. **NVLS dominates intra-node and 2-node scale** with 837 GB/s all_reduce at 1 node (+22% over ring) and 714 GB/s at 2 nodes. NVLS leverages NVLink multicast for efficient intra-node reduction.

3. **NVLS collapses at 64 nodes**: all_reduce NVLS drops to 278 GB/s — worse than ring (288 GB/s). Don't force `NCCL_NVLS_ENABLE=1` for large-scale training.

4. **CollNet scaling is remarkably flat**: all_reduce CollNet holds ~384 GB/s from 4 to 32 nodes (0% variation), only dropping 6% at 64 nodes.

5. **The NVLink→IB cliff is 15x for P2P**: sendrecv drops from 655 GB/s intra-node to 43.3 GB/s at 2 nodes. P2P is single-HCA limited (43.3 / 50 = 87% utilization). This makes pipeline parallelism (PP) the inter-node bottleneck.

6. **Larger messages improve bandwidth at scale**: ring 64n gains 29% from 2G→8G (223→288 GB/s). This validates GBS tuning (more gradient accumulation) as a communication optimization strategy.

7. **Tree algorithm is consistently worst** for large messages: ~192 GB/s ceiling regardless of node count at 4–64 nodes. Never force `NCCL_ALGO=Tree` for LLM training.

8. **Sendrecv P2P is message-size invariant at multi-node**: 43.3 GB/s at 2 nodes regardless of 2G/4G/8G, confirming pure link-bandwidth limitation. Bandwidth drops to 16 GB/s at 64 nodes.

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

Results directory: `/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/collective-scaling/20260425_111753`

Slurm jobs: 84939–85009

Submit script: `~/together-nccl-tests/benchmarks/B200/collective-scaling/submit_all.sh`

Manifest: `manifest.txt` in results directory
