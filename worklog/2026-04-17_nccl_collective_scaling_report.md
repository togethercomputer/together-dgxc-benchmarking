# NCCL Collective Scaling Benchmark Report — 2026-04-17

NCCL collective performance across 1–64 node scale (8–512 GPUs) on the B200 DGXC cluster, testing all_reduce, all_gather, and sendrecv P2P across five algorithm configurations and three message sizes.

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
| Excluded | use3a-ss-b200-gpu-[145,190,227-228,233,239] |

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
| 1n (8) | 658.3 | **821.0** | -- | -- | -- | NVLS |
| 2n (16) | 341.9 | 687.5 | 371.1 | **687.6** | 327.3 | NVLSTree |
| 4n (32) | 328.7 | 315.0 | **379.2** | 257.6 | 184.5 | CollNet SHARP |
| 8n (64) | 309.8 | 309.6 | **382.5** | 256.0 | 188.8 | CollNet SHARP |
| 16n (128) | 287.9 | 251.8 | **357.7** | 253.5 | 187.9 | CollNet SHARP |
| 32n (256) | 269.3 | 249.6 | **336.5** | 251.1 | 186.6 | CollNet SHARP |
| 64n (512) | 223.8 | 244.3 | **275.7** | 249.9 | 185.8 | CollNet SHARP |

### all_reduce @ 4 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree | Best |
|--------------|------|------|---------------|----------|------|------|
| 1n (8) | 668.0 | **833.9** | -- | -- | -- | NVLS |
| 2n (16) | 343.3 | **705.5** | 375.4 | 705.4 | 332.5 | NVLS |
| 4n (32) | 331.2 | 315.4 | **384.3** | 266.1 | 186.6 | CollNet SHARP |
| 8n (64) | 307.3 | 313.3 | **378.1** | 266.4 | 191.3 | CollNet SHARP |
| 16n (128) | 307.7 | 309.9 | **384.3** | 264.9 | 190.8 | CollNet SHARP |
| 32n (256) | 288.3 | 264.4 | **361.1** | 264.6 | 190.2 | CollNet SHARP |
| 64n (512) | 269.7 | 261.7 | **337.0** | 264.0 | 189.5 | CollNet SHARP |

### all_reduce @ 8 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree | Best |
|--------------|------|------|---------------|----------|------|------|
| 1n (8) | 676.3 | **837.1** | -- | -- | -- | NVLS |
| 2n (16) | 343.9 | **713.3** | 375.4 | 713.2 | 335.1 | NVLS |
| 4n (32) | 332.9 | 311.4 | **384.3** | 263.5 | 187.8 | CollNet SHARP |
| 8n (64) | 307.4 | 312.5 | **385.2** | 264.5 | 192.6 | CollNet SHARP |
| 16n (128) | 305.2 | 307.2 | **379.5** | 265.3 | 192.4 | CollNet SHARP |
| 32n (256) | 309.5 | 311.1 | **385.2** | 267.6 | 192.1 | CollNet SHARP |
| 64n (512) | 288.0 | 266.7 | **364.4** | 267.3 | 192.0 | CollNet SHARP |

---

## Results — all_gather Peak busBW (GB/s)

tree and nvls_tree are not supported by NCCL for AllGather.

### all_gather @ 2 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |
|--------------|------|------|---------------|------|
| 1n (8) | **640.7** | 635.5 | -- | Ring |
| 2n (16) | 338.8 | 338.6 | **357.4** | CollNet SHARP |
| 4n (32) | 325.1 | 325.3 | **370.9** | CollNet SHARP |
| 8n (64) | 319.7 | 313.5 | **378.4** | CollNet SHARP |
| 16n (128) | 285.8 | 286.3 | **353.8** | CollNet SHARP |
| 32n (256) | 232.3 | 234.6 | **283.4** | CollNet SHARP |
| 64n (512) | 225.3 | 224.8 | **275.8** | CollNet SHARP |

### all_gather @ 4 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |
|--------------|------|------|---------------|------|
| 1n (8) | 653.9 | **654.8** | -- | NVLS |
| 2n (16) | 341.8 | 341.4 | **368.1** | CollNet SHARP |
| 4n (32) | 329.8 | 330.2 | **381.7** | CollNet SHARP |
| 8n (64) | 321.3 | 317.7 | **375.9** | CollNet SHARP |
| 16n (128) | 313.4 | 316.1 | **382.3** | CollNet SHARP |
| 32n (256) | 286.5 | 285.8 | **358.6** | CollNet SHARP |
| 64n (512) | 234.4 | 235.3 | **282.3** | CollNet SHARP |

### all_gather @ 8 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |
|--------------|------|------|---------------|------|
| 1n (8) | 660.3 | **660.9** | -- | NVLS |
| 2n (16) | 343.2 | 342.9 | **363.5** | CollNet SHARP |
| 4n (32) | 333.2 | 332.6 | **382.7** | CollNet SHARP |
| 8n (64) | 324.9 | 318.8 | **384.0** | CollNet SHARP |
| 16n (128) | 312.2 | 314.7 | **378.3** | CollNet SHARP |
| 32n (256) | 315.8 | 313.5 | **384.2** | CollNet SHARP |
| 64n (512) | 286.3 | 285.9 | **362.3** | CollNet SHARP |

---

## Results — sendrecv P2P Peak busBW (GB/s)

| Nodes (GPUs) | 2 GB | 4 GB | 8 GB |
|--------------|------|------|------|
| 1n (8) | 646.9 | 651.6 | **655.6** |
| 2n (16) | 42.8 | **42.8** | **42.8** |
| 4n (32) | 43.4 | 43.4 | **43.5** |
| 8n (64) | 24.1 | **24.1** | 24.1 |
| 16n (128) | **14.8** | **14.8** | **14.8** |
| 32n (256) | **15.3** | **15.3** | **15.3** |
| 64n (512) | **15.7** | **15.7** | **15.7** |

---

## Scaling Efficiency — all_reduce @ 8 GB

Efficiency = busBW(N nodes) / busBW(baseline) × 100%. Baseline: 1-node for ring/nvls; 2-node for others.

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|--------------|------|------|---------------|----------|------|
| 1n (8) | **100.0%** | 100.0% | -- | -- | -- |
| 2n (16) | 50.8% | 85.2% | **100.0%** | 100.0% | 100.0% |
| 4n (32) | 49.2% | 37.2% | **102.4%** | 36.9% | 56.0% |
| 8n (64) | 45.5% | 37.3% | **102.6%** | 37.1% | 57.5% |
| 16n (128) | 45.1% | 36.7% | **101.1%** | 37.2% | 57.4% |
| 32n (256) | 45.8% | 37.2% | **102.6%** | 37.5% | 57.3% |
| 64n (512) | 42.6% | 31.9% | **97.1%** | 37.5% | 57.3% |

## Scaling Efficiency — all_gather @ 8 GB

Efficiency = busBW(N nodes) / busBW(baseline) × 100%. Baseline: 1-node for ring/nvls; 2-node for others.

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP |
|--------------|------|------|---------------|
| 1n (8) | **100.0%** | 100.0% | -- |
| 2n (16) | 52.0% | 51.9% | **100.0%** |
| 4n (32) | 50.5% | 50.3% | **105.3%** |
| 8n (64) | 49.2% | 48.2% | **105.6%** |
| 16n (128) | 47.3% | 47.6% | **104.1%** |
| 32n (256) | 47.8% | 47.4% | **105.7%** |
| 64n (512) | 43.4% | 43.3% | **99.7%** |

## Scaling Efficiency — sendrecv P2P @ 8 GB

| Nodes (GPUs) | Efficiency |
|--------------|-----------|
| 1n (8) | 100.0% |
| 2n (16) | 6.5% |
| 4n (32) | 6.6% |
| 8n (64) | 3.7% |
| 16n (128) | 2.3% |
| 32n (256) | 2.3% |
| 64n (512) | 2.4% |

---

## Step-by-Step Bandwidth Drop — all_reduce @ 8 GB

Identifies scaling cliffs between consecutive node counts.

| Transition | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|------------|------|------|---------------|----------|------|
| 1n → 2n | **-49.2%** | -14.8% | -- | -- | -- |
| 2n → 4n | -3.2% | **-56.3%** | +2.4% | **-63.1%** | **-44.0%** |
| 4n → 8n | -7.6% | +0.4% | +0.2% | +0.4% | +2.6% |
| 8n → 16n | -0.7% | -1.7% | -1.5% | +0.3% | -0.1% |
| 16n → 32n | +1.4% | +1.3% | +1.5% | +0.9% | -0.2% |
| 32n → 64n | -6.9% | -14.3% | -5.4% | -0.1% | -0.0% |

---

## Message Size Sensitivity — all_reduce 8G/2G Ratio

How much 8 GB messages outperform 2 GB, revealing bandwidth saturation effects at scale.

| Nodes | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|-------|------|------|---------------|----------|------|
| 1n | 1.03x | 1.02x | -- | -- | -- |
| 2n | 1.01x | 1.04x | 1.01x | 1.04x | 1.02x |
| 4n | 1.01x | 0.99x | 1.01x | 1.02x | 1.02x |
| 8n | 0.99x | 1.01x | 1.01x | 1.03x | 1.02x |
| 16n | 1.06x | 1.22x | 1.06x | 1.05x | 1.02x |
| 32n | 1.15x | 1.25x | 1.14x | 1.07x | 1.03x |
| 64n | 1.29x | 1.09x | **1.32x** | 1.07x | 1.03x |

## Message Size Sensitivity — all_gather 8G/2G Ratio

How much 8 GB messages outperform 2 GB, revealing bandwidth saturation effects at scale.

| Nodes | Ring | NVLS | CollNet SHARP |
|-------|------|------|---------------|
| 1n | 1.03x | 1.04x | -- |
| 2n | 1.01x | 1.01x | 1.02x |
| 4n | 1.03x | 1.02x | 1.03x |
| 8n | 1.02x | 1.02x | 1.01x |
| 16n | 1.09x | 1.10x | 1.07x |
| 32n | **1.36x** | **1.34x** | **1.36x** |
| 64n | 1.27x | 1.27x | **1.31x** |

---

## Algorithm Head-to-Head @ 8 GB

### all_reduce — Best Algorithm per Scale

| Nodes (GPUs) | Best Algorithm | busBW (GB/s) | 2nd Best | busBW (GB/s) | Gap |
|--------------|---------------|-------------|----------|-------------|-----|
| 1n (8) | **NVLS** | 837.1 | Ring | 676.3 | +23.8% |
| 2n (16) | **NVLS** | 713.3 | NVLSTree | 713.2 | +0.0% |
| 4n (32) | **CollNet SHARP** | 384.3 | Ring | 332.9 | +15.5% |
| 8n (64) | **CollNet SHARP** | 385.2 | NVLS | 312.5 | +23.3% |
| 16n (128) | **CollNet SHARP** | 379.5 | NVLS | 307.2 | +23.5% |
| 32n (256) | **CollNet SHARP** | 385.2 | NVLS | 311.1 | +23.8% |
| 64n (512) | **CollNet SHARP** | 364.4 | Ring | 288.0 | +26.5% |

### all_gather — Best Algorithm per Scale

| Nodes (GPUs) | Best Algorithm | busBW (GB/s) | 2nd Best | busBW (GB/s) | Gap |
|--------------|---------------|-------------|----------|-------------|-----|
| 1n (8) | **NVLS** | 660.9 | Ring | 660.3 | +0.1% |
| 2n (16) | **CollNet SHARP** | 363.5 | Ring | 343.2 | +5.9% |
| 4n (32) | **CollNet SHARP** | 382.7 | Ring | 333.2 | +14.8% |
| 8n (64) | **CollNet SHARP** | 384.0 | Ring | 324.9 | +18.2% |
| 16n (128) | **CollNet SHARP** | 378.3 | NVLS | 314.7 | +20.2% |
| 32n (256) | **CollNet SHARP** | 384.2 | Ring | 315.8 | +21.6% |
| 64n (512) | **CollNet SHARP** | 362.3 | Ring | 286.3 | +26.5% |

---

## Interconnect Topology Analysis

### NVLink vs InfiniBand Boundary

| Metric | NVLink 1n (GB/s) | IB 2n (GB/s) | IB 64n (GB/s) | 1n→2n Drop |
|--------|-----------------|-------------|--------------|-----------|
| all_reduce (NVLS) | 837.1 | 713.3 | 266.7 | **-14.8%** |
| all_reduce (Ring) | 676.3 | 343.9 | 288.0 | **-49.2%** |
| sendrecv P2P | 655.6 | 42.8 | 15.7 | **-93.5%** |

### Bandwidth Utilization vs Theoretical @ 2 Nodes, 8 GB

| Collective | Best Algorithm | busBW (GB/s) | IB Theoretical | Utilization |
|------------|---------------|-------------|----------------|-------------|
| all_reduce | NVLS | 713.3 | 400 GB/s (8 HCAs) | 178.3% |
| all_gather | CollNet SHARP | 363.5 | 400 GB/s (8 HCAs) | 90.9% |
| sendrecv | P2P | 42.8 | 50 GB/s (1 HCA) | 85.6% |

---

## Delta vs Previous Run

### all_reduce @ 8 GB — Delta (%)

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|--------------|------|------|---------------|----------|------|
| 1n (8) | -0.7% | -0.2% | -- | -- | -- |
| 2n (16) | -0.0% | -0.0% | **+6.7%** | **+17.1%** | +0.9% |
| 4n (32) | +0.2% | **-6.3%** | -0.0% | -1.0% | -1.3% |
| 8n (64) | +0.6% | -2.4% | +0.0% | +2.1% | +0.2% |
| 16n (128) | -0.5% | +0.8% | +0.4% | **+23.3%** | +0.9% |
| 32n (256) | +2.1% | +2.6% | +0.0% | **+27.3%** | +0.2% |
| 64n (512) | +2.9% | **+28.9%** | +4.1% | **+28.9%** | +0.3% |

### all_gather @ 8 GB — Delta (%)

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP |
|--------------|------|------|---------------|
| 1n (8) | +0.8% | -0.1% | -- |
| 2n (16) | +0.7% | +0.4% | +0.3% |
| 4n (32) | +0.3% | +0.1% | -0.1% |
| 8n (64) | +0.6% | +0.9% | +0.0% |
| 16n (128) | +4.6% | -1.3% | +0.1% |
| 32n (256) | +0.2% | +4.7% | +0.0% |
| 64n (512) | +3.6% | +3.7% | +3.6% |

### sendrecv P2P @ 8 GB — Delta (%)

| Nodes (GPUs) | Delta |
|--------------|-------|
| 1n (8) | +0.3% |
| 2n (16) | +0.8% |
| 4n (32) | -1.8% |
| 8n (64) | **-6.0%** |
| 16n (128) | -3.1% |
| 32n (256) | +4.1% |
| 64n (512) | +3.8% |

### Significant Regressions (>5% drop)

- all_reduce nvls 4n @ 8 GB: 332.4 → 311.4 GB/s (-6.3%)
- sendrecv p2p 8n @ 2 GB: 25.7 → 24.1 GB/s (-6.0%)
- sendrecv p2p 8n @ 8 GB: 25.7 → 24.1 GB/s (-6.0%)
- sendrecv p2p 8n @ 4 GB: 25.7 → 24.1 GB/s (-6.0%)

### Significant Improvements (>5% gain)

- all_reduce nvls_tree 64n @ 8 GB: 207.4 → 267.3 GB/s (+28.9%)
- all_reduce nvls 64n @ 8 GB: 206.9 → 266.7 GB/s (+28.9%)
- all_reduce nvls_tree 32n @ 8 GB: 210.2 → 267.6 GB/s (+27.3%)
- all_reduce nvls_tree 16n @ 8 GB: 215.1 → 265.3 GB/s (+23.3%)
- all_gather collnet_sharp 32n @ 2 GB: 233.0 → 283.4 GB/s (+21.6%)
- all_gather collnet_sharp 64n @ 4 GB: 233.2 → 282.3 GB/s (+21.1%)
- all_reduce nvls_tree 2n @ 8 GB: 609.3 → 713.2 GB/s (+17.1%)
- all_reduce nvls_tree 2n @ 4 GB: 622.9 → 705.4 GB/s (+13.2%)
- all_reduce nvls_tree 64n @ 4 GB: 233.9 → 264.0 GB/s (+12.9%)
- all_reduce collnet_sharp 64n @ 2 GB: 245.8 → 275.7 GB/s (+12.2%)
- all_gather collnet_sharp 64n @ 2 GB: 247.9 → 275.8 GB/s (+11.3%)
- all_reduce nvls_tree 2n @ 2 GB: 619.1 → 687.6 GB/s (+11.1%)
- all_gather ring 64n @ 2 GB: 205.5 → 225.3 GB/s (+9.7%)
- all_gather nvls 64n @ 4 GB: 215.5 → 235.3 GB/s (+9.2%)
- all_gather ring 64n @ 4 GB: 214.8 → 234.4 GB/s (+9.1%)
- all_reduce ring 64n @ 2 GB: 205.3 → 223.8 GB/s (+9.1%)
- all_gather nvls 64n @ 2 GB: 206.7 → 224.8 GB/s (+8.8%)
- all_gather nvls 32n @ 2 GB: 216.3 → 234.6 GB/s (+8.4%)
- all_reduce nvls 64n @ 4 GB: 242.9 → 261.7 GB/s (+7.7%)
- all_reduce collnet_sharp 2n @ 8 GB: 351.9 → 375.4 GB/s (+6.7%)
- all_reduce nvls_tree 32n @ 4 GB: 248.9 → 264.6 GB/s (+6.3%)
- all_gather ring 16n @ 4 GB: 296.4 → 313.4 GB/s (+5.7%)

---

## Observations

1. **CollNet SHARP is the clear winner at 4+ nodes** across both all_reduce (~384 GB/s) and all_gather, despite SHARP hardware not being operational. NCCL's CollNet fallback outperforms ring by 24% at 32 nodes. All CollNet jobs logged `SHARP coll init error: Cannot create SHARP job` — actual SHARP offload would likely push numbers higher.

2. **NVLS dominates intra-node and 2-node scale** with 837 GB/s all_reduce at 1 node (+24% over ring) and 713 GB/s at 2 nodes. NVLS leverages NVLink multicast for efficient intra-node reduction.

3. **NVLS collapses at 64 nodes**: all_reduce NVLS drops to 267 GB/s — worse than ring (288 GB/s). Don't force `NCCL_NVLS_ENABLE=1` for large-scale training.

4. **CollNet scaling is remarkably flat**: all_reduce CollNet holds ~384 GB/s from 4 to 32 nodes (0% variation), only dropping 5% at 64 nodes.

5. **The NVLink→IB cliff is 15x for P2P**: sendrecv drops from 656 GB/s intra-node to 42.8 GB/s at 2 nodes. P2P is single-HCA limited (42.8 / 50 = 86% utilization). This makes pipeline parallelism (PP) the inter-node bottleneck.

6. **Larger messages improve bandwidth at scale**: ring 64n gains 29% from 2G→8G (224→288 GB/s). This validates GBS tuning (more gradient accumulation) as a communication optimization strategy.

7. **Tree algorithm is consistently worst** for large messages: ~191 GB/s ceiling regardless of node count at 4–64 nodes. Never force `NCCL_ALGO=Tree` for LLM training.

8. **Sendrecv P2P is message-size invariant at multi-node**: 42.8 GB/s at 2 nodes regardless of 2G/4G/8G, confirming pure link-bandwidth limitation. Bandwidth continues dropping with node count (43→16 GB/s) due to bisection bandwidth scaling.

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

Results directory: `/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/collective-scaling/20260417_173551/`

Slurm jobs: 84174–84244

Submit script: `~/together-nccl-tests/benchmarks/B200/collective-scaling/submit_all.sh`

Manifest: `manifest.txt` in results directory
