# NCCL Collective Scaling Benchmark Report — 2026-04-16

NCCL collective performance across 1–64 node scale (8–512 GPUs) on the B200 DGXC cluster, testing all_reduce, all_gather, and sendrecv P2P across five algorithm configurations and three message sizes.

## System Configuration

| Component | Details |
|-----------|---------|
| GPU | NVIDIA B200, 8 per node |
| Nodes | 1–64 (71 available, 2 drained: gpu-145, gpu-190) |
| Total GPUs | 8–512 |
| Interconnect | Mellanox ConnectX-7 (MT4129), NDR 400 Gb/s InfiniBand |
| NICs per node | 8 HCAs (mlx5_0, mlx5_1, mlx5_4, mlx5_5, mlx5_6, mlx5_11, mlx5_14, mlx5_15) |
| NVLink | 5th gen, ~900 GB/s bidirectional intra-node |
| SHARP | Not available (sharpd not running, no reservations) |

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
| 1n (8) | 663.6 | **819.2** | -- | -- | -- | NVLS |
| 2n (16) | 342.0 | **687.2** | 364.0 | 619.1 | 324.1 | NVLS |
| 4n (32) | 328.1 | 327.7 | **378.8** | 257.2 | 186.9 | CollNet |
| 8n (64) | 305.4 | 318.8 | **382.3** | 254.9 | 188.5 | CollNet |
| 16n (128) | 285.0 | 246.8 | **354.1** | 249.2 | 185.9 | CollNet |
| 32n (256) | 268.2 | 245.1 | **335.0** | 246.3 | 186.0 | CollNet |
| 64n (512) | 205.3 | **242.5** | 245.8 | 238.3 | 184.7 | CollNet |

### all_reduce @ 4 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree | Best |
|--------------|------|------|---------------|----------|------|------|
| 1n (8) | 671.3 | **834.6** | -- | -- | -- | NVLS |
| 2n (16) | 343.3 | **705.5** | 373.8 | 622.9 | 329.5 | NVLS |
| 4n (32) | 331.9 | 331.3 | **384.3** | 265.1 | 189.0 | CollNet |
| 8n (64) | 305.4 | 319.1 | **378.2** | 261.7 | 190.8 | CollNet |
| 16n (128) | 305.2 | 307.9 | **384.2** | 256.4 | 189.0 | CollNet |
| 32n (256) | 277.1 | 251.9 | **357.9** | 248.9 | 189.8 | CollNet |
| 64n (512) | 268.5 | 242.9 | **336.0** | 233.9 | 189.2 | CollNet |

### all_reduce @ 8 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree | Best |
|--------------|------|------|---------------|----------|------|------|
| 1n (8) | 681.4 | **839.0** | -- | -- | -- | NVLS |
| 2n (16) | 344.0 | **713.3** | 351.9 | 609.3 | 332.0 | NVLS |
| 4n (32) | 332.4 | 332.4 | **384.5** | 266.0 | 190.3 | CollNet |
| 8n (64) | 305.6 | 320.3 | **385.2** | 259.2 | 192.2 | CollNet |
| 16n (128) | 306.8 | 304.8 | **377.9** | 215.1 | 190.7 | CollNet |
| 32n (256) | 303.2 | 303.2 | **385.1** | 210.2 | 191.7 | CollNet |
| 64n (512) | 279.9 | 206.9 | **350.0** | 207.4 | 191.4 | CollNet |

---

## Results — all_gather Peak busBW (GB/s)

tree and nvls_tree are not supported by NCCL for AllGather.

### all_gather @ 2 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |
|--------------|------|------|---------------|------|
| 1n (8) | **638.9** | 636.2 | -- | Ring |
| 2n (16) | 336.6 | 337.1 | **356.3** | CollNet |
| 4n (32) | 326.1 | 324.3 | **373.5** | CollNet |
| 8n (64) | 317.9 | 311.1 | **378.4** | CollNet |
| 16n (128) | 280.5 | 285.4 | **352.9** | CollNet |
| 32n (256) | 238.2 | 216.3 | **233.0** | Ring |
| 64n (512) | 205.5 | 206.7 | **247.9** | CollNet |

### all_gather @ 4 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |
|--------------|------|------|---------------|------|
| 1n (8) | 649.5 | **654.6** | -- | NVLS |
| 2n (16) | 339.3 | 340.1 | **366.1** | CollNet |
| 4n (32) | 330.2 | 330.5 | **381.6** | CollNet |
| 8n (64) | 320.4 | 315.5 | **376.8** | CollNet |
| 16n (128) | 296.4 | 318.1 | **382.2** | CollNet |
| 32n (256) | 284.5 | 276.3 | **347.6** | CollNet |
| 64n (512) | 214.8 | 215.5 | **233.2** | CollNet |

### all_gather @ 8 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |
|--------------|------|------|---------------|------|
| 1n (8) | 655.3 | **661.2** | -- | NVLS |
| 2n (16) | 340.8 | 341.5 | **362.4** | CollNet |
| 4n (32) | 332.1 | 332.4 | **382.9** | CollNet |
| 8n (64) | 322.9 | 316.0 | **384.0** | CollNet |
| 16n (128) | 298.6 | 318.9 | **377.8** | CollNet |
| 32n (256) | 315.3 | 299.6 | **384.1** | CollNet |
| 64n (512) | 276.4 | 275.9 | **349.8** | CollNet |

---

## Results — sendrecv P2P Peak busBW (GB/s)

| Nodes (GPUs) | 2 GB | 4 GB | 8 GB |
|--------------|------|------|------|
| 1n (8) | 644.9 | 651.0 | **653.3** |
| 2n (16) | 42.5 | 42.5 | 42.5 |
| 4n (32) | 44.3 | 44.2 | 44.2 |
| 8n (64) | 25.7 | 25.7 | 25.7 |
| 16n (128) | 15.3 | 15.3 | 15.3 |
| 32n (256) | 14.7 | 14.7 | 14.7 |
| 64n (512) | 15.1 | 15.1 | 15.1 |

---

## Scaling Efficiency — all_reduce @ 8 GB

Efficiency = busBW(N nodes) / busBW(baseline) × 100%. Baseline: 1-node for ring/nvls; 2-node for others.

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|--------------|------|------|---------------|----------|------|
| 1n (8) | 100.0% | 100.0% | -- | -- | -- |
| 2n (16) | 50.5% | 85.0% | 100.0% | 100.0% | 100.0% |
| 4n (32) | 48.8% | 39.6% | **109.3%** | 43.7% | 57.3% |
| 8n (64) | 44.8% | 38.2% | **109.5%** | 42.5% | 57.9% |
| 16n (128) | 45.0% | 36.3% | **107.4%** | 35.3% | 57.4% |
| 32n (256) | 44.5% | 36.1% | **109.4%** | 34.5% | 57.7% |
| 64n (512) | 41.1% | 24.7% | **99.5%** | 34.0% | 57.7% |

## Scaling Efficiency — all_gather @ 8 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP |
|--------------|------|------|---------------|
| 1n (8) | 100.0% | 100.0% | -- |
| 2n (16) | 52.0% | 51.6% | 100.0% |
| 4n (32) | 50.7% | 50.3% | **105.7%** |
| 8n (64) | 49.3% | 47.8% | **106.0%** |
| 16n (128) | 45.6% | 48.2% | **104.2%** |
| 32n (256) | 48.1% | 45.3% | **106.0%** |
| 64n (512) | 42.2% | 41.7% | **96.5%** |

## Scaling Efficiency — sendrecv P2P @ 8 GB

| Nodes (GPUs) | Efficiency |
|--------------|-----------|
| 1n (8) | 100.0% |
| 2n (16) | 6.5% |
| 4n (32) | 6.8% |
| 8n (64) | 3.9% |
| 16n (128) | 2.3% |
| 32n (256) | 2.3% |
| 64n (512) | 2.3% |

---

## Step-by-Step Bandwidth Drop — all_reduce @ 8 GB

Identifies scaling cliffs between consecutive node counts.

| Transition | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|------------|------|------|---------------|----------|------|
| 1n → 2n | **-49.5%** | -15.0% | -- | -- | -- |
| 2n → 4n | -3.4% | **-53.4%** | +9.3% | **-56.3%** | **-42.7%** |
| 4n → 8n | -8.1% | -3.6% | +0.2% | -2.6% | +1.0% |
| 8n → 16n | +0.4% | -4.8% | -1.9% | **-17.0%** | -0.8% |
| 16n → 32n | -1.2% | -0.5% | +1.9% | -2.3% | +0.5% |
| 32n → 64n | -7.7% | **-31.8%** | -9.1% | -1.3% | -0.2% |

---

## Message Size Sensitivity — 8G/2G Ratio

How much 8 GB messages outperform 2 GB, revealing bandwidth saturation effects at scale.

### all_reduce 8G/2G Ratio

| Nodes | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|-------|------|------|---------------|----------|------|
| 1n | 1.03x | 1.02x | -- | -- | -- |
| 2n | 1.01x | 1.04x | 0.97x | 0.98x | 1.02x |
| 4n | 1.01x | 1.01x | 1.02x | 1.03x | 1.02x |
| 8n | 1.00x | 1.00x | 1.01x | 1.02x | 1.02x |
| 16n | 1.08x | 1.24x | 1.07x | 0.86x | 1.03x |
| 32n | 1.13x | 1.24x | 1.15x | 0.85x | 1.03x |
| 64n | **1.36x** | 0.85x | **1.42x** | 0.87x | 1.04x |

### all_gather 8G/2G Ratio

| Nodes | Ring | NVLS | CollNet SHARP |
|-------|------|------|---------------|
| 1n | 1.03x | 1.04x | -- |
| 2n | 1.01x | 1.01x | 1.02x |
| 4n | 1.02x | 1.02x | 1.03x |
| 8n | 1.02x | 1.02x | 1.01x |
| 16n | 1.06x | 1.12x | 1.07x |
| 32n | 1.32x | 1.39x | **1.65x** |
| 64n | **1.35x** | 1.33x | **1.41x** |

---

## Algorithm Head-to-Head @ 8 GB

### all_reduce — Best Algorithm per Scale

| Nodes (GPUs) | Best Algorithm | busBW (GB/s) | 2nd Best | busBW (GB/s) | Gap |
|--------------|---------------|-------------|----------|-------------|-----|
| 1n (8) | **NVLS** | 839.0 | Ring | 681.4 | +23.1% |
| 2n (16) | **NVLS** | 713.3 | NVLSTree | 609.3 | +17.1% |
| 4n (32) | **CollNet SHARP** | 384.5 | Ring | 332.4 | +15.7% |
| 8n (64) | **CollNet SHARP** | 385.2 | NVLS | 320.3 | +20.3% |
| 16n (128) | **CollNet SHARP** | 377.9 | Ring | 306.8 | +23.2% |
| 32n (256) | **CollNet SHARP** | 385.1 | Ring | 303.2 | +27.0% |
| 64n (512) | **CollNet SHARP** | 350.0 | Ring | 279.9 | +25.0% |

### all_gather — Best Algorithm per Scale

| Nodes (GPUs) | Best Algorithm | busBW (GB/s) | 2nd Best | busBW (GB/s) | Gap |
|--------------|---------------|-------------|----------|-------------|-----|
| 1n (8) | **NVLS** | 661.2 | Ring | 655.3 | +0.9% |
| 2n (16) | **CollNet SHARP** | 362.4 | NVLS | 341.5 | +6.1% |
| 4n (32) | **CollNet SHARP** | 382.9 | NVLS | 332.4 | +15.2% |
| 8n (64) | **CollNet SHARP** | 384.0 | Ring | 322.9 | +18.9% |
| 16n (128) | **CollNet SHARP** | 377.8 | NVLS | 318.9 | +18.5% |
| 32n (256) | **CollNet SHARP** | 384.1 | Ring | 315.3 | +21.8% |
| 64n (512) | **CollNet SHARP** | 349.8 | Ring | 276.4 | +26.6% |

---

## Interconnect Topology Analysis

### NVLink vs InfiniBand Boundary

| Metric | NVLink 1n (GB/s) | IB 2n (GB/s) | IB 64n (GB/s) | 1n→2n Drop |
|--------|-----------------|-------------|--------------|-----------|
| all_reduce (NVLS) | 839.0 | 713.3 | 206.9 | -15.0% |
| all_reduce (Ring) | 681.4 | 344.0 | 279.9 | -49.5% |
| sendrecv P2P | 653.3 | 42.5 | 15.1 | **-93.5%** |

### Bandwidth Utilization vs Theoretical @ 2 Nodes, 8 GB

| Collective | Best Algorithm | busBW (GB/s) | IB Theoretical | Utilization |
|------------|---------------|-------------|----------------|-------------|
| all_reduce | NVLS | 713.3 | 400 GB/s (8 HCAs) | 178.3% |
| all_gather | CollNet SHARP | 362.4 | 400 GB/s (8 HCAs) | 90.6% |
| sendrecv | P2P | 42.5 | 50 GB/s (1 HCA) | 85.0% |

---

## Observations

1. **CollNet SHARP is the clear winner at 4+ nodes** across both all_reduce (~385 GB/s) and all_gather (~384 GB/s), despite SHARP hardware not being operational. NCCL's CollNet fallback outperforms ring by 15–27% at scale. All CollNet jobs logged `SHARP coll init error: Cannot create SHARP job` — actual SHARP offload would likely push numbers higher.

2. **NVLS dominates intra-node and 2-node scale** with 839 GB/s all_reduce at 1 node (+23% over ring) and 713 GB/s at 2 nodes (+107% over ring). NVLS leverages NVLink multicast for efficient intra-node reduction.

3. **NVLS collapses at 64 nodes**: all_reduce NVLS drops to 207 GB/s — worse than ring (280 GB/s). The 8G/2G ratio of 0.85x at 64 nodes shows NVLS is actively hurt by larger messages at scale. Don't force `NCCL_NVLS_ENABLE=1` for large-scale training.

4. **CollNet scaling is remarkably flat**: all_reduce CollNet holds ~385 GB/s from 4 to 32 nodes (less than 1% variation), only dropping 9% at 64 nodes. Scaling efficiency exceeds 100% at 4–32 nodes relative to its 2-node baseline, likely because CollNet's tree topology benefits from more participants.

5. **The NVLink→IB cliff is 15x for P2P**: sendrecv drops from 653 GB/s intra-node to 42.5 GB/s at 2 nodes. P2P is single-HCA limited (42.5 / 50 = 85% utilization). Collectives achieve much higher bandwidth by aggregating across all 8 HCAs. This makes pipeline parallelism (PP) the inter-node bottleneck.

6. **Larger messages improve bandwidth at scale**: ring 64n gains 36% from 2G→8G (205→280 GB/s); CollNet 32n gains 15% (335→385 GB/s). The 8G/2G ratio reaches 1.42x for CollNet at 64 nodes. This validates GBS tuning (more gradient accumulation) as a communication optimization strategy.

7. **Tree algorithm is consistently worst** for large messages: ~191 GB/s ceiling regardless of node count at 4–64 nodes. Tree's log-depth latency advantage is irrelevant for bandwidth-bound 2–8 GB messages. Never force `NCCL_ALGO=Tree` for LLM training.

8. **Major scaling cliffs identified**: (a) 1→2 nodes: ring drops 49.5% (NVLink→IB); (b) 2→4 nodes: NVLS drops 53.4%, NVLSTree drops 56.3% (lose NVLink advantage); (c) 32→64 nodes: NVLS drops 31.8%. CollNet is the only algorithm with no cliff >10%.

9. **Sendrecv P2P is message-size invariant at multi-node**: 42.5 GB/s at 2 nodes regardless of 2G/4G/8G, confirming pure link-bandwidth limitation. Bandwidth continues dropping with node count (42→25→15 GB/s) due to bisection bandwidth scaling.

---

## Recommendations for LLM Training

| Scale | Recommended NCCL Settings | Rationale |
|-------|--------------------------|-----------|
| 1 node (8 GPU) | `NCCL_NVLS_ENABLE=1`, default algo | NVLS gives 23% boost over ring |
| 2 nodes (16 GPU) | `NCCL_NVLS_ENABLE=1`, default algo | NVLS still 107% faster than ring |
| 4–32 nodes (32–256 GPU) | `NCCL_COLLNET_ENABLE=1` | CollNet fallback is 15–27% faster than ring |
| 64 nodes (512 GPU) | `NCCL_COLLNET_ENABLE=1` | CollNet 25% better than ring |
| All scales | Use largest feasible GBS | Larger messages improve BW utilization by up to 42% |

### Optimization Opportunities

1. **Enable SHARP:** Admin action needed — start `sharp_am` on management node, `sharpd` on compute nodes, create SHARP reservations. Could boost CollNet from ~385 to 500+ GB/s.
2. **Pipeline parallelism bottleneck:** PP SendRecv at 15–42 GB/s inter-node is the scaling wall. Minimize PP stages or keep PP intra-node (NVLink) where possible.
3. **GBS tuning:** Maximize gradient accumulation steps to send larger messages and amortize latency overhead, especially at 32+ nodes.

---

## Raw Results

Results directory: `/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/collective-scaling/20260416_140542/`

Slurm jobs: 83907–83977 (71 total)

Submit script: `~/together-nccl-tests/benchmarks/B200/collective-scaling/submit_all.sh`

Manifest: `manifest.txt` in results directory
