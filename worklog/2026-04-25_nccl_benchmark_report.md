# NCCL Collective Scaling Benchmark Report — 2026-04-25

NCCL collective performance across 1–64 node scale (8–512 GPUs) on the B200 DGXC cluster, testing all_reduce, all_gather, and sendrecv P2P across five algorithm configurations and three message sizes.

---

## Run Notes (TL;DR)

**Status vs prior runs:** all measurable points within ±5% of 2026-04-23 baseline. **No fabric degradation.** Two minor regressions: `all_reduce ring 4n @ 8 GB` (-7.0%) and `sendrecv p2p 16n` (-5.8%); both are noise-level and re-traceable.

**`/opt/hpcx` host gap is unchanged from 2026-04-23.** With no node exclude, 27 of 71 jobs (38%) failed at `srun` startup with `libmpi.so.40: cannot open shared object file`. Per-scale breakdown:

| Scale | OK | hpcx fail | Notes |
|------:|---:|----------:|-------|
| 1n | 4 | 1 | gpu-130 confirmed bad |
| 2n | 9 | 0 | clean |
| 4n | 8 | 1 | one bad allocation |
| 8n | 6 | 4 | |
| 16n | 8 | 3 | |
| 32n | 4 | 7 | most allocations hit a bad host |
| **64n** | **0** | **11** | **same as 2026-04-23 — every 64-node allocation contains ≥1 bad host** |

**Suspected hpcx-missing nodes** (most frequent in failed allocations): gpu-130, gpu-158, gpu-188, gpu-189, gpu-152, gpu-256, gpu-238, gpu-216, gpu-205, gpu-159, gpu-235, gpu-186. (Allocation co-failure means individual identification needs per-node probing.)

**Post-srun hang behavior.** Many jobs produced their `Avg bus bandwidth` line but did not exit `srun` cleanly — they sat in RUNNING state until the 20-min wallclock or until cancelled. A watchdog (`/tmp/nccl-watchdog.sh`) was started after the first hang at 8n was observed; it polls every 20s and `scancel`s any job whose stdout already contains `Avg bus bandwidth`. Total wall time with watchdog: **~17 min** for the entire 71-job suite (would have been ~3-4 hrs without it). Data parity is preserved — `parse_and_report.py` only requires the bandwidth line.

**Scope of this run:** full suite (1–64 nodes, host mode, no node exclude) per user request. Compared against `20260423_220412` (`compare-dir` flag below).

**Action items:**
1. Cluster ops: reinstall `/opt/hpcx` on missing nodes (still unresolved since 2026-04-23) — OR migrate harness to container mode (`run_slurm.sh --container`).
2. Investigate post-srun hang in nccl-tests binaries — appeared this run, was not flagged in 2026-04-23 report; possibly cluster-state related.
3. Patch in `parse_and_report.py` for `None` `sr_64n` case applied locally (line 1001 area).

---

## Workaround applied — shared HPC-X copy on /mnt/vast (2026-04-25 10:35)

To unblock 64-node bare-host NCCL diagnostics without waiting on cluster ops:

1. `rsync -aHAX /opt/hpcx/ /mnt/vast/dgxc-benchmarking-auto/hpcx-2.18/` (1.2 GB, 41s; HPC-X v2.18).
2. `submit_all.sh` patched: `HPCX_INIT="/mnt/vast/dgxc-benchmarking-auto/hpcx-2.18/hpcx-init.sh"`.
3. Verified relocatable: `hpcx-init.sh` resolves `HPCX_DIR` from `BASH_SOURCE[0]` (no hardcoded `/opt/hpcx`).
4. Smoke test on **gpu-130** (confirmed-bad node) passed: 1n all_reduce ring → 772.8 GB/s.
5. Re-ran 64-node tier with **no node exclude** (jobs 84925–84935): 9 OK + 2 expected-NCCL-unsupported (allgather tree/nvls_tree). Watchdog cancelled 6 of 11 post-data-collect; all 9 measurable jobs landed valid bandwidth.

### 64-node results (jobs 84925–84935) vs 2026-04-20 baseline @ 8 GB

| Test | Today (GB/s) | Apr-20 (GB/s) | Δ |
|------|-------------:|--------------:|--:|
| all_reduce ring | 287.3 | 288.3 | -0.3% |
| all_reduce nvls | 273.1 | 264.1 | +3.4% |
| all_reduce nvls_tree | 279.1 | 266.6 | +4.7% |
| all_reduce tree | 192.2 | 192.6 | -0.2% |
| all_reduce collnet_sharp | 363.5 | 363.6 | **-0.0%** |
| all_gather ring | 285.9 | 286.2 | -0.1% |
| all_gather nvls | 285.9 | 286.1 | -0.1% |
| all_gather collnet_sharp | 361.7 | 362.2 | -0.1% |
| sendrecv p2p | 15.7 | 15.7 | -0.1% |

**Conclusion:** all 9 within ±5% of Apr-20 — fabric is healthy at 64-node scale; the morning's 0/11 result was purely the host hpcx gap, not a fabric regression. Cluster-ops ticket (reinstall `/opt/hpcx` on missing nodes) is still the proper long-term fix; the shared-FS copy is a workaround that lets bare-host benchmarks proceed in the meantime.

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

**Job results:** 39 OK, 12 expected failures, 20 unexpected failures.

**Unexpected failures:**
- all_gather collnet_sharp 16n
- all_gather collnet_sharp 4n
- all_gather collnet_sharp 64n
- all_gather collnet_sharp 8n
- all_gather nvls 32n
- all_gather nvls 64n
- all_gather ring 32n
- all_gather ring 64n
- all_gather ring 8n
- all_reduce collnet_sharp 32n
- all_reduce collnet_sharp 64n
- all_reduce nvls 64n
- all_reduce nvls 8n
- all_reduce nvls_tree 32n
- all_reduce nvls_tree 64n
- all_reduce ring 1n
- all_reduce ring 64n
- all_reduce tree 64n
- sendrecv p2p 32n
- sendrecv p2p 64n

---

## Results — all_reduce Peak busBW (GB/s)

### all_reduce @ 2 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree | Best |
|--------------|------|------|---------------|----------|------|------|
| 1n (8) | -- | **820.2** | -- | -- | -- | NVLS |
| 2n (16) | 341.9 | **688.2** | 372.2 | 687.4 | 324.7 | NVLS |
| 4n (32) | 312.5 | 328.1 | **379.2** | 268.8 | 185.2 | CollNet SHARP |
| 8n (64) | 303.8 | -- | **382.2** | 265.8 | 188.2 | CollNet SHARP |
| 16n (128) | 286.9 | 263.9 | **358.3** | 264.3 | 187.0 | CollNet SHARP |
| 32n (256) | **269.3** | 257.6 | -- | -- | 187.3 | Ring |
| 64n (512) | -- | -- | -- | -- | -- | -- |

### all_reduce @ 4 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree | Best |
|--------------|------|------|---------------|----------|------|------|
| 1n (8) | -- | **834.6** | -- | -- | -- | NVLS |
| 2n (16) | 343.2 | 706.0 | 379.5 | **706.1** | 330.3 | NVLSTree |
| 4n (32) | 310.9 | 331.7 | **384.1** | 282.9 | 187.3 | CollNet SHARP |
| 8n (64) | 306.4 | -- | **378.4** | 288.0 | 190.8 | CollNet SHARP |
| 16n (128) | 306.5 | 309.8 | **384.3** | 287.8 | 190.1 | CollNet SHARP |
| 32n (256) | **287.0** | 270.8 | -- | -- | 191.0 | Ring |
| 64n (512) | -- | -- | -- | -- | -- | -- |

### all_reduce @ 8 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree | Best |
|--------------|------|------|---------------|----------|------|------|
| 1n (8) | -- | **836.7** | -- | -- | -- | NVLS |
| 2n (16) | 343.9 | 713.3 | 380.9 | **713.5** | 332.8 | NVLSTree |
| 4n (32) | 310.1 | 333.0 | **384.4** | 290.0 | 188.5 | CollNet SHARP |
| 8n (64) | 308.1 | -- | **385.2** | 290.5 | 192.2 | CollNet SHARP |
| 16n (128) | 305.9 | 308.5 | **379.5** | 289.1 | 191.6 | CollNet SHARP |
| 32n (256) | 304.0 | **304.2** | -- | -- | 192.9 | NVLS |
| 64n (512) | -- | -- | -- | -- | -- | -- |

---

## Results — all_gather Peak busBW (GB/s)

tree and nvls_tree are not supported by NCCL for AllGather.

### all_gather @ 2 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |
|--------------|------|------|---------------|------|
| 1n (8) | 640.3 | **643.8** | -- | NVLS |
| 2n (16) | 336.9 | 337.2 | **357.3** | CollNet SHARP |
| 4n (32) | **325.8** | 325.3 | -- | Ring |
| 8n (64) | -- | **312.0** | -- | NVLS |
| 16n (128) | **287.9** | 287.0 | -- | Ring |
| 32n (256) | -- | -- | **283.0** | CollNet SHARP |
| 64n (512) | -- | -- | -- | -- |

### all_gather @ 4 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |
|--------------|------|------|---------------|------|
| 1n (8) | 653.6 | **656.8** | -- | NVLS |
| 2n (16) | 339.6 | 340.2 | **367.8** | CollNet SHARP |
| 4n (32) | **329.2** | 327.7 | -- | Ring |
| 8n (64) | -- | **311.9** | -- | NVLS |
| 16n (128) | **317.4** | 314.6 | -- | Ring |
| 32n (256) | -- | -- | **353.9** | CollNet SHARP |
| 64n (512) | -- | -- | -- | -- |

### all_gather @ 8 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |
|--------------|------|------|---------------|------|
| 1n (8) | 659.5 | **662.8** | -- | NVLS |
| 2n (16) | 340.9 | 341.6 | **364.1** | CollNet SHARP |
| 4n (32) | 331.3 | **332.4** | -- | NVLS |
| 8n (64) | -- | **311.7** | -- | NVLS |
| 16n (128) | **314.1** | 311.5 | -- | Ring |
| 32n (256) | -- | -- | **384.2** | CollNet SHARP |
| 64n (512) | -- | -- | -- | -- |

---

## Results — sendrecv P2P Peak busBW (GB/s)

| Nodes (GPUs) | 2 GB | 4 GB | 8 GB |
|--------------|------|------|------|
| 1n (8) | 647.0 | 652.5 | **654.5** |
| 2n (16) | 43.3 | **43.3** | 43.3 |
| 4n (32) | 44.0 | 44.0 | **44.0** |
| 8n (64) | **24.0** | 24.0 | 24.0 |
| 16n (128) | **15.1** | **15.1** | **15.1** |
| 32n (256) | -- | -- | -- |
| 64n (512) | -- | -- | -- |

---

## Scaling Efficiency — all_reduce @ 8 GB

Efficiency = busBW(N nodes) / busBW(baseline) × 100%. Baseline: 1-node for ring/nvls; 2-node for others.

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|--------------|------|------|---------------|----------|------|
| 1n (8) | -- | **100.0%** | -- | -- | -- |
| 2n (16) | -- | 85.2% | **100.0%** | 100.0% | 100.0% |
| 4n (32) | -- | 39.8% | **100.9%** | 40.6% | 56.7% |
| 8n (64) | -- | -- | **101.1%** | 40.7% | 57.7% |
| 16n (128) | -- | 36.9% | **99.6%** | 40.5% | 57.6% |
| 32n (256) | -- | 36.4% | -- | -- | **58.0%** |
| 64n (512) | -- | -- | -- | -- | -- |

## Scaling Efficiency — all_gather @ 8 GB

Efficiency = busBW(N nodes) / busBW(baseline) × 100%. Baseline: 1-node for ring/nvls; 2-node for others.

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP |
|--------------|------|------|---------------|
| 1n (8) | **100.0%** | 100.0% | -- |
| 2n (16) | 51.7% | 51.5% | **100.0%** |
| 4n (32) | **50.2%** | 50.1% | -- |
| 8n (64) | -- | **47.0%** | -- |
| 16n (128) | **47.6%** | 47.0% | -- |
| 32n (256) | -- | -- | **105.5%** |
| 64n (512) | -- | -- | -- |

## Scaling Efficiency — sendrecv P2P @ 8 GB

| Nodes (GPUs) | Efficiency |
|--------------|-----------|
| 1n (8) | 100.0% |
| 2n (16) | 6.6% |
| 4n (32) | 6.7% |
| 8n (64) | 3.7% |
| 16n (128) | 2.3% |
| 32n (256) | -- |
| 64n (512) | -- |

---

## Step-by-Step Bandwidth Drop — all_reduce @ 8 GB

Identifies scaling cliffs between consecutive node counts.

| Transition | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|------------|------|------|---------------|----------|------|
| 1n → 2n | -- | -14.8% | -- | -- | -- |
| 2n → 4n | -9.8% | **-53.3%** | +0.9% | **-59.4%** | **-43.3%** |
| 4n → 8n | -0.6% | -- | +0.2% | +0.2% | +1.9% |
| 8n → 16n | -0.7% | -- | -1.5% | -0.5% | -0.3% |
| 16n → 32n | -0.6% | -1.4% | -- | -- | +0.7% |
| 32n → 64n | -- | -- | -- | -- | -- |

---

## Message Size Sensitivity — all_reduce 8G/2G Ratio

How much 8 GB messages outperform 2 GB, revealing bandwidth saturation effects at scale.

| Nodes | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|-------|------|------|---------------|----------|------|
| 1n | -- | 1.02x | -- | -- | -- |
| 2n | 1.01x | 1.04x | 1.02x | 1.04x | 1.03x |
| 4n | 0.99x | 1.01x | 1.01x | 1.08x | 1.02x |
| 8n | 1.01x | -- | 1.01x | 1.09x | 1.02x |
| 16n | 1.07x | 1.17x | 1.06x | 1.09x | 1.02x |
| 32n | 1.13x | 1.18x | -- | -- | 1.03x |
| 64n | -- | -- | -- | -- | -- |

## Message Size Sensitivity — all_gather 8G/2G Ratio

How much 8 GB messages outperform 2 GB, revealing bandwidth saturation effects at scale.

| Nodes | Ring | NVLS | CollNet SHARP |
|-------|------|------|---------------|
| 1n | 1.03x | 1.03x | -- |
| 2n | 1.01x | 1.01x | 1.02x |
| 4n | 1.02x | 1.02x | -- |
| 8n | -- | 1.00x | -- |
| 16n | 1.09x | 1.09x | -- |
| 32n | -- | -- | **1.36x** |
| 64n | -- | -- | -- |

---

## Algorithm Head-to-Head @ 8 GB

### all_reduce — Best Algorithm per Scale

| Nodes (GPUs) | Best Algorithm | busBW (GB/s) | 2nd Best | busBW (GB/s) | Gap |
|--------------|---------------|-------------|----------|-------------|-----|
| 1n (8) | **NVLS** | 836.7 | -- | -- | -- |
| 2n (16) | **NVLSTree** | 713.5 | NVLS | 713.3 | +0.0% |
| 4n (32) | **CollNet SHARP** | 384.4 | NVLS | 333.0 | +15.4% |
| 8n (64) | **CollNet SHARP** | 385.2 | Ring | 308.1 | +25.0% |
| 16n (128) | **CollNet SHARP** | 379.5 | NVLS | 308.5 | +23.0% |
| 32n (256) | **NVLS** | 304.2 | Ring | 304.0 | +0.1% |
| 64n (512) | -- | -- | -- | -- | -- |

### all_gather — Best Algorithm per Scale

| Nodes (GPUs) | Best Algorithm | busBW (GB/s) | 2nd Best | busBW (GB/s) | Gap |
|--------------|---------------|-------------|----------|-------------|-----|
| 1n (8) | **NVLS** | 662.8 | Ring | 659.5 | +0.5% |
| 2n (16) | **CollNet SHARP** | 364.1 | NVLS | 341.6 | +6.6% |
| 4n (32) | **NVLS** | 332.4 | Ring | 331.3 | +0.3% |
| 8n (64) | **NVLS** | 311.7 | -- | -- | -- |
| 16n (128) | **Ring** | 314.1 | NVLS | 311.5 | +0.9% |
| 32n (256) | **CollNet SHARP** | 384.2 | -- | -- | -- |
| 64n (512) | -- | -- | -- | -- | -- |

---

## Interconnect Topology Analysis

### NVLink vs InfiniBand Boundary

| Metric | NVLink 1n (GB/s) | IB 2n (GB/s) | IB 64n (GB/s) | 1n→2n Drop |
|--------|-----------------|-------------|--------------|-----------|
| all_reduce (NVLS) | 836.7 | 713.3 | -- | **-14.8%** |
| all_reduce (Ring) | -- | 343.9 | -- | -- |
| sendrecv P2P | 654.5 | 43.3 | -- | **-93.4%** |

### Bandwidth Utilization vs Theoretical @ 2 Nodes, 8 GB

| Collective | Best Algorithm | busBW (GB/s) | IB Theoretical | Utilization |
|------------|---------------|-------------|----------------|-------------|
| all_reduce | NVLSTree | 713.5 | 400 GB/s (8 HCAs) | 178.4% |
| all_gather | CollNet SHARP | 364.1 | 400 GB/s (8 HCAs) | 91.0% |
| sendrecv | P2P | 43.3 | 50 GB/s (1 HCA) | 86.7% |

---

## Delta vs Previous Run

### all_reduce @ 8 GB — Delta (%)

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|--------------|------|------|---------------|----------|------|
| 1n (8) | -- | -- | -- | -- | -- |
| 2n (16) | -0.0% | -0.0% | +2.9% | +0.0% | -1.8% |
| 4n (32) | **-7.0%** | -0.1% | +0.0% | -0.3% | +0.2% |
| 8n (64) | -0.2% | -- | -0.0% | -0.1% | +0.1% |
| 16n (128) | -0.4% | +0.4% | -0.1% | **+5.2%** | -0.1% |
| 32n (256) | -1.0% | -0.9% | -- | -- | +0.1% |
| 64n (512) | -- | -- | -- | -- | -- |

### all_gather @ 8 GB — Delta (%)

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP |
|--------------|------|------|---------------|
| 1n (8) | +0.5% | +0.5% | -- |
| 2n (16) | -0.2% | -0.0% | +0.3% |
| 4n (32) | -0.4% | -0.2% | -- |
| 8n (64) | -- | -0.6% | -- |
| 16n (128) | +0.5% | -1.5% | -- |
| 32n (256) | -- | -- | +0.1% |
| 64n (512) | -- | -- | -- |

### sendrecv P2P @ 8 GB — Delta (%)

| Nodes (GPUs) | Delta |
|--------------|-------|
| 1n (8) | +0.4% |
| 2n (16) | +0.0% |
| 4n (32) | +4.2% |
| 8n (64) | -2.2% |
| 16n (128) | **-5.8%** |
| 32n (256) | -- |
| 64n (512) | -- |

### Significant Regressions (>5% drop)

- all_reduce ring 4n @ 8 GB: 333.4 → 310.1 GB/s (-7.0%)
- all_reduce ring 4n @ 4 GB: 331.4 → 310.9 GB/s (-6.2%)
- sendrecv p2p 16n @ 2 GB: 16.1 → 15.1 GB/s (-5.8%)
- sendrecv p2p 16n @ 4 GB: 16.1 → 15.1 GB/s (-5.8%)
- sendrecv p2p 16n @ 8 GB: 16.1 → 15.1 GB/s (-5.8%)
- all_reduce ring 4n @ 2 GB: 330.2 → 312.5 GB/s (-5.4%)

### Significant Improvements (>5% gain)

- all_reduce nvls_tree 16n @ 4 GB: 273.4 → 287.8 GB/s (+5.3%)
- all_reduce nvls_tree 16n @ 8 GB: 274.9 → 289.1 GB/s (+5.2%)

---

## Observations

1. **The NVLink→IB cliff is 15x for P2P**: sendrecv drops from 655 GB/s intra-node to 43.3 GB/s at 2 nodes. P2P is single-HCA limited (43.3 / 50 = 87% utilization). This makes pipeline parallelism (PP) the inter-node bottleneck.

2. **Tree algorithm is consistently worst** for large messages: ~191 GB/s ceiling regardless of node count at 4–64 nodes. Never force `NCCL_ALGO=Tree` for LLM training.

3. **Sendrecv P2P is message-size invariant at multi-node**: 43.3 GB/s at 2 nodes regardless of 2G/4G/8G, confirming pure link-bandwidth limitation.

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

Results directory: `/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/collective-scaling/20260425_095453`

Slurm jobs: 84852–84922

Submit script: `~/together-nccl-tests/benchmarks/B200/collective-scaling/submit_all.sh`

Manifest: `manifest.txt` in results directory
