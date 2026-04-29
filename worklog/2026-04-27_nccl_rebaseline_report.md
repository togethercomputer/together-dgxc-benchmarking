# NCCL Collective Scaling Benchmark Report — 2026-04-27

NCCL collective performance across 1–64 node scale (8–512 GPUs) on the B200 DGXC cluster, testing all_reduce, all_gather, and sendrecv P2P across five algorithm configurations and three message sizes.

## Session Notes (overrides auto-generated text below)

**Headline:** SHARP is operational and stable. all_reduce CollNet SHARP gains **+40 to +49%** vs the 04-25 baseline at every scale 2n–64n. Ring/NVLS/NVLSTree/Tree are flat (±2%), so the SHARP shift is the sole driver — not a stack-wide change.

| Scale | CollNet SHARP Δ vs 04-25 (8 GB all_reduce) |
|---:|:---|
| 2n | +42.1% |
| 4n | +41.1% |
| 8n | +41.3% |
| 16n | +43.7% |
| 32n | +41.5% |
| **64n** | **+48.9%** |

**Correction to auto-generated text below:**
- "SHARP | Not available (sharpd not running, no reservations)" — **wrong**, kept from a stale template. SHARP is operational. `all_reduce_collnet_sharp_64nodes.out` confirms `sharp_job_id:1`, `tree_type:LLT`/`SAT` reservations active. This is real SHARP offload, not CollNet fallback.
- Observation 1's "All CollNet jobs logged SHARP coll init error" is also from the stale template and does not match today's logs.

**Run details:**
- Initial sweep: jobs 85344-85414 (71 jobs); 32n + 64n cut by:
  - watchdog 1200s timeout (long jobs killed mid-run)
  - exclude list `[197,201,211]` blocking 3 batch-idle nodes → 64n could never fit (only 61 schedulable)
- Resubmit: jobs 85415-85428 (14 jobs) with `--exclude=[133]` only, watchdog timeout bumped to 3600s
- Both passes used identical NCCL stack (2.29.7+cuda12.9, HPC-X 2.18) — results below are merged into a single dataset
- 10 expected failures: NCCL does not support `all_gather` × `tree`/`nvls_tree` (silent gap in `submit_all.sh` — should be skipped, not submitted)

**Comparison vs 04-09 canonical baseline** (`~/reports/NCCL_Benchmark_Report_20260409.md`):
- 04-09 64n SHARP all_reduce peak: 363.66 GB/s
- 04-27 64n SHARP all_reduce peak: **542.1 GB/s @ 8GB (+49.1%)**
- This re-baselines the 04-09 report. Future regression checks should compare against `20260427_132654`, not `20260409`.

**Open items:**
- `submit_all.sh` should skip the invalid `all_gather` × `tree`/`nvls_tree` matrix entries (10 wasted jobs per sweep).
- `parse_and_report.py` has hard-coded "SHARP not available" boilerplate — should detect from the SHARP init log lines instead.

## Comparison vs 2026-04-26 Per-Group Sweep

Yesterday's `submit_per4_groups.sh` sweep was all_reduce only, but disjointly partitioned every idle node into K-tuples and reported per-group/median results. Today's `submit_all.sh` is the full 3-op × 5-algo matrix with each (op,algo,N) drawn as a single Slurm allocation. Apples-to-apples on `Avg bus bandwidth` (averaged across 2/4/8/16 GB sizes):

| Scale | Today Ring | 04-26 Ring | Δ Ring | Today SHARP | 04-26 SHARP | Δ SHARP | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| 4n | 310.9 | 332.0 healthy median | −6.4% | 507.9 | 514.0 healthy median | −1.2% | today's job landed on 217/226/227 (yesterday's slow-cohort) |
| 8n | 308.1 | 320.5 | −3.9% | 511.0 | 521.4 | −2.0% | today included 168-171 + 229-230 |
| 16n | 302.2 | 316.5 | −4.5% | 512.3 | 522.4 | −1.9% | single random allocation, may straddle suspect nodes |
| 32n A | 291.0 | 294.2 control | −1.1% | 512.6 | 512.3 control | +0.06% | both used full pool — match |
| 64n | 271.6 | 271.5 | +0.04% | 511.3 | 511.4 | −0.02% | full pool both runs — identical |

**Per-size verification at 64n** (today vs yesterday, GB/s):

| Size | Today SHARP | 04-26 SHARP | Today Ring | 04-26 Ring |
|---:|---:|---:|---:|---:|
| 2 GB | 415.0 | 415.1 | 223.3 | 223.1 |
| 4 GB | 541.0 | 541.3 | 269.2 | 269.6 |
| 8 GB | 542.1 | 542.4 | 287.9 | 287.8 |
| 16 GB | 544.5 | 544.3 | 305.3 | 305.2 |

Same machine producing the same numbers twice — within 0.1% across every size and both algos.

### Interpretation

- **At 32n+: today and yesterday are identical** (≤0.1% delta on SHARP and ring). The +40% SHARP shift first seen 04-26 vs 04-25 is fully reproducible and stable.
- **At ≤16n: today's lower numbers are node-placement noise, not a regression.** Yesterday reported the median of 13 healthy 4-node groups; today's single 4n allocation happened to draw from the slow cohort (`use3a-ss-b200-gpu-[216,217,226,227]` — three of those are in yesterday's 12 "4n-slow" list). The 1-2% SHARP gap exactly matches yesterday's per-group ratios for slow-cohort nodes (506-508 vs 514 healthy).
- **Practical takeaway:** Use 04-26 worklog for per-node fabric diagnostic data; use **04-27 (this report)** as the canonical regression baseline going forward.

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
| Excluded | use3a-ss-b200-gpu-[133,190,197,199,201,211,233,239] |

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

**Expected failures:** tree and nvls_tree are not supported by NCCL for AllGather (10 jobs failed as expected).

**Job results:** 61 OK, 10 expected failures, 0 unexpected failures.

---

## Results — all_reduce Peak busBW (GB/s)

### all_reduce @ 2 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree | Best |
|--------------|------|------|---------------|----------|------|------|
| 1n (8) | 670.9 | **815.9** | -- | -- | -- | NVLS |
| 2n (16) | 342.3 | 686.8 | 390.7 | **687.5** | 325.2 | NVLSTree |
| 4n (32) | 310.0 | 328.8 | **404.0** | 268.6 | 184.3 | CollNet SHARP |
| 8n (64) | 306.9 | 306.0 | **410.2** | 268.0 | 187.2 | CollNet SHARP |
| 16n (128) | 287.8 | 265.8 | **412.8** | 262.9 | 187.4 | CollNet SHARP |
| 32n (256) | 269.1 | 258.7 | **412.9** | 258.6 | 187.4 | CollNet SHARP |
| 64n (512) | 223.3 | 255.9 | **415.0** | 255.5 | 185.9 | CollNet SHARP |

### all_reduce @ 4 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree | Best |
|--------------|------|------|---------------|----------|------|------|
| 1n (8) | 680.0 | **831.6** | -- | -- | -- | NVLS |
| 2n (16) | 343.6 | 705.6 | 534.9 | **705.9** | 330.6 | NVLSTree |
| 4n (32) | 309.8 | 331.1 | **540.7** | 287.0 | 186.3 | CollNet SHARP |
| 8n (64) | 308.2 | 306.5 | **543.5** | 287.0 | 189.7 | CollNet SHARP |
| 16n (128) | 309.5 | 304.3 | **544.0** | 277.5 | 190.5 | CollNet SHARP |
| 32n (256) | 287.1 | 271.6 | **544.7** | 272.7 | 191.1 | CollNet SHARP |
| 64n (512) | 269.2 | 271.7 | **541.0** | 271.9 | 190.0 | CollNet SHARP |

### all_reduce @ 8 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree | Best |
|--------------|------|------|---------------|----------|------|------|
| 1n (8) | 686.6 | **836.2** | -- | -- | -- | NVLS |
| 2n (16) | 344.3 | 713.1 | 542.9 | **713.4** | 332.7 | NVLSTree |
| 4n (32) | 311.2 | 333.3 | **542.3** | 289.5 | 187.5 | CollNet SHARP |
| 8n (64) | 308.6 | 307.6 | **544.2** | 290.3 | 191.2 | CollNet SHARP |
| 16n (128) | 306.6 | 303.8 | **545.3** | 280.1 | 192.1 | CollNet SHARP |
| 32n (256) | 304.0 | 304.5 | **545.1** | 276.8 | 192.9 | CollNet SHARP |
| 64n (512) | 287.9 | 274.2 | **542.1** | 275.4 | 192.2 | CollNet SHARP |

---

## Results — all_gather Peak busBW (GB/s)

tree and nvls_tree are not supported by NCCL for AllGather.

### all_gather @ 2 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |
|--------------|------|------|---------------|------|
| 1n (8) | **636.9** | 623.9 | -- | Ring |
| 2n (16) | 337.5 | 338.9 | **356.9** | CollNet SHARP |
| 4n (32) | 323.6 | 325.9 | **373.6** | CollNet SHARP |
| 8n (64) | 311.8 | 311.0 | **378.1** | CollNet SHARP |
| 16n (128) | 286.1 | 286.1 | **352.2** | CollNet SHARP |
| 32n (256) | 232.3 | 234.2 | **330.8** | CollNet SHARP |
| 64n (512) | 224.8 | 225.7 | **324.1** | CollNet SHARP |

### all_gather @ 4 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |
|--------------|------|------|---------------|------|
| 1n (8) | **647.4** | 643.3 | -- | Ring |
| 2n (16) | 340.4 | 341.7 | **367.4** | CollNet SHARP |
| 4n (32) | 330.3 | 328.2 | **381.9** | CollNet SHARP |
| 8n (64) | 315.4 | 311.3 | **376.1** | CollNet SHARP |
| 16n (128) | 310.2 | 313.5 | **382.3** | CollNet SHARP |
| 32n (256) | 286.7 | 286.2 | **355.5** | CollNet SHARP |
| 64n (512) | 231.9 | 230.8 | **335.4** | CollNet SHARP |

### all_gather @ 8 GB

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |
|--------------|------|------|---------------|------|
| 1n (8) | **652.9** | 649.7 | -- | Ring |
| 2n (16) | 341.4 | 343.1 | **362.9** | CollNet SHARP |
| 4n (32) | 332.3 | 332.4 | **383.0** | CollNet SHARP |
| 8n (64) | 315.0 | 311.3 | **384.0** | CollNet SHARP |
| 16n (128) | 308.8 | 314.3 | **378.0** | CollNet SHARP |
| 32n (256) | 313.2 | 312.2 | **384.2** | CollNet SHARP |
| 64n (512) | 285.7 | 286.1 | **361.2** | CollNet SHARP |

---

## Results — sendrecv P2P Peak busBW (GB/s)

| Nodes (GPUs) | 2 GB | 4 GB | 8 GB |
|--------------|------|------|------|
| 1n (8) | 648.1 | 653.3 | **655.5** |
| 2n (16) | 43.3 | 43.3 | **43.3** |
| 4n (32) | 43.2 | 43.2 | **43.2** |
| 8n (64) | 24.0 | 24.0 | **24.5** |
| 16n (128) | 14.7 | 14.7 | **14.8** |
| 32n (256) | **15.7** | **15.7** | **15.7** |
| 64n (512) | 15.7 | **15.7** | **15.7** |

---

## Scaling Efficiency — all_reduce @ 8 GB

Efficiency = busBW(N nodes) / busBW(baseline) × 100%. Baseline: 1-node for ring/nvls; 2-node for others.

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|--------------|------|------|---------------|----------|------|
| 1n (8) | **100.0%** | 100.0% | -- | -- | -- |
| 2n (16) | 50.1% | 85.3% | **100.0%** | 100.0% | 100.0% |
| 4n (32) | 45.3% | 39.9% | **99.9%** | 40.6% | 56.4% |
| 8n (64) | 44.9% | 36.8% | **100.2%** | 40.7% | 57.5% |
| 16n (128) | 44.7% | 36.3% | **100.4%** | 39.3% | 57.7% |
| 32n (256) | 44.3% | 36.4% | **100.4%** | 38.8% | 58.0% |
| 64n (512) | 41.9% | 32.8% | **99.9%** | 38.6% | 57.8% |

## Scaling Efficiency — all_gather @ 8 GB

Efficiency = busBW(N nodes) / busBW(baseline) × 100%. Baseline: 1-node for ring/nvls; 2-node for others.

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP |
|--------------|------|------|---------------|
| 1n (8) | **100.0%** | 100.0% | -- |
| 2n (16) | 52.3% | 52.8% | **100.0%** |
| 4n (32) | 50.9% | 51.2% | **105.5%** |
| 8n (64) | 48.3% | 47.9% | **105.8%** |
| 16n (128) | 47.3% | 48.4% | **104.1%** |
| 32n (256) | 48.0% | 48.0% | **105.8%** |
| 64n (512) | 43.8% | 44.0% | **99.5%** |

## Scaling Efficiency — sendrecv P2P @ 8 GB

| Nodes (GPUs) | Efficiency |
|--------------|-----------|
| 1n (8) | 100.0% |
| 2n (16) | 6.6% |
| 4n (32) | 6.6% |
| 8n (64) | 3.7% |
| 16n (128) | 2.3% |
| 32n (256) | 2.4% |
| 64n (512) | 2.4% |

---

## Step-by-Step Bandwidth Drop — all_reduce @ 8 GB

Identifies scaling cliffs between consecutive node counts.

| Transition | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|------------|------|------|---------------|----------|------|
| 1n → 2n | **-49.9%** | -14.7% | -- | -- | -- |
| 2n → 4n | -9.6% | **-53.3%** | -0.1% | **-59.4%** | **-43.6%** |
| 4n → 8n | -0.8% | -7.7% | +0.3% | +0.3% | +2.0% |
| 8n → 16n | -0.6% | -1.3% | +0.2% | -3.5% | +0.4% |
| 16n → 32n | -0.8% | +0.2% | -0.0% | -1.2% | +0.4% |
| 32n → 64n | -5.3% | -9.9% | -0.5% | -0.5% | -0.3% |

---

## Message Size Sensitivity — all_reduce 8G/2G Ratio

How much 8 GB messages outperform 2 GB, revealing bandwidth saturation effects at scale.

| Nodes | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|-------|------|------|---------------|----------|------|
| 1n | 1.02x | 1.02x | -- | -- | -- |
| 2n | 1.01x | 1.04x | **1.39x** | 1.04x | 1.02x |
| 4n | 1.00x | 1.01x | **1.34x** | 1.08x | 1.02x |
| 8n | 1.01x | 1.01x | **1.33x** | 1.08x | 1.02x |
| 16n | 1.07x | 1.14x | **1.32x** | 1.07x | 1.02x |
| 32n | 1.13x | 1.18x | **1.32x** | 1.07x | 1.03x |
| 64n | 1.29x | 1.07x | **1.31x** | 1.08x | 1.03x |

## Message Size Sensitivity — all_gather 8G/2G Ratio

How much 8 GB messages outperform 2 GB, revealing bandwidth saturation effects at scale.

| Nodes | Ring | NVLS | CollNet SHARP |
|-------|------|------|---------------|
| 1n | 1.03x | 1.04x | -- |
| 2n | 1.01x | 1.01x | 1.02x |
| 4n | 1.03x | 1.02x | 1.03x |
| 8n | 1.01x | 1.00x | 1.02x |
| 16n | 1.08x | 1.10x | 1.07x |
| 32n | **1.35x** | **1.33x** | 1.16x |
| 64n | 1.27x | 1.27x | 1.11x |

---

## Algorithm Head-to-Head @ 8 GB

### all_reduce — Best Algorithm per Scale

| Nodes (GPUs) | Best Algorithm | busBW (GB/s) | 2nd Best | busBW (GB/s) | Gap |
|--------------|---------------|-------------|----------|-------------|-----|
| 1n (8) | **NVLS** | 836.2 | Ring | 686.6 | +21.8% |
| 2n (16) | **NVLSTree** | 713.4 | NVLS | 713.1 | +0.0% |
| 4n (32) | **CollNet SHARP** | 542.3 | NVLS | 333.3 | +62.7% |
| 8n (64) | **CollNet SHARP** | 544.2 | Ring | 308.6 | +76.4% |
| 16n (128) | **CollNet SHARP** | 545.3 | Ring | 306.6 | +77.8% |
| 32n (256) | **CollNet SHARP** | 545.1 | NVLS | 304.5 | +79.0% |
| 64n (512) | **CollNet SHARP** | 542.1 | Ring | 287.9 | +88.3% |

### all_gather — Best Algorithm per Scale

| Nodes (GPUs) | Best Algorithm | busBW (GB/s) | 2nd Best | busBW (GB/s) | Gap |
|--------------|---------------|-------------|----------|-------------|-----|
| 1n (8) | **Ring** | 652.9 | NVLS | 649.7 | +0.5% |
| 2n (16) | **CollNet SHARP** | 362.9 | NVLS | 343.1 | +5.8% |
| 4n (32) | **CollNet SHARP** | 383.0 | NVLS | 332.4 | +15.2% |
| 8n (64) | **CollNet SHARP** | 384.0 | Ring | 315.0 | +21.9% |
| 16n (128) | **CollNet SHARP** | 378.0 | NVLS | 314.3 | +20.3% |
| 32n (256) | **CollNet SHARP** | 384.2 | Ring | 313.2 | +22.7% |
| 64n (512) | **CollNet SHARP** | 361.2 | NVLS | 286.1 | +26.3% |

---

## Interconnect Topology Analysis

### NVLink vs InfiniBand Boundary

| Metric | NVLink 1n (GB/s) | IB 2n (GB/s) | IB 64n (GB/s) | 1n→2n Drop |
|--------|-----------------|-------------|--------------|-----------|
| all_reduce (NVLS) | 836.2 | 713.1 | 274.2 | **-14.7%** |
| all_reduce (Ring) | 686.6 | 344.3 | 287.9 | **-49.9%** |
| sendrecv P2P | 655.5 | 43.3 | 15.7 | **-93.4%** |

### Bandwidth Utilization vs Theoretical @ 2 Nodes, 8 GB

| Collective | Best Algorithm | busBW (GB/s) | IB Theoretical | Utilization |
|------------|---------------|-------------|----------------|-------------|
| all_reduce | NVLSTree | 713.4 | 400 GB/s (8 HCAs) | 178.3% |
| all_gather | CollNet SHARP | 362.9 | 400 GB/s (8 HCAs) | 90.7% |
| sendrecv | P2P | 43.3 | 50 GB/s (1 HCA) | 86.6% |

---

## Delta vs Previous Run

### all_reduce @ 8 GB — Delta (%)

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |
|--------------|------|------|---------------|----------|------|
| 1n (8) | +0.0% | -0.1% | -- | -- | -- |
| 2n (16) | +0.3% | -0.1% | **+42.1%** | -0.0% | -0.8% |
| 4n (32) | -0.5% | +0.2% | **+41.1%** | +0.8% | -0.1% |
| 8n (64) | +1.5% | +0.5% | **+41.3%** | -0.4% | -0.3% |
| 16n (128) | -0.3% | -1.5% | **+43.7%** | +0.7% | -0.5% |
| 32n (256) | -0.7% | -0.5% | **+41.5%** | +1.1% | +0.1% |
| 64n (512) | +0.1% | -1.4% | **+48.9%** | -0.5% | +0.2% |

### all_gather @ 8 GB — Delta (%)

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP |
|--------------|------|------|---------------|
| 1n (8) | -1.3% | -2.0% | -- |
| 2n (16) | +0.1% | +0.4% | -0.0% |
| 4n (32) | -0.5% | -0.3% | +0.0% |
| 8n (64) | +0.3% | -0.6% | +0.0% |
| 16n (128) | +1.0% | +1.4% | +0.1% |
| 32n (256) | -0.8% | +0.6% | +0.0% |
| 64n (512) | -0.1% | +0.0% | -0.0% |

### sendrecv P2P @ 8 GB — Delta (%)

| Nodes (GPUs) | Delta |
|--------------|-------|
| 1n (8) | +0.2% |
| 2n (16) | +0.0% |
| 4n (32) | -2.0% |
| 8n (64) | -1.6% |
| 16n (128) | +0.6% |
| 32n (256) | +2.2% |
| 64n (512) | +0.0% |

### Significant Regressions (>5% drop)

- all_reduce nvls 32n @ 4 GB: 287.4 → 271.6 GB/s (-5.5%)

### Significant Improvements (>5% gain)

- all_reduce collnet_sharp 64n @ 4 GB: 337.4 → 541.0 GB/s (+60.4%)
- all_reduce collnet_sharp 64n @ 2 GB: 274.5 → 415.0 GB/s (+51.2%)
- all_reduce collnet_sharp 32n @ 4 GB: 361.9 → 544.7 GB/s (+50.5%)
- all_reduce collnet_sharp 64n @ 8 GB: 364.0 → 542.1 GB/s (+48.9%)
- all_reduce collnet_sharp 16n @ 8 GB: 379.5 → 545.3 GB/s (+43.7%)
- all_reduce collnet_sharp 8n @ 4 GB: 378.4 → 543.5 GB/s (+43.6%)
- all_reduce collnet_sharp 2n @ 8 GB: 382.0 → 542.9 GB/s (+42.1%)
- all_reduce collnet_sharp 16n @ 4 GB: 384.3 → 544.0 GB/s (+41.6%)
- all_reduce collnet_sharp 32n @ 8 GB: 385.2 → 545.1 GB/s (+41.5%)
- all_reduce collnet_sharp 2n @ 4 GB: 378.2 → 534.9 GB/s (+41.4%)
- all_reduce collnet_sharp 8n @ 8 GB: 385.2 → 544.2 GB/s (+41.3%)
- all_reduce collnet_sharp 4n @ 8 GB: 384.4 → 542.3 GB/s (+41.1%)
- all_reduce collnet_sharp 4n @ 4 GB: 384.2 → 540.7 GB/s (+40.8%)
- all_reduce collnet_sharp 32n @ 2 GB: 336.6 → 412.9 GB/s (+22.6%)
- all_gather collnet_sharp 64n @ 4 GB: 281.0 → 335.4 GB/s (+19.4%)
- all_gather collnet_sharp 64n @ 2 GB: 274.7 → 324.1 GB/s (+18.0%)
- all_gather collnet_sharp 32n @ 2 GB: 282.8 → 330.8 GB/s (+17.0%)
- all_reduce collnet_sharp 16n @ 2 GB: 358.3 → 412.8 GB/s (+15.2%)
- all_reduce collnet_sharp 8n @ 2 GB: 382.6 → 410.2 GB/s (+7.2%)
- all_reduce collnet_sharp 4n @ 2 GB: 379.3 → 404.0 GB/s (+6.5%)

---

## Observations

1. **CollNet SHARP is the clear winner at 4+ nodes** across both all_reduce (~542 GB/s) and all_gather, despite SHARP hardware not being operational. NCCL's CollNet fallback outperforms ring by 79% at 32 nodes. All CollNet jobs logged `SHARP coll init error: Cannot create SHARP job` — actual SHARP offload would likely push numbers higher.

2. **NVLS dominates intra-node and 2-node scale** with 836 GB/s all_reduce at 1 node (+22% over ring) and 713 GB/s at 2 nodes. NVLS leverages NVLink multicast for efficient intra-node reduction.

3. **NVLS collapses at 64 nodes**: all_reduce NVLS drops to 274 GB/s — worse than ring (288 GB/s). Don't force `NCCL_NVLS_ENABLE=1` for large-scale training.

4. **CollNet scaling is remarkably flat**: all_reduce CollNet holds ~542 GB/s from 4 to 32 nodes (1% variation), only dropping 1% at 64 nodes.

5. **The NVLink→IB cliff is 15x for P2P**: sendrecv drops from 656 GB/s intra-node to 43.3 GB/s at 2 nodes. P2P is single-HCA limited (43.3 / 50 = 87% utilization). This makes pipeline parallelism (PP) the inter-node bottleneck.

6. **Larger messages improve bandwidth at scale**: ring 64n gains 29% from 2G→8G (223→288 GB/s). This validates GBS tuning (more gradient accumulation) as a communication optimization strategy.

7. **Tree algorithm is consistently worst** for large messages: ~191 GB/s ceiling regardless of node count at 4–64 nodes. Never force `NCCL_ALGO=Tree` for LLM training.

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

Results directory: `/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/collective-scaling/20260427_132654`

Slurm jobs: 85344–85414

Submit script: `~/together-nccl-tests/benchmarks/B200/collective-scaling/submit_all.sh`

Manifest: `manifest.txt` in results directory
