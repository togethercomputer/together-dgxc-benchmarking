# NCCL 64-Node Benchmark Report — 2026-04-16

NCCL collective performance at 64-node scale (512 GPUs) on the B200 DGXC cluster, testing three collectives across three algorithm configurations.

## System Configuration

| Component | Details |
|-----------|---------|
| GPU | NVIDIA B200, 8 per node |
| Nodes | 64 (use3a-ss-b200-gpu-154 through gpu-256) |
| Total GPUs | 512 (ranks 0–511) |
| Interconnect | Mellanox ConnectX-7 (MT4129), NDR 400 Gb/s InfiniBand |
| HCA Firmware | 28.47.1088 |
| NICs per node | 4 (3x 400 Gb/s + 1x 100 Gb/s) |

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
| Message sizes | 1 MB → 8 GB (2x step factor) |
| Iterations | 5 warmup + 20 measured |
| Data type | float32 |
| Reduction op | sum (all_reduce, reduce_scatter) / none (all_gather) |
| Validation | enabled (Out of bounds check) |
| GPUs per rank | 1 (`-g 1`) |

## Test Matrix

9 total configs — 3 collectives x 3 algorithm settings:

| Config Label | NCCL_ALGO | NCCL_NVLS_ENABLE | NCCL_COLLNET_ENABLE |
|--------------|-----------|------------------|---------------------|
| ring | RING | 0 | 0 |
| nvls | (auto) | 1 | 0 |
| collnet_sharp | (auto) | 0 | 1 |

## Results — Peak busBW (GB/s) at 8 GB Message Size

| Collective | Ring | NVLS | CollNet SHARP | Best |
|------------|------|------|---------------|------|
| all_reduce | 279.80 | 202.14 | **350.14** | CollNet |
| all_gather | 275.26 | 275.40 | **350.62** | CollNet |
| reduce_scatter | 286.76 | 281.90 | **353.25** | CollNet |

## Comparison vs Previous Run (2026-04-09)

Previous run: `20260409_195719_slurm_64nodes` — same NCCL version (2.29.7+cuda12.9), same cluster, same test config.

### Peak busBW @ 8 GB — Summary

| Test / Config | Apr-09 (GB/s) | Apr-16 (GB/s) | Delta (GB/s) | Delta (%) |
|---------------|---------------|---------------|--------------|-----------|
| all_reduce / ring | 285.05 | 279.80 | -5.25 | -1.8% |
| all_reduce / nvls | 219.51 | 202.14 | -17.37 | **-7.9%** |
| all_reduce / collnet_sharp | 363.94 | 350.14 | -13.80 | **-3.8%** |
| all_gather / ring | 282.85 | 275.26 | -7.59 | -2.7% |
| all_gather / nvls | 283.18 | 275.40 | -7.78 | -2.7% |
| all_gather / collnet_sharp | 363.79 | 350.62 | -13.17 | **-3.6%** |
| reduce_scatter / ring | 289.37 | 286.76 | -2.61 | -0.9% |
| reduce_scatter / nvls | 290.67 | 281.90 | -8.77 | -3.0% |
| reduce_scatter / collnet_sharp | 362.49 | 353.25 | -9.24 | -2.5% |

All 9 configs regressed. Average regression: -3.2%. Worst: all_reduce/nvls at -7.9%.

### all_reduce Detailed Comparison (busBW GB/s, out-of-place)

#### all_reduce / ring

| Size | Apr-09 | Apr-16 | Delta | Delta (%) |
|------|--------|--------|-------|-----------|
| 1 MB | 0.96 | 0.96 | +0.00 | +0.0% |
| 2 MB | 1.70 | 1.68 | -0.02 | -1.2% |
| 4 MB | 1.20 | 1.20 | +0.00 | +0.0% |
| 8 MB | 3.48 | 3.47 | -0.01 | -0.3% |
| 16 MB | 5.20 | 5.00 | -0.20 | -3.8% |
| 32 MB | 10.38 | 10.03 | -0.35 | -3.4% |
| 64 MB | 18.78 | 14.86 | -3.92 | **-20.9%** |
| 128 MB | 35.74 | 28.44 | -7.30 | **-20.4%** |
| 256 MB | 65.83 | 53.00 | -12.83 | **-19.5%** |
| 512 MB | 99.84 | 84.24 | -15.60 | **-15.6%** |
| 1 GB | 159.91 | 142.72 | -17.19 | **-10.7%** |
| 2 GB | 223.89 | 205.79 | -18.10 | **-8.1%** |
| 4 GB | 267.89 | 268.37 | +0.48 | +0.2% |
| 8 GB | 285.05 | 279.80 | -5.25 | -1.8% |

Significant regression at 64 MB–2 GB range (15–21% slower). Recovers at 4–8 GB.

#### all_reduce / nvls

| Size | Apr-09 | Apr-16 | Delta | Delta (%) |
|------|--------|--------|-------|-----------|
| 1 MB | 7.91 | 7.88 | -0.03 | -0.4% |
| 2 MB | 13.88 | 13.54 | -0.34 | -2.4% |
| 4 MB | 22.00 | 21.09 | -0.91 | -4.1% |
| 8 MB | 29.70 | 29.31 | -0.39 | -1.3% |
| 16 MB | 53.78 | 53.21 | -0.57 | -1.1% |
| 32 MB | 53.52 | 54.93 | +1.41 | +2.6% |
| 64 MB | 100.57 | 101.07 | +0.50 | +0.5% |
| 128 MB | 145.98 | 147.07 | +1.09 | +0.7% |
| 256 MB | 185.44 | 186.63 | +1.19 | +0.6% |
| 512 MB | 215.55 | 215.34 | -0.21 | -0.1% |
| 1 GB | 238.16 | 237.30 | -0.86 | -0.4% |
| 2 GB | 253.08 | 248.95 | -4.13 | -1.6% |
| 4 GB | 259.66 | 226.91 | -32.75 | **-12.6%** |
| 8 GB | 219.51 | 202.14 | -17.37 | **-7.9%** |

NVLS regression concentrated at 4–8 GB. Smaller sizes are stable or slightly improved.

#### all_reduce / collnet_sharp

| Size | Apr-09 | Apr-16 | Delta | Delta (%) |
|------|--------|--------|-------|-----------|
| 1 MB | 7.69 | 7.61 | -0.08 | -1.0% |
| 2 MB | 13.67 | 13.95 | +0.28 | +2.0% |
| 4 MB | 20.82 | 21.04 | +0.22 | +1.1% |
| 8 MB | 26.87 | 26.92 | +0.05 | +0.2% |
| 16 MB | 51.29 | 51.43 | +0.14 | +0.3% |
| 32 MB | 95.64 | 94.33 | -1.31 | -1.4% |
| 64 MB | 117.02 | 116.42 | -0.60 | -0.5% |
| 128 MB | 120.03 | 119.54 | -0.49 | -0.4% |
| 256 MB | 169.64 | 169.91 | +0.27 | +0.2% |
| 512 MB | 113.77 | 99.69 | -14.08 | **-12.4%** |
| 1 GB | 175.96 | 153.08 | -22.88 | **-13.0%** |
| 2 GB | 277.74 | 245.57 | -32.17 | **-11.6%** |
| 4 GB | 336.56 | 334.36 | -2.20 | -0.7% |
| 8 GB | 363.94 | 350.14 | -13.80 | -3.8% |

CollNet regression concentrated at 512 MB–2 GB (11–13% slower). Small sizes stable.

## Detailed Results — all_reduce

### all_reduce / ring (NCCL_ALGO=RING)

| Size | Time (us) | algBW (GB/s) | busBW (GB/s) |
|------|-----------|-------------|-------------|
| 1 MB | 2,169.8 | 0.48 | 0.96 |
| 2 MB | 2,497.8 | 0.84 | 1.68 |
| 4 MB | 7,005.7 | 0.60 | 1.20 |
| 8 MB | 4,830.9 | 1.74 | 3.47 |
| 16 MB | 6,691.9 | 2.51 | 5.00 |
| 32 MB | 6,679.7 | 5.02 | 10.03 |
| 64 MB | 9,012.5 | 7.45 | 14.86 |
| 128 MB | 9,421.4 | 14.25 | 28.44 |
| 256 MB | 10,109 | 26.55 | 53.00 |
| 512 MB | 12,721 | 42.20 | 84.24 |
| 1 GB | 15,017 | 71.50 | 142.72 |
| 2 GB | 20,830 | 103.10 | 205.79 |
| 4 GB | 31,945 | 134.45 | 268.37 |
| 8 GB | 61,281 | 140.17 | 279.80 |

### all_reduce / nvls (NCCL_NVLS_ENABLE=1)

| Size | Time (us) | algBW (GB/s) | busBW (GB/s) |
|------|-----------|-------------|-------------|
| 1 MB | 265.8 | 3.95 | 7.88 |
| 2 MB | 309.3 | 6.78 | 13.54 |
| 4 MB | 397.0 | 10.57 | 21.09 |
| 8 MB | 571.2 | 14.69 | 29.31 |
| 16 MB | 629.4 | 26.66 | 53.21 |
| 32 MB | 1,219.3 | 27.52 | 54.93 |
| 64 MB | 1,325.3 | 50.64 | 101.07 |
| 128 MB | 1,821.7 | 73.68 | 147.07 |
| 256 MB | 2,871.0 | 93.50 | 186.63 |
| 512 MB | 4,976.5 | 107.88 | 215.34 |
| 1 GB | 9,032.1 | 118.88 | 237.30 |
| 2 GB | 17,219 | 124.72 | 248.95 |
| 4 GB | 37,782 | 113.68 | 226.91 |
| 8 GB | 84,822 | 101.27 | 202.14 |

### all_reduce / collnet_sharp (NCCL_COLLNET_ENABLE=1)

| Size | Time (us) | algBW (GB/s) | busBW (GB/s) |
|------|-----------|-------------|-------------|
| 1 MB | 275.1 | 3.81 | 7.61 |
| 2 MB | 300.0 | 6.99 | 13.95 |
| 4 MB | 397.9 | 10.54 | 21.04 |
| 8 MB | 622.0 | 13.49 | 26.92 |
| 16 MB | 651.1 | 25.77 | 51.43 |
| 32 MB | 710.0 | 47.26 | 94.33 |
| 64 MB | 1,150.6 | 58.32 | 116.42 |
| 128 MB | 2,241.1 | 59.89 | 119.54 |
| 256 MB | 3,153.6 | 85.12 | 169.91 |
| 512 MB | 10,750 | 49.94 | 99.69 |
| 1 GB | 14,001 | 76.69 | 153.08 |
| 2 GB | 17,455 | 123.03 | 245.57 |
| 4 GB | 25,640 | 167.51 | 334.36 |
| 8 GB | 48,970 | 175.41 | 350.14 |

## Detailed Results — all_gather

### all_gather / ring (NCCL_ALGO=RING)

| Size | Time (us) | algBW (GB/s) | busBW (GB/s) |
|------|-----------|-------------|-------------|
| 1 MB | 1,107.6 | 0.95 | 0.94 |
| 2 MB | 1,262.1 | 1.66 | 1.66 |
| 4 MB | 3,426.1 | 1.22 | 1.22 |
| 8 MB | 2,379.9 | 3.52 | 3.52 |
| 16 MB | 3,243.6 | 5.17 | 5.16 |
| 32 MB | 3,258.3 | 10.30 | 10.28 |
| 64 MB | 4,503.3 | 14.90 | 14.87 |
| 128 MB | 4,696.3 | 28.58 | 28.52 |
| 256 MB | 5,030.5 | 53.36 | 53.26 |
| 512 MB | 6,325.7 | 84.87 | 84.71 |
| 1 GB | 7,567.6 | 141.89 | 141.61 |
| 2 GB | 10,301 | 208.47 | 208.07 |
| 4 GB | 19,979 | 214.97 | 214.55 |
| 8 GB | 31,145 | 275.80 | 275.26 |

### all_gather / nvls (NCCL_NVLS_ENABLE=1)

| Size | Time (us) | algBW (GB/s) | busBW (GB/s) |
|------|-----------|-------------|-------------|
| 1 MB | 1,102.0 | 0.95 | 0.95 |
| 2 MB | 1,258.6 | 1.67 | 1.66 |
| 4 MB | 3,508.3 | 1.20 | 1.19 |
| 8 MB | 2,390.9 | 3.51 | 3.50 |
| 16 MB | 3,269.8 | 5.13 | 5.12 |
| 32 MB | 3,254.8 | 10.31 | 10.29 |
| 64 MB | 4,498.6 | 14.92 | 14.89 |
| 128 MB | 4,687.5 | 28.63 | 28.58 |
| 256 MB | 5,023.6 | 53.44 | 53.33 |
| 512 MB | 6,305.9 | 85.14 | 84.97 |
| 1 GB | 7,583.8 | 141.58 | 141.31 |
| 2 GB | 10,391 | 206.67 | 206.26 |
| 4 GB | 19,905 | 215.77 | 215.35 |
| 8 GB | 31,130 | 275.94 | 275.40 |

### all_gather / collnet_sharp (NCCL_COLLNET_ENABLE=1)

| Size | Time (us) | algBW (GB/s) | busBW (GB/s) |
|------|-----------|-------------|-------------|
| 1 MB | 1,114.6 | 0.94 | 0.94 |
| 2 MB | 1,257.9 | 1.67 | 1.66 |
| 4 MB | 3,497.3 | 1.20 | 1.20 |
| 8 MB | 2,412.2 | 3.48 | 3.47 |
| 16 MB | 4,768.0 | 3.52 | 3.51 |
| 32 MB | 4,773.5 | 7.03 | 7.02 |
| 64 MB | 4,779.2 | 14.04 | 14.01 |
| 128 MB | 4,800.2 | 27.96 | 27.91 |
| 256 MB | 4,942.2 | 54.31 | 54.21 |
| 512 MB | 5,371.7 | 99.94 | 99.75 |
| 1 GB | 6,966.8 | 154.12 | 153.82 |
| 2 GB | 8,667.3 | 247.77 | 247.29 |
| 4 GB | 18,414 | 233.25 | 232.79 |
| 8 GB | 24,451 | 351.31 | 350.62 |

## Detailed Results — reduce_scatter

### reduce_scatter / ring (NCCL_ALGO=RING)

| Size | Time (us) | algBW (GB/s) | busBW (GB/s) |
|------|-----------|-------------|-------------|
| 1 MB | 1,123.4 | 0.93 | 0.93 |
| 2 MB | 1,252.0 | 1.67 | 1.67 |
| 4 MB | 3,406.1 | 1.23 | 1.23 |
| 8 MB | 2,353.9 | 3.56 | 3.56 |
| 16 MB | 3,134.6 | 5.35 | 5.34 |
| 32 MB | 3,271.9 | 10.26 | 10.24 |
| 64 MB | 4,507.6 | 14.89 | 14.86 |
| 128 MB | 4,691.6 | 28.61 | 28.55 |
| 256 MB | 5,067.8 | 52.97 | 52.87 |
| 512 MB | 6,293.3 | 85.31 | 85.14 |
| 1 GB | 7,643.8 | 140.47 | 140.20 |
| 2 GB | 10,226 | 210.01 | 209.59 |
| 4 GB | 20,086 | 213.82 | 213.41 |
| 8 GB | 29,897 | 287.32 | 286.76 |

### reduce_scatter / nvls (NCCL_NVLS_ENABLE=1)

| Size | Time (us) | algBW (GB/s) | busBW (GB/s) |
|------|-----------|-------------|-------------|
| 1 MB | 1,131.2 | 0.93 | 0.93 |
| 2 MB | 1,259.5 | 1.67 | 1.66 |
| 4 MB | 3,475.1 | 1.21 | 1.20 |
| 8 MB | 2,367.2 | 3.54 | 3.54 |
| 16 MB | 3,127.8 | 5.36 | 5.35 |
| 32 MB | 3,275.2 | 10.24 | 10.22 |
| 64 MB | 4,510.6 | 14.88 | 14.85 |
| 128 MB | 4,699.1 | 28.56 | 28.51 |
| 256 MB | 5,041.6 | 53.24 | 53.14 |
| 512 MB | 6,278.3 | 85.51 | 85.34 |
| 1 GB | 7,599.0 | 141.30 | 141.02 |
| 2 GB | 10,256 | 209.39 | 208.98 |
| 4 GB | 20,143 | 213.23 | 212.81 |
| 8 GB | 30,412 | 282.46 | 281.90 |

### reduce_scatter / collnet_sharp (NCCL_COLLNET_ENABLE=1)

| Size | Time (us) | algBW (GB/s) | busBW (GB/s) |
|------|-----------|-------------|-------------|
| 1 MB | 1,125.9 | 0.93 | 0.93 |
| 2 MB | 1,246.2 | 1.68 | 1.68 |
| 4 MB | 3,537.2 | 1.19 | 1.18 |
| 8 MB | 2,406.8 | 3.49 | 3.48 |
| 16 MB | 4,705.3 | 3.57 | 3.56 |
| 32 MB | 4,826.0 | 6.95 | 6.94 |
| 64 MB | 4,785.6 | 14.02 | 14.00 |
| 128 MB | 4,809.1 | 27.91 | 27.85 |
| 256 MB | 4,940.6 | 54.33 | 54.23 |
| 512 MB | 5,360.3 | 100.16 | 99.96 |
| 1 GB | 6,937.0 | 154.78 | 154.48 |
| 2 GB | 8,689.1 | 247.15 | 246.66 |
| 4 GB | 18,224 | 235.67 | 235.21 |
| 8 GB | 24,269 | 353.94 | 353.25 |

## Observations

1. **CollNet SHARP is the clear winner** across all three collectives at 64-node scale, achieving ~350 GB/s peak busBW — 25-73% ahead of ring/NVLS.

2. **SHARP not fully operational**: All collnet_sharp jobs logged `SHARP coll init error: Cannot create SHARP job(-11)` on every node. SHARP fell back to non-SHARP CollNet transport, which still outperformed ring and NVLS. Actual SHARP offload (requires `sharpd` daemon running on switches) would likely push numbers higher.

3. **all_reduce NVLS regression at large sizes**: NVLS busBW peaked at ~249 GB/s at 2 GB, then dropped to 202 GB/s at 8 GB. This suggests memory pressure or algorithm fallback at the largest message sizes. At smaller sizes (16–256 MB), NVLS significantly outperforms ring for all_reduce due to lower latency.

4. **Ring and NVLS nearly identical for all_gather/reduce_scatter**: Both converge to 275–287 GB/s at 8 GB, suggesting the bottleneck is inter-node IB bandwidth rather than intra-node algorithm choice at this scale.

5. **CollNet latency overhead at small sizes**: For messages under 64 MB, CollNet configs show higher latency than ring/NVLS (e.g., all_gather at 16 MB: CollNet 4,768 us vs ring 3,244 us). CollNet only breaks ahead at 512 MB+.

6. **Uniform regression vs Apr-09 baseline**: All 9 configs regressed (avg -3.2%, worst -7.9%). Same NCCL version, same test parameters. The regression is not algorithm-specific — it affects ring, NVLS, and CollNet alike, pointing to a system-level cause rather than software change.

7. **Ring regression concentrated at mid-range sizes (64 MB–2 GB)**: all_reduce/ring dropped 15–21% at 64 MB–2 GB but recovered at 4–8 GB. This pattern suggests possible IB congestion, routing changes, or degraded links on some nodes in the current allocation.

8. **CollNet mid-range regression (512 MB–2 GB)**: CollNet showed 11–13% drops at 512 MB–2 GB while small sizes remained stable. This is consistent with inter-node bandwidth degradation affecting the CollNet tree at intermediate sizes.

9. **Possible causes for regression**: The cluster underwent a reboot between Apr-09 and Apr-16. Different node allocation (gpu-154–256 vs prior set), potential IB fabric changes, or degraded links on specific nodes could explain the uniform drop. Worth investigating with per-node IB health checks.

## Raw Results

Results directory: `/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/20260416_092154_slurm_64nodes/`

Slurm jobs: 83894–83902
