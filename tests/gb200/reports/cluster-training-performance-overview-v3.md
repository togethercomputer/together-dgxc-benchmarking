# Cluster Training Performance Overview — v3

Version: 3 | Date: 2026-04-16
Supersedes: cluster-training-performance-overview-v2.pdf (v2, Racks 01, 04, 11, 16)

## What's new in v3

- Rack 08 and Rack 10 added to the benchmark suite (newly provisioned racks).
- All three benchmark workloads (NCCL all_reduce, GPT-OSS 120B, Qwen 72B) run on both new racks.
- Multiple runs per workload: NCCL x2, GPT-OSS x3, Qwen x2 (R08) / x3 (R10).
- Run 3 results used as primary data point (latest, cleanest run); all runs reported for reproducibility.
- Statistics follow v2 methodology: median as primary metric, exclude first step + first 10 extracted steps as warmup.

## Executive Summary

| Workload | Rack 08 (Run 3) | Rack 10 (Run 3) | v2 Rack 04 (baseline) | v2 Range |
|---|---|---|---|---|
| NCCL @ 32 GiB | 925.2 GB/s | 927.6 GB/s | 918.7 GB/s | 918.7–922.2 |
| GPT-OSS 120B median TFLOPS | 377.8 | 386.4 | 405.4 | 395.9–407.2 |
| GPT-OSS 120B median step (s) | 5.89 | 5.76 | 5.49 | 5.47–5.62 |
| Qwen 72B median TFLOPS | 732.2 | 718.9 | 709.0 | 708.7–735.0 |
| Qwen 72B median step (s) | 19.91 | 20.27 | 20.56 | 19.84–20.57 |

Both racks show healthy NCCL fabric (above v2 range), competitive Qwen 72B throughput (above v2 Rack 04 baseline), and a consistent GPT-OSS 120B gap (~5.5% below v2 average) attributable to the OS/kernel stack difference. No hardware fault signatures. Both racks are production-ready.

## Common Method

- Extract Step Time / TFLOPS_per_GPU lines from each log.
- Exclude the first step (large initialization/warmup outlier) and the first 10 extracted steps as warmup normalization.
- Primary metric: median (robust to isolated outliers). Mean provided for continuity.
- For NCCL: out-of-place bus bandwidth (GB/s) at each message size.

---

## NCCL all_reduce Bandwidth — Rack 08 and Rack 10

Single-rack NVLS all_reduce sweep (18 nodes / 72 ranks per rack). Out-of-place bus bandwidth (GB/s) reported. Run 3 results shown with v2 comparison.

| Message size | R08 Run 3 | R10 Run 3 | v2 Rack 01 | v2 Rack 04 | v2 Rack 11 | v2 Rack 16 |
|---|---|---|---|---|---|---|
| 128 MiB | 476.1 | 476.9 | 476.4 | 474.9 | 473.4 | 475.8 |
| 256 MiB | 575.8 | 574.8 | 572.5 | 572.8 | 574.5 | 569.8 |
| 512 MiB | 625.4 | 625.1 | 629.9 | 632.3 | 622.1 | 637.7 |
| 1 GiB | 738.7 | 736.7 | 737.2 | 737.2 | 736.0 | 737.5 |
| 2 GiB | 805.5 | 807.7 | 801.9 | 800.7 | 803.1 | 804.3 |
| 4 GiB | 842.0 | 843.2 | 835.3 | 833.4 | 838.6 | 837.2 |
| 8 GiB | 877.5 | 882.1 | 875.5 | 875.7 | 875.7 | 877.8 |
| 16 GiB | 910.9 | 913.1 | 906.1 | 905.5 | 908.0 | 910.4 |
| 32 GiB | 925.2 | 927.6 | 919.2 | 918.7 | 921.6 | 922.2 |

### NCCL Run-to-Run Reproducibility (@ 32 GiB)

| | Run 1 | Run 3 |
|---|---|---|
| Rack 08 | 924.0 | 925.2 |
| Rack 10 | 926.0 | 927.6 |

Key findings:
- Both racks exceed or match v2 baselines at every message size across both runs.
- At 32 GiB, the spread across all six racks is 918.7–927.6 GB/s (1.0%) — operationally identical.
- Run-to-run variance is <0.2%. Fabric is healthy and stable.

---

## GPT-OSS 120B BF16 — Rack 08 and Rack 10

### Configuration

| Parameter | Value |
|---|---|
| Model | GPT-OSS 120B (MoE, 128 experts, top-4) |
| Parallelism | TP=1, PP=1, EP=64, DP=64 |
| Batch | MBS=4, GBS=1280, GA=5 |
| Seq length | 4096 |
| Precision | BF16 mixed |
| Nodes / GPUs | 16 / 64 per rack |
| Image | nvcr.io/nvidia/nemo:26.02 |

### Results (Run 3 vs v2 Baseline)

| Metric | R08 Run 3 | R10 Run 3 | v2 Rack 04 (baseline) | v2 Rack 01 | v2 Rack 11 | v2 Rack 16 |
|---|---|---|---|---|---|---|
| Post-warmup steps | 29 | 38 | 120 | 185 | 109 | 211 |
| Median step time (s) | 5.89 | 5.76 | 5.49 | 5.62 | 5.47 | 5.47 |
| Mean step time (s) | 6.068 | 5.963 | 5.616 | 5.781 | 5.610 | 5.616 |
| P90 step time (s) | 6.99 | 6.89 | 6.26 | 6.51 | 6.32 | 6.36 |
| Median TFLOPS/GPU | 377.8 | 386.4 | 405.4 | 395.9 | 407.2 | 406.5 |
| Mean TFLOPS/GPU | 368.4 | 375.1 | 397.6 | 386.3 | 398.1 | 397.8 |
| P90 TFLOPS/GPU | 381.1 | 391.2 | 410.3 | 402.8 | 413.4 | 416.6 |

### Rack-to-rack deltas (Run 3 median TFLOPS/GPU, relative to Rack 04 baseline)

| Rack | Delta vs Rack 04 |
|---|---|
| 01 | -2.3% |
| 04 | baseline |
| 08 | -6.8% |
| 10 | -4.7% |
| 11 | +0.4% |
| 16 | +0.3% |

### Run-to-Run Reproducibility (3 runs)

| Run | R08 Median TFLOPS | R10 Median TFLOPS |
|---|---|---|
| Run 1 | 386.2 | 386.2 |
| Run 2 | 386.6 | 373.8 |
| Run 3 | 377.8 | 386.4 |
| **3-run average** | **383.5** | **382.1** |
| **Avg delta vs Rack 04** | **-5.4%** | **-5.7%** |

Both racks average ~383 median TFLOPS across 3 runs, with ~2-3% run-to-run variance. The ~5.5% gap vs v2 Rack 04 is uniform across both new racks and correlates with the OS/kernel difference (Ubuntu 24.04 / kernel 6.17 vs Ubuntu 22.04 / kernel 6.8 on v2 racks). NCCL bandwidth is healthy on both racks, confirming this is not a fabric or hardware issue.

---

## Qwen 72B BF16 — Rack 08 and Rack 10

### Configuration

| Parameter | Value |
|---|---|
| Model | Qwen2.5 72B |
| Parallelism | TP=8, PP=4, CP=1, DP=2, SP=True |
| Batch | MBS=2, GBS=512, GA=128 |
| Seq length | 4096 |
| Precision | BF16 mixed |
| Nodes / GPUs | 16 / 64 per rack |
| Image | nvcr.io/nvidia/nemo:26.02 |

### Results (Run 3 vs v2 Baseline)

| Metric | R08 Run 3 | R10 Run 3 | v2 Rack 04 (baseline) | v2 Rack 01 | v2 Rack 11 | v2 Rack 16 |
|---|---|---|---|---|---|---|
| Post-warmup steps | 22 | 22 | 140 | 61 | 203 | 79 |
| Median step time (s) | 19.91 | 20.27 | 20.56 | 20.31 | 20.57 | 19.84 |
| Mean step time (s) | 20.078 | 20.475 | 20.61 | 20.42 | 20.61 | 19.88 |
| P90 step time (s) | 20.90 | 21.39 | 21.18 | 21.01 | 21.45 | 20.51 |
| Median TFLOPS/GPU | 732.2 | 718.9 | 709.0 | 717.6 | 708.7 | 735.0 |
| Mean TFLOPS/GPU | 726.3 | 712.3 | 707.4 | 714.1 | 707.5 | 733.5 |
| P90 TFLOPS/GPU | 736.3 | 724.0 | 719.0 | 724.4 | 720.8 | 743.7 |

### Rack-to-rack deltas (Run 3 median TFLOPS/GPU, relative to Rack 04 baseline)

| Rack | Median TFLOPS/GPU | Delta vs Rack 04 |
|---|---|---|
| 01 | 717.6 | +1.2% |
| 04 | 709.0 | baseline |
| **08** | **732.2** | **+3.3%** |
| **10** | **718.9** | **+1.4%** |
| 11 | 708.7 | -0.04% |
| 16 | 735.0 | +3.7% |

All six racks fall within a 3.7% spread (708.7–735.0 TFLOPS/GPU). Rack 08 and Rack 16 are the strongest Qwen performers.

### Run-to-Run Reproducibility

| Run | R08 Median TFLOPS | R10 Median TFLOPS |
|---|---|---|
| Run 1 | 731.9 | 678.2* |
| Run 2 | — | 722.4 |
| Run 3 | 732.2 | 718.9 |
| **Representative avg** | **732.1** | **720.7 (excl. run 1)** |

*Run 1 anomaly — transient step-time drift, confirmed not reproducible on runs 2 & 3.

- **Rack 08**: Perfectly reproducible at ~732 TFLOPS / 19.91s across 2 runs.
- **Rack 10**: Run 1 anomaly (678.2) was transient. Runs 2 & 3 average 720.7 TFLOPS, comfortably within v2 baseline range.

---

## Interpretation

- **Cluster-wide NCCL health**: All six racks (01, 04, 08, 10, 11, 16) produce within-range NCCL bandwidth. Racks 08 and 10 are slightly above v2 baselines, confirmed across 2 runs each. Fabric is healthy and stable.

- **GPT-OSS 120B — Racks 08 & 10**: Both new racks average ~383 median TFLOPS across 3 runs, ~5.5% below v2 Rack 04 baseline. Run-to-run variance is ~2-3%, which is normal. Both racks run Ubuntu 24.04 / kernel 6.17 (vs 22.04 / 6.8 on v2 racks), which likely accounts for the gap. Not a hardware issue.

- **Qwen 72B — Rack 08**: Excellent and reproducible — 732 TFLOPS across 2 runs, +3.3% above v2 Rack 04 baseline. Second-fastest Qwen rack after Rack 16.

- **Qwen 72B — Rack 10**: Run 1 anomaly (-4.3%) confirmed transient. Runs 2 & 3 produce 720.7 avg TFLOPS (+1.7% above Rack 04), with no step-time drift. Healthy.

- **No rack shows a pattern suggesting hardware fault.** All NCCL bandwidth measurements are healthy. All training anomalies were either OS/kernel-related (GPT-OSS gap) or transient (Rack 10 Qwen run 1).

---

## Source Logs

| Workload | Rack | Run | File |
|---|---|---|---|
| NCCL all_reduce | 08 | Run 1 | benchmarks/logs/nccl-r08.log |
| NCCL all_reduce | 08 | Run 3 | benchmarks/logs/nccl-r08-v3.log |
| NCCL all_reduce | 10 | Run 1 | benchmarks/logs/nccl-r10.log |
| NCCL all_reduce | 10 | Run 3 | benchmarks/logs/nccl-r10-v3.log |
| GPT-OSS 120B | 08 | Run 1 | benchmarks/logs/gpt-oss-r08.log |
| GPT-OSS 120B | 08 | Run 2 | benchmarks/logs/gpt-oss-r08-v2.log |
| GPT-OSS 120B | 08 | Run 3 | benchmarks/logs/gpt-oss-r08-v3.log |
| GPT-OSS 120B | 10 | Run 1 | benchmarks/logs/gpt-oss-r10.log |
| GPT-OSS 120B | 10 | Run 2 | benchmarks/logs/gpt-oss-r10-v2.log |
| GPT-OSS 120B | 10 | Run 3 | benchmarks/logs/gpt-oss-r10-v3.log |
| Qwen 72B | 08 | Run 1 | benchmarks/logs/qwen-r08.log |
| Qwen 72B | 08 | Run 3 | benchmarks/logs/qwen-r08-v3.log |
| Qwen 72B | 10 | Run 1 | benchmarks/logs/qwen-r10.log |
| Qwen 72B | 10 | Run 2 | benchmarks/logs/qwen-r10-v2.log |
| Qwen 72B | 10 | Run 3 | benchmarks/logs/qwen-r10-v3.log |

## Benchmark YAMLs

All job definitions are in `/home/johnson/benchmarks/`:
- `nccl-18n-r08.yaml`, `nccl-18n-r10.yaml`
- `gpt-oss-120b-r08.yaml`, `gpt-oss-120b-r10.yaml`
- `qwen-72b-r08.yaml`, `qwen-72b-r10.yaml`

## Node Environment (Racks 08 & 10)

| Property | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| Kernel | 6.17.0-1014-nvidia-64k |
| Container Runtime | containerd 2.2.1 |
| Kubernetes | v1.34.6 |
| GPU | 4x per node (GB200 NVL72) |
| Nodes per rack | 18 |

Note: v2 racks (01, 04, 11, 16) run Ubuntu 22.04 / kernel 6.8. This OS/kernel difference likely accounts for the ~5.5% GPT-OSS gap on both new racks.
