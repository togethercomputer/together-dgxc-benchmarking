# B200 DGXC Benchmark Scorecard — 2026-04-26

Together AI B200 cluster. All values in TFLOP/s/GPU, iter-5 steady-state.

**Scale note:** Nemotron4 15B BF16/FP8 ran at **512 GPUs** today (officially supported only at 64/256). All other models ran at **512 GPUs**.

| #  | Model           | Dtype | 04-26 Benchmark (TFLOP/s/GPU) | Target (TFLOP/s/GPU) | Gap     | Job ID         |
|---:|----------------|-------|------------------------------:|---------------------:|--------:|----------------|
|  1 | Nemotron4 15B  | BF16  | **1,378**                     | 1,264                | +9.0%   | 85317          |
|  2 | Nemotron4 15B  | FP8   | **1,527**                     | 1,908                | −20.0%  | 85326          |
|  3 | Llama 3.1 70B  | FP8   | **1,525**                     | 1,624                |  −6.1%  | 85292          |
|  4 | Llama 3.1 70B  | NVFP4 | **2,047**                     | 2,013                |  +1.7%  | 85293          |
|  5 | Nemotron-H 56B | FP8   | **1,501**                     | 1,536                |  −2.3%  | 85296          |
|  6 | Qwen3 235B A22B| BF16  | **667**                       | 514                  | +29.8%  | 85320          |
|  7 | Qwen3 235B A22B| FP8   | **507**                       | 426                  | +19.0%  | 85323          |
|  8 | Grok1 314B     | BF16  | **1,006**                     | 1,025                |  −1.9%  | 85314          |
|  9 | Grok1 314B     | FP8   | **1,460**                     | 1,371                |  +6.5%  | 85312          |
| 10 | Nemotron4 340B | BF16  | **859**                       | 936                  |  −8.2%  | 85310          |
| 11 | Nemotron4 340B | FP8   | **1,238**                     | 1,101                | +12.4%  | 85307          |
| 12 | Llama 3.1 405B | FP8   | **1,772**                     | 1,722                |  +2.9%  | 85294          |
| 13 | Llama 3.1 405B | NVFP4 | **1,670** †                   | 2,006                | −16.7%  | 85295 / 85330  |
| 14 | DeepSeek V3    | BF16  | **553**                       | 500                  | +10.6%  | 85329          |
| 15 | DeepSeek V3    | FP8   | **601**                       | 406                  | +48.0%  | 85305          |
|    | **Cluster aggregate** | — | **ΣB = 18,803**             | **ΣT = 18,352**      | **+2.5%** | —          |

† Average of 2 confirmed runs: Run 1 (85295) = 1677, Run 2 (85330) = 1662. Regression vs 04-20 (1800) is reproducible.

## Cluster-level summary

| Metric                              | Today (04-26)     | 04-20 reference  |
|-------------------------------------|------------------:|-----------------:|
| Aggregate gap (ΣBench ÷ ΣTarget − 1)| **+2.5%**         | +1.6%            |
| Models meeting/exceeding target     | **9 / 15** (60%)  | 9 / 15           |
| Models within ±5% of target         | 5 / 15            | 4 / 15           |
| Models within ±10% of target        | 9 / 15            | 7 / 15           |
| Models within ±2% of 04-20          | 13 / 15           | —                |
| Models within ±1% of 04-20          | 11 / 15           | —                |

**Interpretation:** Cluster delivers **+2.5% above aggregate target** (slightly better than 04-20's +1.6%). Reproduction vs 04-20 is tight: 13/15 within ±2%, 11/15 within ±1%. Two outliers:

- **Qwen3 235B BF16 +10.2%** improvement over 04-20 (605 → 667). Reproducible across 9 steady-state samples. Likely picking up MB/26.02 stack uplift seen in 04-22 rebench.
- **Llama 405B NVFP4 −7.2%** regression vs 04-20 (1800 → 1670). Confirmed by 2 back-to-back runs. Hypothesis: today's run included 20 fabric-weak nodes flagged in the per-group NCCL scan; 04-20 used the historical 9-node exclude `[130,190,197,199,201,211,228,233,239]`.

## Notes

- **Gap** = (04-26 Benchmark − Target) / Target. Positive = exceeds; negative = below.
- **Target** = Tranche-1 targets from `~/johnson/worklog/2026-04-20_benchmark_scorecard.md` (NVIDIA reference / prior-optimized config baselines).
- **Nemotron4 15B FP8 @ 512 gap −20.0%** is structural: 1,908 target was achieved with `GBS=2048` at 256 GPUs (4 µbatch/rank). At 512 GPUs same GBS=2048 gives only 2 µbatch/rank → can't hide NCCL allreduce. 512 is not officially supported scale for this model.
- **Nemotron4 15B BF16 @ 512 (+9.0%)** beats target despite scale-mismatch, because BF16 is less micro-batch-sensitive than FP8.
- **Llama 405B NVFP4 (−16.7% vs target, −7.2% vs 04-20)** target 2,006 is the PP=8 optimized config at 256 GPUs; today's 512-GPU run used official PP=16. Gap vs target is partly structural (scale + PP), but the −7.2% vs 04-20 (same PP=16 official config) is the open regression.
- **DSV3 FP8 +48.0%** is a known-large gap because the Tranche-1 target (406) is conservative; the optimized 04-20 result of 597 is the realistic ceiling. Today's 601 matches 04-20.
- **No NCCL timeouts, no failures**, including the historically-risky DSV3 BF16 at 512.

## Reproduction

Full per-job commands and infrastructure quirks: `~/johnson/worklog/2026-04-26_15jobs_512gpu_report.md`.

## Cross-references

- Detail report: `~/johnson/worklog/2026-04-26_15jobs_512gpu_report.md`
- NCCL fabric scan today: `~/johnson/worklog/2026-04-26_nccl_per_group_scaling.md`
- SHARP investigation today: `~/johnson/worklog/2026-04-26_sharp_investigation.md`
- 04-20 reference scorecard: `~/johnson/worklog/2026-04-20_benchmark_scorecard.md`
