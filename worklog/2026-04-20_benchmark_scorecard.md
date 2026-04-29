# B200 DGXC Benchmark Scorecard — 2026-04-20

Together AI B200 cluster. All values in TFLOP/s/GPU, iter-5 steady-state.

**Scale note:** Nemotron4 15B BF16/FP8 ran at **256 GPUs** (metadata.yaml caps b200 scales at 256 with `exact_scales: true` — 512 is unsupported). All other models ran at **512 GPUs**.

| #  | Model          | Dtype | 04-20 Benchmark (TFLOP/s/GPU) | Target (TFLOP/s/GPU) | Gap    | Job ID         |
|---:|----------------|-------|------------------------------:|---------------------:|-------:|----------------|
|  1 | Nemotron4 15B  | BF16  | **1,489** (256 GPU)           | 1,264                | +17.8% | 84488          |
|  2 | Nemotron4 15B  | FP8   | **1,665** (256 GPU)           | 1,908                | −12.7% | 84490          |
|  3 | Llama 3.1 70B  | FP8   | **~1,540**                    | 1,624                |  −5.2% | 84459          |
|  4 | Llama 3.1 70B  | NVFP4 | **~2,050**                    | 2,013                |  +1.8% | 84374          |
|  5 | Nemotron-H 56B | FP8   | **~1,500**                    | 1,536                |  −2.3% | 84464          |
|  6 | Qwen3 235B A22B| BF16  | **604.7**                     | 514                  | +17.6% | 84466          |
|  7 | Qwen3 235B A22B| FP8   | **499.2**                     | 426                  | +17.2% | 84469          |
|  8 | Grok1 314B     | BF16  | **1,008**                     | 1,025                |  −1.7% | 84475          |
|  9 | Grok1 314B     | FP8   | **1,456**                     | 1,371                |  +6.2% | 84477          |
| 10 | Nemotron4 340B | BF16  | **855**                       | 936                  |  −8.7% | 84473          |
| 11 | Nemotron4 340B | FP8   | **1,236**                     | 1,101                | +12.3% | 84471          |
| 12 | Llama 3.1 405B | FP8   | **~1,790**                    | 1,722                |  +3.9% | 84463          |
| 13 | Llama 3.1 405B | NVFP4 | **~1,800**                    | 2,006                | −10.3% | 84460          |
| 14 | DeepSeek V3    | BF16  | **554.9**                     | 500                  | +11.0% | 84479          |
| 15 | DeepSeek V3    | FP8   | **~597**                      | 406                  | +47.0% | 84467          |
|    | **Cluster aggregate** | —  | **ΣB = 18,645**           | **ΣT = 18,352**      | **+1.60%** | —          |

### Cluster-level summary

| Metric                              | Value              |
|-------------------------------------|-------------------:|
| Aggregate gap (ΣBench ÷ ΣTarget − 1)| **+1.60%**         |
| Simple mean of per-model gaps       | +6.27%             |
| Mean absolute deviation from target | 11.72%             |
| Models meeting/exceeding target     | **9 / 15** (60%)   |
| Models within ±5% of target         | 4 / 15             |
| Models within ±10% of target        | 7 / 15             |

**Interpretation:** The cluster as a whole delivers **+1.6% above aggregate target** — net slightly above the 256-GPU reference even though 14/15 configs are running at 512 GPUs (an unsupported or untuned scale for most models). Dispersion is wide: 6 models underperform (15B FP8 worst at −12.7%; 405B NVFP4 −10.3%), while 5 exceed target by >10% (DSV3 FP8 +47%, 15B BF16 +17.8%, Qwen3 BF16 +17.6%, Qwen3 FP8 +17.2%, N4-340B FP8 +12.3%, DSV3 BF16 +11.0%).

## Notes

- **Gap** = (04-20 Benchmark − Target) / Target. Positive = exceeds target; negative = below target.
- **Target** = "best achieved at 256 GPUs" baseline from `~/johnson/worklog/2026-04-14_b200_benchmark_summary.md` (NVIDIA reference / prior-optimized config), except Nemotron4 15B which uses the Tranche-1 targets (BF16=1,264, FP8=1,908).
- **Nemotron4 15B FP8 @ 256 GPU gap −12.7%** is structural: the 1,908 target was achieved with `GBS=2048` (4 μbatch/rank), but today's run used the launcher-default `GBS=1024` (2 μbatch/rank). Rerun with `GBS=2048` to reproduce the target.
- **Nemotron4 15B at 512 GPUs (if previously run):** BF16=1,376 (84447), FP8=1,528/1,531 (84455/84486). Listed here for context but excluded from the scorecard because 512 is not an officially supported scale for this model.
- **Llama 3.1 70B NVFP4 (+1.8%)**: Today's 512-GPU run is 16% slower than Friday 2026-04-17 (confirmed reproducible: 84374 and 84480 both at 7.2s/iter vs Friday 6.19s/iter). Target comparison still positive, but Friday→Today regression is an **open investigation** — details in `~/johnson/worklog/2026-04-20_512gpu_rerun_report.md`.
- **Llama 3.1 405B NVFP4 (−10.3%)**: Target 2,006 was the PP=8 optimized configuration at 256 GPUs; today's 512-GPU result used the default launcher preset. Gap is expected if the 512-GPU config isn't using the same PP=8 tuning.

## Reproduction

All jobs submitted via:
- **Nemotron4 15B (256 GPU today)**: `SCALE=256 bash /home/johnson/johnson/worklog/submit_session_B.sh n15b_bf16|n15b_fp8`
- **512-GPU jobs**: `llmb-run submit …` for modern (Megatron-Bridge) workloads, `bash /tmp/submit_legacy.sh …` for legacy 25.07/25.09 workloads.

See full details: `~/johnson/worklog/2026-04-20_all15_models_512gpu_report.md` and `~/johnson/worklog/session_B_report_2026-04-20.md`.
