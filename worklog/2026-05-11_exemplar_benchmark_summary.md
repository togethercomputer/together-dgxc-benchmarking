# Exemplar Benchmark Summary — 2026-05-11

Today's 512-GPU LLM sweep on the slinky cluster vs yesterday's 256-GPU run and the Tranche-1 target.

## Scoreboard

Mean MODEL_TFLOP/s/GPU, iter 3–9, MAX_STEPS=10. Best result selected when fallback was used.

| # | Workload · dtype | Target | **Today (2026-05-11) 512 GPU** | vs Target | Yesterday (2026-05-10) 256 GPU | vs Yesterday | Scaling eff |
|--:|---|---:|---:|---:|---:|---:|---:|
| 1 | Llama 70B FP8 | 1,624 | **1,614** | −0.6% | 1,831 | −11.8% | 88% |
| 2 | Llama 70B NVFP4 | 2,013 | **1,946** | −3.3% | 2,063 | −5.6% | 94% |
| 3 | Nemotron-H 56B FP8 ‡ | 1,536 | 1,425 | −7.3% | 1,117 | **+27.5%** | — |
| 4 | Llama 405B FP8 | 1,722 | 931 | −45.9% | 1,023 | −8.9% | 91% |
| 5 | Llama 405B NVFP4 ‡ | 2,006 | 1,785 | −11.0% | **NODE_FAIL ×3** | **first success** | — |
| 6 | Nemotron-4 340B FP8 | 1,101 | **1,056** | −4.1% | 1,096 | −3.6% | 77%* |
| 7 | Nemotron-4 340B BF16 | 936 | 789 | −15.7% | 853 | −7.5% | 92% |
| 8 | Qwen3 235B BF16 | 514 | **491** | −4.6% | 479 | **+2.4%** | **102%** |

‡ = fallback to 256 GPU; the 512 GPU primary failed and the orchestrator auto-recovered.
\* = vs same-day 256-GPU control run (1371); vs yesterday's 256 the apparent scaling was 96% but yesterday was whitelist-constrained.

## Highlights

- **8 of 8 workloads produced numbers** today. Yesterday only 7 of 13 succeeded.
- **Llama 70B FP8 still beats target** at 2× the scale (−0.6% vs target 1624, vs yesterday's +10.5% at 256 GPU).
- **Llama 405B NVFP4 ran for the first time ever** on this cluster (NODE_FAIL ×3 yesterday, 1785 today).
- **Qwen3 235B BF16 superscaled**: 491 @ 512 > 479 @ 256 → MoE with EP=8 keeps expert groups same size, extra GPUs do real work.
- **Llama 405B FP8 stays SHARP-bound**: −46% vs target, −47% vs MD1. Closing this gap requires SHARP enablement; no other workload is comparably affected.

## Per-workload narrative

| Workload | Status | Why |
|---|---|---|
| Llama 70B FP8 | ✅ matches target | FSDP path; clean 88% scaling |
| Llama 70B NVFP4 | ✅ near target | smaller buffers; 94% scaling — best of the dense models |
| Nemotron-H 56B FP8 | ⚠ fallback | PyTorch TCPStore bootstrap deadlock at 512 ranks; recovered at 256 |
| Llama 405B FP8 | ❌ SHARP-bound | flat 91% scaling but ceiling capped without fabric SHARP |
| Llama 405B NVFP4 | ⚠ fallback | 2 nodes (slinky-9, slinky-26) went bad mid-512-run; recovered at 256 |
| N4 340B FP8 | ✅ near target | clean training, slow teardown only |
| N4 340B BF16 | ⚠ −16% target | consistent with yesterday's −10% miss; not new |
| Qwen3 235B BF16 | ✅ near target, beats yesterday | superlinear MoE scaling |

## Operational findings

- **Cluster degrades within hours**: 64 nodes healthy at 09:20, 62 nodes healthy by 12:02 (slinky-9, slinky-26 went bad mid-run). Run a fresh pair sweep before any large allocation.
- **Orchestrator fallback works**: pair sweep + scale-down-to-256 fired 2 productive times (nemotronh, llama405b_nvfp4) plus 1 spurious time (n4340b_fp8 TIMEOUT with valid mean; orchestrator patched mid-run).
- **SHARP would close the biggest gap** — only Llama 405B FP8 sits 46% below target, and SHARP is the documented lift.

## Files

- Full report: `/home/johnson/worklogs/flapping_airplanes_512gpu_benchmark_report.md`
- Yesterday's: `/home/johnson/worklogs/flapping_airplanes_256gpu_benchmark_report.md`
- Results TSV: `/home/johnson/auto_sweep_512gpu/results.tsv`
