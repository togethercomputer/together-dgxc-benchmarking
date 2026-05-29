# DGXC Benchmarking Worklog

Engineering notes, analysis reports, and profiler findings from B200 cluster optimization work.

**Cluster:** Together AI B200 DGXC (74 nodes, 592 GPUs)
**Period:** 2026-04-08 to present

## Reports (chronological)

| Date | File | Summary |
|------|------|---------|
| Apr 8 | `2026-04-08_llama70b_fp8_scaling.md` | 70B FP8 64-592 GPU scaling study |
| Apr 8 | `2026-04-08_llama70b_fp8_64gpu.md` | 70B FP8 64-GPU detailed report |
| Apr 8 | `2026-04-08_llama70b_fp8_512gpu.md` | 70B FP8 512-GPU detailed report |
| Apr 9 | `2026-04-09_nccl_benchmark.md` | Initial NCCL collective benchmarks |
| Apr 10 | `2026-04-10_405b_nvfp4_256gpu_results.md` | 405B NVFP4 parallelism optimization results |
| Apr 10 | `2026-04-10_405b_nvfp4_512gpu.md` | 405B NVFP4 512-GPU scaling report |
| Apr 10 | `2026-04-10_405b_nvfp4_slack_update.md` | 405B slack update for team |
| Apr 10 | `2026-04-10_nccl_p2p_scaling.md` | NCCL SendRecv P2P bandwidth scaling analysis |
| Apr 10 | `2026-04-10_network_fabric_analysis.md` | IB fabric oversubscription analysis |
| Apr 10 | `2026-04-10_next_steps_plan.md` | Optimization roadmap after profiling |
| Apr 11 | `2026-04-11_405b_megatron_bridge_upgrade.md` | Megatron-Bridge upgrade: +7% on 405B |
| Apr 11 | `2026-04-11_dgxc_benchmarking_update.md` | Daily update: all models within 3-8% of targets |
| Apr 11 | `2026-04-11_sharp_not_available.md` | IB SHARP feasibility report (not available) |
| Apr 13 | `2026-04-13_nemotron4_15b_fp8_256gpu.md` | Nemotron4 15B FP8 final benchmark report |
| Apr 13 | `2026-04-13_nemotron4_15b_slack_update.md` | Nemotron4 15B slack update |
| Apr 14 | `2026-04-14_official_baselines_vs_optimized.md` | NVIDIA official configs vs our optimized results, gaps to run |
| Apr 14 | `2026-04-14_benchmark_comparison.md` | Reference TFLOP/s vs Johnson's results, all models side-by-side |
| May 1 | `2026-05-01_dgxc_installation.md` | dgxc-benchmarking install on slinky B200 cluster (issues + fixes) |
| May 2 | `2026-05-02_256gpu_benchmark_report.md` | 256-GPU benchmark sweep on use3a-ss B200 cluster |
| May 10 | `2026-05-10_flapping_airplanes_256gpu_benchmark_report.md` | 256-GPU sweep on flapping-airplanes (15 workloads) |
| May 11 | `2026-05-11_exemplar_benchmark_summary.md` | 512-GPU sweep summary vs 256-GPU and Tranche-1 targets |
| May 11 | `2026-05-11_flapping_airplanes_512gpu_benchmark_report.md` | First full-cluster 512-GPU LLM sweep (8 workloads) |
| May 12 | `2026-05-12_flapping_airplanes_256gpu_benchmark_report.md` | Post-SHARP-abort 256-GPU recovery run (7/8 valid) |
| May 22 | `2026-05-22_rack07_readiness_report.md` | GB200 R07 readiness + per-node + 17n + 16n-with-r07-08 re-runs: r07-06 NVLink degraded (−26%); r07-08 cleared (GPT-OSS 16n incl. r07-08 = 396.6 matches 397 baseline) |
| —     | `template_256gpu_benchmark_report.md` | Template for new N-GPU sweep reports |

## Profiler Reports

| File | Model | Key Finding |
|------|-------|-------------|
| `profiler_reports/405b_nvfp4_nsys_bottleneck.md` | 405B NVFP4 | 67% NCCL, 43% pipeline bubble |
| `profiler_reports/qwen3_235b_bf16_256gpu_profiler.md` | Qwen3 235B BF16 | 54% bubble, MoE AllToAll dominates |
| `profiler_reports/qwen3_235b_bf16_nsys_bottleneck.md` | Qwen3 235B BF16 | 73% comm / 27% compute on early PP stages |
| `profiler_reports/qwen3_235b_fp8mx_256gpu_profiler.md` | Qwen3 235B FP8 | CUDA Graphs +10%, pipeline P2P stalls |
| `profiler_reports/nemotron4_15b_fp8_256gpu_profiler.md` | Nemotron4 15B FP8 | 53% NCCL → 24% with GBS=2048 |

## Analysis Tools

| File | Purpose |
|------|---------|
| `tools/analyze_trace.py` | Parse PyTorch profiler traces |
| `tools/analyze_trace_fast.py` | Fast streaming trace parser |
| `tools/analyze_trace_streaming.py` | Streaming trace analysis for large files |
| `tools/profiling.py` | Nsys profiler analysis utilities |

## Profiler Trace Locations (on cluster)

- Nsys: `/home/johnson/johnson/traces/nsys/` (~6.6 GB)
- PyTorch: `/home/johnson/johnson/traces/pytorch/` (~1.6 GB)
- Qwen3 profiler traces: `/mnt/vast/llmb_/workloads/pretrain_qwen3/profiler_traces/`
