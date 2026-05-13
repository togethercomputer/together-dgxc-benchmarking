# B200 DGXC [N]-GPU Benchmark Sweep — [DATE]

[CLUSTER_NAME] B200 cluster ([CLUSTER_ID]). All [N] official-config workloads at [N] GPUs ([N] nodes). [Brief context: waves, reruns, notable issues.]

## Summary

[N] of [N] jobs produced valid results. [N] of [N] configs beat or meet their Tranche-1 target. [1–2 sentence headline — top result, any structural misses, key fix applied.]

## Scoreboard

Steady-state TFLOP/s/GPU, mean of iterations 3–9.

| # | Model · dtype | Target | **[DATE] ([N] GPU)** | vs Target | May-02 (256 GPU) | vs May-02 | Job ID |
|--:|---|---:|---:|---:|---:|---:|---:|
| 1 | Llama 70B FP8 | 1,624 | — | — | 1,503 | — | — |
| 2 | Llama 70B NVFP4 | 2,013 | — | — | 2,043 | — | — |
| 3 | Llama 405B FP8 | 1,722 | — | — | 1,766 | — | — |
| 4 | Llama 405B NVFP4 | 2,006 | — | — | 1,977 | — | — |
| 5 | Nemotron-H 56B FP8 ‡ | 1,536 | — | — | 1,527 | — | — |
| 6 | DSV3 FP8 | 406 | — | — | 616 | — | — |
| 7 | N4 340B FP8 | 1,101 | — | — | 1,245 | — | — |
| 8 | N4 340B BF16 | 936 | — | — | 868 | — | — |
| 9 | Grok1 314B FP8 | 1,371 | — | — | 1,464 | — | — |
| 10 | Grok1 314B BF16 § | 1,025 | — | — | 1,000 | — | — |
| 11 | N4 15B BF16 | 1,264 | — | — | 1,490 | — | — |
| 12 | Qwen3 235B BF16 | 514 | — | — | 614 | — | — |
| 13 | Qwen3 235B FP8 | 426 | — | — | 514 | — | — |
| 14 | DSV3 BF16 | 500 | — | — | 569 | — | — |
| 15 | N4 15B FP8 | 1,908 | — | — | 1,674 | — | — |

‡ [Note on straggler iters if applicable]
§ [Note on TP comm overlap or PMIx workaround if applicable]

## Notable Results

**Strong:** [models beating target — list with deltas]

**Structural misses:** [models below target — note if consistent with prior runs]

**[Model] fix:** [if a rerun or fix was applied, document root cause and fix here]

## [Optional] [Fix Title] — Root Cause and Workaround

[If a container/PMIx/config fix was needed, document it here with the same structure as the Grok1 PMIx fix in the May-02 report.]

## Infrastructure Notes

**Node exclude list (today):** `[cluster]-gpu-[node_list]` — [N] nodes excluded; [any notable bad nodes].

**[Any script bugs fixed today, container-env issues, or other infra changes.]**

## Configurations

| Model · dtype | Container | Parallelism | GBS |
|---|---|---|---:|
| Llama 70B FP8/NVFP4 | nemo+26.02 | TP=2 PP=4 CP=1 VP=5 | 256 |
| Llama 405B FP8 | nemo+26.02 | TP=4 PP=8 CP=2 VP=8 | 3072 |
| Llama 405B NVFP4 | nemo+26.02 | TP=4 PP=8 CP=1 VP=4 | 1536 |
| Nemotron-H 56B FP8 | nemo+26.02 | TP=2 PP=1 CP=1 | 192 |
| DSV3 FP8/BF16 | nemo+26.02 | TP=1 PP=16 CP=1 EP=8 | 4096 |
| N4 340B FP8/BF16 | nemo+25.07 | TP=8 PP=4 CP=1 VP=12 | 64 |
| Grok1 314B FP8/BF16 | nemo+25.09 | TP=4 PP=4 CP=1 VP=8 EP=8 ETP=1 | 512 |
| N4 15B BF16 | nemo+26.02 (compat_runner) | TP=1 PP=1 CP=1 VP=1 | 2048 |
| N4 15B FP8 | nemo+25.09 | TP=1 PP=1 CP=1 VP=1 | 2048 |
| Qwen3 235B BF16/FP8 | nemo+26.02 | TP=1 PP=8 CP=1 VP=4 EP=8 | 8192 |

[Update any configs that changed from the above.]

## Job Timeline

| Wave | Job | Slurm ID | Notes |
|------|-----|----------|-------|
| 1 | llama70b_fp8 | — | — |
| 1 | llama70b_nvfp4 | — | — |
| 1 | llama405b_fp8 | — | — |
| 1 | llama405b_nvfp4_pp8 | — | — |
| 1 | nemotronh56b_fp8 | — | — |
| 2 | dsv3_fp8 | — | — |
| 3 | n4340b_fp8 | — | — |
| 3 | n4340b_bf16 | — | — |
| 3 | grok1_fp8 | — | — |
| 3 | grok1_bf16 | — | — |
| 4 | n15b_bf16 | — | — |
| 4 | qwen3_bf16 | — | — |
| 4 | qwen3_fp8 | — | — |
| 4 | dsv3_bf16 | — | — |
| 5 | n15b_fp8 | — | — |

## Open Items

- [Any nodes to drain, models to profile, or fixes to upstream]

## Cross-references

- **May-02 256-GPU report:** `2026-05-02_256gpu_benchmark_report.md`
- **Monitor script:** `~/together-dgxc-benchmarking/worklog/bench256_monitor.sh`
- **State and per-job TSV:** `/tmp/bench256_run/results.tsv`
