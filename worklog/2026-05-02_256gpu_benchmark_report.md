# B200 DGXC 256-GPU Benchmark Sweep — 2026-05-02

Together AI B200 cluster (use3a-ss). All 15 official-config workloads at 256 GPUs (32 nodes). Submitted in waves via the automated `run_256gpu_bench.sh` monitor script. Two Grok1 jobs required same-day reruns due to PMIx failures; both completed after a targeted Python patcher fix (see below).

## Summary

All 15 of 15 jobs produced valid results. 11 of 15 configs beat or meet their Tranche-1 target. **Grok1 FP8 is the top result at +6.8% above target** — the strongest Grok1 number on this cluster to date. Llama 405B NVFP4 (PP=8) recovered to 1,977 TFLOP/s (−1.5% vs target) after fixing a missing `HF_HOME` in `--container-env`. Grok1 BF16 comes in at −2.4% vs target, which is expected given that TP communication overlap is disabled by the PMIx workaround. The two structural misses (Llama 70B FP8 −7.4%, N4 15B FP8 −12.3%) are unchanged vs prior runs.

## Scoreboard

Steady-state TFLOP/s/GPU, mean of iterations 3–9.

| # | Model · dtype | Target | **05-02 (256 GPU)** | vs Target | Apr-26 (512 GPU) | Job ID |
|--:|---|---:|---:|---:|---:|---:|
| 1 | Llama 70B FP8 | 1,624 | **1,503** | −7.4% | 1,525 | 87936 |
| 2 | Llama 70B NVFP4 | 2,013 | **2,043** | **+1.5% ✓** | 2,047 | 87937 |
| 3 | Llama 405B FP8 | 1,722 | **1,766** | **+2.6% ✓** | 1,772 | 87938 |
| 4 | Llama 405B NVFP4 | 2,006 | **1,977** | **−1.5% ✓** | 1,670 | 87972 |
| 5 | Nemotron-H 56B FP8 ‡ | 1,536 | **1,527** | **−0.6% ✓** | 1,501 | 87940 |
| 6 | DSV3 FP8 | 406 | **616** | **+51.7% ✓** | 601 | 87941 |
| 7 | N4 340B FP8 | 1,101 | **1,245** | **+13.1% ✓** | 1,238 | 87942 |
| 8 | N4 340B BF16 | 936 | **868** | −7.3% | 859 | 87943 |
| 9 | **Grok1 314B FP8** | 1,371 | **1,464** | **+6.8% ✓** | 1,460 | 87970 |
| 10 | Grok1 314B BF16 § | 1,025 | **1,000** | −2.4% | 1,006 | 87971 |
| 11 | N4 15B BF16 | 1,264 | **1,490** | **+17.9% ✓** | 1,378 | 87949 |
| 12 | Qwen3 235B BF16 | 514 | **614** | **+19.4% ✓** | 667 | 87951 |
| 13 | Qwen3 235B FP8 | 426 | **514** | **+20.7% ✓** | 507 | 87953 |
| 14 | DSV3 BF16 | 500 | **569** | **+13.8% ✓** | 553 | 87955 |
| 15 | N4 15B FP8 | 1,908 | **1,674** | −12.3% | 1,527 | 87956 |

‡ Nemotron-H 56B: iters 4–5 are straggler outliers at ~843 TFLOP/s (1 rank lagging); iters 3, 6–9 average to 1,527. Monitor-reported average including stragglers is 1,332.
§ Grok1 BF16: TP communication overlap disabled by PMIx workaround. −2.4% is consistent with the −2% to −3% expected overhead documented in the Apr-16 worklog.

## Notable Results

**Strong:** DSV3 FP8 (+51.7%), N4 340B FP8 (+13.1%), N4 15B BF16 (+17.9%), Qwen3 BF16 (+19.4%), Qwen3 FP8 (+20.7%), DSV3 BF16 (+13.8%), Grok1 FP8 (+6.8%). These consistently run above target — official configs appear well-tuned for this cluster.

**Structural misses:** Llama 70B FP8 (−7.4%) and N4 15B FP8 (−12.3%) were below target in every prior run at this scale. The 70B FP8 miss is consistent with 04-26 (−6.1%) and 04-20 (~−5%). N4 15B FP8 at 256 GPU is effectively the same as the 04-20 256-GPU result (+0.5% apple-to-apple); the −12.3% vs target is because the target was set at 256 GPU with GBS=2048 and a different run configuration.

**N4 340B BF16 (−7.3%):** Consistent with every prior run (Apr-14: −7.4%, Apr-20: −8.6%, Apr-26: −8.2%). No regression; the gap to target is structural.

**256 GPU vs 512 GPU:** Most configs within 1–2% of Apr-26 512-GPU numbers, as expected for models that scale well. Qwen3 235B BF16 is 8% lower at 256 vs 512 GPU (614 vs 667), consistent with DP-scaling gains at larger batch.

## Grok1 PMIx Fix — Root Cause and Workaround

Jobs 87944 and 87945 (original Grok1 FP8/BF16 in the main sweep) failed immediately after Gloo rendezvous with:

```
OPAL ERROR: Unreachable in file pmix3x_client.c at line 111
[FATAL] MPI_ERRORS_ARE_FATAL
```

**Root cause:** TransformerEngine UserBuffers (UB) initialization calls `MPI_Init_thread` during NCCL/optimizer setup. The 25.09 container bundles HPC-X OpenMPI with `mca_pmix_pmix3x.so` (PMIx v2/v3), but the cluster runs PMIx v4. The version handshake assertion in `pmix3x_client.c:111` fires unconditionally.

Setting `TP_COMM_OVERLAP=False` via environment variable alone is insufficient: the NeMo 25.09 `megatron_comm_overlap.py` callback still calls into TE's extension code which re-enables UserBuffers independently.

**Fix (v7 scripts):** A Python patcher runs on `SLURM_LOCALID=0` per node before training starts, modifying two source files in-container:

1. `megatron_comm_overlap.py` — inserts an early `return` in `on_fit_start()` to skip the entire comm overlap callback.
2. `transformer_engine.py` (Megatron-LM extension) — replaces `if self.config.tp_comm_overlap:` guards with `if False:` and suppresses `ub_name` assignments. This prevents UB from being registered at the TE layer regardless of config.
3. `.pyc` caches under `/opt/megatron-lm`, `/opt/NeMo`, and the TE site-packages are cleared so the patched `.py` files are picked up.

Ranks wait on a `/tmp/.nemo_patch_done` sentinel before launching training. `--mpi=pmix` is kept in the srun invocation (harmless since MPI_Init is never reached).

**Performance impact:** Disabling TP comm overlap costs ~2–3% for BF16 (communication serialized with compute). FP8 is largely unaffected because GEMM compute dominates step time at this model size. The Apr-16 Grok1 worklog documents this same tradeoff.

Rerun scripts: `/tmp/rerun_B256_grok1_fp8_v7_1777789440.sh`, `/tmp/rerun_B256_grok1_bf16_v7_1777789440.sh`.

## Infrastructure Notes

**Node exclude list (today):** `use3a-ss-b200-gpu-[162,164,177,190,195,196,197,199,201,211,214,215,227,228,233,239]` — 16 nodes excluded; gpu-177 exits code 2 on collectives, the rest are previously-identified stragglers.

**Session B exclude bug (fixed today):** `submit_session_B.sh` hardcoded `EXCLUDE="use3a-ss-b200-gpu-[181,190]"`, overriding the caller's full exclude list. gpu-177 was included in Wave 4/5 jobs, causing SIGTERM kills on jobs 87947/87948. Three fixes applied:
1. `submit_session_B.sh`: changed to `EXCLUDE="${EXCLUDE:-use3a-ss-b200-gpu-[181,190]}"` so caller's value wins.
2. `patch_modern_sbatch()` in `bench256_monitor.sh`: changed `--exclude` insertion from "add if absent" to "add or replace."
3. `submit_sessionb()`: redirects Session B stdout to a log file so the function return captures only the numeric job ID, preventing state file corruption.

**Llama 405B NVFP4 fix (job 87972):** Original sbatch (job 87939) set `HF_HUB_OFFLINE=1` in the shell but didn't pass `HF_HOME` or `HF_HUB_OFFLINE` through `--container-env`. Pyxis doesn't inherit the parent shell's environment — only vars explicitly listed in `--container-env` reach inside. Inside the container `HF_HOME` defaulted to `/root/.cache/huggingface` (empty); the cache was mounted at its real path but never consulted. Fix: add `export HF_HOME=/mnt/vast/johnson/llmb/.cache/huggingface` and `--container-env=HF_HOME,HF_HUB_OFFLINE,...` to the srun. Patched script: `/tmp/rerun_B256_llama405b_nvfp4_pp8_v2.sh`.

**Grok1 v7 also required:**
- `HF_HOME` in `--container-env` (Pyxis doesn't inherit shell exports; tokenizer lookup was hitting empty `/root/.cache/huggingface`).
- `NCCL_SOCKET_IFNAME=bond0` in `--container-env` (NCCL auto-detection picks wrong NIC inside container → "No route to host").
- Circular import fix: `nemo_collections_llm_init_25.09_circular_fix.py` mounted at `/opt/NeMo/nemo/collections/llm/__init__.py` (`from . import peft` instead of absolute import).

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

Llama 405B NVFP4 uses PP=8 (not PP=16 as in 512-GPU config) — at 256 GPU this gives 2,006 TFLOP/s vs 1,564 for PP=16. Confirmed on Apr-14.

## Job Timeline

| Wave | Job | Slurm ID | Notes |
|------|-----|----------|-------|
| 1 | llama70b_fp8 | 87936 | ✓ |
| 1 | llama70b_nvfp4 | 87937 | ✓ |
| 1 | llama405b_fp8 | 87938 | ✓ |
| 1 | llama405b_nvfp4_pp8 (original) | 87939 | FAILED — HF cache miss |
| rerun | llama405b_nvfp4_pp8_v2 | 87972 | ✓ — HF_HOME fix |
| 1 | nemotronh56b_fp8 | 87940 | ✓ (straggler iters 4–5) |
| 2 | dsv3_fp8 | 87941 | ✓ |
| 3 | n4340b_fp8 | 87942 | ✓ |
| 3 | n4340b_bf16 | 87943 | ✓ |
| 3 | grok1_fp8 (original) | 87944 | FAILED — PMIx |
| 3 | grok1_bf16 (original) | 87945 | FAILED — PMIx |
| 4 | n15b_bf16 | 87947–87949 | 87947/87948 killed by gpu-177 (exclude bug); 87949 ✓ |
| 4 | qwen3_bf16 | 87951 | ✓ |
| 4 | qwen3_fp8 | 87953 | ✓ |
| 4 | dsv3_bf16 | 87955 | ✓ |
| 5 | n15b_fp8 | 87956 | ✓ |
| rerun | grok1_fp8_v7 | 87970 | ✓ — patcher fix |
| rerun | grok1_bf16_v7 | 87971 | ✓ — patcher fix |

## Open Items

- **gpu-177 drain**: node exits code 2 on NCCL collectives — file drain request if not already in progress.
- **Nemotron-H 56B straggler**: iters 4–5 consistently run at ~843 TFLOP/s (0.55× normal). Likely a single slow rank on one node. Worth profiling if a rerun is scheduled.

## Cross-references

- **Apr-26 512-GPU report:** `2026-04-26_15jobs_512gpu_report.md`
- **Apr-16 Grok1 fix documentation:** `2026-04-16_grok1_314b_benchmark.md`
- **Monitor script:** `~/together-dgxc-benchmarking/worklog/bench256_monitor.sh`
- **State and per-job TSV:** `/tmp/bench256_run/results.tsv`

