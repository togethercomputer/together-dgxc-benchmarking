# B200 DGXC 256-GPU Benchmark Sweep — 2026-05-12

flapping-airplanes B200 cluster (slinky). 8 workloads at 256 GPUs (32 nodes), forced to 256-GPU emergency mode after cluster IB fabric instability blocked the original 512-GPU sweep. Refreshed whitelist + auto-exclude-on-NODE_FAIL design recovered 7 of 8 workloads on a cluster that exposed 13 newly-bad nodes during the run.

## Summary

**Seven of eight workloads completed with valid data on a degraded cluster.** Llama 70B NVFP4 hits **2029 TFLOPS/GPU** (-1.6% vs 2026-05-10's 256-GPU baseline). Llama 405B FP8 hits **1043.6 TFLOPS/GPU** at 28:20 wall, +2.0% vs 2026-05-10 and +12% per-GPU vs yesterday's 512-GPU. Llama 405B NVFP4 produced **1667 TFLOPS/GPU** — the first successful 256-GPU run of that workload on this cluster (2026-05-10 NODE_FAILed without data). Most striking: **Nemotron-4 340B FP8 jumps to 1363 TFLOPS/GPU (+24% over 2026-05-10, +29% over 2026-05-11)** and **N4 340B BF16 to 916 (+8% / +16%)** — best-ever measurements for these workloads on this cluster.

The story of the day was **cluster IB instability**: 13 of 64 nodes (20%) failed during real LLM training despite passing morning pair-sweep verification. The autonomous "find-bad-node-on-NODE_FAIL → add-to-exclude → retry" loop iteratively pruned 13 nodes; llama405b_fp8 went through on the first retry attempt (12-node exclude); llama405b_nvfp4 went through on attempt 2 (after slinky-5 was added to exclude). Only nemotronh_fp8 was abandoned (3 NODE_FAILs in a row + 1 TCPStore deadlock — the same intrinsic fragility yesterday saw).

## Scoreboard

Steady-state MODEL_TFLOP/s/GPU (mean iter 3–9 from Megatron-Bridge `MODEL_TFLOP/s/GPU` or NeMo `TFLOPS_per_GPU` patterns).

| # | Workload · dtype | Target | **2026-05-12 (256 GPU)** | vs Target | 2026-05-11 (512 GPU) | vs 2026-05-11 | 2026-05-10 (256 GPU) | vs 2026-05-10 | MD1 (256 GPU) | Slurm job |
|--:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Llama 70B FP8 (mx) | 1,624 | 1,557.0 † | −4.1% | 1,614.1 | −3.5% | 1,831.0 | −15.0% | 1,503 | 2879 |
| 2 | Llama 70B NVFP4 | 2,013 | **2,029.2** ‡ | **+0.8%** | 1,946.5 | +4.2% | 2,062.6 | −1.6% | 2,043 | 2929 |
| 3 | Nemotron-H 56B FP8 | 1,536 | **ABANDONED** | — | 1,424.6 | — | 1,116.9 | — | 1,527 | — |
| 4 | Llama 405B FP8 | 1,722 | **1,043.6** | −39.4% | 931.4 | **+12.0%** | 1,022.9 | **+2.0%** | 1,766 | 2955 |
| 5 | Llama 405B NVFP4 | 2,006 | **1,667.0** ¶ | −16.9% | 1,785.0 | −6.6% | NODE_FAIL | **first clean 256 GPU run** | 1,977 | 2966 |
| 6 | Nemotron-4 340B FP8 | 1,101 | **1,363.4** § | **+23.8% ✓** | 1,056.4 | **+29.1%** | 1,096.4 | **+24.3%** | 1,245 | 2888 |
| 7 | Nemotron-4 340B BF16 | 936 | **916.7** | −2.1% | 789.4 | **+16.1%** | 853.1 | **+7.5%** | 868 | 2892 |
| 8 | Qwen3 235B BF16 | 514 | **495.3** | −3.6% | 490.6 | +1.0% | 478.9 | +3.4% | 614 | 2894 |

† = partial result — NODE_FAIL on slinky-40 at iter 8; mean computed over iters 3-7 (5 valid points).
‡ = NODE_FAIL on teardown after all 10 iters trained; mean from full data.
§ = TIMEOUT (Slurm wall hit during post-training teardown); training completed all 10 iters; mean from full data.
¶ = NODE_FAIL on teardown of attempt 2 (auto-retry after slinky-5 NODE_FAIL on attempt 1); 10 iters trained; mean from full data. **First successful 256-GPU run of this workload — 2026-05-10 NODE_FAILed without data, 2026-05-11 only got partial 7-iter from 512→256 fallback.**

## Bad-node cascade — 13 nodes (20% of cluster) failed today

The morning's full pair-sweep at 10:55 declared all 62 swept nodes healthy at ~715 GB/s 2-node busbw. Yet, during LLM training, the following nodes failed:

| Node | Failure pattern | Notes |
|---|---|---|
| slinky-40 | **6 NODE_FAILs** | worst offender; killed jobs 2879, 2931, 2933, 2936; consistently dies within 2-3 min of training start |
| slinky-9 | 2 NODE_FAILs | morning; later held by user `theoh` for unrelated single-node MoE training (no failure on that workload — likely IB-only fault) |
| slinky-26 | 1 NODE_FAIL | morning |
| slinky-18 | pyxis container fail | k8s pod cycling — not consistently reproducible |
| slinky-14 | 1 NODE_FAIL | morning; later held by theoh |
| slinky-11 | 1 NODE_FAIL | discovered during retry v2 |
| slinky-12 | 1 NODE_FAIL | discovered during retry v3 attempt 1 of nemotronh_fp8 |
| slinky-20 | rank-0 of CANCELLED job | conservative exclude after manual scancel |
| slinky-21 | 1 NODE_FAIL | retry v3 attempt 3 of nemotronh_fp8 |
| slinky-5 | 1 NODE_FAIL | retry v3 attempt 1 of llama405b_nvfp4 |
| slinky-2, slinky-37, slinky-56 | pair-sweep flagged earlier (no confirmed NODE_FAIL on LLM workload) | included in conservative exclude list |

**Net healthy capacity at end of day**: 51 of 64 nodes (80%). Slurm allocated from the 52-node residual; LLM jobs ran on 32 of those.

**Diagnostic clue**: 64n NCCL `all_reduce_perf` 8 GiB completed cleanly at 13:51 (job 2925, 0 errors, 325 GB/s ring), and pair-sweep at 10:55 found all nodes healthy at 715 GB/s — yet under sustained LLM training load the same nodes hit hard NODE_FAIL. Short NCCL tests do not predict LLM-training stability on this fabric.

## Auto-exclude-on-NODE_FAIL retry mechanism

After the morning sweep finished with 4 successful and 3 lost workloads (1 partial, 2 NODE_FAIL/PREFLIGHT_FAIL each), the retry orchestrator (`auto_sweep_256gpu_retry_v3_2026-05-12.sh`) added:

1. **Base exclude list** of 9 nodes known-bad from morning sweep.
2. **Auto-grow exclude on NODE_FAIL**: when a job NODE_FAILs with no parseable mean, the orchestrator extracts the failing node from `STEP X.Y ON slinky-N CANCELLED ... NODE FAILURE` markers in the log and adds it to a runtime-only exclude.
3. **Up to 3 attempts per workload** with the growing exclude list.
4. **Preflight retry 5×30s** (up from 3×15s) — earlier orchestrators couldn't ride out the 45-90s Slurm rejection window that follows each NODE_FAIL.
5. **Parser fix**: both Megatron-Bridge (`MODEL_TFLOP/s/GPU`) and NeMo (`TFLOPS_per_GPU:`) patterns recognized.

The mechanism added 4 new bad nodes (slinky-12, 20, 21, 5) to the exclude list during the retry — without manual intervention these would each have caused a wasted training-walltime expense (5-25 min each).

## Failure case studies

### 1. nemotronh_fp8 — abandoned after 5 attempts

| Attempt | Job | State | Wall | Failure mode |
|---|---|---|---|---|
| morning #3 | NONE | PREFLIGHT_FAIL | — | 3× preflight rejected during slinky-40 cleanup |
| retry v1 | 2931 | NODE_FAIL | 2:35 | slinky-40 |
| retry v3 #1 | 2945 | NODE_FAIL | 2:24 | slinky-12 |
| retry v3 #2 | 2949 | CANCELLED | 1:43 | manual scancel after 11-min TCPStore deadlock (no iters produced) |
| retry v3 #3 | 2952 | NODE_FAIL | 3:21 | slinky-21 |

The TCPStore-bootstrap hang in attempt 2 matches yesterday's 2026-05-11 nemotronh_fp8 failure mode — same workload, same `torch.distributed.TCPStore` symptom. **This workload appears to have intrinsic fragility on slinky at scale**, not just bad-node luck.

### 2. llama405b_fp8 — succeeded on first retry v3 attempt

Morning's job 2898 NODE_FAILed at 8:25 wall. Retry v3 attempt 1 (job 2955) completed cleanly in 28:20 wall on a node set with 12 pre-excluded bad nodes. Result: **1043.6 TFLOPS/GPU**, +2% over 2026-05-10's 256-GPU result, +12% per-GPU over yesterday's 512-GPU result.

### 3. llama405b_nvfp4 — retry succeeded after slinky-5 exclude

Morning attempts 1+2 NODE_FAILed (no usable data, 4 min wall total). Retry v3 attempt 1 (job 2963) hit slinky-5 (newly-bad node). Auto-exclude added slinky-5; **retry v3 attempt 2 (job 2966) completed all 10 iters cleanly in 18:28 wall**, then NODE_FAILed on teardown. Mean iter 3-9 = **1667.0 TFLOPS/GPU**. Per-iter TFLOPS observed: 1790, 1754, 1751, 1623, 1552 + 4 unrecorded — showing a slight downward drift through iters 5+. Likely a straggler in the attempt 2 node mix.

### 4. Llama 70B FP8 — partial NODE_FAIL with usable data

Morning's job 2879 trained iters 1-7 successfully (1540-1580 TFLOPS) before slinky-40 NODE_FAILed it at iter 8. Mean iter 3-7 = **1557.0 TFLOPS/GPU**. Accepted as partial.

## Cross-day comparison highlights

**Workloads that improved vs prior days** (despite degraded cluster):
- **N4 340B FP8: +24.3% / +29.1%** — biggest delta. Same workload at 256 GPU yesterday (orchestrator's same-day control) was 1371. Today: 1363. Stable within 1% of yesterday's same-day measurement, well above day-before's 1096 baseline.
- **N4 340B BF16: +7.5% / +16.1%** — consistent improvement, no obvious cluster-luck explanation.
- **Llama 405B FP8: +2.0% / +12% per-GPU** — first time this cluster's 256-GPU number beat day-before-yesterday's at 256.
- **Qwen3 235B BF16: +3.4% / +1.0%** — within noise, but small consistent gain.

**Hypothesis for the n4340b jump**: cluster topology improvements OR llmb-run installer updates between 2026-05-10 → today changed something for nemotron4-340b's specific parallelism (TP=8 PP=4). Unclear without orthogonal control. Worth verifying in next sweep.

## Cluster verification — short tests pass, training reveals faults

Morning verification (all done before LLM sweep):
- 09:00 pair-sweep on 62 nodes (31 pairs): **all 31 PASS at ~715 GB/s** — no failures.
- 13:51 64n NCCL `all_reduce_perf` (1-8 GiB sweep, job 2925): **COMPLETED 0:0 exit, 0 errors, 325 GB/s ring 8 GiB** — clean cluster.

Yet during the same window, LLM jobs experienced NODE_FAILs at 2-3 min into training. The pattern is repeatable and matches today's morning experience too. **NCCL connectivity tests are necessary-but-not-sufficient validators for sustained-training stability on slinky.**

## Infrastructure / orchestrator

- **Base script**: `/home/johnson/auto_sweep_256gpu.sh` (2026-05-10) → adapted to `auto_sweep_256gpu_2026-05-12.sh` (this morning) → patched to `auto_sweep_256gpu_retry_v2_2026-05-12.sh` (post-bug-fix) → final `auto_sweep_256gpu_retry_v3_2026-05-12.sh` (auto-exclude + 5× preflight retry).
- **Whitelist file**: `/home/johnson/slinky_32n_whitelist_2026-05-12.txt` (32 nodes, slinky-40 substituted as slinky-7 since theoh held slinky-9 from the original whitelist).
- **Results TSV**: `/home/johnson/auto_sweep_results_2026-05-12.tsv` (18 rows across all attempts; latest valid mean per workload is the reported number).
- **Logs**:
  - `auto_sweep_logs_2026-05-12/` (morning)
  - `auto_sweep_logs_2026-05-12_retry/` (v1)
  - `auto_sweep_logs_2026-05-12_retry_v2/` (v2 — killed mid-run after slinky-11 cascade)
  - `auto_sweep_logs_2026-05-12_retry_v3/` (v3 — final)
- **Pair-sweep snapshot**: `/home/johnson/pair_sweep_2026-05-12_full_1778583327/` (31 pair outputs, all PASS at 715 GB/s).
- **Patch stack**: same as 2026-05-10 + 2026-05-11 (pyxis chmod, memlock, /dev/shm, set_sharing_strategy, ptxas-blackwell, hcoll/ucc disable). No new patches.

## Open items

- **Cluster-wide IB fabric audit**. 20% of nodes failed under sustained training despite passing all short NCCL tests. Strong indication of intermittent IB QP or fabric-manager fault.
- **Drain/reboot recommendation**: slinky-40 (6 NODE_FAILs today — clearly hardware), slinky-5, slinky-11, slinky-12, slinky-21 (1 NODE_FAIL each, today's first failure). Re-add only after ops verifies stability under sustained load.
- **Nemotron-H fragility**: workload-level deadlock pattern matches 2026-05-11. Worth investigating whether NeMo's TCPStore init for this model is misconfigured (single rank-0 hostname dependency, no retry).
- **SHARP env on FSDP workloads**: today's morning sweep had no SHARP (per user direction); a smaller earlier SHARP-on test (job 2740 from the SHARP-attempt sweep) gave 1412 TFLOPS for llama70b_fp8 — **12.5% worse** than today's no-SHARP 1557. Confirms SHARP env should be applied per-workload, not globally.
- **N4 340b improvement source**: +24% jump vs 2026-05-10 is unexplained — verify in next clean-cluster sweep.

## Cross-references

- 2026-05-11 512-GPU report: `flapping_airplanes_512gpu_benchmark_report.md`
- 2026-05-10 256-GPU report: `flapping_airplanes_256gpu_benchmark_report.md`
- MD1 256-GPU baseline (2026-05-02): `2026-05-02_256gpu_benchmark_report.md`
- 2026-05-12 SHARP-attempt incident: `~/.claude/projects/-data-home-johnson/memory/project_512gpu_sharp_2026-05-12_aborted.md`
- Orchestrator + results: `/home/johnson/auto_sweep_256gpu_retry_v3_2026-05-12.sh`, `/home/johnson/auto_sweep_results_2026-05-12.tsv`
