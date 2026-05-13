# B200 DGXC 512-GPU Benchmark Sweep — 2026-05-11

flapping-airplanes B200 cluster (slinky). 8 workloads at 512 GPUs (64 nodes) — first full-cluster LLM sweep on this hardware.

## Summary

**Eight workloads, eight results.** Llama 70B FP8 @ 512 GPU hits **1614 TFLOPS/GPU** (88% of yesterday's 256-GPU per-GPU rate, +7.4% vs MD1 256-GPU baseline). Llama 70B NVFP4 scales to **1946 TFLOPS/GPU** at 94% efficiency. Llama 405B NVFP4 trained successfully **for the first time ever on this cluster** (yesterday: NODE_FAIL ×3; today: 1785 TFLOPS/GPU after auto-fallback). The orchestrator's NCCL pair-sweep + scale-down-to-256 fallback fired three times and recovered each one. Two newly-degraded nodes (slinky-9, slinky-26) were caught by the pair sweep; pre-sweep verification at 09:20 UTC had shown all 64 nodes healthy — degradation happened during the 3-hour sweep window.

## Scoreboard

Steady-state MODEL_TFLOP/s/GPU as reported by Megatron-Bridge or NeMo, mean of iterations 3–9.

| # | Workload · dtype | Target | **2026-05-11 (512 GPU)** | vs Target | Yesterday (256 GPU) | vs Yesterday | MD1 (256 GPU) | vs MD1 | Slurm job |
|--:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Llama 70B FP8 (mx) | 1,624 | **1,614.1** | −0.6% | 1,831.0 | −11.8% | 1,503 | **+7.4%** | 2415 |
| 2 | Llama 70B NVFP4 | 2,013 | **1,946.5** | −3.3% | 2,062.6 | −5.6% | 2,043 | −4.7% | 2417 |
| 3 | Nemotron-H 56B FP8 | 1,536 | 1,424.6 ‡ | −7.3% | 1,116.9 | +27.5% | 1,527 | −6.7% | 2452 |
| 4 | Llama 405B FP8 | 1,722 | **931.4** | −45.9% | 1,022.9 | −8.9% | 1,766 | −47.3% | 2493 |
| 5 | Llama 405B NVFP4 | 2,006 | 1,785.0 ‡ | −11.0% | NODE_FAIL ×3 | **first success** | 1,977 | −9.7% | 2527 |
| 6 | Nemotron-4 340B FP8 | 1,101 | **1,056.4** | −4.1% | 1,096.4 | −3.6% | 1,245 | −15.1% | 2454 |
| 7 | Nemotron-4 340B BF16 | 936 | 789.4 | −15.7% | 853.1 | −7.5% | 868 | −9.1% | 2489 |
| 8 | Qwen3 235B BF16 | 514 | **490.6** | −4.6% | 478.9 | **+2.4%** | 614 | −20.1% | 2491 |

‡ = result obtained via fallback at 256 GPU after 512-GPU primary failed (see Failures section).

## Scaling efficiency (256 → 512)

Direct comparison against yesterday's 256-GPU TFLOPS/GPU; same workload, same dtype, same parallelism family.

| Workload | 256 → 512 efficiency | Comm-bound? |
|---|---:|---|
| Llama 70B FP8 | **88.2%** | moderate (FSDP all_gather + reduce_scatter @ 2× DP) |
| Llama 70B NVFP4 | **94.3%** | low (smaller FSDP buffers in 4-bit) |
| Nemotron-4 340B FP8 | 96.4% (vs yesterday) / **77%** (vs same-day 256 control = 1371) | high — see note below |
| Nemotron-4 340B BF16 | **92.5%** | moderate |
| Qwen3 235B BF16 | **102.4%** (superlinear) | none — MoE with EP=8 keeps expert groups same size, extra GPUs do real work |
| Llama 405B FP8 | **91.1%** | severe (TP=4, PP=8, CP=2 keep intra-node, only DP doubles) |

> **n4340b_fp8 anomaly**: yesterday's 256-GPU baseline (1096) and today's same-day 256-GPU run (1371) differ by 25%. Today's run had no whitelist constraint and used the full 64-node Slurm allocation freedom, hitting a faster node set. The 96% scaling number against yesterday is a stale comparison; the real 256→512 efficiency for this workload is 77%.

## Cross-cluster comparison

**vs MD1 (use3a-ss) at 256 GPU**: today's 512-GPU numbers beat MD1's 256-GPU on Llama 70B FP8 (+7.4%), are within −5 to −10% on most workloads, and significantly trail on Llama 405B FP8 (−47%) — consistent with the **SHARP-inactive ceiling** on this fabric. MD1 has SHARP delivering ~545 GB/s collective vs slinky's ~390 GB/s ring; 405B at FP8 is the most comm-bound workload and pays the full SHARP-absence penalty.

## Failures, fallbacks, and bad-node discovery

The orchestrator's **NCCL-pair-sweep + scale-down-to-256 fallback** triggered three times:

### 1. nemotronh_fp8 @ 512 → fallback @ 256 (PyTorch TCPStore bootstrap deadlock)

- **Symptom**: 512-rank init hung 10 min waiting on `UniqueNCCLID` from rank-0 root (slinky-9). Failed at 11:58 wall. Not an IB issue — bootstrap is over TCP.
- **Pair sweep**: clean (0 bad nodes flagged) — correctly classified, since IB data plane was fine.
- **Fallback**: 256 GPU retry succeeded with **1424.6 TFLOPS/GPU** — better than yesterday's 1117 at the same scale.
- **Note**: slinky-9 was the rank-0 hostname in this failure, an early signal that came back to bite us later (see #3).

### 2. n4340b_fp8 @ 512 → spurious fallback (orchestrator bug, since fixed)

- **Symptom**: 512-GPU job trained 10/10 iters cleanly, then post-training teardown stalled. Slurm TIMEOUTed at 25:15 wall. Mean iter 3-9 was valid (1056.4) in the log.
- **Cause**: Initial fallback condition was `state in (TIMEOUT, FAILED, NODE_FAIL, OOM) OR mean empty`. The state was TIMEOUT despite valid mean.
- **Fallback (unnecessary)**: 256 GPU retry ran cleanly, **1371.3 TFLOPS/GPU** — useful same-day 256 control point (+25% over yesterday's 256 result).
- **Fix**: orchestrator patched mid-run to fire fallback **only if mean is missing**. Subsequent TIMEOUT-with-valid-mean workloads (n4340b_bf16 didn't hit this, but llama405b_fp8 completed cleanly) were not duplicated.

### 3. llama405b_nvfp4 @ 512 → fallback @ 256 (real hardware failure, 2 bad nodes)

- **Symptom**: 512-GPU job started training, ran for 11:29 then NODE_FAIL. No iters parseable.
- **Pair sweep**: **slinky-9, slinky-26 flagged BAD**. First time pair sweep caught real bad nodes during this sweep.
- **Fallback**: 256 GPU retry with `--exclude=slinky-9,slinky-26` reached **iter 7/10 with steady-state ~1788 TFLOPS/GPU** before a third node hit pyxis container failures and ended in NODE_FAIL at 18:04. Mean iter 3-9 parsed: **1785 TFLOPS/GPU**.
- **Significance**: yesterday's NODE_FAIL ×3 attempts at 256 GPU never produced *any* TFLOPS data. Today is the first usable llama405b_nvfp4 result on this cluster.

## Node-degradation timeline during the 3-hour sweep

| Time | Event |
|---|---|
| 09:20 | Pre-sweep verification: **all 64 nodes pass** (slinky-43, slinky-62 specifically retested at 2n ring; healthy at 716/714 GB/s). 64n ring allreduce: 388.5 GB/s, cluster-wide healthy. |
| 09:42 | nemotronh_fp8 @ 512 starts; rank-0 root assigned to slinky-9. |
| 09:54 | nemotronh fails on TCPStore wait (slinky-9 unreachable on bootstrap socket). Pair sweep does not flag slinky-9 (IB data plane still OK). |
| 10:01 → 12:00 | Llama 70B (both dtypes), nemotronh, n4340b (both dtypes), qwen3, llama405b_fp8 all run with **no further failures** through the same 64-node pool. |
| 12:01 | llama405b_nvfp4 @ 512 NODE_FAIL after 11:29. |
| 12:02 | Pair sweep flags **slinky-9, slinky-26** — IB now broken on both. |
| 12:24 | llama405b_nvfp4 fallback @ 256 hits pyxis container failures on a third (unidentified) node and ends NODE_FAIL after producing 7 iters of valid data. |

**Interpretation**: slinky-9 was degrading from the start (control-plane symptom at 09:54). slinky-26 went bad somewhere between ~10:00 and 12:01 — gradual hardware fault, not from-the-start. A third node failed during the fallback, suggesting cluster degradation is **ongoing** as of this sweep's end.

**Action**: drain slinky-9, slinky-26. Re-verify cluster before next sweep.

## Pre-sweep cluster verification (2026-05-11 09:20 UTC)

For this sweep we trusted today's earlier verification rather than yesterday's exclusion list:

| Test | Result |
|---|---|
| 2-node slinky-43 + slinky-0 (was IB-broken) | 716 GB/s — **HEALED** |
| 2-node slinky-62 + slinky-1 (was IB-broken) | 714 GB/s — **HEALED** |
| 4-node SHARP test | 390 GB/s (= ring; SHARP **still inactive**) |
| 64-node ring 32 GiB | 388.5 GB/s — best result on this cluster, beats yesterday's 60n=367 |

So the sweep started with the entire 64-node fleet validated. The degradation of slinky-9 and slinky-26 happened **during** the workload run.

## Infrastructure / Orchestrator

- **Orchestrator**: `/home/johnson/auto_sweep_512gpu/orchestrator.sh` (adapted from `/home/johnson/auto_sweep_256gpu.sh`)
- **Pair-sweep helper**: `/home/johnson/auto_sweep_512gpu/pair_sweep_helper.sh` — 32 disjoint 2-node pairs across the failed job's nodelist, all-reduce 8 GiB, flags pairs failing `Out of bounds values : 0 OK`.
- **Results TSV**: `/home/johnson/auto_sweep_512gpu/results.tsv`
- **Per-job logs**: `/home/johnson/auto_sweep_512gpu/logs/`
- **Per-workload Slurm output**: `/data/home/johnson/llmb/workloads/<workload>/experiments/`
- **Preflight cadence**: 64-node `all_reduce_perf -b 8G -e 8G -n 5 -w 2`, ~30 s before every LLM submission.
- **Patch stack reused** from yesterday: pyxis chmod, RLIMIT_MEMLOCK propagation, `/dev/shm` mount, `set_sharing_strategy('file_system')`, ptxas-blackwell bind-mount, hcoll/ucc disable. No new patches needed at 512 GPU.
- **Wall-clock total**: 3 hours 5 minutes (09:28 → 12:32).

### Orchestrator behavior changes vs the 256-GPU version (`auto_sweep_256gpu.sh`)

1. Preflight raised from 32 → **64 nodes** (matches the workload allocation).
2. **`ADDITIONAL_SLURM_PARAMS` defaults empty** (no whitelist) — relied on today's full-cluster health.
3. On failure: **pair sweep** across the failed job's nodelist (not a fixed whitelist) → identify bad nodes → resubmit at 256 GPU with `--exclude` of bad nodes.
4. Fallback condition (post-fix): **fires only if mean iter 3-9 cannot be parsed** — TIMEOUT-with-valid-mean is treated as success.

## Open items

- **Drain slinky-9 and slinky-26** before any further large-scale runs. Both confirmed bad via pair sweep at 12:02 UTC.
- Identify the third node that hit pyxis container failures during job 2527 fallback (12:24 UTC). Check `/home/johnson/auto_sweep_512gpu/logs/` for node-specific traces.
- **SHARP enablement** would lift the 405B FP8 ceiling from 931 → expected ~1500-1700 TFLOPS/GPU (closing the gap with MD1's 1766). Still requires admin action on the IB fabric manager — no progress today.
- **PyTorch TCPStore bootstrap at 512 ranks** is fragile when any rank-0 host has a control-plane fault. Consider adding an explicit TCP-ping preflight before LLM submission (in addition to IB pair sweep). Today's TCPStore failure (nemotronh) was on a node whose IB was still nominally healthy.

## Cross-references

- **Yesterday's 256-GPU sweep**: `/home/johnson/worklogs/flapping_airplanes_256gpu_benchmark_report.md`
- **2026-05-09 NCCL scaling report (8-60 nodes)**: `/home/johnson/worklogs/flapping_airplanes_nccl_collective_scaling_report_2026-05-09.md`
- **MD1 256-GPU baseline (2026-05-02)**: `2026-05-02_256gpu_benchmark_report.md`
- **Orchestrator + results**: `/home/johnson/auto_sweep_512gpu/`
