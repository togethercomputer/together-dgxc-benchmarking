# Cluster 661303b8 Readiness Report — 2026-11-03

GB200 NVL72 cluster `661303b8` (control plane `oh1bl-hn02`). Acceptance benchmark pass for a **partial rack: 12 GB200 nodes / 48 GPUs** (`661303b8-{04,05,08,09,11..18}`), all in a single NVLink clique (`1849c6a7-9009-4917-9e85-66803cbe32df.32766`). Nodes 01–03, 06, 07, 10 are not joined to the cluster. Workload set and methodology mirror the R01 / R07 / R08 / R10 passes, rescaled to fit 12 nodes (see *Config deltas*).

Reservation: all 12 GPU nodes are labeled + tainted `reserved-for=infra-perf:NoSchedule`, enforced by a `ValidatingAdmissionPolicy` (`reserved-for-infra-perf`) so only namespace `infra-perf` may tolerate the taint. All jobs ran in `infra-perf`. Reservation script / policy: `tests/gb200/GB200/c66-tools/reserve/`.

## Summary

**Verdict: NOT READY — cluster-wide NVLink bandwidth deficit (~28%).** The fabric is uniformly slow, not a straggler: every node, single-node and multi-node, delivers ~73% of the healthy GB200 NVLink all-reduce bandwidth. Training throughput follows it down (Qwen −13 to −18%, GPT-OSS −18 to −23% vs prior racks). All results are extremely reproducible (≤0.15% run-to-run), so this is a systematic condition, not noise.

- NVML reports the fabric **Healthy / Bandwidth: Full**, all 18 NVLinks per GPU up at 53.125 GB/s, no thermal/power throttling (1200 W limit) — i.e. nothing visible from the host flags the problem.
- **Container stack ruled out:** re-running NCCL with `pytorch:25.09` (CUDA 13.0.1 / NCCL 2.27.7, matching the host driver's CUDA 13.0) gives the identical 663 GB/s as `pytorch:26.02` (CUDA 13.1 forward-compat / NCCL 2.29.2).
- Remaining suspects (platform / NVIDIA to investigate): host driver **580.196.07** vs other clusters, NVSwitch tray firmware / fabric manager config, partial-rack NVSwitch configuration (only 12 of 18 compute trays joined).

Workloads: NCCL 12n ×1 (26.02) + ×1 (25.09), per-node single-node NVLink sweep (11 nodes), Qwen 72B ×2, GPT-OSS 120B ×2. One job at a time, torn down (pods + ComputeDomain) between stages.

## Scoreboard

| # | Workload | 661303b8 (run1 / run2 → mean) | R07 (Jul) | R01 (Jul) | R08 (Apr) | R10 (Apr) | vs prior range |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | NCCL 12n busbw @ 32 GiB (GB/s) | 661.1 (26.02) / 663.0 (25.09) | 928.2 (17n) | 914.9 (18n) | 925.2 (16n) | 927.6 (16n) | **−28%** |
| 2 | Single-node NVLink busbw (GB/s) | 515.6 @ 8 GiB (all 11 nodes) | 690.8 @ 32 GiB | — | — | — | **~−25%** |
| 3 | Qwen 72B TFLOP/s/GPU (median) | 607.6 / 608.5 → **608.1** | 723.0 | 700.3 | 732.2 | 718.9 | −13% / −18% |
| 4 | GPT-OSS 120B TFLOP/s/GPU (median) | 314.8 / 315.0 → **314.9** | 408.5 | 399.0 | 383.5 | 382.1 | −18% / −23% |

Medians = last 30 steady-state steps, averaged across the two runs. Training ran at 8 nodes / 32 GPUs (vs 16 / 64 on prior racks) — per-GPU numbers are comparable to first order but not identical; see caveats.

## Config deltas vs the R01/R07 suite

| | Prior racks | 661303b8 | Why |
|---|---|---|---|
| Networking | Multus IPoIB NADs + `rdma/rdma_shared_device_{a..d}` + `NCCL_IB_HCA` | removed | Cluster has no Multus / RDMA shared-device plugin. Single-rack traffic is all MNNVL/NVLS anyway (IB not on the critical path). |
| Reservation | `together.ai/benchmark=johnson` taint | `reserved-for=infra-perf` toleration + nodeSelector, namespace `infra-perf` | Cluster reservation policy |
| NCCL | 16–18 nodes | **12 nodes / 48 GPUs** (`nvlsRanks 48`) | Only 12 nodes present |
| Qwen 72B | 16n, TP8 PP4 DP2, GBS 512 | **8n, TP8 PP4 DP1, GBS 256** | 48 GPUs doesn't factor TP8×PP4×DP; GBS halved to keep **GA=128** per replica (same pipeline-bubble ratio) |
| GPT-OSS 120B | 16n, EP64 DP64, GBS 1280 | **8n, EP32 DP32, GBS 640** | EP must divide 128 experts; GBS halved to keep **GA=5** |
| HF token | `mpiercy-hf-token` secret | `infra-perf/hf-token` secret | |

YAMLs: `tests/gb200/GB200/{nccl-12n-c66,nccl-12n-c66-2509,qwen-72b-c66,gpt-oss-120b-c66}.yaml`, generated from the R01 YAMLs by `tests/gb200/GB200/c66-tools/gen_yamls.py`. Campaign driver: `c66-tools/run_campaign.sh`.

## Per-Run Detail

### NCCL all_reduce_perf (12 nodes, 48 GPUs)

| Run | Image | NCCL | Log | busbw @ 32 GiB | algbw @ 32 GiB | Sweep avg busbw |
|---|---|---|---|--:|--:|--:|
| 1 | `pytorch:26.02-py3` | 2.29.2+cuda13.1 | `nccl-c66-12n.log` | 661.05 | 337.6 | 562.2 |
| 2 | `pytorch:25.09-py3` | 2.27.7+cuda13.0 | `nccl-c66-12n-2509.log` | 662.96 | 338.5 | 562.9 |

Single NVLS domain (`nvlsRanks 48`, MNNVL `state 3 healthMask 0xaa` on all GPUs, 24 NVLS channels), 0 out-of-bounds. Algorithm bandwidth ~338 GB/s vs ~465–470 on healthy racks. Two different CUDA/NCCL stacks agree within 0.3% → not a software-stack issue.

### Per-node single-node NVLink sweep (4 GPUs each, 8 GiB all_reduce)

Run concurrently inside the NCCL worker pods (11 of 12 nodes; `-17` was the already-exited rank-0 pod).

| Nodes | algbw (GB/s) | busbw (GB/s) |
|---|--:|--:|
| 04, 05, 08, 09, 11, 12, 13, 14, 15, 16, 18 | 343.7 ± 0.1 | **515.6 ± 0.2** |

Every node is identical — no outlier to exclude. For reference, R07's degraded node r07-06 measured 513 (vs healthy 690); here *every* node sits at that degraded level, and the 12-node algbw (338) matches the single-node algbw (344), i.e. the per-GPU NVLink rate itself is capped.

Host state on all nodes: driver 580.196.07, VBIOS 97.00.B9.00.98, 18/18 NVLinks @ 53.125 GB/s, fabric `Completed / Success / Healthy / Bandwidth: Full`, no SW/HW thermal slowdown, power limit 1200 W.

### Qwen 72B pretraining (8 nodes, 32 GPUs; TP8 PP4 DP1, MBS 2, GBS 256, BF16)

| Run | Log | Steps | Median TFLOP/s/GPU (last 30) | Last-30 range | Step time |
|---|---|--:|--:|--:|--:|
| 1 | `qwen-c66-run1.log` | 96 | 607.6 | 582.6–609.4 | 24.0 s |
| 2 | `qwen-c66-run2.log` | 95 | 608.5 | 582.2–610.9 | 24.0 s |

Mean **608.1**, run-to-run 0.15%. Run 1 nodes: 05, 08, 11, 12, 15, 16, 17, 18. TP=8 spans two trays, so every TP collective crosses the (slow) NVLink fabric.

### GPT-OSS 120B pretraining (8 nodes, 32 GPUs; EP32 DP32, MBS 4, GBS 640, BF16)

| Run | Log | Steps | Median TFLOP/s/GPU (last 30) | Last-30 range | Step time |
|---|---|--:|--:|--:|--:|
| 1 | `gpt-oss-c66-run1.log` | 95 | 314.8 | 272.6–317.4 | 7.07 s |
| 2 | `gpt-oss-c66-run2.log` | 95 | 315.0 | 274.6–318.5 | 7.06 s |

Mean **314.9**, run-to-run 0.06%. Occasional single-step dips to ~273 (8.1 s; GC interval) don't move the median. EP all-to-all over 32 GPUs is NVLink-bound, consistent with the larger relative hit than Qwen.

## Caveats

- **Training scale differs** from prior racks (32 vs 64 GPUs, EP32 vs EP64). GA is held constant so the pipeline / accumulation structure matches, but EP32 places 4 experts per GPU (vs 2) and changes all-to-all fan-out. Part of the GPT-OSS gap may be config, not fabric; the NCCL deficit alone is unambiguous.
- Single-node sweep used 8 GiB (vs 32 GiB for the R07 690.8 reference); at 8 GiB a healthy node is a few % below its 32 GiB peak — not enough to explain −25%.

## Actions / Follow-ups

- **Escalate to platform / NVIDIA:** cluster-wide ~28% NVLink all-reduce deficit with fabric reporting Healthy/Full. Ask for: NVSwitch tray FW + fabric-manager / NMX config check, driver 580.196.07 vs the driver on cluster `3209e979`, and whether the partial-rack (12/18 trays) NVSwitch partitioning is expected to cap per-GPU NVLink bandwidth.
- **Re-run** this suite after any fix (`c66-tools/run_campaign.sh`; NCCL first — target ≥ ~915 GB/s at full rack, ≥ ~690 single-node).
- **Reservation**: still in place. Release via the rollback in `c66-tools/reserve/apply.sh` — but note its last step deletes namespace `infra-perf` (and the `hf-token` / `nccl-ssh-keys` secrets).

Run artifacts (uncommitted, in `~/`): `nccl-c66-12n.log`, `nccl-c66-12n-2509.log`, `qwen-c66-run{1,2}.log`, `gpt-oss-c66-run{1,2}.log`, `gb200-661303b8/campaign.log`.
