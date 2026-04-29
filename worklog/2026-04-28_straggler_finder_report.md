# Straggler Finder — Tool Build & Cluster Report — 2026-04-28

Built `find_stragglers.py` from scratch today and ran it 7 times against the use3a-ss B200 cluster (66 idle nodes). Tool location: `~/together-nccl-tests/stragglers/find_stragglers.py`. This worklog summarizes the bad-node findings; for tool design see the auto-memory entry `project_find_stragglers_tool.md`.

## TL;DR — Slurm exclude string

```
# Conservative (exclude all confirmed + borderline):
use3a-ss-b200-gpu-[141,161,162,184,216,217,226,227]

# Strict (only confirmed):
use3a-ss-b200-gpu-[141,161,184,216,217,226,227]
```

## Confirmed bad nodes

| Node | Failure mode | Severity | Evidence | Recommended action |
|---|---|---|---|---|
| **gpu-141** | Missing `libcudart.so.12` | **Critical — cannot run CUDA jobs** | All ranks scheduled on 141 exit 127 with `error while loading shared libraries: libcudart.so.12`; pmix peers time out. Reproduces every run when 141 is in the idle pool. Distinct from the libmpi/hpcx-missing case (which is fixed by `stage-hpcx-to-vartmp.sh`). | **Drain immediately** + investigate node-local CUDA install. Likely needs reimaging. |
| **gpu-161** | 4n-slow on ring + SHARP | High | Drags any 4n group it joins. ring −6 to −7% (z ≈ −13), SHARP −1.4 to −1.5% (z ≈ −3). 7/7 runs. | Drain or exclude. |
| **gpu-184** | SHARP/CollNet path degraded | High | Ring is **completely normal** (332 GB/s); SHARP collapses to 440 GB/s (−14.4%, z ≈ −29). 7/7 runs, near-identical numbers each time. | Drain. If unable, exclude from any workload that uses CollNet/SHARP (most training doesn't; inter-node collective benchmarks do). |
| **gpu-216** | 4n-slow on ring + SHARP | High | ring −5 to −6%, SHARP −1.5%. 7/7 runs. | Drain or exclude. |
| **gpu-217** | 4n-slow on ring + SHARP | High | ring −5%, SHARP −1.7%. 7/7 runs. | Drain or exclude. |
| **gpu-226** | 4n-slow on ring + SHARP | High | ring −5 to −6%, SHARP −1.4%. 7/7 runs. | Drain or exclude. |
| **gpu-227** | 4n-slow on ring (confirmed); SHARP topology-sensitive | High–medium | ring −5% (z ≈ −10) consistently; SHARP retest sometimes lands at z=-1.9 (above the −2 bad threshold) depending on which healthy peers it's paired with. | Drain or exclude. |

## Borderline / monitoring

| Node | Behavior | Notes |
|---|---|---|
| **gpu-162** | Tested `candidate_bad` in 4 runs, `candidate_healthy` in 3 runs (out of 7) | Likely topology-sensitive — its measured bandwidth depends on which healthy peers it's paired with. Same node was in the 04-26 manual 4n-slow list. **Conservative**: include in exclude list; **lenient**: monitor across multiple sweeps. |

## Inferred root causes (not confirmed by hardware diagnostics)

- **gpu-141**: node-local CUDA runtime install is broken. Likely needs reimaging.
- **gpu-184**: SHARP/CollNet trees route through one or more degraded NIC ports specific to this node. Ring uses a different IB path and is fine. Check SHARP tree-assignment logs and HCA error counters on 184.
- **gpu-161, 216, 217, 226, 227**: a "4n-slow" cluster — these nodes drag *any* 4-node group by 5-7% on ring. The cluster-wide cross-pair test in Round 3 was clean (8n median 306.8 ± 0.5, no inter-link suspects), so the issue is *not* a generic spine degradation; it's specific to 4n groups that include these nodes. Likely cause: each has one or two underperforming HCA ports that handicap the small ring topology of 4n but get masked at larger scales.
- **gpu-162**: marginal version of the same issue — bandwidth path through it is sometimes slow, sometimes not. Suggests intermittent rather than permanent degradation.

## Tool measurements (cluster baseline as of 2026-04-28)

Saved to `~/together-nccl-tests/baselines/use3a-ss/healthy_4n.json`:

| Metric | Value |
|---|---|
| 4n ring all_reduce healthy median | 332.7 GB/s, stddev 0.24 |
| 4n CollNet SHARP all_reduce healthy median | 514.5 GB/s, stddev 1.20 |
| 8n ring all_reduce healthy median (cross-pair) | 306.8 GB/s, stddev 0.5–1.0 |
| Healthy 4n groups in latest run | 13 of 16 (excluding grp01 bootstrap-fail and grp04/09/15 perf-severe) |

## Cross-validation against historical findings

The 04-26 per-group manual sweep flagged 12 4n-slow nodes: 160, 161, 162, 164, 186, 188-189, 195, 217, 226-227, 229.

| | 04-26 manual | 2026-04-28 tool |
|---|---|---|
| Identified as bad | 160, 161, 162, 164, 186, 188-189, 195, 217, 226-227, 229 | 161, 162, 184, 216, 217, 226-227 |
| **Overlap** | 161, 162, 217, 226-227 (5 nodes) | |
| **Cleared since 04-26** | 160, 164, 186, 188-189, 195, 229 — test healthy now | |
| **New findings 04-28** | gpu-184 (SHARP-only, not in 04-26 list); gpu-216 (not in 04-26 list); gpu-141 (CUDA runtime issue, distinct class) | |

Cluster has improved meaningfully on the 4n-slow front (7 nodes recovered) but two new issues surfaced (184 SHARP-only and 141 missing libcudart).

## Reproducing the report

```bash
# Single command — runs all 4 rounds (~12-15 min wall on 64-node cluster):
~/together-nccl-tests/stragglers/find_stragglers.py

# Output:
#   /mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/find-stragglers/<timestamp>/results.json
#   stdout: per-round tables + final node verdicts
```

Today's run timestamps for direct inspection:
- `20260428_192957` — first full run, found localized bad nodes
- `20260428_201915` — verification round (5 runs in)
- `20260428_211023` — final verification (this report's reference run)

## Recommended next steps

1. **Operational**: hand the exclude list to the cluster operator; drain gpu-141 (CUDA-broken) and review the others for hardware checks (HCA error counters, IB cable diagnostics).
2. **Monitoring**: schedule `find_stragglers.py` daily so cluster drift like the 04-22→04-27 405B regression gets caught early. The tool's run is cheap enough (~12 min, ≤32 GPU-allocations at peak) that a daily slot is reasonable.
3. **Open question on gpu-184**: SHARP signal is unusually clean (z=-29 every run). Worth a targeted investigation — likely points at a specific switch port or HCA. Check SHARP tree topology logs.
4. **Open question on gpu-162**: intermittent. Two sweeps separated by hours give different verdicts. Either run the tool 3 times consecutively to settle the question or wait for next cluster restart.
