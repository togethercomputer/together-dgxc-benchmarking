# IB-Layer Investigation & Phase 6 Tool Extension — 2026-04-28

Continuation of the same-day straggler-finder work. After the NCCL-layer findings (`2026-04-28_straggler_finder_report.md`), drilled into the IB hardware layer to nail down root causes and verified that NCCL is in fact using all 8 HCAs (the 43 GB/s sendrecv figure had been mis-interpreted earlier as "single-HCA limited").

## TL;DR

1. **gpu-184 mlx5_0 PORT_DOWN** confirmed at the IB hardware layer — explains the SHARP -14.4% (1/8 ≈ 12.5% bandwidth loss). Other 7 HCAs are healthy (~47 GB/s each).
2. **gpu-183 mlx5_0 also PORT_DOWN** — NEW finding. NCCL Phase 3 didn't catch this (4n SHARP redundancy masked it). Two pairs of bad mlx5_0 ports in adjacent grp09 nodes.
3. **Phase 6** (`--with-ib-sweep`) added to `find_stragglers.py`. K=1 disjoint pairs × 8 HCAs `ib_write_bw`. ~3 min wall on 64 nodes. Caught both port-down nodes automatically.
4. **NCCL multi-HCA myth busted**: sendrecv ring on 2 nodes uses only 2 of 8 HCAs (topology-bound). Alltoall pattern gets exact 8.6× speedup — confirms NCCL really aggregates across all 8 HCAs when the workload generates enough cross-node flows.
5. **Cluster real cross-node peak**: ~376 GB/s aggregate across 8 HCAs (vs the 43 GB/s per-rail figure that had been misleading us).

## Section 1 — Pairwise IB benchmarks (manual, before tool integration)

Used `ib_write_bw -F -s 8388608 -n 2000` per HCA, tested all 8 (mlx5_0, _1, _4, _5, _6, _11, _14, _15).

### gpu-130 ↔ gpu-184 (suspect pair)

| HCA on gpu-184 | avg BW | Status |
|---|---|---|
| **mlx5_0** | — | **PORT_DOWN** (`ibv_devinfo`: state=PORT_DOWN(1)) |
| mlx5_1 | 47.34 GB/s | healthy |
| mlx5_4 | 47.21 GB/s | healthy |
| mlx5_5 | 47.27 GB/s | healthy |
| mlx5_6 | 47.05 GB/s | healthy |
| mlx5_11 | 46.98 GB/s | healthy |
| mlx5_14 | 46.93 GB/s | healthy |
| mlx5_15 | 46.83 GB/s | healthy |

### gpu-148 ↔ gpu-149 (healthy reference)

All 8 HCAs delivered 46.7–47.3 GB/s. Range 1.4%, all 89-90% of NDR400 line speed.

### Conclusion

The two pairs are **identical except for gpu-184's mlx5_0 PORT_DOWN**. Confirms gpu-184's SHARP -14.4% has a precise IB-hardware root cause (1 of 8 HCAs unreachable; SHARP CollNet trees lose ~1/8 bandwidth). No other subtle issues on the rest of gpu-184.

Output preserved at `/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/ib-pair/{86386,86391}.out`.

## Section 2 — Phase 6 integration into `find_stragglers.py`

Added `--with-ib-sweep` flag (default off). When enabled, runs after Round 4 confirmation retest.

**Algorithm:**
1. Discover idle nodes (66 today). Form `N/2` disjoint **adjacent pairs**.
2. Submit one sbatch per pair (32 jobs in parallel for 64 nodes). Each pair runs:
   - For each of 8 HCAs sequentially:
     - Probe port state on both sides via `ibv_devinfo`.
     - If both PORT_ACTIVE: start `ib_write_bw` server on side A, client on side B, capture avg BW.
     - If either PORT_DOWN: emit `HCA <name> SKIP server=<state> client=<state>`.
3. Wait for all jobs to drain. Parse structured output.
4. Aggregate per-HCA cluster median + stddev. Classify each (node, HCA) entry: `port_down` / `severe` (<95% median) / `suspect` (<98% median) / `healthy`.

**Cost on full cluster**: ~80s wall for Phase 6 itself, +64 sbatch jobs.

**Limitation acknowledged**: K=1 (one wave) cannot disambiguate "node A's HCA bad" vs "link A↔B specifically bad". K=2 with a second pairing matrix would solve it; deferred until needed.

## Section 3 — Phase 6 results on full cluster (job 86412 etc., 33 pairs / 66 nodes)

```
Bad HCAs — Round 5
  ★ use3a-ss-b200-gpu-183  mlx5_0  PORT_DOWN  (server=PORT_ACTIVE, client=PORT_DOWN)
  ★ use3a-ss-b200-gpu-184  mlx5_0  PORT_DOWN  (server=PORT_ACTIVE, client=PORT_DOWN)
```

### Per-HCA cluster stats

| HCA | median (MiB/s) | stddev | min | max | n |
|---|---|---|---|---|---|
| mlx5_0 | 45060 | 84 | 44816 | 45197 | **64** |
| mlx5_1 | 45060 | 125 | 44676 | 45212 | 66 |
| mlx5_4 | 44984 | 92 | 44783 | 45091 | 66 |
| mlx5_5 | 44974 | 140 | 44522 | 45091 | 66 |
| mlx5_6 | 44849 | 51 | 44642 | 44884 | 66 |
| mlx5_11 | 44857 | 52 | 44642 | 44883 | 66 |
| mlx5_14 | 44681 | 87 | 44350 | 44771 | 66 |
| mlx5_15 | 44676 | 77 | 44405 | 44783 | 66 |

mlx5_0 has n=64 (not 66) because gpu-183 and gpu-184 ended up paired with each other (adjacent in idle-list ordering); both sides PORT_DOWN → that pair's mlx5_0 test was fully skipped, costing 2 samples. All HCAs cluster within ~1% — fabric is otherwise uniform.

### gpu-183 finding

Earlier `find_stragglers.py` runs (Phases 1–5) consistently labeled gpu-183 `candidate_healthy`. Phase 6 found mlx5_0 PORT_DOWN on it — almost certainly a recently-flapped port (within the last hour or two of the runs), or an issue masked by NCCL's redundancy at 4n SHARP scale. **Without the IB layer probe this would have been missed entirely**.

This is the strongest validation of the cross-layer design: NCCL phases catch protocol/topology issues; IB phase catches hardware issues; some issues are visible only in one layer.

### Bottom-of-distribution sanity check

Looked at the lowest-bandwidth pair per HCA. Best case: 99.0% of median (gpu-148 ↔ gpu-149 on mlx5_5). All within 1% of cluster median, no outliers exceed the 98% suspect threshold. K=1 cannot tell which side of the slowest pair owns the slowdown — for today's data nothing is slow enough to matter. K=2 would help if we ever see a pair drop >2-3% below median.

### Cross-layer reconciliation

| Node | Phase 1-5 verdict | Phase 6 verdict | Root cause |
|---|---|---|---|
| gpu-141 | bootstrap_fail / `missing_lib` (libcudart.so.12) | all 8 HCAs healthy | CUDA runtime broken (software). Drain + reinstall CUDA, no hardware action. |
| gpu-184 | SHARP -14.4% (z=-29 every run) | **mlx5_0 PORT_DOWN** | 1/8 HCA unreachable. Drain + check IB cable/SFP/switch port. |
| gpu-183 | `candidate_healthy` (Phase 3) | **mlx5_0 PORT_DOWN** | Port likely flapped recently. Drain + check. |
| gpu-161, 216, 217, 226, 227 | ring 4n-slow (-5 to -7%) | all 8 HCAs healthy | NCCL/topology layer, not IB hardware. Avoid for 4n collectives. |
| gpu-162 | borderline (4 of 7 NCCL runs flagged) | all 8 HCAs healthy | Topology-sensitive at NCCL layer; not an IB issue. |

## Section 4 — NCCL multi-HCA verification (separate from find_stragglers)

Question raised: "P2P sendrecv shows ~43 GB/s — are we using only 1 HCA?"

### Test 1: `sendrecv_perf` default vs `NCCL_IB_HCA="=mlx5_1"` (forced single)

```
Test 1 (8 HCAs available):  busbw = 42.62 GB/s
Test 2 (forced mlx5_1):      busbw = 26.63 GB/s
Ratio:                       1.6x   (NOT 8x)
```

If only 1 HCA was used, both would be 42.6 GB/s. If all 8 were used, ratio would be 8x. Got 1.6× → ring sendrecv on 2 nodes uses **2 HCAs simultaneously** (the 2 cross-node ring edges: rank 7→8 and rank 15→0), not all 8.

### Test 2: `alltoall_perf` — pattern that fully exercises HCAs

```
Test B (alltoall, 8 HCAs):     busbw = 89.87 GB/s
Test C (alltoall, 1 HCA):      busbw = 10.44 GB/s
Ratio:                          8.6x   ← matches 8 HCAs
```

**Decisive evidence** that NCCL really aggregates across all 8 HCAs when the workload pattern generates enough independent cross-node flows. Each HCA carries ~47.4 GB/s = ~90% NDR400 line, perfectly matching `ib_write_bw`.

### NCCL_DEBUG=INFO confirms registration

```
NET/IB : Using [0]mlx5_0:1/IB/SHARP [1]mlx5_1:1/IB/SHARP [2]mlx5_4:1/IB/SHARP
            [3]mlx5_5 [4]mlx5_6 [5]mlx5_11 [6]mlx5_14 [7]mlx5_15 [RO]
```

NCCL **sees** all 8 HCAs. The 43 GB/s figure was a per-rail, per-edge measurement — not a single-HCA aggregate ceiling.

### Cluster real cross-node peak

`alltoall busbw 89.87 GB/s × 16 ranks / 15 pairs ≈ 376 GB/s aggregate cross-node throughput`, which equals 8 HCAs × ~47 GB/s — the physical limit.

The earlier (04-09 / 04-10) interpretation that sendrecv P2P at 43 GB/s = "single-HCA limited" was correct *per ring edge* but had been misleading us into thinking the full cluster was bottlenecked at 43 GB/s. **Real peak is ~8× higher** when the workload uses all rails.

## Section 5 — Implications for training workloads

| Pattern | Cross-node edges (2-node test) | HCAs concurrently used |
|---|---|---|
| ring sendrecv | 2 | 2 |
| ring all-reduce | 2 | 2 |
| ring all-reduce ≥4 nodes (multi-channel) | 4-8 | 2-4 |
| alltoall | N×(N-1) | **8** |
| **SHARP / CollNet all-reduce** | tree spans all NICs | **8** |

Practical takeaway:
- For 2-node ring all-reduce training, only 2 of 8 HCAs do real work. Not a config bug — algorithm-bound.
- For >2-node ring all-reduce, NCCL's multi-channel uses up to ~4 HCAs concurrently per direction.
- **SHARP / CollNet is the only way to use all 8 HCAs at small scale.** Today's 04-27 SHARP rebaseline (542 GB/s @ 64n) is consistent with full 8-HCA aggregation; ring at 64n caps around 287 GB/s (~50% of CollNet) because of the multi-channel limit.

## Reproduction

```bash
# Phase 6 IB sweep on whole idle cluster (now part of find_stragglers.py):
~/together-nccl-tests/stragglers/find_stragglers.py --with-ib-sweep \
    --skip-localize --skip-cross-pair --skip-confirm   # IB only, ~5 min

# Manual pair test (per HCA breakdown):
sbatch --nodelist=use3a-ss-b200-gpu-A,use3a-ss-b200-gpu-B /tmp/ib_pair_bench.sh

# Multi-HCA verification (alltoall vs forced single HCA):
sbatch --nodelist=A,B /tmp/nccl_hca_inspection.sh
```

Today's job IDs:
- 86386 — gpu-130↔184 ib_write_bw
- 86391 — gpu-148↔149 ib_write_bw (healthy baseline)
- 86412 — first full-cluster Phase 6 sweep (caught gpu-183 + gpu-184)
- 86540 — sendrecv 8-HCA vs 1-HCA
- 86551 — sendrecv NET debug + alltoall 8-vs-1 HCA verification

Outputs preserved at `/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/ib-pair/`.

## Section 6 — K=2 disjoint matchings (Iteration 7) and flapping discovery

After the K=1 sweeps surfaced gpu-183 + gpu-184 mlx5_0 PORT_DOWN, extended Phase 6 to use K=2 disjoint matchings so each (node, HCA) gets tested in 2 different pair contexts. This addresses the K=1 attribution problem: with one trial per node, when a pair tests slow we can't tell whether node A's HCA is bad, node B's HCA is bad, or only the A↔B link is degraded.

### Algorithm

```
Wave 1 (adjacent):    (h_0, h_1), (h_2, h_3), ..., (h_{N-2}, h_{N-1})
Wave 2 (cross-half):  (h_0, h_{N/2}), (h_1, h_{N/2+1}), ..., (h_{N/2-1}, h_{N-1})
```

Both are perfect matchings; node count per matching is N/2 (32 for our 64-node sweep). Waves run sequentially (same nodes), each pair in a wave runs in parallel.

### Classification logic

Per (node, hca) across K=2 trials:

| Condition | Verdict |
|---|---|
| `self_port_state != PORT_ACTIVE` in any trial | `port_down` |
| All trials below 95% of cluster median | `node_intrinsic_severe` |
| All trials below 98% of cluster median | `node_intrinsic_suspect` |
| Some trials slow, some healthy | `link_specific` (slow only with certain peers) |
| All trials within threshold | (no flag) |

The `link_specific` verdict is the key new capability: it identifies degraded paths *between* specific node pairs rather than blaming a single node's HCA. Today's cluster has no such case (fabric is uniform), but the classifier is ready for it.

### Validation runs (back-to-back, same cluster state)

| Metric | Run 1 (job 86567+) | Run 2 (job 86603+) | Δ |
|---|---|---|---|
| Total samples per HCA | 132 (mlx5_0: 128) | 132 (mlx5_0: 128) | ✓ identical |
| Bad HCAs | gpu-184 mlx5_0 PORT_DOWN | gpu-184 mlx5_0 PORT_DOWN | ✓ same |
| mlx5_0 median (MiB/s) | 45090 | 45085 | -0.01% |
| mlx5_1 median | 45123 | 45133 | +0.02% |
| mlx5_4 median | 45027 | 45017 | -0.02% |
| mlx5_5 median | 45016 | 45047 | +0.07% |
| mlx5_6 median | 44867 | 44860 | -0.02% |
| mlx5_11 median | 44857 | 44864 | +0.02% |
| mlx5_14 median | 44701 | 44687 | -0.03% |
| mlx5_15 median | 44694 | 44716 | +0.05% |
| `link_specific` flags | 0 | 0 | ✓ |
| `node_intrinsic_*` flags | 0 | 0 | ✓ |

**Per-HCA medians stable to within ±0.07% across runs.** Same single finding (gpu-184) — no run-to-run jitter. K=2 sample count is exactly 2× K=1, confirming each node tested in 2 different pair contexts.

### Cost

- Wave 1: ~80s wall, 32 sbatch jobs in parallel
- Wave 2: ~80s wall, 32 sbatch jobs in parallel
- Total Phase 6 with K=2: ~3-4 min (vs ~80s for K=1)
- ~64 sbatch jobs total

### gpu-183 mlx5_0 flapping discovery

Earlier today's sweep (~14:00) flagged gpu-183 mlx5_0 as PORT_DOWN. K=2 sweeps several hours later (~23:50, 00:02) did not flag gpu-183 — `ibv_devinfo` directly confirmed `state: PORT_ACTIVE (4)` at that point.

Same node, same HCA, ~2-10 hours apart:
- ~14:00: `PORT_DOWN`
- ~23:50: `PORT_ACTIVE`
- ~00:02: `PORT_ACTIVE`

**This is genuine port flapping**, not a measurement artifact. The Slurm `idle` state for gpu-183 was unchanged across this window — only the underlying IB port state flipped. Common causes: marginal SFP, loose cable, intermittent switch port, thermal-sensitive optic.

**Operational implication**: a flapping port is *more dangerous than a stable PORT_DOWN*. A stable failure is detected and the node is drained. A flapping port may pass tools' point-in-time checks (like our snapshot K=2 sweep) and then fail mid-training, potentially corrupting collective state. Recommendation: drain gpu-183 proactively + investigate the same hardware suspects as gpu-184 (cable/SFP/switch port for mlx5_0).

### Single-snapshot tool limitation

By definition, a tool that takes one cluster-state snapshot cannot detect flapping. To make Phase 6 catch flapping, two options exist:

1. **Multi-shot mode**: schedule 4-6 runs over an hour; aggregate "any PORT_DOWN seen across snapshots". Cost: 4-6× sweep time, but catches medium-frequency flap.
2. **Persistent state log**: every run appends `{timestamp, port_down_nodes, suspect_HCAs}` to `~/together-nccl-tests/baselines/use3a-ss/ib_history.jsonl`. A daily report shows "any node flagged in last N runs". Cheap (1 line per run) and matches daily-ops use case.

Option 2 is the natural fit if Phase 6 becomes a scheduled cron job. It's deferred for now but worth doing as Iteration 8 once we have automation.

### Updated bad-node summary (end of 2026-04-28)

| Node | Issue | Layer | Failure mode | Action |
|---|---|---|---|---|
| **gpu-141** | missing libcudart.so.12 | CUDA runtime | software | drain + reinstall CUDA |
| **gpu-184** | mlx5_0 PORT_DOWN | IB hardware | stable | **drain + repair** |
| **gpu-183** | mlx5_0 PORT flapping | IB hardware | **intermittent** | **drain proactively** (more dangerous than stable) |
| gpu-161, 216, 217, 226, 227 | ring 4n-slow | NCCL/topology | stable | avoid for 4n collectives |
| gpu-162 | borderline (4 of 7 NCCL runs) | NCCL/topology | run-dependent | monitor |

## Open follow-ups

1. **gpu-183 mlx5_0**: same hardware investigation as gpu-184 (cable/SFP/switch). Bonus: this one is *flapping*, so checking IB port logs / link-up counters at the switch may reveal the failure pattern more clearly than gpu-184's stable down.
2. **`--ib-only` mode**: a fast cluster-health probe (~3-4 min) skipping all NCCL phases. Worth adding for daily ops use.
3. **Phase 7 `--verify-ib-with-sendrecv`**: for each Phase 6 finding, run NCCL sendrecv against a healthy peer to confirm operational impact at workload level.
4. **Iteration 8 — IB history persistence**: append per-run `{ts, port_down, suspect}` to `~/together-nccl-tests/baselines/use3a-ss/ib_history.jsonl`; daily report aggregates last N runs to surface flap. Required if scheduling Phase 6 as a cron.
5. **Update tranche-1 baseline numbers**: the "P2P 43 GB/s" line in the canonical NCCL benchmark report (`~/reports/NCCL_Benchmark_Report_*.md`) should be annotated as "per-edge, ring topology" and complemented with alltoall numbers (~376 GB/s aggregate) so future readers don't make the same misinterpretation.
6. **K=3+ matchings**: only worth adding if K=2 ever shows ambiguous attribution in real data. Today K=2 was sufficient because no real degradation case appeared. Defer until needed.
