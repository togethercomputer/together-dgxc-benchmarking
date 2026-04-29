# NCCL Per-Group Scaling Benchmark — 2026-04-26

**Cluster:** Together AI B200 (use3a-ss)
**Context:** Cluster is fully idle. Run a per-group all_reduce sweep that disjointly partitions every idle node into fixed-size groups and pins each job via `--nodelist`, so every node is exercised and any group-level slowdown localizes to its members. Then probe how the localized weaknesses scale up to production sizes.
**Suite:** `~/together-nccl-tests/benchmarks/B200/collective-scaling/submit_per4_groups.sh` (new today).

---

## 1. Methodology

- **Script:** `submit_per4_groups.sh` — pulls truly-idle nodes from `batch` partition, partitions deterministically into groups of `--group-size`, submits one all_reduce job per group with `--nodelist=<group>`. Same NCCL env block as `submit_all.sh` (HPC-X, NCCL_IB_HCA, bond0, CUDA_DEVICE_MAX_CONNECTIONS=32). Supports `--algo ring|nvls|tree|nvls_tree|collnet_sharp|auto`, `--exclude`, `--sizes "MIN MAX FACTOR"`, `--dry-run`.
- **Bug fixed during dev:** `sinfo -t idle` lumps drained nodes into the idle bucket → the script now filters on the exact STATE column (`awk '$2=="idle"'`) so the 4 drained nodes (141, 143-144, 228) are correctly skipped.
- **Watchdog (`/tmp/nccl_watchdog.sh`):** polls each output file for the `Avg bus bandwidth` line and `scancel`s as soon as it appears — the post-srun-hang pattern documented in `feedback_nccl_post_srun_hang.md`. Cuts wall-time from the 20 min time-limit to ~1 min per round.
- **Sizes:** 2G→16G factor 2, 5 warmup + 20 measured iters.
- **NCCL libs:** `/mnt/vast/dgxc-benchmarking-auto/nccl-libs` (NCCL 2.29.7+cuda12.9), HPC-X 2.18.

---

## 2. Per-4-node sweep (16 groups × 4 nodes = 64 nodes)

Pinned every idle node (64 total) into 16 disjoint 4-node groups. Identified 3 underperformers under both ring and SHARP — same 3 groups in both algos, so it's a real fabric signal, not a ring-only NCCL artifact.

| Group | Nodes | Ring | SHARP | Ring Δ | SHARP Δ |
|------:|---|---:|---:|:---|:---|
| 01 | 130, 145, 146, 148 | 331.9 | 513.8 | — | — |
| 02 | 149, 151, 152, 155 | 332.1 | 512.3 | — | — |
| 03 | 156, 157, 158, 159 | 332.2 | 513.2 | — | — |
| **04** | **160, 161, 162, 164** | **314.3** | **507.8** | **−5.3%** | **−1.4%** |
| 05 | 165, 166, 167, 168 | 331.8 | 515.4 | — | — |
| 06 | 169, 170, 171, 173 | 331.8 | 513.7 | — | — |
| 07 | 174, 175, 176, 177 | 331.8 | 513.7 | — | — |
| 08 | 178, 179, 180, 181 | 332.0 | 516.2 | — | — |
| 09 | 182, 183, 184, 185 | 331.7 | 514.0 | — | — |
| **10** | **186, 188, 189, 195** | **312.7** | **507.2** | **−5.8%** | **−1.5%** |
| 11 | 196, 197, 198, 200 | 331.3 | 514.3 | — | — |
| 12 | 201, 202, 203, 205 | 331.5 | 514.5 | — | — |
| 13 | 209, 210, 211, 212 | 332.1 | 516.8 | — | — |
| 14 | 213, 214, 215, 216 | 332.0 | 513.2 | — | — |
| **15** | **217, 226, 227, 229** | **319.2** | **506.3** | **−3.8%** | **−1.7%** |
| 16 | 230, 235, 238, 256 | 332.0 | 514.3 | — | — |

Healthy median: ring **332**, SHARP **514**. SHARP at 4n today is **+33% above** the 04-25 baseline (384.4) — picking up the same MB/26.02 stack uplift logged in `project_rebench_256_2026-04-22.md`.

---

## 3. Per-8-node sweep (excluding 12 nodes from groups 04 + 10 + 15)

Excluded `[160-162,164,186,188-189,195,217,226-227,229]`, partitioned remaining 52 nodes into 6 disjoint 8-tuples (4 leftover).

| Group | Nodes | Ring | SHARP |
|------:|---|---:|---:|
| 01 | 130, 145-149, 151-152, 155 | 320.5 | 521.4 |
| **02** | **156-159 + 165-168** | **308.1** | **512.5** |
| 03 | 169-171, 173, 174-177 | 320.4 | 521.4 |
| 04 | 178-185 | 320.5 | 521.4 |
| 05 | 196-198, 200-203, 205 | 320.2 | 520.8 |
| 06 | 209-216 | 320.6 | 522.1 |

**New finding:** group 02 (156-159 + 165-168) is slow, **even though both 4-node halves tested healthy individually** at 4n (grp03: 332.2, grp05: 331.8). The slowdown only manifests when the two quads are joined → indicates a degraded **inter-quad fabric link**, not a single bad node.

---

## 4. Per-16-node sweep (same exclusion list)

3 disjoint 16-tuples from 48 of the 52 non-excluded nodes (4 leftover: 230, 235, 238, 256).

| Group | Nodes | Ring | SHARP |
|------:|---|---:|---:|
| **01** | **130, 145-149, 151-152, 155, 156-159, 165-168** | **301.98** | **512.81** |
| 02 | 169-171, 173, 174-185 | 316.59 | 522.30 |
| 03 | 196-198, 200-203, 205, 209-216 | 316.37 | 522.47 |

Group 01 contains the suspect 156-168 cohort and again runs slow (ring −4.6%, SHARP −1.8%). Other 8 nodes in grp01 (130, 145-149, 151-152, 155) tested healthy in 8n grp01, so they're not the cause.

**Ring vs SHARP magnitude consistent across 4n/8n/16n:**
- ring delta is 2-3× the SHARP delta — the bad inter-quad path is on the ring path but largely off the SHARP CollNet path.

---

## 5. 32-node A/B test (does the suspect cohort drag at scale?)

| Set | Composition | Ring | SHARP |
|---|---|---:|---:|
| **A** (control) | 32 confirmed-healthy | 294.23 | 512.31 |
| **B** (test) | 24 healthy + 8 suspicious (156-159, 165-168) | 292.61 | 511.93 |
| **Δ** | B − A | **−0.55%** | **−0.07%** |

**B vs A is within noise on both algorithms.** The slowdown signal disappears at 32 nodes.

---

## 6. 48-node and 64-node

Single-job benchmarks at production scales.

| Scale | Composition | Ring | SHARP |
|---:|---|---:|---:|
| 48n | 40 healthy + 8 suspicious (12 pre-excluded dropped) | **272.0** | **512.1** |
| 64n | all 64 idle (every node, including 12 pre-excluded) | **271.5** | **511.4** |

Adding the 12 pre-excluded "4n-slow" nodes back into a 64-node group costs **−0.2%** ring / **−0.1%** SHARP vs the cleaner 48n. **Inside noise.**

---

## 7. Combined scaling table

| Scale | Composition | Ring | SHARP |
|---:|---|---:|---:|
| 4n | healthy quad | 332.0 | 514.0 |
| 8n | healthy 8-tuple median | 320.5 | 521.4 |
| 16n | healthy 16-tuple median | 316.5 | 522.4 |
| 32n A | 32 healthy | 294.2 | 512.3 |
| 32n B | 24 healthy + 8 suspicious | 292.6 | 511.9 |
| 48n | 40 healthy + 8 suspicious | 272.0 | 512.1 |
| 64n | all 64 idle (incl. 12 pre-excluded) | 271.5 | 511.4 |

---

## 8. Comparison vs prior baselines

| Test | Scale | Prior | Today | Δ |
|---|---:|---:|---:|:---|
| ring all_reduce | 32n | 293.0 (4/23) | 294.2 | +0.4% |
| ring all_reduce | 32n | 306.3 (4/9) | 294.2 | −4.0% |
| SHARP all_reduce | 16n | 379.6 (4/25 avg) | 522.4 | **+37.6%** |
| SHARP all_reduce | 64n | 363.7 (4/25 avg) | 511.4 | **+40.6%** |

**SHARP throughput jumped ~40% across all scales 4n–64n vs the 04-25 baseline.** Same magnitude as the MB/26.02 stack wins on 70B NVFP4 / Qwen3 235B in the 04-22 rebench. Worth flagging — likely a NCCL-libs or SHARP-plugin update on the cluster between 04-25 and today.

Ring is essentially unchanged vs 04-23.

---

## 9. Bottom line — actionable

1. **Cluster is healthy at production scale.** All 64 idle nodes can be used for ≥32-node training jobs without measurable collective slowdown.
2. **Localized fabric weaknesses identified, but only matter at small-N:**
   - **12 nodes 4n-slow** (160-162, 164, 186, 188-189, 195, 217, 226-227, 229) — hit a degraded path in 4-node ring. Wash out by 32n.
   - **8 nodes inter-quad-slow** (156-159, 165-168) — fast individually, slow as a single 8-node block; degraded inter-quad link. Wash out by 32n.
3. **For diagnostic / small-scale work (≤16n)**, prefer the 40 confirmed-healthy nodes:
   ```
   use3a-ss-b200-gpu-[130,145-146,148-149,151-152,155,169-171,173-185,196-198,200-203,205,209-216]
   ```
4. **SHARP is +40% faster than the 04-25 baseline** at every scale tested (4n through 64n). Likely a stack update — should rebaseline `~/reports/NCCL_Benchmark_Report_*.md` with this stack before the next regression check.

---

## 10. Reproduction

```bash
# Per-group sweep — replace --group-size and --algo as needed
~/together-nccl-tests/benchmarks/B200/collective-scaling/submit_per4_groups.sh \
    --group-size 4 --algo ring

# With exclusions and a fixed nodelist (e.g. for the 32n A/B)
~/together-nccl-tests/benchmarks/B200/collective-scaling/submit_per4_groups.sh \
    --group-size 32 --algo collnet_sharp \
    --nodes "use3a-ss-b200-gpu-[130,145-146,148-149,151-152,155,169-171,173-185,196-198,200-203,205]"

# Watchdog (auto-cancels post-srun hang on Avg bus bandwidth line)
/tmp/nccl_watchdog.sh <results_dir> <jobid> [<jobid> ...]
```

**Result dirs (today):**
- 4n:    `/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/per4-groups/20260426_110415_ring/`
- 4n:    `.../per4-groups/20260426_111148_collnet_sharp/`
- 8n:    `.../per8-groups/20260426_111711_ring/`
- 8n:    `.../per8-groups/20260426_111846_collnet_sharp/`
- 16n:   `.../per16-groups/20260426_112608_ring/`
- 16n:   `.../per16-groups/20260426_112748_collnet_sharp/`
- 32n A: `.../per32-groups/20260426_113606_ring/`
- 32n B: `.../per32-groups/20260426_113712_ring/`
- 32n A: `.../per32-groups/20260426_113837_collnet_sharp/`
- 32n B: `.../per32-groups/20260426_114022_collnet_sharp/`
- 48n:   `.../per48-groups/20260426_114437_ring/`
- 48n:   `.../per48-groups/20260426_114613_collnet_sharp/`
- 64n:   `.../per64-groups/20260426_114959_ring/`
- 64n:   `.../per64-groups/20260426_115135_collnet_sharp/`

**Job IDs:** 85152..85209 (all completed/cancelled-after-data, no failed jobs).
