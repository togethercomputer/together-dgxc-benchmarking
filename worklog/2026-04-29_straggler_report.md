# Straggler Report — use3a-ss — 2026-04-29

Ran `find_stragglers.py --with-ib-sweep` 3× today (2 valid NCCL runs, 1 invalid due to job contention from parallel submission). 60 idle nodes tested (down from 66 on 04-28 — consistent with draining yesterday's bad nodes). All runs flagged the same 8 bad nodes. IB layer clean across all runs.

Full per-phase report committed to: `~/together-nccl-tests/baselines/use3a-ss/straggler-report-20260429.md`

Run timestamps:
- `20260429_210316` — primary run (with IB sweep)
- `20260429_213056` — confirmation run #1 (with IB sweep)
- `20260429_213057` — confirmation run #2 — **INVALID**: all NCCL Phase 1 jobs hit `no_output` timeout due to node contention with run #1 (both submitted simultaneously). IB sweep completed fine. Future parallel confirmation runs must be sequential.

## TL;DR — Slurm exclude string

```
use3a-ss-b200-gpu-[162,164,195,196,214,215,227,228]
```

## Confirmed bad nodes

| Node | Ring Δ% | Ring z | Sharp Δ% | Sharp z | Notes |
|------|---------|--------|----------|---------|-------|
| **gpu-162** | −5.4 to −5.8% | −10.7 to −11.6 | −1.3 to −1.5% | −2.6 to −2.9 | Sharp topology-sensitive in 1/2 retests; ring confirmed_bad in both |
| **gpu-164** | −6.1 to −6.4% | −12.2 to −12.7 | −1.5% | −2.9 to −3.1 | Confirmed both algos both runs |
| **gpu-195** | −5.1 to −5.4% | −10.3 to −10.8 | −1.7% | −3.3 | Confirmed both algos both runs |
| **gpu-196** | −5.1 to −5.5% | −10.2 to −11.1 | −1.4 to −1.5% | −2.8 to −2.9 | Confirmed both algos both runs |
| **gpu-214** | −4.2 to −4.6% | −8.4 to −9.1  | −1.5% | −3.0 to −3.1 | Ring topology-sensitive in 1/2 retests; sharp confirmed_bad in both |
| **gpu-215** | −5.7 to −6.2% | −11.5 to −12.4 | −1.5 to −1.6% | −2.9 to −3.1 | Confirmed both algos both runs |
| **gpu-227** | −5.1 to −5.2% | −10.1 to −10.5 | −1.3 to −1.4% | −2.6 to −2.9 | Confirmed both algos both runs. Also bad on 04-28. |
| **gpu-228** | −5.4 to −5.6% | −10.9 to −11.1 | −1.6 to −1.7% | −3.1 to −3.3 | Confirmed both algos both runs |

## Cross-validation against 04-28 findings

| | 04-28 bad nodes | 04-29 bad nodes |
|---|---|---|
| All flagged | gpu-141, 161, 162, 184, 216, 217, 226, 227 | gpu-162, 164, 195, 196, 214, 215, 227, 228 |
| **Overlap** | gpu-162, gpu-227 (persistent bad nodes) | |
| **Cleared since 04-28** | gpu-141, 161, 184, 216, 217, 226 — no longer in idle pool (likely drained) | |
| **New today** | gpu-164, 195, 196, 214, 215, 228 — not flagged on 04-28 | |

6 of yesterday's bad nodes are no longer in the idle pool — consistent with drains acting on the 04-28 report (66→60 idle nodes). 6 new bad nodes surfaced today, suggesting ongoing cluster degradation or that these nodes were previously occupied and untested.

## Phase 4 — Inter-link (cross-pair)

No inter-link suspects in either valid run. `grp07+grp09` consistently slightly fast (+3.8–4.0%, z~+2.4) — within noise, not flagged. Fabric/spine appears healthy.

## Phase 6 — IB Sweep

All 8 HCAs (`mlx5_0/1/4/5/6/11/14/15`) healthy on all 60 nodes across all runs. Medians stable at 44.7–45.2 GB/s. Degradation is NCCL-layer (NVLink or intra-node interconnect), not IB fabric.

## Recommended next steps

1. **Drain** all 8 nodes in the exclude list above.
2. **Investigate gpu-162 and gpu-214** topology-sensitivity — both flagged bad in all runs but one algo flips to topology-sensitive on retest. Worth an extra targeted sweep when these nodes are isolated.
3. **Root cause the 6 new nodes** (164, 195, 196, 214, 215, 228) — these were healthy or untested on 04-28. Either recently degraded or were occupied during yesterday's sweep.
4. **Run confirmation sequentially**, not in parallel — the cluster only has ~60 idle nodes which get fully consumed by one run.
