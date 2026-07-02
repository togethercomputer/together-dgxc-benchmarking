# Rack 01 Readiness Report — 2026-07-01

GB200 NVL72 cluster `3209e979`. Acceptance benchmark pass for Rack 01 (`3209e979-r01-{01..18}`). Workload set and methodology mirror the R07 / R08 / R10 passes. Rack 01 was already in production (46-day-old nodes, untainted) and was reserved for this delivery benchmark via a `together.ai/benchmark=johnson:NoSchedule` taint on all 18 nodes (plain cordon did not hold — lightricks scheduling automation un-cordons; the taint is the durable reservation, per Cody's guidance).

## Summary

**Verdict: READY, with one noted caveat.** All three readiness workloads pass, run ×2 for reproducibility. NCCL 18-node fabric and GPT-OSS 120B are on/above cluster norms. **Qwen 72B runs ~4–6% below the R07/R08 bests and ~2.6% below R10 — reproducibly (5 runs, all 696–711).** This is a stable per-rack characteristic on the Qwen config, not noise and not a fabric fault (NCCL is healthy at 915 GB/s). Recommend flagging to platform for BIOS/FW/clock investigation; it does not block readiness.

6 readiness jobs ran (NCCL ×2, Qwen ×2, GPT-OSS ×2), plus an earlier single-pass + two ad-hoc Qwen runs earlier the same day. Each workload was torn down and the rack drained before the next (single rack = 18 nodes, one job at a time).

## Scoreboard

| # | Workload | R01 (run1 / run2 → mean) | R07 | R08 | R10 | R01 vs R08/R10 |
|--:|---|--:|--:|--:|--:|--:|
| 1 | NCCL 18n peak busbw @ 32 GiB (GB/s) | 914.6 / 915.2 → **914.9** | 934 (16n) | 925 | 928 | −1.1% / −1.4% |
| 2 | Qwen 72B TFLOP/s/GPU (median) | 704.4 / 696.2 → **700.3** | 746 | 732 | 719 | −4.3% / −2.6% |
| 3 | GPT-OSS 120B TFLOP/s/GPU (median) | 399.4 / 398.6 → **399.0** | 397 | 383 | 382 | **+4.2% / +4.4%** |

Medians taken over the last 30 steady-state steps of each run (warmup + periodic checkpoint-step dips excluded), then averaged across the two runs. NCCL is the full-rack 18-node value; R07's 934 was a 16-node figure (its 18n crosses a clique boundary → ~710), so R01's 18n number is not directly comparable to R07's headline but confirms all 18 nodes form one healthy 72-GPU NVLS domain.

## Per-Run Detail

### NCCL all_reduce_perf (18 nodes, 72 GPUs)

| Run | Log | Peak busbw @ 32 GiB (GB/s) |
|---|---|--:|
| 1 | `nccl-r01-run1.log` | 914.63 |
| 2 | `nccl-r01-run2.log` | 915.18 |

Both flat, no stragglers or fabric retries. `nvlsRanks 72` confirms all 18 nodes participate in a single NVLink (NVLS) domain — the whole rack is one clique (`4efdce90-…-32766`). Reproducibility within 0.06%.

### Qwen 72B pretraining (16 nodes, 64 GPUs) — TP=8 PP=4 DP=2, GBS=512, BF16

| Run | Log | Steps | Median TFLOP/s/GPU (last 30) | Step time |
|---|---|--:|--:|--:|
| 1 | `qwen-r01-run1.log` | 90 | 704.4 | ~20.5s |
| 2 | `qwen-r01-run2.log` | 90 | 696.2 | ~20.6s |

Earlier same-day data points (single pass + 2 ad-hoc): 709.65, ~704, ~711. **All five Qwen runs land in 696–711** — the softness is highly reproducible. Step time ~20.5s vs R07's ~19.5s (~5% slower), consistent with the throughput gap.

### GPT-OSS 120B pretraining (16 nodes, 64 GPUs) — TP=1 PP=1 EP=64 DP=64, MBS=4 GBS=1280, BF16

| Run | Log | Steps | Median TFLOP/s/GPU (last 30) | Step time |
|---|---|--:|--:|--:|
| 1 | `gpt-oss-r01-run1.log` | 90 | 399.4 | ~5.5s |
| 2 | `gpt-oss-r01-run2.log` | 90 | 398.6 | ~5.5s |

Above the R07 baseline (397) and clearly above R08/R10 (383/382). Reproducibility within ~0.2%.

## Qwen Softness — Analysis

The gap is **isolated to the Qwen 72B config** and is not explained by the fabric:
- NCCL 18n is healthy (915 GB/s, single 72-GPU NVLS domain) — rules out NVLink/IB degradation.
- GPT-OSS 120B (same nodes, same DRA/ComputeDomain path, EP=64 alltoall-heavy) is *above* baseline — rules out a rack-wide compute/clock deficit affecting all collectives.
- The Qwen config is TP=8-heavy (dense 72B, TP-comms bound); the ~5% step-time inflation points at a per-rack TP-collective or GPU-clock/BIOS/FW delta rather than a broken component.

Recommend platform compare BIOS/firmware/clock settings on r01 vs r07/r08/r10, and check for any power-cap or clock-throttle difference. **Not a readiness blocker** — the rack trains both models correctly with no rank failures.

## Infrastructure Notes

- **Reservation:** all 18 r01 nodes tainted `together.ai/benchmark=johnson:NoSchedule` (Cody-endorsed approach; cordon alone is reverted by lightricks automation). All three YAMLs carry the matching toleration. **Taint still active at time of writing — remove + re-cordon to release the rack.**
- Clique: `4efdce90-b191-4acc-9a4e-9af0e63d1c46.32766` (all 18 nodes). No acceptance taint (rack in production).
- Secrets used: `nccl-ssh-keys`, `mpiercy-hf-token` (GPT-OSS gated access).
- JobSet names kept short (`gpt120b-r01`, `qwen72b-r01`, `nccl-18n-r01`) — well under the 64-char pod-FQDN limit that bit the R07 run.

## Artifacts

Run logs (not committed; live in `~/`):

```
nccl-r01-run1.log      nccl-r01-run2.log         # NCCL 18n x2
qwen-r01-run1.log      qwen-r01-run2.log         # Qwen 72B 16n x2
gpt-oss-r01-run1.log   gpt-oss-r01-run2.log      # GPT-OSS 120B 16n x2
rack01-full-campaign.log                          # orchestrator campaign log
# earlier same-day: nccl-r01-18n.log, qwen-r01.log, qwen-r01-v2.log, gpt-oss-r01.log
```

YAMLs: `tests/gb200/GB200/{nccl-18n,qwen-72b,gpt-oss-120b}-r01.yaml` (all carry the benchmark toleration).

## Remaining Work

1. **Report the Qwen ~5% softness** to platform with the analysis above (fabric-clean, GPT-OSS-above-baseline → per-rack TP/BIOS/FW/clock suspect).
2. **Release the rack** when the delivery is signed off: `kubectl taint nodes -l nvidia.com/gpu.clique=4efdce90-b191-4acc-9a4e-9af0e63d1c46.32766 together.ai/benchmark:NoSchedule-` then re-cordon per Cody's intended state.
3. **Sign-off message** to the rack-acceptance channel: propose READY, note the Qwen caveat.

## Cross-Reference

NCCL fabric numbers also belong in `together-nccl-tests/baselines/<cluster>/` per repo convention; that sibling repo is not present on this host, so this report is the primary record (same caveat as R07). See `2026-05-22_rack07_readiness_report.md` for methodology.
