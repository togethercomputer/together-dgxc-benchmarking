# Rack 07 Re-Readiness Report — 2026-07-13

GB200 NVL72 cluster `3209e979`. Re-benchmark of Rack 07 (`3209e979-r07-{01..18}`) following node maintenance since the 2026-05-22 readiness pass. Two nodes changed state since then:

- **r07-06** — previously flagged with ~26% degraded single-node NVLink (513 vs ~690 GB/s). Reimaged (~2026-06-30, kernel bumped to `6.17.0-1021-nvidia-64k`); re-verified healthy today.
- **r07-15** — now `NotReady` (down ~6 days): `kernel-soft-lockup` + `filesystem-readonly` taints, clique label dropped. Genuine host fault, excluded from this pass and flagged to platform.

Platform (Ankit Goyal) confirmed the rack is released for benchmarking **barring r07-15**. Rack reserved for the run with the `together.ai/benchmark=johnson:NoSchedule` taint (+ toleration) on the 17 healthy nodes, then uncordoned (Cody-endorsed pattern; taint keeps lightricks automation off).

## Summary

**Verdict: READY (16-node training set + full 17-node fabric).** NCCL 17-node fabric is healthy across all 17 usable nodes, r07-06's NVLink degradation is resolved, GPT-OSS 120B beats baseline, and Qwen 72B is reproducible and fabric-clean (~3% under the prior R07 number on a different node set, consistent with the known TP-heavy Qwen softness, not a hardware fault).

Workloads: r07-06 single-node NVLink verify ×1, NCCL 17n ×1, Qwen 72B ×2, GPT-OSS 120B ×2.

## Node set

- **NCCL 17n:** all healthy nodes = r07-{01..14,16,17,18} (excl. r07-15 only).
- **Training 16n (Qwen, GPT-OSS):** excl. r07-15 (down) + r07-17 (spare, to fit 64 GPUs). **Includes the reimaged r07-06 and the previously-cleared r07-08.**

## Scoreboard

| # | Workload | R07 (2026-07-13) | R07 (2026-05-22) | R08 (Apr) | R10 (Apr) | R01 (Jul) | Note |
|--:|---|--:|--:|--:|--:|--:|---|
| 1 | NCCL 17n bus BW @ 32 GiB (GB/s) | **928.2** | 929.7 (17n) | 925.2 (16n) | 927.6 (16n) | 914.9 (18n) | −0.2% vs prior 17n |
| 2 | Qwen 72B TFLOPS/GPU (median) | **723.0** | 746 | 732.2 | 718.9 | 700.3 | different node set |
| 3 | GPT-OSS 120B TFLOPS/GPU (median) | **408.5** | 397 | 383.5 | 382.1 | 399.0 | above baseline |

NCCL = single 17-node run. Qwen/GPT-OSS medians = last-30 steady-state steps, averaged across the two runs.

## Per-Run Detail

### r07-06 single-node NVLink verification (4 GPUs, NVLink only)

| Metric | 2026-05-22 (degraded) | 2026-07-13 (today) | Healthy baseline |
|---|--:|--:|--:|
| Peak busbw @ 32 GiB | 513 | **690.8** | ~690 |
| Sweep avg busbw | 482 | **655.7** | ~655 |

**r07-06 NVLink degradation is RESOLVED** — the reimage fixed it. Peak matches the ~690 GB/s healthy baseline exactly (and r07-08's healthy 655 sweep avg). Log: `~/nccl-r07-06-1n.log`.

### NCCL all_reduce_perf (17 nodes, 68 GPUs)

| Log | Peak BW @ 32 GiB (busbw, GB/s) | Avg bus BW across sweep |
|---|--:|--:|
| `nccl-r07-17n.log` | 928.2 | 758.5 |

Single 17-node NVLS domain (`nvlsRanks 68`), all 17 healthy nodes participating incl. r07-06 and r07-08. Within 0.2% of the prior 17n reading (929.7); no stragglers, no out-of-bounds, no fabric retries. Confirms the full usable rack is healthy at the fabric layer.

### Qwen 72B pretraining (16 nodes, 64 GPUs; TP8 PP4 DP2 GBS512)

| Run | Log | Steps | Median TFLOPS/GPU (last 30) | Step time |
|---|---|--:|--:|--:|
| 1 | `qwen-r07-run1.log` | 102 | 718.6 | ~20.1s |
| 2 | `qwen-r07-run2.log` | 100 | 727.5 | ~20.1s |

Mean **723.0 TFLOPS/GPU**, reproducible within ~1.2%. Steady-state p50 ≈ 720 over 93 steps (range 683–736), no severe checkpoint dips. ~3% under the prior R07 746, but that number used a different node set (excl. r07-06/08, incl. r07-15/17). Above R10 (719) and R01 (700). NCCL healthy (928) rules out fabric; matches the per-rack TP-heavy Qwen softness first seen on R01 — not a hardware fault, not blocking.

### GPT-OSS 120B pretraining (16 nodes, 64 GPUs; EP64 MBS4 GBS1280)

| Run | Log | Steps | Median TFLOPS/GPU (last 30) | Step time |
|---|---|--:|--:|--:|
| 1 | `gpt-oss-r07-run1.log` | 50 | 409.2 | ~5.55s |
| 2 | `gpt-oss-r07-run2.log` | 50 | 407.9 | ~5.55s |

Mean **408.5 TFLOPS/GPU**, reproducible within ~0.3% — above the prior R07 baseline (397), R01 (399), and the v2 baseline (405). Confirms the reimaged r07-06 and r07-08 perform at/above spec in the MoE workload.

## Actions / Follow-ups

- **r07-15**: down (kernel soft-lockup → root fs remounted RO). Flagged to platform (Ankit); awaiting fix. Not part of the usable rack.
- **r07-06**: reimage resolved the NVLink fault — cleared for production.
- **Qwen softness**: ~3% per-rack/config softness (fabric-clean); recommend platform compare r07 vs r08 BIOS/FW/power-cap if revisited. Same pattern as R01.
- **Rack release**: remove reservation taint + re-cordon after sign-off — `kubectl taint nodes -l nvidia.com/gpu.clique=36d8bca1-2323-4a1a-bcb9-cf673591464d.32766 together.ai/benchmark:NoSchedule-` then re-cordon.

Run artifacts (uncommitted, in `~/`): `nccl-r07-06-1n.log`, `nccl-r07-17n.log`, `qwen-r07-run{1,2}.log`, `gpt-oss-r07-run{1,2}.log`.
