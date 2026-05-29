# Rack 07 investigation YAMLs

One-off manifests from the 2026-05-22 R07 readiness pass and follow-up node
investigation. These are **not** the canonical readiness configs — for ongoing
R07 readiness use the 16-node YAMLs in the parent directory
(`nccl-16n-r07.yaml`, `gpt-oss-120b-r07.yaml`, `qwen-72b-r07.yaml`), which match
the r08/r10 convention and produce throughput numbers comparable to the 16n
baselines.

See `worklog/2026-05-22_rack07_readiness_report.md` for the full write-up.

## Per-node investigation (single-node, NVLink-only)

Single-node 4-GPU `all_reduce`. Bypass the DRA/IMEX/IB path entirely
(no ComputeDomain, no RDMA shared device claims) to isolate intra-node NVLink
health. Healthy GB200 NVL72 baseline ≈ 690 GB/s busbw @ 32 GiB.

| File | Node | Result |
|---|---|---|
| `nccl-1n-r07-06.yaml` | r07-06 | 513 GB/s peak — ~26% low, confirmed NVLink/GPU regression |
| `nccl-1n-r07-08.yaml` | r07-08 | 691 GB/s peak — healthy (matches baseline) |

## 17-node structural validation (incl. r07-08, excl. r07-06)

Purpose is **structural validation that r07-08 works through the full readiness
pattern at every layer (CD-enrolled, IB rails, NCCL convergence)** — *not*
throughput comparison. 17 is prime, so parallelism had to be retuned away from
the 16n baselines; the resulting LLM TFLOPS are **not** apples-to-apples.

| File | Notes |
|---|---|
| `nccl-17n-r07.yaml` | 17n incl. r07-06 (early exploratory) |
| `nccl-17n-r07-no06.yaml` | 17n incl. r07-08, excl. r07-06 — 929.7 GB/s, within 0.5% of 16n baseline |
| `qwen-72b-17n-r07.yaml` | TP=2 PP=2 DP=17, MBS=1, full activation recompute (782 TFLOPS — not comparable) |
| `gpt-oss-120b-17n-r07.yaml` | EP=4, Megatron-FSDP (69.6 TFLOPS — not comparable; FSDP+MoE comm overhead) |

## 16-node including r07-08 (production-config validation)

| File | Notes |
|---|---|
| `gpt-oss-120b-16n-with08-r07.yaml` | EP=64 production config, 16n excl. {r07-06, r07-17} so r07-08 participates — **396.6 TFLOPS, matches the 397 baseline → r07-08 production-cleared**. JobSet name kept at `gpt-w08-r07` to stay under the 64-char pod-FQDN limit. |
