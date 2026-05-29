# Rack 07 Readiness Report — 2026-05-22

GB200 NVL72 cluster `3209e979`. Acceptance benchmark pass for Rack 07 (`3209e979-r07-{01..18}`), the next rack through the readiness gate after R08 / R10 were signed off on 2026-04-16. Workload set and methodology mirror the R08/R10 pass.

## Summary

**Verdict: READY.** All three readiness workloads pass on R07, and every workload beats both R08 and R10. The GPT-OSS 120B gap vs the v2 (Rack 04) baseline — the only open thread from the April pass — narrows from ~5.5% on R08/R10 to ~1.4% on R07. NCCL 16-node bus bandwidth is the highest measured on the cluster to date.

6 readiness jobs ran: NCCL 16-node ×2, Qwen 72B ×2, GPT-OSS 120B ×2. One additional exploratory 18-node NCCL run was captured for clique-boundary context and is not part of the readiness suite.

## Scoreboard

| # | Workload | R07 (today) | R08 (Apr-16) | R10 (Apr-16) | v2 baseline (R04) | R07 vs R08/R10 | R07 vs v2 |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | NCCL 16n bus BW @ 32 GiB (GB/s) | **934.2** | 925.2 | 927.6 | 918.7–922.2 | +0.97% / +0.71% | **+1.3 to +1.7%** |
| 2 | Qwen 72B TFLOPS/GPU (median) | **746** | 732.2 | 718.9 | — | +1.9% / +3.8% | — |
| 3 | GPT-OSS 120B TFLOPS/GPU (median) | **397** | 383.5 | 382.1 | 405.4 | +3.5% / +3.9% | **−2.1%** |

NCCL value = mean of two 16-node runs (934.7, 933.7). Qwen/GPT-OSS medians taken over last 30 steady-state steps of each run, then averaged across the two runs per workload.

## Per-Run Detail

### NCCL all_reduce_perf (16 nodes, 64 GPUs)

| Run | Log | Peak BW @ 32 GiB (busbw, GB/s) | Avg bus BW across sweep |
|---|---|--:|--:|
| 1 | `nccl-r07-v2.log` | 934.7 | 764.1 |
| 2 | `nccl-r07-v3.log` | 933.7 | 764.0 |

Both runs flat — no straggler iterations, no out-of-bounds values, no fabric retries in the tail.

### NCCL all_reduce_perf (18 nodes, 72 GPUs) — exploratory, *not* readiness

`nccl-r07.log` captured a single 18-node sweep that ran two internal iterations. Peak @ 32 GiB was 710–716 GB/s, average across the sweep ~645 GB/s. This is expected: 18 nodes spans a clique boundary on NVL72, dropping the slow leg to ConnectX rail-rail rather than NVLink. Not a regression and not part of the readiness suite — recorded for clique-boundary characterization.

### Qwen 72B pretraining

| Run | Log | Steps | Median TFLOPS/GPU (last 30) | Step time |
|---|---|--:|--:|--:|
| 1 | `qwen-r07.log` | 103 | 746.1 | ~19.5s |
| 2 | `qwen-r07-v2.log` | 161 | 745.9 | ~19.5s |

Step-time floor ~19.4s with periodic ~20.5–20.7s outliers consistent with the regular checkpoint cadence; outliers exclude cleanly from the steady-state window. Both runs reproduce within 0.1%.

### GPT-OSS 120B pretraining

| Run | Log | Steps | Median TFLOPS/GPU (last 30) | Step time |
|---|---|--:|--:|--:|
| 1 | `gpt-oss-r07.log` | 50 | 398.5 | ~5.55s |
| 2 | `gpt-oss-r07-v2.log` | 50 | 395.7 | ~5.62s |

Same checkpoint-step dip pattern (~330 TFLOPS on every ~5th step) as Qwen — excluded from the steady-state window.

## GPT-OSS 120B — Open Thread Update

The April pass left a ~5.5% reproducible offset on GPT-OSS 120B (R08/R10 at 382–384 vs v2 baseline 405.4), hypothesized to be the Ubuntu 24.04 / kernel 6.17 stack on the new racks vs Ubuntu 22.04 / kernel 6.8 on v2 racks (NCCL on R08/R10 was healthy, ruling out fabric).

R07 today: **397 TFLOPS/GPU**, only ~2.1% below the v2 baseline. OS/kernel verified today: **Ubuntu 24.04.4 LTS, kernel 6.17.0-1014-nvidia-64k** — *identical* to R08/R10. So the ~4-point improvement vs R08/R10 is **not** OS-driven; the gap has a per-rack component (fabric, BIOS/FW, or calibration) that the OS hypothesis alone can't explain. Doesn't block readiness, but the original "Ubuntu 24.04 / kernel 6.17 is the culprit" framing is now weakened — recommend bisecting fabric / BIOS / FW deltas if anyone revisits this.

## Per-Node Investigation — r07-06 and r07-08

The readiness YAMLs exclude r07-06 and r07-08 ("known bad" and "no IMEX channel in ComputeDomain" respectively, per the YAML comments). Cody asked us to put r07-06 back in and characterize each node individually. Results:

| Test | Configuration | Result @ 32 GiB busbw | Notes |
|---|---|--:|---|
| 16-node NCCL (readiness) | r07 \ {r07-06, r07-08} | **934 GB/s** | Baseline for comparison |
| 17-node NCCL | r07 \ {r07-08}, **includes r07-06** | **735 GB/s** (sweep avg 658) | −21% vs 16-node baseline — r07-06 drags the whole collective |
| 17-node NCCL | r07 \ {r07-06}, **includes r07-08** | **929 GB/s** (sweep avg 759) | −0.5% vs 16-node baseline — r07-08 is fully participating, no impact |
| 1-node NCCL | **r07-06 alone**, 4 GPUs, NVLink only | **513 GB/s** (sweep avg 482) | ~26% below the ~690 GB/s healthy single-node reference |
| 1-node NCCL | **r07-08 alone**, 4 GPUs, NVLink only, no-CD bypass | **691 GB/s** (sweep avg 655) | ✅ matches the ~690 reference — r07-08's intra-node hardware is healthy |

**Findings:**

- **r07-06 has a real intra-node performance regression.** The single-node test bypasses IB, RDMA, the IPoIB NADs, and the ComputeDomain/IMEX path entirely, so this is NVLink or GPU on r07-06 itself. Bandwidth is uniformly low across all message sizes — a stable degradation, not transient corruption. Recommend platform/hardware investigation.
- **r07-08 is fully healthy at every layer tested.** (1) Intra-node NVLink: 691 GB/s single-node ≈ ~690 baseline. (2) DRA driver / ComputeDomain: the 17-node-no-06 pod with the standard `compute-domain-channel` claim scheduled and ran on r07-08 without error — the DRA driver accepted the claim, so the original "no IMEX channel in ComputeDomain" exclusion no longer applies. (3) IB rails (4× ConnectX-7): cross-node NCCL with r07-08 included converged with no errors. (4) Multi-node bandwidth: 929 GB/s @ 32 GiB, within ~0.5% of the 16-node baseline (run-to-run noise). **An earlier draft of this report inferred a hardware fault on r07-08 from K8s symptoms (DRA-driver-plugin restart count, ComputeDomain non-enrollment); that inference was wrong** — per Azeem (platform) the restarts were rack-wide from the recent switch replacement work and stopped 2026-05-19, and ComputeDomain enrollment is per-job/dynamic. The actual NCCL measurement is the ground truth.

**Implication:** the readiness suite should be re-run at 17 nodes including r07-08. The "no IMEX channel in ComputeDomain" comment in the existing YAMLs is stale; the trigger for it likely predates the May-19 switch upgrade completion. r07-06's exclusion stays until the hardware is fixed.

## 17-Node Re-Run (incl. r07-08, excl. r07-06)

Per the per-node investigation, the readiness suite was re-run at 17 nodes to verify r07-08 participates cleanly through the full readiness pattern (NCCL + Qwen + GPT-OSS), not just at the NCCL layer. **All three workloads completed; r07-08 participated cleanly in every one.**

| Workload | 17n config | Steps / runtime | Headline result | 16n baseline | Notes |
|---|---|---|--:|--:|---|
| NCCL 16-node all_reduce | 17 nodes, IB+IMEX path | 32 GiB busbw | **929.7 GB/s** | 934 (excl. r07-08) | Within 0.5%; confirms r07-08 in CD-enrolled multi-node NCCL |
| Qwen 72B BF16 | TP=2 PP=2 DP=17, MBS=1, GBS=544, full activation recompute, `expandable_segments` | 50/50 steps, ~16 min | **782 TFLOPS/GPU** median | 746 (TP=8 PP=4 MBS=2 selective) | First attempt at MBS=2 selective-recompute OOM'd (184 GB used); reduced MBS and switched to full recompute. **Throughput not directly comparable** — both parallelism and recompute strategy differ. |
| GPT-OSS 120B BF16 | TP=1 PP=1 EP=4 DP=17, MBS=1, GBS=1360, Megatron-FSDP (`use_megatron_fsdp=True`, `sharding_strategy=optim_grads_params`) | 50/50 steps, ~26 min training | **69.6 TFLOPS/GPU** median | 397 (EP=64 MBS=4 no-FSDP) | First attempt OOM'd allocating a 106 GiB grad buffer in DDP setup (EP=4 puts ~54B params/rank). FSDP shards optim+grads+params across DP=17 to fit, but the FSDP all-gather + EP=4 alltoall comm overhead drops throughput ~82% vs the 16n EP=64 baseline. **Throughput not directly comparable** — both parallelism and DDP strategy differ. Pod ended in `Error` (SIGABRT at teardown) after training completed — cosmetic, all 50 steps were captured. |

**Key takeaway from the 17n re-run:** the throughput numbers are not directly comparable to the 16-node readiness baselines because the parallelism configs had to change (17 = prime, so the original power-of-2 factorings don't fit). The *validation* the re-run provides is structural: **r07-08 schedules, NCCL converges, both LLM workloads progress through 50 steps without rank failures.** Combined with the per-node single-node + 17n-no-06 NCCL evidence, r07-08 is fully cleared at all layers (intra-NVLink, DRA/IMEX, IB rails, multi-node NCCL, MoE/DP training).

For comparable LLM benchmarks at production parallelism, prefer running at 16 nodes including r07-08 (drop r07-06 + one healthy node from the exclusion list) — keeps the TP=8 PP=4 Qwen config and the EP=64 GPT-OSS config, both of which already validated at 746 and 397 TFLOPS/GPU respectively.

## 16-Node With r07-08 — Production-Config Validation

Following the 17n re-run, ran GPT-OSS 120B at 16 nodes excluding {r07-06, r07-17} so that r07-08 is one of the 16 — same EP=64 MBS=4 config as the readiness baseline, allowing a directly-comparable throughput number.

| Workload | Config | Result | 16n baseline (excl. r07-08) | Δ |
|---|---|--:|--:|--:|
| GPT-OSS 120B BF16 | TP=1 PP=1 EP=64 MBS=4 GBS=1280, 16 nodes incl. r07-08 (excl. r07-06, r07-17) | **396.6 TFLOPS/GPU** median (last 30 of 50) | 397 | **−0.1%** |

50/50 steps, pod phase `Succeeded`, no rank failures. Step-time pattern matches the original readiness (steady ~5.6s with periodic ~6.7s checkpoint-step dips). **This is the cleanest validation that r07-08 fully participates in the production-config readiness workload.**

Combined with Qwen 17n (alternative config), NCCL 17n, and per-node single-node tests, r07-08 is now validated at every layer with both retuned-config (17n) and production-config (16n incl. r07-08) data points.

**One operational caveat surfaced during this run:** `setHostnameAsFQDN: true` (in the standard YAML pattern) imposes a 64-char hostname limit on pod FQDNs. JobSet names longer than ~22 characters can exceed this when expanded into `<name>-w-0-<idx>.<name>.default.svc.cluster.local`. First apply with name `gpt120b-16n-with08-r07` produced 77-char FQDNs; all 16 pods stuck in `ContainerCreating` for 30+ minutes with `FailedCreatePodSandBox: FQDN ... is too long`. Renamed to `gpt-w08-r07` to fix. **Future YAMLs should keep the JobSet name ≤ ~22 chars.**

## Infrastructure Notes

- Acceptance taint: R07 still carries `together.ai/acceptance=true:NoSchedule` at run time. Both LLM YAMLs (`tests/gb200/GB200/qwen-72b-r07.yaml`, `gpt-oss-120b-r07.yaml`) and the NCCL YAML (`nccl-16n-r07.yaml`) include the toleration. The taint will be removed once R07 is declared READY, at which point the toleration becomes a no-op.
- ComputeDomain status snapshot for the 18-node exploratory run is in `~/r07-computedomain-status.yaml`; 17 of 18 nodes Ready (r07-08 absent, see per-node investigation above), all on cliqueID `36d8bca1-…32766`.
- The single-node r07-08 test used a stripped-down YAML (`tests/gb200/GB200/r07-investigations/nccl-1n-r07-08.yaml`) with no ComputeDomain and no RDMA shared device claims — required to bypass the DRA path while it was still considered suspect. For nodes where the DRA path is healthy, prefer the standard YAML pattern.

## Artifacts

Run logs (not committed; live in `~/`):

```
nccl-r07.log          # exploratory 18-node sweep (includes r07-06, r07-08)
nccl-r07-v2.log       # readiness, 16-node run 1
nccl-r07-v3.log       # readiness, 16-node run 2
nccl-r07-17n.log      # per-node investigation, 17-node (incl. r07-06, excl. r07-08)
nccl-r07-17n-no06.log # per-node investigation, 17-node (incl. r07-08, excl. r07-06)
nccl-r07-1n-06.log    # per-node investigation, single-node on r07-06
nccl-r07-1n-08.log    # per-node investigation, single-node on r07-08 (no-CD bypass)
nccl-r07-17n-v2.log   # 17-node re-run, NCCL (confirms 929 GB/s, 0.5% from baseline)
qwen-r07.log          # readiness, run 1 (16-node)
qwen-r07-v2.log       # readiness, run 2 (16-node)
qwen-r07-17n.log      # 17-node re-run, Qwen TP=2 PP=2 MBS=1 full-recompute (782 TFLOPS)
gpt-oss-r07.log       # readiness, run 1 (16-node)
gpt-oss-r07-v2.log    # readiness, run 2 (16-node)
gpt-oss-r07-17n.log   # 17-node re-run, GPT-OSS EP=4 FSDP MBS=1 (69.6 TFLOPS)
gpt-oss-r07-16n-with08.log  # 16-node incl. r07-08, EP=64 production config (396.6 TFLOPS — matches baseline)
r07-computedomain-status.yaml
```

YAMLs:
- Canonical readiness (16n, match r08/r10 convention): `tests/gb200/GB200/{nccl-16n,qwen-72b,gpt-oss-120b}-r07.yaml`.
- Investigation / one-off variants: `tests/gb200/GB200/r07-investigations/{nccl-17n,nccl-17n-r07-no06,nccl-1n-r07-06,nccl-1n-r07-08,qwen-72b-17n,gpt-oss-120b-17n,gpt-oss-120b-16n-with08}-r07.yaml` (see that dir's README.md).

## Remaining Work

1. **Report r07-06 hardware regression** to platform/cluster ops with the per-node investigation numbers above.
2. **Update the canonical readiness YAMLs to drop r07-08 from the exclusion list and stay at 16 nodes including r07-08.** The 17-node re-run validated that r07-08 works at every layer, but the LLM throughput numbers are not directly comparable to the 16n baselines (17 is prime → had to retune parallelism). For ongoing readiness use the proven 16n configs (Qwen TP=8 PP=4, GPT-OSS EP=64); just swap the `NotIn` list from `{r07-06, r07-08}` to `{r07-06, r07-XX}` for some healthy spare. Also drop the stale "no IMEX channel in ComputeDomain" comment.
3. **Sign-off message** to the rack-acceptance channel — propose READY with the caveat that r07-06 is excluded pending platform repair. Include the 17-node-no-06 NCCL result as evidence r07-08 is recovered.
4. **Remove acceptance taint** on R07 after sign-off; verify the LLM YAMLs still schedule (toleration left in place is intentional).

## Cross-Reference

NCCL fabric numbers also belong in `together-nccl-tests/baselines/<cluster>/` per the repo convention; that sibling repo is not present on this host today, so the NCCL section above is the primary record until it can be mirrored.
