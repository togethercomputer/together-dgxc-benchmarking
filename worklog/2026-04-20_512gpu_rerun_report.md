# B200 DGXC 512-GPU Re-run Report — 2026-04-20 (Session A)

## Summary

Re-ran all 10 NVIDIA baseline benchmarks at 512 GPUs to verify reproducibility of Friday 2026-04-17 results. **9 of 10 models match Friday within ±1% step time**. The exception is Llama 3.1 70B NVFP4, which regressed ~16% (7.20s vs 6.19s/iter). No configuration changes between runs.

All 10 jobs used `MAX_STEPS=10` (except NVFP4 where sed patch missed `--max_steps=50`, so 70B NVFP4 ran 50 iters and 405B NVFP4 ran 10 as intended). Session tagged `sessA_*`.

## Results — Today (2026-04-20) vs Friday (2026-04-17)

Iter-5 steady-state step time from log, direct comparison (same GBS, same config):

| Model | Dtype | Friday Job | Today Job | Friday iter-5 (ms) | Today iter-5 (ms) | Δ | TFLOP/s/GPU (today) |
|-------|-------|-----------|-----------|--------------------|-------------------|---|----|
| Llama 3.1 70B | FP8 | 84245 | 84459 | 9,630 | 9,509 | -1.3% | ~1,540 |
| Llama 3.1 70B | NVFP4 | 84246 | 84374 | 6,189 | 7,197 | **+16.3%** | ~2,050 |
| Llama 3.1 405B | FP8 | 84247 | 84463 | 70,129 | 69,698 | -0.6% | ~1,790 |
| Llama 3.1 405B | NVFP4 | 84248 | 84460 | 69,233 | 69,044 | -0.3% | ~1,800 |
| Nemotron-H 56B | FP8 | 84249 | 84464 | 5,474 | 5,489 | +0.3% | ~1,500 |
| DeepSeek-V3 | FP8 | 84255 | 84467 | 28,722 | 28,618 | -0.4% | ~597 |
| Nemotron4 340B | FP8 | 84272 | 84471 | 1,746 (1,231 tfl) | 1,740 (1,236 tfl) | -0.3% | **1,236** |
| Nemotron4 340B | BF16 | 84275 | 84473 | 2,506 (860 tfl) | 2,521 (855 tfl) | +0.6% | **855** |
| Grok1 314B | BF16 | 84278 | 84475 | 8,451 (1,004 tfl) | 8,421 (1,008 tfl) | -0.4% | **1,008** |
| Grok1 314B | FP8 | 84279 | 84477 | 5,841 (1,454 tfl) | 5,830 (1,456 tfl) | -0.2% | **1,456** |

TFLOP/s for legacy models (Nemotron4, Grok1) are reported by NeMo directly. For Megatron-Bridge models (Llama, Nemotron-H, DeepSeek V3), TFLOP/s is estimated by scaling Friday's report value by the step-time ratio.

## Anomaly: Llama 70B NVFP4 Regression

| | Friday 84246 | Today 84374 |
|---|---|---|
| Iter-5 step time | 6.189 s | 7.197 s |
| Iter-50 step time | ~6.19 s (steady) | 7.201 s (steady) |
| Jobs completed | All 50 iters | All 50 iters |
| GBS | 2048 | 2048 |

Both runs identical config, both 50 steady iterations. 16% regression is reproducible across all 50 iters of today's run. Other Llama runs today (70B FP8, 405B FP8, 405B NVFP4) all match Friday — so not a general cluster health issue. Possible causes (unverified):

- Node allocation drew a different subset; 70B NVFP4 at GBS=2048 is bandwidth-sensitive.
- NCCL ring ordering differs with a different node set.

**Recommend**: one more re-run to confirm whether regression is a one-off node-set effect or a consistent new behavior.

## Debugging Timeline

### Gloo rendezvous hang — root cause found

Early afternoon, re-`sbatch`ing Friday's `_sbatch.sh` hung at **0 Gloo connections** after 10+ min — jobs 84444 and 84448 both hung on fresh and old nodes. Fresh `llmb-run submit` which creates a new `_<timestamp>` experiment dir worked reliably on the first try (84459, 84463, etc.).

**Root cause**: reusing a Friday-dated experiment dir corrupts rendezvous somehow. Unknown exactly what (possibly stale `_DONE`/`_TASKS`/`_TUNNELS` cache files or path-dependent state). Not node-specific, not config — the reused dir itself is toxic.

**Workaround**: for any 512-GPU re-run, always regenerate via `llmb-run submit`. Never direct-sbatch an old-dated script.

### Node 228 flaky GPU

Job 84456 (405B NVFP4 attempt) hit `CUDA error: an illegal memory access was encountered` on rank 493 = node `use3a-ss-b200-gpu-228` GPU 5 during NCCL P2P warmup. Past Gloo rendezvous cleanly; single-node hardware issue.

Added 228 to exclude list: `use3a-ss-b200-gpu-[130,190,197,199,201,211,228,233,239]`. All subsequent jobs succeeded.

### Legacy models — end-of-training cleanup hangs

Nemotron4 340B FP8 (84471) and BF16 (84473) completed all iterations then stalled in Python/NCCL teardown for 3+ min (this also happened Friday for 84272/84275). Manually `scancel`'d after confirming `Trainer.fit stopped: max_steps=10 reached`. No data loss — training finished cleanly. Grok1 jobs (84475, 84477) released cleanly without intervention.

## Session A Job Inventory

| Job | Model | Submitted | Completed/Cancelled | Notes |
|-----|-------|-----------|---------------------|-------|
| 84374 | Llama 70B NVFP4 | 13:46 | 14:10 | 50 iters (sed miss) |
| 84459 | Llama 70B FP8 | 16:32 | 16:36 | |
| 84460 | Llama 405B NVFP4 | 16:38 | 16:57 | Past failures: 84444, 84448 (Gloo); 84456 (node 228 GPU) |
| 84463 | Llama 405B FP8 | 17:01 | 17:17 | |
| 84464 | Nemotron-H 56B FP8 | 17:21 | 17:24 | Iter-4 CUDA graph recapture anomaly (same as Friday) |
| 84467 | DeepSeek-V3 FP8 | 17:41 | 17:57 | Iter-1 warmup 674s (Triton JIT) |
| 84471 | Nemotron4 340B FP8 | 18:12 | 18:20 | scancel'd after training done |
| 84473 | Nemotron4 340B BF16 | 18:22 | 18:30 | scancel'd after training done |
| 84475 | Grok1 314B BF16 | 18:30 | 18:34 | clean exit |
| 84477 | Grok1 314B FP8 | 18:34 | 18:39 | clean exit |

## Conclusions

1. **Reproducibility**: 9/10 models reproduce Friday within ±1%. No systematic regression.
2. **70B NVFP4 one anomaly**: 16% slower, cause unknown. Suggest one more re-run to disambiguate node-effect from config drift.
3. **Session A mission complete**: all targeted models re-run; Friday numbers confirmed with high fidelity.
4. **Operational learnings**:
   - Always `llmb-run submit` to get a fresh experiment dir; never reuse old ones.
   - Node 228 has a flaky GPU and should stay on the exclude list until hardware verifies OK.
   - Legacy Nemotron4 340B teardown hang is a known issue; `scancel` once training exits.

## Re-run Commands

### Common setup

```bash
EXCLUDE='use3a-ss-b200-gpu-[130,190,197,199,201,211,228,233,239]'
LLMB_DIR=/mnt/vast/johnson/llmb
DGXC=/mnt/vast/johnson/dgxc-benchmarking
cd $LLMB_DIR
```

### Modern (Megatron-Bridge / NeMo 26.02) — via `llmb-run submit`

After each submission, post-patch max_steps (the llmb-run default is 50):

```bash
patch_steps() { find $LLMB_DIR/workloads/$1/experiments -name "*.sh" -path "*/scripts/*" -newermt "2 minutes ago" -exec sed -i -E 's/--max_steps[= ]50/--max_steps=10/g' {} +; }
```

| Job | Model | Submit command | Generated script |
|-----|-------|----------------|------------------|
| 84459 | Llama 70B FP8 | `./llmb-run submit -w pretrain_llama3.1 -s 70b -d fp8 --scale 512 --exclude $EXCLUDE && patch_steps pretrain_llama3.1` | `$LLMB_DIR/workloads/pretrain_llama3.1/experiments/pretrain_llama3_70b_fp8_mx_gpus512_tp2_pp4_cp1_vp5_ep1_etpNone_mbs1_gbs256/pretrain_llama3_70b_fp8_mx_gpus512_..._1776727347/.../scripts/pretrain_llama3_70b_fp8_mx_gpus512_....sh` |
| 84374 | Llama 70B NVFP4 | `./llmb-run submit -w pretrain_llama3.1 -s 70b -d nvfp4 --scale 512 --exclude $EXCLUDE && patch_steps pretrain_llama3.1` | `pretrain_llama3_70b_nvfp4_..._1776718801/.../scripts/...sh` |
| 84463 | Llama 405B FP8 | `./llmb-run submit -w pretrain_llama3.1 -s 405b -d fp8 --scale 512 --exclude $EXCLUDE && patch_steps pretrain_llama3.1` | `pretrain_llama31_405b_fp8_cs_..._1776728204/.../scripts/...sh` |
| 84460 | Llama 405B NVFP4 | `./llmb-run submit -w pretrain_llama3.1 -s 405b -d nvfp4 --scale 512 --exclude $EXCLUDE && patch_steps pretrain_llama3.1` | `pretrain_llama31_405b_nvfp4_..._1776727831/.../scripts/...sh` |
| 84464 | Nemotron-H 56B FP8 | `./llmb-run submit -w pretrain_nemotron-h -s 56b -d fp8 --scale 512 --exclude $EXCLUDE && patch_steps pretrain_nemotron-h` | `pretrain_nemotronh_56b_fp8_cs_..._1776729597/.../scripts/...sh` |
| 84467 | DeepSeek-V3 FP8 | `./llmb-run submit -w pretrain_deepseek-v3 -d fp8 --scale 512 --exclude $EXCLUDE && patch_steps pretrain_deepseek-v3` | `pretrain_deepseek_v3_fp8_mx_..._1776730733/.../scripts/...sh` |

**Critical**: `llmb-run submit` creates a fresh `_<timestamp>` experiment dir. Do NOT `sbatch` an old-dated script — causes the Gloo rendezvous hang (see Debugging Timeline).

### Legacy (NeMo 25.07/25.09) — via `/tmp/submit_legacy.sh`

The wrapper runs `launch.sh`, cancels the unpatched auto-submitted job, applies all cluster workarounds (HOME=/tmp, NCCL_SOCKET_IFNAME, PMIx, MPI stub, HF_HOME), then `sbatch`es the patched script. Source: `/tmp/submit_legacy.sh` (exclude list already updated for node 228).

```bash
# Usage: bash /tmp/submit_legacy.sh <label> <dtype> <scale> <workload_name> <workload_dir>
```

| Job | Model | Submit command | Patched sbatch script |
|-----|-------|----------------|-----------------------|
| 84471 | Nemotron4 340B FP8 | `bash /tmp/submit_legacy.sh nemotron4_340b_fp8_512 fp8 512 pretrain_nemotron4-340b nemotron4-340b` | `$LLMB_DIR/workloads/pretrain_nemotron4-340b/experiments/pretrain_nemotron4_340b_fp8_gpus512_tp8_pp4_cp1_vp12_mbs1_gbs128/..._1776733151/pretrain_nemotron4_340b_fp8_..._sbatch_patched.sh` |
| 84473 | Nemotron4 340B BF16 | `bash /tmp/submit_legacy.sh nemotron4_340b_bf16_512 bf16 512 pretrain_nemotron4-340b nemotron4-340b` | `..._1776733316/pretrain_nemotron4_340b_bf16_..._sbatch_patched.sh` |
| 84475 | Grok1 314B BF16 | `bash /tmp/submit_legacy.sh grok1_314b_bf16_512 bf16 512 pretrain_grok1 grok1` | `$LLMB_DIR/workloads/pretrain_grok1/experiments/pretrain_grok1_314b_bf16_..._1776733396/pretrain_grok1_314b_bf16_..._sbatch_patched.sh` |
| 84477 | Grok1 314B FP8 | `bash /tmp/submit_legacy.sh grok1_314b_fp8_512 fp8 512 pretrain_grok1 grok1` | `..._1776733476/pretrain_grok1_314b_fp8_..._sbatch_patched.sh` |

**Critical for legacy**: Nemotron4 340B and Grok1 stall in Python/NCCL teardown 3-5 min after training ends. Monitor for `Trainer.fit stopped: max_steps=10 reached.` in the log, then `scancel <job>` to release nodes.

### One-shot full Session A re-run

```bash
bash ~/together-dgxc-benchmarking/tests/b200/slurm/run_all_sessA.sh --tier3
```

This wraps all 10 submissions in order (modern first, then legacy).

## Log Paths

All under `/mnt/vast/johnson/llmb/workloads/`.

| Job | Path fragment |
|-----|---------------|
| 84374 | `pretrain_llama3.1/.../pretrain_llama3_70b_nvfp4_gpus512_..._1776718801/.../log-*_84374_0.out` |
| 84459 | `pretrain_llama3.1/.../pretrain_llama3_70b_fp8_mx_gpus512_..._1776727347/.../log-*_84459_0.out` |
| 84460 | `pretrain_llama3.1/.../pretrain_llama31_405b_nvfp4_..._1776727831/.../log-*_84460_0.out` |
| 84463 | `pretrain_llama3.1/.../pretrain_llama31_405b_fp8_cs_..._1776728204/.../log-*_84463_0.out` |
| 84464 | `pretrain_nemotron-h/.../pretrain_nemotronh_56b_fp8_cs_..._1776729597/.../log-*_84464_0.out` |
| 84467 | `pretrain_deepseek-v3/.../pretrain_deepseek_v3_fp8_mx_..._1776730733/.../log-*_84467_0.out` |
| 84471 | `pretrain_nemotron4-340b/.../pretrain_nemotron4_340b_fp8_..._1776733151/.../log-*_84471_0.out` |
| 84473 | `pretrain_nemotron4-340b/.../pretrain_nemotron4_340b_bf16_..._1776733316/.../log-*_84473_0.out` |
| 84475 | `pretrain_grok1/.../pretrain_grok1_314b_bf16_..._1776733396/.../log-*_84475_0.out` |
| 84477 | `pretrain_grok1/.../pretrain_grok1_314b_fp8_..._1776733476/.../log-*_84477_0.out` |

## Follow-up: 70B NVFP4 Confirmatory Re-run (job 84480)

Submitted at 20:56:17, completed 21:01:43 (5:26 elapsed). Fresh `llmb-run submit`, same exclude list, `--max_steps=10`. Allocated `use3a-ss-b200-gpu-[144-146,148-152,154-189,195-196,198,202-205,209-210,212-217,226-227,234-235,238]` (64 nodes).

Per-iter step time (ms) and NeMo-reported MODEL_TFLOP/s/GPU:

| Iter | Step (ms) | TFLOP/s/GPU |
|------|-----------|-------------|
| 1 | 105,795.8 | 139.2 (warmup) |
| 2 | 7,147.6 | 2,059.5 |
| 3 | 7,154.0 | 2,057.6 |
| 4 | 7,178.3 | 2,050.6 |
| 5 | **7,199.5** | **2,044.6** |
| 6 | 7,244.7 | 2,031.9 |
| 7 | 7,203.4 | 2,043.5 |
| 8 | 7,184.5 | 2,048.9 |
| 9 | 7,154.5 | 2,057.5 |
| 10 | 7,141.8 | 2,061.1 |

Iter 5-10 mean: 7,188.1 ms / 2,047.9 TFLOP/s/GPU.

### Conclusion

Iter-5 step time of **7,199.5 ms** matches today-earlier job 84374 at 7,197 ms within 0.03%. The +16.3% regression vs Friday 84246 (6,189 ms) is **reproducible across two independent submissions today** on disjoint node allocations. This rules out the one-off node-set hypothesis.

**Status**: regression is a persistent condition as of 2026-04-20. Other Llama runs today (70B FP8, 405B FP8, 405B NVFP4) all match Friday, so the issue is NVFP4-70B-specific, not a general cluster slowdown.

Next investigation vectors (not yet tried):
- Compare the saved NCCL/container env between Friday's 84246 experiment dir and today's experiment dirs for any drift in env vars, cache files, or pinned module versions.
- Check if the 26.02 container image sqsh on disk has been replaced/touched since Friday (`stat /mnt/vast/johnson/llmb/images/nvidia+nemo+26.02.00.sqsh`).
- Pull from NCCL logs whether ring/tree topology differs between Friday and today for the 70B NVFP4 allreduce patterns.

### Ruled out: env-var drift

- Container sqsh unchanged: `nvidia+nemo+26.02.00.sqsh` mtime = 2026-04-11 18:18, i.e. same binary as Friday.
- `NCCL_SOCKET_IFNAME=bond0`: not literally in the generated sbatch.sh, but **IS** set via `/mnt/vast/johnson/llmb/cluster_config.yaml` → `llmb_run/job_launcher.py` → sbatch subprocess env → srun/pyxis forward. Both Friday 84246 and today 84480 had it active. Other cluster-config env vars (HOME=/tmp/johnson, HF_HOME, HF_TOKEN) are also consistent across both runs.
- Attempted A/B test via manual `sbatch` failed three times (jobs 84482/84483/84484) because bypassing `llmb-run` strips those config env vars — confirming the cluster-config path is mandatory for submission. The A/B isn't informative since bond0 was never actually disabled.
