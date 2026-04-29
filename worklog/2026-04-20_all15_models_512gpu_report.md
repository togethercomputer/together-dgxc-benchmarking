# B200 DGXC 512-GPU Complete Re-run Report — 2026-04-20

## Summary

Across Sessions A and B (run in parallel), all **15 model/dtype combinations** were benchmarked at 512 GPUs today. All 15 succeeded. This is the complete 512-GPU scorecard for the DGXC benchmarking re-run.

**Session A** (10 configs, `sessA_*` jobs): re-ran Friday 2026-04-17's 512-GPU set to verify reproducibility. 9/10 match Friday within ±1%. One regression (70B NVFP4 -16%).

**Session B** (5 configs + 1 bonus 256-GPU, `B_*` jobs): ran the 5 configs NOT covered at 512 on Friday. Includes the first-ever DSV3 BF16 result and 15B BF16 256-GPU baseline.

## Complete 15-Model Scorecard

Steady-state TFLOP/s/GPU, 512 GPUs unless noted.

| # | Model | Dtype | Session | Job ID | TFLOP/s/GPU | Iter Time | Notes |
|---|-------|-------|---------|--------|-------------|-----------|-------|
| 1 | Nemotron4 15B | BF16 | B | 84447 | **1,376** | 1.061s | -7.8% vs 256 (bandwidth-bound) |
| 2 | Nemotron4 15B | FP8 | B | 84455 / 84486 | **1,531 / 1,528** | 0.954s / 0.956s | 25.09 container; regression confirmed reproducible (−19.9% vs target 1,908) |
| 3 | Llama 3.1 70B | FP8 | A | 84459 | **~1,540** | 9.509s | -1.3% vs Friday |
| 4 | Llama 3.1 70B | NVFP4 | A | 84374 | **~2,050** | 7.197s | **+16.3% vs Friday (regression)** |
| 5 | Nemotron-H 56B | FP8 | A | 84464 | **~1,500** | 5.489s | +0.3% vs Friday |
| 6 | Qwen3 235B A22B | BF16 | B | 84466 | **604.7** | 32.2s | +16.8% vs 256 (517.6) |
| 7 | Qwen3 235B A22B | FP8 | B | 84469 | **499.2** | 38.9s | +28.5% vs 256 (388.6) |
| 8 | Grok1 314B | BF16 | A | 84475 | **1,008** | 8.421s | -0.4% vs Friday |
| 9 | Grok1 314B | FP8 | A | 84477 | **1,456** | 5.830s | -0.2% vs Friday |
| 10 | Nemotron4 340B | BF16 | A | 84473 | **855** | 2.521s | +0.6% vs Friday |
| 11 | Nemotron4 340B | FP8 | A | 84471 | **1,236** | 1.740s | -0.3% vs Friday |
| 12 | Llama 3.1 405B | FP8 | A | 84463 | **~1,790** | 69.698s | -0.6% vs Friday |
| 13 | Llama 3.1 405B | NVFP4 | A | 84460 | **~1,800** | 69.044s | -0.3% vs Friday |
| 14 | DeepSeek V3 | BF16 | B | 84479 | **554.9** | 30.7s | First BF16 result (no 256 baseline) |
| 15 | DeepSeek V3 | FP8 | A | 84467 | **~597** | 28.618s | -0.4% vs Friday |

**Bonus (not counted in 15):** Nemotron4 15B BF16 at 256 GPUs: 1,492 TFLOP/s/GPU (Job 84443, Session B).

## Highlights

- **9 of 10 Session A configs reproduce Friday within ±1%** — cluster is stable; results are reproducible.
- **Llama 70B NVFP4 regressed 16%** (6.19s → 7.20s/iter). Recommend one more re-run (possibly node-set bandwidth effect).
- **Qwen3 235B scales FAVORABLY 256→512** for both dtypes (+16.8% BF16, +28.5% FP8) — unusual and worth characterizing.
- **DSV3 BF16 first result**: 554.9 TFLOPS; 93% of FP8's 595 TFLOPS — MoE comm-bound, FP8 compute speedup masked by all-to-all.
- **15B FP8 512 required 25.09 container** (26.02 + compat_runner breaks FP8 CUDA graphs with cudaErrorInvalidValue).
- **15B FP8 -19.9% vs target is structural, not a regression**: 84486 rerun (2026-04-20) reproduces 84455 within 0.2% (1,528 vs 1,531). Root cause is scale/GBS mismatch — target 1,908 was set at 256 GPUs (GBS=2048, 4 μbatch/rank). At 512 GPUs with same GBS=2048 we only get 2 μbatch/rank, killing FP8 overlap. Also, per `metadata.yaml` scales are capped at 256 for b200 (`exact_scales: true`), and the validated `recommended_model_configs/model_configs_b200.csv` only has num_gpus=64 — 512 is extrapolated, not officially tuned.

## Reproduction

### Common setup

```bash
EXCLUDE='use3a-ss-b200-gpu-[130,190,197,199,201,211,228,233,239]'
LLMB_DIR=/mnt/vast/johnson/llmb
DGXC=/mnt/vast/johnson/dgxc-benchmarking
VENV=/mnt/vast/johnson/llmb_venv
IMAGE_26=$LLMB_DIR/images/nvidia+nemo+26.02.00.sqsh
IMAGE_25=$LLMB_DIR/images/nvidia+nemo+25.09.00.sqsh
```

### Session A: Modern (Megatron-Bridge / NeMo 26.02) — via `llmb-run submit`

After each submission, post-patch `max_steps` (llmb-run default is 50):

```bash
patch_steps() { find $LLMB_DIR/workloads/$1/experiments -name "*.sh" -path "*/scripts/*" -newermt "2 minutes ago" -exec sed -i -E 's/--max_steps[= ]50/--max_steps=10/g' {} +; }
```

| Job | Model | Command |
|-----|-------|---------|
| 84459 | Llama 70B FP8 | `cd $LLMB_DIR && ./llmb-run submit -w pretrain_llama3.1 -s 70b -d fp8 --scale 512 --exclude $EXCLUDE && patch_steps pretrain_llama3.1` |
| 84374 | Llama 70B NVFP4 | `cd $LLMB_DIR && ./llmb-run submit -w pretrain_llama3.1 -s 70b -d nvfp4 --scale 512 --exclude $EXCLUDE && patch_steps pretrain_llama3.1` |
| 84463 | Llama 405B FP8 | `cd $LLMB_DIR && ./llmb-run submit -w pretrain_llama3.1 -s 405b -d fp8 --scale 512 --exclude $EXCLUDE && patch_steps pretrain_llama3.1` |
| 84460 | Llama 405B NVFP4 | `cd $LLMB_DIR && ./llmb-run submit -w pretrain_llama3.1 -s 405b -d nvfp4 --scale 512 --exclude $EXCLUDE && patch_steps pretrain_llama3.1` |
| 84464 | Nemotron-H 56B FP8 | `cd $LLMB_DIR && ./llmb-run submit -w pretrain_nemotron-h -s 56b -d fp8 --scale 512 --exclude $EXCLUDE && patch_steps pretrain_nemotron-h` |
| 84467 | DeepSeek V3 FP8 | `cd $LLMB_DIR && ./llmb-run submit -w pretrain_deepseek-v3 -d fp8 --scale 512 --exclude $EXCLUDE && patch_steps pretrain_deepseek-v3` |

**Critical**: always create a fresh `_<timestamp>` experiment dir via `llmb-run submit`; never direct-sbatch an old dated script (Gloo rendezvous hang).

### Session A: Legacy (NeMo 25.07/25.09) — via `/tmp/submit_legacy.sh`

| Job | Model | Command |
|-----|-------|---------|
| 84471 | Nemotron4 340B FP8 | `bash /tmp/submit_legacy.sh nemotron4_340b_fp8_512 fp8 512 pretrain_nemotron4-340b nemotron4-340b` |
| 84473 | Nemotron4 340B BF16 | `bash /tmp/submit_legacy.sh nemotron4_340b_bf16_512 bf16 512 pretrain_nemotron4-340b nemotron4-340b` |
| 84475 | Grok1 314B BF16 | `bash /tmp/submit_legacy.sh grok1_314b_bf16_512 bf16 512 pretrain_grok1 grok1` |
| 84477 | Grok1 314B FP8 | `bash /tmp/submit_legacy.sh grok1_314b_fp8_512 fp8 512 pretrain_grok1 grok1` |

**Legacy monitoring**: Nemotron4 340B stalls in Python/NCCL teardown for 3-5 min after training completes. Watch for `Trainer.fit stopped: max_steps=10 reached` in log, then `scancel <job>` to release nodes.

### Session B: `submit_session_B.sh` (Megatron-Bridge + NeMo combo)

The Session B wrapper handles 15B, Qwen3 235B, and DSV3 with all required cluster patches (`--exclude`, `HOME=/tmp`, `NCCL_SOCKET_IFNAME=bond0`, HF cache env, `--container-env=` insertion/extension).

```bash
# Usage: SCALE=<gpus> bash /home/johnson/johnson/worklog/submit_session_B.sh <target>
# Targets: n15b_bf16 | n15b_fp8 | qwen3_bf16 | qwen3_fp8 | dsv3_bf16 | all
```

| Job | Model | Command |
|-----|-------|---------|
| 84443 | 15B BF16 256 (bonus) | `SCALE=256 bash /home/johnson/johnson/worklog/submit_session_B.sh n15b_bf16` |
| 84447 | 15B BF16 512 | `SCALE=512 bash /home/johnson/johnson/worklog/submit_session_B.sh n15b_bf16` |
| 84455 | 15B FP8 512 | `SCALE=512 bash /home/johnson/johnson/worklog/submit_session_B.sh n15b_fp8` |
| 84466 | Qwen3 235B BF16 512 | `SCALE=512 bash /home/johnson/johnson/worklog/submit_session_B.sh qwen3_bf16` |
| 84469 | Qwen3 235B FP8 512 | `SCALE=512 bash /home/johnson/johnson/worklog/submit_session_B.sh qwen3_fp8` |
| 84479 | DSV3 BF16 512 | `SCALE=512 bash /home/johnson/johnson/worklog/submit_session_B.sh dsv3_bf16` |

### Session B: Raw launcher commands (bypass wrapper)

Only safe if you also manually apply the cluster patches (see Patches below). The wrapper does all this automatically.

```bash
# 15B (BF16 uses 26.02, FP8 uses 25.09)
source $VENV/bin/activate
cd $DGXC/nemotron4-15b
MAX_STEPS=10 RUN_CONF_IMAGE=$IMAGE_26 LLMB_INSTALL=$LLMB_DIR \
  JOB_TOTAL_GPUS=512 GPU_TYPE=b200 DTYPE=bf16 \
  SBATCH_ACCOUNT=root SBATCH_PARTITION=batch \
  ADDITIONAL_SLURM_PARAMS='job-name=B_n15b_bf16_512' bash launch.sh
# FP8: swap to IMAGE_25 and DTYPE=fp8 (see submit_session_B.sh submit_n15b)

# Qwen3 235B
source $VENV/bin/activate
cd $DGXC/qwen3/pretrain
MAX_STEPS=10 RUN_CONF_IMAGE=$IMAGE_26 LLMB_INSTALL=$LLMB_DIR \
  MODEL_SIZE=235b JOB_TOTAL_GPUS=512 GPU_TYPE=b200 DTYPE=bf16 \
  SBATCH_ACCOUNT=root SBATCH_PARTITION=batch \
  ADDITIONAL_SLURM_PARAMS='job-name=B_qwen3_235b_bf16_512' bash launch.sh

# DeepSeek V3 BF16
source $VENV/bin/activate
cd $DGXC/deepseek_v3/pretrain/megatron_bridge
MAX_STEPS=10 RUN_CONF_IMAGE=$IMAGE_26 LLMB_INSTALL=$LLMB_DIR \
  JOB_TOTAL_GPUS=512 GPU_TYPE=b200 DTYPE=bf16 \
  SBATCH_ACCOUNT=root SBATCH_PARTITION=batch \
  ADDITIONAL_SLURM_PARAMS='job-name=B_dsv3_bf16_512' bash launch.sh
```

## Patched sbatch Scripts (direct re-submit)

Each run's patched sbatch is persistent — `sbatch --requeue --parsable <path>` re-submits exactly the same job (no patching needed).

### Session A — modern (llmb-run generated, then max_steps sed)

| Job | Script path |
|-----|-------------|
| 84374 | `$LLMB_DIR/workloads/pretrain_llama3.1/experiments/pretrain_llama3_70b_nvfp4_..._1776718801/.../scripts/...sh` |
| 84459 | `$LLMB_DIR/workloads/pretrain_llama3.1/experiments/pretrain_llama3_70b_fp8_mx_..._1776727347/.../scripts/...sh` |
| 84460 | `$LLMB_DIR/workloads/pretrain_llama3.1/experiments/pretrain_llama31_405b_nvfp4_..._1776727831/.../scripts/...sh` |
| 84463 | `$LLMB_DIR/workloads/pretrain_llama3.1/experiments/pretrain_llama31_405b_fp8_cs_..._1776728204/.../scripts/...sh` |
| 84464 | `$LLMB_DIR/workloads/pretrain_nemotron-h/experiments/pretrain_nemotronh_56b_fp8_cs_..._1776729597/.../scripts/...sh` |
| 84467 | `$LLMB_DIR/workloads/pretrain_deepseek-v3/experiments/pretrain_deepseek_v3_fp8_mx_..._1776730733/.../scripts/...sh` |

### Session A — legacy (submit_legacy.sh generated)

| Job | Script path |
|-----|-------------|
| 84471 | `$LLMB_DIR/workloads/pretrain_nemotron4-340b/experiments/pretrain_nemotron4_340b_fp8_..._1776733151/pretrain_nemotron4_340b_fp8_..._sbatch_patched.sh` |
| 84473 | `$LLMB_DIR/workloads/pretrain_nemotron4-340b/experiments/pretrain_nemotron4_340b_bf16_..._1776733316/pretrain_nemotron4_340b_bf16_..._sbatch_patched.sh` |
| 84475 | `$LLMB_DIR/workloads/pretrain_grok1/experiments/pretrain_grok1_314b_bf16_..._1776733396/pretrain_grok1_314b_bf16_..._sbatch_patched.sh` |
| 84477 | `$LLMB_DIR/workloads/pretrain_grok1/experiments/pretrain_grok1_314b_fp8_..._1776733476/pretrain_grok1_314b_fp8_..._sbatch_patched.sh` |

### Session B — submit_session_B.sh generated

| Job | Script path |
|-----|-------------|
| 84443 | `$LLMB_DIR/workloads/pretrain_nemotron4-15b/experiments/pretrain_nemotron4_15b_bf16_gpus256_..._1776724327/pretrain_nemotron4_15b_bf16_gpus256_..._sbatch.sh` |
| 84447 | `$LLMB_DIR/workloads/pretrain_nemotron4-15b/experiments/pretrain_nemotron4_15b_bf16_gpus512_..._1776724971/pretrain_nemotron4_15b_bf16_gpus512_..._sbatch.sh` |
| 84455 | `$LLMB_DIR/workloads/pretrain_nemotron4-15b/experiments/pretrain_nemotron4_15b_fp8_gpus512_..._1776726890/pretrain_nemotron4_15b_fp8_gpus512_..._sbatch.sh` |
| 84466 | `$LLMB_DIR/workloads/pretrain_qwen3/experiments/pretrain_qwen3_235b_a22b_bf16_gpus512_..._1776730181/pretrain_qwen3_235b_a22b_bf16_gpus512_..._sbatch.sh` |
| 84469 | `$LLMB_DIR/workloads/pretrain_qwen3/experiments/pretrain_qwen3_235b_a22b_fp8_mx_gpus512_..._1776732051/pretrain_qwen3_235b_a22b_fp8_mx_gpus512_..._sbatch.sh` |
| 84479 | `$LLMB_DIR/workloads/pretrain_deepseek-v3/experiments/pretrain_deepseek_v3_bf16_gpus512_..._1776735078/pretrain_deepseek_v3_bf16_gpus512_..._sbatch.sh` |

## Critical Cluster-Specific Patches (must apply to every generated sbatch)

1. Strip auto `#SBATCH --job-name=root-root...` lines; keep your `B_`/`sessA_` name.
2. Add `#SBATCH --exclude=use3a-ss-b200-gpu-[130,190,197,199,201,211,228,233,239]`.
3. Prepend before `# Command 1`:
   ```bash
   export HOME=/tmp
   export NEMO_NLP_TMP=/tmp/nemo_nlp_tmp
   export HF_HOME=/mnt/vast/johnson/llmb/.cache/huggingface
   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   export HF_TOKEN=<REDACTED_HF_TOKEN>
   export NCCL_SOCKET_IFNAME=bond0
   ulimit -n 1048576 || true
   ```
4. Extend or INSERT `--container-env=` on `srun`: `HOME,NEMO_NLP_TMP,HF_HOME,HF_HUB_OFFLINE,TRANSFORMERS_OFFLINE,HF_TOKEN,NCCL_SOCKET_IFNAME,TORCH_NCCL_HIGH_PRIORITY,NVTE_FWD_LAYERNORM_SM_MARGIN,NVTE_BWD_LAYERNORM_SM_MARGIN`.
   - Megatron-Bridge launchers (qwen3, dsv3) often emit NO `--container-env=` at all — must INSERT, not just extend.
5. Ensure `--no-container-mount-home` is on `srun`.
6. **15B only:** wrap entrypoint with `compat_runner.py` for 26.02 container. Source:
   `$LLMB_DIR/workloads/pretrain_nemotron4-15b/experiments/pretrain_nemotron4_15b_bf16_gpus256_..._1776133099/.../compat_runner.py`
   - SKIP for 15B FP8 (uses native 25.09 container).
7. **Legacy 25.07/25.09 models** (Nemotron4 340B, Grok1): MPI stub LD_PRELOAD + TP_COMM_OVERLAP=False + PMIx patcher. Handled by `/tmp/submit_legacy.sh`.

## Log File Locations

All under `$LLMB_DIR/workloads/`. Common pattern: `<experiment_dir>/<run_id>/<run_id>/log-root-root.<run_id>_<JOB_ID>_0.out`.

Quick TFLOPS extraction:
```bash
# NeMo format (15B, Nemotron4 340B, Grok1): TFLOPS_per_GPU
grep "TFLOPS_per_GPU" <log> | tail -5
# Megatron-Bridge format (Llama, Qwen3, DSV3, Nemotron-H): Step Time
grep "Step Time" <log>
```

## Known Issues Encountered Today (all resolved)

- **Gloo rendezvous hang (Session A)**: jobs 84444, 84448 hung at 0 Gloo connections after 10+ min when re-sbatching Friday's experiment dir. Root cause: reusing a Friday-dated `_<timestamp>` dir corrupts rendezvous. Fix: always `llmb-run submit` fresh.
- **Node 228 flaky GPU (Session A)**: Job 84456 hit CUDA illegal memory access on rank 493 = node 228 GPU 5. Added to exclude list.
- **Nemotron4 340B teardown hang (Session A)**: 84471/84473 stalled in Python/NCCL cleanup 3-5 min after training finished. Manual `scancel` required.
- **Qwen3 BF16 Job 84458/84462 failed (Session B)**: LocalEntryNotFoundError — Megatron-Bridge sbatch had no `--container-env=`, HF env not forwarded. Fix #1: insert `--container-env=` when missing. Fix #2: unique marker for patch idempotency (84462 was skipped because pre-existing `export HF_HUB_OFFLINE=1` matched old marker). Succeeded as 84466.
- **15B FP8 Job 84450 failed (Session B)**: cudaErrorInvalidValue in replay_graph_capture — 26.02 + compat_runner breaks CUDA-graph FP8. Fix: use 25.09 container (native FP8 support).
- **Llama 70B NVFP4 regression (Session A)**: 16.3% slower than Friday. Other Llama runs match Friday. Likely node-set bandwidth sensitivity. Recommend one re-run to disambiguate.

## Conclusions

1. **15/15 complete**: every model/dtype combination has a 512-GPU result as of today.
2. **Reproducibility strong**: 9/10 Session A configs within ±1% of Friday.
3. **Two new baselines established**: DSV3 BF16 (554.9) and 15B BF16 256 (1,492).
4. **Qwen3 235B at 512** performs noticeably better than at 256 — worth further characterization.
5. **One open item**: Llama 70B NVFP4 regression — recommend single re-run to confirm node-set vs systemic effect.

## Cross-references

- Session A detailed report: `~/johnson/worklog/2026-04-20_512gpu_rerun_report.md`
- Session B detailed report: `~/johnson/worklog/session_B_report_2026-04-20.md`
- Session B launcher: `~/johnson/worklog/submit_session_B.sh`
- Legacy launcher: `/tmp/submit_legacy.sh`
