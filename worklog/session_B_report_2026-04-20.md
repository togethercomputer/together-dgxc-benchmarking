# Session B Report — 2026-04-20

Session B covered the 5 model/dtype configs that were NOT run at 512 GPUs on Friday 2026-04-17:
**Nemotron4 15B BF16 (256+512)**, **15B FP8 512**, **Qwen3 235B BF16 512**, **Qwen3 235B FP8 512**, **DeepSeek V3 BF16 512**.

All 6 jobs (5 configs + bonus 15B BF16 256) completed successfully with log-validated TFLOPS.

## Results Summary

| Model | Dtype | GPUs | Job ID | Steady TFLOP/s/GPU | Iter Time | vs 256-GPU |
|-------|-------|------|--------|-------------------|-----------|------------|
| Nemotron4 15B | BF16 | 256 | 84443 | **1,492** | 0.979s | baseline |
| Nemotron4 15B | BF16 | 512 | 84447 | **1,376** | 1.061s | -7.8% |
| Nemotron4 15B | FP8  | 512 | 84455 | **1,531** | 0.954s | n/a |
| Qwen3 235B    | BF16 | 512 | 84466 | **604.7** | 32.2s   | +16.8% (vs 517.6) |
| Qwen3 235B    | FP8  | 512 | 84469 | **499.2** | 38.9s   | +28.5% (vs 388.6) |
| DeepSeek V3   | BF16 | 512 | 84479 | **554.9** | 30.7s   | no 256 baseline |

Notes:
- 15B BF16 at 512 drops 7.8% vs 256 because bandwidth-bound at small model.
- Qwen3 235B at 512 IMPROVES over 256 for both dtypes (scale benefits MoE).
- DSV3 BF16 is 93% of FP8 (595 TFLOPS) — MoE comm-bound, FP8 speedup masked.
- 15B FP8 512 required **25.09 container** (26.02+compat_runner incompat with CUDA graph FP8).

## Reproduction

### Prerequisites (one-time setup)
- Cluster: Together AI B200 DGXC (login `use3a-ss-b200-login-01`), SLURM partition `batch`
- Venv: `/mnt/vast/johnson/llmb_venv`
- Containers:
  - 26.02: `/mnt/vast/johnson/llmb/images/nvidia+nemo+26.02.00.sqsh`
  - 25.09: `/mnt/vast/johnson/llmb/images/nvidia+nemo+25.09.00.sqsh` (only for 15B FP8)
- Launcher wrapper: `/home/johnson/johnson/worklog/submit_session_B.sh`
- Unhealthy node exclude list in wrapper: `use3a-ss-b200-gpu-[130,190,197,199,201,211,233,239]`

### One-command rerun of any config

```bash
# Usage: SCALE=<gpus> bash submit_session_B.sh <target>
# Target is one of: n15b_bf16, n15b_fp8, qwen3_bf16, qwen3_fp8, dsv3_bf16, all

# Session B exact reproduction:
SCALE=256 bash /home/johnson/johnson/worklog/submit_session_B.sh n15b_bf16   # Job 84443
SCALE=512 bash /home/johnson/johnson/worklog/submit_session_B.sh n15b_bf16   # Job 84447
SCALE=512 bash /home/johnson/johnson/worklog/submit_session_B.sh n15b_fp8    # Job 84455 (uses 25.09 container)
SCALE=512 bash /home/johnson/johnson/worklog/submit_session_B.sh qwen3_bf16  # Job 84466
SCALE=512 bash /home/johnson/johnson/worklog/submit_session_B.sh qwen3_fp8   # Job 84469
SCALE=512 bash /home/johnson/johnson/worklog/submit_session_B.sh dsv3_bf16   # Job 84479
```

The wrapper auto-handles: (1) run upstream `launch.sh` to generate sbatch → auto-submits → (2) cancel that job → (3) patch generated sbatch (B_ name, `--exclude=`, `HOME=/tmp`, `NCCL_SOCKET_IFNAME=bond0`, HF cache env, `--container-env=` extension/insertion) → (4) resubmit as patched job.

### Generated sbatch scripts (the actual thing Slurm ran)

These files are persistent — can be re-submitted directly without running the wrapper.

| Job | Patched sbatch path | Direct resubmit |
|-----|---------------------|-----------------|
| 84443 | `/mnt/vast/johnson/llmb/workloads/pretrain_nemotron4-15b/experiments/pretrain_nemotron4_15b_bf16_gpus256_tp1_pp1_cp1_vp1_mbs2_gbs1024/pretrain_nemotron4_15b_bf16_gpus256_tp1_pp1_cp1_vp1_mbs2_gbs1024_1776724327/pretrain_nemotron4_15b_bf16_gpus256_tp1_pp1_cp1_vp1_mbs2_gbs1024_sbatch.sh` | `sbatch --requeue --parsable <path>` |
| 84447 | `/mnt/vast/johnson/llmb/workloads/pretrain_nemotron4-15b/experiments/pretrain_nemotron4_15b_bf16_gpus512_tp1_pp1_cp1_vp1_mbs2_gbs2048/pretrain_nemotron4_15b_bf16_gpus512_tp1_pp1_cp1_vp1_mbs2_gbs2048_1776724971/pretrain_nemotron4_15b_bf16_gpus512_tp1_pp1_cp1_vp1_mbs2_gbs2048_sbatch.sh` | same |
| 84455 | `/mnt/vast/johnson/llmb/workloads/pretrain_nemotron4-15b/experiments/pretrain_nemotron4_15b_fp8_gpus512_tp1_pp1_cp1_vp1_mbs2_gbs2048/pretrain_nemotron4_15b_fp8_gpus512_tp1_pp1_cp1_vp1_mbs2_gbs2048_1776726890/pretrain_nemotron4_15b_fp8_gpus512_tp1_pp1_cp1_vp1_mbs2_gbs2048_sbatch.sh` | same |
| 84466 | `/mnt/vast/johnson/llmb/workloads/pretrain_qwen3/experiments/pretrain_qwen3_235b_a22b_bf16_gpus512_tp1_pp8_cp1_vp4_ep8_etp1_mbs1_gbs8192/pretrain_qwen3_235b_a22b_bf16_gpus512_tp1_pp8_cp1_vp4_ep8_etp1_mbs1_gbs8192_1776730181/pretrain_qwen3_235b_a22b_bf16_gpus512_tp1_pp8_cp1_vp4_ep8_etp1_mbs1_gbs8192_sbatch.sh` | same |
| 84469 | `/mnt/vast/johnson/llmb/workloads/pretrain_qwen3/experiments/pretrain_qwen3_235b_a22b_fp8_mx_gpus512_tp1_pp8_cp1_vpNone_ep8_etp1_mbs1_gbs8192/pretrain_qwen3_235b_a22b_fp8_mx_gpus512_tp1_pp8_cp1_vpNone_ep8_etp1_mbs1_gbs8192_1776732051/pretrain_qwen3_235b_a22b_fp8_mx_gpus512_tp1_pp8_cp1_vpNone_ep8_etp1_mbs1_gbs8192_sbatch.sh` | same |
| 84479 | `/mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3/experiments/pretrain_deepseek_v3_bf16_gpus512_tp1_pp16_cp1_vpNone_ep8_etp1_mbs1_gbs4096/pretrain_deepseek_v3_bf16_gpus512_tp1_pp16_cp1_vpNone_ep8_etp1_mbs1_gbs4096_1776735078/pretrain_deepseek_v3_bf16_gpus512_tp1_pp16_cp1_vpNone_ep8_etp1_mbs1_gbs4096_sbatch.sh` | same |

### Raw launcher commands (bypass the wrapper)

Direct invocations if you want to run without `submit_session_B.sh` (you'll then need to manually patch the generated sbatch — see wrapper for the patch steps):

```bash
# 15B BF16 (256 or 512 by adjusting JOB_TOTAL_GPUS)
source /mnt/vast/johnson/llmb_venv/bin/activate
cd /mnt/vast/johnson/dgxc-benchmarking/nemotron4-15b
MAX_STEPS=10 \
  RUN_CONF_IMAGE=/mnt/vast/johnson/llmb/images/nvidia+nemo+26.02.00.sqsh \
  LLMB_INSTALL=/mnt/vast/johnson/llmb \
  JOB_TOTAL_GPUS=256 \
  GPU_TYPE=b200 DTYPE=bf16 \
  SBATCH_ACCOUNT=root SBATCH_PARTITION=batch \
  ADDITIONAL_SLURM_PARAMS='job-name=B_n15b_bf16_256' \
  bash launch.sh

# 15B FP8 — NOTE 25.09 container (26.02 + compat_runner breaks with cudaErrorInvalidValue)
source /mnt/vast/johnson/llmb_venv/bin/activate
cd /mnt/vast/johnson/dgxc-benchmarking/nemotron4-15b
MAX_STEPS=10 \
  RUN_CONF_IMAGE=/mnt/vast/johnson/llmb/images/nvidia+nemo+25.09.00.sqsh \
  LLMB_INSTALL=/mnt/vast/johnson/llmb \
  JOB_TOTAL_GPUS=512 \
  GPU_TYPE=b200 DTYPE=fp8 \
  SBATCH_ACCOUNT=root SBATCH_PARTITION=batch \
  ADDITIONAL_SLURM_PARAMS='job-name=B_n15b_fp8_512' \
  bash launch.sh

# Qwen3 235B BF16 or FP8
source /mnt/vast/johnson/llmb_venv/bin/activate
cd /mnt/vast/johnson/dgxc-benchmarking/qwen3/pretrain
MAX_STEPS=10 \
  RUN_CONF_IMAGE=/mnt/vast/johnson/llmb/images/nvidia+nemo+26.02.00.sqsh \
  LLMB_INSTALL=/mnt/vast/johnson/llmb \
  MODEL_SIZE=235b \
  JOB_TOTAL_GPUS=512 \
  GPU_TYPE=b200 DTYPE=bf16 \
  SBATCH_ACCOUNT=root SBATCH_PARTITION=batch \
  ADDITIONAL_SLURM_PARAMS='job-name=B_qwen3_235b_bf16_512' \
  bash launch.sh

# DeepSeek V3 BF16
source /mnt/vast/johnson/llmb_venv/bin/activate
cd /mnt/vast/johnson/dgxc-benchmarking/deepseek_v3/pretrain/megatron_bridge
MAX_STEPS=10 \
  RUN_CONF_IMAGE=/mnt/vast/johnson/llmb/images/nvidia+nemo+26.02.00.sqsh \
  LLMB_INSTALL=/mnt/vast/johnson/llmb \
  JOB_TOTAL_GPUS=512 \
  GPU_TYPE=b200 DTYPE=bf16 \
  SBATCH_ACCOUNT=root SBATCH_PARTITION=batch \
  ADDITIONAL_SLURM_PARAMS='job-name=B_dsv3_bf16_512' \
  bash launch.sh
```

## Critical Cluster-Specific Patches

The wrapper (`submit_session_B.sh`) applies these patches to every generated sbatch. If running `launch.sh` directly, you must apply them manually or the job WILL fail.

1. **Strip auto-generated `#SBATCH --job-name=root-root...`** (NeMo Run emits two; last wins)
2. **Add `#SBATCH --exclude=use3a-ss-b200-gpu-[130,190,197,199,201,211,233,239]`** (known-bad nodes)
3. **Prepend block before `# Command 1`:**
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
4. **Extend (or insert when missing) `--container-env=` on the `srun` line** with:
   `HOME,NEMO_NLP_TMP,HF_HOME,HF_HUB_OFFLINE,TRANSFORMERS_OFFLINE,HF_TOKEN,NCCL_SOCKET_IFNAME,TORCH_NCCL_HIGH_PRIORITY,NVTE_FWD_LAYERNORM_SM_MARGIN,NVTE_BWD_LAYERNORM_SM_MARGIN`
   - Megatron-Bridge launchers (qwen3, dsv3) emit NO `--container-env=` at all — must INSERT, not just extend
5. **Add `--no-container-mount-home`** to srun (usually already present; verify)
6. **For 15B only:** wrap entrypoint with `compat_runner.py` (copied from a successful Apr 15 run):
   - Source: `/mnt/vast/johnson/llmb/workloads/pretrain_nemotron4-15b/experiments/pretrain_nemotron4_15b_bf16_gpus256_tp1_pp1_cp1_vp1_mbs2_gbs1024/pretrain_nemotron4_15b_bf16_gpus256_tp1_pp1_cp1_vp1_mbs2_gbs1024_1776133099/pretrain_nemotron4_15b_bf16_gpus256_tp1_pp1_cp1_vp1_mbs2_gbs1024/compat_runner.py`
   - Skip for 15B FP8 (native 25.09 container is compatible)

## Log File Locations

Each job's training log is alongside its sbatch:
```
<experiment_dir>/<run_id>/<run_id>/log-root-root.<run_id>_<JOB_ID>_0.out
```

Quick TFLOPS extraction:
```bash
# 15B (NeMo format): TFLOPS_per_GPU
grep "TFLOPS_per_GPU" <log> | tail -5
# Qwen3/DSV3 (Megatron-Bridge format): Step Time
grep "Step Time" <log>
```

## Known Issues Encountered (all resolved)

- **Qwen3 BF16 512 Job 84458 failed**: `LocalEntryNotFoundError: Qwen/Qwen3-235B-A22B` — MB sbatch had no `--container-env=`, so HF env vars weren't forwarded into container → offline cache lookup fell back to online mode. **Fix**: patch now INSERTS `--container-env=` when missing. Passed on retry (84466).
- **Qwen3 BF16 512 Job 84462 failed (same error)**: idempotency check for the pyxis-workaround block was matching a pre-existing `export HF_HUB_OFFLINE=1` line from NeMo Run, causing the patch to skip. **Fix**: changed marker to unique `# --- pyxis HOME + NCCL socket workaround`.
- **15B FP8 512 Job 84450 failed**: `cudaErrorInvalidValue` in `replay_graph_capture` — 26.02 container + compat_runner's TransformerConfig shim breaks CUDA-graph FP8 path. **Fix**: use 25.09 container for 15B FP8 (26.02 not needed there — 25.09 supports FP8 natively for this recipe).

## Session B scoreboard

```
15B BF16 256:       1,492 TFLOP/s/GPU (84443) ✅
15B BF16 512:       1,376 TFLOP/s/GPU (84447) ✅
15B FP8  512:       1,531 TFLOP/s/GPU (84455) ✅
Qwen3 235B BF16:    604.7 TFLOP/s/GPU (84466) ✅ +16.8% vs 256
Qwen3 235B FP8:     499.2 TFLOP/s/GPU (84469) ✅ +28.5% vs 256
DSV3 BF16:          554.9 TFLOP/s/GPU (84479) ✅ (first BF16 result for DSV3)
```
