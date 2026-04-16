# Nemotron4 15B FP8 — 64 GPU Benchmark: Context & Status

## Objective

Run the NVIDIA Nemotron4 15B pretrain benchmark in **FP8** precision on **64 GPUs (8 nodes)** on the Together AI B200 Slurm cluster. This is part of DGXC benchmarking to validate cluster performance against NVIDIA reference numbers.

## Current Status: COMPLETE (job 81053)

All 50 training steps completed successfully. CUDA graphs disabled due to FP8 tensor copy bug.

### Results

| Metric | Value |
|--------|-------|
| Job ID | 81053 |
| Steps completed | 50/50 |
| Steady-state TFLOPS/GPU (steps 3-49) | **1,845 – 1,926** |
| Median TFLOPS/GPU (steps 3-49) | **~1,895** |
| Step time (steady-state) | 0.76 – 0.79s |
| Peak memory reserved | 78.9 GB |
| Peak memory allocated | 78.7 GB |
| Final loss | 5.551 |
| CUDA graphs | **Disabled** (FP8 cudaErrorInvalidValue on replay) |
| Container | `nvidia+nemo+26.02.00.sqsh` |

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Nemotron4 15B |
| Precision | FP8 (hybrid, tensorwise recipe) |
| GPUs | 64 (8 nodes x 8 GPUs/node) |
| TP / PP / CP / VP / EP | 1 / 1 / 1 / 1 / 1 |
| MBS / GBS | 2 / 256 |
| Seq Length | 4096 |
| Max Steps | 50 |
| Container | `nvidia+nemo+26.02.00.sqsh` (upgraded from 25.09) |
| Tokenizer | `nvidia/Nemotron-4-340B-Base` (HuggingFace, cached) |
| Data | MockDataModule (synthetic) |
| Optimizer | Adam (distributed), lr=4.5e-5, weight_decay=0.1 |

## Key File Paths

- **Sbatch script (user-created):** `~/johnson/scripts/nemotron4_15b/64gpus_fp8/sbatch.sh`
- **NeMoRun experiment dir:** `/mnt/vast/johnson/llmb/workloads/pretrain_nemotron4-15b/experiments/pretrain_nemotron4_15b_fp8_gpus64_tp1_pp1_cp1_vp1_mbs2_gbs256/pretrain_nemotron4_15b_fp8_gpus64_tp1_pp1_cp1_vp1_mbs2_gbs256_1776121660/`
- **Working dir (mounted as /nemo_run):** `...pretrain_nemotron4_15b_fp8_gpus64_tp1_pp1_cp1_vp1_mbs2_gbs256/` (inside the above)
- **Training config:** `configs/pretrain_nemotron4_15b_fp8_gpus64_tp1_pp1_cp1_vp1_mbs2_gbs256_config.yaml`
- **Training script:** `scripts/pretrain_nemotron4_15b_fp8_gpus64_tp1_pp1_cp1_vp1_mbs2_gbs256.sh`
- **Compat wrapper:** `compat_runner.py` (monkey-patches for 25.09->26.02 compat)
- **Container image:** `/mnt/vast/johnson/llmb/images/nvidia+nemo+26.02.00.sqsh`
- **HF cache:** `/mnt/vast/llmb_/.cache/huggingface`
- **Log (successful):** `.../log-nemotron4_15b_fp8_64gpu_81053_0.out`

## Job History (all on 2026-04-13)

| Job ID | Status | Notes |
|--------|--------|-------|
| 81028 | FAILED | Initial NeMoRun-generated sbatch |
| 81030 | FAILED | Tokenizer offline loading failure |
| 81032 | FAILED | Tokenizer + tensorstore import error on teardown |
| 81044 | FAILED | `no_weight_decay_cond` optimizer TypeError (25.09 container) |
| 81046 | FAILED | PermissionError on `/mnt/vast/llmb_/.cache/huggingface/nemo_nlp_tmp` |
| 81048 | FAILED | tensorstore ImportError in 26.02 container (first fix attempt) |
| 81049 | FAILED | tensorstore ImportError (different fix approach) |
| 81051 | FAILED | CUDA error: invalid argument on FP8 graph replay (step 1) |
| **81053** | **SUCCESS** | **All 50 steps, ~1,895 TFLOPS/GPU median** |

## Fixes Applied in compat_runner.py

The `compat_runner.py` wrapper applies three monkey-patches before running the NeMo fdl_runner:

1. **Tensorstore stub** — Creates a dummy `megatron.core.dist_checkpointing.strategies.tensorstore` module in `sys.modules` to prevent ImportError (module missing from 26.02 Megatron-Core build, but NeMo imports it at module level).

2. **CUDA graph disable** — Patches `TransformerConfig.__post_init__` to force `enable_cuda_graph = False`. The 25.09 config enables CUDA graphs, but FP8 quantized tensor copy fails with `cudaErrorInvalidValue` during graph replay in 26.02.

3. **Optimizer kwarg stripping** — Patches `get_megatron_optimizer()` to strip kwargs not in its signature (e.g. `no_weight_decay_cond` passed by NeMo but rejected by some Megatron-Core versions).

## Sbatch Script Fixes

- Added `export NEMO_NLP_TMP=/tmp/nemo_nlp_tmp` to redirect NeMo's mkdir away from read-only shared HF cache
- Switched container from `25.09.00` to `26.02.00` to fix NeMo/Megatron-Core API mismatch
- Added `NEMO_NLP_TMP` to `--container-env` list

## Potential Improvements

- **Re-enable CUDA graphs:** The ~1,895 TFLOPS/GPU is without CUDA graphs. Enabling them could improve throughput significantly but requires fixing the FP8 tensor copy bug (may need a newer container or TransformerEngine patch).
- **BF16 sibling job:** Same benchmark in BF16 — scripts at `~/johnson/scripts/nemotron4_15b/64gpus_bf16/sbatch.sh`.

## Cluster Environment Notes

- **Cluster:** Together AI B200 (Slurm-managed)
- **Partition:** `batch`
- **Network:** IB with bond0, 8x mlx5 HCAs
- **NCCL env:** NVLS disabled, IB HCAs explicitly set, `CUDA_DEVICE_MAX_CONNECTIONS=32`
- **Pyxis:** `HOME=/tmp` and `--no-container-mount-home` required (known pyxis home dir issue)
- **SHARP:** Not available (sharpd not running on this cluster)
- **Available containers:** `25.07.01`, `25.09.00`, `26.02.00` (all at `/mnt/vast/johnson/llmb/images/`)
