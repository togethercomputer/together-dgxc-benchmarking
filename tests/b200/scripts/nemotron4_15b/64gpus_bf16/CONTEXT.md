# Nemotron4 15B BF16 — 64 GPU Benchmark: Context & Status

## Objective

Run the NVIDIA Nemotron4 15B pretrain benchmark in **BF16** precision on **64 GPUs (8 nodes)** on the Together AI B200 Slurm cluster. This is part of DGXC benchmarking to validate cluster performance against NVIDIA reference numbers.

## Current Status: BLOCKED — Multiple distinct errors across runs

The job has been submitted ~7 times. No run has completed successfully. Each attempt has hit a different error, suggesting iterative fixes were applied between runs but new issues surfaced each time.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Nemotron4 15B |
| Precision | BF16 (bf16-mixed) |
| GPUs | 64 (8 nodes × 8 GPUs/node) |
| TP / PP / CP / VP / EP | 1 / 1 / 1 / 1 / 1 |
| MBS / GBS | 2 / 256 |
| Seq Length | 4096 |
| Max Steps | 50 |
| Container | `nvidia+nemo+25.09.00.sqsh` |
| Tokenizer | `nvidia/Nemotron-4-340B-Base` (HuggingFace, cached) |
| Data | MockDataModule (synthetic) |
| Optimizer | Adam (distributed), lr=4.5e-5, weight_decay=0.1, `use_precision_aware_optimizer: true` |

**Key difference from FP8 config:** BF16 has `use_precision_aware_optimizer: true` (FP8 has `false`) and no FP8-specific plugin fields (`fp8`, `fp8_recipe`, etc.).

## Key File Paths

- **Sbatch script (user-created):** `~/johnson/scripts/nemotron4_15b/64gpus_bf16/sbatch.sh`
- **NeMoRun experiment dir:** `/mnt/vast/johnson/llmb/workloads/pretrain_nemotron4-15b/experiments/pretrain_nemotron4_15b_bf16_gpus64_tp1_pp1_cp1_vp1_mbs2_gbs256/pretrain_nemotron4_15b_bf16_gpus64_tp1_pp1_cp1_vp1_mbs2_gbs256_1776121405/`
- **Working dir (mounted as /nemo_run):** `.../pretrain_nemotron4_15b_bf16_gpus64_tp1_pp1_cp1_vp1_mbs2_gbs256/` (inside the above)
- **Training config:** `configs/pretrain_nemotron4_15b_bf16_gpus64_tp1_pp1_cp1_vp1_mbs2_gbs256_config.yaml`
- **Training script:** `scripts/pretrain_nemotron4_15b_bf16_gpus64_tp1_pp1_cp1_vp1_mbs2_gbs256.sh`
- **Compat wrapper:** `compat_runner.py` (patches `enable_cuda_graph` → `cuda_graph_impl` for 25.09→26.02 API change)
- **Container image:** `/mnt/vast/johnson/llmb/images/nvidia+nemo+25.09.00.sqsh`
- **HF cache:** `/mnt/vast/llmb_/.cache/huggingface`

## Job History (all on 2026-04-13)

| Job ID | Error | Notes |
|--------|-------|-------|
| 81026 | Pyxis container startup failure (`spank_pyxis.so: task_init() failed`) | First attempt; NeMoRun-generated sbatch; pyxis couldn't start container on some nodes |
| 81027 | (sbatch output only, same pyxis issue likely) | |
| 81029 | Tokenizer: can't load `nvidia/Nemotron-4-340B-Base` offline | HF cache not reachable inside container |
| 81031 | Same tokenizer offline error | |
| 81038 | Same tokenizer offline error | |
| 81043 | `get_megatron_optimizer() got unexpected keyword 'no_weight_decay_cond'` | NeMo/Megatron API mismatch — **same error as FP8 job 81044** |
| 81045 | `ModuleNotFoundError: No module named 'netrc'` | New error in aiohttp import chain during Fiddle deserialization |

## Error Details

### Error 1: Pyxis container startup (job 81026)

**Root cause:** `spank_pyxis.so: task_init() failed with rc=-1` and `pyxis: couldn't start container`. The NeMoRun-generated sbatch script likely didn't have `--no-container-mount-home` or `HOME=/tmp` set. Later user-created sbatch scripts include these fixes.

**Status:** Likely fixed in the user-created sbatch script.

### Error 2: Tokenizer offline loading (jobs 81029, 81031, 81038)

**Root cause:** `TRANSFORMERS_OFFLINE=1` is set, but the tokenizer for `nvidia/Nemotron-4-340B-Base` can't be found in the mounted HF cache. The container mount for the HF cache may not have been correctly configured, or the tokenizer files were missing from cache.

**Status:** Likely fixed in later runs (81043 got past tokenizer loading). The HF cache is at `/mnt/vast/llmb_/.cache/huggingface` and should contain `hub/models--nvidia--Nemotron-4-340B-Base/`.

### Error 3: `no_weight_decay_cond` optimizer error (job 81043)

**Root cause:** NeMo's `MegatronOptimizerModule` passes `no_weight_decay_cond` to Megatron-Core's `get_megatron_optimizer()`, but the Megatron-Core version in the 25.09 container doesn't accept this kwarg.

**Traceback path:**
```
nemo.lightning.pytorch.optim.megatron:110 → setup_megatron_optimizer()
→ nemo.lightning._strategy_lib:712 → get_megatron_optimizer()
→ TypeError: got unexpected keyword argument 'no_weight_decay_cond'
```

**This is the same error that the FP8 job 81044 hit.** It's a container-level NeMo/Megatron-Core API incompatibility.

**Fix options:**
1. Try a different container — `26.02.00` is available at `/mnt/vast/johnson/llmb/images/nvidia+nemo+26.02.00.sqsh` (but may introduce new issues; the `compat_runner.py` already handles one such API change)
2. Monkey-patch `get_megatron_optimizer` in `compat_runner.py` to strip `no_weight_decay_cond` before calling, similar to the existing cuda_graph patch
3. Investigate whether the NeMoRun version that generated this config is mismatched with the container

### Error 4: `No module named 'netrc'` (job 81045)

**Root cause:** During Fiddle deserialization, importing `Nemotron4Config15B` triggers a long import chain:
```
nemo.collections.llm → bert/data → specter.py → datasets → aiohttp → helpers.py → import netrc
→ ModuleNotFoundError: No module named 'netrc'
```

The `netrc` module is part of Python's standard library, so this suggests the container's Python environment is broken or `HOME=/tmp` is interfering with Python's module resolution. This is a new error that appeared after the earlier tokenizer/optimizer issues.

**Fix options:**
1. Check if `HOME=/tmp` causes Python to skip standard library paths — try setting `PYTHONHOME` or ensuring `/usr/lib/python3.12/netrc.py` exists in the container
2. The `netrc` import comes from `aiohttp` via `datasets` via NeMo's BERT data module — this is an unnecessary import chain for Nemotron4 pretraining. A targeted fix might pre-create a dummy `netrc` module or set `NETRC` env var
3. Switching to the `26.02.00` container may also resolve this if the Python environment is better packaged there

## Cluster Environment Notes

- **Cluster:** Together AI B200 (Slurm-managed)
- **Partition:** `batch`
- **Network:** IB with bond0, 8× mlx5 HCAs
- **NCCL env:** NVLS disabled, IB HCAs explicitly set, `CUDA_DEVICE_MAX_CONNECTIONS=32`
- **Pyxis:** `HOME=/tmp` and `--no-container-mount-home` required (known pyxis home dir issue)
- **SHARP:** Not available (sharpd not running on this cluster)
- **Available containers:** `25.07.01`, `25.09.00`, `26.02.00` (all at `/mnt/vast/johnson/llmb/images/`)

## Recommended Next Steps

1. **Fix the `netrc` error (most recent blocker):** Investigate whether `HOME=/tmp` breaks Python stdlib module resolution. Try adding `PYTHONDONTWRITEBYTECODE=1` or ensuring the `netrc` module is importable. Alternatively, switching to the `26.02.00` container may resolve this.
2. **Fix the optimizer error:** Either switch containers or extend `compat_runner.py` to also patch out the `no_weight_decay_cond` kwarg from `get_megatron_optimizer()`.
3. **Both errors need fixing** since the `netrc` error (81045) and the optimizer error (81043) appeared in consecutive runs — fixing one will expose the other.
4. **Resubmit** with `sbatch ~/johnson/scripts/nemotron4_15b/64gpus_bf16/sbatch.sh`
5. **Verify success** by checking for training step logs showing TFLOP/s/GPU in the output.

## FP8 Sibling Job

There is a parallel FP8 job with the same base config (TP=1/PP=1, 64 GPUs). It shares the `no_weight_decay_cond` optimizer error and additionally has a `PermissionError` on `/mnt/vast/llmb_/.cache/huggingface/nemo_nlp_tmp`. Context at `~/johnson/scripts/nemotron4_15b/64gpus_fp8/CONTEXT.md`.
