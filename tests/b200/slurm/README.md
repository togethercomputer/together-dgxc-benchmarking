# B200 Slurm Benchmark Results

Pretraining benchmark scripts and results on Together AI B200 cluster (74 nodes, 592 GPUs).

**Cluster:** Together AI B200 DGXC
- 74 nodes, 8x NVIDIA B200 per node, 592 GPUs total
- 8x ConnectX-7 IB 400Gb/s NICs per node
- Container: `nvidia+nemo+26.02.00` (enroot/pyxis)
- Scheduler: Slurm with Pyxis plugin
- Storage: `/mnt/vast/` (shared), `/home/johnson/` (login only)

**Date range:** 2026-04-08 to 2026-04-17

## Running Benchmarks

Use `run_all_official.sh` to submit all official NVIDIA baseline benchmarks:

```bash
# Run specific tiers
bash run_all_official.sh --tier1              # 64-GPU jobs (8 nodes each, concurrent)
bash run_all_official.sh --tier2              # 256-GPU jobs (32 nodes each, concurrent)
bash run_all_official.sh --tier3              # 512-GPU jobs (64 nodes each, sequential)
bash run_all_official.sh --tier1 --tier2      # 64+256 GPU (default if no tier specified)

# Preview without submitting
bash run_all_official.sh --tier3 --dry-run
```

All jobs run MAX_STEPS=10, measuring TFLOPS/GPU from iterations 3-9 (skipping compilation + warmup).

### Tier 1: 64-GPU (8 nodes)

| Model | Dtype | Submission |
|-------|-------|------------|
| Llama 3.1 70B | FP8 | `llmb-run` (26.02) |
| Nemotron-H 56B | FP8 | `llmb-run` (26.02) |
| Qwen3 30B | BF16 | `llmb-run` (26.02) |
| Nemotron4 15B | BF16 | `launch.sh` (26.02) |
| Nemotron4 15B | FP8 | `launch.sh` (26.02) |

### Tier 2: 256-GPU (32 nodes)

| Model | Dtype | Submission |
|-------|-------|------------|
| Llama 3.1 70B | FP8 | `llmb-run` (26.02) |
| Llama 3.1 405B | FP8 | `llmb-run` (26.02) |
| Llama 3.1 405B | NVFP4 | `llmb-run` (26.02) |
| DeepSeek-V3 | FP8 | `llmb-run` (26.02) |
| Nemotron-H 56B | FP8 | `llmb-run` (26.02) |
| Qwen3 235B | BF16 | `llmb-run` (26.02) |
| Nemotron4 15B | BF16 | `launch.sh` (26.02) |
| Nemotron4 15B | FP8 | `launch.sh` (26.02) |

### Tier 3: 512-GPU (64 nodes, sequential)

Each job uses the full cluster. They run one at a time.

| Model | Dtype | Submission | Notes |
|-------|-------|------------|-------|
| Llama 3.1 70B | FP8 | `llmb-run` (26.02) | |
| Llama 3.1 70B | NVFP4 | `llmb-run` (26.02) | |
| Llama 3.1 405B | FP8 | `llmb-run` (26.02) | |
| Llama 3.1 405B | NVFP4 | `llmb-run` (26.02) | |
| Nemotron-H 56B | FP8 | `llmb-run` (26.02) | |
| Nemotron4 340B | FP8 | `launch.sh` (25.07) | Auto-patched |
| Nemotron4 340B | BF16 | `launch.sh` (25.07) | Auto-patched |
| Grok1 314B | BF16 | `launch.sh` (25.09) | Auto-patched |
| Grok1 314B | FP8 | `launch.sh` (25.09) | Auto-patched |
| DeepSeek-V3 | FP8 | `llmb-run` (26.02) | BF16 variant known to NCCL-timeout |

## 512-GPU Results (2026-04-17)

| Model | Dtype | Job ID | TFLOPS/GPU | 256-GPU Baseline | Delta |
|-------|-------|--------|-----------|-----------------|-------|
| Llama 3.1 70B | FP8 | 84245 | 1,526 | 1,546 | -1.3% |
| Llama 3.1 70B | NVFP4 | 84246 | 2,378 | 1,943 | +22.4% |
| Llama 3.1 405B | FP8 | 84247 | 1,766 | 1,724 | +2.4% |
| Llama 3.1 405B | NVFP4 | 84248 | 1,789 | 1,746 | +2.5% |
| Nemotron-H 56B | FP8 | 84249 | 1,502 | 1,529 | -1.8% |
| Nemotron4 340B | FP8 | 84272 | 1,234 | 1,244 | -0.8% |
| Nemotron4 340B | BF16 | 84275 | 860 | 866 | -0.7% |
| Grok1 314B | BF16 | 84278 | 1,004 | 994 | +1.0% |
| Grok1 314B | FP8 | 84279 | 1,454 | 1,219 | +19.3% |
| DeepSeek-V3 | FP8 | 84255 | 594 | 551 | +7.8% |

Most models scale within +/-2% from 256 to 512 GPUs. Llama 70B NVFP4 (+22.4%) and Grok1 FP8 (+19.3%) show significant improvement at 512, likely due to higher effective GBS.

## 256-GPU Results (2026-04-16)

| Model | Dtype | Job ID | TFLOPS/GPU |
|-------|-------|--------|-----------|
| Llama 3.1 70B | FP8 | 83996 | 1,546 |
| Llama 3.1 70B | NVFP4 | 84023 | 1,943 |
| Llama 3.1 405B | FP8 | 83997 | 1,724 |
| Llama 3.1 405B | NVFP4 | 84023 | 1,746 |
| Nemotron-H 56B | FP8 | 83998 | 1,529 |
| Nemotron4 340B | FP8 | 84064 | 1,244 |
| Nemotron4 340B | BF16 | 84065 | 866 |
| Grok1 314B | BF16 | 84056 | 994 |
| Grok1 314B | FP8 | 84061 | 1,219 |
| DeepSeek-V3 | FP8 | 84003 | 551 |
| DeepSeek-V3 | BF16 | 84043 | 381 |
| Qwen3 235B | BF16 | 84025 | 516 |
| Nemotron4 15B | FP8 | 83999 | 1,916 |
| Nemotron4 15B | BF16 | 84000 | 1,432 |
| Qwen3 30B | BF16 | 83994 | 1,520 |

## Architecture: Two Submission Paths

### 1. `llmb-run` models (26.02 container)

Models with native Megatron-LM Bridge support. Uses the 26.02 NeMo container which is compatible with the host Slurm/PMIx stack.

```bash
cd /mnt/vast/johnson/llmb
./llmb-run submit -w pretrain_llama3.1 -s 70b -d fp8 --scale 512
```

The `submit_llmb()` function in `run_all_official.sh` handles submission and post-patches `max_steps` from 50 to 10 in the generated training script.

### 2. `launch.sh` models (legacy 25.07/25.09 containers)

Nemotron4 340B and Grok1 require older NeMo containers (25.07 and 25.09 respectively) which have a PMIx v3/v4 mismatch with the host Slurm.

These models use `dgxc-benchmarking/launch.sh` which generates a NeMo experiment and sbatch script via `fdl_runner`. The generated scripts need patching to work on this cluster.

`run_all_official.sh` handles this automatically:
1. Calls `launch.sh` with `TP_COMM_OVERLAP=False` (baked into serialized config)
2. Cancels the auto-submitted (unpatched) job
3. Applies `patch_legacy_sbatch()` which adds:
   - `HOME=/tmp`, `NEMO_NLP_TMP=/tmp` (pyxis HOME workaround)
   - `HF_HOME=/mnt/vast/johnson/llmb/.cache/huggingface` (tokenizer cache)
   - `NCCL_SOCKET_IFNAME=bond0` (correct IB interface)
   - `PMIX_MCA_gds=hash`, `OMPI_MCA_plm=isolated` (PMIx workarounds)
   - `MPI4PY_RC_INITIALIZE=false`, `MPI4PY_RC_THREADS=false`, `MPI4PY_RC_FINALIZE=false`
   - `LD_PRELOAD=/mpi_stub/libmpi_stub.so` + mpi_stub container mount (MPI stub bypass)
4. Resubmits the patched script

These models also require `llmb_venv` (for the `fiddle` dependency):
```bash
source /mnt/vast/johnson/llmb_venv/bin/activate
```

## Directory Structure

```
slurm/
├── run_all_official.sh          # Master submission script (all tiers)
├── OFFICIAL_BASELINES.md        # NVIDIA reference numbers
├── README.md                    # This file
├── llama3.1-70b/
│   ├── fp8/                     # 64-512 GPU
│   └── nvfp4/                   # 512 GPU
├── llama3.1-405b/
│   ├── fp8/                     # 256+512 GPU
│   └── nvfp4/                   # 256+512 GPU, PP optimization
├── nemotron-h/
│   └── fp8/                     # 64-512 GPU
├── nemotron4-15b/
│   ├── bf16/                    # 64+256 GPU
│   └── fp8/                     # 64+256 GPU
├── nemotron4-340b/
│   ├── bf16/                    # 256+512 GPU
│   └── fp8/                     # 256+512 GPU
├── grok1/
│   ├── bf16/                    # 256+512 GPU
│   └── fp8/                     # 256+512 GPU
├── deepseek-v3/
│   ├── bf16/                    # 256 GPU (512 NCCL timeout)
│   └── fp8/                     # 256+512 GPU
├── qwen3-235b/
│   └── bf16/                    # 256 GPU
└── qwen3-30b/
    └── bf16/                    # 64 GPU
```

Each model directory contains:
- `<dtype>/official/<scale>/run.sh` — standalone launch script for a single config
- `<dtype>/official/<scale>/sbatch.sh` — direct sbatch script (where applicable)

## Known Cluster Issues

1. **Pyxis HOME mount:** Container cannot create `/home/johnson`. Must use `--no-container-mount-home` + `export HOME=/tmp` before srun.
2. **PMIx v3/v4 mismatch:** Legacy containers (25.07, 25.09) ship OpenMPI with PMIx v3, host Slurm runs PMIx v4. Requires MPI stub (`LD_PRELOAD`) and `TP_COMM_OVERLAP=False`.
3. **SHARP not available:** `sharpd` not running, no reservations configured — admin setup needed.
4. **HF tokenizer cache:** With `HOME=/tmp`, HuggingFace can't find cached tokenizers. Must set `HF_HOME=/mnt/vast/johnson/llmb/.cache/huggingface`.
5. **NCCL interface:** Must set `NCCL_SOCKET_IFNAME=bond0` for correct IB interface.
6. **NeMo 25.09->26.02 compat:** Use `compat_runner.py` for tensorstore stub, CUDA graph API, optimizer kwargs (Nemotron4 15B only).

## Experiment Logs

All experiments are stored under `/mnt/vast/johnson/llmb/workloads/<model>/experiments/`. Each experiment directory contains:
- `*_sbatch.sh` — the sbatch script used
- `*_sbatch_patched.sh` — patched version (legacy containers)
- `scripts/*.sh` — the inner training script
- `configs/*.yaml` — serialized model config
- `log-*_<jobid>_0.out` — per-task training log with TFLOPS data
- `sbatch_*_<jobid>.out` — sbatch-level output log
