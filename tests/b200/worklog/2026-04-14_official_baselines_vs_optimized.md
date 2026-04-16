# NVIDIA Official B200 Baseline Configurations

Reference configs from [NVIDIA/dgxc-benchmarking](https://github.com/NVIDIA/dgxc-benchmarking).
Each `official/` directory contains a `run.sh` that submits the exact NVIDIA reference config.

## How to Run

```bash
# Megatron-Bridge models (via llmb-run):
bash tests/b200/slurm/llama3.1-405b/nvfp4/official/256gpu/run.sh

# Add --dry-run to preview without submitting:
bash tests/b200/slurm/llama3.1-405b/nvfp4/official/256gpu/run.sh --dry-run

# NeMo2 models (Nemotron4):
cd /mnt/vast/johnson/dgxc-benchmarking/nemotron4-15b
JOB_TOTAL_GPUS=256 GPU_TYPE=b200 DTYPE=fp8 bash launch.sh
```

## Official Configs vs Your Optimized Results

### Nemotron4 15B

| Precision | GPUs | Official Config | Official GBS | Your Best GBS | Your TFLOP/s | Official TFLOP/s | Delta |
|-----------|------|-----------------|-------------|---------------|--------------|-----------------|-------|
| FP8 | 64 | TP=1,PP=1,CG=on | 256 | 256 (CG off) | 1,916 (81053) | — | same config* |
| FP8 | 256 | TP=1,PP=1,CG=on | **1024** | **2048** | 1,904 (81072) | ~1,800 (81060) | **+6%** |
| BF16 | 64 | TP=1,PP=1,CG=on | 256 | 256 | 1,571 (81054) | — | same config |
| BF16 | 256 | TP=1,PP=1,CG=on | 1024 | 1024 | 1,439 (81056) | — | same config |

*CUDA graphs disabled in your runs due to FP8 tensor copy bug in 26.02 container.

### Llama 3.1 405B

| Precision | GPUs | Official Config | Your Best Config | Your TFLOP/s | Official TFLOP/s | Delta |
|-----------|------|-----------------|-----------------|--------------|-----------------|-------|
| NVFP4 | 256 | TP=4,**PP=16**,VP=8 | TP=4,**PP=8**,VP=4 | 2,006 (80076) | 1,564 (80073) | **+28%** |
| NVFP4 | 512 | TP=4,PP=16,VP=8 | TP=4,PP=8,VP=4 | 1,818 (79953) | 1,384 (79952) | +31% |
| FP8 | 256 | TP=4,PP=8,**CP=2**,VP=8 | — | NOT RUN | — | — |

### Llama 3.1 70B

| Precision | GPUs | Official Config | Your Config | Your TFLOP/s | Notes |
|-----------|------|-----------------|-------------|--------------|-------|
| FP8 | 64 | TP=1,PP=1,FSDP | TP=1,PP=1 | 1,590 (79120) | **matches official** |
| FP8 | 256 | **TP=2,PP=4,VP=5** | TP=1,PP=1 | 1,494 (79139) | **different config** |
| FP8 | 512 | **TP=2,PP=4,VP=5** | TP=1,PP=1 | 1,365 (79133) | **different config** |
| NVFP4 | 512 | TP=2,PP=4,VP=5 | — | — | not compared |

### Qwen3 235B

| Precision | GPUs | Official Config | Your Config | Your TFLOP/s | Notes |
|-----------|------|-----------------|-------------|--------------|-------|
| BF16 | 256 | TP=1,PP=8,EP=8,VP=4 | same | 514 (80077) | **matches official exactly** |

### Qwen3 30B

| Precision | GPUs | Official Config | Your Config | Notes |
|-----------|------|-----------------|-------------|-------|
| BF16 | 64 | TP=1,PP=1,EP=8 | FP8 MX only | official is BF16 only |

### DeepSeek-V3 671B

| Precision | GPUs | Official Config | Your Config | Notes |
|-----------|------|-----------------|-------------|-------|
| BF16 | 512 | TP=1,PP=16,EP=8 | same | **both failed NCCL timeout** |

## Key Differences Summary

1. **Nemotron4 15B FP8 256GPU**: Official GBS=1024 vs your GBS=2048 (+6% gain)
2. **Llama 405B NVFP4**: Official PP=16 vs your PP=8 (+28% gain)
3. **Llama 70B FP8 256+GPU**: Official uses TP=2/PP=4/VP=5; you ran TP=1/PP=1 (not yet compared)
4. **Llama 405B FP8**: Official config (TP=4,PP=8,CP=2) not yet run
5. **Qwen3 235B BF16**: Your config matches official exactly
6. **DeepSeek-V3**: Your config matches official; both fail on this cluster

## Container Notes

| Model | Official Container | Cluster Container | Notes |
|-------|-------------------|-------------------|-------|
| Nemotron4 15B | 25.09.00 | 26.02.00 + compat_runner.py | API patches needed |
| Llama 3.1 | 26.02.00 | 26.02.00 | matches |
| Qwen3 | 26.02.00 | 26.02.00 | matches |
| DeepSeek-V3 | 26.02.00 | 26.02.00 | matches |

Available images on cluster: `/mnt/vast/johnson/llmb/images/`
- nvidia+nemo+25.07.01.sqsh
- nvidia+nemo+25.09.00.sqsh
- nvidia+nemo+26.02.00.sqsh
