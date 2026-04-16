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

## Consolidated Benchmark Results (256 GPU)

All results on Together AI B200 DGXC cluster, 32 nodes (256 GPUs).

| Model | Dtype | Target | Official config | Optimized | Key optimization technicals |
|-------|-------|-------:|----------------:|----------:|----------------------------|
| DeepSeek V3 671B | BF16 | 500 | 500 | — | No optimization attempted |
| DeepSeek V3 671B | FP8 | 406 | 406 | — | No optimization attempted |
| Grok1 314B | BF16 | 1,025 | 1,025 | — | No optimization attempted |
| Grok1 314B | FP8 | 1,371 | 1,371 | — | No optimization attempted |
| Llama 3.1 70B | FP8 | 983 | 1,624 | — | 64GPU FSDP official config |
| Llama 3.1 70B | NVFP4 | 1,940 | 2,013 | — | 512GPU official config |
| Llama 3.1 405B | FP8 | 1,536 | 1,722 | — | Official config +12% over target |
| Llama 3.1 405B | NVFP4 | 1,883 | 1,564 | **2,006** | PP=16→PP=8, bubble 43%→16%, DP 4→8, MB upgrade +7% |
| Nemotron-H 56B | FP8 | 1,479 | 1,536 | — | 64/256/512GPU official config |
| Nemotron4 15B | BF16 | 1,076 | 1,439 | 1,571 | 26.02 container + compat_runner.py |
| Nemotron4 15B | FP8 | 1,407 | 1,800 | **1,916** | GBS 1024→2048, grad accum hides 24% NCCL time |
| Nemotron4 340B | BF16 | 936 | 936 | — | No optimization attempted |
| Nemotron4 340B | FP8 | 1,101 | 1,101 | — | No optimization attempted |
| Qwen3 235B | BF16 | 556 | 514 | — | -8% below target, IB oversubscription, 43% MFU ceiling |
| Qwen3 235B | FP8 | 420 | 426 | — | CUDA graphs +10% over base FP8 |

All values in TFLOP/s/GPU.

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
| FP8 | 256 | TP=4,PP=8,CP=2,VP=8 | same (official) | **1,722** (82931) | — | official config |

### Llama 3.1 70B

| Precision | GPUs | Official Config | Your Config | Your TFLOP/s | Notes |
|-----------|------|-----------------|-------------|--------------|-------|
| FP8 | 64 | TP=1,PP=1,FSDP | same (official) | **1,624** (83724) | FSDP official config |
| FP8 | 256 | **TP=2,PP=4,VP=5** | TP=1,PP=1 | 1,494 (79139) | **different config** |
| FP8 | 512 | **TP=2,PP=4,VP=5** | TP=1,PP=1 | 1,365 (79133) | **different config** |
| NVFP4 | 512 | TP=2,PP=4,VP=5 | same (official) | **2,013** (83725) | official config |

### Qwen3 235B

| Precision | GPUs | Official Config | Your Config | Your TFLOP/s | Notes |
|-----------|------|-----------------|-------------|--------------|-------|
| BF16 | 256 | TP=1,PP=8,EP=8,VP=4 | same | 514 (80077) | **matches official exactly** |

### Qwen3 30B

| Precision | GPUs | Official Config | Your Config | Your TFLOP/s | Notes |
|-----------|------|-----------------|-------------|--------------|-------|
| BF16 | 64 | TP=1,PP=1,EP=8,ETP=1 | same (official) | **204** (83726) | official config |

### Nemotron-H 56B

| Precision | GPUs | Official Config | Your Config | Your TFLOP/s | Notes |
|-----------|------|-----------------|-------------|--------------|-------|
| FP8 | 64 | TP=2,PP=1,GBS=192,FP8_cs | same (official) | **1,563** (83727) | 5.27s/step |
| FP8 | 256 | TP=2,PP=1,GBS=768,FP8_cs | same (official) | **1,536** (83728) | 5.36s/step |
| FP8 | 512 | TP=2,PP=1,GBS=1536,FP8_cs | same (official) | **1,477** (83731) | 5.58s/step |

### DeepSeek-V3 671B

| Precision | GPUs | Official Config | Your Config | Your TFLOP/s | Notes |
|-----------|------|-----------------|-------------|--------------|-------|
| BF16 | 256 | TP=1,PP=16,EP=8 | same (official) | **500.22** | |
| FP8 | 256 | TP=1,PP=16,EP=8,FP8_mx | same (official) | **405.59** | |
| BF16 | 512 | TP=1,PP=16,EP=8 | same | — | **NCCL timeout** |
| FP8 | 512 | TP=1,PP=16,EP=8,FP8_mx | same (official) | — (83732) | **NCCL timeout** (641s on PP group) |

**Note**: 512-GPU configs fail with NCCL timeout. PP=16 requires NCCL P2P across 16 pipeline
stages spanning many nodes. NCCL falls back to socket transport for some P2P connections, but
the socket IPs (7.247.226.x subnet) are not routable on this cluster — only IB IPs (7.247.232.x)
work. 256-GPU configs completed successfully.

### Nemotron4 340B

| Precision | GPUs | Official Config | Your Config | Your TFLOP/s | Notes |
|-----------|------|-----------------|-------------|--------------|-------|
| FP8 | 256 | TP=8,PP=4,VP=12,GBS=64,CG=on | same (official) | **1,101.1** | MPI stub required (PMIx v3/v4) |
| BF16 | 256 | TP=8,PP=4,VP=12,GBS=64,CG=on | same (official) | **936.33** | MPI stub required (PMIx v3/v4) |

Container: nvidia+nemo+25.07.01 (NeMo2 framework, separate from Megatron-Bridge models)

### Grok1 314B

| Precision | GPUs | Official Config | Your Config | Your TFLOP/s | Notes |
|-----------|------|-----------------|-------------|--------------|-------|
| BF16 | 256 | TP=4,PP=4,EP=8,VP=8,GBS=512 | same (official) | **1,025.4** | MPI stub required (PMIx v3/v4) |
| FP8 | 256 | TP=4,PP=4,EP=8,VP=8,GBS=512,FP8_cs | same (official) | **1,370.8** | MPI stub required (PMIx v3/v4) |

Container: nvidia+nemo+25.09.00 (NeMo2 framework)

## Key Differences Summary

1. **Nemotron4 15B FP8 256GPU**: Official GBS=1024 vs your GBS=2048 (+6% gain)
2. **Llama 405B NVFP4**: Official PP=16 vs your PP=8 (+28-31% gain)
3. **Llama 405B FP8 256GPU**: Official config (TP=4,PP=8,CP=2,VP=8) → 1,722 TFLOP/s
4. **Llama 70B FP8 64GPU**: Official FSDP config → 1,624 TFLOP/s (+2% over non-FSDP 1,590)
5. **Llama 70B NVFP4 512GPU**: Official config → 2,013 TFLOP/s
6. **Llama 70B FP8 256+GPU**: Official uses TP=2/PP=4/VP=5; you ran TP=1/PP=1 (not yet compared)
7. **Qwen3 235B BF16**: Your config matches official exactly (514 TFLOP/s)
8. **Qwen3 30B BF16**: Official config → 204 TFLOP/s
9. **Nemotron-H 56B FP8**: 1,563/1,536/1,477 TFLOP/s at 64/256/512 GPU (34.7%/34.1%/32.8% MFU)
10. **DeepSeek-V3 256GPU**: BF16=500.22, FP8=405.59 (512-GPU NCCL timeout)
11. **Nemotron4 340B 256GPU**: FP8=1,101.1, BF16=936.33 (via MPI stub for PMIx v3/v4 compat)
12. **Grok1 314B 256GPU**: FP8=1,370.8, BF16=1,025.4 (via MPI stub for PMIx v3/v4 compat)

## Container Notes

| Model | Official Container | Cluster Container | Notes |
|-------|-------------------|-------------------|-------|
| Nemotron4 15B | 25.09.00 | 26.02.00 + compat_runner.py | API patches needed |
| Llama 3.1 | 26.02.00 | 26.02.00 | matches |
| Qwen3 | 26.02.00 | 26.02.00 | matches |
| DeepSeek-V3 | 26.02.00 | 26.02.00 | matches |
| Nemotron-H | 26.02.00 | 26.02.00 | matches |
| Nemotron4 340B | 25.07.01 | 25.07.01 | matches |
| Grok1 | 25.09.00 | 25.09.00 | matches; helpers.py patched for megatron-core API |

Available images on cluster: `/mnt/vast/johnson/llmb/images/`
- nvidia+nemo+25.07.01.sqsh
- nvidia+nemo+25.09.00.sqsh
- nvidia+nemo+26.02.00.sqsh
