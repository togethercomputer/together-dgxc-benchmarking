# B200 DGXC Benchmark Summary

Together AI B200 DGXC cluster, 256 GPUs (32 nodes) unless noted.
Date: 2026-04-14

## Results

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

## Column Definitions

- **Target**: NVIDIA published reference TFLOP/s/GPU from dgxc-benchmarking
- **Official config**: Result from running NVIDIA's exact reference config on our cluster
- **Optimized**: Best result after tuning parallelism/GBS/container beyond official config
- **—**: No optimization attempted; official config is the only result

## Key Wins

1. **Llama 3.1 405B NVFP4 (+28%)**: PP=16→PP=8 halved pipeline depth, bubble fraction dropped from 43% to 16%, P2P comm reduced 84%, DP doubled from 4 to 8. Megatron-Bridge upgrade added another +7%.
2. **Nemotron4 15B FP8 (+6%)**: GBS 1024→2048 hides 24% NCCL allreduce time behind 4 gradient accumulation steps. Meets Tranche-1 target of 1,908.
3. **Nemotron4 15B BF16 (+34% over target)**: NeMo 26.02 container with compat_runner.py patches delivers large uplift over 25.09 reference container.

## Known Gaps

- **Qwen3 235B BF16 (-8%)**: IB fabric oversubscription limits MFU to 43% ceiling. 5 optimization experiments (EP tuning, CUDA graphs, A2A overlap) showed no improvement.
- **DeepSeek V3 512GPU**: NCCL timeout. PP=16 requires cross-node P2P; NCCL falls back to socket transport on non-routable 7.247.226.x subnet.
- **Grok1/Nemotron4 340B**: Required MPI stub (LD_PRELOAD) for PMIx v3/v4 mismatch in legacy containers (25.07.01, 25.09.00). No optimization beyond official config.
