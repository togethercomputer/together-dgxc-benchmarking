# B200 Slurm Benchmark Results

Pretraining benchmark scripts and results on Together AI B200 cluster (74 nodes, 592 GPUs).

**Cluster:** Together AI B200 DGXC
- 74 nodes, 8x NVIDIA B200 per node, 592 GPUs total
- 8x ConnectX-7 IB 400Gb/s NICs per node
- Container: `nvidia+nemo+26.02.00` (enroot/pyxis)
- Scheduler: Slurm with Pyxis plugin
- Storage: `/mnt/vast/` (shared), `/home/johnson/` (login only)

**Date range:** 2026-04-08 to 2026-04-14

## Directory Structure

```
slurm/
├── llama3.1-70b/
│   ├── fp8/           # 64-592 GPU scaling study
│   └── nvfp4/         # 512 GPU
├── llama3.1-405b/
│   └── nvfp4/         # 256+512 GPU, PP optimization
├── qwen3-235b/
│   ├── bf16/          # 256 GPU, parallelism experiments
│   └── fp8mx/         # 256 GPU, CUDA graph optimization
├── nemotron4-15b/
│   ├── bf16/          # 64+256 GPU
│   └── fp8/           # 64+256 GPU, GBS sweep
├── deepseek-v3/
│   └── bf16/          # 512 GPU (NCCL timeout, WIP)
└── qwen3-30b/
    └── fp8mx/         # 64 GPU
```

Each model directory contains:
- `sbatch.sh` / `train.sh` — reproducible launch scripts
- `results.md` — summary table with job IDs, TFLOP/s/GPU, and key findings
- `compat_runner.py` — container compatibility wrapper (Nemotron4 only)

## Quick Reference: Best Results

| Model | Precision | GPUs | TFLOP/s/GPU | Target | Gap | Job ID |
|-------|-----------|------|-------------|--------|-----|--------|
| Llama 70B | FP8 | 64 | 1,590 | — | — | 79120 |
| Llama 70B | FP8 | 512 | 1,365 | — | — | 79133 |
| Llama 405B | NVFP4 | 256 | 2,006 | 2,189 | -8% | 80076 |
| Llama 405B | NVFP4 | 512 | 1,818 | — | — | 79953 |
| Qwen3 235B | BF16 | 256 | 514 | 557 | -8% | 80077 |
| Qwen3 235B | FP8 MX | 256 | 426 | 436 | -3% | 80084 |
| Nemotron4 15B | BF16 | 64 | 1,571 | 1,264 | +24% | 81054 |
| Nemotron4 15B | BF16 | 256 | 1,439 | 1,264 | +14% | 81056 |
| Nemotron4 15B | FP8 | 64 | 1,916 | 1,908 | -0.4% | 81053 |
| Nemotron4 15B | FP8 | 256 | 1,904 | 1,908 | -0.16% | 81072 |
| DeepSeek-V3 | BF16 | 512 | — | — | — | 81073 (FAILED) |

## Known Cluster Issues

1. **Pyxis HOME mount:** Must use `--no-container-mount-home` + `export HOME=/tmp` before srun
2. **SHARP not available:** `sharpd` not running, no reservations configured — admin setup needed
3. **IB P2P bandwidth degradation:** 42 GB/s at 2-4 nodes → 14.5 GB/s at 32+ nodes
4. **NeMo 25.09→26.02 compat:** Use `compat_runner.py` for tensorstore stub, CUDA graph API, optimizer kwargs

## How to Re-run

```bash
# Example: Nemotron4 15B FP8 256 GPUs
cd tests/b200/slurm/nemotron4-15b/fp8/256gpu/
sbatch sbatch.sh
```

Logs go to `/mnt/vast/` paths specified in each sbatch.sh.
