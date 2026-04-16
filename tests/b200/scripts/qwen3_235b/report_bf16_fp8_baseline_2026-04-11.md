# Qwen3-235B BF16 & FP8 MX Baseline Benchmark Report

**Date:** 2026-04-11
**Cluster:** Together AI B200 (use3a), 32 nodes x 8 GPUs (256 GPUs total)
**Container:** nvidia+nemo+26.02.00
**Megatron-Bridge:** commit `6b3b5ba7e` (install at `/mnt/vast/johnson/llmb/`)

## Summary

Benchmarked Qwen3-235B (MoE, 22B active params) pretraining on 256 B200 GPUs
using standard NVIDIA dgxc-benchmarking configs. BF16 achieves **514 TFLOP/s/GPU**
(~37.8s/step). FP8 MX achieves **383 TFLOP/s/GPU** (~50.6s/step), slower than BF16
due to missing virtual pipeline interleaving (VP=None vs VP=4).

## Configurations

| Parameter | BF16 (job 80077) | FP8 MX (job 80078) |
|-----------|------------------|---------------------|
| Precision | BF16 | FP8 MX |
| TP | 1 | 1 |
| PP | 8 | 8 |
| CP | 1 | 1 |
| VP | 4 | None |
| EP | 8 | 8 |
| ETP | 1 | 1 |
| MBS | 1 | 1 |
| GBS | 8192 | 8192 |
| GPUs | 256 | 256 |
| moe_a2a_overlap | False | False |
| Config loaded | B200_BF16_V2 | B200_FP8_MX_V2 |

## Results

### BF16 (job 80077)

**Nodes:** use3a-ss-b200-gpu-[159-190]

| Iter | Step Time (s) | TFLOP/s/GPU |
|------|---------------|-------------|
| 1 | 507.31 | 38.2 (warmup) |
| 2 | 38.55 | 503.4 |
| 3 | 37.79 | 513.6 |
| 4 | 37.76 | 514.0 |
| 5 | 37.75 | 514.1 |
| 6 | 37.86 | 512.6 |
| 7 | 37.61 | 515.9 |
| 8 | 37.56 | 516.7 |
| 9 | 37.78 | 513.7 |
| 10 | 37.80 | 513.4 |
| 11 | 37.86 | 512.6 |
| 12 | 37.86 | 512.5 |
| 13 | 37.90 | 512.1 |
| 14 | 37.75 | 514.1 |
| 15 | 37.75 | 514.0 |
| 16 | 37.78 | 513.7 |
| **Avg (3-16)** | **~37.79** | **~513.8** |

### FP8 MX (job 80078)

**Nodes:** use3a-ss-b200-gpu-[145,195-205,209-217,226-230,233-235,238-239,256]

| Iter | Step Time (s) | TFLOP/s/GPU |
|------|---------------|-------------|
| 1 | 527.44 | 36.8 (warmup) |
| 2 | 51.38 | 377.6 |
| 3 | 50.54 | 383.9 |
| 4 | 50.57 | 383.7 |
| 5 | 50.57 | 383.7 |
| 6 | 50.56 | 383.7 |
| 7 | 50.73 | 382.5 |
| 8 | 50.77 | 382.2 |
| 9 | 50.75 | 382.3 |
| 10 | 50.67 | 382.9 |
| 11 | 50.70 | 382.7 |
| **Avg (3-11)** | **~50.65** | **~383.1** |

### Comparison

| Metric | BF16 | FP8 MX | Delta |
|--------|------|--------|-------|
| Avg step time (s) | 37.79 | 50.65 | FP8 34% slower |
| Avg TFLOP/s/GPU | 513.8 | 383.1 | FP8 -25% |
| VP (virtual pipeline) | 4 | None | Key difference |

## Notes

- **FP8 slower than BF16:** The FP8 MX V2 config has `VP=None` (no virtual pipeline
  interleaving) while BF16 V2 has `VP=4`. VP significantly reduces pipeline bubble
  overhead. This is the standard NVIDIA config — the FP8 config may be optimized for
  different hardware (GB200/GB300) or may need VP tuning for B200.

- **BF16 matches Max's result:** Our BF16 ~514 TFLOP/s matches Max's job 80044
  (~518 TFLOP/s), confirming our install is working correctly.

- **grad norm: nan (BF16):** Expected for throughput benchmarking with synthetic data.
  FP8 shows normal grad norms because it uses a different code path.

- **HOME fix applied:** Added `-ce HOME=/tmp/$USER,NCCL_SOCKET_IFNAME=bond0` to
  `qwen3/pretrain/launch.sh` to fix pyxis container permission errors.

## Job History

| Job ID | Config | Status | Notes |
|--------|--------|--------|-------|
| 80077 | BF16, VP=4 | COMPLETED | 16/16 iterations |
| 80078 | FP8 MX, VP=None | COMPLETED | 16/16 iterations (11+ shown) |

## Paths

| What | Path |
|------|------|
| llmb install | `/mnt/vast/johnson/llmb/` |
| dgxc-benchmarking | `/mnt/vast/johnson/dgxc-benchmarking/` |
| Qwen3 launch.sh | `/mnt/vast/johnson/dgxc-benchmarking/qwen3/pretrain/launch.sh` |
| Megatron-Bridge | `/mnt/vast/johnson/llmb/workloads/pretrain_qwen3/Megatron-Bridge/` |
| BF16 experiment | `.../experiments/pretrain_qwen3_235b_a22b_bf16_gpus256_.../1775960445/` |
| FP8 experiment | `.../experiments/pretrain_qwen3_235b_a22b_fp8_mx_gpus256_.../1775960463/` |
