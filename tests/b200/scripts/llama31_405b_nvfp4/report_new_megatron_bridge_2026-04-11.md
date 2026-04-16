# Llama 3.1 405B NVFP4 256-GPU: New Megatron-Bridge Benchmark Report

**Date:** 2026-04-11
**Cluster:** Together AI B200 (use3a), 32 nodes x 8 GPUs (256 GPUs total)
**Container:** nvidia+nemo+26.02.00

## Summary

Upgraded Megatron-Bridge from commit `4df8c9739` to `6b3b5ba7e` and benchmarked
Llama 3.1 405B NVFP4 pretraining on 256 B200 GPUs. The new MB delivers **~7%
throughput improvement** on this dense model. PP=8 achieves **~1,997 TFLOP/s/GPU**,
a new best for this workload.

## Environment

| Component | Old | New |
|-----------|-----|-----|
| Megatron-Bridge | `4df8c9739` | `6b3b5ba7e` |
| llmb install | `/mnt/vast/llmb_/` | `/mnt/vast/johnson/llmb/` |
| dgxc-benchmarking | `/home/johnson/johnson/scripts/` (manual) | `/mnt/vast/johnson/dgxc-benchmarking/` |
| Container | nvidia+nemo+26.02.00 | nvidia+nemo+26.02.00 (same) |

## Results

### PP=16 Baseline (job 80073)

**Config:** TP=4, PP=16, CP=1, VP=8, MBS=1, GBS=1536, recompute_num_layers=1
**Nodes:** use3a-ss-b200-gpu-[159-190]

| Iter | Step Time (s) | TFLOP/s/GPU |
|------|---------------|-------------|
| 1 | 533.65 | 232.5 (warmup) |
| 2 | 78.67 | 1,576.9 |
| 3 | 78.74 | 1,575.6 |
| 4 | 79.21 | 1,566.1 |
| 5 | 78.27 | 1,585.0 |
| 6 | 78.90 | 1,572.3 |
| 7 | 79.43 | 1,561.8 |
| 8 | 79.80 | 1,554.7 |
| 9 | 78.66 | 1,577.1 |
| 10 | 78.60 | 1,578.3 |
| 11 | 79.53 | 1,559.9 |
| 12 | 80.29 | 1,545.1 |
| **Avg (2-12)** | **79.10** | **~1,568** |

### PP=8 (job 80076)

**Config:** TP=4, PP=8, CP=1, VP=4, MBS=1, GBS=1536, recompute_num_layers=8
**Nodes:** use3a-ss-b200-gpu-[145,195-205,209-217,226-230,233-235,238-239,256]

| Iter | Step Time (s) | TFLOP/s/GPU |
|------|---------------|-------------|
| 1 | 298.83 | 415.1 (warmup) |
| 2 | 61.91 | 2,004.0 |
| 3 | 62.15 | 1,996.3 |
| 4 | 62.12 | 1,997.2 |
| 5 | 62.16 | 1,995.9 |
| 6 | 62.14 | 1,996.4 |
| 7 | 62.15 | 1,996.0 |
| 8 | 62.15 | 1,996.2 |
| **Avg (2-8)** | **62.10** | **~1,997** |

### Comparison with Old Megatron-Bridge

| Config | Old MB (`4df8c97`) | New MB (`6b3b5ba`) | Improvement |
|--------|--------------------|--------------------|-------------|
| PP=16, TP=4, VP=8 | ~84s / 1,471 TFLOP/s | **79.1s / 1,568 TFLOP/s** | **+6.6%** |
| PP=8, TP=4, VP=4 | ~67s / 1,860 TFLOP/s | **62.1s / 1,997 TFLOP/s** | **+7.4%** |

## Notes

- **PP=8 OOM with recompute=1:** Job 80074 (PP=8 without recompute override) hit CUDA
  OOM after iter 1. The V1 fallback config only sets `recompute_num_layers=1`, which is
  insufficient for PP=8. The old MB scripts used `recompute_num_layers=8`. Fixed in
  job 80076 by passing `-rl 8`.

- **No B200_NVFP4_V2 config:** The new MB does not include a
  `LLAMA31_405B_PRETRAIN_CONFIG_B200_NVFP4_V2` variant. `setup_experiment.py` falls
  back to V1 (128-GPU config) with a warning. `launch.sh` overrides `num_gpus=256`
  and `GBS=1536` via CLI.

- **HOME directory fix:** Added `-ce HOME=/tmp/johnson,NCCL_SOCKET_IFNAME=bond0` to
  `launch.sh` to fix pyxis container creation failure on compute nodes where
  `/home/johnson` is local disk (not shared NFS). Jobs 80070-80071 failed without this.

- **launch.sh modifications:** Added `RECOMPUTE_NUM_LAYERS` env var support and the
  `-ce` container env fix to `/mnt/vast/johnson/dgxc-benchmarking/llama3.1/launch.sh`.

## Job History

| Job ID | Config | Status | Notes |
|--------|--------|--------|-------|
| 80070 | PP=16 | FAILED | pyxis HOME dir permission error |
| 80071 | PP=8 | FAILED | pyxis HOME dir permission error |
| 80072 | PP=16 | CANCELLED | cancelled before fix verified |
| 80073 | PP=16 | SUCCESS | with `-ce HOME=/tmp/johnson` fix |
| 80074 | PP=8 | FAILED | CUDA OOM (recompute_num_layers=1) |
| 80075 | PP=8 | CANCELLED | `-rl 8` not passed (launch.sh bug) |
| 80076 | PP=8 | SUCCESS | with recompute_num_layers=8 |

## Paths

| What | Path |
|------|------|
| New llmb install | `/mnt/vast/johnson/llmb/` |
| Megatron-Bridge | `/mnt/vast/johnson/llmb/workloads/pretrain_llama3.1/Megatron-Bridge/` |
| dgxc-benchmarking | `/mnt/vast/johnson/dgxc-benchmarking/` |
| launch.sh (modified) | `/mnt/vast/johnson/dgxc-benchmarking/llama3.1/launch.sh` |
| PP=16 experiment | `.../experiments/pretrain_llama31_405b_nvfp4_gpus256_tp4_pp16_.../1775958352/` |
| PP=8 experiment | `.../experiments/pretrain_llama31_405b_nvfp4_gpus256_tp4_pp8_.../1775959023/` |
