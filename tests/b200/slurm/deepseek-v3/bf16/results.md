# DeepSeek-V3 671B BF16 — B200 Benchmark Results

**Cluster:** Together AI B200 (8x B200/node, ConnectX-7 IB 400Gb/s)
**Container:** nvidia+nemo+26.02.00
**Config:** TP=1, PP=16, EP=8, ETP=1, MBS=1, GBS=4096, 512 GPUs (64 nodes)
**Date:** 2026-04-13 to 2026-04-14

## Results

| GPUs | Job ID | TFLOP/s/GPU | Status | Failure |
|------|--------|-------------|--------|---------|
| 512  | 81073  | — | FAILED | NCCL collective timeout (~637-638s, exceeding 600s limit) |
| 512  | 81076  | — | FAILED | NCCL collective timeout (~639-640s) + DistBackendError |

Both jobs failed before producing any training iteration metrics.

## Reference Performance (from NVIDIA docs)

| Platform | GPUs | Precision | tokens/sec/GPU | TFLOP/s/GPU |
|----------|------|-----------|----------------|-------------|
| DGX-GB200 | 256 | BF16 | 3,139 | 782 |
| DGX-B200 | 256 | FP8-MX | 2,139 | 557 |

## Next Steps

- Debug NCCL timeout: likely IB fabric issue at 512-GPU scale with PP=16 (cross-node P2P)
- Try PP=8 to reduce cross-node P2P hops (same optimization that helped 405B)
- Try smaller scale (256 GPUs) first to validate config works
- Check if NCCL_TIMEOUT needs to be increased for this model size

## Log Locations

- `/mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3/experiments/`
