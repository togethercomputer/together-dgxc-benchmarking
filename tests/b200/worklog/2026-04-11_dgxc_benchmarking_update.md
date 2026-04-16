# DGXC Benchmarking Optimization Summary — 2026-04-11

256 B200 GPUs (32 nodes)

| Model | Old DGXC (original) | Optimized (Job ID) | Improvement | Key Optimizations | Tranche-1 Target | Gap |
|-------|---------------------|---------------------|-------------|-------------------|-------------------|-----|
| Qwen3-235B BF16 | 338 TFLOP/s | 514 TFLOP/s (#80077) | +52% | New LLMB + docker26.02 + NCCL_SOCKET_IFNAME specified | 557 TFLOP/s | -8% |
| Qwen3-235B FP8 MX | 254 TFLOP/s | 425 TFLOP/s (#80084) | +67% | New LLMB + docker26.02 + NCCL_SOCKET_IFNAME specified, CUDA Graphs | 436 TFLOP/s | -3% |
| Llama3.1-405B NVFP4 | 1,352 TFLOP/s | 2,004 TFLOP/s (#80076) | +48% | New LLMB + docker26.02 + NCCL_SOCKET_IFNAME specified, PP=16→PP=8 | 2,189 TFLOP/s | -8% |

## Key Takeaways

1. **New LLMB install + docker26.02 + NCCL_SOCKET_IFNAME specified = +52% gain**, the dominant improvement across all models.
2. **All three models within 3-8% of Tranche-1 targets.** Remaining gap likely from IB fabric oversubscription.
3. **CUDA Graphs is the only effective FP8 optimization** (+10%). Four other techniques tested — none helped, A2A Overlap regressed -42%.
4. **17% FP8-to-BF16 gap is hardware-intrinsic** (MXFP8 GEMM + quantization overhead). Needs newer NVIDIA kernels to close.
5. **405B NVFP4: 1,352 → 2,004 TFLOP/s/GPU (+48%)** via PP=8 + new install, near the 2,189 target.
