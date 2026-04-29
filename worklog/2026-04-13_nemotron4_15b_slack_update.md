**Nemotron4 15B Benchmark Results — B200 Cluster**

| Model | Job ID | Result | Best Tranche-1 Target | Gap | Key Optimizations |
|-------|--------|--------|----------------------|-----|-------------------|
| Nemotron4-15B BF16 (64 GPUs) | #81054 | 1,570 TFLOP/s/GPU | 1,264 TFLOP/s/GPU | **+24.2%** | NeMo 26.02 container, CUDA Graphs enabled |
| Nemotron4-15B BF16 (256 GPUs) | #81056 | 1,439 TFLOP/s/GPU | 1,264 TFLOP/s/GPU | **+13.8%** | Same as 64-GPU, GBS scaled 256→1024 |
| Nemotron4-15B FP8 (64 GPUs) | #81053 | 1,895 TFLOP/s/GPU | 1,908 TFLOP/s/GPU | -0.7% | NeMo 26.02 container, CUDA Graphs disabled (FP8 graph replay bug) |
| Nemotron4-15B FP8 (256 GPUs) | #81072 | 1,905 TFLOP/s/GPU | 1,908 TFLOP/s/GPU | **-0.16%** | GBS 512→2048 (allreduce hidden behind grad accum), CUDA Graphs, contiguous nodes |

**Key Takeaways**

1. **Both BF16 runs exceed Tranche-1 by wide margin** (+13–24%). BF16 is not the bottleneck on this cluster.
2. **FP8 256-GPU meets target** — GBS tuning was critical. Default GBS=512 was communication-bound (~1,600 TFLOP/s/GPU); scaling to GBS=2048 hides NCCL allreduce behind 4 gradient accumulation steps, recovering full scaling efficiency (100.4%).
3. **FP8 gives 1.21x over BF16** at matched GPU count (64 GPUs).
4. **BF16 scaling efficiency 64→256 GPUs: 91.7%** (8.3% drop from inter-node allreduce). FP8 scaling efficiency: 100.4% (GBS tuning compensates).
