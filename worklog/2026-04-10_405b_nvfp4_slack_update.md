**405B NVFP4 Benchmark — B200 Cluster Network Bottleneck Identified**

Llama 3.1 405B NVFP4 on 256 B200 GPUs (32 nodes) with the NVIDIA reference config (TP=4, PP=16, GBS=1536) hit **1,352 TFLOP/s/GPU** — 32% below the ~1,990 achieved on a previous tranche. Profiler traces showed P2P pipeline communication at ~56s/step dominating the ~22s of compute, with 43% pipeline bubble. This pointed to cross-node P2P bandwidth as the bottleneck, not compute.

Reduced pipeline parallelism from PP=16 to PP=8, doubling data parallelism (DP=4→8) to compensate. Result: **1,868 TFLOP/s/GPU (+38%)**. P2P time dropped from ~56s to ~9s/step, bubble fraction from 43% to 15.5%. Other variants tested — CP=2 was 44% worse, VP=8 neutral, TP=8 hung during NCCL init.

NCCL SendRecv benchmarks at 2/4/8/16/32 nodes confirmed the network issue: P2P bandwidth holds at ~42 GB/s for 2-4 nodes, drops to ~26 GB/s at 8 nodes (-41%), then to ~14.5 GB/s at 16-32 nodes (-66% from baseline). All-reduce showed similar degradation at scale. PP=16 operates entirely in the degraded regime (14.5 GB/s); PP=8 partially avoids it (~26 GB/s).

This scaling pattern — healthy within a small node group, steep cliff beyond — is consistent with oversubscribed spine uplinks in the IB fabric. With full bandwidth, estimated throughput would reach ~2,080 TFLOP/s/GPU. Enabling SHARP (currently unconfigured — AM has no reservations) would further help with the 8.6s/step DP gradient sync overhead.
