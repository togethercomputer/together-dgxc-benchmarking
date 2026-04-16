# DGXC Benchmark Comparison: Reference vs Johnson's B200 Results

**Date:** 2026-04-14
**Cluster:** Together AI B200 DGXC (74 nodes, 592 GPUs)

## Results (TFLOP/s/GPU)

| Model | Dtype | Run1 | Run2 | Run3 | Run4 | Johnson's Best | Johnson's Job ID | Scale | Notes |
|-------|-------|------|------|------|------|----------------|-----------------|-------|-------|
| DeepSeek V3 | BF16 | 500.22 | | | | — | 81073, 81076 | 512 GPU | Both failed NCCL timeout |
| DeepSeek V3 | FP8 | 405.59 | 486.84 | | | — | — | — | Not run |
| Grok1 | BF16 | 1025.4 | 1024.4 | 1031.6 | | — | — | — | Not run |
| Grok1 | FP8 | 1370.8 | 1366.6 | 1479 | | — | — | — | Not run |
| Llama3.1 70B | FP8 | 983.28 | 982.44 | 984.26 | 982.86 | **1,590** | 79120 | 64 GPU | TP=1,PP=1; also 1,365 at 512 GPU (79133) |
| Llama3.1 70B | NVFP4 | 1939.82 | | | | — | — | 512 GPU | Script exists, no TFLOP/s extracted |
| Nemotron-h | FP8 | 1478.58 | 1478.58 | 1579.26 | 1579.76 | — | — | — | Not run |
| Nemotron4 15B | BF16 | 1075.5 | 1075.5 | | | **1,571** | 81054 | 64 GPU | Also 1,439 at 256 GPU (81056) |
| Nemotron4 15B | FP8 | 1407.4 | 1410.7 | | | **1,904** | 81072 | 256 GPU | GBS=2048 optimized; also 1,916 at 64 GPU (81053) |
| Nemotron4 340B | BF16 | 936.33 | 934.68 | | | — | — | — | Not run |
| Nemotron4 340B | FP8 | 1101.1 | 1101.1 | 1403.4 | 1401.8 | — | — | — | Not run |
| Qwen3 235B | BF16 | 556.18 | 556 | | | **514** | 80077 | 256 GPU | TP=1,PP=8,EP=8,VP=4 (matches official) |
| Qwen3 235B | FP8 | 419.51 | | | | **426** | 80084 | 256 GPU | FP8 MX + CUDA Graphs |
| Llama3.1 405B | FP8 | 1535.81 | | | | — | — | — | Official config (TP=4,PP=8,CP=2) not yet run |
| Llama3.1 405B | NVFP4 | 1883.05 | | | | **2,006** | 80076 | 256 GPU | TP=4,PP=8 optimized (+28% over PP=16 baseline) |

## Summary of Comparison

| Model | Dtype | Reference Best | Johnson's Best | Delta | Notes |
|-------|-------|---------------|----------------|-------|-------|
| Llama3.1 70B | FP8 | 984.26 | **1,590** | **+62%** | Different scale (ref unknown vs 64 GPU) |
| Nemotron4 15B | BF16 | 1,075.5 | **1,571** | **+46%** | 64 GPU; also +34% at 256 GPU (1,439) |
| Nemotron4 15B | FP8 | 1,410.7 | **1,904** | **+35%** | 256 GPU GBS=2048 optimized |
| Qwen3 235B | BF16 | 556.18 | 514 | **-8%** | Below reference |
| Qwen3 235B | FP8 | 419.51 | **426** | **+2%** | CUDA Graphs enabled |
| Llama3.1 405B | NVFP4 | 1,883.05 | **2,006** | **+7%** | PP=8 optimization |
| Llama3.1 405B | FP8 | 1,535.81 | — | — | Not yet run |
| DeepSeek V3 | BF16 | 500.22 | — | — | Failed NCCL timeout |
| DeepSeek V3 | FP8 | 486.84 | — | — | Not run |
| Grok1 | BF16 | 1,031.6 | — | — | Not run |
| Grok1 | FP8 | 1,479 | — | — | Not run |
| Nemotron-h | FP8 | 1,579.76 | — | — | Not run |
| Nemotron4 340B | BF16 | 936.33 | — | — | Not run |
| Nemotron4 340B | FP8 | 1,403.4 | — | — | Not run |

## Gaps to Fill

Models/configs we have NOT yet run on this cluster:

1. **Llama3.1 405B FP8** — reference: 1,535.81. Official config: TP=4,PP=8,CP=2,VP=8
2. **DeepSeek V3 BF16/FP8** — BF16 failed at 512 GPU (NCCL timeout), FP8 not attempted
3. **Grok1 BF16/FP8** — 314B MoE, TP=4,PP=4,EP=8, scales 256-1024
4. **Nemotron-h FP8** — 56B hybrid, TP=2, scales 64-512
5. **Nemotron4 340B BF16/FP8** — TP=8,PP=4, scales 128-1024
6. **Llama3.1 70B FP8 at 256+ GPUs** — with official TP=2,PP=4,VP=5 config (we only ran TP=1,PP=1)
7. **Llama3.1 70B NVFP4** — script exists but no metrics extracted

## Models Where We Beat Reference

| Model | Dtype | Improvement | Key Optimization |
|-------|-------|-------------|-----------------|
| Llama3.1 70B | FP8 | +62% | May be scale difference |
| Nemotron4 15B | BF16 | +46% | New LLMB + 26.02 container |
| Nemotron4 15B | FP8 | +35% | GBS=2048 for comm/compute overlap |
| Llama3.1 405B | NVFP4 | +7% | PP=16→PP=8 parallelism rebalancing |
| Qwen3 235B | FP8 | +2% | CUDA Graphs |

## Models Where We Are Below Reference

| Model | Dtype | Gap | Reason |
|-------|-------|-----|--------|
| Qwen3 235B | BF16 | -8% | ~43% MFU is the ceiling for 22B-active MoE; ref may be from a different cluster/config |
