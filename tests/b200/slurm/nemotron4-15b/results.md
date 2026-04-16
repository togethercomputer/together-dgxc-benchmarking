# Nemotron4 15B — B200 Benchmark Results

**Cluster:** Together AI B200 (8x B200/node, ConnectX-7 IB 400Gb/s)
**Container:** nvidia+nemo+26.02.00 (via compat_runner.py for 25.09→26.02 compat)
**Base config:** TP=1, PP=1, CP=1, VP=1, EP=1, MBS=2, SeqLen=4096
**Date:** 2026-04-13

## BF16 Results

| GPUs | Nodes | GBS | Job ID | TFLOP/s/GPU | Step Time (s) | Target | Gap | Status |
|------|-------|-----|--------|-------------|---------------|--------|-----|--------|
| 64   | 8     | 256  | 81054 | 1,571 | 0.930 | 1,264 | **+24.2%** | OK |
| 256  | 32    | 1024 | 81056 | 1,439 | 1.015 | 1,264 | **+13.8%** | OK |

**Scaling efficiency (64→256):** 91.7%

## FP8 Results — 64 GPUs

| GPUs | GBS | Job ID | TFLOP/s/GPU | Step Time (s) | Target | Gap | Status |
|------|-----|--------|-------------|---------------|--------|-----|--------|
| 64   | 256 | 81053  | 1,916 | 0.763 | 1,908 | -0.4% | OK |

Note: CUDA graphs disabled at 64 GPUs due to FP8 tensor copy bug (cudaErrorInvalidValue on graph replay).

## FP8 Results — 256 GPUs (Optimization Experiments)

| Experiment | GBS | TP | Job ID | TFLOP/s/GPU | Step Time (s) | vs Baseline | Status |
|-----------|------|-----|--------|-------------|---------------|-------------|--------|
| Baseline | 512 | 1 | 81057 | 1,594 | 0.457 | — | OK (comm-bound, 53% NCCL) |
| Baseline | 512 | 1 | 81058 | 1,601 | 0.456 | — | OK (repeat) |
| Baseline | 512 | 1 | 81059 | 1,594 | 0.458 | — | OK (repeat) |
| exp1: GBS=1024 | 1024 | 1 | 81060 | 1,798 | 0.813 | +12.5% | OK |
| exp2: TP=2 | 512 | 2 | 81066 | 1,308 | 0.558 | -18% | OK (TE segfault, tp_comm_overlap off) |
| exp3: TP=2+GBS1024 | 1024 | 2 | 81067 | 1,375 | 1.063 | -14% | OK |
| exp4: NVLS | 512 | 1 | 81061 | 1,536 | 0.476 | -4% | NVLS hurts at TP=1/DP=256 |
| exp6: GBS=2048 | 2048 | 1 | 81069 | 1,892 | 1.544 | +18.7% | OK |
| **exp6: GBS=2048 (rerun)** | **2048** | **1** | **81072** | **1,904** | **1.535** | **+19.4%** | **OK (final submission)** |
| exp7: GBS=2560 | 2560 | 1 | 81071 | 1,923 | 1.899 | +20.6% | OK (best absolute) |
| profile: GBS=1024 | 1024 | 1 | 81068 | 1,800 | 0.813 | — | profiler run |
| profile: GBS=2048 | 2048 | 1 | 81070 | 1,914 | 1.524 | — | profiler run |

**Tranche-1 Target:** 1,908 TFLOP/s/GPU
**Best match:** Job 81072 (GBS=2048) = 1,904 (**-0.16% from target**)
**Best absolute:** Job 81071 (GBS=2560) = 1,923 (+0.8% above target)

## Failed Jobs (Infrastructure Debugging)

Jobs 81026-81052 (BF16 64-GPU) had multiple failure modes resolved sequentially:
1. Pyxis container startup (`spank_pyxis.so: task_init() failed`) → `--no-container-mount-home` + `HOME=/tmp`
2. Tokenizer offline loading → correct HF cache mounts
3. `no_weight_decay_cond` optimizer TypeError → compat_runner.py patch
4. `No module named 'netrc'` → fixed in compat_runner.py tensorstore stub
5. PermissionError on NeMo NLP tmp → `NEMO_NLP_TMP=/tmp/nemo_nlp_tmp`
6. FP8 CUDA graph replay bug → unconditionally disable enable_cuda_graph

## Key Findings

- GBS tuning is the dominant optimization: GBS=512→2048 hides NCCL allreduce behind gradient accumulation (4 microbatches/rank)
- FP8 scaling efficiency 64→256: **100.4%** (super-linear, GBS tuning compensates)
- BF16 scaling efficiency 64→256: 91.7%
- Contiguous node allocation gives ~1.3% boost from shorter IB paths
- TP=2 not viable: TE user buffer segfault when config is monkey-patched, forced tp_comm_overlap off
- compat_runner.py required for 25.09 NeMo configs on 26.02 container

## Profiler Analysis

- FP8 profiler: `~/johnson/worklog/profiler_reports/nemotron4_15b_fp8_256gpu_profiler.md`

## Log Locations

- `/mnt/vast/johnson/llmb/workloads/pretrain_nemotron4-15b/experiments/`
- `/home/johnson/johnson/scripts/nemotron4_15b/*/`
