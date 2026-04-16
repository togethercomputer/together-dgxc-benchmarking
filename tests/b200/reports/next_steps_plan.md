# Next Steps Plan — B200 Cluster Performance Optimization

**Date:** 2026-04-10
**Based on:**
- NCCL Benchmark Report (2026-04-09) — cluster health, NCCL scaling, compute benchmarks
- 405B NVFP4 256-GPU Profiler Report (2026-04-10) — training bottleneck analysis

---

## Executive Summary

The cluster hardware is healthy (592 GPUs, 96–97% MFU on isolated GEMM, SHARP at 96% IB line rate). However, the Llama 3.1 405B NVFP4 training at 256 GPUs shows **communication takes 4.3× longer than compute**, with PP=16 pipeline P2P being the dominant bottleneck (65% of all comm time). Effective training MFU is ~15.9% (1,428 TFLOP/s/GPU vs 9,000 TFLOP/s peak), far below the 93% MFU achieved in isolated NVFP4 GEMM benchmarks. The gap is almost entirely due to pipeline parallelism overhead and communication stalls.

---

## Phase 1: Quick Wins (No Parallelism Changes)

### 1.1 Enable SHARP for 405B Training
**Priority:** High | **Effort:** Low | **Expected impact:** 15–30% faster collectives

The current 405B training does NOT set `NCCL_COLLNET_ENABLE=1`. The NCCL report shows SHARP delivers 385 GB/s vs Ring's 306 GB/s at 32 nodes — a 25% advantage for AllReduce/ReduceScatter/AllGather.

From the profiler: AllReduce (20.4s) + ReduceScatter (14.6s) + AllGather (7.9s) = **42.9s of collective comm**. With SHARP this could reduce to ~33s, saving ~10s/step.

**Action:**
- Add `export NCCL_COLLNET_ENABLE=1` to `sbatch.sh`
- Note: The Qwen3 235B test showed NCCL algo doesn't matter for PP-dominated workloads. However, the 405B has DP=4 gradient sync that IS collective-based, and the absolute comm time is much larger. Worth testing.

### 1.2 Verify Communication-Compute Overlap
**Priority:** High | **Effort:** Medium | **Expected impact:** Potentially large

Check if gradient AllReduce/ReduceScatter is properly overlapped with backward pass:
- `CUDA_DEVICE_MAX_CONNECTIONS=32` is already set (good)
- Verify Megatron-Bridge enables `--overlap-grad-reduce` and `--overlap-param-gather`
- Profile with overlap enabled/disabled to measure actual overlap efficiency

**Action:**
- Inspect Megatron-Bridge run_script.py and config for overlap flags
- Run a comparison: overlap on vs off

### 1.3 Eliminate aten::item Synchronization
**Priority:** Medium | **Effort:** Low | **Expected impact:** ~2.75s/step saved

A single `aten::item` call blocks for 2.75s (full GPU sync). This is likely from loss-scale or grad-norm computation.

**Action:**
- Grep Megatron-Bridge for `aten::item` / `.item()` calls in the training loop
- Replace with async alternatives where possible

---

## Phase 2: Parallelism Strategy Exploration

### 2.1 Test PP=8 Configuration (Reduce Pipeline Depth)
**Priority:** Critical | **Effort:** Medium | **Expected impact:** Potentially 30–50% faster

PP=16 is the #1 bottleneck: 111.7s of P2P send/recv + 37.1s GPU idle bubbles. The NCCL report's Qwen3 235B analysis confirms PP dominates inter-node communication even more than collectives.

Reducing PP from 16 to 8 would:
- Halve P2P send/recv volume (~55s savings potential)
- Reduce bubble fraction from ~43% to ~25% (with VP adjustment)
- Require compensating with TP=8 (from TP=4) to fit the 405B model

**Trade-off:** TP=8 increases AllReduce volume (more intra-op comm), but SHARP handles this well at 385 GB/s. The net effect should be strongly positive since P2P is far more expensive than TP AllReduce on NVSwitch.

**Action:**
- Create new config: TP=8, PP=8, CP=1, VP=TBD, DP=4
- Verify memory fits (405B / 8 PP stages = 50.6B params per stage, TP=8 further shards)
- Run benchmark and compare step time

### 2.2 Test CP=2 Configuration (Context Parallelism)
**Priority:** Medium | **Effort:** Medium | **Expected impact:** Moderate

Alternative to increasing TP: use CP=2 to reduce PP depth.
- TP=4, PP=8, CP=2, DP=4 → 256 GPUs
- CP ring-attention allows sequence dimension sharding with good overlap

**Action:**
- Create config: TP=4, PP=8, CP=2, VP=TBD
- Run benchmark and compare

### 2.3 Compare with Megatron-Bridge Default Config
**Priority:** Medium | **Effort:** Low | **Expected impact:** Informational

The memory notes show Config B (Megatron-Bridge) uses TP=4, PP=8, CP=2, VP=4 — a very different parallelism strategy from the dgxc-benchmarking Config A (TP=4, PP=16, CP=1, VP=8).

**Action:**
- Run Config B (`/home/johnson/johnson/256gpus_405b_nvfp4_mbs1_mbridge/sbatch.sh`)
- Compare step time, TFLOP/s, and profiler results against Config A

---

## Phase 3: Scale-Out Validation

### 3.1 512-GPU (64 Node) 405B NVFP4 Run
**Priority:** High (after Phase 2 winner identified) | **Effort:** Medium

The NCCL report shows ~5% bandwidth drop at 64 nodes. Need to validate that training scales gracefully.

**Action:**
- Take the best parallelism config from Phase 2
- Scale to 512 GPUs and measure step time degradation
- Profile 1 step to check for new bottlenecks at scale

### 3.2 512-GPU 70B NVFP4 Benchmark
**Priority:** Medium | **Effort:** Low (scripts already exist)

The memory notes indicate scripts are ready at `/home/johnson/johnson/512gpus_nvfp4_mbs1/`. Config: TP=2, PP=4, CP=1, VP=5, MBS=1, GBS=2048.

**Action:**
- Submit the 70B job (was pending reboot)
- Compare TFLOP/s/GPU with the 405B results
- The 70B with PP=4 should have much less pipeline overhead — useful comparison point

---

## Phase 4: Deep Profiling

### 4.1 Multi-Rank PyTorch Profiling
**Priority:** Medium | **Effort:** Medium

Current profiler trace is rank 0 only (PP stage 0). To fully understand pipeline bubble distribution:
- Profile rank 0 (first PP stage) + rank 60 (last PP stage) simultaneously
- Compare when each rank is idle vs active

### 4.2 nsys Cross-Stage Analysis
**Priority:** Medium | **Effort:** Low (traces already exist)

Job 79846 produced 3 nsys traces (rank 0, 32, 60 = first/middle/last PP stages). Analyze these with Nsight Systems GUI to visualize:
- Pipeline bubble pattern across stages
- P2P send/recv timing alignment
- Whether forward/backward waves are balanced

### 4.3 NCCL P2P Bandwidth Benchmark
**Priority:** Low | **Effort:** Medium

The NCCL report benchmarks collectives (AllReduce, AllGather, ReduceScatter) but NOT P2P send/recv. Since P2P is the dominant bottleneck:
- Run `sendrecv_perf` from nccl-tests across node pairs
- Measure point-to-point IB bandwidth between PP stage neighbors
- Identify if any node pairs have degraded P2P performance

---

## Priority Order

| Phase | Task | Priority | Expected Impact |
|-------|------|----------|----------------|
| 2.1 | Test PP=8 (TP=8, PP=8) | **Critical** | 30–50% step time reduction |
| 2.3 | Run Megatron-Bridge config (TP=4, PP=8, CP=2) | **High** | Direct comparison |
| 1.1 | Enable SHARP for 405B | **High** | ~10s/step savings on collectives |
| 1.2 | Verify comm-compute overlap | **High** | Potentially large |
| 3.2 | 512-GPU 70B NVFP4 benchmark | **High** | Scripts ready, quick win |
| 3.1 | 512-GPU 405B scale-out | **High** | After Phase 2 |
| 1.3 | Fix aten::item sync | **Medium** | 2.75s/step |
| 2.2 | Test CP=2 config | **Medium** | Alternative to TP=8 |
| 4.2 | Analyze nsys traces (already have them) | **Medium** | Better bubble understanding |
| 4.1 | Multi-rank PyTorch profiling | **Medium** | Pipeline stage comparison |
| 4.3 | NCCL P2P bandwidth benchmark | **Low** | Diagnostic |

---

## Key Metrics to Track

For each experiment, record:
- Step time (ms) — steady state, average of iterations 3–8
- TFLOP/s/GPU — from training log
- GPU utilization % — from profiler if available
- Comm time vs compute time ratio — from profiler
- Pipeline bubble fraction — from profiler
