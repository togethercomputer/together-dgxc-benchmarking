# Llama 70B FP8 Scaling Benchmark Report — DGXC B200 Cluster

**Date:** 2026-04-08
**Cluster:** us-east-3a-forge-exemplar-testing (Together AI)
**Conducted by:** Johnson

---

## 1. Overview

This report documents the **weak-scaling benchmark** for **Llama 3 70B pretraining with FP8 precision** across five GPU configurations on the DGXC B200 cluster: 64, 128, 256, 512, and 592 GPUs (full cluster). The goal is to characterize per-GPU throughput, aggregate throughput, and scaling efficiency as the job size increases from 8 to 74 nodes.

---

## 2. Configuration

### Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA B200 (183 GB HBM per GPU) |
| GPUs per Node | 8 |
| Cluster Size | 74 batch nodes (592 GPUs) |
| Interconnect | InfiniBand (mlx5, 8 HCAs per node) |
| Network Interface | bond0 (7.247.232.x/24) |
| NUMA Topology | 2 NUMA nodes per server |
| Partition | batch |

### Software

| Component | Version |
|-----------|---------|
| Container | nvidia/nemo:25.11.01 (squashfs) |
| Framework | Megatron-Bridge + NeMo Run |
| CUDA | 12.9 |
| NVIDIA Driver | 575.57.08 |
| NCCL | 2.28.3 |

### Model & Training Configuration (Common)

| Parameter | Value |
|-----------|-------|
| Model | Llama 3 70B |
| Precision | FP8 (Current Scaling / cs recipe) |
| Tensor Parallelism (TP) | 1 |
| Pipeline Parallelism (PP) | 1 |
| Context Parallelism (CP) | 1 |
| Expert Parallelism (EP) | 1 |
| FSDP | Enabled (Megatron FSDP) |
| CPU Offloading Layers | 5 |
| Micro Batch Size (MBS) | 1 |
| Max Steps | 50 |
| Manual GC | Enabled (interval=100) |

### Per-Scale Configuration

| GPUs | Nodes | GBS | Samples/GPU | Time Limit |
|------|-------|-----|-------------|------------|
| 64 | 8 | 128 | 2 | 00:30:00 |
| 128 | 16 | 256 | 2 | 00:30:00 |
| 256 | 32 | 512 | 2 | 00:30:00 |
| 512 | 64 | 1024 | 2 | 00:30:00 |
| 592 | 74 | 1184 | 2 | 00:30:00 |

> GBS scales linearly with GPU count (weak scaling: 2 samples per GPU per step).

---

## 3. Scaling Results

### Summary

| GPUs | Nodes | Avg Step Time | TFLOP/s/GPU | Total TFLOP/s | Scaling Eff. | Wall Time | Job ID(s) |
|------|-------|---------------|-------------|---------------|--------------|-----------|-----------|
| 64 | 8 | 4.63s | 1,589.7 | 101,737 | baseline | 00:08:34 | 79118-79120 |
| 128 | 16 | 4.63s | 1,591.5 | 203,712 | 100.1% | 00:08:50 | 79138 |
| 256 | 32 | 4.92s | 1,498.8 | 383,693 | 94.3% | 00:09:23 | 79139 |
| 512 | 64 | 5.39s | 1,366.6 | 699,688 | 86.0% | 00:10:36 | 79133-79135 |
| 592 | 74 | 5.59s | 1,318.5 | 780,573 | 82.9% | 00:11:07 | 79140 |

### Key Performance Numbers

| Metric | Value |
|--------|-------|
| **Peak Per-GPU Throughput** | **1,591.5 MODEL_TFLOP/s/GPU** (128 GPUs) |
| **Peak Aggregate Throughput** | **780,573 MODEL_TFLOP/s** (592 GPUs, full cluster) |
| **Best Scaling Efficiency** | **100.1%** (64 → 128 GPUs) |
| **Full-Cluster Efficiency** | **82.9%** (64 → 592 GPUs) |
| **Reproducibility (512 GPUs)** | **0.08% variance** across 3 runs |

### Warmup & Initialization

| GPUs | Warmup (Step 1) | Init to Step 1 | Total Overhead |
|------|-----------------|-----------------|----------------|
| 64 | 37.3s | ~3 min | ~3.6 min |
| 128 | 37.8s | ~3.5 min | ~4.1 min |
| 256 | 38.8s | ~4 min | ~4.6 min |
| 512 | 39.5s | ~5 min | ~5.7 min |
| 592 | 41.3s | ~5.5 min | ~6.2 min |

### GPU Memory Usage (Rank 0, after step 1)

| GPUs | Max Allocated | Max Reserved | Headroom |
|------|---------------|--------------|----------|
| 64 | 164.4 GB | 171.9 GB | 11.1 GB |
| 128 | 160.5 GB | 166.8 GB | 16.2 GB |
| 256 | 158.6 GB | 165.9 GB | 17.1 GB |
| 512 | 157.8 GB | 163.1 GB | 19.9 GB |
| 592 | 157.6 GB | 162.6 GB | 20.4 GB |

> Memory per GPU decreases with scale as FSDP shards optimizer states across more ranks.

---

## 4. Scaling Analysis

### Efficiency Curve

```
TFLOP/s/GPU
1600 |*--*
     |     \
1500 |      *
     |       \
1400 |        \
     |         *
1300 |          *
     |
1200 +---+---+---+---+---
     64  128 256 512 592
              GPUs
```

### Scaling Behavior

| Transition | Efficiency | Step Time Delta | Analysis |
|------------|------------|-----------------|----------|
| 64 → 128 | 100.1% | +0.00s | Perfect linear scaling. FSDP all-reduce within 16 nodes is fully overlapped with compute. |
| 128 → 256 | 94.3% | +0.29s | First measurable communication overhead. All-reduce across 32 nodes begins to exceed compute overlap budget. |
| 256 → 512 | 86.0% | +0.47s | Continued degradation from FSDP gradient synchronization across 64 nodes. |
| 512 → 592 | 82.9% | +0.20s | Modest additional drop. The non-power-of-2 topology (74 nodes) does not cause additional penalty beyond the node count increase. |

### Throughput Scaling

| GPUs | Ideal Total TFLOP/s | Actual Total TFLOP/s | Utilization |
|------|---------------------|----------------------|-------------|
| 64 | 101,737 (baseline) | 101,737 | 100% |
| 128 | 203,474 | 203,712 | 100.1% |
| 256 | 406,948 | 383,693 | 94.3% |
| 512 | 813,896 | 699,688 | 86.0% |
| 592 | 943,102 | 780,573 | 82.9% |

---

## 5. Reproducibility (512 GPUs)

Three consecutive runs at 512 GPUs validated reproducibility:

| Metric | Run 1 (79133) | Run 2 (79134) | Run 3 (79135) | Avg |
|--------|---------------|---------------|---------------|-----|
| Step Time | 5.39s | 5.39s | 5.39s | 5.39s |
| TFLOP/s/GPU | 1,366.4 | 1,367.2 | 1,366.1 | 1,366.6 |
| Wall Time | 10m36s | 10m34s | 10m37s | 10m36s |

**Cross-run variance: 0.08%** (std dev 2.2ms)

---

## 6. Per-Step Performance (Representative Runs)

### 64 GPUs (Job 79118)

| Step | Time (ms) | TFLOP/s/GPU | Loss |
|------|-----------|-------------|------|
| 1 | 37,466 | 196.4 | 13.339 |
| 2 | 4,611 | 1,596.2 | 13.399 |
| 10 | 4,628 | 1,600.0 | 7.827 |
| 30 | 4,601 | 1,599.3 | 0.046 |
| 50 | 4,674 | 1,574.7 | 0.012 |

### 512 GPUs (Job 79133)

| Step | Time (ms) | TFLOP/s/GPU | Loss |
|------|-----------|-------------|------|
| 1 | 39,370 | 187.1 | 13.339 |
| 2 | 5,396 | 1,365.5 | 13.400 |
| 10 | 5,402 | 1,363.9 | 3.251 |
| 30 | 5,396 | 1,365.6 | 0.066 |
| 50 | 5,392 | 1,366.6 | 0.007 |

### 592 GPUs (Job 79140)

| Step | Time (ms) | TFLOP/s/GPU | Loss |
|------|-----------|-------------|------|
| 1 | 41,345 | 178.2 | 13.339 |
| 2 | 5,577 | 1,321.2 | 13.400 |
| 10 | 5,602 | 1,315.3 | 3.155 |
| 30 | 5,579 | 1,320.7 | 1.388 |
| 50 | 5,592 | 1,317.6 | 0.011 |

---

## 7. Job Details

### All Runs

| GPUs | Job ID | Status | Wall Time | Nodes |
|------|--------|--------|-----------|-------|
| 64 | 79118 | COMPLETED | 00:08:31 | gpu-[149-152,154-157] |
| 64 | 79119 | COMPLETED | 00:08:38 | gpu-[149-152,154-157] |
| 64 | 79120 | COMPLETED | 00:08:34 | gpu-[149-152,154-157] |
| 128 | 79138 | COMPLETED | 00:08:50 | gpu-[159-174] |
| 256 | 79139 | COMPLETED | 00:09:23 | gpu-[175-190,195-205,233-235,238-239] |
| 512 | 79133 | COMPLETED | 00:10:36 | gpu-[144-145,159-190,195-205,209-217,226-229,233-235,238-239,256] |
| 512 | 79134 | COMPLETED | 00:10:34 | (same as above) |
| 512 | 79135 | COMPLETED | 00:10:37 | (same as above) |
| 592 | 79140 | COMPLETED | 00:11:07 | All 74 batch nodes including gpu-148 |

---

## 8. Issues & Resolutions

### Scale-Specific Issues

| Issue | Affected Scale | Resolution |
|-------|---------------|------------|
| Container permission error | All (initial) | Run as root via `sudo` |
| NCCL wrong interface | All | `NCCL_SOCKET_IFNAME=bond0` + IB HCA pinning |
| HuggingFace cache miss | All | `HF_HOME=/mnt/vast/exemplar/llmb/.cache/huggingface` |
| Missing `code` directory | 128, 256 (new experiments) | Create empty `code/` dir in experiment path |
| Gloo connection refused | 512+ GPUs | Remove `GLOO_SOCKET_IFNAME=bond0` |
| File descriptor exhaustion | 512+ GPUs | `ulimit -n 1048576` |
| Segfault on gpu-148 | 64 GPUs (Job 79113) | Excluded; later 592-GPU run passed without issue |
| OOM with MBS=2 | 512 GPUs (Job 79141) | 171 GB allocated exceeded 178 GB capacity. MBS=2 not feasible without activation checkpointing or more CPU offloading. |

### Sbatch Patch Matrix

| Patch | 64 | 128 | 256 | 512 | 592 |
|-------|----|----|-----|-----|-----|
| NCCL_SOCKET_IFNAME=bond0 | Yes | Yes | Yes | Yes | Yes |
| NCCL_IB_HCA, UCX, OMPI | Yes | Yes | Yes | Yes | Yes |
| HF_HOME | Yes | Yes | Yes | Yes | Yes |
| GLOO_SOCKET_IFNAME=bond0 | Yes | Yes | No | No | No |
| ulimit -n 1048576 | No | No | Yes | Yes | Yes |
| code/ directory | Auto | Manual | Manual | Auto | Manual |

---

## 9. Recommendations

1. **Upstream all sbatch patches** — The NeMo Run template generator should include NCCL interface pinning, HF_HOME, ulimit, and create the `code/` directory by default.

2. **Use the 512-GPU safe pattern for all scales** — Omitting `GLOO_SOCKET_IFNAME` and adding `ulimit -n 1048576` works at all scales. This avoids maintaining separate configs.

3. **MBS=2 is not feasible (tested)** — Job 79141 attempted MBS=2 at 512 GPUs and hit OOM (171 GB allocated vs 178 GB capacity, failed on 896 MiB allocation). The ~20 GB headroom at MBS=1 is insufficient for the doubled activation memory. To enable MBS=2, increase CPU offloading layers (currently 5), enable activation checkpointing, or use pipeline parallelism.

4. **Triage gpu-148** — The node passed at 592 GPUs but previously segfaulted at 64 GPUs. Run isolated diagnostics to confirm reliability.

5. **Consider 128 GPUs as the efficiency sweet spot** — 100% scaling efficiency with 1,591 TFLOP/s/GPU makes 128 GPUs (16 nodes) the most efficient configuration for this model.

---

## 10. Artifact Locations

| Artifact | Path |
|----------|------|
| 64-GPU experiment | `/mnt/vast/exemplar/llmb/workloads/pretrain_llama3.1/experiments/pretrain_llama3_70b_fp8_cs_gpus64_tp1_pp1_cp1_vpNone_ep1_mbs1_gbs128/` |
| 128-GPU experiment | `/mnt/vast/exemplar/llmb/workloads/pretrain_llama3.1/experiments/pretrain_llama3_70b_fp8_cs_gpus128_tp1_pp1_cp1_vpNone_ep1_mbs1_gbs256/` |
| 256-GPU experiment | `/mnt/vast/exemplar/llmb/workloads/pretrain_llama3.1/experiments/pretrain_llama3_70b_fp8_cs_gpus256_tp1_pp1_cp1_vpNone_ep1_mbs1_gbs512/` |
| 512-GPU experiment | `/mnt/vast/exemplar/llmb/workloads/pretrain_llama3.1/experiments/pretrain_llama3_70b_fp8_cs_gpus512_tp1_pp1_cp1_vpNone_ep1_mbs1_gbs1024/` |
| 592-GPU experiment | `/mnt/vast/exemplar/llmb/workloads/pretrain_llama3.1/experiments/pretrain_llama3_70b_fp8_cs_gpus592_tp1_pp1_cp1_vpNone_ep1_mbs1_gbs1184/` |
| Scripts | `~/johnson/` |
| 64-GPU report | `~/johnson/Llama_70B_FP8_Benchmark_Report.md` |
| 512-GPU report | `~/johnson/Llama_70B_FP8_512GPU_Benchmark_Report.md` |
| This report | `~/johnson/Llama_70B_FP8_Scaling_Benchmark_Report.md` |
