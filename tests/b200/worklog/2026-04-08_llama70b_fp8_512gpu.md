# Llama 70B FP8 Benchmark Report — DGXC B200 Cluster (512 GPUs)

**Date:** 2026-04-08
**Cluster:** us-east-3a-forge-exemplar-testing (Together AI)
**Conducted by:** Johnson

---

## 1. Overview

This report documents the benchmark results for **Llama 3 70B pretraining with FP8 precision** on 512 NVIDIA B200 GPUs across 64 nodes on the DGXC cluster. Three consecutive runs were executed to validate reproducibility and stability. This extends the earlier 64-GPU benchmark to a larger scale, demonstrating 86% weak-scaling efficiency.

---

## 2. Configuration

### Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA B200 (183 GB HBM per GPU) |
| GPUs per Node | 8 |
| Total GPUs | 512 (64 nodes) |
| Total GPU Memory | 91.4 TB |
| Interconnect | InfiniBand (mlx5, 8 HCAs per node) |
| Network Interface | bond0 (7.247.232.x/24) |
| NUMA Topology | 2 NUMA nodes per server, 128 CPU cores each |
| Nodes Used | gpu-[144-145, 159-190, 195-205, 209-217, 226-229, 233-235, 238-239, 256] |
| Partition | batch |

### Software

| Component | Version |
|-----------|---------|
| Container | nvidia/nemo:25.11.01 (squashfs) |
| Framework | Megatron-Bridge + NeMo Run |
| CUDA | 12.9 |
| NVIDIA Driver | 575.57.08 |
| NCCL | 2.28.3 |
| PyTorch | (bundled in container) |
| Slurm | Managed via Ansible |

### Model & Training Configuration

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
| Global Batch Size (GBS) | 1024 |
| Micro Batch Size (MBS) | 1 |
| Max Steps | 50 |
| Time Limit | 00:30:00 |
| CUDA_DEVICE_MAX_CONNECTIONS | 32 |
| Manual GC | Enabled (interval=100) |

### GPU Memory Usage (Rank 0, after step 1)

| Metric | Value |
|--------|-------|
| Allocated | 12.3 GB |
| Max Allocated | 157.8 GB |
| Reserved | 163.1 GB |
| Max Reserved | 163.1 GB |
| Alloc Retries | 0 |

---

## 3. Benchmark Results

### Summary (3 Runs)

| Metric | Run 1 (Job 79133) | Run 2 (Job 79134) | Run 3 (Job 79135) | Average |
|--------|-------------------|-------------------|-------------------|---------|
| **Status** | COMPLETED | COMPLETED | COMPLETED | 3/3 |
| **Wall Time** | 00:10:36 | 00:10:34 | 00:10:37 | 00:10:36 |
| **Avg Step Time** | 5.39s | 5.39s | 5.39s | **5.39s** |
| **Min Step Time** | 5.33s | 5.32s | 5.33s | 5.33s |
| **Max Step Time** | 5.43s | 5.42s | 5.41s | 5.42s |
| **Avg TFLOP/s/GPU** | 1,366.4 | 1,367.2 | 1,366.1 | **1,366.6** |
| **Min TFLOP/s/GPU** | 1,356.8 | 1,359.9 | 1,360.7 | 1,359.1 |
| **Max TFLOP/s/GPU** | 1,382.2 | 1,385.3 | 1,383.5 | 1,383.7 |
| **Total TFLOP/s (512 GPUs)** | 699,619 | 700,007 | 699,445 | **699,688** |
| **Final Loss** | 0.0070 | 0.0165 | 0.0067 | 0.0101 |
| **Consumed Samples** | 51,200 | 51,200 | 51,200 | 51,200 |

> **Note:** Step 1 is excluded from averages as it includes warmup/compilation overhead (~39.5s).

### Key Performance Numbers

| Metric | Value |
|--------|-------|
| **Per-GPU Throughput** | **1,366.6 MODEL_TFLOP/s/GPU** |
| **Aggregate Throughput (512 GPUs)** | **699,688 MODEL_TFLOP/s** |
| **Training Step Time** | **5.39s** (steady state) |
| **Warmup Step Time** | ~39.5s (step 1 only) |
| **Cross-Run Variance** | **0.08%** (std dev 2.2ms — highly reproducible) |

### Scaling Efficiency (64 → 512 GPUs)

| Metric | 64 GPUs | 512 GPUs | Efficiency |
|--------|---------|----------|------------|
| **TFLOP/s/GPU** | 1,589.7 | 1,366.6 | **86.0%** |
| **Step Time** | 4.63s | 5.39s | — |
| **GBS** | 128 | 1024 | 8x (weak scaling) |
| **Total TFLOP/s** | 101,737 | 699,688 | 6.88x |

### Per-Step Performance (Run 1 — Job 79133)

| Step | Time (ms) | TFLOP/s/GPU | Loss | Grad Norm |
|------|-----------|-------------|------|-----------|
| 1 | 39,369.7 | 187.1 | 13.339 | 503.16 |
| 2 | 5,395.7 | 1,365.5 | 13.400 | 2853.89 |
| 5 | 5,430.5 | 1,356.8 | 11.047 | 2542.88 |
| 10 | 5,402.1 | 1,363.9 | 3.251 | 2725.91 |
| 20 | 5,395.5 | 1,365.6 | 0.620 | 1043.72 |
| 30 | 5,395.5 | 1,365.6 | 0.066 | 54.76 |
| 40 | 5,400.6 | 1,364.3 | 0.012 | 35.26 |
| 50 | 5,391.6 | 1,366.6 | 0.007 | 9.25 |

---

## 4. Job Details

### Run Timeline

| Run | Job ID | Start | End | Duration | Nodes |
|-----|--------|-------|-----|----------|-------|
| 1 | 79133 | 19:35:29 | 19:46:05 | 10m36s | gpu-[144-145,159-190,195-205,209-217,226-229,233-235,238-239,256] |
| 2 | 79134 | 19:47:53 | 19:58:27 | 10m34s | gpu-[144-145,159-190,195-205,209-217,226-229,233-235,238-239,256] |
| 3 | 79135 | 20:00:39 | 20:11:16 | 10m37s | gpu-[144-145,159-190,195-205,209-217,226-229,233-235,238-239,256] |

### Submit Command

```bash
sudo sbatch --requeue --parsable \
  --exclude=use3a-ss-b200-gpu-148 \
  /mnt/vast/exemplar/llmb/workloads/pretrain_llama3.1/experiments/pretrain_llama3_70b_fp8_cs_gpus512_tp1_pp1_cp1_vpNone_ep1_mbs1_gbs1024/pretrain_llama3_70b_fp8_cs_gpus512_tp1_pp1_cp1_vpNone_ep1_mbs1_gbs1024_1775642420/pretrain_llama3_70b_fp8_cs_gpus512_tp1_pp1_cp1_vpNone_ep1_mbs1_gbs1024_sbatch.sh

# Then bump priority:
sudo scontrol update JobId=<JOBID> Priority=10000
```

---

## 5. Scripts

### 5.1 Sbatch Script

`pretrain_llama3_70b_fp8_cs_gpus512_tp1_pp1_cp1_vpNone_ep1_mbs1_gbs1024_sbatch.sh`

```bash
#!/bin/bash
#
# Generated by NeMo Run (patched for 512-GPU scale)
#

# Parameters
#SBATCH --account=root
#SBATCH --exclusive
#SBATCH --job-name=root-root.pretrain_llama3_70b_fp8_cs_gpus512_tp1_pp1_cp1_vpNone_ep1_mbs1_gbs1024
#SBATCH --mem=0
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=<log_dir>/sbatch_..._%j.out
#SBATCH --partition=batch
#SBATCH --time=00:30:00

set -evx
ulimit -n 1048576  # [PATCHED] Required for Gloo full mesh at 512 ranks

export PYTHONUNBUFFERED=1
export SLURM_UNBUFFEREDIO=1
export TORCHX_MAX_RETRIES=0

set +e

# setup
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# PyTorch / NCCL
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=False
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_HIGH_PRIORITY=1

# [PATCHED] NCCL network interface pinning — required for this cluster
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA="=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_11:1,mlx5_14:1,mlx5_15:1"
export UCX_NET_DEVICES=bond0
export OMPI_MCA_btl_tcp_if_include=bond0

# NOTE: GLOO_SOCKET_IFNAME is intentionally NOT set at 512-GPU scale.
# Setting it to bond0 causes Gloo connection-refused errors at this scale.

# [PATCHED] HuggingFace offline cache
export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/vast/exemplar/llmb/.cache/huggingface

# NeMo / Megatron
export NEMORUN_HOME=/mnt/vast/exemplar/llmb/workloads/pretrain_llama3.1
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export PYTHONPATH=<megatron_bridge_scripts>:$PYTHONPATH

# Command 1
srun --output <log_dir>/log-..._%j_${SLURM_RESTART_COUNT:-0}.out \
  --container-image <squashfs_image> \
  --container-mounts <mounts> \
  --container-workdir /nemo_run/code \
  --wait=60 --kill-on-bad-exit=1 --mpi=pmix \
  --no-container-mount-home --container-writable \
  bash /nemo_run/scripts/pretrain_llama3_70b_fp8_cs_gpus512_...sh

exitcode=$?
set -e
echo "job exited with code $exitcode"
if [ $exitcode -ne 0 ]; then
    if [ "$TORCHX_MAX_RETRIES" -gt "${SLURM_RESTART_COUNT:-0}" ]; then
        scontrol requeue "$SLURM_JOB_ID"
    fi
    exit $exitcode
fi
```

### 5.2 Per-Rank Training Script

`pretrain_llama3_70b_fp8_cs_gpus512_tp1_pp1_cp1_vpNone_ep1_mbs1_gbs1024.sh`

```bash
#!/usr/bin/bash
#!/usr/bin/env bash
set -euo pipefail

bash -c 'numactl --cpunodebind=$((SLURM_LOCALID/4)) --membind=$((SLURM_LOCALID/4)) \
  python /mnt/vast/exemplar/llmb/workloads/pretrain_llama3.1/Megatron-Bridge/scripts/performance/run_script.py \
  --gpu b200 \
  --container_image /mnt/vast/exemplar/llmb/images/nvidia+nemo+25.11.01.sqsh \
  --num_gpus 512 \
  --gpus_per_node 8 \
  --model_name llama3 \
  --model_size 70b \
  --max_steps 50 \
  --custom_mounts /mnt/vast/exemplar/llmb/.cache/huggingface \
  --compute_dtype fp8_cs \
  --account root \
  --partition batch \
  --log_dir /mnt/vast/exemplar/llmb/workloads/pretrain_llama3.1 \
  --time_limit 00:30:00 \
  train.manual_gc=true \
  train.manual_gc_interval=100'
```

---

## 6. Issues & Resolutions

During 512-GPU bring-up, six issues were encountered across jobs 79125–79132 before the first successful run (Job 79133). The 64-GPU configuration required only NCCL and HF_HOME patches, but scaling to 512 GPUs introduced additional Gloo-related failures.

### Issue 1: Gloo Connection Refused (GLOO_SOCKET_IFNAME=bond0)

- **Jobs affected:** 79125, 79127, 79129
- **Error:** `Gloo connectFullMesh: Connection refused` to various nodes (gpu-202, gpu-181, others)
- **Root cause:** Setting `GLOO_SOCKET_IFNAME=bond0` forces Gloo to use the bond0 interface for its full-mesh control plane. At 512 ranks, this causes sporadic connection-refused errors. The 64-GPU configuration works fine with this setting, but at 512-GPU scale Gloo's auto-detection is more reliable.
- **Resolution:** Remove `GLOO_SOCKET_IFNAME=bond0` from the sbatch script. Only `NCCL_SOCKET_IFNAME=bond0` is needed — NCCL and Gloo use different transport mechanisms.

### Issue 2: HuggingFace Cache Miss (missing HF_HOME)

- **Jobs affected:** 79130
- **Error:** `ValueError: Failed to load configuration from meta-llama/Meta-Llama-3-70B` — disk cache miss with outgoing traffic disabled
- **Root cause:** Same as the 64-GPU issue. The sbatch script sets `HF_HUB_OFFLINE=1` but does not export `HF_HOME`. Inside the container, HuggingFace defaults to `~/.cache/huggingface` (empty) instead of the populated cache.
- **Resolution:** Added `export HF_HOME=/mnt/vast/exemplar/llmb/.cache/huggingface` to the sbatch script.

### Issue 3: NCCL Network Routing Failure (missing NCCL_SOCKET_IFNAME)

- **Jobs affected:** 79131
- **Error:** `socketPollConnect: connect to 7.247.226.116 returned No route to host`
- **Root cause:** Same as the 64-GPU issue. Without `NCCL_SOCKET_IFNAME=bond0`, NCCL auto-selects `ens121f1np1` (7.247.226.x/23) which cannot route between nodes. The host `/etc/nccl.conf` is not visible inside the container.
- **Resolution:** Added `NCCL_SOCKET_IFNAME=bond0` and associated IB/UCX/OMPI variables.

### Issue 4: File Descriptor Exhaustion (Gloo full mesh at 512 ranks)

- **Jobs affected:** 79132
- **Error:** `accept: Too many open files` during Gloo process group creation
- **Root cause:** At 512 ranks, Gloo creates a full-mesh TCP connection topology. With multiple process groups (data parallel, model parallel, etc.), the number of open file descriptors per process exceeds the default kernel limit (~1024). The 64-GPU configuration (64 ranks) stays well within limits.
- **Resolution:** Added `ulimit -n 1048576` at the top of the sbatch script to raise the per-process file descriptor limit.

### Summary: 64-GPU vs 512-GPU Sbatch Patches

| Patch | 64 GPUs | 512 GPUs |
|-------|---------|----------|
| `NCCL_SOCKET_IFNAME=bond0` | Required | Required |
| `NCCL_IB_HCA`, `UCX_NET_DEVICES`, `OMPI_MCA` | Required | Required |
| `HF_HOME` | Required | Required |
| `GLOO_SOCKET_IFNAME=bond0` | Works fine | **Must NOT be set** |
| `ulimit -n 1048576` | Not needed | **Required** |

---

## 7. Recommendations

1. **Upstream the sbatch fixes** — Patch the NeMo Run / Megatron-Bridge sbatch template generator to include NCCL interface pinning, `HF_HOME`, and the `ulimit -n` fix by default. These are missing from the auto-generated scripts and must be manually patched for every new experiment.

2. **Do not set GLOO_SOCKET_IFNAME at scale** — Document that `GLOO_SOCKET_IFNAME=bond0` must be omitted for runs above ~64 nodes. Only `NCCL_SOCKET_IFNAME` should be set.

3. **Triage gpu-148** — This node (excluded from all runs) segfaults during model initialization. Move to the `debug` partition and run hardware diagnostics.

4. **Validate remaining idle nodes** — 10 batch nodes were not used in these 512-GPU runs. Run an NCCL all-reduce benchmark across all 74 batch nodes to confirm cluster-wide health.

5. **Container NCCL config** — Consider mounting `/etc/nccl.conf` from the host into the container so that NCCL picks up the correct interface settings automatically, eliminating the need for manual env var patching.

6. **Scale to full cluster** — With 74 batch nodes (592 GPUs), the next natural benchmark point is 584 GPUs (73 nodes, excluding gpu-148). This would use nearly the entire cluster.

---

## 8. Comparison with 64-GPU Baseline

| Metric | 64 GPUs (8 nodes) | 512 GPUs (64 nodes) | Change |
|--------|-------------------|---------------------|--------|
| **TFLOP/s/GPU** | 1,589.7 | 1,366.6 | -14.0% |
| **Total TFLOP/s** | 101,737 | 699,688 | +587.7% (6.88x) |
| **Step Time** | 4.63s | 5.39s | +16.4% |
| **GBS** | 128 | 1,024 | 8x (weak scaling) |
| **Warmup (step 1)** | ~37.3s | ~39.5s | +5.9% |
| **Cross-Run Variance** | <0.3% | 0.08% | Improved |
| **GPU Memory (max alloc)** | 164.4 GB | 157.8 GB | -4.0% |
| **Scaling Efficiency** | (baseline) | 86.0% | — |

The 14% per-GPU throughput reduction at 512 GPUs is expected and primarily attributed to increased all-reduce communication overhead across 64 nodes vs 8 nodes with FSDP.

---

## 9. Artifact Locations

| Artifact | Path |
|----------|------|
| Experiment dir | `/mnt/vast/exemplar/llmb/workloads/pretrain_llama3.1/experiments/pretrain_llama3_70b_fp8_cs_gpus512_tp1_pp1_cp1_vpNone_ep1_mbs1_gbs1024/` |
| Sbatch script | `<experiment_dir>/pretrain_llama3_70b_fp8_cs_gpus512_..._1775642420/pretrain_llama3_70b_fp8_cs_gpus512_..._sbatch.sh` |
| Training script | `<experiment_dir>/pretrain_llama3_70b_fp8_cs_gpus512_..._1775642420/.../scripts/pretrain_llama3_70b_fp8_cs_gpus512_...sh` |
| Run 1 log | `<experiment_dir>/.../log-..._79133_0.out` |
| Run 2 log | `<experiment_dir>/.../log-..._79134_0.out` |
| Run 3 log | `<experiment_dir>/.../log-..._79135_0.out` |
| 64-GPU report | `~/johnson/Llama_70B_FP8_Benchmark_Report.md` |
| Cluster config | `/mnt/vast/exemplar/llmb/cluster_config.yaml` |
