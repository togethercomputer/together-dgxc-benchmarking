#!/bin/bash
#
# Nemotron4 15B FP8 - 256 GPUs - EXPERIMENT 7: GBS=2560
# 2560/(256*2) = 5 microbatches per DP rank. More overlap.
# Current best (GBS=2048): 1,885 TFLOPS/GPU. Target: 1,908.
#

# Parameters
#SBATCH --account=root
#SBATCH --exclusive
#SBATCH --job-name=nem15b_fp8_256gpu_exp7_gbs2560
#SBATCH --mem=0
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=/mnt/vast/johnson/llmb/workloads/pretrain_nemotron4-15b/experiments/pretrain_nemotron4_15b_fp8_gpus256_tp1_pp1_cp1_vp1_mbs2_gbs512/pretrain_nemotron4_15b_fp8_gpus256_tp1_pp1_cp1_vp1_mbs2_gbs512_1776133053/pretrain_nemotron4_15b_fp8_gpus256_tp1_pp1_cp1_vp1_mbs2_gbs512/sbatch_exp7_gbs2560_%j.out
#SBATCH --partition=batch
#SBATCH --time=00:30:00

set -evx
ulimit -n 1048576 || true

export PYTHONUNBUFFERED=1
export SLURM_UNBUFFEREDIO=1
export TORCHX_MAX_RETRIES=0

set +e

# Fix pyxis home directory issue
export HOME=/tmp

# setup
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Performance env vars
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/mnt/vast/llmb_/.cache/huggingface
export TOKENIZERS_PARALLELISM=False
export NCCL_NVLS_ENABLE=0
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=1
export NEMO_LOG_MEMORY_USAGE=1
export NEMORUN_HOME=/mnt/vast/johnson/llmb/workloads/pretrain_nemotron4-15b
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export TORCH_NCCL_HIGH_PRIORITY=1

# Fix NeMo trying to mkdir in read-only shared HF cache
export NEMO_NLP_TMP=/tmp/nemo_nlp_tmp

# EXPERIMENT 7: GBS=2560 (5 microbatches per DP rank)
export OVERRIDE_GBS=2560
export OVERRIDE_NUM_TRAIN_SAMPLES=128000

# Cluster-specific NCCL settings
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA="=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_11:1,mlx5_14:1,mlx5_15:1"
export UCX_NET_DEVICES=bond0
export OMPI_MCA_btl_tcp_if_include=bond0
export CUDA_DEVICE_MAX_CONNECTIONS=32

# Command 1
srun --output /mnt/vast/johnson/llmb/workloads/pretrain_nemotron4-15b/experiments/pretrain_nemotron4_15b_fp8_gpus256_tp1_pp1_cp1_vp1_mbs2_gbs512/pretrain_nemotron4_15b_fp8_gpus256_tp1_pp1_cp1_vp1_mbs2_gbs512_1776133053/pretrain_nemotron4_15b_fp8_gpus256_tp1_pp1_cp1_vp1_mbs2_gbs512/log-exp7_gbs2560_%j_${SLURM_RESTART_COUNT:-0}.out \
  --container-image /mnt/vast/johnson/llmb/images/nvidia+nemo+26.02.00.sqsh \
  --container-mounts /mnt/vast/llmb_/.cache/huggingface,/mnt/vast/johnson/llmb/workloads/pretrain_nemotron4-15b/experiments/pretrain_nemotron4_15b_fp8_gpus256_tp1_pp1_cp1_vp1_mbs2_gbs512/pretrain_nemotron4_15b_fp8_gpus256_tp1_pp1_cp1_vp1_mbs2_gbs512_1776133053/pretrain_nemotron4_15b_fp8_gpus256_tp1_pp1_cp1_vp1_mbs2_gbs512:/nemo_run \
  --container-workdir /nemo_run/code \
  --wait=60 \
  --kill-on-bad-exit=1 \
  --mpi=pmix \
  --no-container-mount-home \
  --container-writable \
  --container-env=TORCH_NCCL_AVOID_RECORD_STREAMS,TRANSFORMERS_OFFLINE,TOKENIZERS_PARALLELISM,NCCL_NVLS_ENABLE,NVTE_FLASH_ATTN,NVTE_FUSED_ATTN,NEMO_LOG_MEMORY_USAGE,NEMORUN_HOME,NCCL_SOCKET_IFNAME,NCCL_IB_HCA,UCX_NET_DEVICES,OMPI_MCA_btl_tcp_if_include,CUDA_DEVICE_MAX_CONNECTIONS,HOME,HF_HOME,NEMO_NLP_TMP,OVERRIDE_GBS,OVERRIDE_NUM_TRAIN_SAMPLES \
  bash /nemo_run/scripts/pretrain_nemotron4_15b_fp8_gpus256_tp1_pp1_cp1_vp1_mbs2_gbs512.sh

exitcode=$?

set -e

echo "job exited with code $exitcode"
if [ $exitcode -ne 0 ]; then
    if [ "$TORCHX_MAX_RETRIES" -gt "${SLURM_RESTART_COUNT:-0}" ]; then
        scontrol requeue "$SLURM_JOB_ID"
    fi
    exit $exitcode
fi
