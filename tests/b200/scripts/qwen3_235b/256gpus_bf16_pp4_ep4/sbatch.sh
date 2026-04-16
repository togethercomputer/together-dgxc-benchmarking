#!/bin/bash
#
# Qwen3 235B A22B BF16 - 256 GPUs (32 nodes)
# OPTIMIZATION EXP 4: PP=4 + EP=4 (attack both PP 57% + EP 16% bottlenecks)
# TP=1, PP=4, CP=1, EP=4, VP=4, MBS=1, GBS=8192, DP=64
# Changes: PP=8→4, EP=8→4
# Baseline: 57.5s/step, 337.7 TFLOP/s/GPU
#

# Parameters
#SBATCH --account=root
#SBATCH --exclusive
#SBATCH --job-name=qwen3_235b_bf16_pp4_ep4
#SBATCH --mem=0
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=/mnt/vast/llmb_/workloads/pretrain_qwen3/experiments/pretrain_qwen3_235b_a22b_bf16_gpus256_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192/pretrain_qwen3_235b_a22b_bf16_gpus256_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192_1775789255/pretrain_qwen3_235b_a22b_bf16_gpus256_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192/sbatch_bf16_pp4_ep4_%j.out
#SBATCH --partition=batch
#SBATCH --time=01:00:00

set -evx
ulimit -n 1048576 || true

# Fix: redirect enroot cache away from broken /home/johnson on compute nodes
export HOME=/tmp

export PYTHONUNBUFFERED=1
export SLURM_UNBUFFEREDIO=1
export TORCHX_MAX_RETRIES=0

set +e

# setup

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_TIMEOUT=1800
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=False
export NCCL_NVLS_ENABLE=0
export NCCL_COLLNET_ENABLE=0
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA="=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_11:1,mlx5_14:1,mlx5_15:1"
export UCX_NET_DEVICES=bond0
export OMPI_MCA_btl_tcp_if_include=bond0
export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/vast/llmb_/.cache/huggingface
export NEMORUN_HOME=/mnt/vast/llmb_/workloads/pretrain_qwen3
export NEMO_HOME=/mnt/vast/llmb_/workloads/pretrain_qwen3
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export PYTHONPATH=/mnt/vast/llmb_/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance:$PYTHONPATH


# Command 1

srun --output /mnt/vast/llmb_/workloads/pretrain_qwen3/experiments/pretrain_qwen3_235b_a22b_bf16_gpus256_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192/pretrain_qwen3_235b_a22b_bf16_gpus256_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192_1775789255/pretrain_qwen3_235b_a22b_bf16_gpus256_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192/log-bf16_pp4_ep4_%j_${SLURM_RESTART_COUNT:-0}.out --container-image /mnt/vast/llmb_/images/nvidia+nemo+26.02.00.sqsh --container-mounts /mnt/vast/llmb_/.cache/huggingface,/mnt/vast/llmb_/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance/run_script.py:/mnt/vast/llmb_/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance/run_script.py,/mnt/vast/llmb_/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance:/mnt/vast/llmb_/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance,/mnt/vast/llmb_/workloads/pretrain_qwen3/experiments/pretrain_qwen3_235b_a22b_bf16_gpus256_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192/pretrain_qwen3_235b_a22b_bf16_gpus256_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192_1775789255/pretrain_qwen3_235b_a22b_bf16_gpus256_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192:/nemo_run --container-workdir /nemo_run/code --wait=60 --kill-on-bad-exit=1 --mpi=pmi2 --no-container-mount-home --container-writable bash /nemo_run/opt_scripts/bf16_pp4_ep4.sh

exitcode=$?

set -e

echo "job exited with code $exitcode"
if [ $exitcode -ne 0 ]; then
    if [ "$TORCHX_MAX_RETRIES" -gt "${SLURM_RESTART_COUNT:-0}" ]; then
        scontrol requeue "$SLURM_JOB_ID"
    fi
    exit $exitcode
fi
