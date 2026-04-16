#!/bin/bash
#
# Llama 3.1 405B NVFP4 - 256 GPUs (32 nodes)
# Config F: TP=4, PP=8, CP=1, VP=8, MBS=1, GBS=1536, DP=8
# Goal: Test more virtual pipeline interleaving (VP=8 vs VP=4)
#

# Parameters
#SBATCH --account=root
#SBATCH --exclusive
#SBATCH --job-name=405b_nvfp4_256gpu_tp4_pp8_vp8
#SBATCH --mem=0
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=/mnt/vast/llmb_/workloads/pretrain_llama3.1/experiments/pretrain_llama31_405b_nvfp4_gpus256_tp4_pp8_cp1_vp8_ep1_mbs1_gbs1536/pretrain_llama31_405b_nvfp4_gpus256_tp4_pp8_cp1_vp8_ep1_mbs1_gbs1536_1775839649/pretrain_llama31_405b_nvfp4_gpus256_tp4_pp8_cp1_vp8_ep1_mbs1_gbs1536/sbatch_405b_nvfp4_256gpu_tp4_pp8_vp8_%j.out
#SBATCH --partition=batch
#SBATCH --time=01:00:00

set -evx
ulimit -n 1048576

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
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=False
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA="=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_11:1,mlx5_14:1,mlx5_15:1"
export UCX_NET_DEVICES=bond0
export OMPI_MCA_btl_tcp_if_include=bond0
export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/vast/llmb_/.cache/huggingface
export NEMORUN_HOME=/mnt/vast/llmb_/workloads/pretrain_llama3.1
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export PYTHONPATH=/mnt/vast/llmb_/workloads/pretrain_llama3.1/Megatron-Bridge/scripts/performance:$PYTHONPATH


# Command 1

srun --output /mnt/vast/llmb_/workloads/pretrain_llama3.1/experiments/pretrain_llama31_405b_nvfp4_gpus256_tp4_pp8_cp1_vp8_ep1_mbs1_gbs1536/pretrain_llama31_405b_nvfp4_gpus256_tp4_pp8_cp1_vp8_ep1_mbs1_gbs1536_1775839649/pretrain_llama31_405b_nvfp4_gpus256_tp4_pp8_cp1_vp8_ep1_mbs1_gbs1536/log-405b_nvfp4_256gpu_tp4_pp8_vp8_%j_${SLURM_RESTART_COUNT:-0}.out --container-image /mnt/vast/llmb_/images/nvidia+nemo+26.02.00.sqsh --container-mounts /mnt/vast/llmb_/.cache/huggingface,/mnt/vast/llmb_/workloads/pretrain_llama3.1/Megatron-Bridge/scripts/performance/run_script.py:/mnt/vast/llmb_/workloads/pretrain_llama3.1/Megatron-Bridge/scripts/performance/run_script.py,/mnt/vast/llmb_/workloads/pretrain_llama3.1/Megatron-Bridge/scripts/performance:/mnt/vast/llmb_/workloads/pretrain_llama3.1/Megatron-Bridge/scripts/performance,/mnt/vast/llmb_/workloads/pretrain_llama3.1/experiments/pretrain_llama31_405b_nvfp4_gpus256_tp4_pp8_cp1_vp8_ep1_mbs1_gbs1536/pretrain_llama31_405b_nvfp4_gpus256_tp4_pp8_cp1_vp8_ep1_mbs1_gbs1536_1775839649/pretrain_llama31_405b_nvfp4_gpus256_tp4_pp8_cp1_vp8_ep1_mbs1_gbs1536:/nemo_run --container-workdir /nemo_run/code --wait=60 --kill-on-bad-exit=1 --mpi=pmix --no-container-mount-home --container-writable bash /nemo_run/scripts/pretrain_llama31_405b_nvfp4_gpus256_tp4_pp8_cp1_vp8_ep1_mbs1_gbs1536.sh

exitcode=$?

set -e

echo "job exited with code $exitcode"
if [ $exitcode -ne 0 ]; then
    if [ "$TORCHX_MAX_RETRIES" -gt "${SLURM_RESTART_COUNT:-0}" ]; then
        scontrol requeue "$SLURM_JOB_ID"
    fi
    exit $exitcode
fi
