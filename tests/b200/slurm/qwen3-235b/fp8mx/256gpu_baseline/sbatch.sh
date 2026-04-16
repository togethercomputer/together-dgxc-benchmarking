#!/bin/bash
#
# Qwen3 235B A22B FP8 MX - Optimized with CUDA graphs + NCCL fixes
# Changes vs baseline:
#   1. Added --cuda_graph_impl=transformer_engine --cuda_graph_scope=moe_router,moe_preprocess
#   2. Added NCCL_SOCKET_IFNAME=bond0 and IB HCA pinning
#   3. Added ulimit -n 1048576 for 512-GPU scale
#   4. Removed GLOO_SOCKET_IFNAME (512-GPU safe pattern)
#

# Parameters
#SBATCH --account=root
#SBATCH --exclusive
#SBATCH --job-name=root-root.pretrain_qwen3_235b_a22b_fp8_mx_gpus512_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192
#SBATCH --mem=0
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=/mnt/vast/exemplar/llmb/workloads/pretrain_qwen3/experiments/pretrain_qwen3_235b_a22b_fp8_mx_gpus512_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192/pretrain_qwen3_235b_a22b_fp8_mx_gpus512_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192_1775713925/pretrain_qwen3_235b_a22b_fp8_mx_gpus512_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192/sbatch_root-root.pretrain_qwen3_235b_a22b_fp8_mx_gpus512_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192_%j.out
#SBATCH --partition=batch
#SBATCH --time=00:40:00

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
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=False
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA="=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_11:1,mlx5_14:1,mlx5_15:1"
export UCX_NET_DEVICES=bond0
export OMPI_MCA_btl_tcp_if_include=bond0
export HF_HUB_OFFLINE=0
export HF_HOME=/mnt/vast/exemplar/llmb/.cache/huggingface
export NEMORUN_HOME=/mnt/vast/exemplar/llmb/workloads/pretrain_qwen3
export NEMO_HOME=/mnt/vast/exemplar/llmb/workloads/pretrain_qwen3
export HF_TOKEN=HF_TOKEN_REDACTED
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export PYTHONPATH=/mnt/vast/exemplar/llmb/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance:$PYTHONPATH


# Command 1

srun --output /mnt/vast/exemplar/llmb/workloads/pretrain_qwen3/experiments/pretrain_qwen3_235b_a22b_fp8_mx_gpus512_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192/pretrain_qwen3_235b_a22b_fp8_mx_gpus512_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192_1775713925/pretrain_qwen3_235b_a22b_fp8_mx_gpus512_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192/log-root-root.pretrain_qwen3_235b_a22b_fp8_mx_gpus512_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192_%j_${SLURM_RESTART_COUNT:-0}.out --container-image /mnt/vast/exemplar/llmb/images/nvidia+nemo+25.11.01.sqsh --container-mounts /mnt/vast/exemplar/llmb/workloads/pretrain_qwen3:/mnt/vast/exemplar/llmb/workloads/pretrain_qwen3,/mnt/vast/exemplar/llmb/.cache/huggingface,/mnt/vast/exemplar/llmb/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance/run_script.py:/mnt/vast/exemplar/llmb/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance/run_script.py,/mnt/vast/exemplar/llmb/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance:/mnt/vast/exemplar/llmb/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance,/mnt/vast/exemplar/llmb/workloads/pretrain_qwen3/experiments/pretrain_qwen3_235b_a22b_fp8_mx_gpus512_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192/pretrain_qwen3_235b_a22b_fp8_mx_gpus512_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192_1775713925/pretrain_qwen3_235b_a22b_fp8_mx_gpus512_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192:/nemo_run --container-workdir /nemo_run/code --wait=60 --kill-on-bad-exit=1 --mpi=pmix --no-container-mount-home --container-writable bash /nemo_run/scripts/pretrain_qwen3_235b_a22b_fp8_mx_gpus512_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192.sh

exitcode=$?

set -e

echo "job exited with code $exitcode"
if [ $exitcode -ne 0 ]; then
    if [ "$TORCHX_MAX_RETRIES" -gt "${SLURM_RESTART_COUNT:-0}" ]; then
        scontrol requeue "$SLURM_JOB_ID"
    fi
    exit $exitcode
fi
