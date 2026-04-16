#!/bin/bash
#
# Qwen3 235B A22B BF16 - 256 GPUs (32 nodes)
# REPRODUCTION of Max's job 80044 (518.5 TFLOP/s/GPU)
# Using Max's Megatron-Bridge (6b3b5ba7e) + dgxc-benchmarking configs
# Only change: output paths point to our directory
#

# Parameters
#SBATCH --account=root
#SBATCH --exclusive
#SBATCH --job-name=qwen3_235b_bf16_mryab_repro
#SBATCH --mem=0
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=/mnt/vast/llmb_/workloads/pretrain_qwen3/experiments/mryab_repro/logs/sbatch_%j.out
#SBATCH --partition=batch
#SBATCH --time=01:00:00

set -evx

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
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=False
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_HIGH_PRIORITY=1
export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/vast/mryab/llmb/.cache/huggingface
export NEMO_HOME=/mnt/vast/mryab/llmb/workloads/pretrain_qwen3
export HOME=/tmp/johnson
export NCCL_SOCKET_IFNAME=bond0
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export PYTHONPATH=/mnt/vast/mryab/llmb/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance:$PYTHONPATH


# Command 1
# Uses Max's Megatron-Bridge, container, HF cache, and experiment dir (for inner script + configs)
# Only srun --output is redirected to our logs dir

srun --output /mnt/vast/llmb_/workloads/pretrain_qwen3/experiments/mryab_repro/logs/log_%j_${SLURM_RESTART_COUNT:-0}.out --container-image /mnt/vast/mryab/llmb/images/nvidia+nemo+26.02.00.sqsh --container-mounts /mnt/vast/mryab/llmb/workloads/pretrain_qwen3:/mnt/vast/mryab/llmb/workloads/pretrain_qwen3,/mnt/vast/mryab/llmb/.cache/huggingface,/mnt/vast/mryab/llmb/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance/run_script.py:/mnt/vast/mryab/llmb/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance/run_script.py,/mnt/vast/mryab/llmb/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance:/mnt/vast/mryab/llmb/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance,/mnt/vast/llmb_/workloads/pretrain_qwen3/experiments/mryab_repro/nemo_run:/nemo_run --container-workdir /nemo_run/code --wait=60 --kill-on-bad-exit=1 --mpi=pmix --no-container-mount-home --container-writable bash /nemo_run/scripts/pretrain_qwen3_235b_a22b_bf16_gpus256_tp1_pp8_cp1_vp4_ep8_etp1_mbs1_gbs8192.sh

exitcode=$?

set -e

echo "job exited with code $exitcode"
if [ $exitcode -ne 0 ]; then
    if [ "$TORCHX_MAX_RETRIES" -gt "${SLURM_RESTART_COUNT:-0}" ]; then
        scontrol requeue "$SLURM_JOB_ID"
    fi
    exit $exitcode
fi
