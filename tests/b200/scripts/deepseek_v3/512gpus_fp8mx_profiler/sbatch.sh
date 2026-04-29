#!/bin/bash
#
# DeepSeek V3 671B FP8 MX - 512 GPUs (64 nodes)
# PROFILER: TP=1, PP=16, CP=1, EP=8, VP=None, MBS=1, GBS=8192
# Goal: PyTorch profiler trace on step 6-7 (rank 0) to identify DSV3 bottlenecks
# Based on qwen3_235b/256gpus_fp8mx_profiler + 84549 validated env vars/exclude list
#

# Parameters
#SBATCH --account=root
#SBATCH --exclusive
#SBATCH --job-name=dsv3_671b_fp8mx_512gpu_profiler
#SBATCH --mem=0
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=/home/johnson/together-dgxc-benchmarking/tests/b200/scripts/deepseek_v3/512gpus_fp8mx_profiler/sbatch_%j.out
#SBATCH --partition=batch
#SBATCH --time=00:45:00
#SBATCH --exclude=use3a-ss-b200-gpu-[190,199,201,202,204,211,228,230,233,239]

set -evx
ulimit -n 1048576 || true

export PYTHONUNBUFFERED=1
export SLURM_UNBUFFEREDIO=1
export TORCHX_MAX_RETRIES=0

set +e

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# NCCL / network (from 84549 validated run)
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_TIMEOUT=7200
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=False
export NCCL_NVLS_ENABLE=0
export NCCL_COLLNET_ENABLE=0
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_IB_HCA="=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_11:1,mlx5_14:1,mlx5_15:1"
export UCX_NET_DEVICES=bond0
export OMPI_MCA_btl_tcp_if_include=bond0
export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/vast/johnson/llmb/.cache/huggingface
export NEMORUN_HOME=/mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3
export NEMO_HOME=/mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3
export HOME=/tmp/johnson

# DSV3-specific (from 84549)
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NVTE_FWD_LAYERNORM_SM_MARGIN=20
export NVTE_BWD_LAYERNORM_SM_MARGIN=20
export NVLINK_DOMAIN_SIZE=8
export USE_MNNVL=0
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8

export PYTHONPATH=/mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3/Megatron-Bridge/scripts/performance:$PYTHONPATH

# srun: bind-mount patched profiling.py (adds CUDA activities to torch.profiler) over
# container's stock copy; bind-mount profiler_traces output dir; mount the patched
# run_script.py (has LD_LIBRARY_PATH ptxas fix) and the performance/ scripts dir.
srun \
  --output /home/johnson/together-dgxc-benchmarking/tests/b200/scripts/deepseek_v3/512gpus_fp8mx_profiler/log_%j_${SLURM_RESTART_COUNT:-0}.out \
  --container-image /mnt/vast/johnson/llmb/images/nvidia+nemo+26.02.00.sqsh \
  --container-mounts /mnt/vast/johnson/llmb/.cache/huggingface,/mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3/profiler_traces:/mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3/profiler_traces,/mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3/Megatron-Bridge/scripts/performance/run_script.py:/mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3/Megatron-Bridge/scripts/performance/run_script.py,/mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3/Megatron-Bridge/scripts/performance:/mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3/Megatron-Bridge/scripts/performance,/mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3/profiler_traces/profiling.py:/opt/Megatron-Bridge/src/megatron/bridge/training/profiling.py \
  --container-workdir /mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3/Megatron-Bridge/scripts/performance \
  --wait=60 \
  --kill-on-bad-exit=1 \
  --mpi=pmix \
  --no-container-mount-home \
  --container-writable \
  bash /home/johnson/together-dgxc-benchmarking/tests/b200/scripts/deepseek_v3/512gpus_fp8mx_profiler/run_profiler.sh

exitcode=$?

set -e

echo "job exited with code $exitcode"
if [ $exitcode -ne 0 ]; then
    if [ "$TORCHX_MAX_RETRIES" -gt "${SLURM_RESTART_COUNT:-0}" ]; then
        scontrol requeue "$SLURM_JOB_ID"
    fi
    exit $exitcode
fi
