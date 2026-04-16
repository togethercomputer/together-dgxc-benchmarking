#!/bin/bash
#
# Grok1 314B BF16 - 256 GPUs (32 nodes)
# Config: TP=4, PP=4, EP=8, VP=8, MBS=1, GBS=512, SeqLen=8192
# Container: nvidia+nemo+25.09.00
# MPI stub + TP_COMM_OVERLAP=False to bypass PMIx v3/v4 mismatch
#

# Parameters
#SBATCH --account=root
#SBATCH --exclusive
#SBATCH --job-name=grok1_314b_bf16_256gpu_no_tp_overlap
#SBATCH --mem=0
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=/mnt/vast/johnson/llmb/workloads/pretrain_grok1/experiments/pretrain_grok1_314b_bf16_gpus256_tp4_pp4_cp1_vp8_ep8_etp1_mbs1_gbs512/pretrain_grok1_314b_bf16_gpus256_tp4_pp4_cp1_vp8_ep8_etp1_mbs1_gbs512_1776219201/pretrain_grok1_314b_bf16_gpus256_tp4_pp4_cp1_vp8_ep8_etp1_mbs1_gbs512/sbatch_grok1_bf16_256gpu_%j.out
#SBATCH --partition=batch
#SBATCH --time=01:15:00
#SBATCH --exclude=use3a-ss-b200-gpu-[150,156,159,164,171,172,181-182,184,190,195,200,202,210-217]

set -evx
ulimit -n 1048576 || true

export PYTHONUNBUFFERED=1
export SLURM_UNBUFFEREDIO=1
export TORCHX_MAX_RETRIES=0

set +e

# Pyxis HOME workaround for this cluster
export HOME=/tmp
export NEMO_NLP_TMP=/tmp

# setup
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Performance env vars
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=False
export NCCL_NVLS_ENABLE=0
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=1
export NEMO_LOG_MEMORY_USAGE=1
export NEMORUN_HOME=/mnt/vast/johnson/llmb/workloads/pretrain_grok1
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export TORCH_NCCL_HIGH_PRIORITY=1
export HF_HOME=/mnt/vast/johnson/llmb/.cache/huggingface

# Cluster-specific NCCL settings
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA="=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_11:1,mlx5_14:1,mlx5_15:1"
export UCX_NET_DEVICES=bond0
export OMPI_MCA_btl_tcp_if_include=bond0

# Disable tp_comm_overlap to bypass PMIx v3/v4 MPI_Init_thread failure
# The 25.09 container has OpenMPI built with PMIx v3, host runs PMIx v4
export TP_COMM_OVERLAP=False

# Prevent mpi4py from auto-calling MPI_Init_thread
export MPI4PY_RC_INITIALIZE=false
export MPI4PY_RC_THREADS=false
export MPI4PY_RC_FINALIZE=false
export PMIX_MCA_gds=hash
export OMPI_MCA_plm=isolated

# MPI stub: intercepts C-level MPI_Init_thread to avoid fatal PMIx abort
# Combined with TP_COMM_OVERLAP=False, userbuffer code won't run (no hang)
export LD_PRELOAD=/mpi_stub/libmpi_stub.so

# Experiment directory (reuse existing experiment with code/configs)
EXP_DIR=/mnt/vast/johnson/llmb/workloads/pretrain_grok1/experiments/pretrain_grok1_314b_bf16_gpus256_tp4_pp4_cp1_vp8_ep8_etp1_mbs1_gbs512/pretrain_grok1_314b_bf16_gpus256_tp4_pp4_cp1_vp8_ep8_etp1_mbs1_gbs512_1776219201/pretrain_grok1_314b_bf16_gpus256_tp4_pp4_cp1_vp8_ep8_etp1_mbs1_gbs512

# Command 1
srun --output ${EXP_DIR}/log-grok1_bf16_256gpu_%j_${SLURM_RESTART_COUNT:-0}.out \
  --container-image /mnt/vast/johnson/llmb/images/nvidia+nemo+25.09.00.sqsh \
  --container-mounts /mnt/vast/johnson/llmb/.cache/huggingface,${EXP_DIR}:/nemo_run,/mnt/vast/johnson/llmb/mpi_stub:/mpi_stub \
  --container-workdir /nemo_run/code \
  --wait=60 \
  --kill-on-bad-exit=1 \
  --mpi=pmix \
  --no-container-mount-home \
  --container-writable \
  --time=01:10:00 \
  --container-env=TORCH_NCCL_AVOID_RECORD_STREAMS,TRANSFORMERS_OFFLINE,TOKENIZERS_PARALLELISM,NCCL_NVLS_ENABLE,NVTE_FLASH_ATTN,NVTE_FUSED_ATTN,NEMO_LOG_MEMORY_USAGE,NEMORUN_HOME,CUDA_DEVICE_MAX_CONNECTIONS,NVTE_FWD_LAYERNORM_SM_MARGIN,NVTE_BWD_LAYERNORM_SM_MARGIN,NCCL_P2P_NET_CHUNKSIZE,TORCH_NCCL_HIGH_PRIORITY,NCCL_SOCKET_IFNAME,NCCL_IB_HCA,UCX_NET_DEVICES,OMPI_MCA_btl_tcp_if_include,HOME,HF_HOME,NEMO_NLP_TMP,TP_COMM_OVERLAP,MPI4PY_RC_INITIALIZE,MPI4PY_RC_THREADS,MPI4PY_RC_FINALIZE,PMIX_MCA_gds,OMPI_MCA_plm,LD_PRELOAD \
  bash /nemo_run/scripts/pretrain_grok1_314b_bf16_gpus256_tp4_pp4_cp1_vp8_ep8_etp1_mbs1_gbs512.sh

exitcode=$?

set -e

echo "job exited with code $exitcode"

# Log-based success validation
if [ $exitcode -ne 0 ]; then
    echo "[LOG_CHECK] Job failed with exit code $exitcode - validating logs..."
    LOG_PATTERN="${EXP_DIR}/log-grok1_bf16_256gpu_${SLURM_JOB_ID}_*.out"
    LATEST_LOG=""
    MAX_RESTART_COUNT=-1
    for log_file in $LOG_PATTERN; do
        if [ -f "$log_file" ]; then
            RESTART_COUNT=$(basename "$log_file" | sed 's/.*_\([0-9]*\)\.out$/\1/')
            if [ "$RESTART_COUNT" -gt "$MAX_RESTART_COUNT" ]; then
                MAX_RESTART_COUNT=$RESTART_COUNT
                LATEST_LOG="$log_file"
            fi
        fi
    done
    if [ -n "$LATEST_LOG" ] && [ -r "$LATEST_LOG" ]; then
        if grep -E -q 'Trainer\.fit.*stopped:.*max_steps=.*reached\.' "$LATEST_LOG"; then
            echo "[LOG_CHECK] Training completed successfully (max_steps reached)"
            echo "[LOG_CHECK] Overriding exit code: $exitcode -> 0"
            exitcode=0
        else
            echo "[LOG_CHECK] Success pattern not found - genuine failure"
        fi
    else
        echo "[LOG_CHECK] Unable to validate logs"
    fi
fi

if [ $exitcode -ne 0 ]; then
    if [ "$TORCHX_MAX_RETRIES" -gt "${SLURM_RESTART_COUNT:-0}" ]; then
        scontrol requeue "$SLURM_JOB_ID"
    fi
    exit $exitcode
fi
