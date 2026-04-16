#!/bin/bash
#SBATCH --account=root
#SBATCH --exclusive
#SBATCH --job-name=test_pyxis_32n
#SBATCH --mem=0
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=8
#SBATCH --partition=batch
#SBATCH --output=/mnt/vast/llmb_/workloads/pretrain_qwen3/test_pyxis_32n_%j.out
#SBATCH --time=00:05:00

set -evx

srun --container-image /mnt/vast/llmb_/images/nvidia+nemo+26.02.00.sqsh \
  --container-mounts /mnt/vast/llmb_/.cache/huggingface,/mnt/vast/llmb_/workloads/pretrain_qwen3/experiments/pretrain_qwen3_235b_a22b_bf16_gpus256_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192/pretrain_qwen3_235b_a22b_bf16_gpus256_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192_1775789255/pretrain_qwen3_235b_a22b_bf16_gpus256_tp1_pp8_cp1_vp4_ep8_mbs1_gbs8192:/nemo_run \
  --container-workdir /nemo_run \
  --no-container-mount-home \
  --container-writable \
  --wait=60 \
  --kill-on-bad-exit=1 \
  --mpi=pmi2 \
  bash -c 'echo "rank=$SLURM_PROCID user=$(whoami) hostname=$(hostname) pwd=$(pwd)"'

exitcode=$?
echo "exit code: $exitcode"
