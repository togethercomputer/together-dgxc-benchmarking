#!/bin/bash
#SBATCH --account=root
#SBATCH --exclusive
#SBATCH --job-name=test_32n_simple
#SBATCH --mem=0
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=/mnt/vast/llmb_/workloads/pretrain_qwen3/test_32n_simple_%j.out
#SBATCH --partition=batch
#SBATCH --time=00:05:00

set -evx
echo "batch script started on $(hostname)"
srun hostname
exitcode=$?
echo "exit code: $exitcode"
