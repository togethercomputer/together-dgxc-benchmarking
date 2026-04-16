#!/bin/bash
#SBATCH --account=root
#SBATCH --exclusive
#SBATCH --job-name=test_pyxis
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --partition=batch
#SBATCH --time=00:05:00

set -evx

srun --container-image /mnt/vast/llmb_/images/nvidia+nemo+26.02.00.sqsh \
  --no-container-mount-home \
  --container-writable \
  bash -c 'echo "rank=$SLURM_PROCID user=$(whoami) hostname=$(hostname)"'

exitcode=$?
echo "exit code: $exitcode"
