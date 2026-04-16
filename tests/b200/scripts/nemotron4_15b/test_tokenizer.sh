#!/bin/bash
#SBATCH --account=root
#SBATCH --job-name=test_tokenizer
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch
#SBATCH --time=00:05:00
#SBATCH --output=/home/johnson/test_tokenizer_%j.out

set -evx

export HOME=/tmp
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/mnt/vast/llmb_/.cache/huggingface

echo "Before srun: HF_HOME=$HF_HOME HOME=$HOME"

srun \
  --container-image /mnt/vast/johnson/llmb/images/nvidia+nemo+26.02.00.sqsh \
  --container-mounts /mnt/vast/llmb_/.cache/huggingface \
  --no-container-mount-home \
  --container-writable \
  --mpi=pmix \
  --container-env=TRANSFORMERS_OFFLINE,HOME,HF_HOME \
  bash -c 'echo "Inside container: HF_HOME=$HF_HOME HOME=$HOME TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"; python3 -c "
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
t = AutoTokenizer(pretrained_model_name=\"nvidia/Nemotron-4-340B-Base\", use_fast=True)
print(\"NeMo AutoTokenizer SUCCESS\")
"'

echo "Test completed with exit code $?"
