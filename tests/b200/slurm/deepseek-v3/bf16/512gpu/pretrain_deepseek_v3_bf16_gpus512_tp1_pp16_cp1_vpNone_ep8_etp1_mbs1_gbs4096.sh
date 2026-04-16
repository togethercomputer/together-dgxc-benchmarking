#!/usr/bin/bash

#!/usr/bin/env bash
set -euo pipefail

# NOTE: DO NOT change the single quotes to double quotes.
bash -c 'numactl --cpunodebind=$((SLURM_LOCALID/4)) --membind=$((SLURM_LOCALID/4)) python /mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3/Megatron-Bridge/scripts/performance/run_script.py --container_image /mnt/vast/johnson/llmb/images/nvidia+nemo+26.02.00.sqsh --compute_dtype bf16 --gpu b200 --num_gpus 512 --gpus_per_node 8 --offline --model_family_name deepseek --model_recipe_name deepseek_v3 --custom_mounts /mnt/vast/johnson/llmb/.cache/huggingface --account root --partition batch --log_dir /mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3 --time_limit 01:30:00 --max_steps 50 train.manual_gc=true train.manual_gc_interval=100'