#!/usr/bin/bash

#!/usr/bin/env bash
set -euo pipefail

# NOTE: DO NOT change the single quotes to double quotes.
bash -c 'numactl --cpunodebind=$((SLURM_LOCALID/4)) --membind=$((SLURM_LOCALID/4)) python /mnt/vast/exemplar/llmb/workloads/pretrain_llama3.1/Megatron-Bridge/scripts/performance/run_script.py --gpu b200 --container_image /mnt/vast/exemplar/llmb/images/nvidia+nemo+25.11.01.sqsh --num_gpus 512 --gpus_per_node 8 --model_name llama3 --model_size 70b --max_steps 50 --custom_mounts /mnt/vast/exemplar/llmb/.cache/huggingface --compute_dtype fp8_cs --account root --partition batch --log_dir /mnt/vast/exemplar/llmb/workloads/pretrain_llama3.1 --time_limit 00:30:00 train.manual_gc=true train.manual_gc_interval=100'