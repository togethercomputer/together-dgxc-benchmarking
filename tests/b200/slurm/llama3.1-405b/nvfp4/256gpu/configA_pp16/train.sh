#!/usr/bin/bash

#!/usr/bin/env bash
set -euo pipefail

# NOTE: DO NOT change the single quotes to double quotes.
bash -c 'numactl --cpunodebind=$((SLURM_LOCALID/4)) --membind=$((SLURM_LOCALID/4)) python /mnt/vast/llmb_/workloads/pretrain_llama3.1/Megatron-Bridge/scripts/performance/run_script.py --gpu b200 --container_image /mnt/vast/llmb_/images/nvidia+nemo+26.02.00.sqsh --num_gpus 256 --gpus_per_node 8 --model_name llama31 --model_size 405b --max_steps 10 --custom_mounts /mnt/vast/llmb_/.cache/huggingface --compute_dtype nvfp4 -tp 4 -pp 16 -cp 1 -vp 8 -mb 1 -gb 1536 --recompute_num_layers 8 --account root --partition batch --log_dir /mnt/vast/llmb_/workloads/pretrain_llama3.1 --time_limit 01:00:00 train.manual_gc=true train.manual_gc_interval=100 profiling.use_pytorch_profiler=true profiling.profile_step_start=3 profiling.profile_step_end=4 profiling.profile_ranks=[0] profiling.record_shapes=true logger.tensorboard_dir=/mnt/vast/llmb_/workloads/pretrain_llama3.1/profiler_traces/405b_nvfp4_256gpu'
