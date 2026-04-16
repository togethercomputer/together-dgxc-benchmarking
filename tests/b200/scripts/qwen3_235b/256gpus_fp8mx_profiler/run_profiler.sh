#!/usr/bin/bash

#!/usr/bin/env bash
set -euo pipefail

# NOTE: DO NOT change the single quotes to double quotes.
bash -c 'numactl --cpunodebind=$((SLURM_LOCALID/4)) --membind=$((SLURM_LOCALID/4)) python /mnt/vast/johnson/llmb/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance/run_script.py --container_image /mnt/vast/johnson/llmb/images/nvidia+nemo+26.02.00.sqsh --offline --compute_dtype fp8_mx --model_family_name qwen --model_recipe_name qwen3_235b_a22b --gpu b200 --num_gpus 256 --gpus_per_node 8 --custom_mounts /mnt/vast/johnson/llmb/.cache/huggingface -ce HOME=/tmp/johnson,NCCL_SOCKET_IFNAME=bond0 --account root --partition batch --log_dir /mnt/vast/johnson/llmb/workloads/pretrain_qwen3 --time_limit 01:00:00 --max_steps 5 train.manual_gc=true train.manual_gc_interval=100 profiling.use_pytorch_profiler=true profiling.profile_step_start=3 profiling.profile_step_end=4 profiling.profile_ranks=[0] profiling.record_shapes=true logger.tensorboard_dir=/mnt/vast/johnson/llmb/workloads/pretrain_qwen3/profiler_traces/qwen3_235b_fp8mx_baseline'
