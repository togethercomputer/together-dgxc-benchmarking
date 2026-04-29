#!/usr/bin/bash
set -euo pipefail

# NOTE: DO NOT change the single quotes to double quotes.
# Inner launch script — runs inside the NeMo container on every rank.
# Mirrors the args 84549 used, then appends hydra overrides to enable torch.profiler.
bash -c 'numactl --cpunodebind=$((SLURM_LOCALID/4)) --membind=$((SLURM_LOCALID/4)) python /mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3/Megatron-Bridge/scripts/performance/run_script.py --container_image /mnt/vast/johnson/llmb/images/nvidia+nemo+26.02.00.sqsh --compute_dtype fp8_mx --gpu b200 --num_gpus 512 --gpus_per_node 8 --offline --model_family_name deepseek --model_recipe_name deepseek_v3 --custom_mounts /mnt/vast/johnson/llmb/.cache/huggingface --account root --partition batch --log_dir /mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3 --time_limit 00:45:00 --max_steps 8 train.manual_gc=true train.manual_gc_interval=100 profiling.use_pytorch_profiler=true profiling.profile_step_start=6 profiling.profile_step_end=7 profiling.profile_ranks=[0] profiling.record_shapes=true logger.tensorboard_dir=/mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3/profiler_traces/dsv3_fp8mx_baseline'
