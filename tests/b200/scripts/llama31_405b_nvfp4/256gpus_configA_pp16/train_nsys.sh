#!/usr/bin/bash

#!/usr/bin/env bash
set -euo pipefail

# Profile only selected ranks with nsys; all other ranks run plain python.
# SLURM_PROCID is the global rank (0..255).
# Ranks: 0 = PP stage 0, 32 = PP stage 8 (middle), 60 = PP stage 15 (last).
# (TP=4, PP=16 → rank = pp_stage * tp_size + tp_rank)

PROFILE_RANKS="[0,32,60]"
RANK=${SLURM_PROCID}
NSYS_OUT="/mnt/vast/llmb_/workloads/pretrain_llama3.1/profiler_traces/405b_nvfp4_256gpu_nsys/profile_${SLURM_JOB_ID}_node${SLURM_NODEID}_gpu${SLURM_LOCALID}"

NUMA_NODE=$((SLURM_LOCALID / 4))
TRAIN_CMD="python /mnt/vast/llmb_/workloads/pretrain_llama3.1/Megatron-Bridge/scripts/performance/run_script.py \
  --gpu b200 \
  --container_image /mnt/vast/llmb_/images/nvidia+nemo+26.02.00.sqsh \
  --num_gpus 256 \
  --gpus_per_node 8 \
  --model_name llama31 \
  --model_size 405b \
  --max_steps 10 \
  --custom_mounts /mnt/vast/llmb_/.cache/huggingface \
  --compute_dtype nvfp4 \
  -tp 4 -pp 16 -cp 1 -vp 8 -mb 1 -gb 1536 \
  --recompute_num_layers 8 \
  --account root \
  --partition batch \
  --log_dir /mnt/vast/llmb_/workloads/pretrain_llama3.1 \
  --time_limit 01:00:00 \
  train.manual_gc=true \
  train.manual_gc_interval=100 \
  profiling.use_nsys_profiler=true \
  profiling.profile_step_start=3 \
  profiling.profile_step_end=6 \
  profiling.profile_ranks=${PROFILE_RANKS}"

if [ "$RANK" = "0" ] || [ "$RANK" = "32" ] || [ "$RANK" = "60" ]; then
  echo "Rank ${RANK}: launching under nsys profiler"
  exec numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE} \
    nsys profile -s none -t nvtx,cuda \
    -o "${NSYS_OUT}" \
    --force-overwrite true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    ${TRAIN_CMD}
else
  exec numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE} \
    ${TRAIN_CMD}
fi
