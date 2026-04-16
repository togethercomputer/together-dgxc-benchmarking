#!/usr/bin/bash

#!/usr/bin/env bash
set -euo pipefail

# Qwen3 235B A22B BF16 — nsys profiling wrapper
# TP=1, PP=8, CP=1, EP=8, VP=4, MBS=1, GBS=8192, DP=32
#
# Rank layout (TP=1, PP=8): rank = pp_rank + dp_rank * 8
#   Node 0 (ranks 0-7):  PP stages 0-7, DP rank 0  (one full pipeline per node)
#   Node 1 (ranks 8-15): PP stages 0-7, DP rank 1
#   ...
#
# Profile ranks: 0 (PP0, first), 4 (PP4, middle), 7 (PP7, last)
# All on node 0 — covers full pipeline + EP/DP inter-node comms.

PROFILE_RANKS="[0,4,7]"
RANK=${SLURM_PROCID}
NSYS_OUT="/mnt/vast/llmb_/workloads/pretrain_qwen3/profiler_traces/qwen3_235b_bf16_256gpu_nsys/profile_${SLURM_JOB_ID}_node${SLURM_NODEID}_gpu${SLURM_LOCALID}"

NUMA_NODE=$((SLURM_LOCALID / 4))
TRAIN_CMD="python /mnt/vast/llmb_/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance/run_script.py \
  --gpu b200 \
  --container_image /mnt/vast/llmb_/images/nvidia+nemo+26.02.00.sqsh \
  --num_gpus 256 \
  --gpus_per_node 8 \
  --model_name qwen3 \
  --model_size 235b_a22b \
  --max_steps 10 \
  --custom_mounts /mnt/vast/llmb_/.cache/huggingface \
  --compute_dtype bf16 \
  -tp 1 -pp 8 -cp 1 -ep 8 -et 1 -vp 4 -mb 1 -gb 8192 \
  --account root \
  --partition batch \
  --log_dir /mnt/vast/llmb_/workloads/pretrain_qwen3 \
  --time_limit 01:00:00 \
  train.manual_gc=true \
  train.manual_gc_interval=100 \
  profiling.use_nsys_profiler=true \
  profiling.profile_step_start=5 \
  profiling.profile_step_end=8 \
  profiling.profile_ranks=${PROFILE_RANKS}"

if [ "$RANK" = "0" ] || [ "$RANK" = "4" ] || [ "$RANK" = "7" ]; then
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
