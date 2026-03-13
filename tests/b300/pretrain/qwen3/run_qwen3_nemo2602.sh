#!/bin/bash
# Direct docker run for pretrain_qwen3 30b bf16 scale=8
# Uses torchrun to launch 8 processes (one per GPU) inside the container
# NeMo version: 26.02.00

LLMB=/scratch/user/johnson/llmb

NEMO_RUN="$LLMB/workloads/pretrain_qwen3/experiments/pretrain_qwen3_30b_a3b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep8_mbs8_gbs512/pretrain_qwen3_30b_a3b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep8_mbs8_gbs512_1773170647/pretrain_qwen3_30b_a3b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep8_mbs8_gbs512"

RUN_SCRIPT="$LLMB/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance/run_script.py"

mkdir -p "$NEMO_RUN"

echo "Starting pretrain_qwen3 30b bf16 scale=8 via docker + torchrun..."
echo "NeMo version: 26.02.00"
echo "Logs: $NEMO_RUN/job.log"

: "${HF_TOKEN:?HF_TOKEN environment variable is required}"

sudo --preserve-env=HF_TOKEN nerdctl run --rm \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --security-opt seccomp=unconfined \
  --cap-add SYS_NICE \
  -w /nemo_run/code \
  -v "$LLMB/workloads/pretrain_qwen3:$LLMB/workloads/pretrain_qwen3" \
  -v "$LLMB/.cache/huggingface:$LLMB/.cache/huggingface" \
  -v "$LLMB/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance:$LLMB/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance" \
  -v "$NEMO_RUN":/nemo_run \
  -e MASTER_ADDR=localhost \
  -e MASTER_PORT=29500 \
  -e TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
  -e TRANSFORMERS_OFFLINE=0 \
  -e TOKENIZERS_PARALLELISM=False \
  -e NCCL_NVLS_ENABLE=0 \
  -e TORCH_NCCL_HIGH_PRIORITY=1 \
  -e HF_HUB_OFFLINE=0 \
  -e NEMORUN_HOME="$LLMB/workloads/pretrain_qwen3" \
  -e NEMO_HOME="$LLMB/workloads/pretrain_qwen3" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e NVTE_FWD_LAYERNORM_SM_MARGIN=20 \
  -e NVTE_BWD_LAYERNORM_SM_MARGIN=20 \
  -e NCCL_GRAPH_REGISTER=0 \
  -e PYTHONPATH="$LLMB/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance" \
  nvcr.io/nvidia/nemo:26.02.00 \
  torchrun \
    --nproc_per_node=8 \
    --master_addr=localhost \
    --master_port=29500 \
    "$RUN_SCRIPT" \
    --compute_dtype bf16 \
    --gpu gb300 \
    --num_gpus 8 \
    --gpus_per_node 8 \
    -m qwen \
    -mr qwen3_30b_a3b \
    --custom_mounts "$LLMB/.cache/huggingface" \
    -tp 1 -pp 1 -cp 1 -ep 8 -et 1 -gb 64 -mb 1 \
    --account default \
    --partition gpu \
    --log_dir "$LLMB/workloads/pretrain_qwen3" \
    --time_limit 01:00:00 \
    --max_steps 50 \
    train.manual_gc=true \
    train.manual_gc_interval=100 \
  2>&1 | tee "$NEMO_RUN/job.log"
