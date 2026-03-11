#!/bin/bash
# Direct docker run for pretrain_llama3 8b bf16 scale=8
# Uses torchrun to launch 8 processes (one per GPU) inside the container

WORKLOAD_DIR="/home/ubuntu/llmb/workloads/pretrain_llama3.1"

RUN_SCRIPT="$WORKLOAD_DIR/Megatron-Bridge/scripts/performance/run_script.py"

LOG_DIR="/home/ubuntu/llmb"

echo "Starting pretrain_llama3 8b bf16 scale=8 via docker + torchrun..."
echo "Logs: $LOG_DIR/llama3_job.log"

sudo docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --security-opt seccomp=unconfined \
  --cap-add SYS_NICE \
  -w /workspace \
  -v "$WORKLOAD_DIR":"$WORKLOAD_DIR" \
  -v /home/ubuntu/llmb/.cache/huggingface:/home/ubuntu/llmb/.cache/huggingface \
  -v "$WORKLOAD_DIR/Megatron-Bridge/scripts/performance":"$WORKLOAD_DIR/Megatron-Bridge/scripts/performance" \
  -e MASTER_ADDR=localhost \
  -e MASTER_PORT=29500 \
  -e TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
  -e TRANSFORMERS_OFFLINE=0 \
  -e TOKENIZERS_PARALLELISM=False \
  -e NCCL_NVLS_ENABLE=0 \
  -e TORCH_NCCL_HIGH_PRIORITY=1 \
  -e HF_HUB_OFFLINE=0 \
  -e NEMORUN_HOME="$WORKLOAD_DIR" \
  -e NEMO_HOME="$WORKLOAD_DIR" \
  -e HF_TOKEN=$HF_TOKEN \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -e NCCL_GRAPH_REGISTER=0 \
  -e PYTHONPATH="$WORKLOAD_DIR/Megatron-Bridge/scripts/performance" \
  nvcr.io/nvidia/nemo:25.11.01 \
  torchrun \
    --nproc_per_node=8 \
    --master_addr=localhost \
    --master_port=29500 \
    "$RUN_SCRIPT" \
    --compute_dtype bf16 \
    --gpu gb300 \
    --num_gpus 8 \
    --gpus_per_node 8 \
    --model_name llama3 \
    --model_size 8b \
    --custom_mounts /home/ubuntu/llmb/.cache/huggingface \
    -tp 1 -pp 1 -cp 1 -gb 128 -mb 4 \
    --account default \
    --partition gpu \
    --log_dir "$WORKLOAD_DIR" \
    --time_limit 00:40:00 \
    --max_steps 50 \
    --hf_token $HF_TOKEN \
    train.manual_gc=true \
    train.manual_gc_interval=100 \
  2>&1 | tee "$LOG_DIR/llama3_job.log"

