#!/usr/bin/bash

#!/usr/bin/env bash
set -euo pipefail

# NOTE: DO NOT change the single quotes to double quotes.
bash -c 'CPUSET=
    MEMNODE=
    case ${SLURM_LOCALID} in
      0) CPUSET=0-7;   MEMNODE=0 ;;
      1) CPUSET=24-31; MEMNODE=0 ;;
      2) CPUSET=96-103; MEMNODE=0 ;;
      3) CPUSET=112-119; MEMNODE=0 ;;
      4) CPUSET=128-135; MEMNODE=1 ;;
      5) CPUSET=144-151; MEMNODE=1 ;;
      6) CPUSET=224-231; MEMNODE=1 ;;
      7) CPUSET=248-255; MEMNODE=1 ;;
      *) echo Unexpected SLURM_LOCALID=${SLURM_LOCALID}; exit 1 ;;
    esac
    numactl --physcpubind=${CPUSET} --membind=${MEMNODE} python /mnt/vast/exemplar/llmb/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance/run_script.py --container_image /mnt/vast/exemplar/llmb/images/nvidia+nemo+25.11.01.sqsh --compute_dtype fp8_mx --gpu b200 --num_gpus 512 --gpus_per_node 8 --model_name qwen3 --model_size 235b_a22b --custom_mounts /mnt/vast/exemplar/llmb/.cache/huggingface -tp 1 -pp 8 -cp 1 -ep 8 -et 1 -gb 8192 -mb 1 -vp 4 --cuda_graph_impl=transformer_engine --cuda_graph_scope=moe_router,moe_preprocess --account root --partition batch --log_dir /mnt/vast/exemplar/llmb/workloads/pretrain_qwen3 --time_limit 00:40:00 --max_steps 50 --hf_token HF_TOKEN_REDACTED train.manual_gc=true train.manual_gc_interval=100'
