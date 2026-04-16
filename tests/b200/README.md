# Llama 70B FP8 Scaling Benchmark — DGXC B200 Cluster

**Date:** 2026-04-08
**Cluster:** us-east-3a-forge-exemplar-testing (Together AI)
**Model:** Llama 3 70B, FP8 Current Scaling
**Framework:** Megatron-Bridge + NeMo Run (nvidia/nemo:25.11.01)

## Directory Structure

```
├── reports/
│   ├── Llama_70B_FP8_Benchmark_Report_64GPU.md
│   ├── Llama_70B_FP8_Benchmark_Report_512GPU.md
│   └── Llama_70B_FP8_Scaling_Benchmark_Report.md
├── 64gpus_mbs1/          # 8 nodes,  GBS=128,  Jobs 79118-79120
│   ├── train.sh          # Per-rank training script (called by srun)
│   ├── sbatch.sh         # Sbatch wrapper script
│   └── logs/             # Job logs (train + sbatch output)
├── 128gpus_mbs1/         # 16 nodes, GBS=256,  Job 79138
│   ├── train.sh
│   ├── sbatch.sh
│   └── logs/
├── 256gpus_mbs1/         # 32 nodes, GBS=512,  Job 79139
│   ├── train.sh
│   ├── sbatch.sh
│   └── logs/
├── 512gpus_mbs1/         # 64 nodes, GBS=1024, Jobs 79133-79135
│   ├── train.sh
│   ├── sbatch.sh
│   └── logs/
├── 512gpus_mbs2/         # 64 nodes, GBS=1024, Job 79141 (OOM)
│   ├── train.sh
│   ├── sbatch.sh
│   └── logs/
├── 592gpus_mbs1/         # 74 nodes, GBS=1184, Job 79140
│   ├── train.sh
│   ├── sbatch.sh
│   └── logs/
└── README.md
```

## Scaling Results Summary

| GPUs | Nodes | Avg Step Time | TFLOP/s/GPU | Scaling Eff. | Jobs |
|------|-------|---------------|-------------|--------------|------|
| 64   | 8     | 4.63s         | 1,589.7     | baseline     | 79118-79120 |
| 128  | 16    | 4.63s         | 1,591.5     | 100.1%       | 79138 |
| 256  | 32    | 4.92s         | 1,498.8     | 94.3%        | 79139 |
| 512  | 64    | 5.39s         | 1,366.6     | 86.0%        | 79133-79135 |
| 592  | 74    | 5.59s         | 1,318.5     | 82.9%        | 79140 |
| 512 (MBS=2) | 64 | N/A      | N/A         | OOM          | 79141 |

## How to Submit

```bash
sudo sbatch --requeue --parsable \
  --exclude=use3a-ss-b200-gpu-148 \
  <config>/sbatch.sh
```

## Log File Naming

Each `logs/` folder contains:
- `job_<JOBID>_train.log` — srun training output (step times, TFLOP/s, loss)
- `job_<JOBID>_sbatch.log` — sbatch wrapper output

## Experiment Data (on /mnt/vast)

Full experiment directories with checkpoints, configs, and all rank logs:

| Config | Path |
|--------|------|
| 64 GPUs  | `/mnt/vast/exemplar/llmb/workloads/pretrain_llama3.1/experiments/pretrain_llama3_70b_fp8_cs_gpus64_.../` |
| 128 GPUs | `/mnt/vast/exemplar/llmb/workloads/pretrain_llama3.1/experiments/pretrain_llama3_70b_fp8_cs_gpus128_.../` |
| 256 GPUs | `/mnt/vast/exemplar/llmb/workloads/pretrain_llama3.1/experiments/pretrain_llama3_70b_fp8_cs_gpus256_.../` |
| 512 GPUs | `/mnt/vast/exemplar/llmb/workloads/pretrain_llama3.1/experiments/pretrain_llama3_70b_fp8_cs_gpus512_.../` |
| 592 GPUs | `/mnt/vast/exemplar/llmb/workloads/pretrain_llama3.1/experiments/pretrain_llama3_70b_fp8_cs_gpus592_.../` |
