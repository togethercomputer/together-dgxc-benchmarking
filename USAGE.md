# Usage Guide — LLM Benchmarking CLI

This guide covers day-to-day usage of the three installed workloads and the
results parser. It assumes `llmb-run` is installed and the environment is
configured (see [README.md](README.md) for installation).

## Environment Setup

```bash
# These should already be in your ~/.bashrc after installation
export LLMB_INSTALL=/path/to/llmb           # set by installer
export SBATCH_ACCOUNT=<your_slurm_account>
export SBATCH_PARTITION=<your_partition>
export GPU_TYPE=h100                         # h100 | b200 | gb200 | gb300

# Convenience alias for the results parser
PARSER=$LLMB_INSTALL/llmb_repo/common/parse_train_timing.sh
```

---

## 1. Llama 3.1 Pretraining (`pretrain_llama3.1`)

### Supported options (H100)

| Model size | Dtypes        | Scales (GPUs)          |
|------------|---------------|------------------------|
| `8b`       | `fp8`, `bf16` | 8, 16, 32, 64, 128     |
| `70b`      | `fp8`, `bf16` | 64, 128, 256, 512, 1024|
| `405b`     | `fp8`, `bf16` | 1024                   |

### Commands

```bash
# 8B FP8, single node (8 GPUs)
llmb-run submit -w pretrain_llama3.1 -s 8b --dtype fp8 --scale 8

# 8B FP8, two nodes (16 GPUs)
llmb-run submit -w pretrain_llama3.1 -s 8b --dtype fp8 --scale 16

# 8B BF16
llmb-run submit -w pretrain_llama3.1 -s 8b --dtype bf16 --scale 8

# Sweep multiple scales in one submission
llmb-run submit -w pretrain_llama3.1 -s 8b --dtype fp8 --scale 8,16,32

# Run 3 repeats per configuration (for statistical confidence)
llmb-run submit -w pretrain_llama3.1 -s 8b --dtype fp8 --scale 8 --repeats 3

# Preview jobs without submitting
llmb-run submit -w pretrain_llama3.1 -s 8b --dtype fp8 --scale 8 --dry-run
```

### Monitor and view results

```bash
squeue -u $USER                             # check running jobs

# Parse results (run from workload directory)
cd $LLMB_INSTALL/workloads/pretrain_llama3.1
bash $PARSER                                # default: iterations 35-44, table
bash $PARSER --max-steps=50                 # explicit 50-step run
bash $PARSER --format=csv > results.csv     # export to CSV
```

### Output location

```
$LLMB_INSTALL/workloads/pretrain_llama3.1/experiments/
  pretrain_llama3_8b_fp8_cs_gpus8_.../
    <name>_<timestamp>/<name>/
      log-*.out        ← training log (step timing, TFLOP/s)
      sbatch_*.out     ← Slurm batch output
```

### Performance reference (H100, FP8)

| Scale | Step time | TFLOP/s/GPU |
|-------|-----------|-------------|
| 8 GPUs (1 node)  | ~9.3 s | ~720 |
| 16 GPUs (2 nodes) | ~9.5 s | ~707 |

---

## 2. Nemotron4-15B Pretraining (`pretrain_nemotron4-15b`)

Minimum scale is **16 GPUs (2 nodes)**.

### Supported options (H100)

| Model size | Dtypes        | Scales (GPUs)                          |
|------------|---------------|----------------------------------------|
| `15b`      | `fp8`, `bf16` | 16, 32, 64, 128, 256, 512, 1024, 2048 |

### Commands

```bash
# 15B FP8, two nodes (16 GPUs)
llmb-run submit -w pretrain_nemotron4-15b --dtype fp8 --scale 16

# 15B BF16, two nodes
llmb-run submit -w pretrain_nemotron4-15b --dtype bf16 --scale 16

# Sweep multiple scales
llmb-run submit -w pretrain_nemotron4-15b --dtype fp8 --scale 16,32

# With SHARP (requires cluster SHARP/AM support — see note below)
llmb-run submit -w pretrain_nemotron4-15b --dtype fp8 --scale 16 --use-sharp

# Preview jobs without submitting
llmb-run submit -w pretrain_nemotron4-15b --dtype fp8 --scale 16 --dry-run
```

> **Note on `--use-sharp`:** Requires InfiniBand SHARP Aggregation Manager (`sharp_am`)
> running on the IB switch fabric, and/or NVLink SHARP (NVLS) enabled via
> `nvidia-fabricmanager` on host nodes. NCCL falls back gracefully if SHARP is
> unavailable (error `-52` in logs is non-fatal).

### Monitor and view results

```bash
squeue -u $USER

cd $LLMB_INSTALL/workloads/pretrain_nemotron4-15b
bash $PARSER                                # default: iterations 35-44
bash $PARSER --max-steps=50
bash $PARSER --format=json
bash $PARSER --format=csv > results.csv
```

### Output location

```
$LLMB_INSTALL/workloads/pretrain_nemotron4-15b/experiments/
  pretrain_nemotron4_15b_fp8_gpus16_.../
    <name>_<timestamp>/<name>/
      log-*.out        ← training log
      sbatch_*.out     ← Slurm batch output
```

### Performance reference (H100, FP8, 16 GPUs)

| Scale | Step time | TFLOP/s/GPU |
|-------|-----------|-------------|
| 16 GPUs (2 nodes) | ~1.79 s | ~817 |

---

## 3. NCCL Microbenchmark (`microbenchmark_nccl`)

Runs 7 collective tests: `all_reduce`, `all_gather`, `reduce_scatter`,
`alltoall` (×2), `sendrecv` (×2). Sweeps message sizes from 8 B to 16 GB.

### Supported options (H100)

| Dtype  | Scales (GPUs)                               |
|--------|---------------------------------------------|
| `fp8`  | 2, 4, 8, 16, 32, 64, 128, 256, 512, ...    |

*(dtype is a label only; tests run `float` for reduce ops and `uint8` for point-to-point)*

### Commands

```bash
# 2 nodes (16 GPUs)
llmb-run submit -w microbenchmark_nccl --dtype fp8 --scale 16

# 1 node (8 GPUs)
llmb-run submit -w microbenchmark_nccl --dtype fp8 --scale 8

# Preview without submitting
llmb-run submit -w microbenchmark_nccl --dtype fp8 --scale 16 --dry-run
```

### View results

```bash
squeue -u $USER                             # monitor running job

EXP=$LLMB_INSTALL/workloads/microbenchmark_nccl/experiments

# Find the latest run directory
LATEST=$(ls -td $EXP/microbenchmark_nccl_*/LOG_*/ | head -1)

# Show all_reduce bandwidth table (from rank 0)
cat "${LATEST}"/1_1_LOG_all_reduce*iter1/rank0*/stdout.txt

# Show all 7 tests
for f in "${LATEST}"/1_*_LOG_*iter1/rank0*/stdout.txt; do
    echo "=== $(basename $(dirname $(dirname $f))) ==="
    tail -5 "$f"
done
```

### Output location

```
$LLMB_INSTALL/workloads/microbenchmark_nccl/experiments/
  microbenchmark_nccl_nccl_fp8_container-nccl_<user>_<jobid>/
    LOG_<date>_<jobid>_h100_sweep_N<nodes>/
      1_1_LOG_all_reduce_*iter1/rank<N>_<node>/stdout.txt  ← bandwidth table
      1_1_LOG_all_reduce_*iter1.txt                        ← rank 0 copy
      1_1_LOG_all_reduce_*iter1.yml                        ← test metadata
      test_sweep_metadata.yml                              ← system info
```

---

## 4. Results Parser (`parse_train_timing.sh`)

Parses `train_step_timing` and `TFLOPS_per_GPU` from all `.out` files in an
experiments directory, reporting mean and standard deviation.

### Synopsis

```
bash $PARSER [OPTIONS] [experiments_directory]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--max-steps=N` | Auto-set analysis window to last 10 iters of an N-step run | — |
| `--min-iter=N` | Start of analysis window (zero-indexed) | `35` |
| `--max-iter=N` | End of analysis window (zero-indexed) | `44` |
| `--format=FORMAT` | Output format: `table`, `csv`, `json` | `table` |
| `--full-names` | Show full filenames instead of shortened | off |
| `[experiments_dir]` | Path to directory containing `.out` files | `./experiments` |

### Examples

```bash
# Run from workload directory (uses ./experiments automatically)
cd $LLMB_INSTALL/workloads/pretrain_llama3.1
bash $PARSER

# Pass the experiments path directly (run from anywhere)
bash $PARSER $LLMB_INSTALL/workloads/pretrain_llama3.1/experiments

# Short run with only 10 steps
bash $PARSER --max-steps=10

# Custom iteration range
bash $PARSER --min-iter=40 --max-iter=49

# CSV output (for spreadsheets or further processing)
bash $PARSER --format=csv
bash $PARSER --format=csv > results.csv

# JSON output (for scripting)
bash $PARSER --format=json

# Full filenames + JSON (for Nemotron4-15B results)
bash $PARSER --format=json --full-names \
  $LLMB_INSTALL/workloads/pretrain_nemotron4-15b/experiments
```

### Example table output

```
Train Step Timing and TFLOPS Analysis (iterations 35-44)
================================================================================
Experiment                                         Status    Time Mean (s)  ...
---------                                          --------  -------------  ...
8b_fp8_cs_gpus8_tp1_pp1_cp1_vpNone_ep1_mbs1_gbs128  Success       9.312      ...

Summary:
  Success experiments: 1
  Failed experiments: 0
  Success rate: 100%
```

---

## `llmb-run submit` — Full Option Reference

```
llmb-run submit [OPTIONS]

  -w, --workload TEXT       Workload name (e.g. pretrain_llama3.1)
  -s, --model-size TEXT     Model size (e.g. 8b, 15b, 70b)
  -d, --dtype TEXT          Data type: fp8, bf16, nvfp4 (comma-separated)
      --scale TEXT          Number of GPUs (comma-separated, e.g. 8,16,32)
      --max-scale INT       Maximum scale for discovery mode
      --min-scale           Only run minimum supported scale
      --exact-scales        Only use scales defined in metadata
  -f, --file TEXT           Path to workload spec file (.txt or .yaml)
  -r, --repeats INT         Number of repeats per configuration [default: 1]
  -p, --profile             Enable Nsight Systems profiling
      --dry-run             Preview jobs without submitting
      --nice INT            Lower Slurm job priority
      --use-sharp           Enable SHARP collectives (requires cluster support)
  -h, --help                Show help and exit
```
