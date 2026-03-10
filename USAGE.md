# Usage Guide — LLM Benchmarking CLI

This guide covers day-to-day usage of the installed workloads and the
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

## 3. Qwen3 Pretraining (`pretrain_qwen3`)

Minimum scale on H100 is **16 GPUs (2 nodes)** for 30B. The 235B model requires
at least 256 GPUs.

### Supported options (H100)

| Model size | Dtypes        | Scales (GPUs)       |
|------------|---------------|---------------------|
| `30b`      | `fp8`, `bf16` | 16, 32, 64, 128     |
| `235b`     | `fp8`, `bf16` | 256, 512            |

> **Note:** Qwen3-30B is a MoE model (30B params, 3B active, EP=8). 8 GPUs is
> not supported on H100 — the expert parallelism alone requires 8 GPUs.

### Commands

```bash
# 30B FP8, two nodes (16 GPUs) — minimum supported scale on H100
llmb-run submit -w pretrain_qwen3 -s 30b --dtype fp8 --scale 16

# 30B BF16
llmb-run submit -w pretrain_qwen3 -s 30b --dtype bf16 --scale 16

# Preview without submitting
llmb-run submit -w pretrain_qwen3 -s 30b --dtype fp8 --scale 16 --dry-run
```

### Monitor and view results

```bash
squeue -u $USER

cd $LLMB_INSTALL/workloads/pretrain_qwen3
bash $PARSER --max-steps=50
bash $PARSER --format=csv > results.csv
```

### Output location

```
$LLMB_INSTALL/workloads/pretrain_qwen3/experiments/
  pretrain_qwen3_30b_a3b_fp8_cs_gpus16_.../
    <name>_<timestamp>/<name>/
      log-*.out        ← training log (step timing, TFLOP/s)
      sbatch_*.out     ← Slurm batch output
```

### DeepEP / HybridEP (GB200/GB300 only)

DeepEP and HybridEP are optimized MoE all-to-all dispatch backends that require
SM100+ architecture (GB200/GB300 with NVLink Switch). They are **not available
on H100** — H100 uses `moe_a2a_overlap=True` instead.

#### Check whether DeepEP is enabled in a running job

```bash
grep -i "moe_enable_deepep\|moe_flex_dispatcher" <path-to-log>.out
```

Expected output with DeepEP active:
```
moe_enable_deepep: true
moe_flex_dispatcher_backend: hybridep
```

#### Disable DeepEP/hybridep on GB300

Edit the config file:
```
$LLMB_INSTALL/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance/configs/qwen3/workload_base_configs.py
```

Find the target GB300 config (e.g. `QWEN3_30B_A3B_GB300_FP8_CS_BASE_CONFIG`) and
set `moe_flex_dispatcher_backend=None`:

```python
# DeepEP DISABLED (standard dispatcher)
QWEN3_30B_A3B_GB300_FP8_CS_BASE_CONFIG = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=8,
    moe_flex_dispatcher_backend=None,         # ← None = standard dispatcher
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)
```

#### Enable DeepEP/hybridep on GB300 (default)

```python
# DeepEP ENABLED (default GB300 config)
QWEN3_30B_A3B_GB300_FP8_CS_BASE_CONFIG = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=8,
    moe_flex_dispatcher_backend="hybridep",   # ← hybridep enables DeepEP
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)
```

The same `moe_flex_dispatcher_backend` field applies to all GB300/GB200 configs
in the file (both `30b` and `235b`). The change takes effect at job generation
time — no container rebuild needed. Resubmit with `llmb-run submit` after editing.

> **Side effect:** `moe_flex_dispatcher_backend="hybridep"` also sets
> `CUDA_DEVICE_MAX_CONNECTIONS=32`. When disabled (`None`), it reverts to `8`.

---

### Installing Qwen3

Use `llmb-install express` which reuses saved cluster config from a prior installation:

```bash
# Check available workload names first
llmb-install express --list-workloads

# Install (HF weights for Qwen3-30B-A3B and Qwen3-235B-A22B are downloaded automatically)
llmb-install express $LLMB_INSTALL -w pretrain_qwen3
```

> The HuggingFace weights are only used for tokenizer/config — they are cached
> at `$LLMB_INSTALL/.cache/huggingface/` and do not need to be re-downloaded
> if already present from a prior install.

> **Git LFS required.** If `llmb-install express --list-workloads` returns
> `Error: Git LFS is not installed`, install it without sudo:
> ```bash
> cd /tmp
> curl -L https://github.com/git-lfs/git-lfs/releases/download/v3.6.1/git-lfs-linux-amd64-v3.6.1.tar.gz | tar xz
> cd git-lfs-3.6.1
> mkdir -p ~/.local/bin && cp git-lfs ~/.local/bin/
> export PATH="$HOME/.local/bin:$PATH"
> echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
> git lfs install
> ```

---

## 4. NCCL Microbenchmark (`microbenchmark_nccl`)

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

## 5. Results Parser (`parse_train_timing.sh`)

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

## 5b. Qwen3 — NCCL Configuration Notes (H100)

### Cluster system setting

`NCCL_ALGO=RING` is set **system-wide** in `/etc/environment` on this cluster.
It overrides any per-job `NCCL_ALGO` setting. **Ring is the correct and stable
algorithm for this cluster** — the admin set this intentionally.

### Checking which algorithm is active

```bash
grep "NCCL INFO NCCL_ALGO" <log>.out
# Expected: NCCL_ALGO set by environment to RING
```

### SHARP availability

The SHARP plugin is installed (`/opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so` v10)
and all 8 IB ports per node support SHARP (`mlx5_X:1/IB/SHARP`). However:

| Attempt | Result |
|---------|--------|
| `NCCL_COLLNET_ENABLE=1` | SHARP plugin loads, but Ring still forced by `/etc/environment` |
| `NCCL_ALGO=CollNetDirect` | **Crashes** — SHARP does not support `ncclFloat32` AllReduce |
| `NCCL_ALGO=NVLS` / `NVLSTree` | **Not applicable** — requires NVSwitch fabric (GB200/GB300 only) |

### Debugging NCCL algorithm selection

```bash
# Submit with NCCL debug logging
MAX_STEPS=10 NCCL_DEBUG=INFO llmb-run submit -w pretrain_qwen3 -s 30b --dtype fp8 --scale 16

# Check algorithm in log
grep "NCCL INFO NCCL_ALGO\|collnet\|SHARP\|NET/IB" <log>.out | grep -v "cudaDriver\|Bootstrap"
```

### Standard submit (recommended for this cluster)

```bash
# Ring is optimal — do not override NCCL_ALGO
llmb-run submit -w pretrain_qwen3 -s 30b --dtype fp8 --scale 16
```

---

## 6. Job Monitoring

```bash
# Check running jobs
squeue -u $USER

# Check completed/failed job status and exit codes
sacct -j <jobid> --format=JobID,State,ExitCode,Start,End,Elapsed

# Monitor a directory growing (e.g. model download or checkpoint)
watch -n 5 du -sh $LLMB_INSTALL/workloads/pretrain_qwen3/

# Tail the training log live
tail -f $LLMB_INSTALL/workloads/<workload>/experiments/<exp>/<name>/log-*.out
```

---

## 7. Troubleshooting

### Parser reports all experiments as "Failed"

**Symptom:** `bash $PARSER` shows `Failed` for all experiments with no timing data.

**Cause:** The parser defaults to iterations 35–44. If your job ran fewer steps
(e.g. 10 or 50), no data falls in that range.

**Fix:** Pass `--max-steps=N` matching your actual run length:

```bash
bash $PARSER --max-steps=10    # for a 10-step job
bash $PARSER --max-steps=50    # for a 50-step job
```

---

### Parser finds no `train_step_timing` data (log format mismatch)

**Symptom:** Parser still shows Failed even with `--max-steps=N`. The log file
exists and has training output, but uses a different format.

**Cause:** Newer NeMo container versions log step timing as:
```
Step Time : 9.3s GPU utilization: 720.0MODEL_TFLOP/s/GPU
```
instead of the `train_step_timing in s:` format the parser expects.

**Workaround:** Extract results manually:
```bash
grep "Step Time" <path-to-log>.out
```

---

### Job hangs or fails: "timeout waiting for task launch"

**Symptom:** `sbatch_*.out` shows:
```
srun: error: timeout waiting for task launch, started 0 of N tasks
srun: StepId=XXX aborted before step completely launched.
```

**Cause:** `--mpi=pmix` is in the `srun` command. The NeMo container's OpenMPI
uses PMIx v3, but the host Slurm uses PMIx v5 — they are incompatible.

**Fix:** Remove `--mpi=pmix` from the workload's `executors.py`:

```bash
# Find the file for your workload
find $LLMB_INSTALL/workloads/<workload> -name "executors.py"

# Edit: remove the "--mpi=pmix" line from srun_args list
```

This fix must be applied separately for each installed workload:
- `pretrain_llama3.1/Megatron-Bridge/scripts/performance/utils/executors.py`
- `pretrain_nemotron4-15b/NeMo/scripts/performance/executors.py`
- `pretrain_qwen3/Megatron-Bridge/scripts/performance/utils/executors.py`

---

### Job hangs: "step creation still disabled"

**Symptom:** `sbatch_*.out` shows `step creation still disabled` and training
never starts. The job eventually times out.

**Cause:** `nemo_run`'s `slurm.sh.j2` template runs `srun hostname` as a
pre-flight step to get the head node IP. On this K8s-backed cluster (where
`slurmd` is PID 1), that first `srun` leaves a zombie `slurmstepd`, blocking
the training `srun`.

**Fix:** Edit the template to use `hostname` directly instead of `srun`:

```bash
# File to edit:
$LLMB_INSTALL/venvs/<venv>/lib/python3.12/site-packages/nemo_run/core/execution/templates/slurm.sh.j2

# Find line containing:
#   head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
# Replace with:
#   head_node_ip=$(hostname --ip-address)
```

> Each workload venv has its own copy — apply to all relevant venvs under
> `$LLMB_INSTALL/venvs/`.

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
