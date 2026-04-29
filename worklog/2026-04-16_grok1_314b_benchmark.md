# Grok1 314B MoE Benchmark Report — 2026-04-16

Grok1 314B Mixture-of-Experts pretraining benchmark on the B200 DGXC cluster. Both BF16 and FP8 runs completed successfully on 256 GPUs (32 nodes), 50 training steps each.

## System Configuration

| Component | Details |
|-----------|---------|
| Model | Grok1 314B MoE (8 experts) |
| GPU | NVIDIA B200, 256 total (32 nodes x 8) |
| Container | nvidia+nemo+25.09.00 |
| Framework | NeMo 2 / Megatron-Core |

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Tensor Parallel (TP) | 4 |
| Pipeline Parallel (PP) | 4 |
| Expert Parallel (EP) | 8 |
| Virtual Pipeline (VP) | 8 |
| Context Parallel (CP) | 1 |
| Expert Tensor Parallel (ETP) | 1 |
| Micro Batch Size (MBS) | 1 |
| Global Batch Size (GBS) | 512 |
| Sequence Length | 8,192 |
| Max Steps | 50 |
| TP Comm Overlap | **Disabled** (PMIx v3/v4 workaround) |

## Results Summary

| Dtype | Job ID | Steps 3-9 Avg (TFLOP/s/GPU) | Step Time (s) | Reference Target | Gap |
|-------|--------|------------------------------|---------------|-----------------|-----|
| BF16 | 84056 | **994** | 8.525 | ~1,025 | -3.0% |
| FP8 | 84061 | **1,219** | 6.964 | ~1,370–1,479 | -11 to 18% |

## BF16 Per-Step Detail (Job 84056)

| Step | Step Time (s) | TFLOP/s/GPU |
|------|---------------|-------------|
| 0 | 85.080 | 100 |
| 1 | 8.354 | 1,016 |
| 2 | 8.343 | 1,018 |
| 3 | 8.427 | 1,007 |
| 4 | 8.496 | 999 |
| 5 | 8.528 | 996 |
| 6 | 8.555 | 992 |
| 7 | 8.538 | 994 |
| 8 | 8.573 | 990 |
| 9 | 8.555 | 992 |
| 10–23 | 8.53–8.66 | 981–994 |
| 24–49 | 8.36–8.59 | 988–1,016 |
| **Avg (1-49)** | **8.50** | **999** |
| **Avg (3-9)** | **8.525** | **994** |

## FP8 Per-Step Detail (Job 84061)

| Step | Step Time (s) | TFLOP/s/GPU |
|------|---------------|-------------|
| 0 | 110.000 | 77 |
| 1 | 7.053 | 1,204 |
| 2 | 7.173 | 1,184 |
| 3 | 6.818 | 1,245 |
| 4 | 6.929 | 1,225 |
| 5 | 7.092 | 1,197 |
| 6 | 6.864 | 1,237 |
| 7 | 6.912 | 1,228 |
| 8 | 7.098 | 1,196 |
| 9 | 7.034 | 1,207 |
| 10–24 | 6.95–7.30 | 1,163–1,234 |
| 25–49 | 6.85–7.22 | 1,177–1,240 |
| **Avg (1-49)** | **7.02** | **1,207** |
| **Avg (3-9)** | **6.964** | **1,219** |

## PMIx v3/v4 Workaround

The 25.09 container ships OpenMPI built with PMIx v3, but the host Slurm runs PMIx v4. This causes `MPI_Init_thread` to fail, which blocks TransformerEngine UserBuffers (UB) initialization — TE's communication overlap mechanism relies on MPI communicators.

### What was disabled

- `TP_COMM_OVERLAP=False` environment variable
- In-container Python patcher applied at launch (SLURM_LOCALID=0 only):
  1. **on_fit_start early return** — skips UB initialization callback in NeMo's `megatron_comm_overlap.py`
  2. **tp_comm_overlap gate bypass** — replaces `if self.config.tp_comm_overlap:` with `if False:` in Megatron-Core's TE extension (2 gates)
  3. **ub_name removal** — removes all `extra_kwargs["ub_name"] = tp_comm_buffer_name` assignments (3 instances)
  4. **.pyc cache clear** — ensures patched source is recompiled
- Other ranks wait for sentinel file `/tmp/.nemo_patch_done` before proceeding

### Performance impact

Disabling TP comm overlap removes compute/communication pipelining for tensor-parallel all-gather and reduce-scatter operations. This accounts for the gap vs reference targets:
- **BF16:** 3.0% below target — modest impact since BF16 compute is slower (more time to overlap)
- **FP8:** 11-18% below target — larger impact because FP8 compute is faster, making the exposed communication latency a bigger fraction of step time

### Resolution path

Fixing the PMIx version mismatch (either updating the container to PMIx v4, or downgrading host Slurm to PMIx v3) would re-enable TP comm overlap and close the gap to reference targets.

## Node Exclusion

Excluded bad nodes: `use3a-ss-b200-gpu-[145,190]` (drained). Jobs ran on:
```
use3a-ss-b200-gpu-[154-155,195-205,209-217,226-229,233-235,238-239,256]
```

## Scripts

- **BF16 sbatch:** `together-dgxc-benchmarking/tests/b200/slurm/grok1/bf16/official/256gpu/sbatch.sh`
- **FP8 sbatch:** `together-dgxc-benchmarking/tests/b200/slurm/grok1/fp8/official/256gpu/sbatch.sh`
- **BF16 experiment:** `/mnt/vast/johnson/llmb/workloads/pretrain_grok1/experiments/pretrain_grok1_314b_bf16_gpus256_tp4_pp4_cp1_vp8_ep8_etp1_mbs1_gbs512/pretrain_grok1_314b_bf16_gpus256_tp4_pp4_cp1_vp8_ep8_etp1_mbs1_gbs512_1776219201/`
- **FP8 experiment:** `/mnt/vast/johnson/llmb/workloads/pretrain_grok1/experiments/pretrain_grok1_314b_fp8_gpus256_tp4_pp4_cp1_vp8_ep8_etp1_mbs1_gbs512/pretrain_grok1_314b_fp8_gpus256_tp4_pp4_cp1_vp8_ep8_etp1_mbs1_gbs512_1776219297/`

## Failed Attempts

| Job | Issue | Root Cause |
|-----|-------|------------|
| 84049 | UB assertion: `_ub_communicators is not None` | Patcher regex didn't match dict key assignments in TE extension |
| 84050 | Same UB assertion + debug output | Confirmed: `extra_kwargs["ub_name"]` uses dict syntax, not kwargs |
| 84053 | Cancelled before launch | Bash single-quote collision with Python raw strings |
