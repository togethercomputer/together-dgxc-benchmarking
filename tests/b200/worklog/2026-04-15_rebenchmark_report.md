# B200 DGXC Re-Benchmark Report — 2026-04-15

Re-ran all NVIDIA official baselines with **26.02 container**, measuring TFLOP/s/GPU averaged over training steps 3-9. Skipped Grok1, Nemotron4 340B (legacy containers), and all 512-GPU configs.

## Cluster State

- **Good nodes** (172-179, 185-190, 195-199, 209-217, 226-229): 32 nodes, all jobs succeed
- **Bad nodes** (145-170, 180-184, 200-205, 233-235, 238-239, 256): crash with `/tmp/pymp` FileNotFoundError and NCCL TCPStore errors. No longer down/drained — must explicitly exclude.
- **Node exclusion:** Use `ADDITIONAL_SLURM_PARAMS="nodelist=use3a-ss-b200-gpu-[172-179,185-190,195-199,209-217,226-229]"` with llmb-run submit. Slurm `--exclude` ranges don't work because some node numbers (147, 153, 158, etc.) don't exist on the cluster.

## Tier 1 Results (64-GPU / 8 nodes) — COMPLETE

| Model | Dtype | TFLOP/s/GPU | Previous | Delta |
|-------|-------|------------|----------|-------|
| Llama 3.1 70B | FP8 | 1,454 | 1,624 | -10.5% |
| Nemotron-H 56B | FP8 | 1,532 | 1,563 | -2.0% |
| Qwen3 30B | BF16 | 205 | 204 | +0.5% |
| Nemotron4 15B | BF16 | 1,468 | 1,571 | -6.6% |
| Nemotron4 15B | FP8 | 1,861 | 1,916 | -2.9% |

## Tier 2 Results (256-GPU / 32 nodes) — 7 of 8 COMPLETE

| Model | Dtype | TFLOP/s/GPU | Previous | Delta | Status |
|-------|-------|------------|----------|-------|--------|
| Llama 3.1 70B | FP8 | 1,546 | 1,494 | +3.5% | Done |
| Llama 3.1 405B | FP8 | 1,724 | 1,722 | +0.1% | Done |
| Llama 3.1 405B | NVFP4 | 1,607 | 1,564 | +2.7% | Done |
| Nemotron-H 56B | FP8 | 1,529 | 1,536 | -0.5% | Done |
| Nemotron4 15B | BF16 | 1,419 | 1,439 | -1.4% | Done |
| Nemotron4 15B | FP8 | 1,624 | 1,800 | -9.8% | Done |
| DeepSeek V3 671B | FP8 | 551 | 406 | +35.7% | Done |
| Qwen3 235B | BF16 | — | — | — | Failed |

### DeepSeek V3 671B FP8 — Steps 3-9 Detail (Job 83853)

| Step | TFLOP/s/GPU | Step Time (s) |
|------|-------------|---------------|
| 3 | 549.5 | 31.01 |
| 4 | 547.9 | 31.10 |
| 5 | 549.8 | 30.99 |
| 6 | 551.0 | 30.93 |
| 7 | 552.2 | 30.85 |
| 8 | 551.7 | 30.89 |
| 9 | 552.5 | 30.84 |
| **Avg** | **550.7** | **30.94** |

Config: TP=1, PP=16, EP=8, ETP=1, MBS=1, GBS=4096, MXFP8

## Remaining Work

1. **Qwen3 235B BF16 256-GPU** — Failed 2 attempts (jobs 83854, 83855). Triton JIT gcc compilation of `__triton_launcher.c` fails inside the 26.02 container. Same class of issue as DeepSeek V3 ptxas/libm.so.6 fix.

   **To fix:** Apply LD_LIBRARY_PATH + TORCHDYNAMO_DISABLE to Qwen3's run_script.py:
   ```
   /mnt/vast/johnson/llmb/workloads/pretrain_qwen3/Megatron-Bridge/scripts/performance/run_script.py
   ```
   Then resubmit:
   ```bash
   cd /mnt/vast/johnson/llmb && ADDITIONAL_SLURM_PARAMS="nodelist=use3a-ss-b200-gpu-[172-179,185-190,195-199,209-217,226-229]" ./llmb-run submit -w pretrain_qwen3 -s 235b -d bf16 --scale 256
   ```

## Issues Encountered and Fixes

### DeepSeek V3: ptxas / libm.so.6 + torch.compile (FIXED)

Triton JIT compilation of `rotary_fwd_kv_kernel` fails with:
```
/usr/local/cuda/bin/ptxas: error while loading shared libraries: libm.so.6: cannot open shared object file
```
Also `torch._inductor` fails with missing `g++` and `Python.h`.

**Fix:** Two patches in `run_script.py` (before any imports):
```python
# Fix 1: LD_LIBRARY_PATH for ptxas/libm.so.6
_ld = os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = f'/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:{_ld}'

# Fix 2: disable torch.compile (container lacks g++ and Python.h)
os.environ['TORCHDYNAMO_DISABLE'] = '1'
```

File: `/mnt/vast/johnson/llmb/workloads/pretrain_deepseek-v3/Megatron-Bridge/scripts/performance/run_script.py`

### Qwen3 235B: Triton gcc compilation (NOT YET FIXED)

Same container issue — Triton JIT needs gcc to compile `__triton_launcher.c` but the compilation fails. Needs the same LD_LIBRARY_PATH fix applied to Qwen3's `run_script.py`.

### Bad Node Exclusion

Bad nodes recovered from down/drained state but still crash jobs. Must explicitly constrain to good nodes. Key discovery: `ADDITIONAL_SLURM_PARAMS` env var (documented in Megatron-Bridge README) passes Slurm parameters through NeMo Run:
```bash
ADDITIONAL_SLURM_PARAMS="nodelist=use3a-ss-b200-gpu-[172-179,185-190,195-199,209-217,226-229]" ./llmb-run submit ...
```

Direct sbatch of NeMo Run-generated scripts does NOT work (Pyxis HOME, HF cache, PYTHONPATH issues). Must always use `llmb-run submit`.

### Nemotron4 15B max_steps

Fiddle-serialized configs embed `max_steps` in `_fn_or_script`, not `_config.yaml`. These jobs ran 50 steps instead of 10 — not blocking (steps 3-9 data still valid).

## Scripts and Commands

- **llmb-run submit:** `cd /mnt/vast/johnson/llmb && ./llmb-run submit -w <workload> -s <size> -d <dtype> --scale <gpus>`
- **Node constraint:** Prefix with `ADDITIONAL_SLURM_PARAMS="nodelist=use3a-ss-b200-gpu-[172-179,185-190,195-199,209-217,226-229]"`
- **Nemotron4 15B:** Pre-existing sbatch scripts in `dgxc-benchmarking/nemotron4-15b/`
- **Master script:** `~/together-dgxc-benchmarking/tests/b200/slurm/run_all_official.sh`
- **Worklog:** `~/johnson/worklog/`

## llmb-run Patches

Added `--exclude` and `--nodelist` CLI options to llmb-run (for future use):
- `/mnt/vast/johnson/llmb_venv/lib/python3.12/site-packages/llmb_run/main.py`
- `/mnt/vast/johnson/llmb_venv/lib/python3.12/site-packages/llmb_run/job_launcher.py`

These work for SbatchLauncher and ConfiguredSbatchLauncher. For Nemo2Launcher (used by DeepSeek V3, Qwen3), use `ADDITIONAL_SLURM_PARAMS` instead.
