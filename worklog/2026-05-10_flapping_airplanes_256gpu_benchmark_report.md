# B200 DGXC 256-GPU Benchmark Sweep — 2026-05-10

flapping-airplanes B200 cluster (slinky). All 15 official-config workloads at 256 GPUs (32 nodes). [Brief context: waves, reruns, notable issues.]

## Summary

**Llama 70B FP8 @ 256 GPU (32 nodes) achieves 1794 MODEL_TFLOP/s/GPU (mean iter 3–9), beating the Tranche-1 target (1624) by +10.5% and the MD1 256-GPU reference (1503) by +19.4%.** The patch stack to make the slinky cluster runnable from `llmb-run` is fully verified (pyxis chmod, RLIMIT_MEMLOCK, /dev/shm sizing, `set_sharing_strategy('file_system')`). The key for 256-GPU runs is a **NCCL pre-flight check** (32-node `all_reduce_perf` ~30s) on a candidate node set; submit the LLM training to the same node set within a 10-second window before pod cycling re-occurs. Without preflight, ~9 of 9 attempts failed at NCCL P2P setup; with preflight, succeeded on first try.

## Scoreboard

Steady-state MODEL_TFLOP/s/GPU as reported by NeMo, mean of iterations 3–9.

| # | Model · dtype | Target | **2026-05-10 (256 GPU)** | vs Target | MD1 (2026-05-02, 256 GPU) | vs MD1 | Job ID |
|--:|---|---:|---:|---:|---:|---:|---:|
| 1 | Llama 70B FP8 (mx, TP=2 PP=4 VP=5) | 1,624 | **1,794** | **+10.5% ✓** | 1,503 | **+19.4%** | 2143 |
| 1a | Llama 70B FP8 (cs, FSDP, 128 GPU sub-scale) | 1,624 | 1,584 | −2.5% | 1,503 | +5.4% | 2139 |
| 2 | Llama 70B NVFP4 | 2,013 | — | — | 2,043 | — | — |
| 3 | Llama 405B FP8 | 1,722 | — | — | 1,766 | — | — |
| 4 | Llama 405B NVFP4 | 2,006 | — | — | 1,977 | — | — |
| 5 | Nemotron-H 56B FP8 ‡ | 1,536 | — | — | 1,527 | — | — |
| 6 | DSV3 FP8 | 406 | — | — | 616 | — | — |
| 7 | N4 340B FP8 | 1,101 | — | — | 1,245 | — | — |
| 8 | N4 340B BF16 | 936 | — | — | 868 | — | — |
| 9 | Grok1 314B FP8 | 1,371 | — | — | 1,464 | — | — |
| 10 | Grok1 314B BF16 § | 1,025 | — | — | 1,000 | — | — |
| 11 | N4 15B BF16 | 1,264 | — | — | 1,490 | — | — |
| 12 | Qwen3 235B BF16 | 514 | — | — | 614 | — | — |
| 13 | Qwen3 235B FP8 | 426 | — | — | 514 | — | — |
| 14 | DSV3 BF16 | 500 | — | — | 569 | — | — |
| 15 | N4 15B FP8 | 1,908 | — | — | 1,674 | — | — |

**Job 2143 (256 GPU primary)** per-iter MODEL_TFLOPS: 123.1 (warmup) / 2014.6 / **1674.8 / 1877.0 / 1859.9 / 1875.2 / 1825.7 / 1785.0 / 1659.6** / 1714.3. Mean iter 3–9 = **1793.9**. Wall: 6:05; ~8.6 s/iter steady state at GBS=1024. Topology: TP=2 PP=4 CP=1 VP=5 EP=1 mbs=1 gbs=256 base (auto-doubled by FSDP path). Container: `nvidia+nemo+26.02.00.sqsh`.

**Job 2139 (128 GPU sub-scale)** per-iter MODEL_TFLOPS: 280.1 (warmup) / 1588.1 / **1576.4 / 1582.9 / 1588.1 / 1602.4 / 1589.9 / 1578.3 / 1569.0** / 1578.7. Mean iter 3–9 = **1583.9**. Wall: 6:45 for 10 iters; 9.27 s/iter steady state at GBS=512. Topology: TP=1 PP=1 (FSDP cs config). Container: same.
‡ Nemotron-H blocked by missing `ptxas-blackwell` binary in the `nemo+26.02.01` container (Mamba/Triton). Non-blocking with admin help (newer container).
§ Grok1 PMIx fix (Apr 2026 v7 patcher) not yet re-applied to slinky install — validate before running.

## Notable Results

**Strong:** [models beating target — list with deltas]

**Structural misses:** [models below target — note if consistent with prior runs]

**[Model] fix:** [if a rerun or fix was applied, document root cause and fix here]

## [Optional] [Fix Title] — Root Cause and Workaround

[If a container/PMIx/config fix was needed, document it here with the same structure as the Grok1 PMIx fix in the MD1 2026-05-02 report.]

## Infrastructure Notes

**Node exclude list (today):** `slinky-43,slinky-62` — 2 nodes excluded. Both fail with `ibv_reg_mr_iova2 failed` / `ibv_create_qp ... Cannot allocate memory` on every multi-node test (verified via 2-node pair tests + 62-node attempt 2026-05-09; codex independently reproduced). Likely root cause: `ulimit -l` not actually unlimited on those two nodes (silent failure under cgroup/PAM), or stale HCA QP/MR state. Drain candidates.

**Earlier "20 IB-broken nodes" diagnosis (2026-05-04) is superseded.** Re-tested 2026-05-09: those nodes pass NCCL allreduce when launched via `srun --mpi=pmix` with HPC-X collectives disabled (`OMPI_MCA_coll=^hcoll,ucc`, `pml=ob1`, `btl=tcp,self`). The earlier diagnosis conflated launcher-environment MPI_Init hangs with hardware. With the corrected launcher (now wired into the LLMB containerized path via Nemo container's own MPI), all 60 non-{43,62} nodes are healthy.

**SHARP is inactive on this fabric** (verified 2026-05-09 with both `NCCL_ALGO=CollnetChain` and `CollnetDirect` — both fail with "no algorithm/protocol available"). NCCL all_reduce is Ring-only at multi-node scale, capping at ~388 GB/s busbw vs ~545 GB/s on MD1 with SHARP. Expect comm-bound workloads (Llama 70B FP8 in particular) to track or slightly miss MD1 numbers, not exceed them.

**Cluster fixes applied today (in `executors.py` / `run_script.py` / `perf_plugins.py`):**
1. `setup_lines`: per-job `sudo chmod 1777 /usr/share/enroot/enroot-data` + `sudo prlimit --memlock=unlimited` + `sudo mount -o remount,size=64g /dev/shm` on each allocated node.
2. `srun_args`: `--propagate=MEMLOCK,STACK,NOFILE`, `--container-remap-root`, `--container-writable`.
3. `mounts.append("/dev/shm:/dev/shm")`.
4. `pre_cmds`: `ulimit -l unlimited || true; ulimit -n 524288 || true` (no `2>/dev/null` — Jinja escapes `>`).
5. **`run_script.py`** top: `torch.multiprocessing.set_sharing_strategy('file_system')` + `os.environ.setdefault("TMPDIR", "/tmp")` — avoids EAGAIN from PyTorch shm-fd hitting pod's RLIMIT_MEMLOCK=8KB.
6. `perf_plugins.py`: appends `dataset.num_workers=0` to hydra overrides (note: ineffective for Nemotron-H — recipe overrides; Llama doesn't hardcode).

**256-GPU pod-cycling workaround (verified 2026-05-10):** Naïve 256-GPU submits failed in 9/9 attempts (jobs 2117–2130) with `ncclSystemError` at the ~3:30–4:40 NCCL P2P setup phase. Root cause: random K8s pod IB-MR allocation failures (`ibv_reg_mr_iova2 failed`, `Cannot allocate memory`) during the heavy concurrent NCCL collective. **Workaround that actually works**: run a **2-min, 32-node NCCL `all_reduce_perf` preflight** on the candidate node set just before submitting the LLM job; if it passes, immediately submit Llama on the SAME node set (10-second window). Implemented in `/home/johnson/preflight_then_llm.sh`. Job 2143 succeeded on first try with this method. The bad nodes are not deterministic across hours — slinky-0/1/17/19 were flapping in the 5:30–5:45 window, but slinky-2-8/10-13/15-16/21-26/30-33/38/41/46/47/50/56/58/60/61 were healthy at 5:50.

**Whitelist (verified 2026-05-10 ~05:48 UTC, may rotate)**: `slinky-{2,3,4,5,6,7,8,10,11,12,13,15,16,21,22,23,24,25,26,30,31,32,33,38,41,46,47,50,56,58,60,61}` — saved at `/home/johnson/slinky_32n_whitelist.txt`. Re-verify with NCCL preflight before each large run.

**Permanent bad nodes** (drain candidates): `slinky-43`, `slinky-62` — `ibv_reg_mr_iova2 failed` on every multi-node test, every hour, since 2026-05-09.

## DGXC Benchmarking Setup

| Component | Details |
|-----------|---------|
| Tool | `llmb-run` v1.10.11 |
| Repo (together fork) | `/data/home/johnson/together-dgxc-benchmarking` |
| Repo (upstream, v26.02.01) | `/data/home/johnson/dgxc-benchmarking` |
| Install root (`LLMB_INSTALL`) | `/data/home/johnson/llmb` |
| Installer venv | `/data/home/johnson/llmb_venv` (Python 3.12, uv) |
| Cluster config | `/data/home/johnson/llmb/cluster_config.yaml` |
| Slurm partition | `all` (= slinky / exeamplar-benchmark) |
| GPU type | b200 |

**Container images** (`/data/home/johnson/llmb/images/`):

| Image | Size |
|-------|------|
| `nvidia+nemo+25.07.01.sqsh` | 27 GB |
| `nvidia+nemo+25.09.00.sqsh` | 30 GB |
| `nvidia+nemo+26.02.00.sqsh` | 36 GB |
| `nvidia+nemo+26.02.01.sqsh` | 36 GB |

**Installed workloads (7):** `pretrain_deepseek-v3`, `pretrain_gpt_oss`, `pretrain_nemotron-h`, `pretrain_qwen3`, `pretrain_grok1`, `pretrain_llama3.1`, `pretrain_nemotron4-340b`

**Not yet installed:** `pretrain_nemotron4-15b` — needed for scoreboard rows 11 (N4 15B BF16) and 15 (N4 15B FP8). Either install via `llmb-install pretrain_nemotron4-15b` or mark those two rows N/A for this run.

**Enroot paths** (redirected to WekaFS for persistence across pod restarts):
```bash
export ENROOT_DATA_PATH=/data/home/johnson/.local/share/enroot
export ENROOT_CACHE_PATH=/data/home/johnson/.cache/enroot
export ENROOT_TEMP_PATH=/data/home/johnson/.cache/enroot/tmp
export LLMB_INSTALL=/data/home/johnson/llmb
```

**Fixes applied to installer** (`/data/home/johnson/llmb_venv` site-packages):
1. `llmb_install/downloads/image.py` — srun time limit raised from 35 → 120 min (36 GB images exceeded 35-min mksquashfs limit)
2. `llmb_install/environment/venv_manager.py` — pip upgrade step added after venv creation (system pip 22.0.2 AssertionError bug with nemo-toolkit deps)

**To activate and run:**
```bash
source /data/home/johnson/llmb_venv/bin/activate
cd /data/home/johnson/llmb
llmb-run submit -w <workload> -s <size> --dtype <dtype> --scale 256
```

## Configurations

| Model · dtype | Container | Parallelism | GBS |
|---|---|---|---:|
| Llama 70B FP8/NVFP4 | nemo+26.02 | TP=2 PP=4 CP=1 VP=5 | 256 |
| Llama 405B FP8 | nemo+26.02 | TP=4 PP=8 CP=2 VP=8 | 3072 |
| Llama 405B NVFP4 | nemo+26.02 | TP=4 PP=8 CP=1 VP=4 | 1536 |
| Nemotron-H 56B FP8 | nemo+26.02 | TP=2 PP=1 CP=1 | 192 |
| DSV3 FP8/BF16 | nemo+26.02 | TP=1 PP=16 CP=1 EP=8 | 4096 |
| N4 340B FP8/BF16 | nemo+25.07 | TP=8 PP=4 CP=1 VP=12 | 64 |
| Grok1 314B FP8/BF16 | nemo+25.09 | TP=4 PP=4 CP=1 VP=8 EP=8 ETP=1 | 512 |
| N4 15B BF16 | nemo+26.02 (compat_runner) | TP=1 PP=1 CP=1 VP=1 | 2048 |
| N4 15B FP8 | nemo+25.09 | TP=1 PP=1 CP=1 VP=1 | 2048 |
| Qwen3 235B BF16/FP8 | nemo+26.02 | TP=1 PP=8 CP=1 VP=4 EP=8 | 8192 |

[Update any configs that changed from the above.]

## Job Timeline

| Wave | Job | Slurm ID | Notes |
|------|-----|----------|-------|
| 1 | llama70b_fp8 | — | — |
| 1 | llama70b_nvfp4 | — | — |
| 1 | llama405b_fp8 | — | — |
| 1 | llama405b_nvfp4_pp8 | — | — |
| 1 | nemotronh56b_fp8 | — | — |
| 2 | dsv3_fp8 | — | — |
| 3 | n4340b_fp8 | — | — |
| 3 | n4340b_bf16 | — | — |
| 3 | grok1_fp8 | — | — |
| 3 | grok1_bf16 | — | — |
| 4 | n15b_bf16 | — | — |
| 4 | qwen3_bf16 | — | — |
| 4 | qwen3_fp8 | — | — |
| 4 | dsv3_bf16 | — | — |
| 5 | n15b_fp8 | — | — |

## Open Items

- [Any nodes to drain, models to profile, or fixes to upstream]

## Cross-references

- **MD1 cluster 256-GPU report (2026-05-02):** `2026-05-02_256gpu_benchmark_report.md`
- **Worklog dir:** `/data/home/johnson/together-dgxc-benchmarking/worklog/`
- **Cluster config:** `/data/home/johnson/llmb/cluster_config.yaml`
- **Workload experiments:** `/data/home/johnson/llmb/workloads/<workload>/experiments/`
