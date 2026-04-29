# Session A Briefing — 512-GPU Re-run of Friday's 10 Models

**Your mission:** Re-run the 10 models that already succeeded at 512 GPUs on Friday (2026-04-17). Scripts are already finalized. Most of the work is babysitting and catching teardown hangs.

**What Session B is doing (don't touch):** The 5 brand-new/never-run configs — Nemotron4 15B BF16/FP8, Qwen3 235B BF16/FP8, DeepSeek V3 BF16 — at 512 GPUs. Session B needs full attention for config debugging.

---

## Context you need

- **Today's date:** 2026-04-20 (Monday). Last successful full-suite run was Friday 2026-04-17.
- **Cluster:** Together AI B200, 70 idle nodes on `batch` partition. Each 512-GPU job uses 64 nodes — only 1 job runs at a time.
- **Unhealthy nodes today** (exclude from any manual sbatch): `use3a-ss-b200-gpu-[130,190,197,199,201,211,233,239]` (130 is missing `/opt/hpcx`).
- **Main scripts live at:** `~/together-dgxc-benchmarking/tests/b200/slurm/`
- **Worklogs:** `~/johnson/worklog/` — especially `2026-04-17_512gpu_benchmark_report.md` (Friday's result + debug log).
- **Friday's patched sbatch scripts** (for direct resubmit fallback): under `/mnt/vast/johnson/llmb/workloads/pretrain_*/experiments/*_gpus512_*/...sbatch_patched.sh` for the 4 legacy models.

---

## The 10 models to re-run

| # | Model | Dtype | Path | Friday TFLOPS/GPU | Friday Job |
|---|---|---|---|---|---|
| 1 | Llama 3.1 70B | FP8 | llmb-run | 1,526 | 84245 |
| 2 | Llama 3.1 70B | NVFP4 | llmb-run | 2,378 | 84246 |
| 3 | Llama 3.1 405B | FP8 | llmb-run | 1,766 | 84247 |
| 4 | Llama 3.1 405B | NVFP4 | llmb-run | 1,789 | 84248 |
| 5 | Nemotron-H 56B | FP8 | llmb-run | 1,502 | 84249 |
| 6 | DeepSeek V3 | FP8 | llmb-run | 594 | 84255 |
| 7 | Nemotron4 340B | FP8 | legacy+patch | 1,234 | 84272 |
| 8 | Nemotron4 340B | BF16 | legacy+patch | 860 | 84275 |
| 9 | Grok1 314B | BF16 | legacy+patch | 1,004 | 84278 |
| 10 | Grok1 314B | FP8 | legacy+patch | 1,454 | 84279 |

**Goal:** confirm stability (±2% of Friday's numbers) and produce a comparison report vs 2026-04-17.

---

## How to run

### Option A — single command (preferred)

```bash
cd ~/together-dgxc-benchmarking/tests/b200/slurm/
bash run_all_official.sh --tier3
```

This submits all 10 sequentially with the correct workflow (`patch_legacy_sbatch()` for 340B/Grok1, `llmb-run` for the rest). Friday's run used exactly this.

### Option B — re-submit Friday's scripts directly (fallback)

For the 4 legacy models, if the `run_all_official.sh` path fails, you can `sbatch` Friday's patched scripts directly:

```
/mnt/vast/johnson/llmb/workloads/pretrain_nemotron4-340b/experiments/.../*_sbatch_patched.sh
/mnt/vast/johnson/llmb/workloads/pretrain_grok1/experiments/.../*_sbatch_patched.sh
```

(Several experiment-dir timestamps exist per model — use the most recent `*_1776491*` dirs which were the final successful Friday runs.)

---

## Known issues to watch for

These are based on today's NCCL benchmark session and Friday's debug timeline:

1. **Teardown hangs** — jobs sometimes complete TFLOPs and then hang in MPI/UCX finalize for 10+ min. If you see the step table printed and `Avg TFLOPS` captured but the job is still RUNNING past expected wall time, check the log — if training is clearly done, `scancel <job_id>` to free the cluster. **Don't wait for the 20-min timeout to kill it.**

2. **Node 130 missing `/opt/hpcx`** — discovered today. Exclude via `--exclude=use3a-ss-b200-gpu-[130,190,197,199,201,211,233,239]` on any manual sbatch. `run_all_official.sh` may already have an exclude; verify.

3. **Legacy container models need patches** — 340B FP8/BF16 + Grok1 BF16/FP8 use 25.07/25.09 containers that hit pyxis/PMIx/UserBuffers/tokenizer errors without patches. The `patch_legacy_sbatch()` helper in `run_all_official.sh` applies all workarounds. Don't try to run raw `launch.sh` output.

4. **NVFP4 max_steps** — a known `sed` gotcha from Friday: the pattern `--max_steps=50` (with `=`) wasn't matched originally. Fix is already in the script (uses `sed -E`). If you see a 512-GPU NVFP4 job running 50 steps instead of 10, that's the bug regressing.

5. **Nemotron-H segfault on Python shutdown** — Friday's Nemotron-H ran all 10 iters successfully, then segfaulted during Python teardown. **Data is still valid** — just don't panic when the exit code is non-zero. Iter-3-9 average is what counts.

6. **CUDA graph recapture anomaly on iter 4** — some models show a one-off slow iter 4 (e.g. Friday's Nemotron-H: 10.1s, 815 TFLOPS). Exclude from the average if you see this pattern.

---

## What counts as a successful re-run

For each model, capture:
- Job ID
- Iter 3-9 average step time (s) and TFLOPS/GPU
- Wall time
- Delta vs Friday's number (±%)

Acceptance: all models within ±5% of Friday's TFLOPS/GPU, no new unexpected failures.

---

## Reporting — produce two things when done

1. **Per-step tables per model** in `~/johnson/worklog/2026-04-20_512gpu_rerun_report.md`, structured like `2026-04-17_512gpu_benchmark_report.md` (same sections: Summary / Cluster State / Results table / Per-Step Details / any debug notes).
2. **Delta table vs 2026-04-17** — just like today's NCCL report did vs 2026-04-17.

If durations differ noticeably from Friday, note that in the Summary. If you hit any new failures, add a Debugging Timeline section.

---

## Coordination with Session B

- **Job name conflicts:** extremely unlikely — Session B will use distinct labels (15B/Qwen3/DeepSeek-BF16), your 10 reuse Friday's label scheme.
- **Cluster contention:** Slurm serializes; your 10 jobs will queue behind / ahead of Session B's 5 jobs based on submit time. First-come first-served. Feel free to start whenever — don't wait for B.
- **Shared files:** `~/together-dgxc-benchmarking/tests/b200/slurm/run_all_official.sh` — Session B **may** edit this to add the 5 new models. If your run fails because of a script edit, check `git status` / `git diff` in that repo.
- **If Session B needs help:** after your run is complete and report written, check in. The comparison report vs Friday is the main artifact.

---

## Memory & references to load on startup

Run these to orient yourself:
- Read `~/johnson/worklog/2026-04-17_512gpu_benchmark_report.md` (full Friday context, debug timeline)
- Read memory file: `~/.claude/projects/-home-johnson/memory/project_512gpu_benchmarks.md`
- Read memory file: `~/.claude/projects/-home-johnson/memory/feedback_pyxis_workarounds.md`
- Read memory file: `~/.claude/projects/-home-johnson/memory/project_mpi_stub_status.md`
- Read memory file: `~/.claude/projects/-home-johnson/memory/feedback_compat_runner.md`

---

## First actions

1. Read the 04-17 report and the 4 memory files above.
2. `sinfo -o "%P %a %D %t %N %G"` — confirm cluster state matches what's described here.
3. `bash run_all_official.sh --tier3` — kick off.
4. `squeue -u johnson` every few minutes; watch for teardown hangs (see issue #1 above).
5. As each job completes, parse iter 3-9 from its log and record. Don't wait for all 10 to finish before starting the report — write incrementally.

Good luck. Ping back if you hit anything surprising — I'll be on Session B working the new-model debug.
