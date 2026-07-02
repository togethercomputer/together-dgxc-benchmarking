# Claude Code context for `together-dgxc-benchmarking`

Skills, memory, and conventions for benchmarking work live in the sibling repo:

```
~/together-nccl-tests/.claude/
```

See `together-nccl-tests/CLAUDE.md` for first-time setup and the full convention list.

## What lives here

- `worklog/` — LLM training benchmark reports, profiler analyses, installation notes (see `worklog/README.md` for the index)
- `worklog/template_256gpu_benchmark_report.md` — starting template for a new sweep report
- `worklog/profiler_reports/` — nsys / PyTorch profiler analyses

## Filing reports

LLM training benchmark reports go in `worklog/` with `YYYY-MM-DD_<descriptor>.md` filename. After writing, update the Reports table in `worklog/README.md`.

NCCL / fabric reports go in `together-nccl-tests/baselines/<cluster>/` instead — see that repo's CLAUDE.md for full routing.

Run artifacts (`.log`, `.out`, `.sbatch`, `.tsv`) stay in `~/` and are not committed.

## GB200 rack readiness benchmarks

Per-rack acceptance suite lives in `tests/gb200/GB200/`, one YAML per workload per rack (`<workload>-<rack>.yaml`, e.g. `qwen-72b-r01.yaml`):

| Workload | File | Nodes / GPUs | Config |
|---|---|---|---|
| NCCL all_reduce | `nccl-18n-<rack>.yaml` | 18 / 72 (full rack) | SSH+MPI `all_reduce_perf` 128M→32G |
| Qwen2.5 72B BF16 | `qwen-72b-<rack>.yaml` | 16 / 64 | TP=8 PP=4 DP=2, GBS=512 |
| GPT-OSS 120B BF16 | `gpt-oss-120b-<rack>.yaml` | 16 / 64 | EP=64 DP=64, MBS=4 GBS=1280 |

Training is **16 nodes, not 18**: the parallelism must divide 64 GPUs cleanly (Qwen TP8×PP4×DP2=64; GPT-OSS EP=64 for a 128-expert MoE) — 72 GPUs doesn't factor. NCCL is the full-rack 18n fabric test. **Methodology:** run each workload ×2, report the median of the last 30 steady-state steps (exclude warmup + periodic checkpoint-step dips), averaged across the two runs. Baselines for comparison (TFLOP/s/GPU unless noted): NCCL 915–934 GB/s, Qwen 700–746, GPT-OSS 382–402 (racks R01/R07/R08/R10). File the report per "Filing reports" above and mirror the R07/R01 report structure.

## Running a rack campaign — gotchas

- **Reserve the rack with a taint, not cordon.** Cordon gets reverted by lightricks scheduling automation; a taint is durable. Apply `together.ai/benchmark=<user>:NoSchedule` to all rack nodes (`kubectl taint nodes -l nvidia.com/gpu.clique=<CLIQUE_ID> ...`) and add the matching toleration to the YAMLs. Release with `...together.ai/benchmark:NoSchedule-` + re-cordon. (Per platform/Cody.)
- **Single rack = one job at a time** (NCCL needs 18, training needs 16) → run stages sequentially, tearing down + waiting for pods to clear between each. Submitting all at once gang-deadlocks.
- **NCCL JobSet never self-completes** — workers `sleep infinity`, only rank-0 exits. Wait for the rank-0 *pod* `Succeeded`, capture the 32 GiB busbw, then `kubectl delete -f` to free the rack.
- **Step-log format** (`nemo:26.02` / Megatron-bridge): `Step Time : Xs GPU utilization: Y MODEL_TFLOP/s/GPU` — NOT `iteration N/M`. Parse `MODEL_TFLOP` lines.
- **~90 steady-state steps** is enough for a stable last-30 median; don't run to `train_iters`.
- **Keep JobSet names ≤ ~22 chars** — `setHostnameAsFQDN: true` + long names exceed the 64-char pod-FQDN limit and pods hang in `ContainerCreating`.
