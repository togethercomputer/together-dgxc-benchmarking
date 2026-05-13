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
