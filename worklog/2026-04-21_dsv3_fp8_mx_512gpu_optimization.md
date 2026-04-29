# DeepSeek V3 FP8 MX 512-GPU — Optimization 2026-04-21

**Workload:** DeepSeek V3 671B, FP8 MX, 512 GPUs / 64 nodes, B200, NeMo 26.02 container
**Base config:** TP=1, PP=16, CP=1, EP=8, VPP=None, MBS=1, GBS=8192, 12 steps (iter 5-12 steady-state)
**Baseline:** 593 TFLOP/s/GPU (job 84552, preset `B200_FP8_MX_V1` unmodified)
**Target:** 690 TFLOP/s/GPU (+16.4%)

---

## Summary

Stacked three independent optimizations on the DSV3 FP8 MX preset. Best config lands at **632.4 TFLOP/s/GPU (+6.6% vs baseline)** — CUDA graphs + drop `mla_up_proj` from recompute. A follow-on `--nccl_ub` experiment showed no gain (within noise, -0.5%) and has been dropped from the best-known config.

**Gap to 690 target: 58 TFLOP/s (+9.2%) remains.** All remaining levers carry structural risk (PP reconfig, accuracy tradeoffs).

---

## Results

| Exp | Job | Config delta | Mean TFLOP/s | Δ vs prev | Δ vs baseline |
|---|---|---|---|---|---|
| Baseline | 84552 | preset B200_FP8_MX_V1 | 593 | — | — |
| Exp 1 | 84559 | + CUDA graphs (`attn,moe_router,moe_preprocess`) | 619.3 | +4.4% | +4.4% |
| **Exp 2** | **84560** | **+ `model.recompute_modules=[]`** | **632.4** | **+2.1%** | **+6.6%** |
| Exp 3 | 84561 | + `--nccl_ub true` | 629.1 | **-0.5%** | +6.1% |

All runs: 64 nodes, excluded persistent bad-node list + node 256, same NCCL env, same NeMo 26.02 container. No OOM, no retries, no NaN.

---

## Exp 1 — CUDA graphs (job 84559, 19:20-19:40)

**Delta:** `--cuda_graph_impl transformer_engine --cuda_graph_scope attn,moe_router,moe_preprocess` (matches `GB200_FP8_MX_V1` preset exactly).

**Per-iter TFLOP/s (iter 5-12):** 617.0 / 619.3 / 618.5 / 618.7 / 619.6 / 620.4 / 619.8 / 621.0
**Mean: 619.3 TFLOP/s, step 27.51s**

**Capture pattern:**
- Iter 1: 671s warmup (unchanged from baseline)
- Iter 4: **55.3s / 308 TFLOP/s spike** (late CUDA graph capture finalizing moe_preprocess)
- Iter 5+: steady

4 graphable layers/rank (2 on PP last stage). Closes ~33% of the gap to 690. Not enough alone.

---

## Exp 2 — Drop `mla_up_proj` recompute (job 84560, 20:04-20:23) — BEST

**Delta (stacked on Exp 1):** add Hydra override `"model.recompute_modules=[]"` (preset default is `["mla_up_proj"]`).

**Per-iter TFLOP/s (iter 5-12):** 627.1 / 631.6 / 632.4 / 633.0 / 632.1 / 634.5 / 634.1 / 634.1
**Mean: 632.4 TFLOP/s, step 26.95s** — tight cluster, slight monotonic upward drift.

**Memory headroom verified (PP=16/MBS=1):**
- Most ranks: max-reserved **106-130 GB** (180 GB B200 HBM)
- Rank 480 (PP last stage, LM head): max-reserved 45 GB
- `mem-alloc-retires: 0` on all ranks

Rationale: removing mla_up_proj recompute saves redundant fwd compute on MLA up-projection; at PP=16/MBS=1 activation-memory headroom is ample, so no OOM risk.

---

## Exp 3 — NCCL user buffer (job 84561, 20:31-20:51) — NEGATIVE RESULT

**Delta (stacked on Exp 2):** add `--nccl_ub true` → sets `ddp.nccl_ub=True` and `ddp.average_in_collective=False` (confirmed in log).

**Per-iter TFLOP/s (iter 5-12):** 624.1 / 629.6 / 629.0 / 628.8 / 630.6 / 630.9 / 628.9 / 630.7
**Mean: 629.1 TFLOP/s, step 27.09s**

**-0.5% vs Exp 2.** Within run-to-run noise but trending slightly worse. Not reproducing Exp 2's 634 upper-range iterations.

**Why it didn't help at this scale:**
- DSV3 bottleneck is GEMM/compute-bound on B200, not gradient all-reduce
- DistributedOptimizer with PP=16/DP=32 → small per-bucket gradient sizes; user-buffer alloc savings are marginal
- `--nccl_ub` docstring explicitly says "for FSDP communication"; we are not FSDP

**Recommendation: drop `--nccl_ub` for DSV3 B200 FP8 MX 512-GPU.**

---

## Recommended best config (going forward)

```bash
--cuda_graph_impl transformer_engine \
--cuda_graph_scope attn,moe_router,moe_preprocess \
# Hydra overrides:
train.manual_gc=true train.manual_gc_interval=100 \
"model.recompute_modules=[]"
```

Reproduction: `/mnt/vast/johnson/scripts/dsv3_512gpus_fp8mx_cg_norecompute/{sbatch,run}.sh`

---

## Remaining levers to close the gap to 690

| Lever | Est. gain | Risk | Notes |
|---|---|---|---|
| PP=8 / VPP=2 (B300_V2 preset style) | moderate | **high** | Major reconfig; unproven on B200 FP8 MX; activation memory doubles |
| PP=16 / VPP=2 custom layout | unknown | **very high** | No preset exists; 61 layers → 32 chunks needs custom pp_layout; stacks poorly with recompute=[] |
| `moe_token_drop` | small | accuracy regression | Not acceptable without validation |
| `cross_entropy_fusion` / vocab parallel CE | small | accuracy regression | Not acceptable without validation |
| `moe_a2a_overlap` | — | **proven -30% on Qwen3 FP8** | Skip |

No more low-risk quick wins available. Closing the last 9% requires either a PP reconfig or accepting accuracy-class changes.

---

## Artifacts

- Exp 1 dir: `/mnt/vast/johnson/scripts/dsv3_512gpus_fp8mx_cudagraph/`
- Exp 2 dir: `/mnt/vast/johnson/scripts/dsv3_512gpus_fp8mx_cg_norecompute/` ← **best**
- Exp 3 dir: `/mnt/vast/johnson/scripts/dsv3_512gpus_fp8mx_cg_nr_nccl_ub/`
- Baseline job: 84552 (ran on a prior day)
- Memory: `project_dsv3_cudagraph_exp{1,2,3}.md`
