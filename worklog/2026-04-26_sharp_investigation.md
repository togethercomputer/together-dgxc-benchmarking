# SHARP Investigation @ Workload Level — 2026-04-26

**Cluster:** Together AI B200 (use3a-ss), 64 idle nodes (512 GPUs)
**Context:** Earlier same day we discovered SHARP came online between 04-25 and 04-26 (`~/johnson/worklog/2026-04-26_nccl_per_group_scaling.md` Section 8: +33-50% on raw NCCL all_reduce 4n→64n vs 04-25 baseline). After Wave 1 of the 15-job 512-GPU sweep showed identical TFLOP/s to 04-20 (SHARP appeared to make no difference), we drilled into whether SHARP actually engages for LLM training collectives.

**Probe vehicle:** Nemotron-H 56B FP8 / 512 GPU, official config (TP=2, PP=1) — most all_reduce-heavy of the Wave 1 jobs.

---

## 1. Tests run

All 5 jobs reuse the exact same llmb-run-generated sbatch template, with progressively more aggressive SHARP forcing. Container: `nvidia+nemo+26.02.00.sqsh` (NCCL 2.28.9 inside, 2.29.7 host plugin).

| Job | Config delta vs original | Wall | Step time | TFLOP/s/GPU | Outcome |
|----:|---|---:|---:|---:|---|
| 85296 | (Wave 1 baseline, no SHARP) | 6:30 | 5.49s | 1501 | ✓ |
| 85300 | + `NCCL_COLLNET_ENABLE=1` + `NCCL_DEBUG=WARN` + `SHARP_COLL_LOG_LEVEL=3` | 6:45 | 5.50s | 1500 | ✓ |
| 85301 | + `NCCL_DEBUG=INFO` + `NCCL_DEBUG_SUBSYS=INIT,COLL,TUNING` | 7:11 | 5.49s | 1500 | ✓ (1.5 GB log) |
| 85302 | + `NCCL_ALGO=CollNetChain` (force) | 3:18 | — | — | ✗ FAIL |
| 85303 | + `NCCL_ALGO=CollNet` (legacy name) | 3:08 | — | — | ✗ FAIL |
| 85304 | + `NCCL_ALGO=Ring,CollNetDirect` (fallback list) | 7:00 | 5.50s | 1498 | ✓ |

Other env always set on top of stock sbatch: `#SBATCH --chdir=/tmp`, `export HOME=/tmp`, `export NCCL_SOCKET_IFNAME=bond0` (the 70B/405B/Nemotron-H Wave 1 jobs go through llmb-run which sets these implicitly; direct sbatch needs them explicit).

---

## 2. Verifications

### SHARP plugin really did initialize (Job 85300, NCCL_DEBUG=WARN)

```
INFO sharp_job_id:1   resv_key:  tree_type:LLT  tree_idx:0  treeID:0  tree_plane:0  caps:0x66
INFO sharp_job_id:1   tree_type:SAT  tree_idx:1  treeID:512 tree_plane:0  caps:0x76
```

12 LLT + 12 SAT tree allocations per process. Zero `Cannot create SHARP job(-11)` warnings. Same handshake we saw in this morning's per-group SHARP test that gave +33-50% busBW.

### NCCL chose RING for every collective during training (Job 85301, NCCL_DEBUG=INFO)

```
NCCL INFO AllReduce:        4 Bytes -> Algo RING proto LL    channel{Lo..Hi}={0..0}
NCCL INFO ReduceScatter: 67108864 Bytes -> Algo RING proto SIMPLE
NCCL INFO AllGather:     67108864 Bytes -> Algo RING proto SIMPLE
```

Across thousands of distinct NCCL operations and many distinct ranks: **never** observed `Algo CollNet*` or `Algo NVLS*`. NCCL's tuner consistently picked RING.

### Forcing CollNet variants — algorithm/datatype incompatibility

Job 85302 (`NCCL_ALGO=CollNetChain`): every rank failed at first AllReduce with
```
NCCL WARN Error : no algorithm/protocol available for function AllReduce 
with datatype ncclFloat32. NCCL_ALGO was set to CollNetChain.
```
Megatron emits ncclFloat32 4-byte AllReduce calls (PyTorch heartbeat / TP-group syncs). CollNetChain in NCCL 2.28.9 does not implement this combination.

Job 85303 (`NCCL_ALGO=CollNet`): bare-name `CollNet` is not a valid algorithm name in NCCL 2.28.9 → `NCCL WARN Unrecognized element token` → every collective falls into `ncclInvalidUsage`.

Job 85304 (`NCCL_ALGO=Ring,CollNetDirect`): runs end-to-end. Same TFLOP/s as Ring-only. Confirms NCCL still picks Ring at training-relevant sizes even when CollNetDirect is explicitly allowed.

---

## 3. Why training doesn't benefit from SHARP

NCCL's auto-tuner is correct to avoid CollNet at these sizes:

- **Most all_reduce calls in Megatron-Bridge are 4 bytes** (PyTorch heartbeat / TP-group sync) — way too small to amortize SHARP setup overhead
- **Gradient sync goes through ReduceScatter + AllGather at ~67 MB chunks** (FSDP-style). Still well below SHARP's win regime.
- **SHARP's measured +33-50% uplift from this morning** was at **2-16 GB** message sizes — orders of magnitude larger than any training collective.
- The Megatron-Bridge / NeMo log explicitly says `sharp_enabled_group: null` — the framework doesn't request SHARP-enabled process groups.
- Even when CollNet is explicitly permitted (Job 85304), NCCL's per-call tuner still selects Ring for every observed call.

---

## 4. Conclusion

| Question | Answer |
|---|---|
| Is the SHARP plugin loaded in the container? | **Yes** — `sharp_job_id:1`, 12+12 trees |
| Does NCCL use it for training? | **No** — every collective uses Ring |
| Can we force NCCL to use it? | **Not safely** — `CollNetChain` breaks the 4-byte fp32 AllReduce calls; `CollNet` is unrecognized; `Ring,CollNetDirect` runs but tuner still picks Ring |
| Should we enable it for production? | **No** — neutral on perf, +15-30s startup, and `NCCL_ALGO=` overrides actively break workloads |

The +40% SHARP uplift we measured this morning is **real but unactionable for LLM training at this stack/scale**. It would matter for inter-node collective primitives that operate at GB-scale message sizes — which Megatron-Bridge gradient sync isn't.

---

## 5. Recipe (for future SHARP work, e.g. NCCL-only benchmarks inside the container)

```bash
# Add to sbatch wrapper (top-of-script)
#SBATCH --chdir=/tmp                        # mandatory — /home/johnson is empty on compute

# Inside the env-export block, add:
export HOME=/tmp                            # standard pyxis workaround
export NCCL_SOCKET_IFNAME=bond0             # required for NCCL bootstrap on this cluster
export NCCL_NVLS_ENABLE=0                   # already set in stock sbatch
export NCCL_COLLNET_ENABLE=1                # turn SHARP on
# Optional verification (verbose):
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,COLL,TUNING
# export SHARP_COLL_LOG_LEVEL=3
```

**Don't add `NCCL_ALGO=CollNet*`** — it breaks training. If you want to verify SHARP engagement post-hoc, grep the log for `sharp_job_id:` (engaged) vs `Cannot create SHARP` (fallback).

---

## 6. Open questions for follow-up

- Would lowering an internal NCCL CollNet threshold (via custom `NCCL_TUNE_FILE`) actually change anything, or would Ring still win on the 67 MB ReduceScatter?
- Does CollNetDirect cover ncclFloat32 (where CollNetChain doesn't)? The Job 85304 result is consistent with "yes, but never selected"; would need NCCL_DEBUG=INFO + grep for any non-Ring algo to confirm.
- Workloads with bigger collectives (e.g. tensor-parallel gradient sync at TP=8 with very large embeddings, or DP-only without ZeRO) might actually exercise SHARP. Worth checking on a different model where AllReduce sizes are bigger than 67 MB.
- Would a NeMo / Megatron-Bridge config change (e.g. `nccl_communicator_config_path` or explicit SHARP-enabled DDP group) force SHARP to be selected? The `sharp_enabled_group: null` log line suggests the framework has knobs we haven't tried.

---

## 7. Reproduction

All inputs/outputs preserved:
- Per-test sbatch scripts: `/tmp/nh_56b_fp8_512_sharp{,_verbose,_forced,_collnet,_ringcollnet}.sh`
- Logs: `/mnt/vast/johnson/llmb/workloads/pretrain_nemotron-h/experiments/pretrain_nemotronh_56b_fp8_cs_gpus512_*/log-*_853{00,01,02,03,04}_0.out`
- Job-ID timeline: `/tmp/run15_log.txt`
- Memory entries: `feedback_sharp_no_workload_uplift.md`, `project_sharp_not_available.md` (rewritten today)
