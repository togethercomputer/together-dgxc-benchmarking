# 405B NVFP4 256-GPU regression investigation — 2026-04-27

## Summary

Spent the afternoon isolating the 1,789 → 1,665 regression on 405B NVFP4 256-GPU between 2026-04-22 and 2026-04-27.

**Conclusion:** the −7% loss is **true cluster drift**, not container, install, or node-selection state. Same container bits (md5 `4a3edb8`), same Megatron-Bridge commit (`6b3b5ba7`), same recipe, on similar nodes — and we still cannot reproduce 04-22's 1,789. Best today is 1,675.

**Bonus finding:** the new `26.02.01` container is ~10% slower than `26.02.00` for B200 NVFP4 — NVIDIA's `metadata.yaml` pin (`b200 → 26.02.00 + MB 6b3b5ba7`) is correct and load-bearing.

## Final scoreboard (256 GPU, TP=4 PP=16 CP=1 VP=8 GBS=1536, MBS=1)

| Job | Date | Install | Container md5 | MB commit | Steady-state TFLOP/s/GPU | vs 04-22 |
|---|---|---|---|---|---:|---:|
| 84591 | **04-22** | old | `4a3edb8` | 6b3b5ba7 | **1,789** | baseline |
| 85430 | 04-27 | old, curated nodes | `4a3edb8` | 6b3b5ba7 | 1,668 | −6.8% |
| 85432 | 04-27 | old, disjoint (incl. 4 fabric-weak) | `4a3edb8` | 6b3b5ba7 | 1,662 | −7.1% |
| 85443 | 04-27 | new | `bdfaa1d` (26.02.01) | aeead1ae | 1,492 | −16.6% |
| 85444 | 04-27 | new, container symlinked to old known-good | `4a3edb8` | 6b3b5ba7 | **1,675** | −6.4% |

Steady-state defined as steps 3+ (skipping step 1 compile and step 2 first-measured).

## Timeline

| Time | Action | Outcome |
|---|---|---|
| 15:55 | Submit 85430 — curated 32 nodes from 145-186 range, old install, original 26.02.00 | 1,668 — confirms today's regression |
| 16:22 | Submit 85432 — disjoint 32 nodes (incl. weak 130/197/201/211), old install | 1,662 — fabric-weak hypothesis dead |
| 17:08 | New install at `/mnt/vast/dgxc-benchmarking-0427/dgxc-benchmarking/llmb` complete | Containers pulled |
| 17:39 | Submit 85439 — new install, container `26.02.00` (re-pushed, md5 `bdfaa1d`) | FAILED at 7:04 — `ncclSystemError: Call to stat failed` |
| 17:53 | Add HOME/NCCL_SOCKET_IFNAME/HF_HOME to new cluster_config.yaml; submit 85440 | Cancelled — env vars don't propagate to sbatch.sh anyway |
| 17:57 | Edit launch.sh to use 26.02.01; submit 85441 | FAILED at 1:32 — `ImportError: cannot import name 'llama3_8b_finetune_config'`. MB commit mismatch with container. |
| 18:06 | Checkout MB to matching `aeead1ae`; submit 85442 | Cancelled — bad watcher matched modelopt UserWarning containing string "ImportError" |
| 18:08 | Resubmit 85443 with smarter watcher | 1,492 — 26.02.01 is real but ~10% slower than 26.02.00 |
| 18:45 | Path A: revert launch.sh to 26.02.00, MB to 6b3b5ba7, **symlink old known-good image (md5 4a3edb8)** into new install's image dir; submit 85444 | 1,675 — matches today's other runs; re-push not the cause |
| 19:23 | Cancel 85444 after 25 measured steps (max_steps patch missed) | 25 steps avg = 1,674.9 |

## What we ruled out

### 1. Nodelist / fabric-weak nodes
- 85430 (curated, all in 145-186 range, no historically-flagged nodes) → 1,668
- 85432 (disjoint, includes 130/197/201/211 fabric-weak) → 1,662
- Δ = 0.4%, well within step-to-step noise.
- Conclusion: node selection contributes <1% at 256 GPU. The 04-26 hypothesis that fabric-weak nodes caused the 512-GPU regression is wrong; the regression is uniform.

### 2. Old install state
- New install at `/mnt/vast/dgxc-benchmarking-0427/dgxc-benchmarking/llmb` reproduces the same number when using identical container + MB code (85444 = 1,675 vs 85430 = 1,668; Δ = 0.4%).
- Conclusion: nothing in the accumulated state at `/mnt/vast/johnson/llmb/` is responsible.

### 3. Container re-push
- `nvcr.io/nvidia/nemo:26.02.00` was re-pushed sometime between 04-11 (when old install pulled) and 04-27 (new install pull). md5s are different (`4a3edb8` vs `bdfaa1d`), sizes identical.
- Path A symlinked the original `4a3edb8` bits into the new install → still 1,675, not 1,789.
- Conclusion: re-push is real and recoverable, but not the cause of the regression.

### 4. Wrong-version interactions
- NVIDIA's `metadata.yaml` pins matched pairs:
  - `b200`: container `26.02.00` + MB `6b3b5ba7` + nemo_run `ab0c4328`
  - `default` (gb200/gb300/b300/h100): container `26.02.01` + MB `aeead1ae` + nemo_run `525d68bf`
- Mismatching them breaks (e.g., `26.02.01 + 6b3b5ba7` → `ImportError: llama3_8b_finetune_config` because the API was renamed in newer MB to `llama3_8b_sft_config`).
- Always change them as a set.

## What we confirmed

### Real cluster drift between 04-22 and 04-27
- Identical container (md5 `4a3edb8`)
- Identical MB code (`6b3b5ba7`)
- Identical recipe / config / parallelism
- Similar 32-node sets in 145-186 range
- Yet 1,789 (04-22) → 1,665–1,675 (04-27)
- The drift is **somewhere on the cluster** — firmware, driver, kernel, fabric configuration, or some host-side state change. **Not anything in our install or scripts.**

### 26.02.01 is slower than 26.02.00 for B200 NVFP4
- Same recipe, just swap container `26.02.00 → 26.02.01` (with matching MB `aeead1ae`): 1,675 → 1,492.
- ~10% loss, consistent across all 9 measured steps in 85443.
- NVIDIA's b200 → 26.02.00 pin in metadata.yaml is intentional and correct. **Do not switch to 26.02.01 on B200.**

## Recommendations

1. **Use the new install with the old container symlinked**: `/mnt/vast/dgxc-benchmarking-0427/dgxc-benchmarking/llmb/images/nvidia+nemo+26.02.00.sqsh` is now a symlink to the original `4a3edb8` image. Ready to reproduce 1,665–1,675 cleanly.
2. **Update Tranche-1 expected numbers**: until cluster drift is fixed, 405B NVFP4 256-GPU baseline should be ~1,670, not 1,789. Same likely true for other workloads.
3. **Investigate cluster drift candidates** in priority order:
   - Driver/firmware updates between 04-22 and 04-26
   - SHARP enablement (operational since 04-26 per NCCL benchmarks)
   - Cluster reboot mentioned in earlier worklog notes around 04-25
   - Kernel or networking config changes
4. **Sweep other workloads** to see which are affected. NVFP4 PP=16 is the most BW-sensitive config; if it loses 7%, FP8 and BF16 may also be affected.

## Files & artifacts

- Old install: `/mnt/vast/johnson/llmb/`
- New install: `/mnt/vast/dgxc-benchmarking-0427/dgxc-benchmarking/llmb/`
- Re-pushed 26.02.00 image preserved at: `/mnt/vast/dgxc-benchmarking-0427/dgxc-benchmarking/llmb/images/nvidia+nemo+26.02.00.repushed-2026-04-27.sqsh`
- Old known-good 26.02.00 image: `/mnt/vast/johnson/llmb/images/nvidia+nemo+26.02.00.sqsh` (md5 `4a3edb8ce9a103e467afbc6f87b364ee`)
- Job IDs: 85429–85444 (16 attempts, 5 completed: 85430, 85432, 85443, 85444)
- Memory: `project_405b_nvfp4_regression_2026-04-27.md`
