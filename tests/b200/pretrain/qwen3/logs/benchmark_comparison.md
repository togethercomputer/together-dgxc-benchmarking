# Qwen3-30B-A3B Pretraining Benchmark Comparison
**Platform:** B300, 8x GPU
**Config:** BF16, GBS=64, TP1 PP1 CP1 EP8 ETP1, MBS=1, 50 steps
**Date:** 2026-03-11

---

## Summary

| Metric | NeMo 25.11.01 | NeMo 26.02.00 | Delta |
|--------|--------------|--------------|-------|
| Avg step time (ms) | ~3,050 | ~2,395 | **-21%** |
| Avg GPU TFLOP/s | ~248 | ~315 | **+27%** |
| Total run (steps 1–50) | ~5m 16s | ~4m 18s | **-18%** |
| Final lm loss (step 50) | 8.174 | 8.175 | ~same |
| Skipped / NaN iterations | 0 / 0 | 0 / 0 | same |

---

## Per-Step Performance (steady state, steps 20–50)

### NeMo 25.11.01
| Step | Elapsed (ms) | TFLOP/s/GPU |
|------|-------------|-------------|
| 20 | 2987.1 | ~252 |
| 25 | 2987.9 | ~245 |
| 30 | 3053.0 | ~246 |
| 35 | 3045.0 | ~248 |
| 40 | 3040.2 | ~248 |
| 45 | 3055.2 | ~247 |
| 50 | 3041.6 | — |
| **Avg** | **~3,050** | **~248** |

### NeMo 26.02.00
| Step | Elapsed (ms) | TFLOP/s/GPU |
|------|-------------|-------------|
| 20 | 2399.3 | ~313 |
| 25 | 2395.1 | ~316 |
| 30 | 2374.4 | ~318 |
| 35 | 2394.5 | ~314 |
| 40 | 2393.7 | ~315 |
| 45 | 2391.2 | ~314 |
| 50 | 2382.8 | — |
| **Avg** | **~2,395** | **~315** |

---

## Training Convergence

Both runs show identical convergence behavior — loss curves and grad norms are consistent across versions, confirming numerical equivalence.

| Step | lm loss 25.11.01 | lm loss 26.02.00 |
|------|-----------------|-----------------|
| 10 | 10.213 | 10.214 |
| 20 | 8.183 | 8.186 |
| 30 | 8.212 | 8.213 |
| 40 | 8.116 | 8.116 |
| 50 | 8.174 | 8.175 |

---

## Conclusion

NeMo 26.02.00 delivers a **~27% throughput improvement** (~248 → ~315 TFLOP/s/GPU) with no regression in training stability or convergence. The gain is consistent across the full run with no skipped or NaN iterations in either version.
