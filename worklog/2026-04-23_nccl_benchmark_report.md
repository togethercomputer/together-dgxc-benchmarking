# NCCL Benchmark Report — 2026-04-23

**Cluster:** Together AI B200 (use3a-ss)
**Context:** Triggered by DeepSeek V3 BF16 512-GPU hang during `SeqNum=1 ALLREDUCE`; ran full NCCL collective suite to look for node-level fabric faults.
**Suite:** `~/together-nccl-tests/benchmarks/B200/collective-scaling/submit_all.sh`
**Results:** `/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/collective-scaling/20260423_220412/`
**Exclude list:** `use3a-ss-b200-gpu-[181,190]` (181 = today's confirmed IB/RoCE mismatch in jobs 84650/53/54; 190 = user-requested safety exclude)

---

## 1. Job outcomes (71 total)

| Scale | OK | FAIL (hpcx missing) | OTHER (algo n/a) |
|------:|---:|--------------------:|-----------------:|
| 1 | 3 | 1 | 1 |
| 2 | 9 | 0 | 2 |
| 4 | 8 | 1 | 2 |
| 8 | 6 | 4 | 1 |
| 16 | 9 | 0 | 2 |
| 32 | 5 | 6 | 0 |
| **64** | **0** | **11** | **0** |
| **Total** | **40** | **23** | **8** |

**OK** = produced `Avg bus bandwidth` line.
**FAIL hpcx** = `libmpi.so.40: cannot open shared object file` — `/opt/hpcx/hpcx-init.sh` missing on at least one host in the allocation.
**OTHER** = algorithm not applicable (`all_gather` has no `tree`/`nvls_tree` in NCCL 2.29.7; `nvls` requires ≥2 nodes). Expected.

---

## 2. Bandwidth results (peak at 8 GB msg, busBW GB/s)

### all_reduce

| Nodes | GPUs | Ring | NVLS | Tree | NVLSTree | CollNet/SHARP | Best |
|------:|-----:|-----:|-----:|-----:|---------:|--------------:|-----:|
| 2 | 16 | 343.5 | 706.2 | 336.9 | 706.5 | 372.7 | **706.5 (NVLSTree)** |
| 4 | 32 | 332.1 | 331.7 | 187.3 | 286.5 | 383.5 | **383.5 (SHARP)** |
| 8 | 64 | 309.3 | — | 190.8 | 286.4 | 382.4 | **382.4 (SHARP)** |
| 16 | 128 | 302.4 | 299.0 | 190.5 | 272.1 | 377.2 | **377.2 (SHARP)** |
| 32 | 256 | 293.0 | 289.5 | 191.1 | 270.7 | — (hpcx) | **293.0 (Ring)** |
| 64 | 512 | — | — | — | — | — | **all hpcx fail** |

### all_gather

| Nodes | GPUs | Ring | NVLS | CollNet/SHARP | Best |
|------:|-----:|-----:|-----:|--------------:|-----:|
| 1 | 8 | 649.6 | 660.2 | — | 660.2 (NVLS) |
| 2 | 16 | 340.7 | 340.8 | 370.4 | **370.4 (SHARP)** |
| 4 | 32 | 330.6 | 330.4 | — | 330.6 (Ring) |
| 8 | 64 | — | 313.7 | — | 313.7 (NVLS) |
| 16 | 128 | 305.9 | 308.7 | 374.6 | **374.6 (SHARP)** |
| 32 | 256 | — | — | 351.6 | 351.6 (SHARP) |
| 64 | 512 | — | — | — | all hpcx fail |

### sendrecv P2P (activation-transfer pattern)

| Nodes | GPUs | Avg busBW | Peak busBW |
|------:|-----:|----------:|-----------:|
| 1 | 8 | 644.2 | 637.9 |
| 2 | 16 | 43.3 | 43.3 |
| 4 | 32 | 42.2 | 42.2 |
| 8 | 64 | 24.6 | 24.8 |
| 16 | 128 | 16.1 | 16.1 |
| 32 | 256 | — (hpcx) | — |
| 64 | 512 | — (hpcx) | — |

---

## 3. Comparison vs 2026-04-09 baseline

(Prior baseline from `~/reports/NCCL_Benchmark_Report_20260409.md`, excluded `gpu-158` faulty IB HCA)

| Test | Scale | 2026-04-09 | 2026-04-23 | Delta |
|---|---:|---:|---:|---:|
| all_reduce / SHARP | 8n | 385.1 | 382.4 | −0.7% |
| all_reduce / SHARP | 16n | 384.3 | 377.2 | −1.8% |
| all_reduce / Ring | 32n | 306.3 | 293.0 | −4.3% |
| all_gather / SHARP | 16n | 382.5 | 374.6 | −2.1% |
| all_gather / Ring | 4n | 330.3 | 330.6 | +0.1% |

**Verdict:** All measured points within **±5%** of the Apr-9 baseline. **No fabric degradation** observed on the scales where tests ran.

---

## 4. 64-node blocker — `/opt/hpcx` missing on host

Every 64-node job (11 configs) failed with:

```
/opt/hpcx/hpcx-init.sh: No such file or directory
libmpi.so.40: cannot open shared object file: No such file or directory
```

- `run_slurm.sh` and `submit_all.sh` both `source /opt/hpcx/hpcx-init.sh` from the host.
- `/opt/hpcx` is present on some nodes but not all.
- At 64 nodes we always draw a host without it → exit 127 before any NCCL init.
- Host `/opt/hpcx` was fine in the 2026-04-09 baseline (67 nodes succeeded at 64-node scale then) — indicating the package has since been lost on some nodes, likely during a reimage.

**Action item:** file with cluster ops to reinstall `/opt/hpcx` on all nodes, **OR** switch the benchmark harness to `run_slurm.sh --container <image.sqsh>` so hpcx is sourced from the container and not the host.

---

## 5. GPU / compute health (companion runs, 84721-84724)

- **GPU-health (64 nodes):** ECC Enabled, 0 correctable, 0 uncorrectable, no throttle, clocks 120 MHz (idle as expected).
- **FA4 attention (64 nodes):** 60-62% efficiency, uniform across all 64 nodes, no compute outlier.
- **FP8 GEMM, NVFP4 GEMM (64 nodes):** COMPLETED exit 0.

**No single-node hardware fault detected on compute side.**

---

## 6. Bottom line — did we find the DSV3 culprit?

**No.** The benchmark could not reach 64-node scale natively to probe the same fabric pattern as the DSV3 hang. At scales that did run (≤32 nodes), bandwidth matches the 2026-04-09 baseline within 5%. Compute and ECC are clean.

**Remaining hypothesis for DSV3 BF16 512-GPU hang:** fabric-level issue that only manifests at the PyTorch/NCCL 2.28.9 + Pyxis container stack at 512 GPUs, not reproducible with bare-host NCCL 2.29.7 tests at smaller scale.

---

## 7. Recommended next step

Run NCCL tests *inside the DSV3 container* to bypass the host `/opt/hpcx` gap and hit the exact NCCL/MPI stack used in training:

```bash
cd ~/together-nccl-tests/benchmarks
bash run_slurm.sh \
    --container /mnt/vast/johnson/llmb/images/nvidia+nemo+26.02.00.sqsh \
    --exclude use3a-ss-b200-gpu-[181,190] \
    --nodes 64 -b 1M -B 8G
```

If that passes at 64 nodes → the DSV3 hang is inside Megatron-Bridge/Torch, not the fabric.
If it hangs → confirms a specific-scale fabric issue; NCCL_DEBUG=INFO on a follow-up reveals the hung peer.
