# SHARP Not Operational on B200 Cluster — Evidence Report

**Date:** 2026-04-11
**Cluster:** Together AI B200 (use3a-ss-b200-gpu-*)
**Author:** Johnson
**Purpose:** Request admin enablement of IB SHARP for NCCL collective acceleration

---

## Summary

IB SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) is **not operational** on the B200 cluster. NCCL jobs requesting SHARP (`NCCL_COLLNET_ENABLE=1`) fail at init and silently fall back to Ring/Tree algorithms. This report documents the evidence from both direct diagnostics and production training workloads.

---

## Evidence

### 1. `sharp_hello` — No Reservation

Run on `use3a-ss-b200-gpu-256` (2026-04-11):

```
$ /opt/mellanox/sharp/bin/sharp_hello

[use3a-ss-b200-gpu-256:0:1078536 - context.c:679] INFO job (ID: 17800447739156601301)
  resource request quota: ( osts:0 user_data_per_ost:0 max_groups:0 max_qps:1
  max_group_channels:1, num_trees:1)
[use3a-ss-b200-gpu-256][warn] - Begin job id: 17800447739156601301 failed with status: No reservation
[use3a-ss-b200-gpu-256] ERROR sharp_get_job_data_len failed: Job error(-35)
[use3a-ss-b200-gpu-256] ERROR SHArP Job init error: No reservation
sharp_coll_init failed: Cannot create SHARP job
```

### 2. `sharpd` Daemon — Not Running

```
$ systemctl status sharpd
Unit sharpd.service could not be found.

$ pgrep -a sharpd
(no output — no processes found)

$ ps aux | grep sharp
(no output — no processes found)
```

### 3. No SHARP Runtime or Configuration Files

```
$ ls /var/run/sharp* /tmp/sharp*
ls: cannot access '/var/run/sharp*': No such file or directory
ls: cannot access '/tmp/sharp*': No such file or directory

$ ls /etc/sharp* /etc/mellanox/sharp*
ls: cannot access '/etc/sharp*': No such file or directory
ls: cannot access '/etc/mellanox/sharp*': No such file or directory
```

### 4. SHARP Binaries Installed but Unconfigured

```
$ ls -la /opt/mellanox/sharp/bin/
sharp_am               2710344 bytes
sharp_cmd                22760 bytes
sharp_coll_dump_config   14488 bytes
sharp_coll_test          74624 bytes
sharp_hello              14744 bytes
```

Binaries were installed as part of Mellanox OFED but the SHARP infrastructure (Aggregate Manager, daemons, reservations) was never configured.

### 5. Production Training Job — SHARP Fallback (Job 80018)

**Workload:** Qwen3 235B A22B BF16, 256 GPUs (32 nodes), `NCCL_COLLNET_ENABLE=1`
**Date:** 2026-04-10, Job ID: 80018
**Nodes:** use3a-ss-b200-gpu-[159-190]
**Script:** `/home/johnson/johnson/scripts/qwen3_235b/256gpus_bf16_sharp/sbatch.sh`
**Log:** `log-bf16_sharp_80018_0.out`

All 8 ranks on the head node hit "No reservation" errors during NCCL init:

```
[use3a-ss-b200-gpu-159][warn] - Begin job id: ... failed with status: No reservation
[use3a-ss-b200-gpu-159] ERROR SHARP Job init error: No reservation
```

NCCL fell back to Ring/Tree. Training ran but showed **no improvement**:

| Config | Avg Step Time (iter 2+) | TFLOP/s/GPU |
|--------|------------------------|-------------|
| Baseline (COLLNET=0) | 55.9s | 337.7 |
| SHARP attempt (COLLNET=1) | 56.5s | 343.7* |

*Only 2 iterations completed before job cancellation; step time is within noise of baseline, confirming SHARP was never active.

### 6. InfiniBand Fabric — Healthy

IB itself is functional (rules out fabric-level issues):

```
$ ibstat
CA 'mlx5_0': State Active, Rate 400, Link layer InfiniBand
CA 'mlx5_1': State Active, Rate 400, Link layer InfiniBand
```

---

## What Is Needed to Enable SHARP

1. **Start SHARP Aggregate Manager (`sharp_am`)** on the management/UFM node
2. **Start `sharpd` daemons** on all compute nodes (via systemd or UFM)
3. **Create SHARP reservations** for jobs (typically managed through UFM)
4. **Verify** with `sharp_hello` returning success on compute nodes

This is typically managed via **Unified Fabric Manager (UFM)** and requires admin-level access to the IB fabric management infrastructure.

---

## Expected Impact

Based on NCCL micro-benchmarks, SHARP could improve **inter-node AllReduce** bandwidth:

- Current (Ring): ~107 GB/s at large message sizes
- Expected (SHARP): ~385 GB/s (3.6x improvement)

Impact on training workloads depends on DP AllReduce fraction:

| Workload | DP AllReduce % of Step | Estimated Speedup |
|----------|----------------------|-------------------|
| Qwen3 235B (EP=8, DP=32, 256 GPU) | ~1.6% | ~1.2% |
| Llama 405B (DP=8, 256 GPU) | Higher DP fraction | More significant |
| Large-scale DP-heavy workloads | Dominant | Up to 10-15% |

SHARP is most impactful for workloads with high data-parallel communication overhead (large DP degree, large gradient sizes, no expert parallelism).

---

## Reproduction Commands

Anyone with access to a compute node can verify:

```bash
# Test 1: SHARP reservation check
/opt/mellanox/sharp/bin/sharp_hello

# Test 2: sharpd daemon status
systemctl status sharpd
pgrep -a sharpd

# Test 3: Runtime files
ls /var/run/sharp* /tmp/sharp*

# Test 4: NCCL with SHARP (multi-node job)
NCCL_COLLNET_ENABLE=1 NCCL_DEBUG=INFO <nccl_test> 2>&1 | grep -i "sharp\|collnet"
```
