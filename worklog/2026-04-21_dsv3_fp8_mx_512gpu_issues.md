# DeepSeek V3 FP8 MX 512-GPU — Cluster Issues 2026-04-21

**Workload:** DeepSeek V3 671B, FP8 MX, 512 GPUs / 64 nodes, B200, NeMo 26.02 container
**Config:** TP=1, PP=16, EP=8, MBS=1, GBS=8192, 50 steps
**Container:** `/mnt/vast/exemplar/llmb/images/nvidia+nemo+26.02.00.sqsh`

---

## Summary

Two independent cluster-level failures blocked the 512-GPU benchmark today:

1. **IB fabric failure (job 84541)** — job ran 32/50 iters at steady state (~618 TFLOP/s/GPU), then died with `NET/IB status=12 vendor err 129` (HCA retry-exceeded) on `mlx5_15`.
2. **NCCL init hang (job 84544)** — identical config on the same 64-node allocation minus one node: all ranks stalled silently at distributed init. 0% GPU, Python procs alive but blocked. Never reached iter 1 after 25+ min.

Both failures happened at **infrastructure layer** — the workload itself is healthy (32 clean iters prove it).

---

## Timeline

| Job | Submitted | Outcome | Notes |
|---|---|---|---|
| 84541 | 13:43:26 | **FAILED** @ iter 32/50 (14:20:32) | IB error; ran well up to failure |
| 84542 | 15:29:08 | FAILED after 1:44 | My resubmit; `sudo sbatch` stripped `HF_HOME` → HF offline cache miss |
| 84543 | 15:33:24 | FAILED after 4:57 | Added `HF_HOME`, but `WorkDir=/home/johnson` (not on compute nodes) → NCCL `stat failed` |
| 84544 | 15:40:04 | **HUNG** (still RUNNING @ 25+ min, 0% GPU) | Fixed WorkDir; now init hang |

---

## Issue 1 — IB / NCCL remote error (job 84541)

**Timing:**
- Iters 2-32 steady at 27.57-27.61 s/iter → **~618 TFLOP/s/GPU**
- Loss descended cleanly (11.9 → 7.6)
- First error at **14:19:05** (~2:30 after iter 32)

**First rank to fail:** 391 (local rank 7 on node index 48 of allocation → **`use3a-ss-b200-gpu-204`**)

**Error:**
```
[rank391] NCCL error: remote process exited or there was a network error, NCCL version 2.28.9
ncclRemoteError: A call failed possibly due to a network error or a remote process exiting prematurely.
Last error:
NET/IB: Got completion from peer 7.247.232.172<39936> with status=12 opcode=129 len=0 vendor err 129 (Recv) hca mlx5_15
```

**Cascading peers (IB fabric IPs):** `7.247.232.{168,171,172,175}`

**Interpretation:** `status=12` is IBV_WC_RETRY_EXC_ERR (retry counter exceeded). mlx5_15 either dropped a remote peer or the peer's HCA stopped responding. 600 s collective timeout → SLURM killed job with ExitCode 15.

**Allocation nodes (64):**
```
use3a-ss-b200-gpu-[144-146,154-162,164-190,195-205,209-217,226-230]
```

**Action taken:** resubmitted with `--exclude=use3a-ss-b200-gpu-204`.

**Ask for infra:**
- Check HCA/IB health on `use3a-ss-b200-gpu-204` and the peers `7.247.232.{168,171,172,175}`; specifically `mlx5_15`.
- Any fabric events (switch flap, SM blip) around 14:19 UTC on 2026-04-21?
- Full log: `/mnt/vast/exemplar/llmb/workloads/pretrain_deepseek-v3/experiments/pretrain_deepseek_v3_fp8_mx_gpus512_*/…_84541_0.out`

---

## Issue 2 — NCCL init hang (job 84544)

**Setup:** Same generated sbatch script as 84541. Submit-time fixes:
- `HF_HOME=/mnt/vast/exemplar/llmb/.cache/huggingface`
- `HOME=/tmp`
- `sbatch -D /mnt/vast/exemplar/llmb/workloads/pretrain_deepseek-v3/Megatron-Bridge`
- `--exclude=use3a-ss-b200-gpu-204`

**Allocation (63 of original 64, + 148-149 as fill):**
```
use3a-ss-b200-gpu-[144-146,148-149,154-162,164-190,195-203,209-217,226-230]
```

**Observation:**
- Start 15:40:04; first log line at 15:41:34 (normal), last log line at 15:42:10.
- **No log output for 23+ minutes** (runtime 00:25:24 at time of escalation).
- `nvidia-smi` on node 144: all 8 GPUs at **0% utilization, 808 MiB per process** (bare cuBLAS/CUDA context, no model loaded).
- Python processes alive (PIDs 604173, 604195, 604203, 604231, 604232, 604258, 604275, 604292) but blocked.
- No error lines in log. No OOM. No NCCL timeout yet.
- SLURM `JobState=RUNNING Reason=None`.

**Comparison to 84541 (same init path, healthy run):**

| Checkpoint | 84541 | 84544 |
|---|---|---|
| +1:30 | first NCCL warnings | first NCCL warnings |
| +2:00 – +18:00 | continuous writes (barriers, optimizer config, NCCL deprecation spam) | **silent** |
| +20:04 | iter 1 starts | still silent |
| +25:24 | already on iter 11 | **no iter 1** |

**Interpretation:** all ranks blocked in an early distributed init collective (likely `init_process_group` barrier or first `all_gather` of model state). Ranks are holding CUDA contexts but not progressing. Not CPU-bound — no Python progress.

**Next step pending user confirmation:** `sudo scancel 84544` + resubmit. If the re-submit hangs identically, it's reproducible and needs infra escalation (suspect a different node in this allocation may have a degraded IB link similar to 204).

**Full log:** `…_84544_0.out` (6660 lines before teardown; 1054 lines of stack traces written at 17:01:43 when SLURM hit TimeLimit).

### UPDATE after teardown (17:01:43): root-cause evidence

When SLURM killed 84544 at its 01:30:00 time limit, the NCCL heartbeat monitor flushed stack traces from all 512 ranks. The first error in the dump is:

```
[rank508]: torch.distributed.DistBackendError: NCCL error … ncclRemoteError
[rank508]: socketPollConnect: connect to 7.247.226.174<42307> returned Connection timed out,
           exceeded error retry count after 35 attempts
```

**IP resolution:** `7.247.226.174` → `use3a-ss-b200-gpu-202-s1.storage` (node 202's **storage-fabric NIC**, not compute IB).

**Rank mapping:** rank 508 = local rank 4 on node index 63 of allocation = `use3a-ss-b200-gpu-230`. I.e. node 230 was the one *trying* to connect, and node 202's storage IP was the unreachable peer.

**Revised root cause for Issue 2 (supersedes initial diagnosis):**

Node 230's Sleeping python state observed live during the hang was a *symptom*, not the cause — its ranks were blocked in TCP bootstrap retries to `7.247.226.174:42307`. The fundamental fault is that **NCCL bootstrap selected the storage fabric (7.247.226.0/24 → `*-s1.storage`) for at least some peer-to-peer connections, and that fabric is not end-to-end routable between all compute nodes.** After 35 retries (~82 min) the heartbeat monitor fired.

This is not a single-node hardware fault — it's a **NCCL socket-interface selection / network fabric topology issue** affecting this allocation. The 84541 failure in contrast *was* a genuine HCA retry-exceeded on mlx5_15 over compute fabric (7.247.232.0/24).

**Cascade on teardown:**
- Rank 388 (node 174, local rank 4): `Failed to check "should dump" flag on TCPStore … Broken pipe` → TCPStore server on `use3a-ss-b200-gpu-144:19544` shut down
- All 512 ranks wrote full stack traces before exiting

**sacct summary:**
```
84544       FAILED   15:0   01:21:43  2026-04-21T17:01:47
84544.batch FAILED   15:0   01:21:43
84544.1     CANCELLED 0:15  01:21:42
```

**Revised ask for infra:**
- Verify whether `7.247.226.0/24` (storage fabric) is supposed to be reachable between compute nodes. If not, NCCL bootstrap should be pinned to the compute IB fabric only (e.g. `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` tightening).
- Specifically check connectivity from `use3a-ss-b200-gpu-230` to `7.247.226.174` (`use3a-ss-b200-gpu-202-s1.storage`) — was this port down at 15:42 PDT 2026-04-21, or is it a never-routable path?
- If the submit script is setting `NCCL_SOCKET_IFNAME` to a permissive pattern (e.g. `^lo,docker`), tightening it to the compute fabric interface will prevent future recurrences.

### Node-level diagnosis (performed ~16:40, 1h into the hang)

Ran `srun --overlap --jobid=84544 -N 64` across all allocated nodes while the job was wedged:

**Symptom across 63 nodes (HEALTHY / spinning):**
- Python process `State: R (running)`
- Main thread @ **~98% CPU** (classic NCCL busy-wait on collective)
- All 8 python workers in stat `Rl`

**Symptom on `use3a-ss-b200-gpu-230` (THE ONE OUTLIER):**
- Python process `State: S (sleeping)`
- Main thread @ **~2% CPU** (not participating)
- All 8 python workers in stat `Sl`
- Main thread kernel stack: `do_poll.constprop.0 → do_sys_poll → __x64_sys_poll`
- Worker threads stuck in `futex(FUTEX_WAIT_BITSET_PRIVATE)` and blocking `read()` on many fds
- 8 GPUs all 0% util, 831 MiB each (bare CUDA context — model never loaded)

Every other node was spin-waiting for 230, which never entered the collective. **Node 230 is the bad actor for the 84544 hang.**

**What is NOT wrong with 230 (ruled out):**
- IB link rates identical to healthy node 226 (all 8 compute ports at 400 Gb/sec 4X NDR, storage/mgmt ports at expected 100/200 Gb/sec)
- IB port state `ACTIVE` on all 16 `mlx5_*` devices
- `NVRM: gpuValidateRegOffset_IMPL` dmesg spam is present (640 entries) but also present on ≥14 other allocation nodes at similar counts (768 each) — **not** unique to 230 and not causally linked to the hang

**What remains suspect on 230:**
- Something in the very early container/CUDA/distributed-init path is blocking on I/O before NCCL setup completes
- The `poll()` + many pending `read()` fds pattern (no fds pointing to `/dev/infiniband`, all 8 GPU handles open to `/dev/nvidia7`) is consistent with a stuck IPC/pipe wait or a stalled NCCL bootstrap TCP socket
- Possible candidates: degraded front-end NIC, NCCL bootstrap socket never completing, kernel/driver resource exhaustion, pyxis/enroot stall on this specific node

**Actions for infra:**
- Drain / health-check `use3a-ss-b200-gpu-230`
- Specifically check: front-end ethernet, NCCL bootstrap connectivity, any hung enroot/pyxis child procs, and `/var/log/syslog` around 15:40-15:42 UTC on 2026-04-21
- This is the **second** bad node in ~2 hours on the same physical pool (after 204's `mlx5_15` HCA retry-exceeded error) — suggests the pool itself needs an audit

**Updated exclude list for next resubmit:** `use3a-ss-b200-gpu-[204,230]` (plus yesterday's persistent bad-node list `[130,190,197,199,201,211,228,233,239]`).

---

## Environment gotchas encountered (for future re-submits)

When resubmitting a root-owned `sbatch` script from a johnson shell via `sudo`, the following env must be set **explicitly** or the job fails:

| Var / flag | Why | Failure mode if missing |
|---|---|---|
| `HF_HOME=/mnt/vast/exemplar/llmb/.cache/huggingface` | `HF_HUB_OFFLINE=1` is set in the sbatch, but without `HF_HOME` HF looks at container `~/.cache/...` which isn't mounted | Job 84542: `ValueError: Failed to load configuration from deepseek-ai/DeepSeek-V3 … couldn't find them in the cached files` |
| `HOME=/tmp` | Pyxis / slurmstepd needs a writable HOME; `/root` is the sudo default but not mounted in container | Memory note + slurmstepd errors |
| `sbatch -D <workdir>` | Sudo's cwd is `/home/johnson` which doesn't exist on compute nodes | Job 84543: `couldn't chdir to /home/johnson`, then `NCCL … Call to stat failed: No such file or directory` across all ranks |

Working resubmit recipe:
```bash
sudo env HF_HOME=/mnt/vast/exemplar/llmb/.cache/huggingface HOME=/tmp \
  sbatch -D /mnt/vast/exemplar/llmb/workloads/pretrain_deepseek-v3/Megatron-Bridge \
    --exclude=<bad_nodes> \
    <generated_sbatch_script>
```

---

## What to report

**To infra / ops Slack:**
- Issue 1 (84541): IB HCA error on `use3a-ss-b200-gpu-204` / peers in 7.247.232.168–175 range; request health check on mlx5_15.
- Issue 2 (84544): NCCL bootstrap hang. Live diagnosis showed `use3a-ss-b200-gpu-230` blocked in `poll()` while 63 others spin-waited at 98% CPU. Post-teardown evidence (ranks 508 on node 230 and 388 on node 174) shows the blocked syscall was `socketPollConnect` to `7.247.226.174` = `use3a-ss-b200-gpu-202-s1.storage`. **NCCL bootstrap was attempting to reach node 202 over the storage fabric**, timing out after 35 retries (~82 min). Needs: storage-fabric reachability check between nodes 230↔202, and tightening of `NCCL_SOCKET_IFNAME`/`NCCL_IB_HCA` env in the submit script to pin bootstrap to the compute IB fabric.

**Cluster-level pattern:** two independent failures on consecutive 64-node DSv3 runs within ~2 hours on the same physical pool, each implicating a different node (204 → IB HCA retry-exceeded; 230 → silent init hang). Suggests a pool-wide reliability issue rather than a workload bug — recommend an IB fabric + host audit across the full allocation.
