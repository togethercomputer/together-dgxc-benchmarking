#!/usr/bin/env python3
"""Analyze PyTorch profiler trace for bottlenecks."""

import gzip
import json
import sys
from collections import defaultdict

TRACE_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/johnson/johnson/use3a-ss-b200-gpu-152.cloud.together.ai_rank0.pt.trace.json.gz"

print(f"Loading trace: {TRACE_PATH}")
with gzip.open(TRACE_PATH, "rt") as f:
    trace = json.load(f)

events = trace.get("traceEvents", [])
print(f"Total events: {len(events)}\n")

# ---- Categorize events ----
cpu_ops = []        # CPU-side operators
gpu_kernels = []    # GPU kernels
cuda_runtime = []   # CUDA runtime calls (memcpy, sync, etc.)
comm_ops = []       # NCCL / communication
memory_events = []

for ev in events:
    cat = ev.get("cat", "")
    name = ev.get("name", "")
    dur = ev.get("dur", 0)  # microseconds
    ph = ev.get("ph", "")

    if ph != "X":  # only duration events
        continue

    if "nccl" in name.lower() or "allreduce" in name.lower() or "allgather" in name.lower() \
       or "reduce_scatter" in name.lower() or "all_to_all" in name.lower() \
       or "broadcast" in name.lower() or "send" in name.lower() or "recv" in name.lower():
        comm_ops.append(ev)
    elif cat == "kernel" or "kernel" in cat:
        gpu_kernels.append(ev)
    elif cat == "cuda_runtime" or "runtime" in cat:
        cuda_runtime.append(ev)
    elif cat == "cpu_op" or cat == "user_annotation" or "operator" in cat or "python_function" in cat:
        cpu_ops.append(ev)

print(f"CPU ops:        {len(cpu_ops)}")
print(f"GPU kernels:    {len(gpu_kernels)}")
print(f"CUDA runtime:   {len(cuda_runtime)}")
print(f"Comm ops (NCCL):{len(comm_ops)}")
print()

# ---- Helper ----
def top_by_duration(events_list, n=20, label=""):
    """Aggregate by name, sort by total duration."""
    agg = defaultdict(lambda: {"count": 0, "total_us": 0, "max_us": 0})
    for ev in events_list:
        name = ev.get("name", "unknown")
        dur = ev.get("dur", 0)
        agg[name]["count"] += 1
        agg[name]["total_us"] += dur
        agg[name]["max_us"] = max(agg[name]["max_us"], dur)

    sorted_ops = sorted(agg.items(), key=lambda x: x[1]["total_us"], reverse=True)
    total_all = sum(v["total_us"] for _, v in sorted_ops)

    print(f"{'='*100}")
    print(f" {label} — Top {n} by total time (total: {total_all/1e6:.2f}s)")
    print(f"{'='*100}")
    print(f"{'Rank':>4} {'Name':<60} {'Total(ms)':>10} {'Count':>8} {'Avg(ms)':>10} {'Max(ms)':>10} {'%':>6}")
    print(f"{'-'*100}")

    for i, (name, stats) in enumerate(sorted_ops[:n]):
        total_ms = stats["total_us"] / 1000
        avg_ms = total_ms / stats["count"] if stats["count"] else 0
        max_ms = stats["max_us"] / 1000
        pct = (stats["total_us"] / total_all * 100) if total_all else 0
        disp_name = name[:58] if len(name) > 58 else name
        print(f"{i+1:>4} {disp_name:<60} {total_ms:>10.1f} {stats['count']:>8} {avg_ms:>10.2f} {max_ms:>10.1f} {pct:>5.1f}%")
    print()


# ---- Analysis ----

# 1. Top GPU kernels
top_by_duration(gpu_kernels, n=25, label="GPU KERNELS")

# 2. Top NCCL / communication
top_by_duration(comm_ops, n=15, label="COMMUNICATION (NCCL)")

# 3. Top CPU ops
top_by_duration(cpu_ops, n=20, label="CPU OPERATORS")

# 4. Top CUDA runtime calls
top_by_duration(cuda_runtime, n=10, label="CUDA RUNTIME")

# ---- Time breakdown summary ----
gpu_total = sum(ev.get("dur", 0) for ev in gpu_kernels)
comm_total = sum(ev.get("dur", 0) for ev in comm_ops)
cpu_total = sum(ev.get("dur", 0) for ev in cpu_ops)
runtime_total = sum(ev.get("dur", 0) for ev in cuda_runtime)

# Separate compute vs memory kernels
compute_kernels_us = 0
memcpy_kernels_us = 0
for ev in gpu_kernels:
    name = ev.get("name", "")
    dur = ev.get("dur", 0)
    if "memcpy" in name.lower() or "memset" in name.lower():
        memcpy_kernels_us += dur
    else:
        compute_kernels_us += dur

# Separate NCCL comm types
nccl_breakdown = defaultdict(int)
for ev in comm_ops:
    name = ev.get("name", "").lower()
    dur = ev.get("dur", 0)
    if "allreduce" in name:
        nccl_breakdown["AllReduce"] += dur
    elif "allgather" in name or "all_gather" in name:
        nccl_breakdown["AllGather"] += dur
    elif "reduce_scatter" in name:
        nccl_breakdown["ReduceScatter"] += dur
    elif "send" in name:
        nccl_breakdown["Send (P2P)"] += dur
    elif "recv" in name:
        nccl_breakdown["Recv (P2P)"] += dur
    elif "broadcast" in name:
        nccl_breakdown["Broadcast"] += dur
    else:
        nccl_breakdown["Other"] += dur

print(f"{'='*100}")
print(f" HIGH-LEVEL TIME BREAKDOWN")
print(f"{'='*100}")
print(f"  GPU Compute Kernels:  {compute_kernels_us/1e6:>8.2f}s")
print(f"  GPU Memcpy/Memset:    {memcpy_kernels_us/1e6:>8.2f}s")
print(f"  Communication (NCCL): {comm_total/1e6:>8.2f}s")
print(f"  CPU Operators:        {cpu_total/1e6:>8.2f}s")
print(f"  CUDA Runtime:         {runtime_total/1e6:>8.2f}s")
print()

if nccl_breakdown:
    print(f"  NCCL Breakdown:")
    for op, us in sorted(nccl_breakdown.items(), key=lambda x: -x[1]):
        print(f"    {op:<20}: {us/1e6:>8.2f}s ({us/comm_total*100:.1f}%)" if comm_total else "")
print()

# ---- Look for gaps / idle time ----
# Find long gaps between GPU kernels (potential bubble / stall)
if gpu_kernels:
    sorted_kernels = sorted(gpu_kernels, key=lambda e: e.get("ts", 0))
    gaps = []
    for i in range(1, len(sorted_kernels)):
        prev_end = sorted_kernels[i-1].get("ts", 0) + sorted_kernels[i-1].get("dur", 0)
        curr_start = sorted_kernels[i].get("ts", 0)
        gap = curr_start - prev_end
        if gap > 1000:  # > 1ms gaps
            gaps.append({
                "gap_us": gap,
                "after": sorted_kernels[i-1].get("name", ""),
                "before": sorted_kernels[i].get("name", ""),
                "ts": prev_end,
            })

    gaps.sort(key=lambda x: -x["gap_us"])
    print(f"{'='*100}")
    print(f" TOP GPU IDLE GAPS (>1ms) — {len(gaps)} gaps found")
    print(f"{'='*100}")
    total_gap = sum(g["gap_us"] for g in gaps)
    print(f"  Total GPU idle time in gaps: {total_gap/1e6:.2f}s")
    print()
    for i, g in enumerate(gaps[:15]):
        print(f"  {i+1:>3}. {g['gap_us']/1000:>8.1f}ms gap  |  after: {g['after'][:40]}  →  before: {g['before'][:40]}")
    print()

print("Analysis complete.")
