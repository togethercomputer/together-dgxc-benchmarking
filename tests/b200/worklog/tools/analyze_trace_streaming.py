#!/usr/bin/env python3
"""Analyze PyTorch profiler trace — streaming parser for large/truncated gzip files."""

import gzip
import json
import sys
from collections import defaultdict

TRACE_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/johnson/johnson/traces/pytorch/qwen3_235b_bf16_baseline_rank0.pt.trace.json.gz"

print(f"Loading trace: {TRACE_PATH}")
print("Reading (truncated gzip safe)...")

# Read as much as possible from potentially truncated gzip
raw = b""
with gzip.open(TRACE_PATH, "rb") as f:
    while True:
        try:
            chunk = f.read(64 * 1024 * 1024)  # 64MB
            if not chunk:
                break
            raw += chunk
            print(f"  read {len(raw)/1e9:.2f} GB ...", end="\r", flush=True)
        except EOFError:
            print(f"\n  Hit truncation at {len(raw)/1e9:.2f} GB — continuing with partial data")
            break

print(f"Total decompressed: {len(raw)/1e9:.2f} GB")
text = raw.decode("utf-8", errors="replace")
del raw  # free memory

# Try to parse as JSON; if truncated, fix the JSON
try:
    trace = json.loads(text)
except json.JSONDecodeError:
    # Truncated — find last complete event (ends with "}")
    # Chrome trace format: {"traceEvents": [...], ...}
    # Try to close the array and object
    # Find last complete }, then close brackets
    last_brace = text.rfind("}")
    if last_brace > 0:
        # Try different truncation repair strategies
        attempt = text[:last_brace + 1] + "]}"
        try:
            trace = json.loads(attempt)
            print(f"  Repaired truncated JSON (closed array at byte {last_brace})")
        except json.JSONDecodeError:
            # Might have trailing comma or partial event — back up further
            # Find last complete event delimiter
            last_complete = text.rfind("},", 0, last_brace)
            if last_complete > 0:
                attempt = text[:last_complete + 1] + "]}"
                try:
                    trace = json.loads(attempt)
                    print(f"  Repaired truncated JSON (backed up to byte {last_complete})")
                except json.JSONDecodeError as e2:
                    print(f"  Cannot repair JSON: {e2}")
                    sys.exit(1)
            else:
                print("  Cannot find valid JSON repair point")
                sys.exit(1)
    else:
        print("  No valid JSON structure found")
        sys.exit(1)

del text  # free memory

events = trace.get("traceEvents", [])
print(f"Total events parsed: {len(events)}\n")

# ---- Categorize events ----
cpu_ops = []
gpu_kernels = []
cuda_runtime = []
comm_ops = []

# NCCL keywords for communication detection
NCCL_KEYWORDS = {"nccl", "allreduce", "allgather", "all_gather", "reduce_scatter",
                 "reducescatter", "all_to_all", "alltoall", "broadcast", "send", "recv",
                 "isend", "irecv", "ncclkernel", "nccldevkernel"}

for ev in events:
    cat = ev.get("cat", "")
    name = ev.get("name", "")
    dur = ev.get("dur", 0)
    ph = ev.get("ph", "")

    if ph != "X":  # only duration events
        continue

    name_lower = name.lower()

    # Check for NCCL/communication
    is_comm = False
    for kw in NCCL_KEYWORDS:
        if kw in name_lower:
            is_comm = True
            break

    if is_comm:
        comm_ops.append(ev)
    elif cat == "kernel" or "kernel" in cat:
        gpu_kernels.append(ev)
    elif cat == "cuda_runtime" or "runtime" in cat:
        cuda_runtime.append(ev)
    elif cat in ("cpu_op", "user_annotation", "python_function") or "operator" in cat:
        cpu_ops.append(ev)

print(f"CPU ops:         {len(cpu_ops)}")
print(f"GPU kernels:     {len(gpu_kernels)}")
print(f"CUDA runtime:    {len(cuda_runtime)}")
print(f"Comm ops (NCCL): {len(comm_ops)}")
print()

# ---- Helper ----
def top_by_duration(events_list, n=20, label=""):
    agg = defaultdict(lambda: {"count": 0, "total_us": 0, "max_us": 0, "min_us": float("inf")})
    for ev in events_list:
        name = ev.get("name", "unknown")
        dur = ev.get("dur", 0)
        agg[name]["count"] += 1
        agg[name]["total_us"] += dur
        agg[name]["max_us"] = max(agg[name]["max_us"], dur)
        agg[name]["min_us"] = min(agg[name]["min_us"], dur)

    sorted_ops = sorted(agg.items(), key=lambda x: x[1]["total_us"], reverse=True)
    total_all = sum(v["total_us"] for _, v in sorted_ops)

    print(f"{'='*120}")
    print(f" {label} — Top {n} by total time (total: {total_all/1e6:.2f}s)")
    print(f"{'='*120}")
    print(f"{'Rank':>4} {'Name':<65} {'Total(ms)':>10} {'Count':>8} {'Avg(ms)':>10} {'Max(ms)':>10} {'%':>6}")
    print(f"{'-'*120}")

    for i, (name, stats) in enumerate(sorted_ops[:n]):
        total_ms = stats["total_us"] / 1000
        avg_ms = total_ms / stats["count"] if stats["count"] else 0
        max_ms = stats["max_us"] / 1000
        pct = (stats["total_us"] / total_all * 100) if total_all else 0
        disp = name[:63] if len(name) > 63 else name
        print(f"{i+1:>4} {disp:<65} {total_ms:>10.1f} {stats['count']:>8} {avg_ms:>10.2f} {max_ms:>10.1f} {pct:>5.1f}%")
    print()
    return sorted_ops, total_all


# ---- Analysis ----

# 1. Top GPU kernels
gpu_sorted, gpu_total_us = top_by_duration(gpu_kernels, n=30, label="GPU KERNELS")

# 2. Top NCCL / communication
comm_sorted, comm_total_us = top_by_duration(comm_ops, n=20, label="COMMUNICATION (NCCL)")

# 3. Top CPU ops
top_by_duration(cpu_ops, n=20, label="CPU OPERATORS")

# 4. Top CUDA runtime calls
top_by_duration(cuda_runtime, n=10, label="CUDA RUNTIME")

# ---- Compute vs Communication Breakdown ----
compute_kernels_us = 0
memcpy_kernels_us = 0
nccl_kernels_us = 0  # NCCL kernels that show up in GPU kernel category
pure_compute_us = 0

for ev in gpu_kernels:
    name = ev.get("name", "").lower()
    dur = ev.get("dur", 0)
    if "memcpy" in name or "memset" in name:
        memcpy_kernels_us += dur
    elif "nccl" in name:
        nccl_kernels_us += dur
    else:
        pure_compute_us += dur

# NCCL comm breakdown by type
nccl_breakdown = defaultdict(lambda: {"total_us": 0, "count": 0})
for ev in comm_ops:
    name = ev.get("name", "").lower()
    dur = ev.get("dur", 0)
    if "allreduce" in name:
        key = "AllReduce"
    elif "allgather" in name or "all_gather" in name:
        key = "AllGather"
    elif "reduce_scatter" in name or "reducescatter" in name:
        key = "ReduceScatter"
    elif "all_to_all" in name or "alltoall" in name:
        key = "AllToAll (MoE)"
    elif "send" in name:
        key = "Send (P2P/PP)"
    elif "recv" in name:
        key = "Recv (P2P/PP)"
    elif "broadcast" in name:
        key = "Broadcast"
    else:
        key = "Other"
    nccl_breakdown[key]["total_us"] += dur
    nccl_breakdown[key]["count"] += 1

comm_total = sum(v["total_us"] for v in nccl_breakdown.values())

print(f"{'='*120}")
print(f" HIGH-LEVEL TIME BREAKDOWN")
print(f"{'='*120}")
total_gpu = pure_compute_us + memcpy_kernels_us + nccl_kernels_us
print(f"  GPU Compute Kernels:  {pure_compute_us/1e6:>10.2f}s  ({pure_compute_us/total_gpu*100:.1f}% of GPU)" if total_gpu else "")
print(f"  GPU NCCL Kernels:     {nccl_kernels_us/1e6:>10.2f}s  ({nccl_kernels_us/total_gpu*100:.1f}% of GPU)" if total_gpu else "")
print(f"  GPU Memcpy/Memset:    {memcpy_kernels_us/1e6:>10.2f}s  ({memcpy_kernels_us/total_gpu*100:.1f}% of GPU)" if total_gpu else "")
print(f"  ────────────────────────────────")
print(f"  Total GPU time:       {total_gpu/1e6:>10.2f}s")
print()
print(f"  Comm ops (CPU-side):  {comm_total/1e6:>10.2f}s")
print()

if nccl_breakdown:
    print(f"  NCCL Communication Breakdown:")
    print(f"  {'Type':<25} {'Total(s)':>10} {'Count':>8} {'Avg(ms)':>10} {'% of Comm':>10}")
    print(f"  {'-'*65}")
    for op, stats in sorted(nccl_breakdown.items(), key=lambda x: -x[1]["total_us"]):
        us = stats["total_us"]
        cnt = stats["count"]
        avg_ms = us / cnt / 1000 if cnt else 0
        pct = us / comm_total * 100 if comm_total else 0
        print(f"  {op:<25} {us/1e6:>10.2f} {cnt:>8} {avg_ms:>10.2f} {pct:>9.1f}%")
print()

# ---- GPU Idle Gaps (bubble/stall detection) ----
if gpu_kernels:
    sorted_kernels = sorted(gpu_kernels, key=lambda e: e.get("ts", 0))
    gaps = []
    total_span = 0
    if len(sorted_kernels) > 1:
        first_ts = sorted_kernels[0].get("ts", 0)
        last_end = sorted_kernels[-1].get("ts", 0) + sorted_kernels[-1].get("dur", 0)
        total_span = last_end - first_ts

    for i in range(1, len(sorted_kernels)):
        prev_end = sorted_kernels[i-1].get("ts", 0) + sorted_kernels[i-1].get("dur", 0)
        curr_start = sorted_kernels[i].get("ts", 0)
        gap = curr_start - prev_end
        if gap > 1000:  # > 1ms
            gaps.append({
                "gap_us": gap,
                "after": sorted_kernels[i-1].get("name", ""),
                "before": sorted_kernels[i].get("name", ""),
                "ts": prev_end,
            })

    gaps.sort(key=lambda x: -x["gap_us"])
    total_gap = sum(g["gap_us"] for g in gaps)

    print(f"{'='*120}")
    print(f" GPU IDLE GAPS (>1ms) — {len(gaps)} gaps found")
    print(f"{'='*120}")
    print(f"  Total GPU idle (gaps >1ms): {total_gap/1e6:.2f}s")
    if total_span > 0:
        print(f"  Total GPU span:             {total_span/1e6:.2f}s")
        print(f"  Bubble fraction:            {total_gap/total_span*100:.1f}%")
    print()
    print(f"  {'Rank':>4} {'Gap(ms)':>10}  {'After':<45} {'Before':<45}")
    print(f"  {'-'*108}")
    for i, g in enumerate(gaps[:20]):
        after = g["after"][:43] if len(g["after"]) > 43 else g["after"]
        before = g["before"][:43] if len(g["before"]) > 43 else g["before"]
        print(f"  {i+1:>4} {g['gap_us']/1000:>10.1f}  {after:<45} {before:<45}")
    print()

    # Distribution of gaps
    gap_buckets = {"1-10ms": 0, "10-100ms": 0, "100ms-1s": 0, ">1s": 0}
    gap_count_buckets = {"1-10ms": 0, "10-100ms": 0, "100ms-1s": 0, ">1s": 0}
    for g in gaps:
        us = g["gap_us"]
        if us < 10000:
            gap_buckets["1-10ms"] += us
            gap_count_buckets["1-10ms"] += 1
        elif us < 100000:
            gap_buckets["10-100ms"] += us
            gap_count_buckets["10-100ms"] += 1
        elif us < 1000000:
            gap_buckets["100ms-1s"] += us
            gap_count_buckets["100ms-1s"] += 1
        else:
            gap_buckets[">1s"] += us
            gap_count_buckets[">1s"] += 1

    print(f"  Gap Distribution:")
    print(f"  {'Bucket':<15} {'Count':>8} {'Total(s)':>10} {'% of idle':>10}")
    print(f"  {'-'*45}")
    for bucket in ["1-10ms", "10-100ms", "100ms-1s", ">1s"]:
        pct = gap_buckets[bucket] / total_gap * 100 if total_gap else 0
        print(f"  {bucket:<15} {gap_count_buckets[bucket]:>8} {gap_buckets[bucket]/1e6:>10.2f} {pct:>9.1f}%")
    print()

# ---- MoE-specific: look for expert-related ops ----
print(f"{'='*120}")
print(f" MoE-SPECIFIC ANALYSIS")
print(f"{'='*120}")

moe_keywords = ["expert", "moe", "topk", "routing", "permut", "unpermut", "all_to_all", "alltoall",
                 "token_dispatch", "token_permut", "grouped_gemm", "groupedgemm"]
moe_ops = []
for ev in events:
    if ev.get("ph") != "X":
        continue
    name_lower = ev.get("name", "").lower()
    for kw in moe_keywords:
        if kw in name_lower:
            moe_ops.append(ev)
            break

if moe_ops:
    top_by_duration(moe_ops, n=20, label="MoE-RELATED OPERATIONS")
else:
    print("  No MoE-specific operations found by keyword search.")
    print("  (Expert routing may be embedded in generic GEMM / NCCL kernels)")
print()

# ---- aten::item / sync stall detection ----
sync_stalls = [ev for ev in events if ev.get("ph") == "X" and "aten::item" in ev.get("name", "")]
if sync_stalls:
    total_stall = sum(e.get("dur", 0) for e in sync_stalls)
    print(f"  aten::item sync stalls: {len(sync_stalls)} calls, total {total_stall/1e6:.2f}s")
else:
    print("  No aten::item sync stalls found.")

print("\nAnalysis complete.")
