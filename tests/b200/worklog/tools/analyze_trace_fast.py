#!/usr/bin/env python3
"""Fast line-by-line PyTorch trace analyzer — avoids full JSON parse of multi-GB files.

Chrome trace format has one event per line in traceEvents array.
We parse each line individually, which is orders of magnitude faster than json.loads on 8GB.
"""

import gzip
import json
import sys
import time
from collections import defaultdict

TRACE_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/johnson/johnson/traces/pytorch/qwen3_235b_bf16_baseline_rank0.pt.trace.json.gz"

print(f"Loading trace: {TRACE_PATH}")
t0 = time.time()

# ---- Categorization state ----
cpu_ops = []
gpu_kernels = []
cuda_runtime = []
comm_ops = []
moe_ops = []
all_events_count = 0
parse_errors = 0

NCCL_KEYWORDS = {"nccl", "allreduce", "allgather", "all_gather", "reduce_scatter",
                 "reducescatter", "all_to_all", "alltoall", "broadcast",
                 "ncclkernel", "nccldevkernel"}
# Send/recv are checked separately to avoid false positives
PP_KEYWORDS = {"isend", "irecv"}

MOE_KEYWORDS = {"expert", "moe", "topk", "routing", "permut", "unpermut",
                "token_dispatch", "token_permut", "grouped_gemm", "groupedgemm"}


def categorize_event(ev):
    """Categorize a single trace event."""
    global all_events_count
    all_events_count += 1

    cat = ev.get("cat", "")
    name = ev.get("name", "")
    dur = ev.get("dur", 0)
    ph = ev.get("ph", "")

    if ph != "X":
        return

    name_lower = name.lower()

    # Check NCCL/communication
    is_comm = False
    for kw in NCCL_KEYWORDS:
        if kw in name_lower:
            is_comm = True
            break
    if not is_comm:
        for kw in PP_KEYWORDS:
            if kw in name_lower:
                is_comm = True
                break
    # Also catch "send" and "recv" but only if it looks like P2P
    if not is_comm and cat == "kernel":
        if ("send" in name_lower or "recv" in name_lower) and "nccl" in name_lower:
            is_comm = True

    # Check MoE
    is_moe = False
    for kw in MOE_KEYWORDS:
        if kw in name_lower:
            is_moe = True
            break

    # Store compactly — only fields we need
    compact = {"name": name, "dur": dur, "ts": ev.get("ts", 0), "cat": cat}

    if is_comm:
        comm_ops.append(compact)
    elif cat == "kernel" or "kernel" in cat:
        gpu_kernels.append(compact)
    elif cat == "cuda_runtime" or "runtime" in cat:
        cuda_runtime.append(compact)
    elif cat in ("cpu_op", "user_annotation", "python_function") or "operator" in cat:
        cpu_ops.append(compact)

    if is_moe:
        moe_ops.append(compact)


# ---- Stream parse ----
print("Streaming line-by-line parse...")
in_events = False
line_count = 0

with gzip.open(TRACE_PATH, "rt", encoding="utf-8", errors="replace") as f:
    # State machine: accumulate lines for multi-line events
    obj_buf = []
    brace_depth = 0

    while True:
        try:
            line = f.readline()
        except EOFError:
            print(f"\n  Hit truncation at line {line_count} — continuing with partial data")
            break
        if not line:
            break

        line_count += 1
        stripped = line.strip()

        # Detect start of traceEvents array
        if not in_events:
            if '"traceEvents"' in stripped:
                in_events = True
            continue

        # Track brace depth to accumulate complete JSON objects
        for ch in stripped:
            if ch == '{':
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1

        if brace_depth > 0:
            obj_buf.append(stripped)
        elif brace_depth == 0 and obj_buf:
            obj_buf.append(stripped)
            obj_str = " ".join(obj_buf)
            obj_buf = []

            # Remove trailing comma
            if obj_str.endswith(","):
                obj_str = obj_str[:-1]

            try:
                ev = json.loads(obj_str)
                if isinstance(ev, dict):
                    categorize_event(ev)
            except json.JSONDecodeError:
                parse_errors += 1
        elif brace_depth == 0 and stripped.startswith("{"):
            # Single-line event
            if stripped.endswith(","):
                stripped = stripped[:-1]
            try:
                ev = json.loads(stripped)
                if isinstance(ev, dict):
                    categorize_event(ev)
            except json.JSONDecodeError:
                parse_errors += 1

        if line_count % 5000000 == 0:
            elapsed = time.time() - t0
            print(f"  {line_count/1e6:.1f}M lines, {all_events_count} events, "
                  f"{len(gpu_kernels)} kernels, {len(comm_ops)} comm, "
                  f"{elapsed:.0f}s elapsed", flush=True)

elapsed = time.time() - t0
print(f"\nParsed {line_count} lines in {elapsed:.1f}s")
print(f"Total events: {all_events_count} (parse errors: {parse_errors})")
print(f"CPU ops:         {len(cpu_ops)}")
print(f"GPU kernels:     {len(gpu_kernels)}")
print(f"CUDA runtime:    {len(cuda_runtime)}")
print(f"Comm ops (NCCL): {len(comm_ops)}")
print(f"MoE ops:         {len(moe_ops)}")
print()


# ---- Helper ----
def top_by_duration(events_list, n=20, label=""):
    agg = defaultdict(lambda: {"count": 0, "total_us": 0, "max_us": 0})
    for ev in events_list:
        name = ev["name"]
        dur = ev["dur"]
        agg[name]["count"] += 1
        agg[name]["total_us"] += dur
        agg[name]["max_us"] = max(agg[name]["max_us"], dur)

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
top_by_duration(cpu_ops, n=25, label="CPU OPERATORS")

# 4. Top CUDA runtime calls
top_by_duration(cuda_runtime, n=10, label="CUDA RUNTIME")

# 5. MoE ops
if moe_ops:
    top_by_duration(moe_ops, n=20, label="MoE-RELATED OPERATIONS")

# ---- Compute vs Communication Breakdown ----
pure_compute_us = 0
memcpy_kernels_us = 0
nccl_kernels_us = 0

for ev in gpu_kernels:
    name_lower = ev["name"].lower()
    dur = ev["dur"]
    if "memcpy" in name_lower or "memset" in name_lower:
        memcpy_kernels_us += dur
    elif "nccl" in name_lower:
        nccl_kernels_us += dur
    else:
        pure_compute_us += dur

# NCCL breakdown by type
nccl_breakdown = defaultdict(lambda: {"total_us": 0, "count": 0})
for ev in comm_ops:
    name_lower = ev["name"].lower()
    dur = ev["dur"]
    if "allreduce" in name_lower:
        key = "AllReduce"
    elif "allgather" in name_lower or "all_gather" in name_lower:
        key = "AllGather"
    elif "reduce_scatter" in name_lower or "reducescatter" in name_lower:
        key = "ReduceScatter"
    elif "all_to_all" in name_lower or "alltoall" in name_lower:
        key = "AllToAll (MoE)"
    elif "send" in name_lower and ("nccl" in name_lower or "isend" in name_lower):
        key = "Send (P2P/PP)"
    elif "recv" in name_lower and ("nccl" in name_lower or "irecv" in name_lower):
        key = "Recv (P2P/PP)"
    elif "broadcast" in name_lower:
        key = "Broadcast"
    else:
        key = "Other"
    nccl_breakdown[key]["total_us"] += dur
    nccl_breakdown[key]["count"] += 1

comm_total = sum(v["total_us"] for v in nccl_breakdown.values())
total_gpu = pure_compute_us + memcpy_kernels_us + nccl_kernels_us

print(f"{'='*120}")
print(f" HIGH-LEVEL TIME BREAKDOWN")
print(f"{'='*120}")
if total_gpu:
    print(f"  GPU Compute Kernels:  {pure_compute_us/1e6:>10.2f}s  ({pure_compute_us/total_gpu*100:.1f}% of GPU)")
    print(f"  GPU NCCL Kernels:     {nccl_kernels_us/1e6:>10.2f}s  ({nccl_kernels_us/total_gpu*100:.1f}% of GPU)")
    print(f"  GPU Memcpy/Memset:    {memcpy_kernels_us/1e6:>10.2f}s  ({memcpy_kernels_us/total_gpu*100:.1f}% of GPU)")
    print(f"  ────────────────────────────────")
    print(f"  Total GPU time:       {total_gpu/1e6:>10.2f}s")
print()
if comm_total:
    print(f"  Comm ops (CPU-side):  {comm_total/1e6:>10.2f}s")
    print()
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
    sorted_kernels = sorted(gpu_kernels, key=lambda e: e["ts"])
    gaps = []
    total_span = 0
    if len(sorted_kernels) > 1:
        first_ts = sorted_kernels[0]["ts"]
        last_end = sorted_kernels[-1]["ts"] + sorted_kernels[-1]["dur"]
        total_span = last_end - first_ts

    for i in range(1, len(sorted_kernels)):
        prev_end = sorted_kernels[i-1]["ts"] + sorted_kernels[i-1]["dur"]
        curr_start = sorted_kernels[i]["ts"]
        gap = curr_start - prev_end
        if gap > 1000:  # > 1ms
            gaps.append({
                "gap_us": gap,
                "after": sorted_kernels[i-1]["name"],
                "before": sorted_kernels[i]["name"],
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
    for i, g in enumerate(gaps[:25]):
        after = g["after"][:43] if len(g["after"]) > 43 else g["after"]
        before = g["before"][:43] if len(g["before"]) > 43 else g["before"]
        print(f"  {i+1:>4} {g['gap_us']/1000:>10.1f}  {after:<45} {before:<45}")
    print()

    # Gap distribution
    buckets = {"1-10ms": [0, 0], "10-100ms": [0, 0], "100ms-1s": [0, 0], ">1s": [0, 0]}
    for g in gaps:
        us = g["gap_us"]
        if us < 10000:
            k = "1-10ms"
        elif us < 100000:
            k = "10-100ms"
        elif us < 1000000:
            k = "100ms-1s"
        else:
            k = ">1s"
        buckets[k][0] += 1
        buckets[k][1] += us

    print(f"  Gap Distribution:")
    print(f"  {'Bucket':<15} {'Count':>8} {'Total(s)':>10} {'% of idle':>10}")
    print(f"  {'-'*45}")
    for bucket in ["1-10ms", "10-100ms", "100ms-1s", ">1s"]:
        cnt, total_us = buckets[bucket]
        pct = total_us / total_gap * 100 if total_gap else 0
        print(f"  {bucket:<15} {cnt:>8} {total_us/1e6:>10.2f} {pct:>9.1f}%")
    print()

# ---- aten::item / sync stall detection ----
print(f"{'='*120}")
print(f" SYNC STALL DETECTION")
print(f"{'='*120}")
sync_keywords = ["aten::item", "aten::_local_scalar_dense", "cudaDeviceSynchronize",
                 "cudaStreamSynchronize"]
for kw in sync_keywords:
    matches = [ev for ev in cpu_ops + cuda_runtime if kw in ev["name"]]
    if matches:
        total_us = sum(e["dur"] for e in matches)
        print(f"  {kw}: {len(matches)} calls, total {total_us/1e6:.2f}s, avg {total_us/len(matches)/1000:.2f}ms")

print(f"\nTotal analysis time: {time.time()-t0:.1f}s")
print("Analysis complete.")
