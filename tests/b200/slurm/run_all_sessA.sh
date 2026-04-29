#!/bin/bash
#
# Master benchmark script — submit all official NVIDIA baselines
# Container: 26.02 for all models
# Iterations: 10 steps (measure steps 3-9)
#
# Usage:
#   bash run_all_official.sh [--tier1] [--tier2] [--tier3] [--dry-run]
#
# --tier1: only 64-GPU jobs (8 nodes each)
# --tier2: only 256-GPU jobs (32 nodes each)
# --tier3: only 512-GPU jobs (64 nodes each, sequential — uses full cluster)
# --dry-run: show what would be submitted without actually submitting
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLMB_DIR=/mnt/vast/johnson/llmb
DGXC_REPO=/mnt/vast/johnson/dgxc-benchmarking
IMAGE_26=/mnt/vast/johnson/llmb/images/nvidia+nemo+26.02.00.sqsh
MAX_STEPS=10
# Session A: exclude unhealthy nodes (130 missing /opt/hpcx; others drained/down per 2026-04-20 briefing)
EXCLUDE_NODES='use3a-ss-b200-gpu-[130,190,197,199,201,211,233,239]'
SESSION_TAG=sessA

# Parse arguments
RUN_TIER1=false
RUN_TIER2=false
RUN_TIER3=false
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --tier1) RUN_TIER1=true ;;
        --tier2) RUN_TIER2=true ;;
        --tier3) RUN_TIER3=true ;;
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done
# If none specified, run tier1+tier2 (tier3 must be explicit)
if ! $RUN_TIER1 && ! $RUN_TIER2 && ! $RUN_TIER3; then
    RUN_TIER1=true
    RUN_TIER2=true
fi

# Track submitted jobs
declare -A JOBS

##############################################################################
# Helpers
##############################################################################

check_cluster() {
    echo "=== Cluster Status ==="
    local idle_nodes
    idle_nodes=$(sinfo -p batch -t idle -h -o "%D" 2>/dev/null || echo "0")
    echo "Idle batch nodes: ${idle_nodes}"
    echo "Down/drained:"
    sinfo -p batch -t down,drained -h -N -o "  %N %T %E" 2>/dev/null || true
    echo ""
    echo "$idle_nodes"
}

submit_llmb() {
    # Submit via llmb-run, then post-patch max_steps in generated script
    local label="$1"
    local workload="$2"
    local size="$3"
    local dtype="$4"
    local scale="$5"

    echo "--- [$label] ${workload} ${size} ${dtype} ${scale}gpu ---"

    # Build llmb-run command (size may be empty for some models like DeepSeek-V3)
    local cmd="./llmb-run submit -w ${workload}"
    if [[ -n "$size" ]]; then
        cmd+=" -s ${size}"
    fi
    cmd+=" -d ${dtype} --scale ${scale} --exclude ${EXCLUDE_NODES}"

    if $DRY_RUN; then
        echo "  [DRY-RUN] cd ${LLMB_DIR} && ${cmd}"
        return
    fi

    # Find experiment dirs before submission (to detect new one)
    local exp_base="${LLMB_DIR}/workloads/${workload}/experiments"
    local before_count
    before_count=$(find "$exp_base" -maxdepth 2 -mindepth 2 -type d 2>/dev/null | wc -l)

    # Submit
    cd "${LLMB_DIR}"
    local output
    output=$(eval "${cmd}" 2>&1) || true
    echo "$output"

    # Extract job ID from output
    local job_id
    job_id=$(echo "$output" | grep -oP 'jobid=\K[0-9]+' | head -1)

    if [[ -z "$job_id" ]]; then
        echo "  [ERROR] Failed to extract job ID"
        return 1
    fi

    echo "  Job ID: ${job_id}"
    JOBS["$label"]="$job_id"
    scontrol update job="$job_id" JobName="${SESSION_TAG}_${label}" 2>/dev/null || true

    # Post-patch: find the newly created experiment script and change max_steps
    sleep 1  # brief wait for filesystem
    local newest_script
    newest_script=$(find "$exp_base" -name "*.sh" -path "*/scripts/*" -newer /tmp/.llmb_marker_$$ 2>/dev/null | head -1)

    # Fallback: search by job ID in the experiment directory
    if [[ -z "$newest_script" ]]; then
        newest_script=$(find "$exp_base" -name "*.sh" -path "*/scripts/*" -newermt "1 minute ago" 2>/dev/null | head -1)
    fi

    if [[ -n "$newest_script" ]]; then
        if grep -qE "max_steps[= ]50" "$newest_script"; then
            sed -i -E "s/--max_steps[= ]50/--max_steps=${MAX_STEPS}/g" "$newest_script"
            echo "  Patched: max_steps 50 -> max_steps ${MAX_STEPS} in $(basename "$newest_script")"
        else
            echo "  [WARN] max_steps=50 not found in script, may already be patched or different default"
        fi
    else
        echo "  [WARN] Could not find generated script to patch max_steps"
    fi
    cd "${SCRIPT_DIR}"
}

patch_legacy_sbatch() {
    # Patch a NeMo-generated sbatch script for legacy containers (25.07/25.09)
    # on this cluster. Adds workarounds for pyxis HOME, HF tokenizer cache,
    # NCCL interface, PMIx v3/v4 mismatch, and MPI stub.
    #
    # Usage: patch_legacy_sbatch <sbatch_script> <patched_output>
    local src="$1"
    local dst="$2"

    if [[ ! -f "$src" ]]; then
        echo "  [ERROR] patch_legacy_sbatch: source not found: $src"
        return 1
    fi

    cp "$src" "$dst"

    # 1. Insert env vars before the srun command
    sed -i '/^# Command 1/a\
# --- Legacy container workarounds (auto-patched) ---\
export HOME=/tmp\
export NEMO_NLP_TMP=/tmp\
export HF_HOME=/mnt/vast/johnson/llmb/.cache/huggingface\
export NCCL_SOCKET_IFNAME=bond0\
export PMIX_MCA_gds=hash\
export OMPI_MCA_plm=isolated\
export MPI4PY_RC_INITIALIZE=false\
export MPI4PY_RC_THREADS=false\
export MPI4PY_RC_FINALIZE=false\
export LD_PRELOAD=/mpi_stub/libmpi_stub.so\
# --- End workarounds ---' "$dst"

    # 2. Add mpi_stub mount to --container-mounts
    sed -i 's|:/nemo_run |:/nemo_run,/mnt/vast/johnson/llmb/mpi_stub:/mpi_stub |g' "$dst"

    # 3. Append workaround env vars to --container-env
    sed -i 's|--container-env=\(.*\) bash|--container-env=\1,HOME,NEMO_NLP_TMP,HF_HOME,NCCL_SOCKET_IFNAME,PMIX_MCA_gds,OMPI_MCA_plm,MPI4PY_RC_INITIALIZE,MPI4PY_RC_THREADS,MPI4PY_RC_FINALIZE,LD_PRELOAD bash|' "$dst"

    echo "  Patched: $(basename "$dst")"
}

submit_nemotron4_15b() {
    # Submit Nemotron4 15B via dgxc-benchmarking launch.sh with 26.02 container
    local label="$1"
    local dtype="$2"
    local scale="$3"

    echo "--- [$label] nemotron4-15b ${dtype} ${scale}gpu ---"

    if $DRY_RUN; then
        echo "  [DRY-RUN] cd ${DGXC_REPO}/nemotron4-15b && MAX_STEPS=${MAX_STEPS} RUN_CONF_IMAGE=${IMAGE_26} JOB_TOTAL_GPUS=${scale} GPU_TYPE=b200 DTYPE=${dtype} bash launch.sh"
        return
    fi

    cd "${DGXC_REPO}/nemotron4-15b"
    local output
    output=$(MAX_STEPS=${MAX_STEPS} \
        RUN_CONF_IMAGE=${IMAGE_26} \
        LLMB_INSTALL=${LLMB_DIR} \
        JOB_TOTAL_GPUS=${scale} \
        GPU_TYPE=b200 \
        DTYPE=${dtype} \
        SBATCH_ACCOUNT=root \
        SBATCH_PARTITION=batch \
        bash launch.sh 2>&1) || true
    echo "$output"

    local job_id
    job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+|jobid=\K[0-9]+' | head -1)

    if [[ -n "$job_id" ]]; then
        echo "  Job ID: ${job_id}"
        JOBS["$label"]="$job_id"
    else
        echo "  [WARN] Could not extract job ID"
    fi
    cd "${SCRIPT_DIR}"
}

submit_nemotron4_340b() {
    # Submit Nemotron4 340B via dgxc-benchmarking launch.sh (25.07 container)
    # Requires llmb_venv for fiddle dependency
    # TP_COMM_OVERLAP=False is required (PMIx v3/v4 mismatch breaks UserBuffers)
    # Generated sbatch is auto-patched with pyxis/PMIx/MPI stub workarounds
    local label="$1"
    local dtype="$2"
    local scale="$3"

    echo "--- [$label] nemotron4-340b ${dtype} ${scale}gpu ---"

    if $DRY_RUN; then
        echo "  [DRY-RUN] cd ${DGXC_REPO}/nemotron4-340b && TP_COMM_OVERLAP=False MAX_STEPS=${MAX_STEPS} JOB_TOTAL_GPUS=${scale} GPU_TYPE=b200 DTYPE=${dtype} bash launch.sh"
        return
    fi

    source /mnt/vast/johnson/llmb_venv/bin/activate

    cd "${DGXC_REPO}/nemotron4-340b"
    local output
    output=$(TP_COMM_OVERLAP=False \
        MAX_STEPS=${MAX_STEPS} \
        LLMB_INSTALL=${LLMB_DIR} \
        JOB_TOTAL_GPUS=${scale} \
        GPU_TYPE=b200 \
        DTYPE=${dtype} \
        SBATCH_ACCOUNT=root \
        SBATCH_PARTITION=batch \
        bash launch.sh 2>&1) || true
    echo "$output"

    local job_id
    job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+|jobid=\K[0-9]+' | head -1)

    if [[ -z "$job_id" ]]; then
        echo "  [WARN] Could not extract job ID"
        cd "${SCRIPT_DIR}"
        return 1
    fi

    echo "  Job ID: ${job_id} (auto-submitted, will cancel and resubmit patched)"

    # Cancel the auto-submitted job (unpatched sbatch will fail)
    scancel "$job_id" 2>/dev/null
    sleep 2

    # Find the generated sbatch script
    local exp_base="${LLMB_DIR}/workloads/pretrain_nemotron4-340b/experiments"
    local sbatch_script
    sbatch_script=$(find "$exp_base" -name "*_sbatch.sh" -newer /tmp/.llmb_marker_$$ -path "*${dtype}*gpus${scale}*" 2>/dev/null | sort | tail -1)

    if [[ -z "$sbatch_script" ]]; then
        echo "  [ERROR] Could not find generated sbatch script"
        cd "${SCRIPT_DIR}"
        return 1
    fi

    local patched_script="${sbatch_script%.sh}_patched.sh"
    patch_legacy_sbatch "$sbatch_script" "$patched_script"

    # Resubmit with patched script (exclude unhealthy nodes, tag Session A)
    local new_job_id
    new_job_id=$(sbatch --parsable --exclude="${EXCLUDE_NODES}" --job-name="${SESSION_TAG}_${label}" "$patched_script" 2>/dev/null)
    if [[ -n "$new_job_id" ]]; then
        echo "  Resubmitted as Job ID: ${new_job_id}"
        JOBS["$label"]="$new_job_id"
    else
        echo "  [ERROR] Failed to resubmit patched script"
    fi
    cd "${SCRIPT_DIR}"
}

submit_grok1() {
    # Submit Grok1 via dgxc-benchmarking launch.sh (25.09 container)
    # Requires llmb_venv for fiddle dependency
    # TP_COMM_OVERLAP=False is required (PMIx v3/v4 mismatch breaks UserBuffers)
    # Generated sbatch is auto-patched with pyxis/PMIx/MPI stub workarounds
    local label="$1"
    local dtype="$2"
    local scale="$3"

    echo "--- [$label] grok1-314b ${dtype} ${scale}gpu ---"

    if $DRY_RUN; then
        echo "  [DRY-RUN] cd ${DGXC_REPO}/grok1 && TP_COMM_OVERLAP=False MAX_STEPS=${MAX_STEPS} JOB_TOTAL_GPUS=${scale} GPU_TYPE=b200 DTYPE=${dtype} bash launch.sh"
        return
    fi

    source /mnt/vast/johnson/llmb_venv/bin/activate

    cd "${DGXC_REPO}/grok1"
    local output
    output=$(TP_COMM_OVERLAP=False \
        MAX_STEPS=${MAX_STEPS} \
        LLMB_INSTALL=${LLMB_DIR} \
        JOB_TOTAL_GPUS=${scale} \
        GPU_TYPE=b200 \
        DTYPE=${dtype} \
        SBATCH_ACCOUNT=root \
        SBATCH_PARTITION=batch \
        bash launch.sh 2>&1) || true
    echo "$output"

    local job_id
    job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+|jobid=\K[0-9]+' | head -1)

    if [[ -z "$job_id" ]]; then
        echo "  [WARN] Could not extract job ID"
        cd "${SCRIPT_DIR}"
        return 1
    fi

    echo "  Job ID: ${job_id} (auto-submitted, will cancel and resubmit patched)"

    # Cancel the auto-submitted job (unpatched sbatch will fail)
    scancel "$job_id" 2>/dev/null
    sleep 2

    # Find the generated sbatch script
    local exp_base="${LLMB_DIR}/workloads/pretrain_grok1/experiments"
    local sbatch_script
    sbatch_script=$(find "$exp_base" -name "*_sbatch.sh" -newer /tmp/.llmb_marker_$$ -path "*${dtype}*gpus${scale}*" 2>/dev/null | sort | tail -1)

    if [[ -z "$sbatch_script" ]]; then
        echo "  [ERROR] Could not find generated sbatch script"
        cd "${SCRIPT_DIR}"
        return 1
    fi

    local patched_script="${sbatch_script%.sh}_patched.sh"
    patch_legacy_sbatch "$sbatch_script" "$patched_script"

    # Resubmit with patched script (exclude unhealthy nodes, tag Session A)
    local new_job_id
    new_job_id=$(sbatch --parsable --exclude="${EXCLUDE_NODES}" --job-name="${SESSION_TAG}_${label}" "$patched_script" 2>/dev/null)
    if [[ -n "$new_job_id" ]]; then
        echo "  Resubmitted as Job ID: ${new_job_id}"
        JOBS["$label"]="$new_job_id"
    else
        echo "  [ERROR] Failed to resubmit patched script"
    fi
    cd "${SCRIPT_DIR}"
}

##############################################################################
# Pre-flight
##############################################################################

echo "=============================================="
echo "  B200 DGXC Official Benchmark Suite"
echo "  Container: 26.02 | Steps: ${MAX_STEPS} | Measure: 3-9"
echo "=============================================="
echo ""

idle_count=$(check_cluster | tail -1)

# Create marker file for finding new experiment dirs
touch /tmp/.llmb_marker_$$

##############################################################################
# Tier 1: 64-GPU jobs (8 nodes each)
##############################################################################

if $RUN_TIER1; then
    echo ""
    echo "========== TIER 1: 64-GPU Jobs (8 nodes) =========="
    if [[ "$idle_count" -lt 8 ]]; then
        echo "[ERROR] Need at least 8 idle nodes for Tier 1, have ${idle_count}"
    else
        # llmb-run models (26.02 container, native support)
        submit_llmb "llama70b_fp8_64"    pretrain_llama3.1  70b  fp8  64
        submit_llmb "nemotron_h_fp8_64"  pretrain_nemotron-h 56b fp8  64
        submit_llmb "qwen3_30b_bf16_64"  pretrain_qwen3     30b  bf16 64

        # Nemotron4 15B (launch.sh with 26.02 override)
        submit_nemotron4_15b "nemotron4_15b_bf16_64" bf16 64
        submit_nemotron4_15b "nemotron4_15b_fp8_64"  fp8  64
    fi
fi

##############################################################################
# Tier 2: 256-GPU jobs (32 nodes each)
##############################################################################

if $RUN_TIER2; then
    echo ""
    echo "========== TIER 2: 256-GPU Jobs (32 nodes) =========="
    if [[ "$idle_count" -lt 32 ]]; then
        echo "[WARN] Need 32 idle nodes for Tier 2, have ${idle_count}. Jobs may queue."
    fi

    # llmb-run models
    submit_llmb "llama70b_fp8_256"    pretrain_llama3.1   70b  fp8   256
    submit_llmb "llama405b_fp8_256"   pretrain_llama3.1   405b fp8   256
    submit_llmb "llama405b_nvfp4_256" pretrain_llama3.1   405b nvfp4 256
    submit_llmb "deepseek_fp8_256"    pretrain_deepseek-v3 ""  fp8   256
    submit_llmb "nemotron_h_fp8_256"  pretrain_nemotron-h  56b fp8   256
    submit_llmb "qwen3_235b_bf16_256" pretrain_qwen3       235b bf16 256

    # Nemotron4 15B (launch.sh with 26.02 override)
    submit_nemotron4_15b "nemotron4_15b_bf16_256" bf16 256
    submit_nemotron4_15b "nemotron4_15b_fp8_256"  fp8  256
fi

##############################################################################
# Tier 3: 512-GPU jobs (64 nodes each) — sequential, uses full cluster
##############################################################################

if $RUN_TIER3; then
    echo ""
    echo "========== TIER 3: 512-GPU Jobs (64 nodes) — Sequential =========="
    if [[ "$idle_count" -lt 64 ]]; then
        echo "[WARN] Need 64 idle nodes for Tier 3, have ${idle_count}. Jobs may queue."
    fi

    # Priority 1: llmb-run models (26.02 container, most reliable)
    submit_llmb "llama70b_fp8_512"     pretrain_llama3.1    70b  fp8   512
    submit_llmb "llama70b_nvfp4_512"   pretrain_llama3.1    70b  nvfp4 512
    submit_llmb "llama405b_fp8_512"    pretrain_llama3.1    405b fp8   512
    submit_llmb "llama405b_nvfp4_512"  pretrain_llama3.1    405b nvfp4 512
    submit_llmb "nemotron_h_fp8_512"   pretrain_nemotron-h  56b  fp8   512

    # Priority 2: dgxc-benchmarking models (legacy containers)
    submit_nemotron4_340b "nemotron4_340b_fp8_512"  fp8  512
    submit_nemotron4_340b "nemotron4_340b_bf16_512" bf16 512
    submit_grok1 "grok1_bf16_512" bf16 512
    submit_grok1 "grok1_fp8_512"  fp8  512

    # Priority 3: risky (BF16 variant known to NCCL-timeout at 512)
    submit_llmb "deepseek_fp8_512" pretrain_deepseek-v3 "" fp8 512
fi

##############################################################################
# Summary
##############################################################################

echo ""
echo "========== Submitted Jobs =========="
for label in "${!JOBS[@]}"; do
    printf "  %-30s  Job %s\n" "$label" "${JOBS[$label]}"
done
echo ""
echo "Monitor: squeue -u $(whoami)"
echo "Done."

# Cleanup
rm -f /tmp/.llmb_marker_$$
