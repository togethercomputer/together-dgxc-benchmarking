#!/bin/bash
#
# Master benchmark script — submit all official NVIDIA baselines
# Container: 26.02 for all models
# Iterations: 10 steps (measure steps 3-9)
#
# Usage:
#   bash run_all_official.sh [--tier1] [--tier2] [--dry-run]
#
# --tier1: only 64-GPU jobs (8 nodes each)
# --tier2: only 256-GPU jobs (32 nodes each)
# --dry-run: show what would be submitted without actually submitting
#
# Skipped models: Grok1, Nemotron4 340B (legacy containers), all 512-GPU configs
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLMB_DIR=/mnt/vast/johnson/llmb
DGXC_REPO=/mnt/vast/johnson/dgxc-benchmarking
IMAGE_26=/mnt/vast/johnson/llmb/images/nvidia+nemo+26.02.00.sqsh
MAX_STEPS=10

# Parse arguments
RUN_TIER1=false
RUN_TIER2=false
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --tier1) RUN_TIER1=true ;;
        --tier2) RUN_TIER2=true ;;
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done
# If neither specified, run both
if ! $RUN_TIER1 && ! $RUN_TIER2; then
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
    cmd+=" -d ${dtype} --scale ${scale}"

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

    # Post-patch: find the newly created experiment script and change max_steps
    sleep 1  # brief wait for filesystem
    local newest_script
    newest_script=$(find "$exp_base" -name "*.sh" -path "*/scripts/*" -newer /tmp/.llmb_marker_$$ 2>/dev/null | head -1)

    # Fallback: search by job ID in the experiment directory
    if [[ -z "$newest_script" ]]; then
        newest_script=$(find "$exp_base" -name "*.sh" -path "*/scripts/*" -newermt "1 minute ago" 2>/dev/null | head -1)
    fi

    if [[ -n "$newest_script" ]]; then
        if grep -q "max_steps[= ]50" "$newest_script"; then
            sed -i "s/--max_steps[= ]50/--max_steps=${MAX_STEPS}/g" "$newest_script"
            echo "  Patched: max_steps 50 -> max_steps ${MAX_STEPS} in $(basename "$newest_script")"
        else
            echo "  [WARN] max_steps=50 not found in script, may already be patched or different default"
        fi
    else
        echo "  [WARN] Could not find generated script to patch max_steps"
    fi
    cd "${SCRIPT_DIR}"
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
