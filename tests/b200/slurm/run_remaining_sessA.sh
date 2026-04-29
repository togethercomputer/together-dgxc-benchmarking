#!/bin/bash
# Session A: submit the 6 remaining 512-GPU jobs (Llama 70B FP8 was cancelled;
# legacy models never submitted). 84374-84377 already queued.
set -euo pipefail

# Source the helper functions from run_all_sessA.sh (but skip its tier dispatcher)
# We do this by disabling the tier flags via arg stripping.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export the same constants the main script uses
LLMB_DIR=/mnt/vast/johnson/llmb
DGXC_REPO=/mnt/vast/johnson/dgxc-benchmarking
IMAGE_26=/mnt/vast/johnson/llmb/images/nvidia+nemo+26.02.00.sqsh
MAX_STEPS=10
EXCLUDE_NODES='use3a-ss-b200-gpu-[130,190,197,199,201,211,233,239]'
SESSION_TAG=sessA
DRY_RUN=false
declare -A JOBS
touch /tmp/.llmb_marker_$$

# Inline-copy the needed submit_* functions (kept local to avoid picking up tier dispatcher)
submit_llmb() {
    local label="$1" workload="$2" size="$3" dtype="$4" scale="$5"
    echo "--- [$label] ${workload} ${size} ${dtype} ${scale}gpu ---"
    local cmd="./llmb-run submit -w ${workload}"
    [[ -n "$size" ]] && cmd+=" -s ${size}"
    cmd+=" -d ${dtype} --scale ${scale} --exclude ${EXCLUDE_NODES}"
    cd "${LLMB_DIR}"
    local output
    output=$(eval "${cmd}" 2>&1) || true
    echo "$output"
    local job_id
    job_id=$(echo "$output" | grep -oP 'jobid=\K[0-9]+' | head -1)
    if [[ -z "$job_id" ]]; then
        echo "  [ERROR] No job ID"
        cd "${SCRIPT_DIR}"
        return 1
    fi
    echo "  Job ID: ${job_id}"
    JOBS["$label"]="$job_id"
    scontrol update job="$job_id" JobName="${SESSION_TAG}_${label}" 2>/dev/null || true
    sleep 1
    local exp_base="${LLMB_DIR}/workloads/${workload}/experiments"
    local newest_script
    newest_script=$(find "$exp_base" -name "*.sh" -path "*/scripts/*" -newermt "1 minute ago" 2>/dev/null | head -1)
    if [[ -n "$newest_script" ]]; then
        if grep -qE "max_steps[= ]50" "$newest_script"; then
            sed -i -E "s/--max_steps[= ]50/--max_steps=${MAX_STEPS}/g" "$newest_script"
            echo "  Patched max_steps in $(basename "$newest_script")"
        fi
    fi
    cd "${SCRIPT_DIR}"
}

patch_legacy_sbatch() {
    local src="$1" dst="$2"
    cp "$src" "$dst"
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
    sed -i 's|:/nemo_run |:/nemo_run,/mnt/vast/johnson/llmb/mpi_stub:/mpi_stub |g' "$dst"
    sed -i 's|--container-env=\(.*\) bash|--container-env=\1,HOME,NEMO_NLP_TMP,HF_HOME,NCCL_SOCKET_IFNAME,PMIX_MCA_gds,OMPI_MCA_plm,MPI4PY_RC_INITIALIZE,MPI4PY_RC_THREADS,MPI4PY_RC_FINALIZE,LD_PRELOAD bash|' "$dst"
    echo "  Patched: $(basename "$dst")"
}

submit_legacy() {
    # $1=label $2=workload_dir (nemotron4-340b or grok1) $3=dtype $4=scale $5=workload_name
    local label="$1" wdir="$2" dtype="$3" scale="$4" wname="$5"
    echo "--- [$label] $wdir $dtype ${scale}gpu ---"
    source /mnt/vast/johnson/llmb_venv/bin/activate
    cd "${DGXC_REPO}/${wdir}"
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
        echo "  [WARN] No job ID"
        cd "${SCRIPT_DIR}"
        return 1
    fi
    echo "  Auto-submitted ${job_id}; cancelling + patching + resubmitting"
    scancel "$job_id" 2>/dev/null
    sleep 2
    local exp_base="${LLMB_DIR}/workloads/${wname}/experiments"
    local sbatch_script
    sbatch_script=$(find "$exp_base" -name "*_sbatch.sh" -newer /tmp/.llmb_marker_$$ -path "*${dtype}*gpus${scale}*" 2>/dev/null | sort | tail -1)
    if [[ -z "$sbatch_script" ]]; then
        echo "  [ERROR] Could not find generated sbatch script"
        cd "${SCRIPT_DIR}"
        return 1
    fi
    local patched_script="${sbatch_script%.sh}_patched.sh"
    patch_legacy_sbatch "$sbatch_script" "$patched_script"
    local new_job_id
    new_job_id=$(sbatch --parsable --exclude="${EXCLUDE_NODES}" --job-name="${SESSION_TAG}_${label}" "$patched_script" 2>/dev/null)
    if [[ -n "$new_job_id" ]]; then
        echo "  Resubmitted as ${new_job_id}"
        JOBS["$label"]="$new_job_id"
    else
        echo "  [ERROR] Resubmit failed"
    fi
    cd "${SCRIPT_DIR}"
}

echo "========== Session A: remaining 6 jobs =========="

# 1. Re-submit Llama 70B FP8 (was 84373, cancelled)
submit_llmb "llama70b_fp8_512" pretrain_llama3.1 70b fp8 512

# 2-3. Nemotron4 340B FP8 + BF16 (legacy 25.07)
submit_legacy "nemotron4_340b_fp8_512"  nemotron4-340b fp8  512 pretrain_nemotron4-340b
submit_legacy "nemotron4_340b_bf16_512" nemotron4-340b bf16 512 pretrain_nemotron4-340b

# 4-5. Grok1 BF16 + FP8 (legacy 25.09)
submit_legacy "grok1_bf16_512" grok1 bf16 512 pretrain_grok1
submit_legacy "grok1_fp8_512"  grok1 fp8  512 pretrain_grok1

# 6. DeepSeek V3 FP8
submit_llmb "deepseek_fp8_512" pretrain_deepseek-v3 "" fp8 512

echo ""
echo "========== Submitted =========="
for label in "${!JOBS[@]}"; do
    printf "  %-30s  %s\n" "$label" "${JOBS[$label]}"
done
rm -f /tmp/.llmb_marker_$$
