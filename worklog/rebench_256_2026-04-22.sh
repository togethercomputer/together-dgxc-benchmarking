#!/bin/bash
# Rebenchmark batch at 256 GPUs — 2026-04-22
# Serial execution: submit, patch max_steps, wait, parse, repeat.
#
# Uses: llmb-run (Llama/N-H/DSV3), submit_session_B.sh (Qwen3).
# Skips already-running job 84589 (Llama 70B NVFP4) — tracked externally.

set -uo pipefail

LLMB_INSTALL=/mnt/vast/johnson/llmb
VENV=/mnt/vast/johnson/llmb_venv
EXCLUDE="use3a-ss-b200-gpu-[130,190,197,199,201,211,228,233,239]"
RESULTS=/home/johnson/johnson/worklog/rebench_256_2026-04-22_results.tsv
LOG=/home/johnson/johnson/worklog/rebench_256_2026-04-22_orchestrator.log

export LLMB_INSTALL
source "${VENV}/bin/activate"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

# results.tsv header (create if missing)
if [[ ! -f "$RESULTS" ]]; then
    echo -e "timestamp\tmodel\tdtype\tjob_id\titer5_step_s\titer5_tflops\tss_tflops_avg\tstatus" > "$RESULTS"
fi

wait_for_job() {
    local jid="$1"
    local name="$2"
    log "  waiting for $name ($jid)"
    while squeue -h -j "$jid" 2>/dev/null | grep -q "$jid"; do
        sleep 30
    done
    log "  $name ($jid) left queue"
}

parse_result() {
    # Args: log_file model dtype job_id
    local lf="$1" model="$2" dtype="$3" jid="$4"
    if [[ ! -f "$lf" ]]; then
        echo -e "$(ts)\t$model\t$dtype\t$jid\t-\t-\t-\tNO_LOG" >> "$RESULTS"
        log "  NO_LOG: $lf missing"
        return 1
    fi
    local iter5 iter5_tf
    iter5=$(grep -E "iteration 5/" "$lf" | head -1 | grep -oE "train_step_timing in s: [0-9.]+" | awk '{print $NF}')
    iter5_tf=$(grep -E "iteration 5/" "$lf" | head -1 | grep -oE "TFLOPS_per_GPU: [0-9.e+]+" | awk '{print $NF}')
    # Steady-state: iters 3-9 TFLOPS average
    local ss_avg
    ss_avg=$(grep -E "iteration [3-9]/" "$lf" | grep -oE "TFLOPS_per_GPU: [0-9.e+]+" | awk '{sum+=$NF; n++} END {if(n>0) printf "%.0f", sum/n; else print "-"}')
    local status="OK"
    if [[ -z "$iter5" ]]; then
        status="NO_ITER5"
        iter5="-"; iter5_tf="-"
    fi
    if ! grep -q "Trainer.fit.*stopped.*max_steps" "$lf" 2>/dev/null; then
        status="INCOMPLETE"
    fi
    echo -e "$(ts)\t$model\t$dtype\t$jid\t$iter5\t$iter5_tf\t$ss_avg\t$status" >> "$RESULTS"
    log "  RESULT $model $dtype: iter5=${iter5}s ${iter5_tf} TFLOPS, ss_avg=${ss_avg} [${status}]"
}

find_log() {
    # Args: workload_subdir job_id -> echoes log file path (latest matching)
    local ws="$1" jid="$2"
    find "${LLMB_INSTALL}/workloads/${ws}/experiments" -type f -name "log-*_${jid}_0.out" 2>/dev/null | head -1
}

find_script() {
    local ws="$1"
    # latest scripts/*.sh under workload dir (modified in last 5 min)
    find "${LLMB_INSTALL}/workloads/${ws}/experiments" -type f -name "*.sh" -path "*/scripts/*" 2>/dev/null \
        | xargs -r ls -t 2>/dev/null | head -1
}

submit_llmb() {
    # Args: short_name workload size dtype (size empty for DSV3/N-H)
    local name="$1" ws_cli="$2" size="$3" dtype="$4"
    log "submitting $name ($ws_cli $size $dtype)"
    local size_arg=""
    [[ -n "$size" ]] && size_arg="-s $size"
    local out
    out=$(llmb-run submit -w "$ws_cli" $size_arg -d "$dtype" --scale 256 --exclude "$EXCLUDE" 2>&1)
    echo "$out" >> "$LOG"
    local jid
    jid=$(echo "$out" | grep -oP 'Job id: \K[0-9]+' | head -1)
    if [[ -z "$jid" ]]; then
        log "  FAILED to get job id for $name"
        echo -e "$(ts)\t$name\t$dtype\t-\t-\t-\t-\tSUBMIT_FAIL" >> "$RESULTS"
        return 1
    fi
    log "  job $jid"
    # Find and patch the new script
    local ws_dir
    case "$ws_cli" in
        pretrain_llama3.1) ws_dir="pretrain_llama3.1" ;;
        pretrain_nemotron-h) ws_dir="pretrain_nemotron-h" ;;
        pretrain_deepseek-v3) ws_dir="pretrain_deepseek-v3" ;;
    esac
    sleep 5
    local script
    script=$(find_script "$ws_dir")
    if [[ -n "$script" ]]; then
        sed -i 's/--max_steps=50/--max_steps=10/g' "$script"
        log "  patched max_steps in $script"
    else
        log "  WARN no script found to patch for $ws_dir"
    fi
    wait_for_job "$jid" "$name"
    sleep 30  # let log flush
    local lf
    lf=$(find_log "$ws_dir" "$jid")
    parse_result "$lf" "$name" "$dtype" "$jid"
}

submit_qwen3() {
    local dtype="$1"
    log "submitting qwen3_235b $dtype via submit_session_B.sh"
    local out
    out=$(SCALE=256 bash /home/johnson/johnson/worklog/submit_session_B.sh "qwen3_${dtype}" 2>&1)
    echo "$out" >> "$LOG"
    local jid
    jid=$(echo "$out" | grep -oP 'Resubmitted as Job \K[0-9]+' | head -1)
    if [[ -z "$jid" ]]; then
        log "  FAILED to submit qwen3 $dtype"
        echo -e "$(ts)\tqwen3_235b\t$dtype\t-\t-\t-\t-\tSUBMIT_FAIL" >> "$RESULTS"
        return 1
    fi
    log "  job $jid"
    wait_for_job "$jid" "qwen3_235b_${dtype}"
    sleep 30
    local lf
    lf=$(find_log "pretrain_qwen3" "$jid")
    parse_result "$lf" "qwen3_235b" "$dtype" "$jid"
}

# Wait for already-running job 84589 (Llama 70B NVFP4), then parse it
log "=== Batch start: 256-GPU rebench 2026-04-22 ==="
log "waiting for pre-existing job 84589 (Llama 70B NVFP4)"
wait_for_job 84589 "llama70b_nvfp4"
sleep 30
LOGFILE=$(find_log "pretrain_llama3.1" 84589)
parse_result "$LOGFILE" "llama70b" "nvfp4" "84589"

# Remaining queue
submit_llmb "llama70b"      "pretrain_llama3.1"   "70b"  "fp8"
submit_llmb "llama405b"     "pretrain_llama3.1"   "405b" "nvfp4"
submit_llmb "llama405b"     "pretrain_llama3.1"   "405b" "fp8"
submit_qwen3 "bf16"
submit_qwen3 "fp8"
submit_llmb "nemotron-h_56b" "pretrain_nemotron-h" "56b"  "fp8"
submit_llmb "deepseek-v3"    "pretrain_deepseek-v3" ""    "fp8"

log "=== Batch complete ==="
log "Results: $RESULTS"
