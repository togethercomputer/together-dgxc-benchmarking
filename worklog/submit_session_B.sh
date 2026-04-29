#!/bin/bash
#
# Session B — submit the 5 models that were NOT run at 512 GPUs on Friday 2026-04-17.
#
# Submission strategy per model (derived from Friday's successful patched scripts):
#   1. Run launch.sh (NeMo or Megatron-Bridge) to generate experiment dir + sbatch
#   2. Capture auto-submitted job ID, cancel (unpatched sbatch fails on this cluster)
#   3. Patch generated sbatch: B_ job name, HOME=/tmp, unhealthy-node exclude
#   4. Resubmit patched sbatch
#
# Pre-requisites (already applied this session):
#   - helpers.py patched for megatron-core 0.13.0rc4 API rename
#     (keep_fp8_transpose_cache -> keep_fp8_transpose_cache_when_using_custom_fsdp)
#   - launch.sh (nemotron4-15b, qwen3, deepseek_v3) patched to use
#     --additional_slurm_params=VALUE (= not space) so argparse doesn't eat
#     --job-name= values starting with --
#
# Usage: bash submit_session_B.sh [MODEL_NAME]
#   MODEL_NAME one of: n15b_bf16 n15b_fp8 qwen3_bf16 qwen3_fp8 dsv3_bf16
#   Omit to run all 5 sequentially.

set -uo pipefail

LLMB_DIR=/mnt/vast/johnson/llmb
DGXC_REPO=/mnt/vast/johnson/dgxc-benchmarking
IMAGE_26=${LLMB_DIR}/images/nvidia+nemo+26.02.00.sqsh
VENV=/mnt/vast/johnson/llmb_venv
MAX_STEPS=10
SCALE=${SCALE:-512}
# Proven compat_runner.py from 15B 256-GPU success (Apr 15)
COMPAT_RUNNER_SRC=/mnt/vast/johnson/llmb/workloads/pretrain_nemotron4-15b/experiments/pretrain_nemotron4_15b_bf16_gpus256_tp1_pp1_cp1_vp1_mbs2_gbs1024/pretrain_nemotron4_15b_bf16_gpus256_tp1_pp1_cp1_vp1_mbs2_gbs1024_1776133099/pretrain_nemotron4_15b_bf16_gpus256_tp1_pp1_cp1_vp1_mbs2_gbs1024/compat_runner.py

# Unhealthy nodes to exclude
# - 181: confirmed NCCL IB/RoCE link-type mismatch on 2026-04-23 (jobs 84650/84653/84654)
# - 190: user request
EXCLUDE="use3a-ss-b200-gpu-[181,190]"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

patch_modern_sbatch() {
    # Patch a NeMo/Megatron-Bridge generated sbatch for this cluster.
    #
    #   - Replace the auto-generated "--job-name=root-root..." line with a B_ name
    #     (some generators emit TWO --job-name lines, only the last wins; we strip
    #      the auto one so the B_ one at the top takes effect)
    #   - Insert --exclude=<unhealthy>
    #   - Prepend HOME=/tmp, NEMO_NLP_TMP=/tmp (pyxis HOME workaround) before srun
    #   - Extend --container-env with HOME,NEMO_NLP_TMP,HF_HOME
    #
    # Usage: patch_modern_sbatch <sbatch> <B_job_name>
    local sbatch="$1"
    local bname="$2"

    if [[ ! -f "$sbatch" ]]; then
        echo "  [ERROR] patch_modern_sbatch: missing $sbatch"
        return 1
    fi

    # 1) Remove auto-generated --job-name=root-root... line (keeps ours)
    sed -i '/^#SBATCH --job-name=root-root/d' "$sbatch"

    # 2) Ensure our B_ job-name is present (idempotent)
    if ! grep -q "^#SBATCH --job-name=${bname}" "$sbatch"; then
        # Replace any leftover --job-name=... with B_ name
        sed -i "s|^#SBATCH --job-name=.*|#SBATCH --job-name=${bname}|" "$sbatch"
    fi

    # 3) Insert --exclude if not present
    if ! grep -q "^#SBATCH --exclude=" "$sbatch"; then
        sed -i "/^#SBATCH --partition=/a #SBATCH --exclude=${EXCLUDE}" "$sbatch"
    fi

    # 4) Insert HOME=/tmp block + NCCL socket before the srun line (before "# Command 1")
    #    NCCL_SOCKET_IFNAME=bond0 and NCCL_IB_HCA are cluster-critical: without them
    #    Gloo/NCCL rendezvous picks wrong iface -> broadcast_object_list segfault
    #    (see Apr 15 83825 vs Apr 20 84439 sbatch diff)
    if ! grep -q "# --- pyxis HOME + NCCL socket workaround" "$sbatch"; then
        sed -i '/^# Command 1/i\
# --- pyxis HOME + NCCL socket workaround (auto-patched by submit_session_B.sh) ---\
export HOME=/tmp\
export NEMO_NLP_TMP=/tmp/nemo_nlp_tmp\
export HF_HOME=/mnt/vast/johnson/llmb/.cache/huggingface\
export HF_HUB_OFFLINE=1\
export TRANSFORMERS_OFFLINE=1\
export HF_TOKEN=${HF_TOKEN:?Please set HF_TOKEN env var before running}\
export NCCL_SOCKET_IFNAME=bond0\
ulimit -n 1048576 || true\
# --- end ---' "$sbatch"
    fi

    # 5) Ensure --container-env= exists on the srun line and contains the vars
    #    we need forwarded. CRITICAL: must include NCCL_SOCKET_IFNAME,
    #    TORCH_NCCL_HIGH_PRIORITY, NVTE_FWD/BWD_LAYERNORM_SM_MARGIN, and HF_HOME
    #    + HF_HUB_OFFLINE + TRANSFORMERS_OFFLINE (else inside-container HF lookup
    #    falls back to online mode and fails — Job 84458 Qwen3-235B).
    local extra_env="HOME,NEMO_NLP_TMP,HF_HOME,HF_HUB_OFFLINE,TRANSFORMERS_OFFLINE,HF_TOKEN,NCCL_SOCKET_IFNAME,TORCH_NCCL_HIGH_PRIORITY,NVTE_FWD_LAYERNORM_SM_MARGIN,NVTE_BWD_LAYERNORM_SM_MARGIN"
    if grep -q '\-\-container-env=' "$sbatch"; then
        # Extend existing container-env (idempotent via HF_HUB_OFFLINE marker)
        if ! grep -q 'container-env=[^ ]*HF_HUB_OFFLINE' "$sbatch"; then
            sed -i "s|--container-env=\([^ ]*\) |--container-env=\1,${extra_env} |g" "$sbatch"
        fi
    else
        # No --container-env= present (Megatron-Bridge launcher behavior).
        # Insert it right after --no-container-mount-home.
        sed -i "s|--no-container-mount-home|--no-container-mount-home --container-env=${extra_env}|" "$sbatch"
    fi

    echo "  Patched: $(basename "$sbatch")"
}

install_compat_runner() {
    # Copy compat_runner.py into the experiment's mounted /nemo_run dir,
    # and rewrite scripts/X.sh to invoke it instead of fdl_runner directly.
    # This replicates the 15B 256-GPU (Apr 15) success pattern.
    #
    # Usage: install_compat_runner <experiment_root_dir>
    local exp_root="$1"
    if [[ ! -d "$exp_root" ]]; then
        echo "  [ERROR] install_compat_runner: missing $exp_root"
        return 1
    fi
    cp "$COMPAT_RUNNER_SRC" "$exp_root/compat_runner.py"

    # Rewrite the scripts/*.sh to call compat_runner.py
    local script
    for script in "$exp_root"/scripts/*.sh; do
        [[ -f "$script" ]] || continue
        sed -i 's|python -m nemo_run\.core\.runners\.fdl_runner|python /nemo_run/compat_runner.py|g' "$script"
    done
    echo "  Installed compat_runner.py (scripts rewritten to use it)"
}

find_latest_sbatch() {
    # Locate the just-generated top-level sbatch for a workload/dtype/scale.
    # Usage: find_latest_sbatch <workload_dir> <dtype> <scale>
    local exp_base="$1"
    local dtype="$2"
    local scale="$3"
    find "$exp_base" -maxdepth 4 -name "*_sbatch.sh" -path "*${dtype}*gpus${scale}*" 2>/dev/null \
        | xargs -r ls -t 2>/dev/null | head -1
}

submit_one() {
    # Capture auto-submitted job, cancel, patch, resubmit with B_ name.
    # Usage: submit_one <short_name> <dtype> <workload_dir> <launcher_cmd>
    local name="$1"
    local dtype="$2"
    local workload_dir="$3"
    local launcher="$4"
    local bname="B_${name}_${dtype}_${SCALE}"

    echo ""
    echo "=============================================================="
    echo "  [${bname}] launching via ${launcher}"
    echo "=============================================================="

    local marker=/tmp/.sessB_marker_$$
    touch "$marker"

    # 1) Invoke launcher; it auto-submits an unpatched sbatch
    local output rc
    output=$(eval "${launcher}" 2>&1)
    rc=$?
    echo "$output" | tail -40

    # 2) Extract job ID (NeMo: 'Launched app: slurm_tunnel://nemo_run/<jid>')
    local auto_jid
    auto_jid=$(echo "$output" | grep -oP 'nemo_run/\K[0-9]+|Submitted batch job \K[0-9]+|jobid=\K[0-9]+' | head -1)

    if [[ -z "$auto_jid" ]]; then
        echo "  [ERROR] Could not find auto-submitted job ID. Exit=${rc}"
        rm -f "$marker"
        return 1
    fi
    echo "  Auto-submitted: ${auto_jid} (will cancel + patch + resubmit)"
    scancel "$auto_jid" 2>/dev/null || true
    sleep 2

    # 3) Locate the generated sbatch
    local sbatch
    sbatch=$(find "${workload_dir}/experiments" -maxdepth 4 -name "*_sbatch.sh" -newer "$marker" 2>/dev/null | head -1)
    rm -f "$marker"
    if [[ -z "$sbatch" ]]; then
        echo "  [ERROR] Could not find generated sbatch under ${workload_dir}/experiments"
        return 1
    fi
    echo "  Generated: ${sbatch}"

    # 4) Patch sbatch
    patch_modern_sbatch "$sbatch" "$bname" || return 1

    # 4b) Install compat_runner.py (15B/NeMo path only — MB models don't need it)
    #     Skip when N15B_USE_COMPAT=0 (e.g. FP8 on 25.09 container which is
    #     natively compatible — compat_runner is only needed for 26.02).
    if [[ "$name" == "n15b" ]] && [[ "${N15B_USE_COMPAT:-1}" == "1" ]]; then
        local exp_root
        exp_root=$(dirname "$sbatch")
        local nested
        nested=$(find "$exp_root" -mindepth 1 -maxdepth 1 -type d -name "pretrain_*" | head -1)
        if [[ -n "$nested" ]]; then
            install_compat_runner "$nested" || return 1
        fi
    fi

    # 5) Resubmit
    local new_jid
    new_jid=$(sbatch --parsable "$sbatch")
    if [[ -z "$new_jid" ]]; then
        echo "  [ERROR] Resubmit failed"
        return 1
    fi
    echo "  Resubmitted as Job ${new_jid} (name: ${bname})"
    echo "${bname} ${new_jid}" >> /tmp/sessB_jobs.log
}

# -----------------------------------------------------------------------------
# Per-model wrappers
# -----------------------------------------------------------------------------

submit_n15b() {
    local dtype="$1"
    source "${VENV}/bin/activate"
    cd "${DGXC_REPO}/nemotron4-15b"
    # 15B FP8 requires 25.09 container: 26.02+compat_runner triggers a CUDA-graph
    # TransformerEngine FP8 incompat (Job 84450: cudaErrorInvalidValue in
    # replay_graph_capture). BF16 works on either.
    local image="${IMAGE_26}"
    local use_compat=1
    if [[ "${dtype}" == "fp8" ]]; then
        image="${LLMB_DIR}/images/nvidia+nemo+25.09.00.sqsh"
        use_compat=0
    fi
    local launcher="MAX_STEPS=${MAX_STEPS} \
        RUN_CONF_IMAGE=${image} \
        LLMB_INSTALL=${LLMB_DIR} \
        JOB_TOTAL_GPUS=${SCALE} \
        GPU_TYPE=b200 \
        DTYPE=${dtype} \
        SBATCH_ACCOUNT=root \
        SBATCH_PARTITION=batch \
        ADDITIONAL_SLURM_PARAMS='job-name=B_n15b_${dtype}_${SCALE}' \
        bash launch.sh"
    N15B_USE_COMPAT="${use_compat}" \
        submit_one "n15b" "${dtype}" "${LLMB_DIR}/workloads/pretrain_nemotron4-15b" "${launcher}"
}

submit_qwen3() {
    local dtype="$1"
    source "${VENV}/bin/activate"
    cd "${DGXC_REPO}/qwen3/pretrain"
    local launcher="MAX_STEPS=${MAX_STEPS} \
        RUN_CONF_IMAGE=${IMAGE_26} \
        LLMB_INSTALL=${LLMB_DIR} \
        MODEL_SIZE=235b \
        JOB_TOTAL_GPUS=${SCALE} \
        GPU_TYPE=b200 \
        DTYPE=${dtype} \
        SBATCH_ACCOUNT=root \
        SBATCH_PARTITION=batch \
        ADDITIONAL_SLURM_PARAMS='job-name=B_qwen3_235b_${dtype}_${SCALE}' \
        bash launch.sh"
    submit_one "qwen3_235b" "${dtype}" "${LLMB_DIR}/workloads/pretrain_qwen3" "${launcher}"
}

submit_dsv3() {
    local dtype="$1"
    source "${VENV}/bin/activate"
    cd "${DGXC_REPO}/deepseek_v3/pretrain/megatron_bridge"
    local launcher="MAX_STEPS=${MAX_STEPS} \
        RUN_CONF_IMAGE=${IMAGE_26} \
        LLMB_INSTALL=${LLMB_DIR} \
        JOB_TOTAL_GPUS=${SCALE} \
        GPU_TYPE=b200 \
        DTYPE=${dtype} \
        SBATCH_ACCOUNT=root \
        SBATCH_PARTITION=batch \
        ADDITIONAL_SLURM_PARAMS='job-name=B_dsv3_${dtype}_${SCALE}' \
        bash launch.sh"
    submit_one "dsv3" "${dtype}" "${LLMB_DIR}/workloads/pretrain_deepseek-v3" "${launcher}"
}

# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------

echo "Session B submission — $(date)"
echo "Cluster idle nodes: $(sinfo -p batch -t idle -h -o "%D" 2>/dev/null || echo '?')"
echo ""

target="${1:-all}"
case "$target" in
    n15b_bf16)  submit_n15b bf16 ;;
    n15b_fp8)   submit_n15b fp8  ;;
    qwen3_bf16) submit_qwen3 bf16 ;;
    qwen3_fp8)  submit_qwen3 fp8  ;;
    dsv3_bf16)  submit_dsv3 bf16 ;;
    all)
        submit_n15b bf16
        submit_n15b fp8
        submit_qwen3 bf16
        submit_qwen3 fp8
        submit_dsv3 bf16
        ;;
    *) echo "Unknown target: $target"; exit 1 ;;
esac

echo ""
echo "=============================================================="
echo "Submitted jobs:"
cat /tmp/sessB_jobs.log 2>/dev/null
echo "=============================================================="
