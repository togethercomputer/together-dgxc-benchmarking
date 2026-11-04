#!/usr/bin/env bash
# Sequential GB200 readiness campaign on cluster 661303b8 (namespace infra-perf).
# Stages: qwen run1 (already running) -> qwen run2 -> gpt-oss run1/run2 -> NCCL 12n on pytorch:25.09.
# One job at a time; each stage torn down (pods + ComputeDomain gone) before the next.
set -uo pipefail
export KUBECONFIG=~/.kube/config
NS=infra-perf
Y=~/together-dgxc-benchmarking/tests/gb200/GB200
STEPS=${STEPS:-95}
log() { echo "[$(date +%H:%M:%S)] $*"; }

pod_of() { kubectl -n $NS get pods -l jobset.sigs.k8s.io/jobset-name=$1,batch.kubernetes.io/job-completion-index=$2 \
             -o jsonpath='{.items[0].metadata.name}' 2>/dev/null; }

teardown() {  # $1 yaml
  kubectl delete -f "$1" --ignore-not-found --wait=true >/dev/null 2>&1
  for i in $(seq 1 90); do
    [ "$(kubectl -n $NS get pods --no-headers 2>/dev/null | wc -l)" = 0 ] && \
    [ "$(kubectl -n $NS get computedomains --no-headers 2>/dev/null | wc -l)" = 0 ] && return 0
    sleep 5
  done
  log "WARN: teardown of $1 not clean after 450s"; kubectl -n $NS get pods,computedomains
}

train() {  # $1 jobset  $2 yaml  $3 outlog  [$4 launch:1/0]
  local J=$1 F=$2 OUT=$3
  if [ "${4:-1}" = 1 ]; then log "launch $J -> $OUT"; kubectl apply -f "$F" >/dev/null || return 1; fi
  local t0=$(date +%s)
  while :; do
    local P=$(pod_of $J 0) N=0
    [ -n "$P" ] && N=$(kubectl -n $NS logs "$P" 2>/dev/null | grep -c 'MODEL_TFLOP' || true)
    if [ "$N" -ge "$STEPS" ]; then kubectl -n $NS logs "$P" > "$OUT"; log "$J reached $N steps"; break; fi
    if kubectl -n $NS get pods -l jobset.sigs.k8s.io/jobset-name=$J --no-headers 2>/dev/null \
         | grep -qE 'Error|Failed|CrashLoop|OOMKilled|CreateContainerConfigError|ImagePull'; then
      kubectl -n $NS get pods -l jobset.sigs.k8s.io/jobset-name=$J
      [ -n "$P" ] && kubectl -n $NS logs "$P" --tail=200 > "$OUT.fail"
      log "FAILED $J (see $OUT.fail)"; return 1
    fi
    [ $(( $(date +%s) - t0 )) -gt 7200 ] && { log "TIMEOUT $J"; [ -n "$P" ] && kubectl -n $NS logs "$P" > "$OUT.fail"; return 1; }
    sleep 30
  done
  grep -oE '[0-9.]+MODEL_TFLOP' "$OUT" | tail -30 | tr -d 'A-Z_' | sort -n | awk '{a[NR]=$1} END{printf "  last-30 median TFLOP/s/GPU = %.1f\n", (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'
  teardown "$F"; return 0
}

nccl() {  # $1 jobset $2 yaml $3 outlog
  local J=$1 F=$2 OUT=$3 t0=$(date +%s) rc=0
  if [ "${4:-1}" = 1 ]; then log "launch $J -> $OUT"; kubectl apply -f "$F" >/dev/null || return 1; fi
  while :; do
    local P=$(pod_of $J 0) S=$(kubectl -n $NS get pod "$(pod_of $J 0)" -o jsonpath='{.status.phase}' 2>/dev/null)
    case "$S" in Succeeded|Failed) kubectl -n $NS logs "$P" > "$OUT"; log "$J rank0 $S"; [ "$S" = Failed ] && rc=1; break;; esac
    [ $(( $(date +%s) - t0 )) -gt 3600 ] && { log "TIMEOUT $J"; rc=1; break; }
    sleep 20
  done
  grep -E '^\s+34359738368' "$OUT" | awk '{print "  32GiB busbw(oop)="$8" algbw="$7}'; grep 'Avg bus bandwidth' "$OUT" || true
  teardown "$F"; return $rc
}

# Usage: run_campaign.sh [START_STAGE] [ATTACH]  -- ATTACH=1: START_STAGE's job is already running.
# (No set -e: grep -c exits 1 on zero matches, which killed v1 of this script mid-poll.)
START=${1:-1}; ATTACH=${2:-0}
stage() {  # $1 n, rest = command; on failure abort campaign with non-zero exit
  local n=$1; shift
  [ "$n" -lt "$START" ] && return 0
  local launch=1; [ "$n" = "$START" ] && [ "$ATTACH" = 1 ] && launch=0
  "$@" $launch || { log "CAMPAIGN ABORTED at stage $n"; exit 1; }
}
stage 1 train qwen72b-c66 $Y/qwen-72b-c66.yaml ~/qwen-c66-run1.log
stage 2 train qwen72b-c66 $Y/qwen-72b-c66.yaml ~/qwen-c66-run2.log
stage 3 train gptoss-c66 $Y/gpt-oss-120b-c66.yaml ~/gpt-oss-c66-run1.log
stage 4 train gptoss-c66 $Y/gpt-oss-120b-c66.yaml ~/gpt-oss-c66-run2.log
stage 5 nccl nccl-12n-c66-2509 $Y/nccl-12n-c66-2509.yaml ~/nccl-c66-12n-2509.log
log "CAMPAIGN DONE"
