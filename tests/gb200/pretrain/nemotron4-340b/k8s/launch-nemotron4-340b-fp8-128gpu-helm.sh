#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../../.." && pwd)

CHART_DIR=${CHART_DIR:-"${REPO_ROOT}/nemotron4-bench"}
WORKLOAD_LAUNCHER=${WORKLOAD_LAUNCHER:-"${CHART_DIR}/launcher.sh"}
WORKLOAD_CONFIG=${WORKLOAD_CONFIG:-"${CHART_DIR}/nemotron4-340b-fp8.py"}
RELEASE_NAME=${RELEASE_NAME:-"nemotron4-340b-128gpu"}
KUBECONFIG_PATH=${KUBECONFIG_PATH:-"${HOME}/.kube/lightricks-poc"}
FIXED_GBS=${FIXED_GBS:-"256"}

require_path() {
    local path="$1"
    local label="$2"

    if [[ ! -e "${path}" ]]; then
        printf 'Missing %s: %s\n' "${label}" "${path}" >&2
        exit 1
    fi
}

require_path "${CHART_DIR}" "chart directory"
require_path "${WORKLOAD_LAUNCHER}" "workload launcher"
require_path "${WORKLOAD_CONFIG}" "workload config"
require_path "${KUBECONFIG_PATH}" "kubeconfig"

helm install "${RELEASE_NAME}" "${CHART_DIR}/" \
  --set-file workload_launcher="${WORKLOAD_LAUNCHER}" \
  --set-file workload_config="${WORKLOAD_CONFIG}" \
  --set workload.gpus=128 \
  --set workload.configFile="$(basename "${WORKLOAD_CONFIG}")" \
  --set-string 'workload.envs[4].name=FIXED_GBS' \
  --set-string "workload.envs[4].value=${FIXED_GBS}" \
  --kubeconfig "${KUBECONFIG_PATH}"
