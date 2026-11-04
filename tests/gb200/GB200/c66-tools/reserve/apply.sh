#!/usr/bin/env bash
# Reserve all 12 GB200 nodes for namespace 'infra-perf'.
set -euo pipefail
D=$(dirname "$0")
K="sudo kubectl --kubeconfig /etc/kubernetes/admin.conf"
NODES=$($K get nodes -l nvidia.com/gpu.present=true -o jsonpath='{.items[*].metadata.name}')

$K create namespace infra-perf --dry-run=client -o yaml | $K apply -f -
$K apply -f "$D/policy.yaml"
$K label nodes $NODES reserved-for=infra-perf --overwrite
$K taint nodes $NODES reserved-for=infra-perf:NoSchedule --overwrite

$K get nodes -o custom-columns='NAME:.metadata.name,LABEL:.metadata.labels.reserved-for,TAINTS:.spec.taints[*].key'

# Rollback:
#   $K taint nodes $NODES reserved-for-
#   $K label nodes $NODES reserved-for-
#   $K delete -f "$D/policy.yaml"
#   $K delete namespace infra-perf
