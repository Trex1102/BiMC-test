#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_CFG="${1:-configs/trainers/bimc_ensemble.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

DATA_CFG="${1:-configs/datasets/cifar100.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

cd "${REPO_ROOT}"

python scripts/prototype_refinement_conservative_qpr.py \
  --train_cfg "${TRAIN_CFG}" \
  --data_cfg "${DATA_CFG}" \
  "$@"
