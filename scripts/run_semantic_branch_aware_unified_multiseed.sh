#!/usr/bin/env bash
set -euo pipefail

TRAIN_CFG="${1:-configs/trainers/bimc_ensemble.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

DATA_CFG="${1:-configs/datasets/cifar100.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

SEEDS=(1 2 3 4 5)
for seed in "${SEEDS[@]}"; do
  echo "=== $(date '+%F %T') | seed=${seed} ==="
  PYTHONUNBUFFERED=1 python scripts/semantic_branch_aware_unified_audit.py \
    --data_cfg "${DATA_CFG}" \
    --train_cfg "${TRAIN_CFG}" \
    --seed_override "${seed}" \
    "$@"
done
