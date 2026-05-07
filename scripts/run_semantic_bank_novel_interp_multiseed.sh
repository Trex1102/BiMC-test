#!/usr/bin/env bash
set -euo pipefail

TRAIN_CFG="${1:-configs/trainers/bimc_ensemble.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

DATA_CFG="${1:-configs/datasets/cifar100_adaptive.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

ALPHAS=("${1:-0.25}" "${2:-0.75}")
if [[ $# -ge 2 ]]; then
  shift 2
fi

SEEDS=(2 3 4 5)
for seed in "${SEEDS[@]}"; do
  echo "=== $(date '+%F %T') | seed=${seed} | alphas=${ALPHAS[*]} ==="
  PYTHONUNBUFFERED=1 python scripts/semantic_bank_novel_interp_audit.py \
    --data_cfg "${DATA_CFG}" \
    --train_cfg "${TRAIN_CFG}" \
    --alphas "${ALPHAS[@]}" \
    --seed_override "${seed}" \
    "$@"
done
