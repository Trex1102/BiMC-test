#!/usr/bin/env bash
set -euo pipefail

TRAIN_CFG="${1:-configs/trainers/bimc_adaptive.yaml}"

for dataset in cifar100 miniimagenet cub200; do
  echo "=== $(date '+%F %T') | ${dataset} | ${TRAIN_CFG} ==="
  bash scripts/run_bimc_adaptive.sh "${dataset}" "${TRAIN_CFG}"
done
