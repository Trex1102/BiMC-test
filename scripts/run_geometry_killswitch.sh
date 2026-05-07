#!/usr/bin/env bash
set -euo pipefail

TRAIN_CFG="${1:-configs/trainers/bimc_ensemble.yaml}"

for data_cfg in \
  configs/datasets/cifar100_adaptive.yaml \
  configs/datasets/miniimagenet_adaptive.yaml \
  configs/datasets/cub200_adaptive.yaml; do
  echo "=== $(date '+%F %T') | ${data_cfg} | ${TRAIN_CFG} ==="
  PYTHONUNBUFFERED=1 python scripts/geometry_killswitch.py --data_cfg "${data_cfg}" --train_cfg "${TRAIN_CFG}"
done
