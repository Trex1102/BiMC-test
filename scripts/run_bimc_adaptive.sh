#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="${1:?usage: scripts/run_bimc_adaptive.sh <cifar100|miniimagenet|cub200> [train_cfg]}"
TRAIN_CFG="${2:-configs/trainers/bimc_adaptive.yaml}"

case "${DATASET_NAME}" in
  cifar100)
    DATA_CFG="configs/datasets/cifar100_adaptive.yaml"
    ;;
  miniimagenet)
    DATA_CFG="configs/datasets/miniimagenet_adaptive.yaml"
    ;;
  cub200)
    DATA_CFG="configs/datasets/cub200_adaptive.yaml"
    ;;
  *)
    echo "Unknown dataset: ${DATASET_NAME}" >&2
    exit 1
    ;;
esac

PYTHONUNBUFFERED=1 python main.py --data_cfg "${DATA_CFG}" --train_cfg "${TRAIN_CFG}"
