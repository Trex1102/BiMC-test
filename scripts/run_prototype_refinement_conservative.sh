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

PYTHONUNBUFFERED=1 python scripts/prototype_refinement_conservative.py --data_cfg "${DATA_CFG}" --train_cfg "${TRAIN_CFG}" "$@"
