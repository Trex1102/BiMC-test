#!/usr/bin/env bash
set -euo pipefail

TRAIN_CFG="${1:-configs/trainers/bimc_ensemble.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

DATA_CFG="${1:-configs/datasets/cub200.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

PYTHONUNBUFFERED=1 python scripts/semantic_bank_novel_composer_audit.py --data_cfg "${DATA_CFG}" --train_cfg "${TRAIN_CFG}" "$@"
