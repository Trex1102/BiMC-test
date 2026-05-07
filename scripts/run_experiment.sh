#!/usr/bin/env bash
set -euo pipefail

DATA_CFG="${1:?usage: scripts/run_experiment.sh <data_cfg> <train_cfg>}"
TRAIN_CFG="${2:?usage: scripts/run_experiment.sh <data_cfg> <train_cfg>}"

PYTHONUNBUFFERED=1 python main.py --data_cfg "${DATA_CFG}" --train_cfg "${TRAIN_CFG}"
