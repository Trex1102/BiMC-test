#!/usr/bin/env bash
set -euo pipefail

for data_cfg in \
  configs/datasets/cifar100_adaptive.yaml \
  configs/datasets/miniimagenet_adaptive.yaml \
  configs/datasets/cub200_adaptive.yaml; do
  for train_cfg in \
    configs/trainers/bimc_adaptive_risk_pseudoprior.yaml \
    configs/trainers/bimc_adaptive_risk_margin.yaml \
    configs/trainers/bimc_adaptive_risk_pseudoprior_margin.yaml; do
    echo "=== $(date '+%F %T') | ${data_cfg} | ${train_cfg} ==="
    bash scripts/run_experiment.sh "${data_cfg}" "${train_cfg}"
  done
done
