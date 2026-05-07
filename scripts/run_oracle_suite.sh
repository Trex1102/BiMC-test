#!/usr/bin/env bash
set -euo pipefail

GROUP="${1:-main}"

run_job() {
  local data_cfg="$1"
  local train_cfg="$2"
  echo "=== $(date '+%F %T') | ${data_cfg} | ${train_cfg} ==="
  bash scripts/run_experiment.sh "${data_cfg}" "${train_cfg}"
}

run_main() {
  for dataset in cifar100_adaptive miniimagenet_adaptive cub200_adaptive; do
    run_job "configs/datasets/${dataset}.yaml" "configs/trainers/bimc.yaml"
    run_job "configs/datasets/${dataset}.yaml" "configs/trainers/bimc_ensemble.yaml"
    run_job "configs/datasets/${dataset}.yaml" "configs/trainers/bimc_adaptive_risk_fixed.yaml"
    run_job "configs/datasets/${dataset}.yaml" "configs/trainers/bimc_oracle_beta.yaml"
    run_job "configs/datasets/${dataset}.yaml" "configs/trainers/bimc_oracle_delta.yaml"
    run_job "configs/datasets/${dataset}.yaml" "configs/trainers/bimc_oracle_tau.yaml"
    run_job "configs/datasets/${dataset}.yaml" "configs/trainers/bimc_oracle_joint.yaml"
    run_job "configs/datasets/${dataset}.yaml" "configs/trainers/bimc_oracle_joint_ensemble.yaml"
  done
}

case "${GROUP}" in
  main)
    run_main
    ;;
  *)
    echo "Unknown group: ${GROUP}" >&2
    exit 1
    ;;
esac
