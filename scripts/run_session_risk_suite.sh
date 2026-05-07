#!/usr/bin/env bash
set -euo pipefail

GROUP="${1:-all}"

run_job() {
  local data_cfg="$1"
  local train_cfg="$2"
  echo "=== $(date '+%F %T') | ${data_cfg} | ${train_cfg} ==="
  bash scripts/run_experiment.sh "${data_cfg}" "${train_cfg}"
}

run_main() {
  for dataset in cifar100_adaptive miniimagenet_adaptive cub200_adaptive; do
    run_job "configs/datasets/${dataset}.yaml" "configs/trainers/bimc.yaml"
    run_job "configs/datasets/${dataset}.yaml" "configs/trainers/bimc_adaptive_risk_fixed.yaml"
    run_job "configs/datasets/${dataset}.yaml" "configs/trainers/bimc_adaptive_risk_baseauto.yaml"
    run_job "configs/datasets/${dataset}.yaml" "configs/trainers/bimc_adaptive_risk_universal.yaml"
    run_job "configs/datasets/${dataset}.yaml" "configs/trainers/bimc_ensemble.yaml"
    run_job "configs/datasets/${dataset}.yaml" "configs/trainers/bimc_adaptive_risk_baseauto_ensemble.yaml"
  done
}

run_stress() {
  for dataset in cifar100_noise30 miniimagenet_noise30 cub200_noise30; do
    run_job "configs/datasets/${dataset}.yaml" "configs/trainers/bimc.yaml"
    run_job "configs/datasets/${dataset}.yaml" "configs/trainers/bimc_adaptive_risk_baseauto.yaml"
  done

  for dataset in cifar100_1shot miniimagenet_1shot cub200_1shot; do
    run_job "configs/datasets/${dataset}.yaml" "configs/trainers/bimc.yaml"
    run_job "configs/datasets/${dataset}.yaml" "configs/trainers/bimc_adaptive_risk_baseauto.yaml"
  done

  run_job "configs/datasets/miniimagenet_lowres32.yaml" "configs/trainers/bimc.yaml"
  run_job "configs/datasets/miniimagenet_lowres32.yaml" "configs/trainers/bimc_adaptive_risk_baseauto.yaml"
}

case "${GROUP}" in
  main)
    run_main
    ;;
  stress)
    run_stress
    ;;
  all)
    run_main
    run_stress
    ;;
  *)
    echo "Unknown group: ${GROUP}" >&2
    exit 1
    ;;
esac
