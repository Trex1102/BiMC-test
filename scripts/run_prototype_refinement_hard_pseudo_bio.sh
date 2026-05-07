#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-cifar100}"
if [[ $# -gt 0 ]]; then
  shift
fi

TRAIN_CFG="${1:-configs/trainers/bimc_ensemble.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${DATASET}" in
  cifar100)
    DATA_CFG="configs/datasets/cifar100.yaml"
    ;;
  miniimagenet|mini_imagenet)
    DATA_CFG="configs/datasets/miniimagenet.yaml"
    ;;
  cub200)
    DATA_CFG="configs/datasets/cub200_adaptive.yaml"
    ;;
  *)
    echo "Unknown dataset: ${DATASET}" >&2
    exit 1
    ;;
esac

if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/.anaconda/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/.anaconda/etc/profile.d/conda.sh"
else
  source "${HOME}/.bashrc"
fi

conda activate fscil-env
cd /media/bio/data/BiMC-test
bash scripts/run_prototype_refinement_hard_pseudo.sh "${TRAIN_CFG}" "${DATA_CFG}" "$@"
