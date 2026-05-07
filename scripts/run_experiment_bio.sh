#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_CFG="${1:?usage: scripts/run_experiment_bio.sh <data_cfg> <train_cfg>}"
TRAIN_CFG="${2:?usage: scripts/run_experiment_bio.sh <data_cfg> <train_cfg>}"

if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/.anaconda/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/.anaconda/etc/profile.d/conda.sh"
else
  source ~/.bashrc
fi

conda activate fscil-env
cd "${REPO_ROOT}"

bash scripts/run_experiment.sh "${DATA_CFG}" "${TRAIN_CFG}"
