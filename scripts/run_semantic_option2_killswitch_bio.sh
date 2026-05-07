#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_CFG="${1:-configs/trainers/bimc_ensemble.yaml}"

if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/.anaconda/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/.anaconda/etc/profile.d/conda.sh"
else
  source ~/.bashrc
fi

conda activate fscil-env
cd "${REPO_ROOT}"

bash scripts/run_semantic_option2_killswitch.sh "${TRAIN_CFG}"
