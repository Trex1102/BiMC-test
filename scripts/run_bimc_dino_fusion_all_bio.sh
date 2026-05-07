#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_CFG="${TRAIN_CFG:-configs/trainers/bimc_dino_fusion.yaml}"

if [[ -n "${SEEDS:-}" ]]; then
  read -r -a SEED_LIST <<< "${SEEDS}"
else
  SEED_LIST=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
fi

DATA_CFGS=(
  "configs/datasets/cifar100.yaml"
  "configs/datasets/miniimagenet.yaml"
  "configs/datasets/cub200_bimc_dino_fusion.yaml"
)

if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/.anaconda/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/.anaconda/etc/profile.d/conda.sh"
else
  source ~/.bashrc
fi

conda activate fscil-env
cd "${REPO_ROOT}"

RUN_ID="$(date '+%Y%m%d-%H%M%S')"
LOG_DIR="logs/fscil-test-${RUN_ID}"
mkdir -p "${LOG_DIR}"
STATUS_FILE="${LOG_DIR}/status.tsv"
MASTER_LOG="${LOG_DIR}/master.log"
printf 'phase\tdataset\tseed\tstatus\tstart\tend\tduration_sec\tlog\n' > "${STATUS_FILE}"

TOTAL_BASELINE=$((${#DATA_CFGS[@]} * ${#SEED_LIST[@]}))
TOTAL_CPR=${TOTAL_BASELINE}
TOTAL_RUNS=$((TOTAL_BASELINE + TOTAL_CPR))
COMPLETED=0
RUN_START_EPOCH=$(date +%s)

log_master() {
  echo "$*" | tee -a "${MASTER_LOG}"
}

eta_line() {
  local now elapsed avg remaining eta_epoch eta_text
  now=$(date +%s)
  elapsed=$((now - RUN_START_EPOCH))
  if [[ ${COMPLETED} -le 0 ]]; then
    return 0
  fi
  avg=$((elapsed / COMPLETED))
  remaining=$((TOTAL_RUNS - COMPLETED))
  eta_epoch=$((now + avg * remaining))
  eta_text=$(date -d "@${eta_epoch}" '+%F %T %Z')
  log_master "ETA after ${COMPLETED}/${TOTAL_RUNS}: ${eta_text} (avg ${avg}s/run, remaining ${remaining})"
}

seeded_train_cfg() {
  local seed="$1"
  local tmp
  tmp=$(mktemp "${LOG_DIR}/train_seed${seed}.XXXX.yaml")
  sed "s/^SEED:.*/SEED: ${seed}/" "${TRAIN_CFG}" > "${tmp}"
  echo "${tmp}"
}

run_one() {
  local phase="$1"
  local data_cfg="$2"
  local seed="$3"
  shift 3
  local dataset log start_text end_text start_epoch end_epoch duration status
  dataset=$(basename "${data_cfg}" .yaml)
  log="${LOG_DIR}/${phase}_${dataset}_seed${seed}.log"
  start_text=$(date '+%F %T %Z')
  start_epoch=$(date +%s)
  log_master "=== START ${phase} dataset=${dataset} seed=${seed} at ${start_text} ==="
  set +e
  PYTHONUNBUFFERED=1 "$@" 2>&1 | tee "${log}"
  status=${PIPESTATUS[0]}
  set -e
  end_text=$(date '+%F %T %Z')
  end_epoch=$(date +%s)
  duration=$((end_epoch - start_epoch))
  if [[ ${status} -eq 0 ]]; then
    printf '%s\t%s\t%s\tOK\t%s\t%s\t%s\t%s\n' "${phase}" "${dataset}" "${seed}" "${start_text}" "${end_text}" "${duration}" "${log}" >> "${STATUS_FILE}"
    log_master "=== END ${phase} dataset=${dataset} seed=${seed} status=OK duration=${duration}s at ${end_text} ==="
  else
    printf '%s\t%s\t%s\tFAIL(%s)\t%s\t%s\t%s\t%s\n' "${phase}" "${dataset}" "${seed}" "${status}" "${start_text}" "${end_text}" "${duration}" "${log}" >> "${STATUS_FILE}"
    log_master "=== END ${phase} dataset=${dataset} seed=${seed} status=FAIL(${status}) duration=${duration}s at ${end_text} ==="
    exit "${status}"
  fi
  COMPLETED=$((COMPLETED + 1))
  eta_line
}

log_master "Run id: ${RUN_ID}"
log_master "Remote repo: ${REPO_ROOT}"
log_master "Train cfg: ${TRAIN_CFG}"
log_master "Datasets: ${DATA_CFGS[*]}"
log_master "Seeds: ${SEED_LIST[*]}"
log_master "Total runs: ${TOTAL_RUNS} (${TOTAL_BASELINE} baseline + ${TOTAL_CPR} CPR)"
log_master "Started: $(date '+%F %T %Z')"

log_master "--- Phase 1: BiMC DinoV2/CLIP visual fusion baseline (omega=0.7) ---"
for data_cfg in "${DATA_CFGS[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    tmp_cfg=$(seeded_train_cfg "${seed}")
    run_one "baseline" "${data_cfg}" "${seed}" python main.py --data_cfg "${data_cfg}" --train_cfg "${tmp_cfg}"
  done
done

log_master "--- Phase 2: conservative query prototype refinement on top of DinoV2 fusion ---"
for data_cfg in "${DATA_CFGS[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    run_one "cpr" "${data_cfg}" "${seed}" python scripts/prototype_refinement_conservative_dino_fusion.py \
      --data_cfg "${data_cfg}" \
      --train_cfg "${TRAIN_CFG}" \
      --seed_override "${seed}" \
      --alpha_grid 0.5 0.75 1.0 \
      --mass_thr 0.3 \
      --min_count 5
  done
done

finish_text=$(date '+%F %T %Z')
log_master "ALL_DONE ${finish_text}"
log_master "Status file: ${STATUS_FILE}"
