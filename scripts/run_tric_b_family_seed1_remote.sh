#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-tric_b_family_tric_seed1_$(date '+%Y%m%d-%H%M%S')}"
EXP_ROOT="experiments/${RUN_ID}"
LOG_DIR="${EXP_ROOT}/logs"
MASTER_LOG="${LOG_DIR}/master.log"
STATUS_FILE="${LOG_DIR}/status.tsv"

mkdir -p "${LOG_DIR}"
printf 'phase\tstatus\tstart\tend\tduration_sec\tlog\n' > "${STATUS_FILE}"

log_master() {
  echo "$*" | tee -a "${MASTER_LOG}"
}

run_phase() {
  local phase="$1"
  shift
  local log="${LOG_DIR}/${phase}.log"
  local start_text end_text start_epoch end_epoch duration status
  start_text="$(date '+%F %T %Z')"
  start_epoch="$(date +%s)"
  log_master "=== START phase=${phase} at ${start_text} ==="
  set +e
  PYTHONUNBUFFERED=1 "$@" 2>&1 | tee "${log}"
  status=${PIPESTATUS[0]}
  set -e
  end_text="$(date '+%F %T %Z')"
  end_epoch="$(date +%s)"
  duration=$((end_epoch - start_epoch))
  if [[ ${status} -eq 0 ]]; then
    printf '%s\tOK\t%s\t%s\t%s\t%s\n' "${phase}" "${start_text}" "${end_text}" "${duration}" "${log}" >> "${STATUS_FILE}"
    log_master "=== END phase=${phase} status=OK duration=${duration}s at ${end_text} ==="
  else
    printf '%s\tFAIL(%s)\t%s\t%s\t%s\t%s\n' "${phase}" "${status}" "${start_text}" "${end_text}" "${duration}" "${log}" >> "${STATUS_FILE}"
    log_master "=== END phase=${phase} status=FAIL(${status}) duration=${duration}s at ${end_text} ==="
    exit "${status}"
  fi
}

log_master "RUN_ID=${RUN_ID}"
log_master "EXP_ROOT=${EXP_ROOT}"

run_phase "support_subspace_prototype_tric" \
  python scripts/support_subspace_prototype_suite.py \
    --frameworks tric_dinov2 \
    --seed 1 \
    --output_root "${EXP_ROOT}"

run_phase "support_subspace_ablation_tric" \
  python scripts/support_subspace_ablation_suite.py \
    --frameworks tric_dinov2 \
    --seed 1 \
    --output_root "${EXP_ROOT}"

run_phase "tric_nonqpr_followups" \
  python scripts/tric_nonqpr_followup_suite.py \
    --seed 1 \
    --output_root "${EXP_ROOT}"

log_master "All phases completed."
log_master "Primary output root: ${EXP_ROOT}"
