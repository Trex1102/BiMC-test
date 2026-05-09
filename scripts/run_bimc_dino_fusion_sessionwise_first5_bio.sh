#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_CFG="${TRAIN_CFG:-configs/trainers/bimc_dino_fusion.yaml}"

if [[ -n "${SEEDS:-}" ]]; then
  read -r -a SEED_LIST <<< "${SEEDS}"
else
  SEED_LIST=(1 2 3 4 5)
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
LOG_DIR="logs/fscil-sessionwise-pseudoval-cpr-${RUN_ID}"
mkdir -p "${LOG_DIR}"
STATUS_FILE="${LOG_DIR}/status.tsv"
RESULTS_INDEX="${LOG_DIR}/results_index.tsv"
MASTER_LOG="${LOG_DIR}/master.log"
printf 'phase\tdataset\tseed\tstatus\tstart\tend\tduration_sec\tlog\tresults_json\n' > "${STATUS_FILE}"
printf 'dataset\tseed\tresults_json\n' > "${RESULTS_INDEX}"

TOTAL_RUNS=$((${#DATA_CFGS[@]} * ${#SEED_LIST[@]}))
COMPLETED=0
ACTUAL_COMPLETED=0
RUN_DURATION_SUM=0
RUN_START_EPOCH=$(date +%s)
shopt -s nullglob

log_master() {
  echo "$*" | tee -a "${MASTER_LOG}"
}

dataset_name() {
  local data_cfg="$1"
  local base
  base="$(basename "${data_cfg}" .yaml)"
  case "${base}" in
    cifar100) echo "CIFAR100" ;;
    cub200_bimc_dino_fusion) echo "cub200" ;;
    *) echo "${base}" ;;
  esac
}

latest_result_json() {
  local dataset="$1"
  local seed="$2"
  local files=(
    "experiments/${dataset}/prototype_refinement_conservative_pseudoval_sessionwise_bimc_dino_fusion/seed${seed}_"*/results.json
  )
  if [[ ${#files[@]} -eq 0 ]]; then
    return 0
  fi
  ls -t "${files[@]}" | head -n 1
}

eta_line() {
  local now elapsed avg remaining eta_epoch eta_text
  now=$(date +%s)
  if [[ ${ACTUAL_COMPLETED} -le 0 ]]; then
    return 0
  fi
  avg=$((RUN_DURATION_SUM / ACTUAL_COMPLETED))
  remaining=$((TOTAL_RUNS - COMPLETED))
  eta_epoch=$((now + avg * remaining))
  eta_text=$(date -d "@${eta_epoch}" '+%F %T %Z')
  log_master "ETA after ${COMPLETED}/${TOTAL_RUNS}: ${eta_text} (avg ${avg}s/actual-run, remaining ${remaining})"
}

aggregate_results() {
  python - "${RESULTS_INDEX}" "${LOG_DIR}" <<'PY' | tee -a "${MASTER_LOG}"
import json
import sys
from collections import defaultdict
from pathlib import Path

index_path = Path(sys.argv[1])
log_dir = Path(sys.argv[2])
records = []
with index_path.open("r", encoding="utf-8") as f:
    next(f, None)
    for line in f:
        dataset, seed, result = line.rstrip("\n").split("\t")
        records.append((dataset, int(seed), Path(result)))

by_dataset = defaultdict(list)
for dataset, seed, result in records:
    with result.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    by_dataset[dataset].append((seed, payload))

summary = {}
for dataset, items in sorted(by_dataset.items()):
    items.sort(key=lambda x: x[0])
    baseline = [item[1]["baseline_acc_list"] for item in items]
    final = [item[1]["final_acc_list"] for item in items]
    width = len(baseline[0])

    def mean_curve(curves):
        return [round(sum(curve[i] for curve in curves) / len(curves), 3) for i in range(width)]

    baseline_curve = mean_curve(baseline)
    final_curve = mean_curve(final)
    gain_curve = [round(final_curve[i] - baseline_curve[i], 3) for i in range(width)]
    summary[dataset] = {
        "seeds": [seed for seed, _ in items],
        "baseline_acc_list": baseline_curve,
        "final_acc_list": final_curve,
        "gain_list": gain_curve,
        "baseline_avg": round(sum(baseline_curve) / len(baseline_curve), 3),
        "final_avg": round(sum(final_curve) / len(final_curve), 3),
        "gain_avg": round(sum(gain_curve) / len(gain_curve), 3),
        "baseline_pd": round(baseline_curve[0] - baseline_curve[-1], 3),
        "final_pd": round(final_curve[0] - final_curve[-1], 3),
    }

json_path = log_dir / "aggregate_first5_sessionwise.json"
txt_path = log_dir / "aggregate_first5_sessionwise.txt"
json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
lines = []
for dataset, payload in summary.items():
    lines.extend([
        f"dataset: {dataset}",
        f"seeds: {payload['seeds']}",
        f"baseline_acc_list: {payload['baseline_acc_list']}",
        f"final_acc_list: {payload['final_acc_list']}",
        f"gain_list: {payload['gain_list']}",
        f"baseline_avg: {payload['baseline_avg']}",
        f"final_avg: {payload['final_avg']}",
        f"gain_avg: {payload['gain_avg']}",
        f"baseline_pd: {payload['baseline_pd']}",
        f"final_pd: {payload['final_pd']}",
        "",
    ])
txt_path.write_text("\n".join(lines), encoding="utf-8")
print("Aggregate results written to:")
print(f"  {json_path}")
print(f"  {txt_path}")
print(txt_path.read_text(encoding="utf-8"))
PY
}

run_one() {
  local data_cfg="$1"
  local seed="$2"
  local dataset log start_text end_text start_epoch end_epoch duration status result_json
  dataset="$(dataset_name "${data_cfg}")"
  log="${LOG_DIR}/sessionwise_pseudoval_cpr_${dataset}_seed${seed}.log"
  result_json="$(latest_result_json "${dataset}" "${seed}")"
  if [[ "${SKIP_COMPLETED:-1}" == "1" && -n "${result_json}" ]]; then
    start_text=$(date '+%F %T %Z')
    printf '%s\t%s\t%s\tSKIP_EXISTING\t%s\t%s\t%s\t%s\t%s\n' "sessionwise_pseudoval_cpr" "${dataset}" "${seed}" "${start_text}" "${start_text}" "0" "" "${result_json}" >> "${STATUS_FILE}"
    printf '%s\t%s\t%s\n' "${dataset}" "${seed}" "${result_json}" >> "${RESULTS_INDEX}"
    log_master "=== SKIP sessionwise_pseudoval_cpr dataset=${dataset} seed=${seed} existing=${result_json} ==="
    COMPLETED=$((COMPLETED + 1))
    eta_line
    return 0
  fi
  start_text=$(date '+%F %T %Z')
  start_epoch=$(date +%s)
  log_master "=== START sessionwise_pseudoval_cpr dataset=${dataset} seed=${seed} at ${start_text} ==="
  set +e
  PYTHONUNBUFFERED=1 python scripts/prototype_refinement_conservative_dino_fusion_sessionwise.py \
    --data_cfg "${data_cfg}" \
    --train_cfg "${TRAIN_CFG}" \
    --seed_override "${seed}" \
    --alpha_grid 0.5 0.75 1.0 \
    --mass_thr 0.3 \
    --min_count 5 \
    --val_fraction 0.2 \
    --base_val_conf_thr 0.5 \
    --base_val_max_per_class 50 \
    --fallback_alpha 0.75 \
    2>&1 | tee "${log}"
  status=${PIPESTATUS[0]}
  set -e
  end_text=$(date '+%F %T %Z')
  end_epoch=$(date +%s)
  duration=$((end_epoch - start_epoch))
  result_json="$(latest_result_json "${dataset}" "${seed}")"
  if [[ ${status} -eq 0 ]]; then
    printf '%s\t%s\t%s\tOK\t%s\t%s\t%s\t%s\t%s\n' "sessionwise_pseudoval_cpr" "${dataset}" "${seed}" "${start_text}" "${end_text}" "${duration}" "${log}" "${result_json}" >> "${STATUS_FILE}"
    printf '%s\t%s\t%s\n' "${dataset}" "${seed}" "${result_json}" >> "${RESULTS_INDEX}"
    log_master "=== END sessionwise_pseudoval_cpr dataset=${dataset} seed=${seed} status=OK duration=${duration}s at ${end_text} ==="
  else
    printf '%s\t%s\t%s\tFAIL(%s)\t%s\t%s\t%s\t%s\t%s\n' "sessionwise_pseudoval_cpr" "${dataset}" "${seed}" "${status}" "${start_text}" "${end_text}" "${duration}" "${log}" "${result_json}" >> "${STATUS_FILE}"
    log_master "=== END sessionwise_pseudoval_cpr dataset=${dataset} seed=${seed} status=FAIL(${status}) duration=${duration}s at ${end_text} ==="
    exit "${status}"
  fi
  COMPLETED=$((COMPLETED + 1))
  ACTUAL_COMPLETED=$((ACTUAL_COMPLETED + 1))
  RUN_DURATION_SUM=$((RUN_DURATION_SUM + duration))
  eta_line
}

log_master "Run id: ${RUN_ID}"
log_master "Remote repo: ${REPO_ROOT}"
log_master "Train cfg: ${TRAIN_CFG}"
log_master "Datasets: ${DATA_CFGS[*]}"
log_master "Seeds: ${SEED_LIST[*]}"
log_master "Total runs: ${TOTAL_RUNS} session-wise pseudo-validation CPR runs"
log_master "Started: $(date '+%F %T %Z')"

for data_cfg in "${DATA_CFGS[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    run_one "${data_cfg}" "${seed}"
  done
done

aggregate_results
finish_text=$(date '+%F %T %Z')
log_master "ALL_DONE ${finish_text}"
log_master "Status file: ${STATUS_FILE}"
log_master "Results index: ${RESULTS_INDEX}"
