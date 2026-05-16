#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-tric_table467_$(date '+%Y%m%d-%H%M%S')}"
EXP_ROOT="experiments/${RUN_ID}"
LOG_DIR="${EXP_ROOT}/logs"
RESULT_DIR="${EXP_ROOT}/branch_audit"
STATUS_FILE="${LOG_DIR}/status.tsv"
RESULTS_INDEX="${LOG_DIR}/results_index.tsv"
MASTER_LOG="${LOG_DIR}/master.log"
TRAIN_CFG="${TRAIN_CFG:-configs/trainers/bimc_dino_fusion.yaml}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20}"

DATA_CFGS=(
  "configs/datasets/cifar100.yaml"
  "configs/datasets/miniimagenet.yaml"
  "configs/datasets/cub200_bimc_dino_fusion.yaml"
)

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"
printf 'phase\tdataset\tseed\tstatus\tstart\tend\tduration_sec\tlog\tresults_json\tparams\n' > "${STATUS_FILE}"
printf 'phase\tdataset\tseed\tresults_json\tparams\n' > "${RESULTS_INDEX}"

log_master() {
  echo "$*" | tee -a "${MASTER_LOG}"
}

dataset_name() {
  local base
  base="$(basename "$1" .yaml)"
  case "${base}" in
    cifar100) echo "CIFAR100" ;;
    cub200_bimc_dino_fusion) echo "cub200" ;;
    *) echo "${base}" ;;
  esac
}

run_one() {
  local data_cfg="$1"
  local seed="$2"
  local dataset log result_json start_text end_text start_epoch end_epoch duration status
  dataset="$(dataset_name "${data_cfg}")"
  log="${LOG_DIR}/table467_${dataset}_seed${seed}.log"
  result_json="${RESULT_DIR}/${dataset}_seed${seed}.json"
  start_text="$(date '+%F %T %Z')"
  start_epoch="$(date +%s)"
  log_master "=== START table467 dataset=${dataset} seed=${seed} at ${start_text} ==="
  set +e
  PYTHONUNBUFFERED=1 python scripts/tric_table467_audit.py \
    --data_cfg "${data_cfg}" \
    --train_cfg "${TRAIN_CFG}" \
    --seed "${seed}" \
    --output_root "${EXP_ROOT}" \
    --output "${result_json}" \
    2>&1 | tee "${log}"
  status=${PIPESTATUS[0]}
  set -e
  end_text="$(date '+%F %T %Z')"
  end_epoch="$(date +%s)"
  duration=$((end_epoch - start_epoch))
  if [[ ${status} -eq 0 && ! -s "${result_json}" ]]; then
    status=97
  fi
  if [[ ${status} -eq 0 ]]; then
    printf '%s\t%s\t%s\tOK\t%s\t%s\t%s\t%s\t%s\t%s\n' "table467" "${dataset}" "${seed}" "${start_text}" "${end_text}" "${duration}" "${log}" "${result_json}" "variants=semantic,clip,dino,clip_dino,tric_no_mask,tric" >> "${STATUS_FILE}"
    printf '%s\t%s\t%s\t%s\t%s\n' "table467" "${dataset}" "${seed}" "${result_json}" "variants=semantic,clip,dino,clip_dino,tric_no_mask,tric" >> "${RESULTS_INDEX}"
    log_master "=== END table467 dataset=${dataset} seed=${seed} status=OK duration=${duration}s result=${result_json} at ${end_text} ==="
  else
    printf '%s\t%s\t%s\tFAIL(%s)\t%s\t%s\t%s\t%s\t%s\t%s\n' "table467" "${dataset}" "${seed}" "${status}" "${start_text}" "${end_text}" "${duration}" "${log}" "${result_json}" "variants=semantic,clip,dino,clip_dino,tric_no_mask,tric" >> "${STATUS_FILE}"
    log_master "=== END table467 dataset=${dataset} seed=${seed} status=FAIL(${status}) duration=${duration}s at ${end_text} ==="
    exit "${status}"
  fi
}

aggregate_results() {
  python - "${RESULTS_INDEX}" "${EXP_ROOT}" <<'PY' | tee -a "${MASTER_LOG}"
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

index = Path(sys.argv[1])
root = Path(sys.argv[2])
records = []
with index.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        records.append((row["dataset"], int(row["seed"]), Path(row["results_json"])))

variant_rows = defaultdict(list)
outcome_rows = defaultdict(list)
for dataset, seed, result_path in records:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    for variant, data in payload["variants"].items():
        variant_rows[(dataset, variant)].append((seed, data["acc_list"], data["avg"], data["pd"]))
    for split, data in payload["classifier_outcome"].items():
        outcome_rows[(dataset, split)].append((seed, data["patterns_sem_visual_tric"]))

def std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0

variant_summary = []
for (dataset, variant), items in sorted(variant_rows.items()):
    items.sort(key=lambda x: x[0])
    width = len(items[0][1])
    session_values = [[acc[i] for _, acc, _, _ in items] for i in range(width)]
    avg_values = [avg for _, _, avg, _ in items]
    pd_values = [pd for _, _, _, pd in items]
    variant_summary.append({
        "dataset": dataset,
        "variant": variant,
        "seeds": [seed for seed, _, _, _ in items],
        "n": len(items),
        "session_mean": [round(statistics.mean(values), 4) for values in session_values],
        "session_std": [round(std(values), 4) for values in session_values],
        "avg_mean": round(statistics.mean(avg_values), 4),
        "avg_std": round(std(avg_values), 4),
        "pd_mean": round(statistics.mean(pd_values), 4),
        "pd_std": round(std(pd_values), 4),
        "final_mean": round(statistics.mean(session_values[-1]), 4),
        "final_std": round(std(session_values[-1]), 4),
    })

outcome_summary = []
for (dataset, split), items in sorted(outcome_rows.items()):
    keys = sorted(items[0][1])
    outcome_summary.append({
        "dataset": dataset,
        "split": split,
        "seeds": [seed for seed, _ in sorted(items)],
        "patterns_sem_visual_tric": {
            key: round(statistics.mean(patterns[key] for _, patterns in items), 4)
            for key in keys
        },
        "patterns_sem_visual_tric_std": {
            key: round(std([patterns[key] for _, patterns in items]), 4)
            for key in keys
        },
    })

payload = {
    "variant_summary": variant_summary,
    "classifier_outcome_summary": outcome_summary,
    "records": [
        {"dataset": dataset, "seed": seed, "result": str(path)}
        for dataset, seed, path in records
    ],
}
(root / "aggregate_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

lines = ["Table 4/6/7 aggregate", "====================", ""]
for row in variant_summary:
    lines.append(f"{row['dataset']} | {row['variant']} | n={row['n']}")
    lines.append(f"  session_mean: {[round(x, 2) for x in row['session_mean']]}")
    lines.append(f"  session_std: {[round(x, 2) for x in row['session_std']]}")
    lines.append(f"  avg: {row['avg_mean']:.2f} +- {row['avg_std']:.2f}")
    lines.append(f"  pd: {row['pd_mean']:.2f} +- {row['pd_std']:.2f}")
    lines.append("")
lines.append("Classifier outcome patterns")
lines.append("Pattern key order is semantic, CLIP-DINO visual, TriC; 1=correct, 0=wrong.")
for row in outcome_summary:
    lines.append(f"{row['dataset']} | {row['split']}")
    for key in sorted(row["patterns_sem_visual_tric"], reverse=True):
        lines.append(f"  {key}: {row['patterns_sem_visual_tric'][key]:.2f} +- {row['patterns_sem_visual_tric_std'][key]:.2f}")
    lines.append("")
(root / "aggregate_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print((root / "aggregate_summary.txt").read_text(encoding="utf-8"))
PY
}

if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/.anaconda/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/.anaconda/etc/profile.d/conda.sh"
else
  source "${HOME}/.bashrc"
fi
conda activate fscil-env

read -r -a SEED_LIST <<< "${SEEDS}"
log_master "RUN_STARTED $(date '+%F %T %Z')"
log_master "Run id: ${RUN_ID}"
log_master "Experiment root: ${REPO_ROOT}/${EXP_ROOT}"
log_master "Train cfg: ${TRAIN_CFG}"
log_master "Datasets: ${DATA_CFGS[*]}"
log_master "Seeds: ${SEEDS}"
log_master "Variants: semantic branch only, CLIP visual only, DINO visual only, CLIP-DINO visual only, TriC without masked ensemble, TriC"

for seed in "${SEED_LIST[@]}"; do
  for data_cfg in "${DATA_CFGS[@]}"; do
    run_one "${data_cfg}" "${seed}"
  done
done

aggregate_results
log_master "ALL_DONE $(date '+%F %T %Z')"
log_master "Experiment root: ${REPO_ROOT}/${EXP_ROOT}"
log_master "Status file: ${REPO_ROOT}/${STATUS_FILE}"
log_master "Results index: ${REPO_ROOT}/${RESULTS_INDEX}"
log_master "Aggregate summary: ${REPO_ROOT}/${EXP_ROOT}/aggregate_summary.txt"
