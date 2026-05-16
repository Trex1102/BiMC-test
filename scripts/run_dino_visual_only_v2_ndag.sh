#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-dino_visual_only_v2_$(date '+%Y%m%d-%H%M%S')}"
EXP_ROOT="experiments/${RUN_ID}"
LOG_DIR="${EXP_ROOT}/logs"
CFG_DIR="${EXP_ROOT}/generated_configs"
STATUS_FILE="${LOG_DIR}/status.tsv"
RESULTS_INDEX="${LOG_DIR}/results_index.tsv"
MASTER_LOG="${LOG_DIR}/master.log"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20}"

DATA_CFGS=(
  "configs/datasets/cifar100.yaml"
  "configs/datasets/miniimagenet.yaml"
  "configs/datasets/cub200_bimc_dino_fusion.yaml"
)

mkdir -p "${LOG_DIR}" "${CFG_DIR}"
if [[ ! -s "${STATUS_FILE}" ]]; then
  printf 'phase\tdataset\tseed\tstatus\tstart\tend\tduration_sec\tlog\tresults_json\tparams\n' > "${STATUS_FILE}"
fi
if [[ ! -s "${RESULTS_INDEX}" ]]; then
  printf 'phase\tdataset\tseed\tresults_json\tparams\n' > "${RESULTS_INDEX}"
fi

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

make_dino_cfg() {
  local dest="$1"
  local seed="$2"
  mkdir -p "$(dirname "${dest}")"
  {
    printf 'OUTPUT:\n'
    printf '  ROOT: "%s"\n' "${EXP_ROOT}"
    printf 'METHOD: "bimc_dino_fusion"\n'
    printf 'DATASET:\n'
    printf '  BETA: 0.0\n'
    printf 'SEED: %s\n' "${seed}"
    printf '\n'
    printf 'DEVICE:\n'
    printf '  DEVICE_NAME: "cuda"\n'
    printf '  GPU_ID: "0;"\n'
    printf '\n'
    printf 'DATALOADER:\n'
    printf '  TRAIN:\n'
    printf '    BATCH_SIZE_BASE: 64\n'
    printf '    BATCH_SIZE_INC: 64\n'
    printf '  TEST:\n'
    printf '    BATCH_SIZE: 100\n'
    printf '  NUM_WORKERS: 4\n'
    printf '\n'
    printf 'MODEL:\n'
    printf '  BACKBONE:\n'
    printf '    NAME: "ViT-B/16"\n'
    printf '\n'
    printf 'TRAINER:\n'
    printf '  BiMC:\n'
    printf '    PREC: "fp16"\n'
    printf '    VISION_CALIBRATION: False\n'
    printf '    LAMBDA_I: 0.1\n'
    printf '    TAU: 16\n'
    printf '    TEXT_CALIBRATION: False\n'
    printf '    LAMBDA_T: 0.0\n'
    printf '    GAMMA_BASE: 1.0\n'
    printf '    GAMMA_INC: 1.0\n'
    printf '    USING_ENSEMBLE: False\n'
  } > "${dest}"
}

ensure_dino_cfg() {
  local seed="$1"
  local dest="${CFG_DIR}/seed${seed}/dino_visual_only.yaml"
  if [[ ! -s "${dest}" ]]; then
    make_dino_cfg "${dest}" "${seed}"
  fi
  printf '%s\n' "${dest}"
}

latest_result_json() {
  local dataset="$1"
  local seed="$2"
  local files=()
  shopt -s nullglob
  files=("${EXP_ROOT}/${dataset}/bimc_dino_fusion/seed${seed}_"*/results.json)
  shopt -u nullglob
  if [[ ${#files[@]} -eq 0 ]]; then
    return 1
  fi
  ls -t "${files[@]}" | head -n 1
}

completed_result_json() {
  local dataset="$1"
  local seed="$2"
  local params="$3"
  awk -F '\t' -v dataset="${dataset}" -v seed="${seed}" -v params="${params}" '
    $1 == "dino_visual_only" && $2 == dataset && $3 == seed && $4 == "OK" && $10 == params { result = $9 }
    END { if (result != "") print result }
  ' "${STATUS_FILE}"
}

run_dino_only() {
  local seed="$1"
  local data_cfg="$2"
  local dataset cfg params log start_text end_text start_epoch end_epoch duration status result_json completed_json
  dataset="$(dataset_name "${data_cfg}")"
  cfg="$(ensure_dino_cfg "${seed}")"
  params="branch=dino_visual_only,beta=0,omega=1,ensemble=false,vision_calibration=false"
  completed_json="$(completed_result_json "${dataset}" "${seed}" "${params}" || true)"
  if [[ -n "${completed_json}" && -f "${completed_json}" ]]; then
    log_master "=== SKIP phase=dino_visual_only dataset=${dataset} seed=${seed} existing=${completed_json} ==="
    return 0
  fi

  log="${LOG_DIR}/dino_visual_only_${dataset}_seed${seed}.log"
  start_text="$(date '+%F %T %Z')"
  start_epoch="$(date +%s)"
  log_master "=== START phase=dino_visual_only dataset=${dataset} seed=${seed} params=${params} at ${start_text} ==="
  set +e
  PYTHONUNBUFFERED=1 python scripts/run_main_with_dino_omega.py --data_cfg "${data_cfg}" --train_cfg "${cfg}" --dino_omega 1.0 2>&1 | tee "${log}"
  status=${PIPESTATUS[0]}
  set -e
  end_text="$(date '+%F %T %Z')"
  end_epoch="$(date +%s)"
  duration=$((end_epoch - start_epoch))
  result_json="$(latest_result_json "${dataset}" "${seed}" || true)"
  if [[ ${status} -eq 0 && -z "${result_json}" ]]; then
    status=97
  fi
  if [[ ${status} -eq 0 ]]; then
    printf '%s\t%s\t%s\tOK\t%s\t%s\t%s\t%s\t%s\t%s\n' "dino_visual_only" "${dataset}" "${seed}" "${start_text}" "${end_text}" "${duration}" "${log}" "${result_json}" "${params}" >> "${STATUS_FILE}"
    printf '%s\t%s\t%s\t%s\t%s\n' "dino_visual_only" "${dataset}" "${seed}" "${result_json}" "${params}" >> "${RESULTS_INDEX}"
    log_master "=== END phase=dino_visual_only dataset=${dataset} seed=${seed} status=OK duration=${duration}s result=${result_json} at ${end_text} ==="
  else
    printf '%s\t%s\t%s\tFAIL(%s)\t%s\t%s\t%s\t%s\t%s\t%s\n' "dino_visual_only" "${dataset}" "${seed}" "${status}" "${start_text}" "${end_text}" "${duration}" "${log}" "${result_json}" "${params}" >> "${STATUS_FILE}"
    log_master "=== END phase=dino_visual_only dataset=${dataset} seed=${seed} status=FAIL(${status}) duration=${duration}s result=${result_json} at ${end_text} ==="
    exit "${status}"
  fi
}

aggregate_results() {
  python - "${EXP_ROOT}" "${RESULTS_INDEX}" <<'AGGPY' | tee -a "${MASTER_LOG}"
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
index = Path(sys.argv[2])
records = []
with index.open("r", encoding="utf-8") as f:
    next(f, None)
    for line in f:
        phase, dataset, seed, result, params = line.rstrip("\n").split("\t")
        records.append((phase, dataset, int(seed), Path(result), params))

by_dataset = defaultdict(list)
for phase, dataset, seed, result, params in records:
    if phase != "dino_visual_only" or not result.exists():
        continue
    data = json.loads(result.read_text(encoding="utf-8"))
    acc = [float(x) for x in data["acc_list"]]
    by_dataset[(dataset, params)].append((seed, acc, result))

def std(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0

rows = []
for (dataset, params), items in sorted(by_dataset.items()):
    items.sort(key=lambda x: x[0])
    curves = [acc for _, acc, _ in items]
    width = len(curves[0])
    session_mean = [statistics.mean(curve[i] for curve in curves) for i in range(width)]
    session_std = [std([curve[i] for curve in curves]) for i in range(width)]
    avg_values = [sum(curve) / len(curve) for curve in curves]
    pd_values = [curve[0] - curve[-1] for curve in curves]
    rows.append({
        "phase": "dino_visual_only",
        "dataset": dataset,
        "params": params,
        "seeds": [seed for seed, _, _ in items],
        "n": len(items),
        "session_mean": [round(x, 4) for x in session_mean],
        "session_std": [round(x, 4) for x in session_std],
        "avg_mean": round(statistics.mean(avg_values), 4),
        "avg_std": round(std(avg_values), 4),
        "pd_mean": round(statistics.mean(pd_values), 4),
        "pd_std": round(std(pd_values), 4),
        "final_mean": round(session_mean[-1], 4),
        "final_std": round(session_std[-1], 4),
    })

payload = {
    "summary": rows,
    "records": [
        {"phase": p, "dataset": d, "seed": s, "result": str(r), "params": pa}
        for p, d, s, r, pa in records
    ],
}
(root / "aggregate_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
lines = ["DINO visual-only aggregate", "==========================", ""]
for row in rows:
    lines.append(f"{row['dataset']} | n={row['n']} | params={row['params']}")
    lines.append(f"  session_mean: {[round(x, 2) for x in row['session_mean']]}")
    lines.append(f"  session_std: {[round(x, 2) for x in row['session_std']]}")
    lines.append(f"  avg_mean: {row['avg_mean']:.4f}")
    lines.append(f"  avg_std: {row['avg_std']:.4f}")
    lines.append(f"  pd_mean: {row['pd_mean']:.4f}")
    lines.append(f"  pd_std: {row['pd_std']:.4f}")
    lines.append(f"  final_mean: {row['final_mean']:.4f}")
    lines.append(f"  final_std: {row['final_std']:.4f}")
    lines.append("")
(root / "aggregate_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print((root / "aggregate_summary.txt").read_text(encoding="utf-8"))
AGGPY
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
log_master "Seeds: ${SEEDS}"
log_master "Branch: dino_visual_only only"

for seed in "${SEED_LIST[@]}"; do
  for data_cfg in "${DATA_CFGS[@]}"; do
    run_dino_only "${seed}" "${data_cfg}"
  done
done

aggregate_results
log_master "ALL_DONE $(date '+%F %T %Z')"
log_master "Experiment root: ${REPO_ROOT}/${EXP_ROOT}"
log_master "Status file: ${REPO_ROOT}/${STATUS_FILE}"
log_master "Results index: ${REPO_ROOT}/${RESULTS_INDEX}"
log_master "Aggregate summary: ${REPO_ROOT}/${EXP_ROOT}/aggregate_summary.txt"
