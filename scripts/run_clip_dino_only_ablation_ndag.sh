#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-clip_dino_only_$(date '+%Y%m%d-%H%M%S')}"
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

make_branch_cfg() {
  local method="$1"
  local dest="$2"
  local seed="$3"
  mkdir -p "$(dirname "${dest}")"
  {
    printf 'OUTPUT:\n'
    printf '  ROOT: "%s"\n' "${EXP_ROOT}"
    if [[ "${method}" == "dino_visual_only" ]]; then
      printf 'METHOD: "bimc_dino_fusion"\n'
    fi
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

ensure_branch_cfg() {
  local method="$1"
  local seed="$2"
  local dest="${CFG_DIR}/seed${seed}/${method}.yaml"
  if [[ ! -s "${dest}" ]]; then
    make_branch_cfg "${method}" "${dest}" "${seed}"
  fi
  printf '%s\n' "${dest}"
}

latest_result_json() {
  local dataset="$1"
  local trainer="$2"
  local seed="$3"
  local files=()
  shopt -s nullglob
  files=("${EXP_ROOT}/${dataset}/${trainer}/seed${seed}_"*/results.json)
  shopt -u nullglob
  if [[ ${#files[@]} -eq 0 ]]; then
    return 1
  fi
  ls -t "${files[@]}" | head -n 1
}

completed_result_json() {
  local phase="$1"
  local dataset="$2"
  local seed="$3"
  local params="$4"
  awk -F '\t' -v phase="${phase}" -v dataset="${dataset}" -v seed="${seed}" -v params="${params}" '
    $1 == phase && $2 == dataset && $3 == seed && $4 == "OK" && $10 == params { result = $9 }
    END { if (result != "") print result }
  ' "${STATUS_FILE}"
}

run_command() {
  local phase="$1"
  local dataset="$2"
  local trainer="$3"
  local seed="$4"
  local params="$5"
  shift 5
  local log start_text end_text start_epoch end_epoch duration status result_json completed_json

  completed_json="$(completed_result_json "${phase}" "${dataset}" "${seed}" "${params}" || true)"
  if [[ -n "${completed_json}" && -f "${completed_json}" ]]; then
    log_master "=== SKIP phase=${phase} dataset=${dataset} seed=${seed} existing=${completed_json} ==="
    return 0
  fi

  log="${LOG_DIR}/${phase}_${dataset}_seed${seed}.log"
  start_text="$(date '+%F %T %Z')"
  start_epoch="$(date +%s)"
  log_master "=== START phase=${phase} dataset=${dataset} seed=${seed} params=${params} at ${start_text} ==="
  set +e
  PYTHONUNBUFFERED=1 "$@" 2>&1 | tee "${log}"
  status=${PIPESTATUS[0]}
  set -e
  end_text="$(date '+%F %T %Z')"
  end_epoch="$(date +%s)"
  duration=$((end_epoch - start_epoch))
  result_json="$(latest_result_json "${dataset}" "${trainer}" "${seed}" || true)"

  if [[ ${status} -eq 0 && -z "${result_json}" ]]; then
    status=97
  fi

  if [[ ${status} -eq 0 ]]; then
    printf '%s\t%s\t%s\tOK\t%s\t%s\t%s\t%s\t%s\t%s\n' "${phase}" "${dataset}" "${seed}" "${start_text}" "${end_text}" "${duration}" "${log}" "${result_json}" "${params}" >> "${STATUS_FILE}"
    printf '%s\t%s\t%s\t%s\t%s\n' "${phase}" "${dataset}" "${seed}" "${result_json}" "${params}" >> "${RESULTS_INDEX}"
    log_master "=== END phase=${phase} dataset=${dataset} seed=${seed} status=OK duration=${duration}s result=${result_json} at ${end_text} ==="
  else
    printf '%s\t%s\t%s\tFAIL(%s)\t%s\t%s\t%s\t%s\t%s\t%s\n' "${phase}" "${dataset}" "${seed}" "${status}" "${start_text}" "${end_text}" "${duration}" "${log}" "${result_json}" "${params}" >> "${STATUS_FILE}"
    log_master "=== END phase=${phase} dataset=${dataset} seed=${seed} status=FAIL(${status}) duration=${duration}s result=${result_json} at ${end_text} ==="
    exit "${status}"
  fi
}

run_clip_only() {
  local seed="$1"
  local data_cfg="$2"
  local dataset cfg
  dataset="$(dataset_name "${data_cfg}")"
  cfg="$(ensure_branch_cfg "clip_visual_only" "${seed}")"
  run_command "clip_visual_only" "${dataset}" "bimc" "${seed}" "branch=clip_visual_only,beta=0,ensemble=false,vision_calibration=false" \
    python main.py --data_cfg "${data_cfg}" --train_cfg "${cfg}"
}

run_dino_only() {
  local seed="$1"
  local data_cfg="$2"
  local dataset cfg
  dataset="$(dataset_name "${data_cfg}")"
  cfg="$(ensure_branch_cfg "dino_visual_only" "${seed}")"
  run_command "dino_visual_only" "${dataset}" "bimc_dino_fusion" "${seed}" "branch=dino_visual_only,beta=0,omega=1,ensemble=false,vision_calibration=false" \
    python scripts/run_main_with_dino_omega.py --data_cfg "${data_cfg}" --train_cfg "${cfg}" --dino_omega 1.0
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

summary = defaultdict(lambda: {"seeds": [], "final": [], "avg": []})
for phase, dataset, seed, result, params in records:
    if not result.exists():
        continue
    data = json.loads(result.read_text(encoding="utf-8"))
    if "acc_list" not in data:
        continue
    acc = [float(x) for x in data["acc_list"]]
    key = (phase, dataset, params)
    summary[key]["seeds"].append(seed)
    summary[key]["final"].append(acc[-1])
    summary[key]["avg"].append(sum(acc) / len(acc))

def std(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0

rows = []
for (phase, dataset, params), s in sorted(summary.items()):
    rows.append({
        "phase": phase,
        "dataset": dataset,
        "params": params,
        "seeds": sorted(s["seeds"]),
        "n": len(s["final"]),
        "final_mean": round(statistics.mean(s["final"]), 4),
        "final_std": round(std(s["final"]), 4),
        "avg_mean": round(statistics.mean(s["avg"]), 4),
        "avg_std": round(std(s["avg"]), 4),
    })

payload = {
    "summary": rows,
    "records": [
        {"phase": p, "dataset": d, "seed": se, "result": str(r), "params": pa}
        for p, d, se, r, pa in records
    ],
}
(root / "aggregate_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
lines = ["Aggregate summary", "=================", ""]
for row in rows:
    lines.append(f"{row['phase']} | {row['dataset']} | n={row['n']} | params={row['params']}")
    lines.append(f"  final_mean: {row['final_mean']}")
    lines.append(f"  final_std: {row['final_std']}")
    lines.append(f"  avg_mean: {row['avg_mean']}")
    lines.append(f"  avg_std: {row['avg_std']}")
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
log_master "Branches: clip_visual_only, dino_visual_only"

for seed in "${SEED_LIST[@]}"; do
  for data_cfg in "${DATA_CFGS[@]}"; do
    run_clip_only "${seed}" "${data_cfg}"
    run_dino_only "${seed}" "${data_cfg}"
  done
done

aggregate_results
log_master "ALL_DONE $(date '+%F %T %Z')"
log_master "Experiment root: ${REPO_ROOT}/${EXP_ROOT}"
log_master "Status file: ${REPO_ROOT}/${STATUS_FILE}"
log_master "Results index: ${REPO_ROOT}/${RESULTS_INDEX}"
log_master "Aggregate summary: ${REPO_ROOT}/${EXP_ROOT}/aggregate_summary.txt"
