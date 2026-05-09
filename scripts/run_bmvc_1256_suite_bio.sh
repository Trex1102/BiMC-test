#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-bmvc_1256_$(date '+%Y%m%d-%H%M%S')}"
EXP_ROOT="experiments/${RUN_ID}"
LOG_DIR="${EXP_ROOT}/logs"
CFG_DIR="${EXP_ROOT}/generated_configs"
STATUS_FILE="${LOG_DIR}/status.tsv"
RESULTS_INDEX="${LOG_DIR}/results_index.tsv"
MASTER_LOG="${LOG_DIR}/master.log"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20}"
SENS_SEEDS="${SENS_SEEDS:-1 2 3 4 5}"

DATA_CFGS=(
  "configs/datasets/cifar100.yaml"
  "configs/datasets/miniimagenet.yaml"
  "configs/datasets/cub200_bimc_dino_fusion.yaml"
)

mkdir -p "${LOG_DIR}" "${CFG_DIR}"
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

make_train_cfg() {
  local src="$1"
  local dest="$2"
  {
    printf 'OUTPUT:\n'
    printf '  ROOT: "%s"\n' "${EXP_ROOT}"
    cat "${src}"
  } > "${dest}"
}

make_train_cfg "configs/trainers/bimc.yaml" "${CFG_DIR}/bimc.yaml"
make_train_cfg "configs/trainers/bimc_ensemble.yaml" "${CFG_DIR}/bimc_ensemble.yaml"
make_train_cfg "configs/trainers/bimc_dino_fusion.yaml" "${CFG_DIR}/bimc_dino_fusion.yaml"

latest_result_json() {
  local dataset="$1"
  local trainer="$2"
  local seed="$3"
  local files=(
    "${EXP_ROOT}/${dataset}/${trainer}/seed${seed}_"*/results.json
  )
  if [[ ${#files[@]} -eq 0 ]]; then
    return 0
  fi
  ls -t "${files[@]}" | head -n 1
}

run_command() {
  local phase="$1"
  local dataset="$2"
  local trainer="$3"
  local seed="$4"
  local params="$5"
  shift 5
  local log start_text end_text start_epoch end_epoch duration status result_json
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
  result_json="$(latest_result_json "${dataset}" "${trainer}" "${seed}")"
  if [[ ${status} -eq 0 ]]; then
    printf '%s\t%s\t%s\tOK\t%s\t%s\t%s\t%s\t%s\t%s\n' "${phase}" "${dataset}" "${seed}" "${start_text}" "${end_text}" "${duration}" "${log}" "${result_json}" "${params}" >> "${STATUS_FILE}"
    printf '%s\t%s\t%s\t%s\t%s\n' "${phase}" "${dataset}" "${seed}" "${result_json}" "${params}" >> "${RESULTS_INDEX}"
    log_master "=== END phase=${phase} dataset=${dataset} seed=${seed} status=OK duration=${duration}s result=${result_json} at ${end_text} ==="
  else
    printf '%s\t%s\t%s\tFAIL(%s)\t%s\t%s\t%s\t%s\t%s\t%s\n' "${phase}" "${dataset}" "${seed}" "${status}" "${start_text}" "${end_text}" "${duration}" "${log}" "${result_json}" "${params}" >> "${STATUS_FILE}"
    log_master "=== END phase=${phase} dataset=${dataset} seed=${seed} status=FAIL(${status}) duration=${duration}s at ${end_text} ==="
    exit "${status}"
  fi
}

run_main_ablation() {
  local phase="$1"
  local cfg="$2"
  local trainer="$3"
  local seed="$4"
  local data_cfg="$5"
  local dataset
  dataset="$(dataset_name "${data_cfg}")"
  run_command "${phase}" "${dataset}" "${trainer}" "${seed}" "train_cfg=$(basename "${cfg}")" \
    python main.py --data_cfg "${data_cfg}" --train_cfg "${cfg}"
}

run_fixed_alpha() {
  local alpha="$1"
  local seed="$2"
  local data_cfg="$3"
  local dataset phase tag
  dataset="$(dataset_name "${data_cfg}")"
  tag="$(python - "${alpha}" <<'PY'
import sys
print("%03d" % int(round(float(sys.argv[1]) * 100)))
PY
)"
  phase="fixed_alpha_${tag}"
  run_command "${phase}" "${dataset}" "prototype_refinement_conservative_bimc_dino_fusion" "${seed}" "alpha=${alpha}" \
    python scripts/prototype_refinement_conservative_dino_fusion.py \
      --data_cfg "${data_cfg}" \
      --train_cfg "${CFG_DIR}/bimc_dino_fusion.yaml" \
      --seed_override "${seed}" \
      --alpha_grid "${alpha}" \
      --mass_thr 0.3 \
      --min_count 5
}

run_pseudoval() {
  local phase="$1"
  local seed="$2"
  local data_cfg="$3"
  shift 3
  local dataset
  dataset="$(dataset_name "${data_cfg}")"
  run_command "${phase}" "${dataset}" "prototype_refinement_conservative_pseudoval_sessionwise_bimc_dino_fusion" "${seed}" "$*" \
    python scripts/prototype_refinement_conservative_dino_fusion_sessionwise.py \
      --data_cfg "${data_cfg}" \
      --train_cfg "${CFG_DIR}/bimc_dino_fusion.yaml" \
      --seed_override "${seed}" \
      "$@"
}

aggregate_results() {
  python - "${EXP_ROOT}" "${RESULTS_INDEX}" <<'PY' | tee -a "${MASTER_LOG}"
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

summary = defaultdict(lambda: {
    "seeds": [],
    "final": [],
    "avg": [],
    "base_final": [],
    "base_avg": [],
    "gain_final": [],
    "gain_avg": [],
})
failures = []
for phase, dataset, seed, result, params in records:
    with result.open("r", encoding="utf-8") as f:
        data = json.load(f)
    key = (phase, dataset, params)
    s = summary[key]
    s["seeds"].append(seed)
    if "acc_list" in data:
        acc = [float(x) for x in data["acc_list"]]
        s["final"].append(acc[-1])
        s["avg"].append(sum(acc) / len(acc))
    elif "final_acc_list" in data:
        final = [float(x) for x in data["final_acc_list"]]
        base = [float(x) for x in data["baseline_acc_list"]]
        gain = [float(x) for x in data["gain_list"]]
        s["final"].append(final[-1])
        s["avg"].append(float(data.get("final_avg", sum(final) / len(final))))
        s["base_final"].append(base[-1])
        s["base_avg"].append(float(data.get("baseline_avg", sum(base) / len(base))))
        s["gain_final"].append(final[-1] - base[-1])
        s["gain_avg"].append(float(data.get("gain_avg", sum(gain) / len(gain))))
        for i, g in enumerate(gain):
            if g < 0:
                failures.append({
                    "phase": phase,
                    "dataset": dataset,
                    "seed": seed,
                    "session": i,
                    "gain": round(g, 6),
                    "baseline": base[i],
                    "final": final[i],
                    "params": params,
                    "result": str(result),
                })
    elif "final_full_acc" in data:
        final_acc = float(data["final_full_acc"])
        base_acc = float(data.get("baseline_full_acc", final_acc))
        s["final"].append(final_acc)
        s["avg"].append(final_acc)
        s["base_final"].append(base_acc)
        s["gain_final"].append(final_acc - base_acc)

def std(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0

rows = []
for (phase, dataset, params), s in sorted(summary.items()):
    row = {
        "phase": phase,
        "dataset": dataset,
        "params": params,
        "seeds": sorted(s["seeds"]),
        "n": len(s["final"]),
    }
    for name in ["base_final", "base_avg", "final", "avg", "gain_final", "gain_avg"]:
        vals = s[name]
        if vals:
            row[f"{name}_mean"] = round(statistics.mean(vals), 4)
            row[f"{name}_std"] = round(std(vals), 4)
    rows.append(row)

payload = {
    "summary": rows,
    "negative_gain_sessions": failures,
    "records": [
        {"phase": p, "dataset": d, "seed": se, "result": str(r), "params": pa}
        for p, d, se, r, pa in records
    ],
}
(root / "aggregate_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

lines = ["Aggregate summary", "=================", ""]
for row in rows:
    lines.append(f"{row['phase']} | {row['dataset']} | n={row['n']} | params={row['params']}")
    for key in [
        "base_final_mean",
        "base_final_std",
        "final_mean",
        "final_std",
        "gain_final_mean",
        "gain_final_std",
        "base_avg_mean",
        "final_avg_mean",
        "gain_avg_mean",
    ]:
        if key in row:
            lines.append(f"  {key}: {row[key]}")
    lines.append("")
lines += ["Negative gain sessions", "======================"]
if failures:
    lines += [json.dumps(x, sort_keys=True) for x in failures[:200]]
else:
    lines.append("None found in sessionwise payloads.")
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
read -r -a SENS_SEED_LIST <<< "${SENS_SEEDS}"

log_master "Run id: ${RUN_ID}"
log_master "Experiment root: ${REPO_ROOT}/${EXP_ROOT}"
log_master "Full seeds: ${SEEDS}"
log_master "Sensitivity seeds: ${SENS_SEEDS}"
log_master "Started: $(date '+%F %T %Z')"

log_master "--- 2: ablations original BiMC, BiMC ensemble, and Dino-fusion baseline ---"
for seed in "${SEED_LIST[@]}"; do
  for data_cfg in "${DATA_CFGS[@]}"; do
    run_main_ablation "ablation_bimc" "${CFG_DIR}/bimc.yaml" "bimc" "${seed}" "${data_cfg}"
    run_main_ablation "ablation_bimc_ensemble" "${CFG_DIR}/bimc_ensemble.yaml" "bimc_ensemble" "${seed}" "${data_cfg}"
    run_main_ablation "ablation_dino_fusion" "${CFG_DIR}/bimc_dino_fusion.yaml" "bimc_dino_fusion_ensemble" "${seed}" "${data_cfg}"
  done
done

log_master "--- 5: fixed-alpha CPR baselines for pseudo-validation comparison ---"
for seed in "${SEED_LIST[@]}"; do
  for data_cfg in "${DATA_CFGS[@]}"; do
    run_fixed_alpha "0.5" "${seed}" "${data_cfg}"
    run_fixed_alpha "0.75" "${seed}" "${data_cfg}"
    run_fixed_alpha "1.0" "${seed}" "${data_cfg}"
  done
done

log_master "--- 1: 20-seed sessionwise pseudo-validation CPR ---"
for seed in "${SEED_LIST[@]}"; do
  for data_cfg in "${DATA_CFGS[@]}"; do
    run_pseudoval "pseudoval_20seed" "${seed}" "${data_cfg}" \
      --alpha_grid 0.5 0.75 1.0 \
      --mass_thr 0.3 \
      --min_count 5 \
      --val_fraction 0.2 \
      --base_val_conf_thr 0.5 \
      --base_val_max_per_class 50 \
      --fallback_alpha 0.75
  done
done

log_master "--- 6: one-factor sensitivity runs on first five seeds ---"
SENSITIVITY_SPECS=(
  "sens_mass020|--alpha_grid 0.5 0.75 1.0 --mass_thr 0.2 --min_count 5 --val_fraction 0.2 --base_val_conf_thr 0.5 --base_val_max_per_class 50 --fallback_alpha 0.75"
  "sens_mass040|--alpha_grid 0.5 0.75 1.0 --mass_thr 0.4 --min_count 5 --val_fraction 0.2 --base_val_conf_thr 0.5 --base_val_max_per_class 50 --fallback_alpha 0.75"
  "sens_mincount003|--alpha_grid 0.5 0.75 1.0 --mass_thr 0.3 --min_count 3 --val_fraction 0.2 --base_val_conf_thr 0.5 --base_val_max_per_class 50 --fallback_alpha 0.75"
  "sens_mincount010|--alpha_grid 0.5 0.75 1.0 --mass_thr 0.3 --min_count 10 --val_fraction 0.2 --base_val_conf_thr 0.5 --base_val_max_per_class 50 --fallback_alpha 0.75"
  "sens_valfrac010|--alpha_grid 0.5 0.75 1.0 --mass_thr 0.3 --min_count 5 --val_fraction 0.1 --base_val_conf_thr 0.5 --base_val_max_per_class 50 --fallback_alpha 0.75"
  "sens_valfrac030|--alpha_grid 0.5 0.75 1.0 --mass_thr 0.3 --min_count 5 --val_fraction 0.3 --base_val_conf_thr 0.5 --base_val_max_per_class 50 --fallback_alpha 0.75"
  "sens_alphagrid025|--alpha_grid 0.25 0.5 0.75 1.0 --mass_thr 0.3 --min_count 5 --val_fraction 0.2 --base_val_conf_thr 0.5 --base_val_max_per_class 50 --fallback_alpha 0.75"
  "sens_alphagrid_sparse|--alpha_grid 0.5 1.0 --mass_thr 0.3 --min_count 5 --val_fraction 0.2 --base_val_conf_thr 0.5 --base_val_max_per_class 50 --fallback_alpha 0.75"
)
for spec in "${SENSITIVITY_SPECS[@]}"; do
  phase="${spec%%|*}"
  args="${spec#*|}"
  read -r -a arg_array <<< "${args}"
  for seed in "${SENS_SEED_LIST[@]}"; do
    for data_cfg in "${DATA_CFGS[@]}"; do
      run_pseudoval "${phase}" "${seed}" "${data_cfg}" "${arg_array[@]}"
    done
  done
done

aggregate_results
log_master "ALL_DONE $(date '+%F %T %Z')"
log_master "Experiment root: ${REPO_ROOT}/${EXP_ROOT}"
log_master "Status file: ${REPO_ROOT}/${STATUS_FILE}"
log_master "Results index: ${REPO_ROOT}/${RESULTS_INDEX}"
log_master "Aggregate summary: ${REPO_ROOT}/${EXP_ROOT}/aggregate_summary.txt"
