import argparse
import json
from pathlib import Path


DATASET_ORDER = ["CIFAR100", "cub200", "miniimagenet"]


def latest_results(root, mode, train_cfg_stem):
    results = {}
    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir():
            continue
        trainer_dir = dataset_dir / f"calibration_meta_{mode}_{train_cfg_stem}"
        if not trainer_dir.exists():
            continue
        for result_file in sorted(trainer_dir.glob("seed*/results.json")):
            try:
                with result_file.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue
            results[str(dataset_dir.name)] = payload
    return results


def mode_decision(mode, payloads):
    gains = {name: float(v.get("method_gain", 0.0)) for name, v in payloads.items()}
    recoveries = [float(v.get("recovery_rate", 0.0) or 0.0) for v in payloads.values()]
    non_negative_all = all(gains.get(dataset, float("-inf")) >= 0.0 for dataset in DATASET_ORDER if dataset in payloads)
    cifar_gain = gains.get("CIFAR100", float("-inf"))
    cub_gain = gains.get("cub200", float("-inf"))
    mini_gain = gains.get("miniimagenet", float("-inf"))
    mean_recovery = sum(recoveries) / max(len(recoveries), 1)

    if mode == "bmvc":
        continue_flag = (
            non_negative_all
            and cifar_gain >= 0.2
            and cub_gain >= 0.2
            and mini_gain >= 0.0
        )
    else:
        continue_flag = (
            non_negative_all
            and cifar_gain >= 0.2
            and cub_gain >= 0.2
            and mini_gain >= 0.0
            and mean_recovery >= 0.25
        )

    return {
        "continue_flag": continue_flag,
        "non_negative_all": non_negative_all,
        "mean_gain": round(sum(gains.values()) / max(len(gains), 1), 4),
        "mean_recovery": round(mean_recovery, 4),
        "cifar_gain": round(cifar_gain, 4) if cifar_gain != float("-inf") else None,
        "cub_gain": round(cub_gain, 4) if cub_gain != float("-inf") else None,
        "mini_gain": round(mini_gain, 4) if mini_gain != float("-inf") else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--mode", choices=["bmvc", "astar"], required=True)
    parser.add_argument("--train_cfg_stem", default="bimc_ensemble")
    args = parser.parse_args()

    root = Path(args.root)
    payloads = latest_results(root, args.mode, args.train_cfg_stem)
    decision = mode_decision(args.mode, payloads)

    lines = []
    for dataset in DATASET_ORDER:
        payload = payloads.get(dataset)
        if payload is None:
            continue
        lines.append(
            {
                "dataset": dataset,
                "baseline_acc": payload.get("baseline_acc"),
                "method_acc": payload.get("method_acc"),
                "oracle_acc": payload.get("oracle_acc"),
                "method_gain": payload.get("method_gain"),
                "oracle_gain": payload.get("oracle_gain"),
                "recovery_rate": payload.get("recovery_rate"),
                "recommendation": payload.get("recommendation"),
                "method_params": payload.get("method_params"),
                "oracle_params": payload.get("oracle_params"),
            }
        )

    print(
        json.dumps(
            {
                "mode": args.mode,
                "train_cfg_stem": args.train_cfg_stem,
                "datasets": lines,
                "kill_switch": {
                    "decision": "continue_direction" if decision["continue_flag"] else "kill_direction",
                    **decision,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
