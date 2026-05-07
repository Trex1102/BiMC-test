import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import setup_cfg
from semantic_option2_killswitch import (
    accuracy_from_probs,
    build_semantic_banks,
    collect_final_queries,
    compose_probs_from_semantic,
    compute_cov_logits,
    compute_knn_logits,
    extract_final_state,
    to_builtin,
)


EXACT_FAMILY_NAMES = [
    "prompt_only",
    "desc_mean_only",
    "desc_atom_direct",
    "prompt_desc_atom",
    "base_anchor_family",
    "prompt_desc_composer",
    "full",
]


def setup_cfg_with_override(data_cfg, train_cfg, seed_override=None):
    cfg = setup_cfg(data_cfg, train_cfg)
    if seed_override is None:
        return cfg
    cfg.defrost()
    cfg.SEED = int(seed_override)
    cfg.freeze()
    return cfg


def create_output_dir(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"semantic_oracle_family_ablation_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def candidate_exact_family(name):
    if name == "baseline_sem":
        return "baseline"
    if name == "prompt_only":
        return "prompt_only"
    if name == "desc_mean_only":
        return "desc_mean_only"
    if name.startswith("desc_atom_") or name.startswith("desc_plus_desc_atom_"):
        return "desc_atom_direct"
    if name.startswith("prompt_plus_desc_atom_") or name == "prompt_desc_top_descatom":
        return "prompt_desc_atom"
    if (
        name.startswith("base_atom_")
        or name.startswith("prompt_plus_base_atom_")
        or name.startswith("desc_plus_base_atom_")
        or name.startswith("prompt_desc_base_atom_")
    ):
        return "base_anchor_family"
    return "other"


def candidate_allowed(name, family_name):
    exact_family = candidate_exact_family(name)
    if name == "baseline_sem":
        return True
    if family_name == "full":
        return True
    if family_name == "prompt_desc_composer":
        return exact_family in {
            "prompt_only",
            "desc_mean_only",
            "desc_atom_direct",
            "prompt_desc_atom",
        }
    return exact_family == family_name


def summarize_choices(chosen_candidates):
    exact_counts = Counter()
    broad_counts = Counter()
    changed = 0
    for name in chosen_candidates.values():
        exact_family = candidate_exact_family(name)
        exact_counts[exact_family] += 1
        if name != "baseline_sem":
            changed += 1
        if exact_family == "baseline":
            broad_counts["baseline"] += 1
        elif exact_family in {"desc_atom_direct", "prompt_desc_atom"}:
            broad_counts["desc_atom_family"] += 1
        elif exact_family in {"prompt_only", "desc_mean_only", "base_anchor_family"}:
            broad_counts[exact_family] += 1
        else:
            broad_counts["other"] += 1
    return {
        "changed_class_rate": round(float(changed / max(len(chosen_candidates), 1)), 4),
        "exact_family_counts": dict(sorted(exact_counts.items())),
        "broad_family_counts": dict(sorted(broad_counts.items())),
    }


def run_restricted_oracle(
    banks,
    baseline_sem,
    query_features,
    query_targets,
    image_proto,
    num_cls,
    num_base_cls,
    beta,
    ensemble_alpha,
    prob_cov,
    prob_knn,
    family_name,
    max_passes,
):
    device = query_features.device
    baseline_probs, _ = compose_probs_from_semantic(
        query_features,
        baseline_sem,
        image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )
    baseline_acc = accuracy_from_probs(baseline_probs, query_targets)

    current_sem = baseline_sem.clone()
    current_probs = baseline_probs
    current_acc = baseline_acc
    chosen_meta = {
        cls_id: {
            "name": "baseline_sem",
            "is_baseline": True,
        }
        for cls_id in range(num_cls)
    }
    novel_classes = list(range(num_base_cls, num_cls))

    for _ in range(max_passes):
        improved_any = False
        for cls_id in novel_classes:
            best_acc = current_acc
            best_probs = current_probs
            best_vec = current_sem[cls_id]
            best_meta = chosen_meta[cls_id]
            for candidate in banks[cls_id]:
                if not candidate_allowed(candidate["name"], family_name):
                    continue
                candidate_vec = candidate["vector"].to(device)
                temp_sem = current_sem.clone()
                temp_sem[cls_id] = candidate_vec
                probs, _ = compose_probs_from_semantic(
                    query_features,
                    temp_sem,
                    image_proto,
                    num_base_cls,
                    beta,
                    ensemble_alpha,
                    prob_cov,
                    prob_knn,
                )
                acc = accuracy_from_probs(probs, query_targets)
                if acc > best_acc + 1e-6:
                    best_acc = acc
                    best_probs = probs
                    best_vec = candidate_vec
                    best_meta = {
                        "name": candidate["name"],
                        "is_baseline": bool(candidate["is_baseline"]),
                    }
            if best_acc > current_acc + 1e-6:
                current_acc = best_acc
                current_probs = best_probs
                current_sem[cls_id] = best_vec
                chosen_meta[cls_id] = best_meta
                improved_any = True
        if not improved_any:
            break

    chosen_candidates = {str(cls_id): chosen_meta[cls_id]["name"] for cls_id in novel_classes}
    summary = summarize_choices(chosen_candidates)
    return {
        "baseline_acc": round(float(baseline_acc), 3),
        "oracle_acc": round(float(current_acc), 3),
        "oracle_gain": round(float(current_acc - baseline_acc), 3),
        "changed_class_rate": summary["changed_class_rate"],
        "chosen_candidates": chosen_candidates,
        "exact_family_counts": summary["exact_family_counts"],
        "broad_family_counts": summary["broad_family_counts"],
    }


def run_dataset(data_cfg, train_cfg, families, max_desc_atoms, base_atom_topk, max_passes, seed_override=None):
    cfg = setup_cfg_with_override(data_cfg, train_cfg, seed_override=seed_override)
    out_dir = create_output_dir(cfg, train_cfg)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    start_time = time.time()
    data_manager, model, merged_state, final_task_id = extract_final_state(cfg)
    query_features, query_targets = collect_final_queries(cfg, data_manager, model, final_task_id)

    num_cls = max(data_manager.class_index_in_task[final_task_id]) + 1
    num_base_cls = len(data_manager.class_index_in_task[0])
    beta = float(cfg.DATASET.BETA)
    lambda_t = float(cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
    ensemble_alpha = float(cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.0)

    device = query_features.device
    image_proto = F.normalize(merged_state["image_proto"][:num_cls].to(device), dim=-1)
    description_features = F.normalize(merged_state["description_features"].to(device), dim=-1)
    description_targets = merged_state["description_targets"].to(device)
    cov_image = merged_state["cov_image"].to(device)

    prob_cov = F.softmax(compute_cov_logits(query_features, image_proto, cov_image) / 512.0, dim=-1)
    prob_knn = F.softmax(
        compute_knn_logits(query_features, description_features, description_targets, num_cls),
        dim=-1,
    )

    baseline_sem, banks = build_semantic_banks(
        merged_state,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        lambda_t=lambda_t,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
    )
    baseline_sem = baseline_sem.to(device)

    results = {}
    for family_name in families:
        results[family_name] = run_restricted_oracle(
            banks=banks,
            baseline_sem=baseline_sem,
            query_features=query_features.to(device),
            query_targets=query_targets.to(device),
            image_proto=image_proto,
            num_cls=num_cls,
            num_base_cls=num_base_cls,
            beta=beta,
            ensemble_alpha=ensemble_alpha,
            prob_cov=prob_cov,
            prob_knn=prob_knn,
            family_name=family_name,
            max_passes=max_passes,
        )

    payload = {
        "dataset": cfg.DATASET.NAME,
        "train_cfg": train_cfg,
        "runtime_sec": round(time.time() - start_time, 3),
        "num_seen_classes": num_cls,
        "num_base_classes": num_base_cls,
        "baseline_beta": beta,
        "baseline_lambda_t": lambda_t,
        "ensemble_alpha": ensemble_alpha,
        "max_desc_atoms": max_desc_atoms,
        "base_atom_topk": base_atom_topk,
        "max_passes": max_passes,
        "families": families,
        "results": results,
    }
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [f"dataset: {cfg.DATASET.NAME}", f"runtime_sec: {payload['runtime_sec']}"]
    for family_name in families:
        family_res = payload["results"][family_name]
        summary_lines.extend([
            f"{family_name}.oracle_gain: {family_res['oracle_gain']}",
            f"{family_name}.changed_class_rate: {family_res['changed_class_rate']}",
        ])
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "dataset": cfg.DATASET.NAME,
        "runtime_sec": payload["runtime_sec"],
        "results": {
            family_name: {
                "oracle_gain": payload["results"][family_name]["oracle_gain"],
                "changed_class_rate": payload["results"][family_name]["changed_class_rate"],
            }
            for family_name in families
        },
    }, indent=2))
    print(f"Saved semantic oracle family ablation to: {out_dir}")
    return out_dir, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--families", nargs="+", default=EXACT_FAMILY_NAMES)
    parser.add_argument("--max_desc_atoms", type=int, default=3)
    parser.add_argument("--base_atom_topk", type=int, default=2)
    parser.add_argument("--max_passes", type=int, default=2)
    parser.add_argument("--seed_override", type=int, default=None)
    args = parser.parse_args()
    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        families=args.families,
        max_desc_atoms=args.max_desc_atoms,
        base_atom_topk=args.base_atom_topk,
        max_passes=args.max_passes,
        seed_override=args.seed_override,
    )


if __name__ == "__main__":
    main()
