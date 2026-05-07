import argparse
import json
import math
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
from models.semantic_branch_composer import NovelSemanticComposer
from semantic_option2_killswitch import (
    accuracy_from_probs,
    collect_final_queries,
    compose_probs_from_semantic,
    compute_cov_logits,
    compute_knn_logits,
    extract_final_state,
    normalize_vector,
    select_diverse_description_atoms,
    to_builtin,
)


def create_output_dir(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"semantic_branch_composer_killswitch_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def latest_semantic_oracle_gain(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    oracle_root = output_root / cfg.DATASET.NAME / f"semantic_killswitch_{Path(train_cfg).stem}"
    if not oracle_root.exists():
        return None, None
    result_files = sorted(oracle_root.glob("*/results.json"))
    if not result_files:
        return None, None
    latest = result_files[-1]
    with latest.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return float(payload["oracle_gain"]), latest.parent


def build_component_bank(merged_state, num_cls, num_base_cls, lambda_t, max_desc_atoms, base_atom_topk, device):
    text_features = F.normalize(merged_state["text_features"][:num_cls].to(device), dim=-1)
    desc_proto = F.normalize(merged_state["description_proto"][:num_cls].to(device), dim=-1)
    desc_features = F.normalize(merged_state["description_features"].to(device), dim=-1)
    desc_targets = merged_state["description_targets"].to(device)
    image_proto = F.normalize(merged_state["image_proto"][:num_cls].to(device), dim=-1)

    baseline_sem = F.normalize((1.0 - lambda_t) * text_features + lambda_t * desc_proto, dim=-1)
    component_banks = {}
    metadata = {}
    for cls_id in range(num_cls):
        desc_group = desc_features[desc_targets == cls_id]
        desc_atoms = select_diverse_description_atoms(desc_group, max_desc_atoms) if desc_group.numel() > 0 else []
        components = [
            ("prompt", text_features[cls_id]),
            ("desc_mean", desc_proto[cls_id]),
        ]
        for atom_idx, atom in enumerate(desc_atoms):
            components.append((f"desc_atom_{atom_idx}", atom.to(device)))
        if cls_id >= num_base_cls and num_base_cls > 0:
            sims = torch.matmul(baseline_sem[cls_id], baseline_sem[:num_base_cls].T)
            topk = min(int(base_atom_topk), num_base_cls)
            top_idx = torch.topk(sims, k=topk).indices.tolist()
            for rank, base_idx in enumerate(top_idx):
                components.append((f"base_anchor_{rank}", baseline_sem[base_idx]))
        names = [name for name, _ in components]
        vectors = torch.stack([normalize_vector(vec) for _, vec in components], dim=0)
        component_banks[cls_id] = vectors
        metadata[cls_id] = names
    return baseline_sem, image_proto, desc_features, desc_targets, component_banks, metadata


def optimize_class_oracle(
    cls_id,
    baseline_semantic,
    current_semantic,
    component_bank,
    image_proto,
    num_base_cls,
    beta,
    ensemble_alpha,
    prob_cov,
    prob_knn,
    query_features,
    query_targets,
    max_steps,
    lr,
    alpha_reg,
    entropy_reg,
):
    device = query_features.device
    baseline_probs, _ = compose_probs_from_semantic(
        query_features,
        current_semantic,
        image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )
    best_acc = accuracy_from_probs(baseline_probs, query_targets)
    best_ce = float(F.nll_loss(torch.log(torch.clamp(baseline_probs, min=1e-12)), query_targets).item())
    best_state = None

    init_schemes = [
        {"alpha_init": 0.05, "bias_index": 1},
        {"alpha_init": 0.35, "bias_index": 1},
        {"alpha_init": 0.50, "bias_index": None},
        {"alpha_init": 0.75, "bias_index": None},
    ]

    for scheme in init_schemes:
        init_logits = torch.zeros(component_bank.shape[0], device=device)
        if scheme["bias_index"] is not None and scheme["bias_index"] < init_logits.numel():
            init_logits[scheme["bias_index"]] = 1.5
        alpha_init = float(scheme["alpha_init"])
        alpha_init = min(max(alpha_init, 1e-4), 1.0 - 1e-4)
        init_alpha_logit = math.log(alpha_init / (1.0 - alpha_init))

        logits = torch.nn.Parameter(init_logits)
        alpha_logit = torch.nn.Parameter(torch.tensor(init_alpha_logit, device=device))
        optimizer = torch.optim.Adam([logits, alpha_logit], lr=lr)

        for _ in range(max_steps):
            optimizer.zero_grad()
            semantic_vec, weights, alpha = NovelSemanticComposer.compose_from_parameters(
                baseline_sem=baseline_semantic[cls_id],
                component_vectors=component_bank,
                component_logits=logits,
                alpha_logit=alpha_logit,
            )
            temp_semantic = current_semantic.clone()
            temp_semantic[cls_id] = semantic_vec
            probs, _ = compose_probs_from_semantic(
                query_features,
                temp_semantic,
                image_proto,
                num_base_cls,
                beta,
                ensemble_alpha,
                prob_cov,
                prob_knn,
            )
            ce = F.nll_loss(torch.log(torch.clamp(probs, min=1e-12)), query_targets)
            entropy = -(weights * torch.log(torch.clamp(weights, min=1e-12))).sum()
            loss = ce + alpha_reg * (alpha ** 2) + entropy_reg * entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_([logits, alpha_logit], 5.0)
            optimizer.step()

            with torch.no_grad():
                acc = accuracy_from_probs(probs, query_targets)
                ce_val = float(ce.item())
                if acc > best_acc + 1e-6 or (abs(acc - best_acc) <= 1e-6 and ce_val < best_ce - 1e-6):
                    best_acc = acc
                    best_ce = ce_val
                    best_state = {
                        "semantic_vec": semantic_vec.detach().clone(),
                        "weights": weights.detach().clone(),
                        "alpha": float(alpha.item()),
                        "acc": best_acc,
                        "ce": best_ce,
                    }
    return best_state


def summarize_component_usage(chosen_components):
    counts = Counter(chosen_components)
    return dict(sorted(counts.items()))


def run_module_oracle(
    merged_state,
    query_features,
    query_targets,
    num_cls,
    num_base_cls,
    beta,
    lambda_t,
    ensemble_alpha,
    max_desc_atoms,
    base_atom_topk,
    max_steps,
    max_passes,
    lr,
    alpha_reg,
    entropy_reg,
):
    device = query_features.device
    baseline_sem, image_proto, description_features, description_targets, component_banks, component_meta = build_component_bank(
        merged_state=merged_state,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        lambda_t=lambda_t,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
        device=device,
    )
    cov_image = merged_state["cov_image"].to(device)
    prob_cov = F.softmax(compute_cov_logits(query_features, image_proto, cov_image) / 512.0, dim=-1)
    prob_knn = F.softmax(compute_knn_logits(query_features, description_features, description_targets, num_cls), dim=-1)

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
    baseline_ce = float(F.nll_loss(torch.log(torch.clamp(baseline_probs, min=1e-12)), query_targets).item())

    current_sem = baseline_sem.clone()
    current_acc = baseline_acc
    current_ce = baseline_ce
    novel_classes = list(range(num_base_cls, num_cls))
    chosen_components = {cls_id: "baseline_sem" for cls_id in novel_classes}
    chosen_alphas = {cls_id: 0.0 for cls_id in novel_classes}

    for _ in range(max_passes):
        improved_any = False
        for cls_id in novel_classes:
            best = optimize_class_oracle(
                cls_id=cls_id,
                baseline_semantic=baseline_sem,
                current_semantic=current_sem,
                component_bank=component_banks[cls_id],
                image_proto=image_proto,
                num_base_cls=num_base_cls,
                beta=beta,
                ensemble_alpha=ensemble_alpha,
                prob_cov=prob_cov,
                prob_knn=prob_knn,
                query_features=query_features,
                query_targets=query_targets,
                max_steps=max_steps,
                lr=lr,
                alpha_reg=alpha_reg,
                entropy_reg=entropy_reg,
            )
            if best is None:
                continue
            if best["acc"] > current_acc + 1e-6 or (abs(best["acc"] - current_acc) <= 1e-6 and best["ce"] < current_ce - 1e-6):
                current_sem[cls_id] = best["semantic_vec"]
                current_acc = best["acc"]
                current_ce = best["ce"]
                top_idx = int(torch.argmax(best["weights"]).item())
                chosen_components[cls_id] = component_meta[cls_id][top_idx]
                chosen_alphas[cls_id] = best["alpha"]
                improved_any = True
        if not improved_any:
            break

    changed_classes = [cls_id for cls_id in novel_classes if chosen_components[cls_id] != "baseline_sem" and chosen_alphas[cls_id] > 1e-6]
    return {
        "baseline_acc": round(float(baseline_acc), 3),
        "oracle_acc": round(float(current_acc), 3),
        "oracle_gain": round(float(current_acc - baseline_acc), 3),
        "changed_class_rate": round(float(len(changed_classes) / max(len(novel_classes), 1)), 4),
        "mean_alpha_all_novel": round(float(sum(chosen_alphas.values()) / max(len(novel_classes), 1)), 4),
        "mean_alpha_changed": round(float(sum(chosen_alphas[cls_id] for cls_id in changed_classes) / max(len(changed_classes), 1)), 4) if changed_classes else 0.0,
        "top_component_counts": summarize_component_usage([chosen_components[cls_id] for cls_id in changed_classes]),
        "chosen_components": {str(k): v for k, v in chosen_components.items()},
        "chosen_alphas": {str(k): round(float(v), 4) for k, v in chosen_alphas.items()},
    }


def run_dataset(
    data_cfg,
    train_cfg,
    max_desc_atoms,
    base_atom_topk,
    max_steps,
    max_passes,
    lr,
    alpha_reg,
    entropy_reg,
):
    cfg = setup_cfg(data_cfg, train_cfg)
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

    full_results = run_module_oracle(
        merged_state=merged_state,
        query_features=query_features,
        query_targets=query_targets,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        beta=beta,
        lambda_t=lambda_t,
        ensemble_alpha=ensemble_alpha,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
        max_steps=max_steps,
        max_passes=max_passes,
        lr=lr,
        alpha_reg=alpha_reg,
        entropy_reg=entropy_reg,
    )
    fused_results = run_module_oracle(
        merged_state=merged_state,
        query_features=query_features,
        query_targets=query_targets,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        beta=beta,
        lambda_t=lambda_t,
        ensemble_alpha=1.0,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
        max_steps=max_steps,
        max_passes=max_passes,
        lr=lr,
        alpha_reg=alpha_reg,
        entropy_reg=entropy_reg,
    )

    semantic_oracle_gain, semantic_oracle_dir = latest_semantic_oracle_gain(cfg, train_cfg)
    recovery = None
    if semantic_oracle_gain is not None and abs(semantic_oracle_gain) > 1e-9:
        recovery = round(float(full_results["oracle_gain"] / semantic_oracle_gain), 4)

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
        "max_steps": max_steps,
        "max_passes": max_passes,
        "lr": lr,
        "alpha_reg": alpha_reg,
        "entropy_reg": entropy_reg,
        "semantic_oracle_gain": semantic_oracle_gain,
        "semantic_oracle_dir": str(semantic_oracle_dir) if semantic_oracle_dir is not None else None,
        "recovery_vs_semantic_oracle": recovery,
        "full": full_results,
        "fused_only": fused_results,
    }
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [
        f"dataset: {payload['dataset']}",
        f"runtime_sec: {payload['runtime_sec']}",
        f"semantic_oracle_gain: {payload['semantic_oracle_gain']}",
        f"recovery_vs_semantic_oracle: {payload['recovery_vs_semantic_oracle']}",
        f"full.oracle_gain: {payload['full']['oracle_gain']}",
        f"full.changed_class_rate: {payload['full']['changed_class_rate']}",
        f"full.mean_alpha_changed: {payload['full']['mean_alpha_changed']}",
        f"fused_only.oracle_gain: {payload['fused_only']['oracle_gain']}",
        f"fused_only.changed_class_rate: {payload['fused_only']['changed_class_rate']}",
        f"fused_only.mean_alpha_changed: {payload['fused_only']['mean_alpha_changed']}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "dataset": payload["dataset"],
        "semantic_oracle_gain": payload["semantic_oracle_gain"],
        "recovery_vs_semantic_oracle": payload["recovery_vs_semantic_oracle"],
        "full": {
            "oracle_gain": payload["full"]["oracle_gain"],
            "changed_class_rate": payload["full"]["changed_class_rate"],
            "mean_alpha_changed": payload["full"]["mean_alpha_changed"],
            "top_component_counts": payload["full"]["top_component_counts"],
        },
        "fused_only": {
            "oracle_gain": payload["fused_only"]["oracle_gain"],
            "changed_class_rate": payload["fused_only"]["changed_class_rate"],
            "mean_alpha_changed": payload["fused_only"]["mean_alpha_changed"],
            "top_component_counts": payload["fused_only"]["top_component_counts"],
        },
    }, indent=2))
    print(f"Saved semantic branch composer killswitch to: {out_dir}")
    return out_dir, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--max_desc_atoms", type=int, default=2)
    parser.add_argument("--base_atom_topk", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--max_passes", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--alpha_reg", type=float, default=0.0)
    parser.add_argument("--entropy_reg", type=float, default=0.0)
    args = parser.parse_args()
    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        max_desc_atoms=args.max_desc_atoms,
        base_atom_topk=args.base_atom_topk,
        max_steps=args.max_steps,
        max_passes=args.max_passes,
        lr=args.lr,
        alpha_reg=args.alpha_reg,
        entropy_reg=args.entropy_reg,
    )


if __name__ == "__main__":
    main()
