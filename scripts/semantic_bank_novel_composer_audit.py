import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import setup_cfg
from semantic_option2_killswitch import (
    average_vectors,
    collect_final_queries,
    compose_probs_from_semantic,
    compute_cov_logits,
    compute_knn_logits,
    extract_final_state,
    normalize_vector,
    to_builtin,
)


def create_output_dir(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"semantic_bank_novel_composer_audit_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def setup_cfg_with_override(data_cfg, train_cfg, seed_override=None):
    cfg = setup_cfg(data_cfg, train_cfg)
    if seed_override is None:
        return cfg
    cfg.defrost()
    cfg.SEED = int(seed_override)
    cfg.freeze()
    return cfg


def accuracy_stats(probs, targets, num_base_cls):
    preds = torch.argmax(probs, dim=1)
    overall = float((preds == targets).float().mean().item() * 100.0)
    base_mask = targets < num_base_cls
    novel_mask = targets >= num_base_cls
    base = float((preds[base_mask] == targets[base_mask]).float().mean().item() * 100.0) if base_mask.any() else 0.0
    novel = float((preds[novel_mask] == targets[novel_mask]).float().mean().item() * 100.0) if novel_mask.any() else 0.0
    return round(overall, 3), round(base, 3), round(novel, 3)


def reference_confusers(image_proto, baseline_sem, beta, num_cls, topk):
    fused_proto = F.normalize(beta * baseline_sem + (1.0 - beta) * image_proto, dim=-1)
    confuser_proto = {}
    for cls_id in range(num_cls):
        sims = torch.matmul(fused_proto, fused_proto[cls_id])
        sims[cls_id] = float("-inf")
        k = min(int(topk), max(1, num_cls - 1))
        conf_idx = torch.topk(sims, k=k).indices
        confuser_proto[cls_id] = fused_proto[conf_idx]
    return fused_proto, confuser_proto


def select_top_atom(desc_group, own_image_proto, own_text_proto, confuser_bank, strategy):
    if desc_group.shape[0] == 0:
        return None
    if strategy == "image_top1":
        score = torch.matmul(desc_group, own_image_proto)
    elif strategy == "prompt_top1":
        score = torch.matmul(desc_group, own_text_proto)
    elif strategy == "confuser_top1":
        own_score = torch.matmul(desc_group, own_image_proto)
        if confuser_bank is None or confuser_bank.numel() == 0:
            conf_score = torch.zeros_like(own_score)
        else:
            conf_score = torch.matmul(desc_group, confuser_bank.T).max(dim=1).values
        score = own_score - conf_score
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    return desc_group[int(torch.argmax(score).item())]


def build_bundle(merged_state, num_cls):
    return {
        "image_proto": F.normalize(merged_state["image_proto"][:num_cls], dim=-1),
        "text_features": F.normalize(merged_state["text_features"][:num_cls], dim=-1),
        "description_proto": F.normalize(merged_state["description_proto"][:num_cls], dim=-1),
        "description_features": F.normalize(merged_state["description_features"], dim=-1),
        "description_targets": merged_state["description_targets"],
        "cov_image": merged_state["cov_image"],
    }


def build_composer_target(bundle, num_cls, num_base_cls, beta, lambda_t, confuser_topk, strategy, compose_mode):
    image_proto = bundle["image_proto"][:num_cls]
    text_features = bundle["text_features"][:num_cls]
    description_proto = bundle["description_proto"][:num_cls]
    description_features = bundle["description_features"]
    description_targets = bundle["description_targets"]

    baseline_sem = F.normalize((1.0 - lambda_t) * text_features + lambda_t * description_proto, dim=-1)
    _, confuser_proto = reference_confusers(image_proto, baseline_sem, beta, num_cls, confuser_topk)

    target_sem = baseline_sem.clone()
    atom_alignment = []
    atom_margin = []
    for cls_id in range(num_base_cls, num_cls):
        desc_group = description_features[description_targets == cls_id]
        atom = select_top_atom(
            desc_group=desc_group,
            own_image_proto=image_proto[cls_id],
            own_text_proto=text_features[cls_id],
            confuser_bank=confuser_proto[cls_id],
            strategy=strategy,
        )
        if atom is None:
            continue
        atom_alignment.append(float(torch.dot(atom, image_proto[cls_id]).item()))
        if confuser_proto[cls_id].numel() == 0:
            atom_margin.append(float(torch.dot(atom, image_proto[cls_id]).item()))
        else:
            margin = torch.dot(atom, image_proto[cls_id]) - torch.matmul(confuser_proto[cls_id], atom).max()
            atom_margin.append(float(margin.item()))

        if compose_mode == "desc_atom_only":
            target_sem[cls_id] = atom
        elif compose_mode == "prompt_desc_atom":
            target_sem[cls_id] = average_vectors([text_features[cls_id], description_proto[cls_id], atom])
        elif compose_mode == "prompt_atom":
            target_sem[cls_id] = average_vectors([text_features[cls_id], atom])
        else:
            raise ValueError(f"unknown compose_mode: {compose_mode}")

    stats = {
        "selected_atom_alignment_mean": round(float(sum(atom_alignment) / max(1, len(atom_alignment))), 6),
        "selected_atom_margin_mean": round(float(sum(atom_margin) / max(1, len(atom_margin))), 6),
        "num_novel_classes": int(max(0, num_cls - num_base_cls)),
    }
    return baseline_sem, F.normalize(target_sem, dim=-1), stats


def interpolate_semantic(baseline_sem, composer_sem, num_base_cls, alpha):
    alpha = float(alpha)
    mixed = baseline_sem.clone()
    if num_base_cls < mixed.shape[0]:
        blended = (1.0 - alpha) * baseline_sem[num_base_cls:] + alpha * composer_sem[num_base_cls:]
        mixed[num_base_cls:] = F.normalize(blended, dim=-1)
    return mixed


def evaluate_semantic(bundle, semantic_proto, query_features, query_targets, num_cls, num_base_cls, beta, ensemble_alpha):
    image_proto = bundle["image_proto"][:num_cls]
    cov_image = bundle["cov_image"]
    knn_description_features = bundle["description_features"]
    knn_description_targets = bundle["description_targets"]

    prob_cov = F.softmax(compute_cov_logits(query_features, image_proto, cov_image) / 512.0, dim=-1)
    prob_knn = F.softmax(compute_knn_logits(query_features, knn_description_features, knn_description_targets, num_cls), dim=-1)
    full_probs, _ = compose_probs_from_semantic(
        query_features,
        semantic_proto,
        image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )
    fused_probs, _ = compose_probs_from_semantic(
        query_features,
        semantic_proto,
        image_proto,
        num_base_cls,
        beta,
        1.0,
        prob_cov,
        prob_knn,
    )
    full_acc, full_base, full_novel = accuracy_stats(full_probs, query_targets, num_base_cls)
    fused_acc, fused_base, fused_novel = accuracy_stats(fused_probs, query_targets, num_base_cls)
    knn_acc, knn_base, knn_novel = accuracy_stats(prob_knn, query_targets, num_base_cls)
    return {
        "full_acc": full_acc,
        "full_base_acc": full_base,
        "full_novel_acc": full_novel,
        "fused_only_acc": fused_acc,
        "fused_only_base_acc": fused_base,
        "fused_only_novel_acc": fused_novel,
        "knn_only_acc": knn_acc,
        "knn_only_base_acc": knn_base,
        "knn_only_novel_acc": knn_novel,
    }


def run_dataset(data_cfg, train_cfg, compose_modes, strategies, alphas, confuser_topk, seed_override=None):
    cfg = setup_cfg_with_override(data_cfg, train_cfg, seed_override=seed_override)
    out_dir = create_output_dir(cfg, train_cfg)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    start_time = time.time()
    data_manager, model, merged_state, final_task_id = extract_final_state(cfg)
    query_features, query_targets = collect_final_queries(cfg, data_manager, model, final_task_id)
    query_features = query_features.to(cfg.DEVICE.DEVICE_NAME)
    query_targets = query_targets.to(cfg.DEVICE.DEVICE_NAME)

    num_cls = max(data_manager.class_index_in_task[final_task_id]) + 1
    num_base_cls = len(data_manager.class_index_in_task[0])
    beta = float(cfg.DATASET.BETA)
    lambda_t = float(cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
    ensemble_alpha = float(cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.0)

    merged_state = {
        key: value.to(cfg.DEVICE.DEVICE_NAME) if isinstance(value, torch.Tensor) else value
        for key, value in merged_state.items()
    }
    bundle = build_bundle(merged_state, num_cls)

    results = {}
    for compose_mode in compose_modes:
        for strategy in strategies:
            baseline_sem, composer_sem, composer_stats = build_composer_target(
                bundle=bundle,
                num_cls=num_cls,
                num_base_cls=num_base_cls,
                beta=beta,
                lambda_t=lambda_t,
                confuser_topk=confuser_topk,
                strategy=strategy,
                compose_mode=compose_mode,
            )
            key = f"{compose_mode}__{strategy}"
            results[key] = {
                "composer_stats": composer_stats,
                "alphas": {},
            }
            for alpha in alphas:
                semantic_proto = interpolate_semantic(baseline_sem, composer_sem, num_base_cls, alpha)
                metrics = evaluate_semantic(
                    bundle=bundle,
                    semantic_proto=semantic_proto,
                    query_features=query_features,
                    query_targets=query_targets,
                    num_cls=num_cls,
                    num_base_cls=num_base_cls,
                    beta=beta,
                    ensemble_alpha=ensemble_alpha,
                )
                results[key]["alphas"][str(alpha)] = metrics

    payload = {
        "dataset": cfg.DATASET.NAME,
        "train_cfg": train_cfg,
        "runtime_sec": round(time.time() - start_time, 3),
        "compose_modes": list(compose_modes),
        "strategies": list(strategies),
        "alphas": [float(alpha) for alpha in alphas],
        "confuser_topk": int(confuser_topk),
        "results": results,
    }
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        f"dataset: {cfg.DATASET.NAME}",
        f"train_cfg: {train_cfg}",
        f"runtime_sec: {payload['runtime_sec']}",
    ]
    for key, value in payload["results"].items():
        lines.append(
            f"[{key}] atom_align={value['composer_stats']['selected_atom_alignment_mean']} atom_margin={value['composer_stats']['selected_atom_margin_mean']}"
        )
        for alpha, metrics in value["alphas"].items():
            lines.append(
                f"[{key}] alpha={alpha} full_acc={metrics['full_acc']} full_base_acc={metrics['full_base_acc']} full_novel_acc={metrics['full_novel_acc']} fused_only_novel_acc={metrics['fused_only_novel_acc']}"
            )
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "dataset": cfg.DATASET.NAME,
        "runtime_sec": payload["runtime_sec"],
        "results": payload["results"],
    }, indent=2))
    print(f"Saved semantic bank novel composer audit to: {out_dir}")
    return out_dir, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--compose_modes", nargs="+", default=["desc_atom_only", "prompt_desc_atom"])
    parser.add_argument("--strategies", nargs="+", default=["image_top1", "prompt_top1", "confuser_top1"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--confuser_topk", type=int, default=3)
    parser.add_argument("--seed_override", type=int, default=None)
    args = parser.parse_args()
    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        compose_modes=args.compose_modes,
        strategies=args.strategies,
        alphas=args.alphas,
        confuser_topk=args.confuser_topk,
        seed_override=args.seed_override,
    )


if __name__ == "__main__":
    main()
