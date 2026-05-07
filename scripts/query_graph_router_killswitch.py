import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import setup_cfg
from models.query_graph_router import (
    build_knn_affinity,
    graph_smoothness,
    route_with_query_alphas,
)
from semantic_option2_killswitch import (
    collect_final_queries,
    compute_cov_logits,
    compute_knn_logits,
    extract_final_state,
    to_builtin,
)


def create_output_dir(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"query_graph_router_killswitch_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def accuracy_stats(probs, targets, num_base_cls):
    preds = torch.argmax(probs, dim=1)
    full_acc = float((preds == targets).float().mean().item() * 100.0)
    base_mask = targets < int(num_base_cls)
    novel_mask = ~base_mask
    base_acc = float((preds[base_mask] == targets[base_mask]).float().mean().item() * 100.0) if base_mask.any() else 0.0
    novel_acc = float((preds[novel_mask] == targets[novel_mask]).float().mean().item() * 100.0) if novel_mask.any() else 0.0
    return full_acc, base_acc, novel_acc


def compute_branch_probs(merged_state, query_features, num_cls, beta, lambda_t):
    device = query_features.device
    image_proto = F.normalize(merged_state["image_proto"][:num_cls].to(device), dim=-1)
    text_features = F.normalize(merged_state["text_features"][:num_cls].to(device), dim=-1)
    description_proto = F.normalize(merged_state["description_proto"][:num_cls].to(device), dim=-1)
    description_features = F.normalize(merged_state["description_features"].to(device), dim=-1)
    description_targets = merged_state["description_targets"].to(device)
    cov_image = merged_state["cov_image"].to(device)

    semantic_proto = F.normalize((1.0 - lambda_t) * text_features + lambda_t * description_proto, dim=-1)
    fused_proto = F.normalize(beta * semantic_proto + (1.0 - beta) * image_proto, dim=-1)
    logits_fused = torch.matmul(query_features, fused_proto.T)
    logits_cov = compute_cov_logits(query_features, image_proto, cov_image)
    logits_knn = compute_knn_logits(query_features, description_features, description_targets, num_cls)

    return {
        "prob_fused": F.softmax(logits_fused, dim=-1),
        "prob_cov": F.softmax(logits_cov / 512.0, dim=-1),
        "prob_knn": F.softmax(logits_knn, dim=-1),
    }


def baseline_probs(prob_fused, prob_cov, prob_knn, num_base_cls, ensemble_alpha):
    alpha_base = prob_fused.new_full((prob_fused.shape[0],), float(ensemble_alpha))
    alpha_novel = prob_fused.new_full((prob_fused.shape[0],), float(ensemble_alpha))
    return route_with_query_alphas(prob_fused, prob_cov, prob_knn, num_base_cls, alpha_base, alpha_novel)


def discrete_oracle(prob_fused, prob_cov, prob_knn, targets, num_base_cls):
    fused_pred = torch.argmax(prob_fused, dim=1)
    cov_pred = torch.argmax(prob_cov, dim=1)
    knn_pred = torch.argmax(prob_knn, dim=1)
    choose_fused_base = fused_pred.eq(targets) | (~cov_pred.eq(targets))
    choose_fused_novel = fused_pred.eq(targets) | (~knn_pred.eq(targets))
    alpha_base = choose_fused_base.float()
    alpha_novel = choose_fused_novel.float()
    probs = route_with_query_alphas(prob_fused, prob_cov, prob_knn, num_base_cls, alpha_base, alpha_novel)
    return {
        "probs": probs,
        "alpha_base": alpha_base,
        "alpha_novel": alpha_novel,
    }


def optimize_graph_router_oracle(
    query_features,
    prob_fused,
    prob_cov,
    prob_knn,
    targets,
    num_base_cls,
    baseline_alpha,
    graph_k,
    smooth_weight,
    reg_weight,
    max_epochs,
    patience,
    lr,
    restarts,
):
    device = query_features.device
    affinity = build_knn_affinity(query_features, k=graph_k, self_loop=True)

    best = None
    prior_logit = math.log(float(baseline_alpha) / max(1e-8, 1.0 - float(baseline_alpha)))

    for restart in range(restarts):
        torch.manual_seed(int(7000 + restart))
        alpha_base_logit = torch.nn.Parameter(torch.full((query_features.shape[0],), float(prior_logit), device=device))
        alpha_novel_logit = torch.nn.Parameter(torch.full((query_features.shape[0],), float(prior_logit), device=device))
        optimizer = torch.optim.Adam([alpha_base_logit, alpha_novel_logit], lr=lr)

        best_state = None
        best_acc = -1.0
        best_ce = float("inf")
        stale = 0

        for _ in range(max_epochs):
            optimizer.zero_grad()
            alpha_base = torch.sigmoid(alpha_base_logit)
            alpha_novel = torch.sigmoid(alpha_novel_logit)
            routed = route_with_query_alphas(prob_fused, prob_cov, prob_knn, num_base_cls, alpha_base, alpha_novel)
            ce = F.nll_loss(torch.log(torch.clamp(routed, min=1e-12)), targets)
            reg = ((alpha_base - baseline_alpha) ** 2).mean() + ((alpha_novel - baseline_alpha) ** 2).mean()
            smooth = graph_smoothness(alpha_base, affinity) + graph_smoothness(alpha_novel, affinity)
            loss = ce + float(reg_weight) * reg + float(smooth_weight) * smooth
            loss.backward()
            torch.nn.utils.clip_grad_norm_([alpha_base_logit, alpha_novel_logit], 5.0)
            optimizer.step()

            with torch.no_grad():
                acc = float((torch.argmax(routed, dim=1) == targets).float().mean().item() * 100.0)
                ce_val = float(ce.item())
                if acc > best_acc + 1e-6 or (abs(acc - best_acc) <= 1e-6 and ce_val < best_ce - 1e-6):
                    best_acc = acc
                    best_ce = ce_val
                    best_state = {
                        "alpha_base": alpha_base.detach().clone(),
                        "alpha_novel": alpha_novel.detach().clone(),
                        "probs": routed.detach().clone(),
                        "acc": best_acc,
                        "ce": best_ce,
                    }
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        break

        if best is None or best_state["acc"] > best["acc"] + 1e-6 or (abs(best_state["acc"] - best["acc"]) <= 1e-6 and best_state["ce"] < best["ce"] - 1e-6):
            best = best_state

    return best


def run_dataset(data_cfg, train_cfg, graph_k, smooth_weight, reg_weight, max_epochs, patience, lr, restarts):
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

    query_features = query_features.to(cfg.DEVICE.DEVICE_NAME)
    targets = query_targets.to(cfg.DEVICE.DEVICE_NAME)
    branch = compute_branch_probs(merged_state, query_features, num_cls, beta, lambda_t)
    prob_fused = branch["prob_fused"]
    prob_cov = branch["prob_cov"]
    prob_knn = branch["prob_knn"]

    baseline = baseline_probs(prob_fused, prob_cov, prob_knn, num_base_cls, ensemble_alpha)
    baseline_full, baseline_base, baseline_novel = accuracy_stats(baseline, targets, num_base_cls)

    graph_oracle = optimize_graph_router_oracle(
        query_features=query_features,
        prob_fused=prob_fused,
        prob_cov=prob_cov,
        prob_knn=prob_knn,
        targets=targets,
        num_base_cls=num_base_cls,
        baseline_alpha=ensemble_alpha,
        graph_k=graph_k,
        smooth_weight=smooth_weight,
        reg_weight=reg_weight,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        restarts=restarts,
    )
    graph_full, graph_base, graph_novel = accuracy_stats(graph_oracle["probs"], targets, num_base_cls)

    discrete = discrete_oracle(prob_fused, prob_cov, prob_knn, targets, num_base_cls)
    disc_full, disc_base, disc_novel = accuracy_stats(discrete["probs"], targets, num_base_cls)

    payload = {
        "dataset": cfg.DATASET.NAME,
        "train_cfg": train_cfg,
        "runtime_sec": round(time.time() - start_time, 3),
        "num_seen_classes": int(num_cls),
        "num_base_classes": int(num_base_cls),
        "beta": beta,
        "lambda_t": lambda_t,
        "ensemble_alpha": ensemble_alpha,
        "graph_k": int(graph_k),
        "smooth_weight": float(smooth_weight),
        "reg_weight": float(reg_weight),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "lr": float(lr),
        "restarts": int(restarts),
        "baseline_full_acc": round(baseline_full, 3),
        "baseline_base_acc": round(baseline_base, 3),
        "baseline_novel_acc": round(baseline_novel, 3),
        "graph_router_oracle_full_acc": round(graph_full, 3),
        "graph_router_oracle_base_acc": round(graph_base, 3),
        "graph_router_oracle_novel_acc": round(graph_novel, 3),
        "graph_router_oracle_gain": round(graph_full - baseline_full, 3),
        "discrete_oracle_full_acc": round(disc_full, 3),
        "discrete_oracle_base_acc": round(disc_base, 3),
        "discrete_oracle_novel_acc": round(disc_novel, 3),
        "discrete_oracle_gain": round(disc_full - baseline_full, 3),
        "mean_alpha_base": round(float(graph_oracle["alpha_base"].mean().item()), 4),
        "mean_alpha_novel": round(float(graph_oracle["alpha_novel"].mean().item()), 4),
        "std_alpha_base": round(float(graph_oracle["alpha_base"].std().item()), 4),
        "std_alpha_novel": round(float(graph_oracle["alpha_novel"].std().item()), 4),
    }
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [
        f"dataset: {cfg.DATASET.NAME}",
        f"baseline_full_acc: {payload['baseline_full_acc']}",
        f"graph_router_oracle_full_acc: {payload['graph_router_oracle_full_acc']}",
        f"graph_router_oracle_gain: {payload['graph_router_oracle_gain']}",
        f"discrete_oracle_gain: {payload['discrete_oracle_gain']}",
        f"baseline_base_acc: {payload['baseline_base_acc']}",
        f"baseline_novel_acc: {payload['baseline_novel_acc']}",
        f"graph_router_oracle_base_acc: {payload['graph_router_oracle_base_acc']}",
        f"graph_router_oracle_novel_acc: {payload['graph_router_oracle_novel_acc']}",
        f"mean_alpha_base: {payload['mean_alpha_base']}",
        f"mean_alpha_novel: {payload['mean_alpha_novel']}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "dataset": cfg.DATASET.NAME,
        "baseline_full_acc": payload["baseline_full_acc"],
        "graph_router_oracle_full_acc": payload["graph_router_oracle_full_acc"],
        "graph_router_oracle_gain": payload["graph_router_oracle_gain"],
        "discrete_oracle_gain": payload["discrete_oracle_gain"],
        "graph_router_oracle_base_acc": payload["graph_router_oracle_base_acc"],
        "graph_router_oracle_novel_acc": payload["graph_router_oracle_novel_acc"],
    }, indent=2))
    print(f"Saved query graph router killswitch results to: {out_dir}")
    return out_dir, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--graph_k", type=int, default=10)
    parser.add_argument("--smooth_weight", type=float, default=0.2)
    parser.add_argument("--reg_weight", type=float, default=0.01)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--restarts", type=int, default=4)
    args = parser.parse_args()
    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        graph_k=args.graph_k,
        smooth_weight=args.smooth_weight,
        reg_weight=args.reg_weight,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        restarts=args.restarts,
    )


if __name__ == "__main__":
    main()
