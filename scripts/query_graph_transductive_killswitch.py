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
from models.query_graph_router import build_knn_affinity, graph_smoothness, route_with_query_alphas
from models.query_graph_transductive import distribution_smoothness, entropy_loss, symmetric_kl
from query_graph_router_killswitch import accuracy_stats, baseline_probs, compute_branch_probs, discrete_oracle
from semantic_option2_killswitch import (
    collect_final_queries,
    extract_final_state,
    to_builtin,
)


def create_output_dir(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"query_graph_transductive_killswitch_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def optimize_transductive_oracle(
    query_features,
    prob_fused,
    prob_cov,
    prob_knn,
    targets,
    num_base_cls,
    baseline_alpha,
    graph_k,
    graph_weight,
    anchor_weight,
    prior_weight,
    alpha_graph_weight,
    entropy_weight,
    max_epochs,
    patience,
    lr,
    restarts,
):
    device = query_features.device
    affinity = build_knn_affinity(query_features, k=graph_k, self_loop=True)
    prior_logit = math.log(float(baseline_alpha) / max(1e-8, 1.0 - float(baseline_alpha)))
    baseline = baseline_probs(prob_fused, prob_cov, prob_knn, num_base_cls, baseline_alpha).float()
    init_logits = torch.log(torch.clamp(baseline, min=1e-12))

    best = None
    for restart in range(restarts):
        torch.manual_seed(int(8100 + restart))
        alpha_base_logit = torch.nn.Parameter(torch.full((query_features.shape[0],), float(prior_logit), device=device))
        alpha_novel_logit = torch.nn.Parameter(torch.full((query_features.shape[0],), float(prior_logit), device=device))
        session_logits = torch.nn.Parameter(init_logits.detach().clone())
        optimizer = torch.optim.Adam([alpha_base_logit, alpha_novel_logit, session_logits], lr=lr)

        with torch.no_grad():
            init_alpha_base = torch.sigmoid(alpha_base_logit)
            init_alpha_novel = torch.sigmoid(alpha_novel_logit)
            init_routed = route_with_query_alphas(
                prob_fused,
                prob_cov,
                prob_knn,
                num_base_cls,
                init_alpha_base,
                init_alpha_novel,
            )
            init_session = F.softmax(session_logits, dim=1)
            init_acc = float((torch.argmax(init_session, dim=1) == targets).float().mean().item() * 100.0)
            init_ce = float(F.nll_loss(torch.log(torch.clamp(init_session, min=1e-12)), targets).item())

        best_state = {
            "alpha_base": init_alpha_base.detach().clone(),
            "alpha_novel": init_alpha_novel.detach().clone(),
            "routed_probs": init_routed.detach().clone(),
            "session_probs": init_session.detach().clone(),
            "acc": init_acc,
            "ce": init_ce,
        }
        best_acc = init_acc
        best_ce = init_ce
        stale = 0

        for _ in range(max_epochs):
            optimizer.zero_grad()
            alpha_base = torch.sigmoid(alpha_base_logit)
            alpha_novel = torch.sigmoid(alpha_novel_logit)
            routed = route_with_query_alphas(prob_fused, prob_cov, prob_knn, num_base_cls, alpha_base, alpha_novel)
            session_probs = F.softmax(session_logits, dim=1)

            ce = F.nll_loss(torch.log(torch.clamp(session_probs, min=1e-12)), targets)
            anchor = symmetric_kl(session_probs, routed)
            graph = distribution_smoothness(session_probs, affinity)
            alpha_graph = graph_smoothness(alpha_base, affinity) + graph_smoothness(alpha_novel, affinity)
            prior = ((alpha_base - baseline_alpha) ** 2).mean() + ((alpha_novel - baseline_alpha) ** 2).mean()
            ent = entropy_loss(session_probs)
            loss = (
                ce
                + float(anchor_weight) * anchor
                + float(graph_weight) * graph
                + float(alpha_graph_weight) * alpha_graph
                + float(prior_weight) * prior
                + float(entropy_weight) * ent
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([alpha_base_logit, alpha_novel_logit, session_logits], 5.0)
            optimizer.step()

            with torch.no_grad():
                acc = float((torch.argmax(session_probs, dim=1) == targets).float().mean().item() * 100.0)
                ce_val = float(ce.item())
                if acc > best_acc + 1e-6 or (abs(acc - best_acc) <= 1e-6 and ce_val < best_ce - 1e-6):
                    best_acc = acc
                    best_ce = ce_val
                    best_state = {
                        "alpha_base": alpha_base.detach().clone(),
                        "alpha_novel": alpha_novel.detach().clone(),
                        "routed_probs": routed.detach().clone(),
                        "session_probs": session_probs.detach().clone(),
                        "acc": best_acc,
                        "ce": best_ce,
                    }
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        break

        if best is None or best_state["acc"] > best["acc"] + 1e-6 or (
            abs(best_state["acc"] - best["acc"]) <= 1e-6 and best_state["ce"] < best["ce"] - 1e-6
        ):
            best = best_state

    return best


def run_dataset(
    data_cfg,
    train_cfg,
    graph_k,
    graph_weight,
    anchor_weight,
    prior_weight,
    alpha_graph_weight,
    entropy_weight,
    max_epochs,
    patience,
    lr,
    restarts,
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

    query_features = query_features.to(cfg.DEVICE.DEVICE_NAME)
    targets = query_targets.to(cfg.DEVICE.DEVICE_NAME)
    branch = compute_branch_probs(merged_state, query_features, num_cls, beta, lambda_t)
    prob_fused = branch["prob_fused"]
    prob_cov = branch["prob_cov"]
    prob_knn = branch["prob_knn"]

    baseline = baseline_probs(prob_fused, prob_cov, prob_knn, num_base_cls, ensemble_alpha)
    baseline_full, baseline_base, baseline_novel = accuracy_stats(baseline, targets, num_base_cls)

    oracle = optimize_transductive_oracle(
        query_features=query_features,
        prob_fused=prob_fused,
        prob_cov=prob_cov,
        prob_knn=prob_knn,
        targets=targets,
        num_base_cls=num_base_cls,
        baseline_alpha=ensemble_alpha,
        graph_k=graph_k,
        graph_weight=graph_weight,
        anchor_weight=anchor_weight,
        prior_weight=prior_weight,
        alpha_graph_weight=alpha_graph_weight,
        entropy_weight=entropy_weight,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        restarts=restarts,
    )
    oracle_full, oracle_base, oracle_novel = accuracy_stats(oracle["session_probs"], targets, num_base_cls)
    routed_full, routed_base, routed_novel = accuracy_stats(oracle["routed_probs"], targets, num_base_cls)

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
        "graph_weight": float(graph_weight),
        "anchor_weight": float(anchor_weight),
        "prior_weight": float(prior_weight),
        "alpha_graph_weight": float(alpha_graph_weight),
        "entropy_weight": float(entropy_weight),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "lr": float(lr),
        "restarts": int(restarts),
        "baseline_full_acc": round(baseline_full, 3),
        "baseline_base_acc": round(baseline_base, 3),
        "baseline_novel_acc": round(baseline_novel, 3),
        "transductive_oracle_full_acc": round(oracle_full, 3),
        "transductive_oracle_base_acc": round(oracle_base, 3),
        "transductive_oracle_novel_acc": round(oracle_novel, 3),
        "transductive_oracle_gain": round(oracle_full - baseline_full, 3),
        "routed_anchor_full_acc": round(routed_full, 3),
        "routed_anchor_base_acc": round(routed_base, 3),
        "routed_anchor_novel_acc": round(routed_novel, 3),
        "routed_anchor_gain": round(routed_full - baseline_full, 3),
        "discrete_oracle_full_acc": round(disc_full, 3),
        "discrete_oracle_base_acc": round(disc_base, 3),
        "discrete_oracle_novel_acc": round(disc_novel, 3),
        "discrete_oracle_gain": round(disc_full - baseline_full, 3),
        "mean_alpha_base": round(float(oracle["alpha_base"].mean().item()), 4),
        "mean_alpha_novel": round(float(oracle["alpha_novel"].mean().item()), 4),
        "std_alpha_base": round(float(oracle["alpha_base"].std().item()), 4),
        "std_alpha_novel": round(float(oracle["alpha_novel"].std().item()), 4),
    }
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [
        f"dataset: {cfg.DATASET.NAME}",
        f"baseline_full_acc: {payload['baseline_full_acc']}",
        f"transductive_oracle_full_acc: {payload['transductive_oracle_full_acc']}",
        f"transductive_oracle_gain: {payload['transductive_oracle_gain']}",
        f"routed_anchor_gain: {payload['routed_anchor_gain']}",
        f"discrete_oracle_gain: {payload['discrete_oracle_gain']}",
        f"baseline_base_acc: {payload['baseline_base_acc']}",
        f"baseline_novel_acc: {payload['baseline_novel_acc']}",
        f"transductive_oracle_base_acc: {payload['transductive_oracle_base_acc']}",
        f"transductive_oracle_novel_acc: {payload['transductive_oracle_novel_acc']}",
        f"mean_alpha_base: {payload['mean_alpha_base']}",
        f"mean_alpha_novel: {payload['mean_alpha_novel']}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Kill switch for transductive graph-based BiMC session optimizer.")
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--graph_k", type=int, default=10)
    parser.add_argument("--graph_weight", type=float, default=0.3)
    parser.add_argument("--anchor_weight", type=float, default=0.3)
    parser.add_argument("--prior_weight", type=float, default=0.01)
    parser.add_argument("--alpha_graph_weight", type=float, default=0.1)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--max_epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--restarts", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        graph_k=args.graph_k,
        graph_weight=args.graph_weight,
        anchor_weight=args.anchor_weight,
        prior_weight=args.prior_weight,
        alpha_graph_weight=args.alpha_graph_weight,
        entropy_weight=args.entropy_weight,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        restarts=args.restarts,
    )


if __name__ == "__main__":
    main()
