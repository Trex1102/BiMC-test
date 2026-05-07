import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import setup_cfg
from models.query_graph_router import (
    QueryGraphBranchRouter,
    build_graph_router_inputs,
    build_knn_affinity,
    graph_smoothness,
    route_with_query_alphas,
    smooth_query_alphas,
)
from query_graph_router_killswitch import (
    accuracy_stats,
    baseline_probs,
    compute_branch_probs,
    optimize_graph_router_oracle,
)
from semantic_option2_killswitch import (
    collect_final_queries,
    extract_final_state,
    to_builtin,
)
from semantic_option2_learnability_audit import extract_base_context, sample_episode


def create_output_dir(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"query_graph_router_trainable_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def split_base_classes(base_classes, seed):
    base_classes = list(map(int, base_classes))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(base_classes)
    n = len(base_classes)
    n_train = max(1, int(round(0.6 * n)))
    n_val = max(1, int(round(0.2 * n)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train = base_classes[:n_train]
    val = base_classes[n_train:n_train + n_val]
    test = base_classes[n_train + n_val:]
    if not test:
        test = val[-1:]
        val = val[:-1]
    return train, val, test


def latest_graph_oracle_gain(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    oracle_root = output_root / cfg.DATASET.NAME / f"query_graph_router_killswitch_{Path(train_cfg).stem}"
    if not oracle_root.exists():
        return None, None
    results_files = sorted(oracle_root.glob("*/results.json"))
    if not results_files:
        return None, None
    latest = results_files[-1]
    with latest.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return float(payload.get("graph_router_oracle_gain", 0.0)), latest.parent


def build_episode_collection(base_state, candidate_classes, cfg, num_episodes, seed_offset):
    episodes = []
    candidate_classes = list(map(int, candidate_classes))
    pseudo_novel_per_episode = int(cfg.DATASET.NUM_INC_CLS)
    if len(candidate_classes) <= pseudo_novel_per_episode:
        raise RuntimeError("candidate class pool is too small for pseudo-novel episodes")

    for idx in range(num_episodes):
        g = torch.Generator()
        g.manual_seed(int(cfg.SEED) + int(seed_offset) + 9973 * (idx + 1))
        perm = torch.randperm(len(candidate_classes), generator=g).tolist()
        pseudo_novel = [candidate_classes[pos] for pos in perm[:pseudo_novel_per_episode]]
        episodes.append(
            sample_episode(
                base_state,
                pseudo_novel,
                cfg,
                seed=int(cfg.SEED) + int(seed_offset) + 1297 * (idx + 1),
            )
        )
    return episodes


def prepare_episode(
    episode,
    cfg,
    graph_k,
    teacher_smooth_weight,
    teacher_reg_weight,
    teacher_max_epochs,
    teacher_patience,
    teacher_lr,
    teacher_restarts,
):
    device = cfg.DEVICE.DEVICE_NAME
    state = {k: v.to(device) if torch.is_tensor(v) else v for k, v in episode["state"].items()}
    query_features = episode["query_features"].to(device)
    targets = episode["query_targets"].to(device)
    num_cls = state["image_proto"].shape[0]
    num_base_cls = int(episode["num_base_cls"])
    beta = float(cfg.DATASET.BETA)
    lambda_t = float(cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
    ensemble_alpha = float(cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.0)

    branch = compute_branch_probs(state, query_features, num_cls, beta, lambda_t)
    inputs = build_graph_router_inputs(
        query_features,
        branch["prob_fused"],
        branch["prob_cov"],
        branch["prob_knn"],
        num_base_cls,
    )
    affinity = build_knn_affinity(inputs["query_features"], k=graph_k, self_loop=True)
    baseline = baseline_probs(
        branch["prob_fused"],
        branch["prob_cov"],
        branch["prob_knn"],
        num_base_cls,
        ensemble_alpha,
    )
    baseline_full, baseline_base, baseline_novel = accuracy_stats(baseline, targets, num_base_cls)

    teacher = optimize_graph_router_oracle(
        query_features=inputs["query_features"],
        prob_fused=branch["prob_fused"],
        prob_cov=branch["prob_cov"],
        prob_knn=branch["prob_knn"],
        targets=targets,
        num_base_cls=num_base_cls,
        baseline_alpha=ensemble_alpha,
        graph_k=graph_k,
        smooth_weight=teacher_smooth_weight,
        reg_weight=teacher_reg_weight,
        max_epochs=teacher_max_epochs,
        patience=teacher_patience,
        lr=teacher_lr,
        restarts=teacher_restarts,
    )
    teacher_full, teacher_base, teacher_novel = accuracy_stats(teacher["probs"], targets, num_base_cls)

    return {
        "query_features": inputs["query_features"],
        "stats_features": inputs["stats_features"],
        "targets": targets,
        "prob_fused": branch["prob_fused"],
        "prob_cov": branch["prob_cov"],
        "prob_knn": branch["prob_knn"],
        "affinity": affinity,
        "num_base_cls": num_base_cls,
        "ensemble_alpha": ensemble_alpha,
        "teacher_alpha_base": teacher["alpha_base"],
        "teacher_alpha_novel": teacher["alpha_novel"],
        "baseline_full_acc": baseline_full,
        "baseline_base_acc": baseline_base,
        "baseline_novel_acc": baseline_novel,
        "teacher_full_acc": teacher_full,
        "teacher_base_acc": teacher_base,
        "teacher_novel_acc": teacher_novel,
        "teacher_gain": teacher_full - baseline_full,
    }


def evaluate_router(model, episode, smooth_mix, smooth_steps):
    outputs = model(episode["query_features"], episode["stats_features"])
    alpha_base, alpha_novel = smooth_query_alphas(
        outputs["alpha_base"],
        outputs["alpha_novel"],
        episode["affinity"],
        mix=smooth_mix,
        steps=smooth_steps,
    )
    probs = route_with_query_alphas(
        episode["prob_fused"],
        episode["prob_cov"],
        episode["prob_knn"],
        episode["num_base_cls"],
        alpha_base,
        alpha_novel,
    )
    return {
        "probs": probs,
        "alpha_base": alpha_base,
        "alpha_novel": alpha_novel,
    }


def train_router(
    model,
    train_episodes,
    val_episodes,
    smooth_mix,
    smooth_steps,
    distill_weight,
    ce_weight,
    reg_weight,
    graph_weight,
    lr,
    max_epochs,
    patience,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val = -1.0
    stale = 0

    for _ in range(max_epochs):
        model.train()
        for episode in train_episodes:
            optimizer.zero_grad()
            outputs = model(episode["query_features"], episode["stats_features"])
            alpha_base_raw = outputs["alpha_base"]
            alpha_novel_raw = outputs["alpha_novel"]
            alpha_base, alpha_novel = smooth_query_alphas(
                alpha_base_raw,
                alpha_novel_raw,
                episode["affinity"],
                mix=smooth_mix,
                steps=smooth_steps,
            )
            probs = route_with_query_alphas(
                episode["prob_fused"],
                episode["prob_cov"],
                episode["prob_knn"],
                episode["num_base_cls"],
                alpha_base,
                alpha_novel,
            )
            ce = F.nll_loss(torch.log(torch.clamp(probs, min=1e-12)), episode["targets"])
            distill = F.mse_loss(alpha_base_raw, episode["teacher_alpha_base"]) + F.mse_loss(
                alpha_novel_raw, episode["teacher_alpha_novel"]
            )
            reg = ((alpha_base_raw - episode["ensemble_alpha"]) ** 2).mean() + (
                (alpha_novel_raw - episode["ensemble_alpha"]) ** 2
            ).mean()
            smooth = graph_smoothness(alpha_base_raw, episode["affinity"]) + graph_smoothness(
                alpha_novel_raw, episode["affinity"]
            )
            loss = (
                float(ce_weight) * ce
                + float(distill_weight) * distill
                + float(reg_weight) * reg
                + float(graph_weight) * smooth
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_accs = []
            for episode in val_episodes:
                routed = evaluate_router(model, episode, smooth_mix=smooth_mix, smooth_steps=smooth_steps)
                full_acc, _, _ = accuracy_stats(routed["probs"], episode["targets"], episode["num_base_cls"])
                val_accs.append(full_acc)
            val_score = float(np.mean(val_accs)) if val_accs else 0.0

        if val_score > best_val + 1e-6:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


def summarize_episode_set(model, episodes, smooth_mix, smooth_steps):
    model.eval()
    baseline_full = []
    teacher_full = []
    routed_full = []
    with torch.no_grad():
        for episode in episodes:
            baseline_full.append(float(episode["baseline_full_acc"]))
            teacher_full.append(float(episode["teacher_full_acc"]))
            routed = evaluate_router(model, episode, smooth_mix=smooth_mix, smooth_steps=smooth_steps)
            full_acc, _, _ = accuracy_stats(routed["probs"], episode["targets"], episode["num_base_cls"])
            routed_full.append(float(full_acc))
    if not episodes:
        return {}
    return {
        "baseline_full_acc_mean": float(np.mean(baseline_full)),
        "teacher_full_acc_mean": float(np.mean(teacher_full)),
        "router_full_acc_mean": float(np.mean(routed_full)),
    }


def run_dataset(
    data_cfg,
    train_cfg,
    num_train_episodes,
    num_val_episodes,
    graph_k,
    teacher_smooth_weight,
    teacher_reg_weight,
    teacher_max_epochs,
    teacher_patience,
    teacher_lr,
    teacher_restarts,
    smooth_mix,
    smooth_steps,
    distill_weight,
    ce_weight,
    reg_weight,
    graph_weight,
    lr,
    max_epochs,
    patience,
    hidden_dim,
    query_proj_dim,
):
    cfg = setup_cfg(data_cfg, train_cfg)
    out_dir = create_output_dir(cfg, train_cfg)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    start_time = time.time()
    base_context = extract_base_context(cfg)
    train_classes, val_classes, _ = split_base_classes(base_context["base_state"]["class_index"], cfg.SEED)
    train_eps_raw = build_episode_collection(base_context["base_state"], train_classes, cfg, num_train_episodes, 0)
    val_eps_raw = build_episode_collection(base_context["base_state"], val_classes, cfg, num_val_episodes, 10000)

    train_episodes = [
        prepare_episode(
            ep,
            cfg,
            graph_k,
            teacher_smooth_weight,
            teacher_reg_weight,
            teacher_max_epochs,
            teacher_patience,
            teacher_lr,
            teacher_restarts,
        )
        for ep in train_eps_raw
    ]
    val_episodes = [
        prepare_episode(
            ep,
            cfg,
            graph_k,
            teacher_smooth_weight,
            teacher_reg_weight,
            teacher_max_epochs,
            teacher_patience,
            teacher_lr,
            teacher_restarts,
        )
        for ep in val_eps_raw
    ]

    model = QueryGraphBranchRouter(
        query_dim=train_episodes[0]["query_features"].shape[1],
        stats_dim=train_episodes[0]["stats_features"].shape[1],
        hidden_dim=hidden_dim,
        query_proj_dim=query_proj_dim,
    ).to(cfg.DEVICE.DEVICE_NAME)

    best_val = train_router(
        model=model,
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        smooth_mix=smooth_mix,
        smooth_steps=smooth_steps,
        distill_weight=distill_weight,
        ce_weight=ce_weight,
        reg_weight=reg_weight,
        graph_weight=graph_weight,
        lr=lr,
        max_epochs=max_epochs,
        patience=patience,
    )

    data_manager, bimc_model, merged_state, final_task_id = extract_final_state(cfg)
    query_features, query_targets = collect_final_queries(cfg, data_manager, bimc_model, final_task_id)
    num_cls = max(data_manager.class_index_in_task[final_task_id]) + 1
    num_base_cls = len(data_manager.class_index_in_task[0])
    beta = float(cfg.DATASET.BETA)
    lambda_t = float(cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
    ensemble_alpha = float(cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.0)

    query_features = query_features.to(cfg.DEVICE.DEVICE_NAME)
    targets = query_targets.to(cfg.DEVICE.DEVICE_NAME)
    branch = compute_branch_probs(merged_state, query_features, num_cls, beta, lambda_t)
    final_inputs = build_graph_router_inputs(
        query_features,
        branch["prob_fused"],
        branch["prob_cov"],
        branch["prob_knn"],
        num_base_cls,
    )
    final_affinity = build_knn_affinity(final_inputs["query_features"], k=graph_k, self_loop=True)
    baseline = baseline_probs(
        branch["prob_fused"],
        branch["prob_cov"],
        branch["prob_knn"],
        num_base_cls,
        ensemble_alpha,
    )
    baseline_full, baseline_base, baseline_novel = accuracy_stats(baseline, targets, num_base_cls)

    model.eval()
    with torch.no_grad():
        outputs = model(final_inputs["query_features"], final_inputs["stats_features"])
        alpha_base, alpha_novel = smooth_query_alphas(
            outputs["alpha_base"],
            outputs["alpha_novel"],
            final_affinity,
            mix=smooth_mix,
            steps=smooth_steps,
        )
        routed = route_with_query_alphas(
            branch["prob_fused"],
            branch["prob_cov"],
            branch["prob_knn"],
            num_base_cls,
            alpha_base,
            alpha_novel,
        )
    final_full, final_base, final_novel = accuracy_stats(routed, targets, num_base_cls)

    graph_oracle_gain, graph_oracle_path = latest_graph_oracle_gain(cfg, train_cfg)
    train_summary = summarize_episode_set(model, train_episodes, smooth_mix, smooth_steps)
    val_summary = summarize_episode_set(model, val_episodes, smooth_mix, smooth_steps)

    payload = {
        "dataset": cfg.DATASET.NAME,
        "train_cfg": train_cfg,
        "runtime_sec": round(time.time() - start_time, 3),
        "num_train_episodes": int(num_train_episodes),
        "num_val_episodes": int(num_val_episodes),
        "graph_k": int(graph_k),
        "teacher_smooth_weight": float(teacher_smooth_weight),
        "teacher_reg_weight": float(teacher_reg_weight),
        "teacher_max_epochs": int(teacher_max_epochs),
        "teacher_patience": int(teacher_patience),
        "teacher_lr": float(teacher_lr),
        "teacher_restarts": int(teacher_restarts),
        "smooth_mix": float(smooth_mix),
        "smooth_steps": int(smooth_steps),
        "distill_weight": float(distill_weight),
        "ce_weight": float(ce_weight),
        "reg_weight": float(reg_weight),
        "graph_weight": float(graph_weight),
        "lr": float(lr),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "hidden_dim": int(hidden_dim),
        "query_proj_dim": int(query_proj_dim),
        "best_val_acc": round(float(best_val), 3),
        "baseline_full_acc": round(float(baseline_full), 3),
        "baseline_base_acc": round(float(baseline_base), 3),
        "baseline_novel_acc": round(float(baseline_novel), 3),
        "trainable_full_acc": round(float(final_full), 3),
        "trainable_base_acc": round(float(final_base), 3),
        "trainable_novel_acc": round(float(final_novel), 3),
        "trainable_gain": round(float(final_full - baseline_full), 3),
        "mean_alpha_base": round(float(alpha_base.mean().item()), 4),
        "mean_alpha_novel": round(float(alpha_novel.mean().item()), 4),
        "std_alpha_base": round(float(alpha_base.std().item()), 4),
        "std_alpha_novel": round(float(alpha_novel.std().item()), 4),
        "graph_oracle_gain": None if graph_oracle_gain is None else round(float(graph_oracle_gain), 3),
        "graph_oracle_path": None if graph_oracle_path is None else str(graph_oracle_path),
        "train_episode_summary": train_summary,
        "val_episode_summary": val_summary,
    }
    if graph_oracle_gain is not None and abs(graph_oracle_gain) > 1e-8:
        payload["recovery_vs_graph_oracle"] = round(float((final_full - baseline_full) / graph_oracle_gain * 100.0), 1)
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [
        f"dataset: {cfg.DATASET.NAME}",
        f"baseline_full_acc: {payload['baseline_full_acc']}",
        f"trainable_full_acc: {payload['trainable_full_acc']}",
        f"trainable_gain: {payload['trainable_gain']}",
        f"graph_oracle_gain: {payload['graph_oracle_gain']}",
        f"baseline_base_acc: {payload['baseline_base_acc']}",
        f"baseline_novel_acc: {payload['baseline_novel_acc']}",
        f"trainable_base_acc: {payload['trainable_base_acc']}",
        f"trainable_novel_acc: {payload['trainable_novel_acc']}",
        f"mean_alpha_base: {payload['mean_alpha_base']}",
        f"mean_alpha_novel: {payload['mean_alpha_novel']}",
        f"std_alpha_base: {payload['std_alpha_base']}",
        f"std_alpha_novel: {payload['std_alpha_novel']}",
        f"best_val_acc: {payload['best_val_acc']}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Trainable graph-smoothed query router for BiMC.")
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--num_train_episodes", type=int, default=10)
    parser.add_argument("--num_val_episodes", type=int, default=4)
    parser.add_argument("--graph_k", type=int, default=10)
    parser.add_argument("--teacher_smooth_weight", type=float, default=0.2)
    parser.add_argument("--teacher_reg_weight", type=float, default=0.01)
    parser.add_argument("--teacher_max_epochs", type=int, default=200)
    parser.add_argument("--teacher_patience", type=int, default=30)
    parser.add_argument("--teacher_lr", type=float, default=0.05)
    parser.add_argument("--teacher_restarts", type=int, default=4)
    parser.add_argument("--smooth_mix", type=float, default=0.5)
    parser.add_argument("--smooth_steps", type=int, default=2)
    parser.add_argument("--distill_weight", type=float, default=1.0)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--reg_weight", type=float, default=0.01)
    parser.add_argument("--graph_weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--query_proj_dim", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        num_train_episodes=args.num_train_episodes,
        num_val_episodes=args.num_val_episodes,
        graph_k=args.graph_k,
        teacher_smooth_weight=args.teacher_smooth_weight,
        teacher_reg_weight=args.teacher_reg_weight,
        teacher_max_epochs=args.teacher_max_epochs,
        teacher_patience=args.teacher_patience,
        teacher_lr=args.teacher_lr,
        teacher_restarts=args.teacher_restarts,
        smooth_mix=args.smooth_mix,
        smooth_steps=args.smooth_steps,
        distill_weight=args.distill_weight,
        ce_weight=args.ce_weight,
        reg_weight=args.reg_weight,
        graph_weight=args.graph_weight,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        query_proj_dim=args.query_proj_dim,
    )


if __name__ == "__main__":
    main()
