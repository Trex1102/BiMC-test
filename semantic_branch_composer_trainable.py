import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import setup_cfg
from models.semantic_branch_composer import (
    NovelSemanticComposer,
    build_component_features,
    build_context_features,
)
from semantic_option2_killswitch import (
    collect_final_queries,
    compose_probs_from_semantic,
    compute_cov_logits,
    compute_knn_logits,
    extract_final_state,
    normalize_vector,
    select_diverse_description_atoms,
    to_builtin,
)
from semantic_option2_learnability_audit import extract_base_context, sample_episode


def create_output_dir(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"semantic_branch_composer_trainable_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def latest_module_oracle_gain(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    oracle_root = output_root / cfg.DATASET.NAME / f"semantic_branch_composer_killswitch_{Path(train_cfg).stem}"
    if not oracle_root.exists():
        return None, None
    results_files = sorted(oracle_root.glob("*/results.json"))
    if not results_files:
        return None, None
    latest = results_files[-1]
    with latest.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    full_results = payload.get("full", {})
    return float(full_results.get("oracle_gain", 0.0)), latest.parent


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
        episodes.append(sample_episode(base_state, pseudo_novel, cfg, seed=int(cfg.SEED) + int(seed_offset) + 1297 * (idx + 1)))
    return episodes


def class_balanced_nll(probs, targets, class_ids):
    idx = torch.arange(targets.size(0), device=targets.device)
    chosen = torch.clamp(probs[idx, targets], min=1e-12)
    nll = -torch.log(chosen)
    losses = []
    for cls_id in class_ids:
        mask = targets == int(cls_id)
        if mask.any():
            losses.append(nll[mask].mean())
    return torch.stack(losses).mean()


def class_subset_accuracy(probs, targets, class_ids):
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for cls_id in class_ids:
        mask |= targets == int(cls_id)
    if not mask.any():
        return torch.tensor(0.0, device=targets.device)
    preds = probs.argmax(dim=1)
    return (preds[mask] == targets[mask]).float().mean()


def component_bank_for_class(state, cls_id, num_base_cls, baseline_sem, max_desc_atoms, base_atom_topk):
    prompt_vec = normalize_vector(state["text_features"][cls_id])
    desc_mean_vec = normalize_vector(state["description_proto"][cls_id])
    image_vec = normalize_vector(state["image_proto"][cls_id])
    desc_group = F.normalize(state["description_features"][state["description_targets"] == cls_id], dim=-1)

    component_names = ["prompt", "desc_mean"]
    component_vectors = [prompt_vec, desc_mean_vec]

    if desc_group.numel() > 0:
        desc_atoms = select_diverse_description_atoms(desc_group, max_desc_atoms)
        for atom_idx, atom in enumerate(desc_atoms):
            component_names.append(f"desc_atom_{atom_idx}")
            component_vectors.append(normalize_vector(atom))

    top_base_anchor = None
    if cls_id >= num_base_cls and num_base_cls > 0:
        sims = torch.matmul(baseline_sem[cls_id], baseline_sem[:num_base_cls].T)
        topk = min(int(base_atom_topk), num_base_cls)
        top_idx = torch.topk(sims, k=topk).indices.tolist()
        for rank, base_idx in enumerate(top_idx):
            base_anchor = normalize_vector(baseline_sem[base_idx])
            if top_base_anchor is None:
                top_base_anchor = base_anchor
            component_names.append(f"base_anchor_{rank}")
            component_vectors.append(base_anchor)

    component_vectors = torch.stack(component_vectors, dim=0)
    context_features = build_context_features(
        prompt_vec=prompt_vec,
        desc_mean_vec=desc_mean_vec,
        image_proto_vec=image_vec,
        desc_group=desc_group if desc_group.numel() > 0 else desc_mean_vec.unsqueeze(0),
        top_base_anchor=top_base_anchor,
    )
    component_features = build_component_features(
        component_vectors=component_vectors,
        prompt_vec=prompt_vec,
        desc_mean_vec=desc_mean_vec,
        image_proto_vec=image_vec,
        top_base_anchor=top_base_anchor,
    )
    return component_names, component_vectors, context_features, component_features


def predict_episode_semantics(
    composer,
    state,
    num_cls,
    num_base_cls,
    pseudo_novel_local,
    lambda_t,
    max_desc_atoms,
    base_atom_topk,
):
    baseline_sem = F.normalize(
        (1.0 - lambda_t) * state["text_features"][:num_cls] + lambda_t * state["description_proto"][:num_cls],
        dim=-1,
    )
    semantic_proto = baseline_sem.clone()

    alphas = []
    entropies = []
    chosen_components = {}

    for cls_id in pseudo_novel_local:
        names, vectors, context_feat, component_feat = component_bank_for_class(
            state=state,
            cls_id=cls_id,
            num_base_cls=num_base_cls,
            baseline_sem=baseline_sem,
            max_desc_atoms=max_desc_atoms,
            base_atom_topk=base_atom_topk,
        )
        outputs = composer(
            baseline_sem=baseline_sem[cls_id].unsqueeze(0),
            component_vectors=vectors.unsqueeze(0),
            context_features=context_feat.unsqueeze(0),
            component_features=component_feat.unsqueeze(0),
        )
        semantic_proto[cls_id] = outputs["semantic"].squeeze(0)
        weights = outputs["weights"].squeeze(0)
        alpha = outputs["alpha"].squeeze(0)
        alphas.append(alpha)
        entropies.append(-(weights * torch.log(torch.clamp(weights, min=1e-12))).sum())
        top_idx = int(torch.argmax(weights).item())
        chosen_components[int(cls_id)] = names[top_idx]

    mean_alpha = torch.stack(alphas).mean() if alphas else torch.tensor(0.0, device=semantic_proto.device)
    mean_entropy = torch.stack(entropies).mean() if entropies else torch.tensor(0.0, device=semantic_proto.device)
    return semantic_proto, {
        "baseline_sem": baseline_sem,
        "mean_alpha": mean_alpha,
        "mean_entropy": mean_entropy,
        "chosen_components": chosen_components,
    }


def precompute_episode_cache(episode, cfg):
    device = torch.device(cfg.DEVICE.DEVICE_NAME)
    state = {k: v.to(device=device, dtype=torch.float32 if v.dtype.is_floating_point else v.dtype) for k, v in episode["state"].items()}
    state["description_targets"] = state["description_targets"].long()

    query_features = episode["query_features"].to(device=device, dtype=torch.float32)
    query_targets = episode["query_targets"].to(device=device, dtype=torch.long)
    num_cls = len(episode["orig_classes"])
    num_base_cls = int(episode["num_base_cls"])

    logits_cov = compute_cov_logits(query_features, state["image_proto"][:num_cls], state["cov_image"])
    prob_cov = F.softmax(logits_cov / 512.0, dim=-1)
    logits_knn = compute_knn_logits(query_features, state["description_features"], state["description_targets"], num_cls)
    prob_knn = F.softmax(logits_knn, dim=-1)

    return {
        "state": state,
        "query_features": query_features,
        "query_targets": query_targets,
        "num_cls": num_cls,
        "num_base_cls": num_base_cls,
        "pseudo_novel_local": list(map(int, episode["pseudo_novel_local"])),
        "prob_cov": prob_cov,
        "prob_knn": prob_knn,
    }


def evaluate_episode(
    composer,
    cache,
    cfg,
    max_desc_atoms,
    base_atom_topk,
):
    beta = float(cfg.DATASET.BETA)
    lambda_t = float(cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
    ensemble_alpha = float(cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.0)

    semantic_proto, composer_info = predict_episode_semantics(
        composer=composer,
        state=cache["state"],
        num_cls=cache["num_cls"],
        num_base_cls=cache["num_base_cls"],
        pseudo_novel_local=cache["pseudo_novel_local"],
        lambda_t=lambda_t,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
    )
    probs, _ = compose_probs_from_semantic(
        cache["query_features"],
        semantic_proto,
        cache["state"]["image_proto"][:cache["num_cls"]],
        cache["num_base_cls"],
        beta,
        ensemble_alpha,
        cache["prob_cov"],
        cache["prob_knn"],
    )

    all_classes = list(range(cache["num_cls"]))
    novel_classes = cache["pseudo_novel_local"]
    loss_all = class_balanced_nll(probs, cache["query_targets"], all_classes)
    loss_novel = class_balanced_nll(probs, cache["query_targets"], novel_classes)
    full_acc = class_subset_accuracy(probs, cache["query_targets"], all_classes)
    novel_acc = class_subset_accuracy(probs, cache["query_targets"], novel_classes)

    return {
        "probs": probs,
        "loss_all": loss_all,
        "loss_novel": loss_novel,
        "full_acc": full_acc,
        "novel_acc": novel_acc,
        "mean_alpha": composer_info["mean_alpha"],
        "mean_entropy": composer_info["mean_entropy"],
        "chosen_components": composer_info["chosen_components"],
    }


def train_composer(
    composer,
    train_caches,
    val_caches,
    cfg,
    max_desc_atoms,
    base_atom_topk,
    lr,
    max_epochs,
    patience,
    novel_loss_weight,
    alpha_reg,
    entropy_reg,
):
    optimizer = torch.optim.Adam(composer.parameters(), lr=lr, weight_decay=1e-4)
    best_state = {k: v.detach().cpu().clone() for k, v in composer.state_dict().items()}
    best_val_full = -1.0
    best_val_novel = -1.0
    best_epoch = -1
    stale_epochs = 0
    history = []

    for epoch in range(max_epochs):
        composer.train()
        train_full = []
        train_novel = []
        train_loss_values = []

        for cache in train_caches:
            optimizer.zero_grad()
            metrics = evaluate_episode(
                composer=composer,
                cache=cache,
                cfg=cfg,
                max_desc_atoms=max_desc_atoms,
                base_atom_topk=base_atom_topk,
            )
            loss = (
                metrics["loss_all"]
                + float(novel_loss_weight) * metrics["loss_novel"]
                + float(alpha_reg) * metrics["mean_alpha"]
                + float(entropy_reg) * metrics["mean_entropy"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(composer.parameters(), 5.0)
            optimizer.step()

            train_loss_values.append(float(loss.item()))
            train_full.append(float(metrics["full_acc"].item()))
            train_novel.append(float(metrics["novel_acc"].item()))

        composer.eval()
        with torch.no_grad():
            val_full = []
            val_novel = []
            val_alpha = []
            for cache in val_caches:
                metrics = evaluate_episode(
                    composer=composer,
                    cache=cache,
                    cfg=cfg,
                    max_desc_atoms=max_desc_atoms,
                    base_atom_topk=base_atom_topk,
                )
                val_full.append(float(metrics["full_acc"].item()))
                val_novel.append(float(metrics["novel_acc"].item()))
                val_alpha.append(float(metrics["mean_alpha"].item()))

        epoch_record = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(train_loss_values) if train_loss_values else 0.0),
            "train_full": float(np.mean(train_full) * 100.0 if train_full else 0.0),
            "train_novel": float(np.mean(train_novel) * 100.0 if train_novel else 0.0),
            "val_full": float(np.mean(val_full) * 100.0 if val_full else 0.0),
            "val_novel": float(np.mean(val_novel) * 100.0 if val_novel else 0.0),
            "val_alpha": float(np.mean(val_alpha) if val_alpha else 0.0),
        }
        history.append(epoch_record)

        improved = (
            epoch_record["val_full"] > best_val_full + 1e-6
            or (
                abs(epoch_record["val_full"] - best_val_full) <= 1e-6
                and epoch_record["val_novel"] > best_val_novel + 1e-6
            )
        )
        if improved:
            best_val_full = epoch_record["val_full"]
            best_val_novel = epoch_record["val_novel"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in composer.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    composer.load_state_dict(best_state)
    return {
        "best_epoch": int(best_epoch),
        "best_val_full": round(float(best_val_full), 3),
        "best_val_novel": round(float(best_val_novel), 3),
        "history": history,
    }


def evaluate_final_session(
    composer,
    merged_state,
    query_features,
    query_targets,
    cfg,
    max_desc_atoms,
    base_atom_topk,
):
    device = torch.device(cfg.DEVICE.DEVICE_NAME)
    num_cls = max(int(query_targets.max().item()) + 1, int(merged_state["image_proto"].shape[0]))
    num_cls = min(num_cls, merged_state["image_proto"].shape[0])
    num_base_cls = int(cfg.DATASET.NUM_INIT_CLS)
    beta = float(cfg.DATASET.BETA)
    lambda_t = float(cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
    ensemble_alpha = float(cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.0)

    state = {
        "image_proto": F.normalize(merged_state["image_proto"][:num_cls].to(device=device, dtype=torch.float32), dim=-1),
        "text_features": F.normalize(merged_state["text_features"][:num_cls].to(device=device, dtype=torch.float32), dim=-1),
        "description_proto": F.normalize(merged_state["description_proto"][:num_cls].to(device=device, dtype=torch.float32), dim=-1),
        "description_features": F.normalize(merged_state["description_features"].to(device=device, dtype=torch.float32), dim=-1),
        "description_targets": merged_state["description_targets"].to(device=device, dtype=torch.long),
        "cov_image": merged_state["cov_image"].to(device=device, dtype=torch.float32),
    }
    query_features = query_features.to(device=device, dtype=torch.float32)
    query_targets = query_targets.to(device=device, dtype=torch.long)

    baseline_sem = F.normalize((1.0 - lambda_t) * state["text_features"] + lambda_t * state["description_proto"], dim=-1)
    prob_cov = F.softmax(compute_cov_logits(query_features, state["image_proto"], state["cov_image"]) / 512.0, dim=-1)
    prob_knn = F.softmax(compute_knn_logits(query_features, state["description_features"], state["description_targets"], num_cls), dim=-1)
    baseline_probs, baseline_logits = compose_probs_from_semantic(
        query_features,
        baseline_sem,
        state["image_proto"],
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )

    novel_class_ids = list(range(num_base_cls, num_cls))
    semantic_proto, composer_info = predict_episode_semantics(
        composer=composer,
        state=state,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        pseudo_novel_local=novel_class_ids,
        lambda_t=lambda_t,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
    )
    full_probs, full_logits = compose_probs_from_semantic(
        query_features,
        semantic_proto,
        state["image_proto"],
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )

    fused_only_probs, _ = compose_probs_from_semantic(
        query_features,
        semantic_proto,
        state["image_proto"],
        num_base_cls,
        beta,
        1.0,
        prob_cov,
        prob_knn,
    )
    baseline_fused_probs, _ = compose_probs_from_semantic(
        query_features,
        baseline_sem,
        state["image_proto"],
        num_base_cls,
        beta,
        1.0,
        prob_cov,
        prob_knn,
    )

    base_ids = list(range(num_base_cls))
    novel_ids = novel_class_ids
    component_counts = Counter(composer_info["chosen_components"].values())

    def pct(acc_tensor):
        return round(float(acc_tensor.item() * 100.0), 3)

    return {
        "baseline_full_acc": pct(class_subset_accuracy(baseline_probs, query_targets, list(range(num_cls)))),
        "baseline_base_acc": pct(class_subset_accuracy(baseline_probs, query_targets, base_ids)),
        "baseline_novel_acc": pct(class_subset_accuracy(baseline_probs, query_targets, novel_ids)),
        "composer_full_acc": pct(class_subset_accuracy(full_probs, query_targets, list(range(num_cls)))),
        "composer_base_acc": pct(class_subset_accuracy(full_probs, query_targets, base_ids)),
        "composer_novel_acc": pct(class_subset_accuracy(full_probs, query_targets, novel_ids)),
        "composer_full_gain": round(pct(class_subset_accuracy(full_probs, query_targets, list(range(num_cls)))) - pct(class_subset_accuracy(baseline_probs, query_targets, list(range(num_cls)))), 3),
        "composer_base_gain": round(pct(class_subset_accuracy(full_probs, query_targets, base_ids)) - pct(class_subset_accuracy(baseline_probs, query_targets, base_ids)), 3),
        "composer_novel_gain": round(pct(class_subset_accuracy(full_probs, query_targets, novel_ids)) - pct(class_subset_accuracy(baseline_probs, query_targets, novel_ids)), 3),
        "baseline_fused_novel_acc": pct(class_subset_accuracy(baseline_fused_probs, query_targets, novel_ids)),
        "composer_fused_novel_acc": pct(class_subset_accuracy(fused_only_probs, query_targets, novel_ids)),
        "composer_fused_novel_gain": round(pct(class_subset_accuracy(fused_only_probs, query_targets, novel_ids)) - pct(class_subset_accuracy(baseline_fused_probs, query_targets, novel_ids)), 3),
        "mean_alpha_all_novel": round(float(composer_info["mean_alpha"].item()), 4),
        "mean_entropy_all_novel": round(float(composer_info["mean_entropy"].item()), 4),
        "top_component_counts": dict(sorted(component_counts.items())),
        "chosen_components": {str(k): v for k, v in composer_info["chosen_components"].items()},
    }


def run_dataset(
    data_cfg,
    train_cfg,
    episodes_train,
    episodes_val,
    max_desc_atoms,
    base_atom_topk,
    lr,
    max_epochs,
    patience,
    hidden_dim,
    novel_loss_weight,
    alpha_reg,
    entropy_reg,
):
    cfg = setup_cfg(data_cfg, train_cfg)
    out_dir = create_output_dir(cfg, train_cfg)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    start_time = time.time()
    base_context = extract_base_context(cfg)
    train_classes, val_classes, heldout_classes = split_base_classes(base_context["base_state"]["class_index"], seed=cfg.SEED)
    train_episodes = build_episode_collection(base_context["base_state"], train_classes, cfg, episodes_train, seed_offset=1000)
    val_episodes = build_episode_collection(base_context["base_state"], val_classes, cfg, episodes_val, seed_offset=2000)

    train_caches = [precompute_episode_cache(ep, cfg) for ep in train_episodes]
    val_caches = [precompute_episode_cache(ep, cfg) for ep in val_episodes]

    device = torch.device(cfg.DEVICE.DEVICE_NAME)
    composer = NovelSemanticComposer(context_dim=7, component_feat_dim=4, hidden_dim=hidden_dim).to(device)
    with torch.no_grad():
        composer.alpha_head.bias.fill_(-2.0)

    train_summary = train_composer(
        composer=composer,
        train_caches=train_caches,
        val_caches=val_caches,
        cfg=cfg,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
        lr=lr,
        max_epochs=max_epochs,
        patience=patience,
        novel_loss_weight=novel_loss_weight,
        alpha_reg=alpha_reg,
        entropy_reg=entropy_reg,
    )

    data_manager, model, merged_state, final_task_id = extract_final_state(cfg)
    query_features, query_targets = collect_final_queries(cfg, data_manager, model, final_task_id)
    final_results = evaluate_final_session(
        composer=composer,
        merged_state=merged_state,
        query_features=query_features,
        query_targets=query_targets,
        cfg=cfg,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
    )

    oracle_gain, oracle_dir = latest_module_oracle_gain(cfg, train_cfg)
    recovery_rate = None
    if oracle_gain is not None and abs(oracle_gain) > 1e-9:
        recovery_rate = round(float(final_results["composer_full_gain"] / oracle_gain), 4)

    runtime_sec = round(time.time() - start_time, 3)
    payload = {
        "dataset": cfg.DATASET.NAME,
        "train_cfg": train_cfg,
        "runtime_sec": runtime_sec,
        "episodes_train": int(episodes_train),
        "episodes_val": int(episodes_val),
        "base_split_sizes": {
            "train": len(train_classes),
            "val": len(val_classes),
            "heldout_unused": len(heldout_classes),
        },
        "max_desc_atoms": int(max_desc_atoms),
        "base_atom_topk": int(base_atom_topk),
        "lr": float(lr),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "hidden_dim": int(hidden_dim),
        "novel_loss_weight": float(novel_loss_weight),
        "alpha_reg": float(alpha_reg),
        "entropy_reg": float(entropy_reg),
        "oracle_gain": oracle_gain,
        "oracle_results_dir": str(oracle_dir) if oracle_dir is not None else None,
        "recovery_rate": recovery_rate,
        "best_epoch": train_summary["best_epoch"],
        "best_val_full": train_summary["best_val_full"],
        "best_val_novel": train_summary["best_val_novel"],
        "history": train_summary["history"],
        **final_results,
    }
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [
        f"dataset: {cfg.DATASET.NAME}",
        f"train_cfg: {train_cfg}",
        f"runtime_sec: {runtime_sec}",
        f"best_epoch: {train_summary['best_epoch']}",
        f"best_val_full: {train_summary['best_val_full']}",
        f"best_val_novel: {train_summary['best_val_novel']}",
        f"baseline_full_acc: {final_results['baseline_full_acc']}",
        f"composer_full_acc: {final_results['composer_full_acc']}",
        f"composer_full_gain: {final_results['composer_full_gain']}",
        f"baseline_base_acc: {final_results['baseline_base_acc']}",
        f"composer_base_acc: {final_results['composer_base_acc']}",
        f"composer_base_gain: {final_results['composer_base_gain']}",
        f"baseline_novel_acc: {final_results['baseline_novel_acc']}",
        f"composer_novel_acc: {final_results['composer_novel_acc']}",
        f"composer_novel_gain: {final_results['composer_novel_gain']}",
        f"baseline_fused_novel_acc: {final_results['baseline_fused_novel_acc']}",
        f"composer_fused_novel_acc: {final_results['composer_fused_novel_acc']}",
        f"composer_fused_novel_gain: {final_results['composer_fused_novel_gain']}",
        f"oracle_gain: {oracle_gain}",
        f"recovery_rate: {recovery_rate}",
        f"mean_alpha_all_novel: {final_results['mean_alpha_all_novel']}",
        f"mean_entropy_all_novel: {final_results['mean_entropy_all_novel']}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "dataset": cfg.DATASET.NAME,
        "baseline_full_acc": final_results["baseline_full_acc"],
        "composer_full_acc": final_results["composer_full_acc"],
        "composer_full_gain": final_results["composer_full_gain"],
        "baseline_novel_acc": final_results["baseline_novel_acc"],
        "composer_novel_acc": final_results["composer_novel_acc"],
        "composer_novel_gain": final_results["composer_novel_gain"],
        "oracle_gain": oracle_gain,
        "recovery_rate": recovery_rate,
        "best_epoch": train_summary["best_epoch"],
        "best_val_full": train_summary["best_val_full"],
        "best_val_novel": train_summary["best_val_novel"],
    }, indent=2))
    print(f"Saved trainable semantic branch composer results to: {out_dir}")
    return out_dir, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--episodes_train", type=int, default=60)
    parser.add_argument("--episodes_val", type=int, default=15)
    parser.add_argument("--max_desc_atoms", type=int, default=2)
    parser.add_argument("--base_atom_topk", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--novel_loss_weight", type=float, default=1.5)
    parser.add_argument("--alpha_reg", type=float, default=0.02)
    parser.add_argument("--entropy_reg", type=float, default=0.002)
    args = parser.parse_args()

    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        episodes_train=args.episodes_train,
        episodes_val=args.episodes_val,
        max_desc_atoms=args.max_desc_atoms,
        base_atom_topk=args.base_atom_topk,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        novel_loss_weight=args.novel_loss_weight,
        alpha_reg=args.alpha_reg,
        entropy_reg=args.entropy_reg,
    )


if __name__ == "__main__":
    main()
