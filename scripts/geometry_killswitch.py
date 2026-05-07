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

from datasets.data_manager import DatasetManager
from main import setup_cfg
from models.bimc import BiMC
from models.bimc_adaptive import BiMCAdaptive
from utils.util import set_gpu, set_seed


CAT_KEYS = [
    "description_proto",
    "description_features",
    "description_targets",
    "text_features",
    "text_targets",
    "image_proto",
    "images_features",
    "images_targets",
    "prompt_reliability",
    "description_reliability",
    "text_reliability",
    "visual_reliability",
    "beta_per_class",
    "lambda_t_per_class",
    "lambda_i_per_class",
    "anchor_reliability",
    "image_counts",
    "prompt_counts",
    "description_counts",
]


def build_model(cfg, data_manager):
    if cfg.METHOD == "bimc_adaptive":
        return BiMCAdaptive(cfg, data_manager.template, cfg.DEVICE.DEVICE_NAME)
    return BiMC(cfg, data_manager.template, cfg.DEVICE.DEVICE_NAME)


def merge_states(states):
    merged = {}
    for key in CAT_KEYS:
        if key in states[0]:
            merged[key] = torch.cat([state[key] for state in states], dim=0)

    total_weight = sum(len(state["class_index"]) for state in states)
    merged["cov_image"] = sum(
        state["cov_image"] * len(state["class_index"]) for state in states
    ) / total_weight
    return merged


def gini_coefficient(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    values = np.clip(values, a_min=0.0, a_max=None)
    total = values.sum()
    if total <= 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = sorted_vals.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_vals) / (n * total)) - (n + 1.0) / n)


def skewness(values, eps=1e-12):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    mean = values.mean()
    std = values.std()
    if std <= eps:
        return 0.0
    centered = (values - mean) / std
    return float(np.mean(centered ** 3))


def resolve_session_beta(model_ref, cfg, merged_state, num_cls, num_base_cls, current_task_class_index):
    beta = cfg.DATASET.BETA
    if hasattr(model_ref, "resolve_beta_prior"):
        beta = model_ref.resolve_beta_prior(beta)

    session_beta = None
    if hasattr(model_ref, "compute_session_joint_beta"):
        session_beta = model_ref.compute_session_joint_beta(
            num_cls,
            beta,
            merged_state.get("beta_per_class"),
            merged_state.get("text_reliability"),
            merged_state.get("visual_reliability"),
            num_base_cls=num_base_cls,
            image_proto=merged_state["image_proto"],
            cov_image=merged_state["cov_image"],
            description_proto=merged_state["description_proto"],
            description_features=merged_state["description_features"],
            description_targets=merged_state["description_targets"],
            text_features=merged_state["text_features"],
            images_features=merged_state.get("images_features"),
            images_targets=merged_state.get("images_targets"),
            lambda_t_per_class=merged_state.get("lambda_t_per_class"),
            current_task_class_index=current_task_class_index,
        )
        if session_beta is not None:
            session_beta = float(session_beta)
    return float(beta), session_beta


def compose_fused_prototypes(model_ref, cfg, merged_state, num_cls, beta, session_beta=None):
    image_proto = merged_state["image_proto"][:num_cls]
    text_features = merged_state["text_features"][:num_cls]
    description_proto = merged_state["description_proto"][:num_cls]

    if "lambda_t_per_class" in merged_state:
        lambda_t = merged_state["lambda_t_per_class"][:num_cls].to(image_proto.device)
    else:
        lambda_value = cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
        lambda_t = image_proto.new_full((num_cls,), float(lambda_value))

    beta_mode = ""
    if hasattr(model_ref, "adapt_cfg"):
        beta_mode = str(model_ref.adapt_cfg.BETA_MODE).lower()

    if session_beta is not None:
        beta_tensor = image_proto.new_full((num_cls,), float(session_beta))
    elif beta_mode not in {"session_joint", "session_risk"} and "beta_per_class" in merged_state:
        beta_tensor = merged_state["beta_per_class"][:num_cls].to(image_proto.device)
    else:
        beta_tensor = image_proto.new_full((num_cls,), float(beta))

    fused_text = (1 - lambda_t.unsqueeze(1)) * text_features + lambda_t.unsqueeze(1) * description_proto
    fused_text = F.normalize(fused_text, dim=-1)
    fused_proto = beta_tensor.unsqueeze(1) * fused_text + (1 - beta_tensor.unsqueeze(1)) * image_proto
    return F.normalize(fused_proto, dim=-1)


@torch.no_grad()
def collect_session_outputs(model, model_ref, cfg, data_manager, task_id, merged_state):
    num_base_cls = len(data_manager.class_index_in_task[0])
    num_cls = max(data_manager.class_index_in_task[task_id]) + 1
    beta, session_beta = resolve_session_beta(
        model_ref,
        cfg,
        merged_state,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        current_task_class_index=data_manager.class_index_in_task[task_id],
    )

    extra_kwargs = {}
    if hasattr(model_ref, "adapt_cfg"):
        for key in [
            "beta_per_class",
            "lambda_t_per_class",
            "lambda_i_per_class",
            "prompt_reliability",
            "description_reliability",
            "text_reliability",
            "visual_reliability",
            "images_features",
            "images_targets",
        ]:
            if key in merged_state:
                extra_kwargs[key] = merged_state[key]
        extra_kwargs["current_task_class_index"] = data_manager.class_index_in_task[task_id]
        if session_beta is not None:
            extra_kwargs["session_beta"] = session_beta

    test_loader = data_manager.get_dataloader(task_id, source="test", mode="test")
    all_features = []
    all_probs = []
    all_targets = []

    for batch in test_loader:
        images = batch["image"].to(cfg.DEVICE.DEVICE_NAME)
        targets = batch["label"].to(cfg.DEVICE.DEVICE_NAME)
        image_features = model_ref.extract_img_feature(images)
        image_features = F.normalize(image_features, dim=-1)
        probs = model.forward_ours(
            images,
            num_cls,
            num_base_cls,
            merged_state["image_proto"],
            merged_state["cov_image"],
            merged_state["description_proto"],
            merged_state["description_features"],
            merged_state["description_targets"],
            merged_state["text_features"],
            beta=beta,
            **extra_kwargs,
        )
        all_features.append(image_features.float().cpu())
        all_probs.append(probs.float().cpu())
        all_targets.append(targets.cpu())

    fused_proto = compose_fused_prototypes(
        model_ref,
        cfg,
        merged_state,
        num_cls=num_cls,
        beta=beta,
        session_beta=session_beta,
    ).float().cpu()

    return {
        "num_cls": num_cls,
        "num_base_cls": num_base_cls,
        "beta_prior": beta,
        "session_beta": session_beta,
        "query_features": torch.cat(all_features, dim=0),
        "query_probs": torch.cat(all_probs, dim=0),
        "query_targets": torch.cat(all_targets, dim=0),
        "fused_proto": fused_proto,
    }


def compute_hubness_metrics(query_features, fused_proto, knn_k):
    sim = torch.matmul(query_features, fused_proto.T)
    k = max(1, min(int(knn_k), fused_proto.shape[0]))
    topk_idx = torch.topk(sim, k=k, dim=1).indices.reshape(-1)
    counts = torch.bincount(topk_idx, minlength=fused_proto.shape[0]).cpu().numpy()
    return {
        "hubness_skew": skewness(counts),
        "k_occurrence_gini": gini_coefficient(counts),
        "k_occurrence_max_over_mean": float(counts.max() / (counts.mean() + 1e-12)),
    }


def compute_density_metrics(query_features, query_targets, num_base_cls, density_k):
    unique_classes = torch.unique(query_targets).tolist()
    base_density = []
    novel_density = []
    for cls_id in unique_classes:
        cls_id = int(cls_id)
        cls_feat = query_features[query_targets == cls_id]
        if cls_feat.shape[0] <= 1:
            continue
        sim = torch.matmul(cls_feat, cls_feat.T)
        dist = 1.0 - sim
        dist.fill_diagonal_(float("inf"))
        k = max(1, min(int(density_k), cls_feat.shape[0] - 1))
        mean_kdist = torch.topk(dist, k=k, largest=False, dim=1).values.mean().item()
        density = 1.0 / max(mean_kdist, 1e-6)
        if cls_id < num_base_cls:
            base_density.append(density)
        else:
            novel_density.append(density)

    base_mean = float(np.mean(base_density)) if base_density else 0.0
    novel_mean = float(np.mean(novel_density)) if novel_density else 0.0
    ratio = base_mean / max(novel_mean, 1e-12) if novel_density else 1.0
    return {
        "base_local_density": base_mean,
        "novel_local_density": novel_mean,
        "base_novel_density_ratio": ratio,
    }


def compute_crowding_metrics(fused_proto, num_base_cls, crowd_k):
    num_cls = fused_proto.shape[0]
    num_novel = max(0, num_cls - num_base_cls)
    if num_novel == 0:
        return {
            "novel_nearest_base_rate": 0.0,
            "novel_base_neighbor_fraction": 0.0,
            "novel_base_neighbor_excess": 0.0,
            "base_class_prior": float(num_base_cls / max(num_cls, 1)),
        }

    sim = torch.matmul(fused_proto, fused_proto.T)
    sim.fill_diagonal_(float("-inf"))
    k = max(1, min(int(crowd_k), num_cls - 1))
    topk = torch.topk(sim, k=k, dim=1).indices
    novel_topk = topk[num_base_cls:]
    base_mask = (novel_topk < num_base_cls).float()
    nearest_base_rate = float(base_mask[:, 0].mean().item())
    neighbor_fraction = float(base_mask.mean().item())
    base_prior = float(num_base_cls / num_cls)
    return {
        "novel_nearest_base_rate": nearest_base_rate,
        "novel_base_neighbor_fraction": neighbor_fraction,
        "novel_base_neighbor_excess": neighbor_fraction - base_prior,
        "base_class_prior": base_prior,
    }


def compute_margin_metrics(query_probs, query_targets, current_task_class_index, num_base_cls):
    if len(current_task_class_index) == 0:
        return {
            "novel_margin_mean": 0.0,
            "novel_margin_median": 0.0,
            "novel_margin_violation_rate": 0.0,
            "novel_margin_severe_rate": 0.0,
        }

    current_task_tensor = torch.tensor(current_task_class_index, dtype=torch.long)
    novel_mask = (query_targets.unsqueeze(1) == current_task_tensor.unsqueeze(0)).any(dim=1)
    if not novel_mask.any():
        return {
            "novel_margin_mean": 0.0,
            "novel_margin_median": 0.0,
            "novel_margin_violation_rate": 0.0,
            "novel_margin_severe_rate": 0.0,
        }

    novel_probs = query_probs[novel_mask]
    novel_targets = query_targets[novel_mask]
    sample_idx = torch.arange(novel_targets.shape[0], dtype=torch.long)
    target_scores = novel_probs[sample_idx, novel_targets]
    base_scores = novel_probs[:, :num_base_cls].max(dim=1).values
    margins = target_scores - base_scores
    return {
        "novel_margin_mean": float(margins.mean().item()),
        "novel_margin_median": float(margins.median().item()),
        "novel_margin_violation_rate": float((margins <= 0).float().mean().item()),
        "novel_margin_severe_rate": float((margins < -0.05).float().mean().item()),
    }


def compute_session_metrics(data_manager, task_id, session_outputs, knn_k, density_k, crowd_k):
    metrics = {
        "task_id": int(task_id),
        "num_seen_classes": int(session_outputs["num_cls"]),
        "num_base_classes": int(session_outputs["num_base_cls"]),
        "beta_prior": round(float(session_outputs["beta_prior"]), 4),
    }
    if session_outputs["session_beta"] is not None:
        metrics["session_beta"] = round(float(session_outputs["session_beta"]), 4)

    metrics.update(
        compute_hubness_metrics(
            session_outputs["query_features"],
            session_outputs["fused_proto"],
            knn_k=knn_k,
        )
    )
    metrics.update(
        compute_density_metrics(
            session_outputs["query_features"],
            session_outputs["query_targets"],
            num_base_cls=session_outputs["num_base_cls"],
            density_k=density_k,
        )
    )
    metrics.update(
        compute_crowding_metrics(
            session_outputs["fused_proto"],
            num_base_cls=session_outputs["num_base_cls"],
            crowd_k=crowd_k,
        )
    )
    metrics.update(
        compute_margin_metrics(
            session_outputs["query_probs"],
            session_outputs["query_targets"],
            current_task_class_index=data_manager.class_index_in_task[task_id].tolist(),
            num_base_cls=session_outputs["num_base_cls"],
        )
    )
    return metrics


def apply_kill_switch(session_metrics):
    base_metrics = session_metrics[0]
    final_metrics = session_metrics[-1]

    conditions = {
        "hubness_growth": (
            final_metrics["hubness_skew"] >= 1.0
            and final_metrics["hubness_skew"] >= base_metrics["hubness_skew"] + 0.25
        ),
        "occurrence_skew_growth": (
            final_metrics["k_occurrence_gini"] >= 0.30
            and final_metrics["k_occurrence_gini"] >= base_metrics["k_occurrence_gini"] + 0.03
        ),
        "density_imbalance": final_metrics["base_novel_density_ratio"] >= 1.20,
        "prototype_crowding": (
            final_metrics["novel_base_neighbor_excess"] >= 0.10
            or final_metrics["novel_nearest_base_rate"] >= final_metrics["base_class_prior"] + 0.15
        ),
        "margin_collapse": (
            final_metrics["novel_margin_violation_rate"] >= 0.35
            or final_metrics["novel_margin_mean"] <= 0.0
        ),
    }

    strong_signal_count = int(sum(bool(flag) for flag in conditions.values()))
    continue_option1 = strong_signal_count >= 3 and (
        conditions["density_imbalance"] or conditions["prototype_crowding"] or conditions["margin_collapse"]
    )

    return {
        "final_metrics": final_metrics,
        "base_metrics": base_metrics,
        "conditions": conditions,
        "strong_signal_count": strong_signal_count,
        "recommendation": "continue_option1" if continue_option1 else "pivot_option2",
    }


def create_output_dir(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"geometry_killswitch_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@torch.no_grad()
def run_diagnostics(data_cfg, train_cfg, knn_k, density_k, crowd_k):
    cfg = setup_cfg(data_cfg, train_cfg)
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)

    data_manager = DatasetManager(cfg)
    model = build_model(cfg, data_manager)
    model.eval()
    model_ref = model.module if hasattr(model, "module") else model

    out_dir = create_output_dir(cfg, train_cfg)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    states = []
    per_session = []
    start_time = time.time()

    for task_id in range(data_manager.num_tasks):
        class_names = np.array(data_manager.class_names)[data_manager.class_index_in_task[task_id]]
        train_loader = data_manager.get_dataloader(task_id, source="train", mode="test", accumulate_past=False)
        state = model.build_task_statistics(
            class_names,
            train_loader,
            class_index=data_manager.class_index_in_task[task_id],
            calibrate_novel_vision_proto=cfg.TRAINER.BiMC.VISION_CALIBRATION,
        )
        states.append(state)
        merged_state = merge_states(states)
        outputs = collect_session_outputs(model, model_ref, cfg, data_manager, task_id, merged_state)
        metrics = compute_session_metrics(
            data_manager,
            task_id,
            outputs,
            knn_k=knn_k,
            density_k=density_k,
            crowd_k=crowd_k,
        )
        per_session.append(metrics)
        print(
            f"task={task_id} | hubness={metrics['hubness_skew']:.3f} | "
            f"gini={metrics['k_occurrence_gini']:.3f} | "
            f"density_ratio={metrics['base_novel_density_ratio']:.3f} | "
            f"crowding_excess={metrics['novel_base_neighbor_excess']:.3f} | "
            f"margin_violation={metrics['novel_margin_violation_rate']:.3f}"
        )

    decision = apply_kill_switch(per_session)
    runtime_sec = round(time.time() - start_time, 3)
    payload = {
        "dataset": cfg.DATASET.NAME,
        "train_cfg": train_cfg,
        "runtime_sec": runtime_sec,
        "knn_k": knn_k,
        "density_k": density_k,
        "crowd_k": crowd_k,
        "per_session": per_session,
        "kill_switch": decision,
    }

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [
        f"dataset: {cfg.DATASET.NAME}",
        f"train_cfg: {train_cfg}",
        f"runtime_sec: {runtime_sec}",
        f"recommendation: {decision['recommendation']}",
        f"strong_signal_count: {decision['strong_signal_count']}",
        "conditions:",
    ]
    summary_lines.extend([f"  {k}: {v}" for k, v in decision["conditions"].items()])
    summary_lines.append("final_metrics:")
    summary_lines.extend([f"  {k}: {v}" for k, v in decision["final_metrics"].items()])
    summary_lines.append("base_metrics:")
    summary_lines.extend([f"  {k}: {v}" for k, v in decision["base_metrics"].items()])
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Saved diagnostics to: {out_dir}")
    print(json.dumps({"dataset": cfg.DATASET.NAME, "recommendation": decision["recommendation"], "strong_signal_count": decision["strong_signal_count"]}, indent=2))
    return out_dir, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--density_k", type=int, default=5)
    parser.add_argument("--crowd_k", type=int, default=5)
    args = parser.parse_args()
    run_diagnostics(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        knn_k=args.knn_k,
        density_k=args.density_k,
        crowd_k=args.crowd_k,
    )


if __name__ == "__main__":
    main()
