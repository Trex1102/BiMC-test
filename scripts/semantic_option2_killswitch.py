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


def create_output_dir(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"semantic_killswitch_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def extract_final_state(cfg):
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)

    data_manager = DatasetManager(cfg)
    model = build_model(cfg, data_manager)
    model.eval()

    states = []
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
    merged = merge_states(states)
    final_task_id = data_manager.num_tasks - 1
    return data_manager, model, merged, final_task_id


@torch.no_grad()
def collect_final_queries(cfg, data_manager, model, final_task_id):
    model_ref = model.module if hasattr(model, "module") else model
    loader = data_manager.get_dataloader(final_task_id, source="test", mode="test")
    all_features = []
    all_targets = []
    for batch in loader:
        images = batch["image"].to(cfg.DEVICE.DEVICE_NAME)
        targets = batch["label"].to(cfg.DEVICE.DEVICE_NAME)
        image_features = model_ref.extract_img_feature(images)
        image_features = F.normalize(image_features, dim=-1)
        all_features.append(image_features)
        all_targets.append(targets)
    return torch.cat(all_features, dim=0), torch.cat(all_targets, dim=0)


def compute_cov_logits(query_features, image_proto, cov_image):
    inv_cov = torch.pinverse(cov_image.to(dtype=torch.float32)).to(dtype=image_proto.dtype)
    logits = []
    for cls_id in range(image_proto.shape[0]):
        dist = query_features - image_proto[cls_id]
        left = torch.matmul(dist, inv_cov)
        maha = torch.sum(left * dist, dim=1)
        logits.append(-maha)
    return torch.stack(logits, dim=1)


def compute_knn_logits(query_features, support_features, support_labels, num_cls):
    similarity = torch.matmul(query_features, support_features.T)
    max_scores = torch.full((query_features.size(0), num_cls), float("-inf"), device=query_features.device, dtype=query_features.dtype)
    expanded_labels = support_labels.unsqueeze(0).expand(query_features.size(0), -1)
    for label in range(num_cls):
        mask = expanded_labels == label
        masked_scores = similarity.masked_fill(~mask, float("-inf"))
        max_scores[:, label] = masked_scores.max(dim=1).values
    return max_scores


def normalize_vector(vec):
    return F.normalize(vec.unsqueeze(0), dim=-1).squeeze(0)


def average_vectors(vectors):
    stacked = torch.stack(vectors, dim=0)
    return normalize_vector(stacked.mean(dim=0))


def to_builtin(value):
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def select_diverse_description_atoms(desc_group, max_atoms):
    desc_group = F.normalize(desc_group, dim=-1)
    if desc_group.shape[0] <= max_atoms:
        return [desc_group[idx] for idx in range(desc_group.shape[0])]

    mean_vec = normalize_vector(desc_group.mean(dim=0))
    sims_to_mean = torch.matmul(desc_group, mean_vec)
    first_idx = torch.argmin(sims_to_mean).item()
    selected = [first_idx]
    while len(selected) < max_atoms:
        remaining = [idx for idx in range(desc_group.shape[0]) if idx not in selected]
        selected_feats = desc_group[selected]
        rem_feats = desc_group[remaining]
        pair_sims = torch.matmul(rem_feats, selected_feats.T)
        min_dissim = (1.0 - pair_sims).min(dim=1).values
        next_idx = remaining[torch.argmax(min_dissim).item()]
        selected.append(next_idx)
    return [desc_group[idx] for idx in selected]


def build_semantic_banks(merged_state, num_cls, num_base_cls, lambda_t, max_desc_atoms=3, base_atom_topk=2):
    text_features = F.normalize(merged_state["text_features"][:num_cls], dim=-1)
    description_proto = F.normalize(merged_state["description_proto"][:num_cls], dim=-1)
    description_features = F.normalize(merged_state["description_features"], dim=-1)
    description_targets = merged_state["description_targets"]

    baseline_sem = F.normalize((1 - lambda_t) * text_features + lambda_t * description_proto, dim=-1)
    banks = []
    for cls_id in range(num_cls):
        desc_group = description_features[description_targets == cls_id]
        desc_atoms = select_diverse_description_atoms(desc_group, max_desc_atoms) if desc_group.numel() > 0 else []

        atoms = [
            {"name": "prompt", "vector": text_features[cls_id], "uses_base": False, "uses_desc_atom": False},
            {"name": "desc_mean", "vector": description_proto[cls_id], "uses_base": False, "uses_desc_atom": False},
        ]
        for atom_idx, atom in enumerate(desc_atoms):
            atoms.append({
                "name": f"desc_atom_{atom_idx}",
                "vector": atom,
                "uses_base": False,
                "uses_desc_atom": True,
            })

        if cls_id >= num_base_cls and num_base_cls > 0:
            base_scores = torch.matmul(baseline_sem[cls_id], baseline_sem[:num_base_cls].T)
            topk = min(base_atom_topk, num_base_cls)
            top_idx = torch.topk(base_scores, k=topk).indices.tolist()
            for rank, base_idx in enumerate(top_idx):
                atoms.append({
                    "name": f"base_atom_{rank}",
                    "vector": baseline_sem[base_idx],
                    "uses_base": True,
                    "uses_desc_atom": False,
                })

        candidates = [{
            "name": "baseline_sem",
            "vector": baseline_sem[cls_id],
            "uses_base": False,
            "uses_desc_atom": False,
            "is_baseline": True,
        }]
        candidates.append({
            "name": "prompt_only",
            "vector": text_features[cls_id],
            "uses_base": False,
            "uses_desc_atom": False,
            "is_baseline": False,
        })
        candidates.append({
            "name": "desc_mean_only",
            "vector": description_proto[cls_id],
            "uses_base": False,
            "uses_desc_atom": False,
            "is_baseline": False,
        })

        for atom in atoms[2:]:
            candidates.append({
                "name": atom["name"],
                "vector": atom["vector"],
                "uses_base": atom["uses_base"],
                "uses_desc_atom": atom["uses_desc_atom"],
                "is_baseline": False,
            })

        for atom in atoms[2:]:
            candidates.append({
                "name": f"prompt_plus_{atom['name']}",
                "vector": average_vectors([text_features[cls_id], atom["vector"]]),
                "uses_base": atom["uses_base"],
                "uses_desc_atom": atom["uses_desc_atom"],
                "is_baseline": False,
            })
            candidates.append({
                "name": f"desc_plus_{atom['name']}",
                "vector": average_vectors([description_proto[cls_id], atom["vector"]]),
                "uses_base": atom["uses_base"],
                "uses_desc_atom": atom["uses_desc_atom"],
                "is_baseline": False,
            })

        desc_atom_vectors = [atom["vector"] for atom in atoms if atom["uses_desc_atom"]]
        if desc_atom_vectors:
            best_desc_combo = average_vectors([text_features[cls_id], description_proto[cls_id], desc_atom_vectors[0]])
            candidates.append({
                "name": "prompt_desc_top_descatom",
                "vector": best_desc_combo,
                "uses_base": False,
                "uses_desc_atom": True,
                "is_baseline": False,
            })

        base_atoms = [atom for atom in atoms if atom["uses_base"]]
        for atom in base_atoms:
            candidates.append({
                "name": f"prompt_desc_{atom['name']}",
                "vector": average_vectors([text_features[cls_id], description_proto[cls_id], atom["vector"]]),
                "uses_base": True,
                "uses_desc_atom": False,
                "is_baseline": False,
            })

        dedup = []
        seen = set()
        for candidate in candidates:
            key = candidate["name"]
            if key in seen:
                continue
            seen.add(key)
            dedup.append(candidate)
        banks.append(dedup)

    return baseline_sem, banks


def compose_probs_from_semantic(
    query_features,
    semantic_proto,
    image_proto,
    num_base_cls,
    beta,
    ensemble_alpha,
    prob_cov,
    prob_knn,
):
    fused_proto = F.normalize(beta * semantic_proto + (1.0 - beta) * image_proto, dim=-1)
    logits_proto = torch.matmul(query_features, fused_proto.T)
    prob_fused = F.softmax(logits_proto, dim=-1)
    if ensemble_alpha >= 1.0:
        return prob_fused, logits_proto
    base_probs = ensemble_alpha * prob_fused[:, :num_base_cls] + (1.0 - ensemble_alpha) * prob_cov[:, :num_base_cls]
    inc_probs = ensemble_alpha * prob_fused[:, num_base_cls:] + (1.0 - ensemble_alpha) * prob_knn[:, num_base_cls:]
    return torch.cat([base_probs, inc_probs], dim=1), logits_proto


def accuracy_from_probs(probs, targets):
    return float((torch.argmax(probs, dim=1) == targets).float().mean().item() * 100.0)


def run_semantic_oracle(
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
    max_passes,
):
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
            "uses_base": False,
            "uses_desc_atom": False,
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
                        "uses_base": bool(candidate["uses_base"]),
                        "uses_desc_atom": bool(candidate["uses_desc_atom"]),
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

    changed_classes = [cls_id for cls_id in novel_classes if not chosen_meta[cls_id]["is_baseline"]]
    used_base_classes = [cls_id for cls_id in novel_classes if chosen_meta[cls_id]["uses_base"]]
    used_desc_atom_classes = [cls_id for cls_id in novel_classes if chosen_meta[cls_id]["uses_desc_atom"]]

    image_proto_cpu = image_proto.detach().cpu()
    baseline_sem_cpu = baseline_sem.detach().cpu()
    current_sem_cpu = current_sem.detach().cpu()
    novel_slice = slice(num_base_cls, num_cls)
    baseline_alignment = 1.0 - torch.sum(image_proto_cpu[novel_slice] * baseline_sem_cpu[novel_slice], dim=1)
    oracle_alignment = 1.0 - torch.sum(image_proto_cpu[novel_slice] * current_sem_cpu[novel_slice], dim=1)

    return {
        "baseline_acc": round(float(baseline_acc), 3),
        "oracle_acc": round(float(current_acc), 3),
        "oracle_gain": round(float(current_acc - baseline_acc), 3),
        "changed_class_rate": round(float(len(changed_classes) / max(len(novel_classes), 1)), 4),
        "base_atom_usage_rate": round(float(len(used_base_classes) / max(len(novel_classes), 1)), 4),
        "desc_atom_usage_rate": round(float(len(used_desc_atom_classes) / max(len(novel_classes), 1)), 4),
        "baseline_alignment_gap": round(float(baseline_alignment.mean().item()), 6) if len(novel_classes) > 0 else 0.0,
        "oracle_alignment_gap": round(float(oracle_alignment.mean().item()), 6) if len(novel_classes) > 0 else 0.0,
        "chosen_candidates": {
            str(cls_id): chosen_meta[cls_id]["name"] for cls_id in novel_classes
        },
    }


def option2_continue_flag(dataset_name, oracle_gain):
    dataset_name = dataset_name.lower()
    if dataset_name == "cub200":
        return oracle_gain >= 1.5
    return oracle_gain >= 0.8


def run_dataset(data_cfg, train_cfg, max_desc_atoms, base_atom_topk, max_passes):
    cfg = setup_cfg(data_cfg, train_cfg)
    out_dir = create_output_dir(cfg, train_cfg)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    start_time = time.time()
    data_manager, model, merged_state, final_task_id = extract_final_state(cfg)
    model_ref = model.module if hasattr(model, "module") else model
    query_features, query_targets = collect_final_queries(cfg, data_manager, model, final_task_id)

    num_cls = max(data_manager.class_index_in_task[final_task_id]) + 1
    num_base_cls = len(data_manager.class_index_in_task[0])
    beta = float(cfg.DATASET.BETA)
    lambda_t = float(cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
    ensemble_alpha = float(cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.0)

    oracle = run_semantic_oracle(
        merged_state=merged_state,
        query_features=query_features.to(cfg.DEVICE.DEVICE_NAME),
        query_targets=query_targets.to(cfg.DEVICE.DEVICE_NAME),
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        beta=beta,
        lambda_t=lambda_t,
        ensemble_alpha=ensemble_alpha,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
        max_passes=max_passes,
    )

    runtime_sec = round(time.time() - start_time, 3)
    recommendation = "continue_option2" if option2_continue_flag(cfg.DATASET.NAME, oracle["oracle_gain"]) else "kill_option2"
    payload = {
        "dataset": cfg.DATASET.NAME,
        "train_cfg": train_cfg,
        "runtime_sec": runtime_sec,
        "num_seen_classes": num_cls,
        "num_base_classes": num_base_cls,
        "baseline_beta": beta,
        "baseline_lambda_t": lambda_t,
        "ensemble_alpha": ensemble_alpha,
        "max_desc_atoms": max_desc_atoms,
        "base_atom_topk": base_atom_topk,
        "max_passes": max_passes,
        "recommendation": recommendation,
        **oracle,
    }
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [
        f"dataset: {cfg.DATASET.NAME}",
        f"train_cfg: {train_cfg}",
        f"runtime_sec: {runtime_sec}",
        f"baseline_acc: {oracle['baseline_acc']}",
        f"oracle_acc: {oracle['oracle_acc']}",
        f"oracle_gain: {oracle['oracle_gain']}",
        f"recommendation: {recommendation}",
        f"changed_class_rate: {oracle['changed_class_rate']}",
        f"base_atom_usage_rate: {oracle['base_atom_usage_rate']}",
        f"desc_atom_usage_rate: {oracle['desc_atom_usage_rate']}",
        f"baseline_alignment_gap: {oracle['baseline_alignment_gap']}",
        f"oracle_alignment_gap: {oracle['oracle_alignment_gap']}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "dataset": cfg.DATASET.NAME,
        "baseline_acc": oracle["baseline_acc"],
        "oracle_acc": oracle["oracle_acc"],
        "oracle_gain": oracle["oracle_gain"],
        "recommendation": recommendation,
        "changed_class_rate": oracle["changed_class_rate"],
        "base_atom_usage_rate": oracle["base_atom_usage_rate"],
        "desc_atom_usage_rate": oracle["desc_atom_usage_rate"],
    }, indent=2))
    print(f"Saved semantic oracle diagnostics to: {out_dir}")
    return out_dir, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--max_desc_atoms", type=int, default=3)
    parser.add_argument("--base_atom_topk", type=int, default=2)
    parser.add_argument("--max_passes", type=int, default=2)
    args = parser.parse_args()
    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        max_desc_atoms=args.max_desc_atoms,
        base_atom_topk=args.base_atom_topk,
        max_passes=args.max_passes,
    )


if __name__ == "__main__":
    main()
