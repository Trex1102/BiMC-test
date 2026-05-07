import argparse
import numpy as np
import torch

from datasets.data_manager import DatasetManager
from main import setup_cfg
from models.bimc import BiMC
from models.bimc_adaptive import BiMCAdaptive
from utils.util import set_gpu, set_seed


def build_model(cfg, data_manager):
    if cfg.METHOD == "bimc_adaptive":
        return BiMCAdaptive(cfg, data_manager.template, cfg.DEVICE.DEVICE_NAME)
    return BiMC(cfg, data_manager.template, cfg.DEVICE.DEVICE_NAME)


def merge_states(states):
    merged = {}
    cat_keys = [
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
    for key in cat_keys:
        if key in states[0]:
            merged[key] = torch.cat([state[key] for state in states], dim=0)

    total_weight = sum(len(state["class_index"]) for state in states)
    merged["cov_image"] = sum(
        state["cov_image"] * len(state["class_index"]) for state in states
    ) / total_weight
    return merged


def evaluate_task1(data_cfg, train_cfg):
    cfg = setup_cfg(data_cfg, train_cfg)
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)

    data_manager = DatasetManager(cfg)
    model = build_model(cfg, data_manager)
    model.eval()

    states = []
    for task_id in range(2):
        class_names = np.array(data_manager.class_names)[data_manager.class_index_in_task[task_id]]
        loader = data_manager.get_dataloader(task_id, source="train", mode="test", accumulate_past=False)
        state = model.build_task_statistics(
            class_names,
            loader,
            class_index=data_manager.class_index_in_task[task_id],
            calibrate_novel_vision_proto=cfg.TRAINER.BiMC.VISION_CALIBRATION,
        )
        states.append(state)

    merged = merge_states(states)
    num_base = len(data_manager.class_index_in_task[0])
    num_cls = max(data_manager.class_index_in_task[1]) + 1

    test_loader = data_manager.get_dataloader(1, source="test", mode="test")
    preds = []
    targets = []
    session_beta = None

    for batch in test_loader:
        data = batch["image"].to(cfg.DEVICE.DEVICE_NAME)
        label = batch["label"].to(cfg.DEVICE.DEVICE_NAME)
        extra_kwargs = {}
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
            if key in merged:
                extra_kwargs[key] = merged[key]

        beta_prior = cfg.DATASET.BETA
        if hasattr(model, "resolve_beta_prior"):
            beta_prior = model.resolve_beta_prior(beta_prior)

        if hasattr(model, "compute_session_joint_beta"):
            session_beta = model.compute_session_joint_beta(
                num_cls,
                beta_prior,
                beta_per_class=merged.get("beta_per_class"),
                text_reliability=merged.get("text_reliability"),
                visual_reliability=merged.get("visual_reliability"),
                num_base_cls=num_base,
                image_proto=merged["image_proto"],
                cov_image=merged["cov_image"],
                description_proto=merged["description_proto"],
                description_features=merged["description_features"],
                description_targets=merged["description_targets"],
                text_features=merged["text_features"],
                images_features=merged.get("images_features"),
                images_targets=merged.get("images_targets"),
                lambda_t_per_class=merged.get("lambda_t_per_class"),
                current_task_class_index=data_manager.class_index_in_task[1],
            )
            if session_beta is not None:
                extra_kwargs["session_beta"] = session_beta

        extra_kwargs["current_task_class_index"] = data_manager.class_index_in_task[1]

        logits = model.forward_ours(
            data,
            num_cls,
            num_base,
            merged["image_proto"],
            merged["cov_image"],
            merged["description_proto"],
            merged["description_features"],
            merged["description_targets"],
            merged["text_features"],
            beta=beta_prior,
            **extra_kwargs,
        )
        preds.append(torch.argmax(logits, dim=1).cpu())
        targets.append(label.cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    base_classes = set(data_manager.class_index_in_task[0].tolist())
    novel_classes = set(data_manager.class_index_in_task[1].tolist())
    novel_mask = np.isin(targets, list(novel_classes))
    novel_preds = preds[novel_mask]
    novel_targets = targets[novel_mask]

    out = {
        "train_cfg": train_cfg,
        "novel_acc": float((novel_preds == novel_targets).mean() * 100),
        "novel_predicted_as_base_pct": float(np.isin(novel_preds, list(base_classes)).mean() * 100),
        "novel_predicted_as_novel_pct": float(np.isin(novel_preds, list(novel_classes)).mean() * 100),
    }
    if "beta_per_class" in merged:
        out["beta_base_mean"] = float(merged["beta_per_class"][:num_base].mean())
        out["beta_novel_mean"] = float(merged["beta_per_class"][num_base:num_cls].mean())
        if session_beta is not None:
            out["beta_session_joint"] = float(session_beta)
    if "lambda_t_per_class" in merged:
        out["lambda_t_base_mean"] = float(merged["lambda_t_per_class"][:num_base].mean())
        out["lambda_t_novel_mean"] = float(merged["lambda_t_per_class"][num_base:num_cls].mean())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", nargs="+", required=True)
    args = parser.parse_args()

    for train_cfg in args.train_cfg:
        result = evaluate_task1(args.data_cfg, train_cfg)
        print(result)


if __name__ == "__main__":
    main()
