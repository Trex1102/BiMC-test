import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from datasets.data_manager import DatasetManager
from main import setup_cfg
from models.bimc_dino_fusion import BiMCDinoFusion
from utils.evaluator import AccuracyEvaluator
from utils.util import set_gpu, set_seed


VARIANTS = [
    "semantic_branch_only",
    "clip_visual_only",
    "dino_visual_only",
    "clip_dino_visual_only",
    "tric_no_mask",
    "tric",
]


def merge_task_states(states):
    result = {}
    concat_keys = [
        "description_proto",
        "description_features",
        "description_targets",
        "text_features",
        "text_targets",
        "image_proto",
        "images_features",
        "images_targets",
        "dino_image_proto",
        "dino_images_features",
        "dino_images_targets",
    ]
    for key in concat_keys:
        if key in states[0]:
            result[key] = torch.cat([state[key] for state in states], dim=0)

    weights = [len(state["class_index"]) for state in states]
    weight_sum = sum(weights)
    for key in ["cov_image", "dino_cov_image"]:
        if key not in states[0]:
            continue
        cov_sum = torch.zeros_like(states[0][key])
        for weight, state in zip(weights, states):
            cov_sum += state[key] * weight
        result[key] = cov_sum / weight_sum

    return result


def softmax_logits(logits, temp=100.0):
    return F.softmax(logits * temp, dim=-1)


def compute_variant_probs(model, images, state, num_cls, num_base_cls, beta):
    clip_feat = F.normalize(model.extract_img_feature(images), dim=-1).float()
    dino_feat = model.extract_dino_img_feature(images).float()

    image_proto = F.normalize(state["image_proto"][:num_cls].to(clip_feat.device).float(), dim=-1)
    dino_proto = F.normalize(state["dino_image_proto"][:num_cls].to(dino_feat.device).float(), dim=-1)
    text_features = F.normalize(state["text_features"][:num_cls].to(clip_feat.device).float(), dim=-1)
    description_proto = F.normalize(state["description_proto"][:num_cls].to(clip_feat.device).float(), dim=-1)

    lambda_t = model.cfg.TRAINER.BiMC.LAMBDA_T if model.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
    semantic_proto = F.normalize((1.0 - lambda_t) * text_features + lambda_t * description_proto, dim=-1)

    semantic_logits = torch.matmul(clip_feat, semantic_proto.T)
    clip_visual_logits = torch.matmul(clip_feat, image_proto.T)
    dino_visual_logits = torch.matmul(dino_feat, dino_proto.T)
    clip_dino_logits = model.OMEGA * dino_visual_logits + (1.0 - model.OMEGA) * clip_visual_logits
    tric_logits = float(beta) * semantic_logits + (1.0 - float(beta)) * clip_dino_logits
    tric_proto_probs = softmax_logits(tric_logits, model.LOGIT_TEMP)

    dino_cov = state["dino_cov_image"].to(dino_feat.device)
    cov_probs = F.softmax(model._mahalanobis_logits(dino_feat, dino_proto, dino_cov) / model.DINO_COV_SCALE, dim=-1)
    knn_probs = F.softmax(
        model._knn_similarity_scores(
            clip_feat,
            state["description_features"].float(),
            state["description_targets"],
            num_cls,
        ),
        dim=-1,
    )

    eta = float(model.cfg.DATASET.ENSEMBLE_ALPHA)
    tric_probs = torch.cat(
        [
            eta * tric_proto_probs[:, :num_base_cls] + (1.0 - eta) * cov_probs[:, :num_base_cls],
            eta * tric_proto_probs[:, num_base_cls:] + (1.0 - eta) * knn_probs[:, num_base_cls:],
        ],
        dim=1,
    )

    return {
        "semantic_branch_only": softmax_logits(semantic_logits, model.LOGIT_TEMP),
        "clip_visual_only": softmax_logits(clip_visual_logits, model.LOGIT_TEMP),
        "dino_visual_only": softmax_logits(dino_visual_logits, model.LOGIT_TEMP),
        "clip_dino_visual_only": softmax_logits(clip_dino_logits, model.LOGIT_TEMP),
        "tric_no_mask": tric_proto_probs,
        "tric": tric_probs,
    }


def percent_patterns(correct):
    patterns = {}
    total = len(next(iter(correct.values())))
    for s in [True, False]:
        for v in [True, False]:
            for t in [True, False]:
                mask = (
                    (correct["semantic_branch_only"] == s)
                    & (correct["clip_dino_visual_only"] == v)
                    & (correct["tric"] == t)
                )
                key = f"{int(s)}{int(v)}{int(t)}"
                patterns[key] = round(100.0 * float(mask.sum()) / float(total), 4)
    return patterns


def run_audit(data_cfg, train_cfg, seed, output_root, output_path):
    cfg = setup_cfg(data_cfg, train_cfg)
    cfg.defrost()
    cfg.SEED = int(seed)
    cfg.OUTPUT.ROOT = str(output_root)
    cfg.METHOD = "bimc_dino_fusion"
    cfg.TRAINER.BiMC.VISION_CALIBRATION = False
    cfg.TRAINER.BiMC.TEXT_CALIBRATION = True
    cfg.TRAINER.BiMC.LAMBDA_T = 0.5
    cfg.TRAINER.BiMC.GAMMA_BASE = 1.0
    cfg.TRAINER.BiMC.GAMMA_INC = 1.0
    cfg.TRAINER.BiMC.USING_ENSEMBLE = True
    cfg.freeze()

    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)

    started = time.time()
    data_manager = DatasetManager(cfg)
    device = cfg.DEVICE.DEVICE_NAME
    model = BiMCDinoFusion(cfg, data_manager.template, device).to(device).eval()
    evaluator = AccuracyEvaluator(data_manager.class_index_in_task)

    variant_results = {
        variant: {"acc_list": [], "task_acc_list": [], "metrics": []}
        for variant in VARIANTS
    }
    outcome_by_session = {}
    task_states = []

    for task_id in range(data_manager.num_tasks):
        class_names = np.array(data_manager.class_names)[data_manager.class_index_in_task[task_id]]
        train_loader = data_manager.get_dataloader(task_id, source="train", mode="test", accumulate_past=False)
        task_state = model.build_task_statistics(
            class_names,
            train_loader,
            class_index=data_manager.class_index_in_task[task_id],
            calibrate_novel_vision_proto=False,
        )
        task_states.append(task_state)
        state = merge_task_states(task_states)

        num_base_cls = len(data_manager.class_index_in_task[0])
        num_cls = max(data_manager.class_index_in_task[task_id]) + 1
        test_loader = data_manager.get_dataloader(task_id, source="test", mode="test")

        all_targets = []
        probs_by_variant = {variant: [] for variant in VARIANTS}
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                targets = batch["label"].to(device)
                probs = compute_variant_probs(model, images, state, num_cls, num_base_cls, cfg.DATASET.BETA)
                all_targets.append(targets)
                for variant in VARIANTS:
                    probs_by_variant[variant].append(probs[variant].detach().cpu())

        targets = torch.cat(all_targets, dim=0).cpu()
        for variant in VARIANTS:
            probs = torch.cat(probs_by_variant[variant], dim=0)
            metrics = evaluator.calc_accuracy(probs, targets, task_id)
            variant_results[variant]["acc_list"].append(metrics["mean_acc"])
            variant_results[variant]["task_acc_list"].append(metrics["task_acc"])
            variant_results[variant]["metrics"].append(metrics)

        if task_id == 0 or task_id == data_manager.num_tasks - 1:
            correct = {}
            for variant in ["semantic_branch_only", "clip_dino_visual_only", "tric"]:
                probs = torch.cat(probs_by_variant[variant], dim=0)
                correct[variant] = (torch.argmax(probs, dim=1) == targets).numpy()
            key = "base" if task_id == 0 else "final"
            outcome_by_session[key] = {
                "task_id": task_id,
                "num_samples": int(targets.numel()),
                "patterns_sem_visual_tric": percent_patterns(correct),
            }

    for variant, payload in variant_results.items():
        acc = payload["acc_list"]
        payload["avg"] = round(float(sum(acc) / len(acc)), 4)
        payload["pd"] = round(float(acc[0] - acc[-1]), 4)

    output = {
        "dataset": cfg.DATASET.NAME,
        "seed": cfg.SEED,
        "data_cfg": data_cfg,
        "train_cfg": train_cfg,
        "variants": variant_results,
        "classifier_outcome": outcome_by_session,
        "runtime_sec": round(time.time() - started, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output_path), "runtime_sec": output["runtime_sec"]}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description="Run branch and TriC audits for BMVC tables 4, 6, and 7.")
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", default="configs/trainers/bimc_dino_fusion.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_audit(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        seed=args.seed,
        output_root=Path(args.output_root),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
