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
from models.bimc_dino_fusion import BiMCDinoFusion
from prototype_refinement_conservative import conservative_refine_once
from query_graph_router_killswitch import accuracy_stats
from semantic_option2_killswitch import to_builtin
from utils.util import set_gpu, set_seed


CAT_KEYS = [
    'description_proto',
    'description_features',
    'description_targets',
    'text_features',
    'text_targets',
    'image_proto',
    'images_features',
    'images_targets',
    'dino_image_proto',
    'dino_images_features',
    'dino_images_targets',
]
COV_KEYS = ['cov_image', 'dino_cov_image']


def normalize_rows(tensor):
    return F.normalize(tensor.float(), dim=-1)


def merge_states(states):
    merged = {}
    for key in CAT_KEYS:
        if key in states[0]:
            merged[key] = torch.cat([state[key] for state in states], dim=0)
    weights = [len(state['class_index']) for state in states]
    total_weight = sum(weights)
    for key in COV_KEYS:
        if key in states[0]:
            merged[key] = sum(state[key] * weights[idx] for idx, state in enumerate(states)) / total_weight
    return merged


def create_output_dir(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"prototype_refinement_conservative_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def extract_final_state(cfg):
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)
    data_manager = DatasetManager(cfg)
    model = BiMCDinoFusion(cfg, data_manager.template, cfg.DEVICE.DEVICE_NAME)
    model.eval()

    states = []
    for task_id in range(data_manager.num_tasks):
        class_names = np.array(data_manager.class_names)[data_manager.class_index_in_task[task_id]]
        train_loader = data_manager.get_dataloader(task_id, source='train', mode='test', accumulate_past=False)
        state = model.build_task_statistics(
            class_names,
            train_loader,
            class_index=data_manager.class_index_in_task[task_id],
            calibrate_novel_vision_proto=cfg.TRAINER.BiMC.VISION_CALIBRATION,
        )
        states.append(state)
    return data_manager, model, merge_states(states), data_manager.num_tasks - 1


@torch.no_grad()
def collect_final_queries(cfg, data_manager, model, final_task_id):
    loader = data_manager.get_dataloader(final_task_id, source='test', mode='test')
    clip_features = []
    dino_features = []
    targets_all = []
    for batch in loader:
        images = batch['image'].to(cfg.DEVICE.DEVICE_NAME)
        targets = batch['label'].to(cfg.DEVICE.DEVICE_NAME)
        clip_features.append(F.normalize(model.extract_img_feature(images), dim=-1))
        dino_features.append(model.extract_dino_img_feature(images))
        targets_all.append(targets)
    return torch.cat(clip_features, dim=0), torch.cat(dino_features, dim=0), torch.cat(targets_all, dim=0)


def knn_logits(query_features, support_features, support_labels, num_cls):
    similarity = torch.matmul(query_features, support_features.T)
    out = torch.full((query_features.size(0), num_cls), float('-inf'), device=query_features.device, dtype=query_features.dtype)
    expanded_labels = support_labels.unsqueeze(0).expand(query_features.size(0), -1)
    for label in range(num_cls):
        mask = expanded_labels == label
        masked = similarity.masked_fill(~mask, float('-inf'))
        out[:, label] = masked.max(dim=1).values
    return out


def mahalanobis_logits(query_features, prototypes, cov):
    inv_cov = torch.pinverse(cov.to(dtype=torch.float32)).to(dtype=prototypes.dtype)
    logits = []
    for cls_id in range(prototypes.shape[0]):
        dist = query_features - prototypes[cls_id]
        left = torch.matmul(dist, inv_cov)
        logits.append(-torch.sum(left * dist, dim=1))
    return torch.stack(logits, dim=1)


def compose_dino_fusion_probs(
    clip_query,
    dino_query,
    semantic_proto,
    clip_image_proto,
    dino_image_proto,
    dino_cov,
    description_features,
    description_targets,
    num_base_cls,
    beta,
    ensemble_alpha,
    omega=0.7,
    logit_temp=100.0,
    cov_scale=1.0,
):
    num_cls = int(dino_image_proto.shape[0])
    text_logits = torch.matmul(clip_query, semantic_proto.T)
    clip_visual_logits = torch.matmul(clip_query, clip_image_proto.T)
    dino_visual_logits = torch.matmul(dino_query, dino_image_proto.T)
    visual_logits = float(omega) * dino_visual_logits + (1.0 - float(omega)) * clip_visual_logits
    fused_logits = float(beta) * text_logits + (1.0 - float(beta)) * visual_logits
    prob_fused = F.softmax(fused_logits * float(logit_temp), dim=-1)

    prob_cov = F.softmax(mahalanobis_logits(dino_query, dino_image_proto, dino_cov) / float(cov_scale), dim=-1)
    prob_knn = F.softmax(knn_logits(clip_query, description_features, description_targets, num_cls), dim=-1)

    if ensemble_alpha >= 1.0:
        return prob_fused
    base_probs = ensemble_alpha * prob_fused[:, :num_base_cls] + (1.0 - ensemble_alpha) * prob_cov[:, :num_base_cls]
    inc_probs = ensemble_alpha * prob_fused[:, num_base_cls:] + (1.0 - ensemble_alpha) * prob_knn[:, num_base_cls:]
    return torch.cat([base_probs, inc_probs], dim=1)


def run_dataset(data_cfg, train_cfg, alpha_grid, mass_thr, min_count, topk_per_class=None, seed_override=None):
    cfg = setup_cfg(data_cfg, train_cfg)
    if seed_override is not None:
        cfg.defrost()
        cfg.SEED = int(seed_override)
        cfg.freeze()

    out_dir = create_output_dir(cfg, train_cfg)
    (out_dir / 'config.yaml').write_text(cfg.dump(), encoding='utf-8')
    start_time = time.time()

    data_manager, model, merged_state, final_task_id = extract_final_state(cfg)
    clip_query, dino_query, query_targets = collect_final_queries(cfg, data_manager, model, final_task_id)
    device = cfg.DEVICE.DEVICE_NAME
    clip_query = normalize_rows(clip_query.to(device))
    dino_query = normalize_rows(dino_query.to(device))
    query_targets = query_targets.to(device)

    num_cls = int(max(data_manager.class_index_in_task[final_task_id]) + 1)
    num_base_cls = int(len(data_manager.class_index_in_task[0]))
    beta = float(cfg.DATASET.BETA)
    lambda_t = float(cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
    ensemble_alpha = float(cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.0)

    text_features = normalize_rows(merged_state['text_features'][:num_cls].to(device))
    description_proto = normalize_rows(merged_state['description_proto'][:num_cls].to(device))
    semantic_proto = normalize_rows((1.0 - lambda_t) * text_features + lambda_t * description_proto)
    clip_image_proto = normalize_rows(merged_state['image_proto'][:num_cls].to(device))
    dino_image_proto = normalize_rows(merged_state['dino_image_proto'][:num_cls].to(device))
    dino_cov = merged_state['dino_cov_image'].to(device).float()
    description_features = normalize_rows(merged_state['description_features'].to(device))
    description_targets = merged_state['description_targets'].to(device)

    baseline_probs = compose_dino_fusion_probs(
        clip_query,
        dino_query,
        semantic_proto,
        clip_image_proto,
        dino_image_proto,
        dino_cov,
        description_features,
        description_targets,
        num_base_cls,
        beta,
        ensemble_alpha,
    )
    baseline_full, baseline_base, baseline_novel = accuracy_stats(baseline_probs, query_targets, num_base_cls)

    sweep = []
    best = None
    for alpha in alpha_grid:
        refined_dino_proto, stats = conservative_refine_once(
            query_features=dino_query,
            query_targets=query_targets,
            baseline_probs=baseline_probs,
            orig_image_proto=dino_image_proto,
            num_base_cls=num_base_cls,
            num_cls=num_cls,
            alpha=alpha,
            mass_thr=mass_thr,
            min_count=min_count,
            topk_per_class=topk_per_class,
        )
        refined_probs = compose_dino_fusion_probs(
            clip_query,
            dino_query,
            semantic_proto,
            clip_image_proto,
            refined_dino_proto,
            dino_cov,
            description_features,
            description_targets,
            num_base_cls,
            beta,
            ensemble_alpha,
        )
        full_acc, base_acc, novel_acc = accuracy_stats(refined_probs, query_targets, num_base_cls)
        record = {
            'alpha': float(alpha),
            'full_acc': round(float(full_acc), 3),
            'base_acc': round(float(base_acc), 3),
            'novel_acc': round(float(novel_acc), 3),
            'gain': round(float(full_acc - baseline_full), 3),
            'updated_class_count': int(stats['updated_class_count']),
            'updated_class_rate': round(float(stats['updated_class_rate']), 4),
            'mean_queries_per_updated_class': round(float(stats['mean_queries_per_updated_class']), 3),
            'mean_selected_per_updated_class': round(float(stats['mean_selected_per_updated_class']), 3),
            'gated_query_count': int(stats['gated_query_count']),
            'gated_base_count': int(stats['gated_base_count']),
            'gated_novel_count': int(stats['gated_novel_count']),
        }
        sweep.append(record)
        if best is None or full_acc > best['full_acc']:
            best = {
                'alpha': float(alpha),
                'full_acc': float(full_acc),
                'base_acc': float(base_acc),
                'novel_acc': float(novel_acc),
                'stats': stats,
            }

    payload = {
        'dataset': cfg.DATASET.NAME,
        'train_cfg': train_cfg,
        'seed': int(cfg.SEED),
        'method': 'bimc_dino_fusion + conservative_dino_query_prototype_refinement',
        'runtime_sec': round(time.time() - start_time, 3),
        'omega': 0.7,
        'beta': beta,
        'lambda_t': lambda_t,
        'ensemble_alpha': ensemble_alpha,
        'num_seen_classes': num_cls,
        'num_base_classes': num_base_cls,
        'alpha_grid': [float(x) for x in alpha_grid],
        'mass_thr': float(mass_thr),
        'min_count': int(min_count),
        'topk_per_class': None if topk_per_class is None else int(topk_per_class),
        'baseline_full_acc': round(float(baseline_full), 3),
        'baseline_base_acc': round(float(baseline_base), 3),
        'baseline_novel_acc': round(float(baseline_novel), 3),
        'best_alpha': round(float(best['alpha']), 3),
        'final_full_acc': round(float(best['full_acc']), 3),
        'final_base_acc': round(float(best['base_acc']), 3),
        'final_novel_acc': round(float(best['novel_acc']), 3),
        'final_gain': round(float(best['full_acc'] - baseline_full), 3),
        'updated_class_count': int(best['stats']['updated_class_count']),
        'updated_class_rate': round(float(best['stats']['updated_class_rate']), 4),
        'gated_query_count': int(best['stats']['gated_query_count']),
        'gated_base_count': int(best['stats']['gated_base_count']),
        'gated_novel_count': int(best['stats']['gated_novel_count']),
        'alpha_sweep': sweep,
    }
    payload = to_builtin(payload)
    with (out_dir / 'results.json').open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    summary_lines = [
        f"dataset: {payload['dataset']}",
        f"seed: {payload['seed']}",
        f"baseline_full_acc: {payload['baseline_full_acc']}",
        f"baseline_base_acc: {payload['baseline_base_acc']}",
        f"baseline_novel_acc: {payload['baseline_novel_acc']}",
        f"best_alpha: {payload['best_alpha']}",
        f"final_full_acc: {payload['final_full_acc']}",
        f"final_base_acc: {payload['final_base_acc']}",
        f"final_novel_acc: {payload['final_novel_acc']}",
        f"final_gain: {payload['final_gain']}",
    ]
    (out_dir / 'summary.txt').write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description='Conservative query prototype refinement on top of BiMC DinoV2 fusion.')
    parser.add_argument('--data_cfg', required=True)
    parser.add_argument('--train_cfg', required=True)
    parser.add_argument('--alpha_grid', type=float, nargs='+', default=[0.5, 0.75, 1.0])
    parser.add_argument('--mass_thr', type=float, default=0.3)
    parser.add_argument('--min_count', type=int, default=5)
    parser.add_argument('--topk_per_class', type=int, default=None)
    parser.add_argument('--seed_override', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        alpha_grid=args.alpha_grid,
        mass_thr=args.mass_thr,
        min_count=args.min_count,
        topk_per_class=args.topk_per_class,
        seed_override=args.seed_override,
    )


if __name__ == '__main__':
    main()
