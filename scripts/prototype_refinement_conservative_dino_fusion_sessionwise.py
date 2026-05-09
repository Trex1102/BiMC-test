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
from prototype_refinement_conservative_dino_fusion import compose_dino_fusion_probs, merge_states, normalize_rows
from query_graph_router_killswitch import accuracy_stats
from semantic_option2_killswitch import to_builtin
from utils.util import set_gpu, set_seed


def create_output_dir(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"prototype_refinement_conservative_pseudoval_sessionwise_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_session_states(cfg):
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
    return data_manager, model, states


@torch.no_grad()
def collect_session_queries(cfg, data_manager, model, task_id):
    loader = data_manager.get_dataloader(task_id, source='test', mode='test')
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


def session_components(cfg, data_manager, merged_state, task_id, clip_query, dino_query):
    device = cfg.DEVICE.DEVICE_NAME
    num_cls = int(max(data_manager.class_index_in_task[task_id]) + 1)
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

    return {
        'num_cls': num_cls,
        'num_base_cls': num_base_cls,
        'beta': beta,
        'lambda_t': lambda_t,
        'ensemble_alpha': ensemble_alpha,
        'semantic_proto': semantic_proto,
        'clip_image_proto': clip_image_proto,
        'dino_image_proto': dino_image_proto,
        'dino_cov': dino_cov,
        'description_features': description_features,
        'description_targets': description_targets,
    }


def compose_from_components(clip_query, dino_query, components, dino_image_proto):
    return compose_dino_fusion_probs(
        clip_query,
        dino_query,
        components['semantic_proto'],
        components['clip_image_proto'],
        dino_image_proto,
        components['dino_cov'],
        components['description_features'],
        components['description_targets'],
        components['num_base_cls'],
        components['beta'],
        components['ensemble_alpha'],
    )


def pseudo_validation_split(
    baseline_probs,
    num_base_cls,
    num_cls,
    mass_thr,
    min_count,
    val_fraction,
    base_val_conf_thr,
    base_val_max_per_class,
    topk_per_class=None,
):
    preds = torch.argmax(baseline_probs, dim=1)
    max_probs = baseline_probs.max(dim=1).values
    novel_probs = baseline_probs[:, num_base_cls:]
    novel_mass = novel_probs.sum(dim=1)
    top_novel_local = torch.argmax(novel_probs, dim=1)
    top_novel_global = top_novel_local + int(num_base_cls)
    novel_gate = (preds >= int(num_base_cls)) & (novel_mass >= float(mass_thr))

    update_indices_by_class = {}
    validation_indices = []
    validation_labels = []
    per_class_counts = {}
    per_class_validation_counts = {}
    per_class_update_counts = {}

    for cls_id in range(num_base_cls, num_cls):
        cls_mask = novel_gate & (top_novel_global == int(cls_id))
        cls_indices = torch.nonzero(cls_mask, as_tuple=False).flatten()
        count = int(cls_indices.numel())
        per_class_counts[int(cls_id)] = count
        if count < int(min_count):
            per_class_update_counts[int(cls_id)] = 0
            per_class_validation_counts[int(cls_id)] = 0
            continue

        cls_scores = baseline_probs[cls_indices, int(cls_id)]
        order = torch.argsort(cls_scores, descending=True)
        cls_indices = cls_indices[order]
        if topk_per_class is not None and count > int(topk_per_class):
            cls_indices = cls_indices[: int(topk_per_class)]
            count = int(cls_indices.numel())

        val_count = int(round(count * float(val_fraction)))
        val_count = max(1, val_count)
        val_count = min(val_count, count - int(min_count))
        if val_count > 0:
            update_indices = cls_indices[:-val_count]
            val_indices = cls_indices[-val_count:]
            validation_indices.append(val_indices)
            validation_labels.append(torch.full_like(val_indices, int(cls_id)))
        else:
            update_indices = cls_indices

        if int(update_indices.numel()) >= int(min_count):
            update_indices_by_class[int(cls_id)] = update_indices
            per_class_update_counts[int(cls_id)] = int(update_indices.numel())
        else:
            per_class_update_counts[int(cls_id)] = 0
        per_class_validation_counts[int(cls_id)] = max(int(val_count), 0)

    for cls_id in range(num_base_cls):
        cls_mask = (preds == int(cls_id)) & (max_probs >= float(base_val_conf_thr))
        cls_indices = torch.nonzero(cls_mask, as_tuple=False).flatten()
        if cls_indices.numel() == 0:
            continue
        cls_scores = max_probs[cls_indices]
        order = torch.argsort(cls_scores, descending=True)
        cls_indices = cls_indices[order]
        if base_val_max_per_class is not None:
            cls_indices = cls_indices[: int(base_val_max_per_class)]
        validation_indices.append(cls_indices)
        validation_labels.append(torch.full_like(cls_indices, int(cls_id)))

    if validation_indices:
        val_indices = torch.cat(validation_indices, dim=0)
        val_labels = torch.cat(validation_labels, dim=0)
    else:
        val_indices = torch.empty(0, device=baseline_probs.device, dtype=torch.long)
        val_labels = torch.empty(0, device=baseline_probs.device, dtype=torch.long)

    return {
        'update_indices_by_class': update_indices_by_class,
        'validation_indices': val_indices,
        'validation_labels': val_labels,
        'novel_gated_query_count': int(novel_gate.sum().item()),
        'novel_validation_count': int(sum(per_class_validation_counts.values())),
        'base_validation_count': int(val_indices.numel()) - int(sum(per_class_validation_counts.values())),
        'updated_class_count': int(len(update_indices_by_class)),
        'per_class_counts': per_class_counts,
        'per_class_update_counts': per_class_update_counts,
        'per_class_validation_counts': per_class_validation_counts,
    }


def refine_from_index_groups(query_features, baseline_probs, orig_image_proto, update_indices_by_class, alpha):
    image_proto = orig_image_proto.clone()
    for cls_id, indices in update_indices_by_class.items():
        if indices.numel() == 0:
            continue
        cls_query_features = query_features[indices]
        weights = baseline_probs[indices, int(cls_id)]
        weights = weights / torch.clamp(weights.sum(), min=1e-12)
        proto = torch.matmul(weights.unsqueeze(0), cls_query_features).squeeze(0)
        proto = F.normalize(proto.unsqueeze(0), dim=-1).squeeze(0)
        mixed = (1.0 - float(alpha)) * orig_image_proto[int(cls_id)] + float(alpha) * proto
        image_proto[int(cls_id)] = F.normalize(mixed.unsqueeze(0), dim=-1).squeeze(0)
    return image_proto


def summarize_pseudo_split(split):
    return {
        'novel_gated_query_count': int(split['novel_gated_query_count']),
        'validation_count': int(split['validation_indices'].numel()),
        'novel_validation_count': int(split['novel_validation_count']),
        'base_validation_count': int(split['base_validation_count']),
        'updated_class_count': int(split['updated_class_count']),
        'per_class_counts': split['per_class_counts'],
        'per_class_update_counts': split['per_class_update_counts'],
        'per_class_validation_counts': split['per_class_validation_counts'],
    }


def choose_alpha_by_pseudo_validation(
    clip_query,
    dino_query,
    components,
    baseline_probs,
    alpha_grid,
    mass_thr,
    min_count,
    val_fraction,
    base_val_conf_thr,
    base_val_max_per_class,
    fallback_alpha,
    topk_per_class=None,
):
    split = pseudo_validation_split(
        baseline_probs=baseline_probs,
        num_base_cls=components['num_base_cls'],
        num_cls=components['num_cls'],
        mass_thr=mass_thr,
        min_count=min_count,
        val_fraction=val_fraction,
        base_val_conf_thr=base_val_conf_thr,
        base_val_max_per_class=base_val_max_per_class,
        topk_per_class=topk_per_class,
    )
    val_indices = split['validation_indices']
    val_labels = split['validation_labels']
    split_summary = summarize_pseudo_split(split)
    if val_indices.numel() == 0 or not split['update_indices_by_class']:
        return float(fallback_alpha), {
            'mode': 'pseudo_validation',
            'fallback_used': True,
            'reason': 'no pseudo-validation samples or no update classes',
            'selected_alpha': float(fallback_alpha),
            'candidates': [],
            'split': split_summary,
        }

    candidates = []
    best = None
    for alpha in alpha_grid:
        refined_dino_proto = refine_from_index_groups(
            query_features=dino_query,
            baseline_probs=baseline_probs,
            orig_image_proto=components['dino_image_proto'],
            update_indices_by_class=split['update_indices_by_class'],
            alpha=alpha,
        )
        refined_probs = compose_from_components(clip_query, dino_query, components, refined_dino_proto)
        selected_probs = refined_probs[val_indices, val_labels]
        pseudo_nll = -torch.log(torch.clamp(selected_probs, min=1e-12)).mean()
        pseudo_agreement = (torch.argmax(refined_probs[val_indices], dim=1) == val_labels).float().mean()
        score = -float(pseudo_nll.item())
        record = {
            'alpha': float(alpha),
            'pseudo_val_score': round(score, 6),
            'pseudo_val_nll': round(float(pseudo_nll.item()), 6),
            'pseudo_val_agreement': round(float(pseudo_agreement.item() * 100.0), 3),
        }
        candidates.append(record)
        if (
            best is None
            or score > best['score'] + 1e-12
            or (abs(score - best['score']) <= 1e-12 and float(alpha) < best['alpha'])
        ):
            best = {
                'alpha': float(alpha),
                'score': score,
            }

    return best['alpha'], {
        'mode': 'pseudo_validation',
        'fallback_used': False,
        'selected_alpha': float(best['alpha']),
        'candidates': candidates,
        'split': split_summary,
    }


def evaluate_one_session(
    cfg,
    data_manager,
    model,
    states,
    task_id,
    alpha_grid,
    mass_thr,
    min_count,
    val_fraction,
    base_val_conf_thr,
    base_val_max_per_class,
    fallback_alpha,
    topk_per_class=None,
):
    merged_state = merge_states(states[:task_id + 1])
    clip_query, dino_query, query_targets = collect_session_queries(cfg, data_manager, model, task_id)
    device = cfg.DEVICE.DEVICE_NAME
    clip_query = normalize_rows(clip_query.to(device))
    dino_query = normalize_rows(dino_query.to(device))
    query_targets = query_targets.to(device)

    components = session_components(cfg, data_manager, merged_state, task_id, clip_query, dino_query)
    dino_image_proto = components['dino_image_proto']
    baseline_probs = compose_from_components(clip_query, dino_query, components, dino_image_proto)
    baseline_full, baseline_base, baseline_novel = accuracy_stats(
        baseline_probs,
        query_targets,
        components['num_base_cls'],
    )

    if task_id == 0:
        return {
            'session': int(task_id),
            'num_seen_classes': int(components['num_cls']),
            'baseline_full_acc': round(float(baseline_full), 3),
            'baseline_base_acc': round(float(baseline_base), 3),
            'baseline_novel_acc': round(float(baseline_novel), 3),
            'best_alpha': None,
            'final_full_acc': round(float(baseline_full), 3),
            'final_base_acc': round(float(baseline_base), 3),
            'final_novel_acc': round(float(baseline_novel), 3),
            'final_gain': 0.0,
            'updated_class_count': 0,
            'gated_query_count': 0,
            'alpha_selection': {
                'mode': 'not_applicable_base_session',
                'selected_alpha': None,
            },
        }

    selected_alpha, alpha_selection = choose_alpha_by_pseudo_validation(
        clip_query=clip_query,
        dino_query=dino_query,
        components=components,
        baseline_probs=baseline_probs,
        alpha_grid=alpha_grid,
        mass_thr=mass_thr,
        min_count=min_count,
        val_fraction=val_fraction,
        base_val_conf_thr=base_val_conf_thr,
        base_val_max_per_class=base_val_max_per_class,
        fallback_alpha=fallback_alpha,
        topk_per_class=topk_per_class,
    )
    refined_dino_proto, stats = conservative_refine_once(
        query_features=dino_query,
        query_targets=query_targets,
        baseline_probs=baseline_probs,
        orig_image_proto=dino_image_proto,
        num_base_cls=components['num_base_cls'],
        num_cls=components['num_cls'],
        alpha=selected_alpha,
        mass_thr=mass_thr,
        min_count=min_count,
        topk_per_class=topk_per_class,
    )
    refined_probs = compose_from_components(clip_query, dino_query, components, refined_dino_proto)
    full_acc, base_acc, novel_acc = accuracy_stats(refined_probs, query_targets, components['num_base_cls'])

    return {
        'session': int(task_id),
        'num_seen_classes': int(components['num_cls']),
        'baseline_full_acc': round(float(baseline_full), 3),
        'baseline_base_acc': round(float(baseline_base), 3),
        'baseline_novel_acc': round(float(baseline_novel), 3),
        'selected_alpha': round(float(selected_alpha), 3),
        'final_full_acc': round(float(full_acc), 3),
        'final_base_acc': round(float(base_acc), 3),
        'final_novel_acc': round(float(novel_acc), 3),
        'final_gain': round(float(full_acc - baseline_full), 3),
        'updated_class_count': int(stats['updated_class_count']),
        'gated_query_count': int(stats['gated_query_count']),
        'gated_base_count': int(stats['gated_base_count']),
        'gated_novel_count': int(stats['gated_novel_count']),
        'alpha_selection': alpha_selection,
    }


def run_dataset(
    data_cfg,
    train_cfg,
    alpha_grid,
    mass_thr,
    min_count,
    val_fraction,
    base_val_conf_thr,
    base_val_max_per_class,
    fallback_alpha,
    topk_per_class=None,
    seed_override=None,
):
    cfg = setup_cfg(data_cfg, train_cfg)
    if seed_override is not None:
        cfg.defrost()
        cfg.SEED = int(seed_override)
        cfg.freeze()

    out_dir = create_output_dir(cfg, train_cfg)
    (out_dir / 'config.yaml').write_text(cfg.dump(), encoding='utf-8')
    start_time = time.time()

    data_manager, model, states = build_session_states(cfg)
    session_results = []
    for task_id in range(data_manager.num_tasks):
        session_result = evaluate_one_session(
            cfg,
            data_manager,
            model,
            states,
            task_id,
            alpha_grid=alpha_grid,
            mass_thr=mass_thr,
            min_count=min_count,
            val_fraction=val_fraction,
            base_val_conf_thr=base_val_conf_thr,
            base_val_max_per_class=base_val_max_per_class,
            fallback_alpha=fallback_alpha,
            topk_per_class=topk_per_class,
        )
        session_results.append(session_result)
        print(
            'session {session}: baseline={baseline_full_acc:.3f} final={final_full_acc:.3f} gain={final_gain:.3f}'.format(
                **session_result
            )
        )

    baseline_acc_list = [float(item['baseline_full_acc']) for item in session_results]
    final_acc_list = [float(item['final_full_acc']) for item in session_results]
    gain_list = [float(item['final_gain']) for item in session_results]

    payload = {
        'dataset': cfg.DATASET.NAME,
        'seed': int(cfg.SEED),
        'train_cfg': train_cfg,
        'method': 'sessionwise conservative query prototype refinement on bimc_dino_fusion with pseudo-validation alpha selection',
        'runtime_sec': round(time.time() - start_time, 3),
        'alpha_grid': [float(x) for x in alpha_grid],
        'alpha_selection': 'pseudo_validation_no_test_labels',
        'mass_thr': float(mass_thr),
        'min_count': int(min_count),
        'val_fraction': float(val_fraction),
        'base_val_conf_thr': float(base_val_conf_thr),
        'base_val_max_per_class': None if base_val_max_per_class is None else int(base_val_max_per_class),
        'fallback_alpha': float(fallback_alpha),
        'topk_per_class': None if topk_per_class is None else int(topk_per_class),
        'baseline_acc_list': baseline_acc_list,
        'final_acc_list': final_acc_list,
        'gain_list': gain_list,
        'baseline_avg': round(float(np.mean(baseline_acc_list)), 3),
        'baseline_pd': round(float(baseline_acc_list[0] - baseline_acc_list[-1]), 3),
        'final_avg': round(float(np.mean(final_acc_list)), 3),
        'final_pd': round(float(final_acc_list[0] - final_acc_list[-1]), 3),
        'gain_avg': round(float(np.mean(gain_list)), 3),
        'session_results': session_results,
    }
    payload = to_builtin(payload)
    with (out_dir / 'results.json').open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    summary_lines = [
        f"dataset: {payload['dataset']}",
        f"seed: {payload['seed']}",
        f"baseline_acc_list: {payload['baseline_acc_list']}",
        f"final_acc_list: {payload['final_acc_list']}",
        f"gain_list: {payload['gain_list']}",
        f"baseline_avg: {payload['baseline_avg']}",
        f"baseline_pd: {payload['baseline_pd']}",
        f"final_avg: {payload['final_avg']}",
        f"final_pd: {payload['final_pd']}",
        f"gain_avg: {payload['gain_avg']}",
    ]
    (out_dir / 'summary.txt').write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description='Session-wise conservative query prototype refinement on BiMC DinoV2 fusion with pseudo-validation alpha selection.')
    parser.add_argument('--data_cfg', required=True)
    parser.add_argument('--train_cfg', required=True)
    parser.add_argument('--alpha_grid', type=float, nargs='+', default=[0.5, 0.75, 1.0])
    parser.add_argument('--mass_thr', type=float, default=0.3)
    parser.add_argument('--min_count', type=int, default=5)
    parser.add_argument('--val_fraction', type=float, default=0.2)
    parser.add_argument('--base_val_conf_thr', type=float, default=0.5)
    parser.add_argument('--base_val_max_per_class', type=int, default=50)
    parser.add_argument('--fallback_alpha', type=float, default=0.75)
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
        val_fraction=args.val_fraction,
        base_val_conf_thr=args.base_val_conf_thr,
        base_val_max_per_class=args.base_val_max_per_class,
        fallback_alpha=args.fallback_alpha,
        topk_per_class=args.topk_per_class,
        seed_override=args.seed_override,
    )


if __name__ == '__main__':
    main()
