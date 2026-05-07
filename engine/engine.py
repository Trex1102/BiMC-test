import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.data_manager import DatasetManager
from tqdm import tqdm
from utils.evaluator import AccuracyEvaluator
from models.bimc import BiMC
from models.bimc_adaptive import BiMCAdaptive
from models.bimc_dino_fusion import BiMCDinoFusion
import numpy as np


class Runner:

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_manager = DatasetManager(cfg)
        self.device = cfg.DEVICE.DEVICE_NAME
        self.trainer_name = self._resolve_trainer_name()
        self.model = self._build_model()

        # device
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.is_distributed = True
        else:
            self.is_distributed = False

        self.acc_list = []
        self.task_acc_list = []
        self.task_results = []
        self.evaluator = AccuracyEvaluator(self.data_manager.class_index_in_task)

        self.project_root = Path(__file__).resolve().parent.parent
        output_root = Path(self.cfg.OUTPUT.ROOT)
        if not output_root.is_absolute():
            output_root = self.project_root / output_root
        self.output_root = output_root
        self.experiment_dir = self._create_experiment_dir()
        self._save_config()

    def _create_experiment_dir(self):
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        experiment_dir = self.output_root / self.cfg.DATASET.NAME / self.trainer_name / f'seed{self.cfg.SEED}_{timestamp}'
        experiment_dir.mkdir(parents=True, exist_ok=True)
        print(f'Saving experiment artifacts to: {experiment_dir}')
        return experiment_dir

    def _save_config(self):
        config_path = self.experiment_dir / 'config.yaml'
        config_path.write_text(self.cfg.dump(), encoding='utf-8')

    def _save_results(self, total_runtime_sec):
        final_metrics = self.task_results[-1]['metrics'] if self.task_results else {}
        summary = {
            'dataset': self.cfg.DATASET.NAME,
            'trainer': self.trainer_name,
            'seed': self.cfg.SEED,
            'runtime_sec': round(total_runtime_sec, 3),
            'acc_list': self.acc_list,
            'task_acc_list': self.task_acc_list,
            'task_results': self.task_results,
        }

        results_path = self.experiment_dir / 'results.json'
        with results_path.open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        summary_lines = [
            f"dataset: {summary['dataset']}",
            f"trainer: {summary['trainer']}",
            f"seed: {summary['seed']}",
            f"runtime_sec: {summary['runtime_sec']}",
            f"final_acc: {self.acc_list[-1] if self.acc_list else 'N/A'}",
            f"final_harmonic_acc: {final_metrics.get('harmonic_acc', 'N/A') if self.task_results else 'N/A'}",
            f"final_nll: {final_metrics.get('nll', 'N/A') if self.task_results else 'N/A'}",
            f"final_brier: {final_metrics.get('brier', 'N/A') if self.task_results else 'N/A'}",
            f"final_ece: {final_metrics.get('ece', 'N/A') if self.task_results else 'N/A'}",
            f"acc_list: {self.acc_list}",
            'task-wise acc:',
        ]
        for key in ['beta_prior', 'session_beta', 'oracle_mode', 'oracle_search_metric', 'oracle_beta', 'oracle_delta', 'oracle_tau']:
            if key in final_metrics:
                summary_lines.append(f'{key}: {final_metrics[key]}')
        summary_lines.extend([f'task {i:2d}: {task_acc}' for i, task_acc in enumerate(self.task_acc_list)])
        (self.experiment_dir / 'summary.txt').write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')

    def merge_dicts(self, dict_list):
        result = {}

        keys_to_merge = [
            'description_proto',
            'description_features',
            'description_targets',
            'text_features',
            'text_targets',
            'image_proto',
            'images_features',
            'images_targets',
            'prompt_reliability',
            'description_reliability',
            'text_reliability',
            'visual_reliability',
            'beta_per_class',
            'lambda_t_per_class',
            'lambda_i_per_class',
            'anchor_reliability',
            'image_counts',
            'prompt_counts',
            'description_counts',
            'dino_image_proto',
            'dino_images_features',
            'dino_images_targets',
        ]

        for key in keys_to_merge:
            if key in dict_list[0]:
                result[key] = torch.cat([d[key] for d in dict_list], dim=0)

        weights = [len(d['class_index']) for d in dict_list]

        cov_keys = [
            'cov_image',
            'dino_cov_image',
        ]
        cov_keys = [key for key in cov_keys if key in dict_list[0]]
        cov_sums = {key: torch.zeros_like(dict_list[0][key]) for key in cov_keys}
        weight_sum = sum(weights)

        for i, d in enumerate(dict_list):
            for key in cov_keys:
                cov_sums[key] += d[key] * weights[i]

        for key in cov_keys:
            if weight_sum > 0:
                result[key] = cov_sums[key] / weight_sum

        return result

    @torch.no_grad()
    def run(self):
        print(f'Start inferencing on all tasks: [0, {self.data_manager.num_tasks - 1}]')
        state_dict_list = []
        overall_start_time = time.time()
        for i in range(self.data_manager.num_tasks):
            self.model.eval()

            current_class_name = np.array(self.data_manager.class_names)[self.data_manager.class_index_in_task[i]]
            loader = self.data_manager.get_dataloader(i, source='train', mode='test', accumulate_past=False)

            current_state_dict = self.model.build_task_statistics(
                current_class_name,
                loader,
                class_index=self.data_manager.class_index_in_task[i],
                calibrate_novel_vision_proto=self.cfg.TRAINER.BiMC.VISION_CALIBRATION,
            )

            if 'beta_per_class' in current_state_dict:
                beta_stats = current_state_dict['beta_per_class']
                lambda_t_stats = current_state_dict['lambda_t_per_class']
                print(
                    'adaptive stats | '
                    f'beta mean={beta_stats.mean().item():.4f}, '
                    f'beta min={beta_stats.min().item():.4f}, '
                    f'beta max={beta_stats.max().item():.4f}, '
                    f'lambda_t mean={lambda_t_stats.mean().item():.4f}'
                )
                model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                if hasattr(model_ref, 'estimated_beta_prior') and model_ref.estimated_beta_prior is not None:
                    print(f'adaptive stats | resolved base beta prior={float(model_ref.estimated_beta_prior):.4f}')

            state_dict_list.append(current_state_dict)
            merged_state_dict = self.merge_dicts(state_dict_list)

            start_time = time.time()
            acc = self.inference_task_covariance(i, merged_state_dict)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'+++++++++++  task {i}, time: {elapsed_time} ++++++++++++++++')

            print(f'=> Task [{i}], Acc: {acc["mean_acc"]:.3f}')
            self.acc_list.append(round(acc['mean_acc'], 3))
            self.task_acc_list.append(acc['task_acc'])
            self.task_results.append({
                'task_id': i,
                'elapsed_time_sec': round(elapsed_time, 3),
                'metrics': acc,
            })

        print(f'Final acc:{self.acc_list}')
        print('Task-wise acc:')
        for i, task_acc in enumerate(self.task_acc_list):
            print(f'task {i:2d}, acc:{task_acc}')

        total_runtime_sec = time.time() - overall_start_time
        self._save_results(total_runtime_sec)
        print(f'Saved results to: {self.experiment_dir}')

    @torch.no_grad()
    def inference_task_covariance(self, task_id, state_dict):

        default_beta = self.cfg.DATASET.BETA

        image_proto = state_dict['image_proto']
        cov_image = state_dict['cov_image']
        text_features = state_dict['text_features']
        description_proto = state_dict['description_proto']
        description_features = state_dict['description_features']
        description_targets = state_dict['description_targets']

        num_base_class = len(self.data_manager.class_index_in_task[0])
        num_accumulated_class = max(self.data_manager.class_index_in_task[task_id]) + 1

        test_loader = self.data_manager.get_dataloader(task_id, source='test', mode='test')
        all_logits = []
        all_targets = []
        all_img_feat = []

        model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(model_ref, 'resolve_beta_prior'):
            beta = model_ref.resolve_beta_prior(default_beta)
        else:
            beta = default_beta

        session_beta = None
        if hasattr(model_ref, 'compute_session_joint_beta'):
            session_beta = model_ref.compute_session_joint_beta(
                num_accumulated_class,
                beta,
                state_dict.get('beta_per_class'),
                state_dict.get('text_reliability'),
                state_dict.get('visual_reliability'),
                num_base_cls=num_base_class,
                image_proto=image_proto,
                cov_image=cov_image,
                description_proto=description_proto,
                description_features=description_features,
                description_targets=description_targets,
                text_features=text_features,
                images_features=state_dict.get('images_features'),
                images_targets=state_dict.get('images_targets'),
                lambda_t_per_class=state_dict.get('lambda_t_per_class'),
                current_task_class_index=self.data_manager.class_index_in_task[task_id],
            )
            print(f'beta prior: {float(beta):.4f}')
            if session_beta is not None:
                print(f'joint session beta: {float(session_beta):.4f}')

        extra_kwargs = {}
        for key in [
            'dino_image_proto',
            'dino_cov_image',
            'dino_images_features',
            'dino_images_targets',
        ]:
            if key in state_dict:
                extra_kwargs[key] = state_dict[key]

        if hasattr(model_ref, 'adapt_cfg'):
            for key in [
                'beta_per_class',
                'lambda_t_per_class',
                'lambda_i_per_class',
                'prompt_reliability',
                'description_reliability',
                'text_reliability',
                'visual_reliability',
                'images_features',
                'images_targets',
            ]:
                if key in state_dict:
                    extra_kwargs[key] = state_dict[key]
            extra_kwargs['current_task_class_index'] = self.data_manager.class_index_in_task[task_id]
            if session_beta is not None:
                extra_kwargs['session_beta'] = session_beta

        oracle_cfg = getattr(self.cfg.TRAINER, 'BiMCOracle', None)
        oracle_enabled = bool(getattr(oracle_cfg, 'ENABLE', False)) and hasattr(model_ref, 'oracle_calibrate_session')

        for _, batch in enumerate(tqdm(test_loader)):
            data, targets = self.parse_batch(batch)
            if oracle_enabled:
                img_feat = model_ref.extract_img_feature(data)
                img_feat = F.normalize(img_feat, dim=-1)
                all_img_feat.append(img_feat)
                all_targets.append(targets)
            else:
                logits = self.model.forward_ours(
                    data,
                    num_accumulated_class,
                    num_base_class,
                    image_proto,
                    cov_image,
                    description_proto,
                    description_features,
                    description_targets,
                    text_features,
                    beta=beta,
                    **extra_kwargs,
                )
                all_logits.append(logits)
                all_targets.append(targets)

        all_targets = torch.cat(all_targets, dim=0)
        if oracle_enabled:
            all_img_feat = torch.cat(all_img_feat, dim=0)
            oracle_beta = float(session_beta) if session_beta is not None else float(beta)
            oracle_result = model_ref.oracle_calibrate_session(
                img_feat=all_img_feat,
                targets=all_targets,
                num_cls=num_accumulated_class,
                num_base_cls=num_base_class,
                image_proto=image_proto,
                cov_image=cov_image,
                description_proto=description_proto,
                description_features=description_features,
                description_targets=description_targets,
                text_features=text_features,
                beta=oracle_beta,
                lambda_t_per_class=state_dict.get('lambda_t_per_class'),
            )
            all_logits = oracle_result['probs']
            print(
                'oracle session params | '
                f"mode={oracle_cfg.MODE}, "
                f"beta={oracle_result['beta']:.4f}, "
                f"delta={oracle_result['delta']:.4f}, "
                f"tau={oracle_result['tau']:.4f}"
            )
        else:
            all_logits = torch.cat(all_logits, dim=0)

        eval_acc = self.evaluator.calc_accuracy(all_logits, all_targets, task_id)
        eval_acc['beta_prior'] = round(float(beta), 4)
        if session_beta is not None:
            eval_acc['session_beta'] = round(float(session_beta), 4)
        if oracle_enabled:
            eval_acc['oracle_mode'] = str(oracle_cfg.MODE).lower()
            eval_acc['oracle_search_metric'] = str(oracle_cfg.SEARCH_METRIC).lower()
            eval_acc['oracle_beta'] = round(float(oracle_result['beta']), 4)
            eval_acc['oracle_delta'] = round(float(oracle_result['delta']), 4)
            eval_acc['oracle_tau'] = round(float(oracle_result['tau']), 4)
        print(f"Test acc mean: {eval_acc['mean_acc']}, task-wise acc: {eval_acc['task_acc']}")
        return eval_acc

    def parse_batch(self, batch):
        data = batch['image']
        targets = batch['label']
        data = data.to(self.device)
        targets = targets.to(self.device)
        return data, targets

    def _build_model(self):
        method = getattr(self.cfg, 'METHOD', '').lower()
        if method == 'bimc_adaptive':
            return BiMCAdaptive(self.cfg, self.data_manager.template, self.device)
        if method == 'bimc_dino_fusion':
            return BiMCDinoFusion(self.cfg, self.data_manager.template, self.device)
        return BiMC(self.cfg, self.data_manager.template, self.device)

    def _resolve_trainer_name(self):
        method = getattr(self.cfg, 'METHOD', '').lower()
        if method == 'bimc_adaptive':
            oracle_cfg = getattr(self.cfg.TRAINER, 'BiMCOracle', None)
            oracle_enabled = bool(getattr(oracle_cfg, 'ENABLE', False))
            if oracle_enabled:
                oracle_mode = str(oracle_cfg.MODE).lower()
                trainer_name = f'bimc_oracle_{oracle_mode}'
                if self.cfg.TRAINER.BiMC.USING_ENSEMBLE:
                    trainer_name = f'{trainer_name}_ensemble'
                return trainer_name
            trainer_name = 'bimc_adaptive'
            beta_mode = str(self.cfg.TRAINER.BiMCAdaptive.BETA_MODE).lower()
            if beta_mode == 'session_joint':
                trainer_name = 'bimc_adaptive_joint'
            elif beta_mode == 'session_risk':
                trainer_name = 'bimc_adaptive_risk'
            prior_mode = str(self.cfg.TRAINER.BiMCAdaptive.PRIOR_MODE).lower()
            if beta_mode == 'session_risk' and prior_mode != 'fixed':
                if prior_mode == 'base_pseudo':
                    trainer_name = f'{trainer_name}_pseudo_prior'
                else:
                    trainer_name = f'{trainer_name}_{prior_mode}'
            objective_mode = str(self.cfg.TRAINER.BiMCAdaptive.SESSION_RISK_OBJECTIVE).lower()
            if beta_mode == 'session_risk' and objective_mode != 'nll':
                trainer_name = f'{trainer_name}_{objective_mode}'
            if self.cfg.TRAINER.BiMC.USING_ENSEMBLE:
                trainer_name = f'{trainer_name}_ensemble'
            return trainer_name
        if method == 'bimc_dino_fusion':
            return 'bimc_dino_fusion_ensemble' if self.cfg.TRAINER.BiMC.USING_ENSEMBLE else 'bimc_dino_fusion'
        return 'bimc_ensemble' if self.cfg.TRAINER.BiMC.USING_ENSEMBLE else 'bimc'
