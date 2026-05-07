import torch
import torch.nn.functional as F
import models.clip.clip as clip

from models.bimc import BiMC


class BiMCAdaptive(BiMC):
    def __init__(self, cfg, template, device):
        super().__init__(cfg, template, device)
        self.adapt_cfg = cfg.TRAINER.BiMCAdaptive
        self.prompt_templates = list(self.adapt_cfg.PROMPT_TEMPLATES)
        if not self.adapt_cfg.USE_PROMPT_ENSEMBLE:
            self.prompt_templates = list(template)
        self.base_visual_reliability = None
        self.estimated_beta_prior = None

    @torch.no_grad()
    def inference_prompt_bank(self, class_names, cls_begin_index):
        all_embeddings = []
        all_targets = []
        k = cls_begin_index
        for classname in class_names:
            classname = classname.replace('_', ' ')
            classname = classname.replace('-', ' ')
            texts = [t.format(classname) for t in self.prompt_templates]
            tokens = clip.tokenize(texts).to(self.device)
            class_embeddings = self.clip_model.encode_text(tokens)
            class_embeddings = F.normalize(class_embeddings, dim=-1)
            all_embeddings.append(class_embeddings)
            all_targets.append(torch.full((len(self.prompt_templates),), k, dtype=torch.long, device=class_embeddings.device))
            k += 1
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        return all_embeddings, all_targets

    def _group_class_features(self, features, targets, class_index):
        grouped = []
        for cls_id in class_index:
            idx = torch.where(targets == int(cls_id))[0]
            grouped.append(features[idx])
        return grouped

    def _estimate_reliability(self, features):
        eps = self.adapt_cfg.EPS
        if features.numel() == 0:
            return features.new_tensor(eps)

        features = F.normalize(features, dim=-1)
        if features.shape[0] <= 1:
            return features.new_tensor(self.adapt_cfg.SINGLETON_KAPPA)

        resultant = features.sum(dim=0)
        mean_resultant = resultant.norm() / features.shape[0]
        mean_resultant = torch.clamp(mean_resultant, min=0.0, max=1.0 - 1e-4)
        dim = features.shape[-1]
        kappa = mean_resultant * (dim - mean_resultant.pow(2)) / torch.clamp(1 - mean_resultant.pow(2), min=eps)
        kappa = torch.nan_to_num(
            kappa,
            nan=self.adapt_cfg.SINGLETON_KAPPA,
            posinf=self.adapt_cfg.MAX_KAPPA,
            neginf=0.0,
        )
        kappa = torch.clamp(kappa, min=0.0, max=self.adapt_cfg.MAX_KAPPA)
        count_scale = float(features.shape[0]) ** self.adapt_cfg.RELIABILITY_COUNT_POWER
        return features.new_tensor(count_scale) * kappa

    def _agreement_gate(self, proto_a, proto_b):
        if not self.adapt_cfg.USE_AGREEMENT_GATE:
            return proto_a.new_tensor(1.0)
        agreement = torch.dot(F.normalize(proto_a, dim=-1), F.normalize(proto_b, dim=-1))
        agreement = torch.clamp(agreement, min=self.adapt_cfg.MIN_AGREEMENT, max=1.0)
        return agreement.pow(self.adapt_cfg.AGREEMENT_POWER)

    def _prior_logit_weight(self, prior, numerator, denominator):
        eps = self.adapt_cfg.EPS
        prior_tensor = numerator.new_tensor(float(prior))
        prior_tensor = torch.clamp(prior_tensor, min=eps, max=1.0 - eps)
        prior_logit = self._logit(prior_tensor)
        evidence_delta = torch.log(numerator + eps) - torch.log(denominator + eps)
        return torch.sigmoid(prior_logit + self.adapt_cfg.RELIABILITY_LOGIT_SCALE * evidence_delta)

    def _logit(self, value):
        eps = self.adapt_cfg.EPS
        value = torch.clamp(value, min=eps, max=1.0 - eps)
        return torch.log(value) - torch.log1p(-value)

    def resolve_beta_prior(self, default_beta):
        mode = str(self.adapt_cfg.PRIOR_MODE).lower()
        if mode in {'base_auto', 'base_pseudo'} and self.estimated_beta_prior is not None:
            prior = float(self.estimated_beta_prior)
        elif mode == 'universal':
            prior = float(self.adapt_cfg.UNIVERSAL_BETA)
        else:
            prior = float(default_beta)
        prior = min(max(prior, self.adapt_cfg.EPS), 1.0 - self.adapt_cfg.EPS)
        return prior

    def _beta_grid(self, grid_size, device):
        grid_size = max(int(grid_size), 2)
        return torch.linspace(
            float(self.adapt_cfg.MIN_WEIGHT),
            float(self.adapt_cfg.MAX_WEIGHT),
            steps=grid_size,
            device=device,
        )

    def _delta_grid(self, device):
        oracle_cfg = self.cfg.TRAINER.BiMCOracle
        grid_size = max(int(oracle_cfg.DELTA_GRID_SIZE), 2)
        return torch.linspace(
            float(oracle_cfg.DELTA_MIN),
            float(oracle_cfg.DELTA_MAX),
            steps=grid_size,
            device=device,
        )

    def _tau_grid(self, device):
        tau_values = [float(v) for v in self.cfg.TRAINER.BiMCOracle.TAU_VALUES]
        if not tau_values:
            tau_values = [1.0]
        return torch.tensor(tau_values, device=device)

    def _posthoc_calibrate_probabilities(self, probs, num_base_cls, tau=1.0, delta=0.0):
        eps = self.adapt_cfg.EPS
        logits = torch.log(torch.clamp(probs, min=eps))
        logits = logits / max(float(tau), eps)
        if num_base_cls < logits.shape[1] and abs(float(delta)) > 0:
            logits = logits.clone()
            logits[:, num_base_cls:] = logits[:, num_base_cls:] + float(delta)
        return F.softmax(logits, dim=-1)

    def _score_probabilities(self, probs, targets):
        sample_idx = torch.arange(targets.shape[0], device=targets.device)
        target_probs = probs[sample_idx, targets]
        nll = -torch.log(torch.clamp(target_probs, min=self.adapt_cfg.EPS)).mean()
        acc = (torch.argmax(probs, dim=1) == targets).float().mean()
        return acc, nll

    def _oracle_is_better(self, acc, nll, best_acc, best_nll, metric='acc'):
        metric = str(metric).lower()
        if best_acc is None or best_nll is None:
            return True
        tol = 1e-8
        if metric == 'nll':
            if nll.item() < best_nll.item() - tol:
                return True
            if abs(nll.item() - best_nll.item()) <= tol and acc.item() > best_acc.item() + tol:
                return True
            return False
        if acc.item() > best_acc.item() + tol:
            return True
        if abs(acc.item() - best_acc.item()) <= tol and nll.item() < best_nll.item() - tol:
            return True
        return False

    @torch.no_grad()
    def oracle_calibrate_session(
        self,
        img_feat,
        targets,
        num_cls,
        num_base_cls,
        image_proto,
        cov_image,
        description_proto,
        description_features,
        description_targets,
        text_features,
        beta,
        lambda_t_per_class=None,
    ):
        oracle_cfg = self.cfg.TRAINER.BiMCOracle
        if not bool(oracle_cfg.ENABLE):
            return None

        mode = str(oracle_cfg.MODE).lower()
        if mode not in {'beta', 'delta', 'tau', 'joint'}:
            return None

        device = img_feat.device
        metric = str(oracle_cfg.SEARCH_METRIC).lower()
        base_beta = float(beta)
        base_delta = float(oracle_cfg.DEFAULT_DELTA)
        base_tau = float(oracle_cfg.DEFAULT_TAU)

        if lambda_t_per_class is None:
            lambda_t = image_proto.new_full((num_cls,), self.cfg.TRAINER.BiMC.LAMBDA_T)
        else:
            lambda_t = lambda_t_per_class[:num_cls].to(image_proto.device)

        def compose(beta_value):
            beta_tensor = image_proto.new_full((num_cls,), float(beta_value))
            return self._compose_probabilities_from_features(
                img_feat,
                num_cls,
                num_base_cls,
                image_proto,
                cov_image,
                description_proto,
                description_features,
                description_targets,
                text_features,
                beta_tensor,
                lambda_t,
            )

        best = {
            'beta': base_beta,
            'delta': base_delta,
            'tau': base_tau,
            'probs': None,
            'acc': None,
            'nll': None,
        }

        def update_best(raw_probs, beta_value, delta_value, tau_value):
            calibrated = self._posthoc_calibrate_probabilities(
                raw_probs,
                num_base_cls=num_base_cls,
                tau=tau_value,
                delta=delta_value,
            )
            acc, nll = self._score_probabilities(calibrated, targets)
            if self._oracle_is_better(acc, nll, best['acc'], best['nll'], metric=metric):
                best['beta'] = float(beta_value)
                best['delta'] = float(delta_value)
                best['tau'] = float(tau_value)
                best['probs'] = calibrated
                best['acc'] = acc.detach()
                best['nll'] = nll.detach()

        if mode == 'beta':
            for beta_value in self._beta_grid(oracle_cfg.BETA_GRID_SIZE, device):
                update_best(compose(beta_value), beta_value, base_delta, base_tau)
        elif mode == 'delta':
            raw_probs = compose(base_beta)
            for delta_value in self._delta_grid(device):
                update_best(raw_probs, base_beta, delta_value, base_tau)
        elif mode == 'tau':
            raw_probs = compose(base_beta)
            for tau_value in self._tau_grid(device):
                update_best(raw_probs, base_beta, base_delta, tau_value)
        else:
            tau_grid = self._tau_grid(device)
            delta_grid = self._delta_grid(device)
            for beta_value in self._beta_grid(oracle_cfg.BETA_GRID_SIZE, device):
                raw_probs = compose(beta_value)
                log_probs = torch.log(torch.clamp(raw_probs, min=self.adapt_cfg.EPS))
                for tau_value in tau_grid:
                    scaled_logits = log_probs / max(float(tau_value), self.adapt_cfg.EPS)
                    if num_base_cls < num_cls:
                        for delta_value in delta_grid:
                            calibrated_logits = scaled_logits.clone()
                            calibrated_logits[:, num_base_cls:] = calibrated_logits[:, num_base_cls:] + float(delta_value)
                            calibrated = F.softmax(calibrated_logits, dim=-1)
                            acc, nll = self._score_probabilities(calibrated, targets)
                            if self._oracle_is_better(acc, nll, best['acc'], best['nll'], metric=metric):
                                best['beta'] = float(beta_value)
                                best['delta'] = float(delta_value)
                                best['tau'] = float(tau_value)
                                best['probs'] = calibrated
                                best['acc'] = acc.detach()
                                best['nll'] = nll.detach()
                    else:
                        calibrated = F.softmax(scaled_logits, dim=-1)
                        acc, nll = self._score_probabilities(calibrated, targets)
                        if self._oracle_is_better(acc, nll, best['acc'], best['nll'], metric=metric):
                            best['beta'] = float(beta_value)
                            best['delta'] = 0.0
                            best['tau'] = float(tau_value)
                            best['probs'] = calibrated
                            best['acc'] = acc.detach()
                            best['nll'] = nll.detach()

        if best['probs'] is None:
            default_probs = compose(base_beta)
            best['probs'] = self._posthoc_calibrate_probabilities(
                default_probs,
                num_base_cls=num_base_cls,
                tau=base_tau,
                delta=base_delta,
            )
            best['acc'], best['nll'] = self._score_probabilities(best['probs'], targets)

        return best

    def _knn_similarity_scores(self, queries, support_features, support_labels, num_cls):
        device = queries.device
        support_features = support_features.to(device)
        support_labels = support_labels.to(device)
        similarity_scores = torch.matmul(queries, support_features.T)
        max_scores = torch.full((queries.size(0), num_cls), float('-inf'), device=device)
        expanded_labels = support_labels.unsqueeze(0).expand(queries.size(0), -1)
        for label in range(num_cls):
            label_mask = expanded_labels == label
            masked_scores = similarity_scores.masked_fill(~label_mask, float('-inf'))
            max_scores[:, label] = torch.max(masked_scores, dim=1).values
        return max_scores

    def _mahalanobis(self, dist, cov_inv):
        left_term = torch.matmul(dist, cov_inv)
        mahal = torch.matmul(left_term, dist.T)
        return torch.diag(mahal)

    def _cov_forward(self, feat, proto, cov, num_cls):
        maha_dist = []
        inv_covmat = torch.pinverse(cov.to(dtype=torch.float32))
        inv_covmat = inv_covmat.to(dtype=proto.dtype)
        for cl in range(num_cls):
            distance = feat - proto[cl]
            dist = self._mahalanobis(distance, inv_covmat)
            maha_dist.append(dist)
        maha_dist = torch.stack(maha_dist)
        logits = -maha_dist.T
        return logits

    def _compose_probabilities_from_features(
        self,
        img_feat,
        num_cls,
        num_base_cls,
        image_proto,
        cov_image,
        description_proto,
        description_features,
        description_targets,
        text_features,
        beta_tensor,
        lambda_t,
        use_diversity=None,
    ):
        fused_text = (1 - lambda_t.unsqueeze(1)) * text_features[:num_cls] + lambda_t.unsqueeze(1) * description_proto[:num_cls]
        fused_text = F.normalize(fused_text, dim=-1)
        fused_proto = beta_tensor.unsqueeze(1) * fused_text + (1 - beta_tensor.unsqueeze(1)) * image_proto[:num_cls]
        fused_proto = F.normalize(fused_proto, dim=-1)

        logits_proto_fused = img_feat @ fused_proto.t()
        prob_fused_proto = F.softmax(logits_proto_fused, dim=-1)

        if use_diversity is None:
            use_diversity = self.cfg.TRAINER.BiMC.USING_ENSEMBLE
        if not use_diversity:
            return prob_fused_proto

        logits_cov = self._cov_forward(img_feat, image_proto[:num_cls], cov_image, num_cls)
        logits_knn = self._knn_similarity_scores(img_feat, description_features, description_targets, num_cls)
        prob_cov = F.softmax(logits_cov / 512, dim=-1)
        prob_knn = F.softmax(logits_knn, dim=-1)

        ensemble_alpha = self.cfg.DATASET.ENSEMBLE_ALPHA
        base_probs = ensemble_alpha * prob_fused_proto[:, :num_base_cls] + (1 - ensemble_alpha) * prob_cov[:, :num_base_cls]
        inc_probs = ensemble_alpha * prob_fused_proto[:, num_base_cls:] + (1 - ensemble_alpha) * prob_knn[:, num_base_cls:]
        return torch.cat([base_probs, inc_probs], dim=1)

    def _class_mask(self, targets, class_index):
        mask = targets == int(class_index[0])
        for cls_id in class_index[1:]:
            mask = mask | (targets == int(cls_id))
        return mask

    def _membership_mask(self, targets, values):
        values = list(values) if values is not None else []
        if len(values) == 0:
            return targets.new_zeros(targets.shape, dtype=torch.bool)
        mask = targets == int(values[0])
        for value in values[1:]:
            mask = mask | (targets == int(value))
        return mask

    def _select_support_features(self, images_features, images_targets, class_index, max_per_class):
        selected_features = []
        selected_targets = []
        for cls_id in class_index:
            idx = torch.where(images_targets == int(cls_id))[0]
            if idx.numel() == 0:
                continue
            if max_per_class > 0 and idx.numel() > max_per_class:
                idx = idx[:max_per_class]
            selected_features.append(images_features[idx])
            selected_targets.append(images_targets[idx])
        if not selected_features:
            return None, None
        return torch.cat(selected_features, dim=0), torch.cat(selected_targets, dim=0)

    def _beta_objective(
        self,
        beta_value,
        prior_beta,
        support_features,
        support_targets,
        num_cls,
        num_base_cls,
        image_proto,
        cov_image,
        description_proto,
        description_features,
        description_targets,
        text_features,
        lambda_t_per_class,
        use_diversity,
        reg_lambda,
        objective_mode='nll',
        novel_class_index=None,
        base_class_index=None,
    ):
        beta_tensor = image_proto.new_full((num_cls,), float(beta_value))
        if lambda_t_per_class is None:
            lambda_t = image_proto.new_full((num_cls,), self.cfg.TRAINER.BiMC.LAMBDA_T)
        else:
            lambda_t = lambda_t_per_class[:num_cls].to(image_proto.device)

        probs = self._compose_probabilities_from_features(
            support_features,
            num_cls,
            num_base_cls,
            image_proto,
            cov_image,
            description_proto,
            description_features,
            description_targets,
            text_features,
            beta_tensor,
            lambda_t,
            use_diversity=use_diversity,
        )
        sample_idx = torch.arange(support_targets.shape[0], device=support_targets.device)
        nll = -torch.log(torch.clamp(probs[sample_idx, support_targets], min=self.adapt_cfg.EPS)).mean()
        margin_loss = support_features.new_tensor(0.0)

        objective_mode = str(objective_mode).lower()
        if objective_mode in {'margin', 'nll_margin'} and base_class_index is not None and novel_class_index is not None:
            novel_mask = self._membership_mask(support_targets, novel_class_index)
            base_class_index = list(base_class_index)
            if novel_mask.any() and len(base_class_index) > 0:
                novel_probs = probs[novel_mask]
                novel_targets = support_targets[novel_mask]
                novel_sample_idx = torch.arange(novel_targets.shape[0], device=novel_targets.device)
                target_scores = novel_probs[novel_sample_idx, novel_targets]
                base_index_tensor = torch.tensor(base_class_index, dtype=torch.long, device=novel_probs.device)
                base_scores = novel_probs.index_select(1, base_index_tensor).max(dim=1).values
                margin = target_scores - base_scores
                margin_loss = torch.relu(
                    margin.new_tensor(float(self.adapt_cfg.SESSION_RISK_MARGIN_TARGET)) - margin
                ).mean()

        beta_scalar = image_proto.new_tensor(float(beta_value))
        prior_scalar = image_proto.new_tensor(float(prior_beta))
        reg = image_proto.new_tensor(float(reg_lambda)) * (self._logit(beta_scalar) - self._logit(prior_scalar)).pow(2)
        if objective_mode == 'margin':
            obj = reg + margin_loss * float(self.adapt_cfg.SESSION_RISK_MARGIN_WEIGHT)
        elif objective_mode == 'nll_margin':
            obj = nll + reg + margin_loss * float(self.adapt_cfg.SESSION_RISK_MARGIN_WEIGHT)
        else:
            obj = nll + reg
        return obj, nll, margin_loss

    def _estimate_base_prior_beta(
        self,
        class_index,
        text_features,
        description_proto,
        description_features,
        description_targets,
        images_features,
        images_targets,
        prior_center_beta,
    ):
        prior_mode = str(self.adapt_cfg.PRIOR_MODE).lower()
        if prior_mode not in {'base_auto', 'base_pseudo'}:
            return None

        objective_mode = str(self.adapt_cfg.SESSION_RISK_OBJECTIVE).lower()

        if prior_mode == 'base_auto':
            calib_fraction = float(self.adapt_cfg.AUTO_PRIOR_CALIB_FRACTION)
            max_calib = int(self.adapt_cfg.AUTO_PRIOR_MAX_CALIB_PER_CLASS)
            min_proto = int(self.adapt_cfg.AUTO_PRIOR_MIN_PROTO_PER_CLASS)

            proto_list = []
            calib_features = []
            calib_targets = []
            for cls_id in class_index:
                idx = torch.where(images_targets == int(cls_id))[0]
                if idx.numel() == 0:
                    continue

                calib_count = int(round(idx.numel() * calib_fraction))
                calib_count = max(1, calib_count)
                calib_count = min(calib_count, max_calib)
                if idx.numel() - calib_count < min_proto:
                    calib_count = max(1, idx.numel() - min_proto)
                if idx.numel() > 1:
                    calib_count = min(calib_count, idx.numel() - 1)
                else:
                    calib_count = 0

                if calib_count > 0:
                    calib_idx = idx[:calib_count]
                    proto_idx = idx[calib_count:]
                else:
                    calib_idx = idx[:0]
                    proto_idx = idx

                if proto_idx.numel() == 0:
                    proto_idx = idx
                proto = images_features[proto_idx].mean(dim=0)
                proto_list.append(F.normalize(proto, dim=-1))

                if calib_idx.numel() > 0:
                    calib_features.append(images_features[calib_idx])
                    calib_targets.append(images_targets[calib_idx])

            if not proto_list or not calib_features:
                return None

            base_image_proto = torch.stack(proto_list, dim=0)
            calib_features = torch.cat(calib_features, dim=0)
            calib_targets = torch.cat(calib_targets, dim=0)

            beta_grid = self._beta_grid(self.adapt_cfg.AUTO_PRIOR_GRID_SIZE, base_image_proto.device)
            best_beta = base_image_proto.new_tensor(float(prior_center_beta))
            best_obj = None
            lambda_t = base_image_proto.new_full((len(class_index),), self.cfg.TRAINER.BiMC.LAMBDA_T)
            base_labels = list(range(len(class_index)))

            for beta_value in beta_grid:
                obj, _, _ = self._beta_objective(
                    beta_value=beta_value,
                    prior_beta=prior_center_beta,
                    support_features=calib_features,
                    support_targets=calib_targets,
                    num_cls=len(class_index),
                    num_base_cls=len(class_index),
                    image_proto=base_image_proto,
                    cov_image=None,
                    description_proto=description_proto,
                    description_features=description_features,
                    description_targets=description_targets,
                    text_features=text_features,
                    lambda_t_per_class=lambda_t,
                    use_diversity=False,
                    reg_lambda=float(self.adapt_cfg.AUTO_PRIOR_REG_LAMBDA),
                    objective_mode=objective_mode,
                    novel_class_index=[],
                    base_class_index=base_labels,
                )
                if best_obj is None or obj.item() < best_obj.item():
                    best_obj = obj
                    best_beta = beta_value

            return float(best_beta)

        image_groups = self._group_class_features(images_features, images_targets, class_index)
        desc_groups = self._group_class_features(description_features, description_targets, class_index)
        num_classes = len(class_index)
        if num_classes <= 1:
            return None

        beta_grid = self._beta_grid(self.adapt_cfg.AUTO_PRIOR_GRID_SIZE, images_features.device)
        objective_sums = images_features.new_zeros(beta_grid.shape[0])
        valid_episodes = 0

        num_episodes = max(1, int(self.adapt_cfg.AUTO_PRIOR_PSEUDO_EPISODES))
        pseudo_novel_count = int(self.adapt_cfg.AUTO_PRIOR_PSEUDO_NOVEL_COUNT)
        if pseudo_novel_count <= 0:
            pseudo_novel_count = max(1, min(int(self.cfg.DATASET.NUM_INC_CLS), num_classes - 1))
        pseudo_novel_count = min(max(1, pseudo_novel_count), num_classes - 1)
        pseudo_shot = int(self.adapt_cfg.AUTO_PRIOR_PSEUDO_SHOT)
        if pseudo_shot <= 0:
            pseudo_shot = max(1, int(self.cfg.DATASET.NUM_INC_SHOT))
        pseudo_calib = max(1, int(self.adapt_cfg.AUTO_PRIOR_PSEUDO_CALIB_PER_CLASS))
        min_base_proto = max(1, int(self.adapt_cfg.AUTO_PRIOR_PSEUDO_MIN_BASE_PROTO))
        use_diversity = bool(self.adapt_cfg.AUTO_PRIOR_PSEUDO_USE_ENSEMBLE and self.cfg.TRAINER.BiMC.USING_ENSEMBLE)

        for episode_id in range(num_episodes):
            generator = torch.Generator()
            generator.manual_seed(int(self.cfg.SEED) + 7919 * (episode_id + 1))
            perm = torch.randperm(num_classes, generator=generator).tolist()
            novel_positions = perm[:pseudo_novel_count]
            novel_position_set = set(novel_positions)
            base_positions = [idx for idx in range(num_classes) if idx not in novel_position_set]
            episode_positions = base_positions + novel_positions

            local_proto = []
            local_text = []
            local_desc_proto = []
            local_desc_features = []
            local_desc_targets = []
            local_calib_features = []
            local_calib_targets = []
            local_cov_source = []
            episode_valid = True

            for local_label, class_pos in enumerate(episode_positions):
                class_images = image_groups[class_pos]
                class_desc = desc_groups[class_pos]
                if class_images.shape[0] <= 1:
                    episode_valid = False
                    break

                class_perm = torch.randperm(class_images.shape[0], generator=generator).tolist()
                class_images = class_images[class_perm]

                is_novel = class_pos in novel_position_set
                if is_novel:
                    if class_images.shape[0] <= pseudo_shot:
                        episode_valid = False
                        break
                    support_count = min(pseudo_shot, class_images.shape[0] - 1)
                    calib_count = min(pseudo_calib, class_images.shape[0] - support_count)
                    if calib_count <= 0:
                        episode_valid = False
                        break
                    proto_feats = class_images[:support_count]
                    calib_feats = class_images[support_count:support_count + calib_count]
                else:
                    max_calib = max(1, class_images.shape[0] - min_base_proto)
                    calib_count = min(pseudo_calib, max_calib)
                    if class_images.shape[0] - calib_count <= 0:
                        episode_valid = False
                        break
                    calib_feats = class_images[:calib_count]
                    proto_feats = class_images[calib_count:]

                if proto_feats.numel() == 0 or calib_feats.numel() == 0:
                    episode_valid = False
                    break

                local_proto.append(F.normalize(proto_feats.mean(dim=0), dim=-1))
                local_text.append(text_features[class_pos])
                local_desc_proto.append(description_proto[class_pos])
                local_desc_features.append(class_desc)
                local_desc_targets.append(torch.full((class_desc.shape[0],), local_label, dtype=torch.long, device=class_desc.device))
                local_calib_features.append(calib_feats)
                local_calib_targets.append(torch.full((calib_feats.shape[0],), local_label, dtype=torch.long, device=calib_feats.device))
                local_cov_source.append(proto_feats)

            if not episode_valid:
                continue

            local_image_proto = torch.stack(local_proto, dim=0)
            local_text_features = torch.stack(local_text, dim=0)
            local_description_proto = torch.stack(local_desc_proto, dim=0)
            local_description_features = torch.cat(local_desc_features, dim=0)
            local_description_targets = torch.cat(local_desc_targets, dim=0)
            local_calib_features = torch.cat(local_calib_features, dim=0)
            local_calib_targets = torch.cat(local_calib_targets, dim=0)
            local_cov_image = None
            if use_diversity:
                cov_source = torch.cat(local_cov_source, dim=0)
                local_cov_image = torch.cov(cov_source.T)

            local_num_base = len(base_positions)
            local_num_cls = len(episode_positions)
            local_base_labels = list(range(local_num_base))
            local_novel_labels = list(range(local_num_base, local_num_cls))
            local_lambda_t = local_image_proto.new_full((local_num_cls,), self.cfg.TRAINER.BiMC.LAMBDA_T)

            for beta_idx, beta_value in enumerate(beta_grid):
                obj, _, _ = self._beta_objective(
                    beta_value=beta_value,
                    prior_beta=prior_center_beta,
                    support_features=local_calib_features,
                    support_targets=local_calib_targets,
                    num_cls=local_num_cls,
                    num_base_cls=local_num_base,
                    image_proto=local_image_proto,
                    cov_image=local_cov_image,
                    description_proto=local_description_proto,
                    description_features=local_description_features,
                    description_targets=local_description_targets,
                    text_features=local_text_features,
                    lambda_t_per_class=local_lambda_t,
                    use_diversity=use_diversity,
                    reg_lambda=float(self.adapt_cfg.AUTO_PRIOR_REG_LAMBDA),
                    objective_mode=objective_mode,
                    novel_class_index=local_novel_labels,
                    base_class_index=local_base_labels,
                )
                objective_sums[beta_idx] = objective_sums[beta_idx] + obj.detach()

            valid_episodes += 1

        if valid_episodes == 0:
            return None

        best_beta = beta_grid[torch.argmin(objective_sums)]
        return float(best_beta)

    def _compute_session_risk_beta(
        self,
        num_cls,
        num_base_cls,
        prior_beta,
        image_proto,
        cov_image,
        description_proto,
        description_features,
        description_targets,
        text_features,
        images_features,
        images_targets,
        lambda_t_per_class,
        current_task_class_index,
    ):
        if images_features is None or images_targets is None or current_task_class_index is None:
            return image_proto.new_tensor(float(prior_beta))

        if self.adapt_cfg.SESSION_RISK_USE_CURRENT_TASK_ONLY:
            support_class_index = list(current_task_class_index)
        else:
            support_class_index = list(range(num_cls))

        support_features, support_targets = self._select_support_features(
            images_features[:images_targets.shape[0]],
            images_targets[:images_targets.shape[0]],
            support_class_index,
            int(self.adapt_cfg.SESSION_RISK_MAX_SUPPORT_PER_CLASS),
        )
        if support_features is None or support_features.numel() == 0:
            return image_proto.new_tensor(float(prior_beta))

        beta_grid = self._beta_grid(self.adapt_cfg.SESSION_RISK_GRID_SIZE, image_proto.device)
        best_beta = image_proto.new_tensor(float(prior_beta))
        best_obj = None
        use_diversity = bool(self.cfg.TRAINER.BiMC.USING_ENSEMBLE and self.adapt_cfg.SESSION_RISK_INCLUDE_ENSEMBLE)
        reg_lambda = float(self.adapt_cfg.SESSION_RISK_REG_LAMBDA)
        objective_mode = str(self.adapt_cfg.SESSION_RISK_OBJECTIVE).lower()
        base_class_index = list(range(num_base_cls)) if num_base_cls is not None else []

        for beta_value in beta_grid:
            obj, _, _ = self._beta_objective(
                beta_value,
                prior_beta,
                support_features,
                support_targets,
                num_cls,
                num_base_cls,
                image_proto,
                cov_image,
                description_proto,
                description_features,
                description_targets,
                text_features,
                lambda_t_per_class,
                use_diversity,
                reg_lambda,
                objective_mode=objective_mode,
                novel_class_index=list(current_task_class_index),
                base_class_index=base_class_index,
            )
            if best_obj is None or obj.item() < best_obj.item():
                best_obj = obj
                best_beta = beta_value

        return torch.clamp(best_beta, min=self.adapt_cfg.MIN_WEIGHT, max=self.adapt_cfg.MAX_WEIGHT)

    def compute_session_joint_beta(
        self,
        num_cls,
        prior_beta,
        beta_per_class=None,
        text_reliability=None,
        visual_reliability=None,
        num_base_cls=None,
        image_proto=None,
        cov_image=None,
        description_proto=None,
        description_features=None,
        description_targets=None,
        text_features=None,
        images_features=None,
        images_targets=None,
        lambda_t_per_class=None,
        current_task_class_index=None,
    ):
        mode = str(self.adapt_cfg.BETA_MODE).lower()
        if mode not in {'session_joint', 'session_risk'}:
            return None

        prior_beta = self.resolve_beta_prior(prior_beta)

        if mode == 'session_risk':
            if image_proto is None or description_proto is None or text_features is None:
                return torch.tensor(prior_beta, device=self.device)
            return self._compute_session_risk_beta(
                num_cls=num_cls,
                num_base_cls=num_base_cls,
                prior_beta=prior_beta,
                image_proto=image_proto,
                cov_image=cov_image,
                description_proto=description_proto,
                description_features=description_features,
                description_targets=description_targets,
                text_features=text_features,
                images_features=images_features,
                images_targets=images_targets,
                lambda_t_per_class=lambda_t_per_class,
                current_task_class_index=current_task_class_index,
            )

        eps = self.adapt_cfg.EPS
        prior = float(prior_beta)
        prior_tensor = torch.tensor(prior, device=self.device)
        prior_logit = self._logit(prior_tensor)

        if text_reliability is not None and visual_reliability is not None:
            text_rel = text_reliability[:num_cls].to(self.device)
            visual_rel = visual_reliability[:num_cls].to(self.device)
            evidence_delta = (torch.log(text_rel + eps) - torch.log(visual_rel + eps)).mean()
            joint_beta = torch.sigmoid(prior_logit + self.adapt_cfg.RELIABILITY_LOGIT_SCALE * evidence_delta)
        elif beta_per_class is not None:
            beta_vals = beta_per_class[:num_cls].to(self.device)
            beta_vals = torch.clamp(beta_vals, min=eps, max=1.0 - eps)
            joint_beta = torch.sigmoid((self._logit(beta_vals)).mean())
        else:
            return prior_tensor

        rho = float(num_cls) / (float(num_cls) + self.adapt_cfg.BETA_SHRINKAGE_NU)
        joint_beta = rho * joint_beta + (1.0 - rho) * prior_tensor
        joint_beta = torch.clamp(joint_beta, min=self.adapt_cfg.MIN_WEIGHT, max=self.adapt_cfg.MAX_WEIGHT)
        return joint_beta

    def _shrink_weights(self, raw_weights, evidence, fallback, nu, min_weight=None, max_weight=None):
        session_prior = raw_weights.mean()
        if fallback is not None:
            session_prior = 0.5 * (session_prior + raw_weights.new_tensor(float(fallback)))
        rho = evidence / (evidence + nu)
        shrunk = rho * raw_weights + (1 - rho) * session_prior
        if min_weight is None:
            min_weight = self.adapt_cfg.MIN_WEIGHT
        if max_weight is None:
            max_weight = self.adapt_cfg.MAX_WEIGHT
        return torch.clamp(shrunk, min=min_weight, max=max_weight)

    def _compute_adaptive_statistics(
        self,
        class_index,
        text_features,
        text_targets,
        description_proto,
        description_features,
        description_targets,
        image_proto,
        images_features,
        images_targets,
    ):
        prompt_groups = self._group_class_features(text_features, text_targets, class_index)
        desc_groups = self._group_class_features(description_features, description_targets, class_index)
        image_groups = self._group_class_features(images_features, images_targets, class_index)

        prompt_rel = []
        desc_rel = []
        text_rel = []
        visual_rel = []
        image_counts = []
        prompt_counts = []
        description_counts = []
        raw_lambda_t = []
        raw_beta = []

        for idx in range(len(class_index)):
            prompt_group = prompt_groups[idx]
            desc_group = desc_groups[idx]
            image_group = image_groups[idx]

            prompt_count = prompt_group.shape[0]
            desc_count = desc_group.shape[0]
            image_count = image_group.shape[0]

            prompt_counts.append(float(prompt_count))
            description_counts.append(float(desc_count))
            image_counts.append(float(image_count))

            prompt_score = self._estimate_reliability(prompt_group)
            desc_score = self._estimate_reliability(desc_group)
            desc_score = desc_score * self._agreement_gate(description_proto[idx], image_proto[idx])
            visual_score = self._estimate_reliability(image_group)
            text_score = prompt_score + desc_score

            prompt_rel.append(prompt_score)
            desc_rel.append(desc_score)
            text_rel.append(text_score)
            visual_rel.append(visual_score)

            raw_lambda_t.append(self._prior_logit_weight(self.cfg.TRAINER.BiMC.LAMBDA_T, desc_score, prompt_score))
            raw_beta.append(self._prior_logit_weight(self.cfg.DATASET.BETA, text_score, visual_score))

        prompt_rel = torch.stack(prompt_rel)
        desc_rel = torch.stack(desc_rel)
        text_rel = torch.stack(text_rel)
        visual_rel = torch.stack(visual_rel)
        image_counts = image_proto.new_tensor(image_counts)
        prompt_counts = image_proto.new_tensor(prompt_counts)
        description_counts = image_proto.new_tensor(description_counts)
        raw_lambda_t = torch.stack(raw_lambda_t)
        raw_beta = torch.stack(raw_beta)

        if self.adapt_cfg.ENABLE_ADAPTIVE_TEXT:
            lambda_t = self._shrink_weights(
                raw_lambda_t,
                prompt_counts + description_counts,
                self.cfg.TRAINER.BiMC.LAMBDA_T,
                self.adapt_cfg.TEXT_SHRINKAGE_NU,
            )
        else:
            lambda_t = image_proto.new_full(raw_lambda_t.shape, self.cfg.TRAINER.BiMC.LAMBDA_T)

        if self.adapt_cfg.ENABLE_ADAPTIVE_BETA:
            beta = self._shrink_weights(
                raw_beta,
                image_counts,
                self.cfg.DATASET.BETA,
                self.adapt_cfg.BETA_SHRINKAGE_NU,
            )
        else:
            beta = image_proto.new_full(raw_beta.shape, self.cfg.DATASET.BETA)

        return {
            'prompt_reliability': prompt_rel,
            'description_reliability': desc_rel,
            'text_reliability': text_rel,
            'visual_reliability': visual_rel,
            'beta_per_class': beta,
            'lambda_t_per_class': lambda_t,
            'image_counts': image_counts,
            'prompt_counts': prompt_counts,
            'description_counts': description_counts,
        }

    def _adaptive_visual_calibration(self, cur_protos, cur_reliability, image_counts):
        if self.base_visual_reliability is None:
            zero = cur_protos.new_zeros(cur_protos.shape[0])
            return cur_protos, zero, zero

        base_protos = F.normalize(self.base_vision_prototype, dim=-1)
        cur_protos = F.normalize(cur_protos, dim=-1)
        weights = torch.mm(cur_protos, base_protos.T) * self.cfg.TRAINER.BiMC.TAU
        norm_weights = torch.softmax(weights, dim=1)
        delta_protos = torch.matmul(norm_weights, base_protos)
        delta_protos = F.normalize(delta_protos, dim=-1)

        anchor_reliability = torch.matmul(norm_weights, self.base_visual_reliability.unsqueeze(1)).squeeze(1)
        raw_lambda_i = self._prior_logit_weight(self.cfg.TRAINER.BiMC.LAMBDA_I, anchor_reliability, cur_reliability)

        if self.adapt_cfg.ENABLE_ADAPTIVE_VISION:
            lambda_i = self._shrink_weights(
                raw_lambda_i,
                image_counts,
                self.cfg.TRAINER.BiMC.LAMBDA_I,
                self.adapt_cfg.VISION_SHRINKAGE_NU,
                min_weight=self.adapt_cfg.MIN_VISION_WEIGHT,
                max_weight=self.adapt_cfg.MAX_VISION_WEIGHT,
            )
        else:
            lambda_i = cur_protos.new_full(raw_lambda_i.shape, self.cfg.TRAINER.BiMC.LAMBDA_I)

        updated_protos = (1 - lambda_i.unsqueeze(1)) * cur_protos + lambda_i.unsqueeze(1) * delta_protos
        updated_protos = F.normalize(updated_protos, dim=-1)
        return updated_protos, lambda_i, anchor_reliability

    def build_task_statistics(self, class_names, loader, class_index, calibrate_novel_vision_proto=False):
        def shrink_cov(cov, alpha1=1.0, alpha2=0.0):
            diag_mean = torch.mean(torch.diagonal(cov))
            off_diag = cov.clone()
            off_diag.fill_diagonal_(0.0)
            mask = off_diag != 0.0
            off_diag_mean = (off_diag * mask).sum() / mask.sum()
            iden = torch.eye(cov.shape[0]).to(cov.device)
            cov_ = cov + (alpha1 * diag_mean * iden) + (alpha2 * off_diag_mean * (1 - iden))
            return cov_

        cls_begin_index = class_index[0]

        text_features, _ = super().inference_text_feature(class_names, self.template, cls_begin_index)
        prompt_bank_features, text_targets = self.inference_prompt_bank(class_names, cls_begin_index)
        description_features, description_targets, description_proto = self.inference_all_description_feature(
            class_names=class_names,
            gpt_path=self.cfg.DATASET.GPT_PATH,
            cls_begin_index=cls_begin_index,
        )
        images_features, images_targets, images_proto = self.inference_all_img_feature(loader, cls_begin_index)

        adaptive_stats = self._compute_adaptive_statistics(
            class_index,
            prompt_bank_features,
            text_targets,
            description_proto,
            description_features,
            description_targets,
            images_proto,
            images_features,
            images_targets,
        )

        if cls_begin_index == 0:
            prior_center_beta = float(self.adapt_cfg.UNIVERSAL_BETA)
            estimated_prior = self._estimate_base_prior_beta(
                class_index=class_index,
                text_features=text_features,
                description_proto=description_proto,
                description_features=description_features,
                description_targets=description_targets,
                images_features=images_features,
                images_targets=images_targets,
                prior_center_beta=prior_center_beta,
            )
            if estimated_prior is not None:
                self.estimated_beta_prior = estimated_prior
                print(f'estimated base beta prior: {self.estimated_beta_prior:.4f}')

            adaptive_stats['beta_per_class'] = images_proto.new_full(
                adaptive_stats['beta_per_class'].shape, self.cfg.DATASET.BETA
            )
            adaptive_stats['lambda_t_per_class'] = images_proto.new_full(
                adaptive_stats['lambda_t_per_class'].shape, self.cfg.TRAINER.BiMC.LAMBDA_T
            )

        if cls_begin_index != 0 and calibrate_novel_vision_proto:
            print(f'adaptive vision calibration on class [{class_index}]')
            images_proto, lambda_i, anchor_reliability = self._adaptive_visual_calibration(
                images_proto,
                adaptive_stats['visual_reliability'],
                adaptive_stats['image_counts'],
            )
            adaptive_stats['lambda_i_per_class'] = lambda_i
            adaptive_stats['anchor_reliability'] = anchor_reliability
        else:
            adaptive_stats['lambda_i_per_class'] = images_proto.new_zeros(images_proto.shape[0])
            adaptive_stats['anchor_reliability'] = images_proto.new_zeros(images_proto.shape[0])

        if cls_begin_index == 0:
            self.base_vision_prototype = images_proto
            self.base_visual_reliability = adaptive_stats['visual_reliability']

        cov_images = torch.cov(images_features.T)
        if cls_begin_index == 0:
            cov_images = shrink_cov(cov_images, alpha1=self.cfg.TRAINER.BiMC.GAMMA_BASE)
        else:
            cov_images = shrink_cov(cov_images, alpha1=self.cfg.TRAINER.BiMC.GAMMA_INC)

        print('finish loading covariance')
        return {
            'description_proto': description_proto,
            'description_features': description_features,
            'description_targets': description_targets,
            'text_features': text_features,
            'text_targets': text_targets,
            'image_proto': images_proto,
            'images_features': images_features,
            'images_targets': images_targets,
            'cov_image': cov_images,
            'class_index': class_index,
            'sample_cnt': len(images_features),
            **adaptive_stats,
        }

    def forward_ours(
        self,
        images,
        num_cls,
        num_base_cls,
        image_proto,
        cov_image,
        description_proto,
        description_features,
        description_targets,
        text_features,
        beta,
        beta_per_class=None,
        lambda_t_per_class=None,
        session_beta=None,
        **kwargs,
    ):
        img_feat = self.extract_img_feature(images)
        img_feat = F.normalize(img_feat, dim=-1)

        if lambda_t_per_class is None:
            lambda_t = image_proto.new_full((num_cls,), self.cfg.TRAINER.BiMC.LAMBDA_T)
        else:
            lambda_t = lambda_t_per_class[:num_cls].to(image_proto.device)

        if session_beta is not None:
            beta_tensor = image_proto.new_full((num_cls,), float(session_beta))
        elif beta_per_class is None:
            beta_tensor = image_proto.new_full((num_cls,), beta)
        else:
            mode = str(self.adapt_cfg.BETA_MODE).lower()
            if mode in {'session_joint', 'session_risk'}:
                joint_beta = self.compute_session_joint_beta(
                    num_cls,
                    beta,
                    beta_per_class=beta_per_class,
                    text_reliability=kwargs.get('text_reliability'),
                    visual_reliability=kwargs.get('visual_reliability'),
                    num_base_cls=num_base_cls,
                    image_proto=image_proto,
                    cov_image=cov_image,
                    description_proto=description_proto,
                    description_features=description_features,
                    description_targets=description_targets,
                    text_features=text_features,
                    images_features=kwargs.get('images_features'),
                    images_targets=kwargs.get('images_targets'),
                    lambda_t_per_class=lambda_t_per_class,
                    current_task_class_index=kwargs.get('current_task_class_index'),
                )
                beta_tensor = image_proto.new_full((num_cls,), float(joint_beta))
            else:
                beta_tensor = beta_per_class[:num_cls].to(image_proto.device)

        return self._compose_probabilities_from_features(
            img_feat,
            num_cls,
            num_base_cls,
            image_proto,
            cov_image,
            description_proto,
            description_features,
            description_targets,
            text_features,
            beta_tensor,
            lambda_t,
        )
