import os

import torch
import torch.nn.functional as F

from models.bimc import BiMC


def _ensure_scaled_dot_product_attention():
    if hasattr(F, "scaled_dot_product_attention"):
        return

    def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        scale_factor = (query.size(-1) ** -0.5) if scale is None else scale
        attn = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        if is_causal:
            q_len, k_len = query.size(-2), key.size(-2)
            causal = torch.ones((q_len, k_len), dtype=torch.bool, device=query.device).tril()
            attn = attn.masked_fill(~causal, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn = attn.masked_fill(~attn_mask, float("-inf"))
            else:
                attn = attn + attn_mask
        attn = torch.softmax(attn, dim=-1)
        if dropout_p and dropout_p > 0.0:
            attn = torch.dropout(attn, dropout_p, train=True)
        return torch.matmul(attn, value)

    F.scaled_dot_product_attention = scaled_dot_product_attention


class BiMCDinoFusion(BiMC):
    """BiMC-compatible DinoV2/CLIP visual score fusion.

    This is an incremental method path: the original BiMC class is untouched.
    It mirrors the notebook ablation that uses naive visual prototypes,
    score-level DinoV2/CLIP visual fusion with omega=0.7, CLIP text
    calibration, and the masked ensemble.
    """

    OMEGA = 0.7
    LOGIT_TEMP = 100.0
    DINO_COV_SCALE = 1.0

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
    DINO_MEAN = (0.485, 0.456, 0.406)
    DINO_STD = (0.229, 0.224, 0.225)

    def __init__(self, cfg, template, device):
        super().__init__(cfg, template, device)
        self.dino_model = self._load_dinov2().to(self.device).eval()
        for param in self.dino_model.parameters():
            param.requires_grad_(False)

        self.register_buffer("clip_mean", self._image_stats(self.CLIP_MEAN), persistent=False)
        self.register_buffer("clip_std", self._image_stats(self.CLIP_STD), persistent=False)
        self.register_buffer("dino_mean", self._image_stats(self.DINO_MEAN), persistent=False)
        self.register_buffer("dino_std", self._image_stats(self.DINO_STD), persistent=False)
        self.base_dino_vision_prototype = None

    def _image_stats(self, values):
        return torch.tensor(values, dtype=torch.float32).view(1, 3, 1, 1)

    def _load_dinov2(self):
        _ensure_scaled_dot_product_attention()
        default_hub = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
        default_weights = os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth")
        hub_dir = os.environ.get("DINOV2_HUB_DIR", default_hub)
        weights_path = os.environ.get("DINOV2_WEIGHTS", default_weights)

        if os.path.isdir(hub_dir):
            print(f"Loading DinoV2 ViT-B/14 from local hub: {hub_dir}")
            model = torch.hub.load(hub_dir, "dinov2_vitb14", source="local", pretrained=False)
        else:
            print("Loading DinoV2 ViT-B/14 from torch.hub cache/repo")
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=False)

        if os.path.isfile(weights_path):
            print(f"Loading DinoV2 weights: {weights_path}")
            state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state, strict=True)
        else:
            raise FileNotFoundError(
                f"DinoV2 weights not found at {weights_path}. Set DINOV2_WEIGHTS to the checkpoint path."
            )
        return model

    def _to_dino_input(self, clip_normalized_images):
        images = clip_normalized_images.float()
        images = images * self.clip_std.to(images.device) + self.clip_mean.to(images.device)
        images = images.clamp(0.0, 1.0)
        return (images - self.dino_mean.to(images.device)) / self.dino_std.to(images.device)

    @torch.no_grad()
    def extract_dino_img_feature(self, images):
        images = images.to(self.device)
        dino_input = self._to_dino_input(images)
        features = self.dino_model(dino_input)
        return F.normalize(features, dim=-1)

    @torch.no_grad()
    def inference_all_dino_img_feature(self, loader, cls_begin_index):
        all_features = []
        all_labels = []
        for batch in loader:
            images, labels = self.parse_batch(batch)
            features = self.extract_dino_img_feature(images)
            all_features.append(features)
            all_labels.append(labels)
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        unique_labels = torch.unique(all_labels)
        print(f'dino all targets:{unique_labels}')
        prototypes = []
        for cls_id in unique_labels:
            class_features = all_features[torch.where(cls_id == all_labels)[0]]
            prototypes.append(class_features.mean(dim=0))
        prototypes = F.normalize(torch.stack(prototypes, dim=0), dim=-1)
        return all_features, all_labels, prototypes

    def _shrink_cov(self, cov, alpha1=1.0, alpha2=0.0):
        diag_mean = torch.mean(torch.diagonal(cov))
        off_diag = cov.clone()
        off_diag.fill_diagonal_(0.0)
        mask = off_diag != 0.0
        off_diag_mean = (off_diag * mask).sum() / mask.sum()
        eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        return cov + (alpha1 * diag_mean * eye) + (alpha2 * off_diag_mean * (1 - eye))

    def build_task_statistics(self, class_names, loader, class_index, calibrate_novel_vision_proto=False):
        cls_begin_index = class_index[0]

        text_features, text_targets = self.inference_text_feature(class_names, self.template, cls_begin_index)
        description_features, description_targets, description_proto = self.inference_all_description_feature(
            class_names=class_names,
            gpt_path=self.cfg.DATASET.GPT_PATH,
            cls_begin_index=cls_begin_index,
        )

        images_features, images_targets, images_proto = self.inference_all_img_feature(loader, cls_begin_index)
        dino_features, dino_targets, dino_proto = self.inference_all_dino_img_feature(loader, cls_begin_index)

        if cls_begin_index != 0 and calibrate_novel_vision_proto:
            print(f'calibrate CLIP/DinoV2 vision proto on class [{class_index}]')
            images_proto = self.soft_calibration(self.base_vision_prototype, images_proto)
            dino_proto = self.soft_calibration(self.base_dino_vision_prototype, dino_proto)
        elif cls_begin_index == 0:
            self.base_vision_prototype = images_proto
            self.base_dino_vision_prototype = dino_proto

        gamma_image = self.cfg.TRAINER.BiMC.GAMMA_BASE if cls_begin_index == 0 else self.cfg.TRAINER.BiMC.GAMMA_INC
        cov_images = self._shrink_cov(torch.cov(images_features.T), alpha1=gamma_image)
        dino_cov = self._shrink_cov(torch.cov(dino_features.T.float()), alpha1=self.cfg.TRAINER.BiMC.GAMMA_BASE)

        print('finish loading CLIP/DinoV2 covariance')
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
            'dino_image_proto': dino_proto,
            'dino_images_features': dino_features,
            'dino_images_targets': dino_targets,
            'dino_cov_image': dino_cov,
            'class_index': class_index,
            'sample_cnt': len(images_features),
        }

    def _mahalanobis_logits(self, features, prototypes, cov):
        inv_cov = torch.pinverse(cov.to(dtype=torch.float32)).to(dtype=prototypes.dtype)
        logits = []
        for cls_id in range(prototypes.shape[0]):
            dist = features - prototypes[cls_id]
            left = torch.matmul(dist, inv_cov)
            logits.append(-torch.sum(left * dist, dim=1))
        return torch.stack(logits, dim=1)

    def _knn_similarity_scores(self, queries, support_features, support_labels, num_cls):
        support_features = support_features.to(queries.device)
        support_labels = support_labels.to(queries.device)
        similarity_scores = torch.matmul(queries, support_features.T)
        max_scores = torch.full(
            (queries.size(0), num_cls),
            float('-inf'),
            device=queries.device,
            dtype=queries.dtype,
        )
        expanded_labels = support_labels.unsqueeze(0).expand(queries.size(0), -1)
        for label in range(num_cls):
            label_mask = expanded_labels == label
            masked_scores = similarity_scores.masked_fill(~label_mask, float('-inf'))
            max_scores[:, label] = torch.max(masked_scores, dim=1).values
        return max_scores

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
        dino_image_proto=None,
        dino_cov_image=None,
        dino_images_features=None,
        dino_images_targets=None,
    ):
        if dino_image_proto is None or dino_cov_image is None:
            raise ValueError('BiMCDinoFusion requires dino_image_proto and dino_cov_image in the merged state.')

        clip_feat = F.normalize(self.extract_img_feature(images), dim=-1).float()
        dino_feat = self.extract_dino_img_feature(images).float()
        image_proto = F.normalize(image_proto[:num_cls].to(clip_feat.device).float(), dim=-1)
        dino_image_proto = F.normalize(dino_image_proto[:num_cls].to(dino_feat.device).float(), dim=-1)

        lambda_t = self.cfg.TRAINER.BiMC.LAMBDA_T if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
        text_features = F.normalize(text_features[:num_cls].to(clip_feat.device).float(), dim=-1)
        description_proto = F.normalize(description_proto[:num_cls].to(clip_feat.device).float(), dim=-1)
        semantic_proto = F.normalize((1 - lambda_t) * text_features + lambda_t * description_proto, dim=-1)

        text_logits = torch.matmul(clip_feat, semantic_proto.T)
        clip_visual_logits = torch.matmul(clip_feat, image_proto.T)
        dino_visual_logits = torch.matmul(dino_feat, dino_image_proto.T)
        visual_logits = self.OMEGA * dino_visual_logits + (1.0 - self.OMEGA) * clip_visual_logits
        fused_logits = float(beta) * text_logits + (1.0 - float(beta)) * visual_logits
        prob_fused_proto = F.softmax(fused_logits * self.LOGIT_TEMP, dim=-1)

        prob_cov = F.softmax(
            self._mahalanobis_logits(dino_feat, dino_image_proto, dino_cov_image.to(dino_feat.device)) / self.DINO_COV_SCALE,
            dim=-1,
        )
        prob_knn = F.softmax(
            self._knn_similarity_scores(clip_feat, description_features.float(), description_targets, num_cls),
            dim=-1,
        )

        if self.cfg.TRAINER.BiMC.USING_ENSEMBLE:
            ensemble_alpha = self.cfg.DATASET.ENSEMBLE_ALPHA
        else:
            ensemble_alpha = 1.0

        base_probs = ensemble_alpha * prob_fused_proto[:, :num_base_cls] + (1 - ensemble_alpha) * prob_cov[:, :num_base_cls]
        inc_probs = ensemble_alpha * prob_fused_proto[:, num_base_cls:] + (1 - ensemble_alpha) * prob_knn[:, num_base_cls:]
        return torch.cat([base_probs, inc_probs], dim=1)
