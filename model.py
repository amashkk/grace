"""
GRACE Model: CLIP ViT-B/32 with PEFT LoRA on the visual encoder.

References:
  - Section 5.2, Eq. (5): W(θ) = W(θ₀) + B_W · A_W
  - Section F.3: PyTorch + Transformers + PEFT
  - Section F.4: Pre-trained weights from OpenAI (HuggingFace)
  - Appendix B.1, Eq. (12)-(14): Zero-shot classifier from text prompts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model
from typing import Dict, List, Tuple

from config import GRACEConfig


# CLIP-style prompt templates for zero-shot classification (Appendix B.1)
CLIP_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a photo of the large {}.",
    "a photo of the small {}.",
    "a photo of many {}.",
    "a rendition of a {}.",
    "a cropped photo of the {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a photo of a {} in the wild.",
]


class GRACEModel(nn.Module):
    """
    CLIP model with PEFT LoRA fine-tuning for the GRACE framework.

    Section 5.2, Eq. (5): Only LoRA adapters Θ = {A_W, B_W}_W are trainable;
    frozen backbone W(θ₀) is fixed. This constrains the adapted parameters
    to remain in a low-rank subspace around the pre-trained weights,
    controlling the KL term in Eq. (2).
    """

    def __init__(self, config: GRACEConfig):
        super().__init__()
        self.config = config

        # Section F.3-F.4: Load CLIP ViT-B/32 from OpenAI via HuggingFace
        self.clip_model = CLIPModel.from_pretrained(config.clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(config.clip_model_name)

        # Freeze all parameters first
        for param in self.clip_model.parameters():
            param.requires_grad_(False)

        # Section 5.2, Eq. (5): Apply LoRA to visual encoder via PEFT
        # W(θ) = W(θ₀) + B_W · A_W, rank r ≪ min(n₁, n₂)
        lora_config = LoraConfig(
            r=config.lora_rank,                          # Table 10: r_W = 64
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,   # q_proj, k_proj, v_proj, out_proj
            modules_to_save=None,
        )
        # Apply LoRA only to the vision model
        self.clip_model.vision_model = get_peft_model(
            self.clip_model.vision_model, lora_config
        )

        # Zero-shot classifier weights: built from text prompts (Eq. 12-14)
        self.register_buffer("classifier_weights", None)

    @torch.no_grad()
    def build_classifier(self, classnames: List[str], device: torch.device):
        """
        Build zero-shot classifier from class names using CLIP text encoder.

        Appendix B.1, Eq. (12)-(13):
          W = [G_φ(t₁)^T; ...; G_φ(t_K)^T] ∈ R^{K×D}
          u(x) = W · z(x)
        """
        self.clip_model.eval()
        zeroshot_weights = []

        for classname in classnames:
            texts = [template.format(classname) for template in CLIP_TEMPLATES]
            # Tokenize using the processor
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()
                      if k in ["input_ids", "attention_mask"]}
            text_outputs = self.clip_model.text_model(**inputs)
            # Use pooled output (CLS token) and project
            text_features = self.clip_model.text_projection(text_outputs.pooler_output)
            # L2-normalize (features on unit hypersphere)
            text_features = F.normalize(text_features, dim=-1)
            # Average over templates for this class
            class_embedding = text_features.mean(dim=0)
            class_embedding = F.normalize(class_embedding, dim=-1)
            zeroshot_weights.append(class_embedding)

        # classifier_weights: (D, num_classes) for logits = features @ weights
        self.classifier_weights = torch.stack(zeroshot_weights, dim=1)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images through visual encoder with LoRA.
        Returns L2-normalized penultimate-layer features on the unit hypersphere.

        Section 4 / Assumption 2: f_θ maps to the unit sphere.
        """
        # Get vision model outputs
        vision_outputs = self.clip_model.vision_model(pixel_values=images)
        # Use the pooled output (CLS token projected)
        features = vision_outputs.pooler_output
        # Project through the visual projection layer
        features = self.clip_model.visual_projection(features)
        # L2-normalize to unit hypersphere
        features = F.normalize(features, dim=-1)
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode images and compute logits against zero-shot classifier.

        Appendix B.1, Eq. (13): u(x) = W · z(x)
        Returns logits of shape (B, num_classes).
        """
        features = self.encode_image(images)  # (B, D)
        # Eq. (13): logits = features @ classifier_weights, scaled by CLIP temperature
        logits = 100.0 * features @ self.classifier_weights
        return logits

    def get_lora_params(self) -> List[nn.Parameter]:
        """Return all trainable LoRA parameters Θ = {A_W, B_W}_W (Eq. 5)."""
        return [p for p in self.clip_model.vision_model.parameters() if p.requires_grad]

    def get_lora_layers(self) -> Dict[str, nn.Module]:
        """Return dict of all LoRA layer modules by name for curvature estimation."""
        lora_layers = {}
        for name, module in self.clip_model.vision_model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                lora_layers[name] = module
        return lora_layers
