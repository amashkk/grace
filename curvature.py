"""
Curvature estimation and rank curriculum for LAR-AWP.

References:
  - Appendix E.1: Full description of curvature estimator and rank curriculum
  - Eq. (28): ĥ_W = n_v · (g_W ⊙ g_W)  [Gauss-Newton proxy]
  - Eq. (29): h_W^(t) = β · h_W^(t-1) + (1-β) · ĥ_W^(t)  [EMA stabilization]
  - Practical Rank Allocation: sharp layers (top 20%) → highest rank,
    flat layers (bottom 20%) → minimal or zero rank
  - Algorithm 1, lines 16-19: Curvature update every K=1000 iterations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict

from config import GRACEConfig


class CurvatureEstimator:
    """
    Maintains per-layer curvature estimates via Gauss-Newton proxy with EMA.
    Allocates AWP perturbation ranks based on curvature percentiles.

    Appendix E.1: "We adopt a tractable first-order proxy given by the
    diagonal of the Gauss-Newton matrix."

    Works with PEFT LoRA layers by collecting named parameters that
    contain 'lora_' in their name.
    """

    def __init__(self, config: GRACEConfig, model: nn.Module):
        self.config = config

        # Discover all LoRA parameter groups by layer prefix
        # e.g., "base_model.model.encoder.layers.0.self_attn.q_proj.lora_A.default.weight"
        # → layer key: "encoder.layers.0.self_attn.q_proj"
        self.layer_params: Dict[str, list] = {}
        for name, param in model.named_parameters():
            if param.requires_grad and "lora_" in name:
                # Extract the layer prefix (everything before "lora_A" or "lora_B")
                for marker in [".lora_A.", ".lora_B."]:
                    if marker in name:
                        layer_key = name.split(marker)[0]
                        if layer_key not in self.layer_params:
                            self.layer_params[layer_key] = []
                        self.layer_params[layer_key].append((name, param))
                        break

        # Algorithm 1, line 2: Initialize h_W ← 0 for all layers W
        self.h_W: Dict[str, float] = {k: 0.0 for k in self.layer_params}

        # Algorithm 1, line 2: Initialize r_AWP^(W) ← 0
        self.r_AWP: Dict[str, int] = {k: 0 for k in self.layer_params}

        self.initialized = False
        print(f"  CurvatureEstimator tracking {len(self.layer_params)} LoRA layer groups")

    def update_curvature(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor):
        """
        Algorithm 1, lines 16-18: Estimate curvature proxy and update ranks.

        Eq. (28): ĥ_W = n_v · (g_W ⊙ g_W)
        where g_W = ∇_W (1/n_v) Σ L(F_θ(x_vi), y_vi)

        Eq. (29): h_W^(t) = β · h_W^(t-1) + (1-β) · ĥ_W^(t)
        """
        model.zero_grad()
        n_v = images.shape[0]

        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()

        beta = self.config.curvature_ema_beta  # β ∈ [0.85, 0.95]

        for layer_key, param_list in self.layer_params.items():
            # Eq. (28): ĥ_W = n_v · (g_W ⊙ g_W) for each param in this layer
            h_hat_W = 0.0
            total_numel = 0

            for param_name, param in param_list:
                if param.grad is not None:
                    g_W = param.grad.detach()
                    # Eq. (28): element-wise squared gradient scaled by n_v
                    h_hat_W += (n_v * (g_W ** 2)).sum().item()
                    total_numel += param.numel()

            # Appendix E.1: h_norm_W = (1/|W|) · Σ ĥ_W
            if total_numel > 0:
                h_hat_W = h_hat_W / total_numel

            # Eq. (29): h_W^(t) = β · h_W^(t-1) + (1-β) · ĥ_W^(t)
            if self.initialized:
                self.h_W[layer_key] = beta * self.h_W[layer_key] + (1.0 - beta) * h_hat_W
            else:
                self.h_W[layer_key] = h_hat_W

        self.initialized = True
        self._update_rank_allocation()
        model.zero_grad()

    def _update_rank_allocation(self):
        """
        Appendix E.1 — Practical Rank Allocation:
          r_AWP ∝ 1{h_norm_W ≥ τ_p}, τ_p = quantile({ĥ_W}, p=0.8)
          "sharp layers (top 20%) receive the highest rank,
           moderately curved layers receive intermediate ranks,
           and flat layers (bottom 20%) receive minimal or zero rank."
        """
        curvature_values = np.array(list(self.h_W.values()))
        names = list(self.h_W.keys())

        if curvature_values.sum() == 0:
            return

        r_max = self.config.awp_max_rank  # Table 10: r_AWP_max = 4
        p = self.config.curvature_percentile  # Table 10: φ_AWP = 0.8

        tau_high = np.percentile(curvature_values, p * 100)      # 80th percentile
        tau_low = np.percentile(curvature_values, (1 - p) * 100)  # 20th percentile

        for name, curv in zip(names, curvature_values):
            if curv >= tau_high:
                self.r_AWP[name] = r_max          # Sharp: top 20% → highest rank
            elif curv >= tau_low:
                self.r_AWP[name] = max(1, r_max // 2)  # Moderate → intermediate
            else:
                self.r_AWP[name] = 0              # Flat: bottom 20% → zero

    def get_rank_allocation(self) -> Dict[str, int]:
        """Return current per-layer AWP rank allocation."""
        return dict(self.r_AWP)
