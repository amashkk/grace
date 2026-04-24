"""
GRACE loss functions following Algorithm 1 exactly.

References:
  - Eq. (4):  L_GRACE = L_task + λ_LAR · L_LAR-AWP + λ_GV · L_GV
  - Eq. (6):  W_pert(θ,Δ) = W(θ₀) + B_W·A_W + B_AWP·A_AWP
  - Eq. (7):  L_LAR-AWP ≈ (1/n) Σ max_{‖δ‖≤ε, ‖Δ‖≤ρ} L(F_{W_pert}(x_i), y_i)
  - Eq. (8):  G_i = [⟨f_j, f_k⟩]_{j,k ∈ {ID,Adv,AWP}} + εI
  - Eq. (9):  L_GV = √|det(G_i)|
  - Eq. (19): PGD: x_{t+1} = Π_S(x_t + α·sign(∇_{x_t} L))
  - Algorithm 1, Steps 1-3 and line 20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List

from config import GRACEConfig


# ============================================================
# Step 1: PGD Attack (Algorithm 1, line 6; Eq. 19)
# ============================================================

def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    step_size: float,
    num_steps: int,
) -> torch.Tensor:
    """
    Algorithm 1, line 6: PGD with radius ε to obtain x_Adv.
    Eq. (19): x_{t+1} = Π_S(x_t + α · sign(∇_{x_t} L(f(x_t), y)))

    Args:
        epsilon: L∞ perturbation radius (Table 10: 4/255)
        step_size: PGD step size α (Table 10: 1/255)
        num_steps: number of PGD steps (10)
    """
    x_adv = images.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv)[0]

        # Eq. (19): ascent step + project onto ε-ball
        x_adv = x_adv.detach() + step_size * grad.sign()
        perturbation = torch.clamp(x_adv - images, -epsilon, epsilon)
        x_adv = torch.clamp(images + perturbation, 0.0, 1.0)

    return x_adv.detach()


# ============================================================
# Step 2: LAR-AWP Inner Maximization (Algorithm 1, lines 8-13; Eq. 6-7)
# ============================================================

def _get_base_layer_for_lora(model: nn.Module, layer_key: str):
    """
    Given a layer_key like 'base_model.model.encoder.layers.0.self_attn.q_proj',
    navigate the module tree and return the base_layer (nn.Linear) whose weight
    we perturb for the AWP delta.
    """
    parts = layer_key.split(".")
    module = model
    for part in parts:
        if hasattr(module, part):
            module = getattr(module, part)
        else:
            return None
    # PEFT wraps the original linear in module.base_layer
    if hasattr(module, "base_layer"):
        return module.base_layer
    return module


def _make_awp_hook(A_AWP, B_AWP):
    """Create a forward hook that adds B_AWP @ A_AWP perturbation differentiably.

    For nn.Linear: output = input @ W.T + bias
    Adding delta_W to W gives: output + input @ delta_W.T
    where delta_W = B_AWP @ A_AWP. This keeps A_AWP, B_AWP in the graph.
    """
    def hook(module, input, output):
        x = input[0]  # (batch, ..., in_features)
        delta_W = B_AWP @ A_AWP  # (out_features, in_features)
        return output + x @ delta_W.t()
    return hook


def lar_awp_inner_max(
    model: nn.Module,
    rank_allocation: Dict[str, int],
    x_adv: torch.Tensor,
    labels: torch.Tensor,
    config: GRACEConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Algorithm 1, lines 8-13: LAR-AWP inner maximization.

    Eq. (6): W_pert(θ,Δ) = W(θ₀) + B_W·A_W + B_AWP·A_AWP
    Eq. (7): L_LAR-AWP ≈ (1/n) Σ max_{‖Δ‖≤ρ} L(F_{W_pert}(x_Adv), y)

    Uses forward hooks to add B_AWP @ A_AWP to each layer's output
    differentiably, keeping A_AWP/B_AWP in the computation graph.

    Returns:
        (L_LAR_AWP, f_AWP): adversarial loss under perturbed weights, and features
    """
    rho = config.awp_rho
    T_AWP = config.awp_inner_steps
    eta = config.awp_inner_lr

    # Collect base layers and their shapes for AWP perturbation
    awp_layers = {}  # layer_key -> (base_layer, A_AWP, B_AWP)

    for layer_key, r_AWP_W in rank_allocation.items():
        if r_AWP_W == 0:
            continue

        base_layer = _get_base_layer_for_lora(model, layer_key)
        if base_layer is None or not hasattr(base_layer, "weight"):
            continue

        out_features, in_features = base_layer.weight.shape

        # Eq. (6): A_AWP ∈ R^{r_AWP × n2}, B_AWP ∈ R^{n1 × r_AWP}
        A_AWP = torch.randn(r_AWP_W, in_features, device=base_layer.weight.device) * 0.01
        B_AWP = torch.zeros(out_features, r_AWP_W, device=base_layer.weight.device)
        A_AWP.requires_grad_(True)
        B_AWP.requires_grad_(True)

        awp_layers[layer_key] = (base_layer, A_AWP, B_AWP)

    if not awp_layers:
        # No layers to perturb (all ranks are 0)
        f_AWP = model.encode_image(x_adv)
        logits = 100.0 * f_AWP @ model.classifier_weights
        L_LAR_AWP = F.cross_entropy(logits, labels)
        return L_LAR_AWP, f_AWP.detach()

    # Algorithm 1, lines 8-12: Inner maximization loop
    for t in range(T_AWP):
        # Register forward hooks to add perturbation differentiably
        hooks = []
        for layer_key, (base_layer, A_AWP, B_AWP) in awp_layers.items():
            h = base_layer.register_forward_hook(_make_awp_hook(A_AWP, B_AWP))
            hooks.append(h)

        # Line 10: compute loss under perturbed weights
        logits = model(x_adv)
        loss = F.cross_entropy(logits, labels)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Gradient w.r.t. AWP params
        awp_params = []
        for layer_key, (_, A, B) in awp_layers.items():
            awp_params.extend([A, B])
        grads = torch.autograd.grad(loss, awp_params, retain_graph=False)

        # Line 10: (A_AWP, B_AWP) ← (A_AWP, B_AWP) + η · ∇_Δ L
        idx = 0
        new_awp_layers = {}
        for layer_key, (base_layer, A_AWP, B_AWP) in awp_layers.items():
            A_AWP = A_AWP.detach() + eta * grads[idx]
            B_AWP = B_AWP.detach() + eta * grads[idx + 1]
            idx += 2

            # Line 11: Project Δ onto ‖Δ‖ ≤ ρ
            norm = torch.sqrt(torch.sum(A_AWP ** 2) + torch.sum(B_AWP ** 2))
            if norm > rho:
                scale = rho / (norm + 1e-12)
                A_AWP = A_AWP * scale
                B_AWP = B_AWP * scale

            A_AWP.requires_grad_(True)
            B_AWP.requires_grad_(True)
            new_awp_layers[layer_key] = (base_layer, A_AWP, B_AWP)

        awp_layers = new_awp_layers

    # Algorithm 1, line 13: Compute f_AWP using final W_pert(θ, Δ) from Eq. (6)
    hooks = []
    for layer_key, (base_layer, A_AWP, B_AWP) in awp_layers.items():
        h = base_layer.register_forward_hook(_make_awp_hook(A_AWP, B_AWP))
        hooks.append(h)

    f_AWP = model.encode_image(x_adv)
    logits_AWP = 100.0 * f_AWP @ model.classifier_weights
    L_LAR_AWP = F.cross_entropy(logits_AWP, labels)

    for h in hooks:
        h.remove()

    return L_LAR_AWP, f_AWP.detach()


# ============================================================
# Step 3: Gram-Volume Alignment Loss (Algorithm 1, lines 14-15; Eq. 8-9)
# ============================================================

def gram_volume_loss(
    f_ID: torch.Tensor,
    f_Adv: torch.Tensor,
    f_AWP: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Algorithm 1, lines 14-15: Gram-Volume feature alignment.

    Eq. (8): G_i = [⟨f_j, f_k⟩]_{j,k ∈ {ID,Adv,AWP}} + εI
    where f_ID, f_Adv, f_AWP ∈ R^D are L2-normalized image features.

    Eq. (9): L_GV = √|det(G_i)|
    "L_GV ≈ 0 when the three representations are close (stable manifold),
     and L_GV grows as perturbations push features into diverging directions."
    """
    B = f_ID.shape[0]

    f_ID = F.normalize(f_ID, dim=-1)
    f_Adv = F.normalize(f_Adv, dim=-1)
    f_AWP = F.normalize(f_AWP, dim=-1)

    # Eq. (8): 3×3 Gram matrix per sample
    F_stack = torch.stack([f_ID, f_Adv, f_AWP], dim=1)  # (B, 3, D)
    G_i = torch.bmm(F_stack, F_stack.transpose(1, 2))     # (B, 3, 3)
    G_i = G_i + eps * torch.eye(3, device=G_i.device).unsqueeze(0)

    # Eq. (9): L_GV = √|det(G_i)|
    det_G = torch.det(G_i)
    L_GV = torch.sqrt(torch.abs(det_G) + 1e-12)

    return L_GV.mean()


# ============================================================
# Combined Loss: Eq. (4) and Algorithm 1, line 20
# ============================================================

def compute_grace_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: GRACEConfig,
    rank_allocation: Dict[str, int],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute full GRACE loss following Algorithm 1, lines 5-15 and 20.

    Eq. (4): L_GRACE = L_task + λ_LAR · L_LAR-AWP + λ_GV · L_GV

    Exact order from Algorithm 1:
      Line 5:     f_ID, L_task
      Line 6:     PGD → x_Adv
      Line 7:     f_Adv
      Lines 8-13: LAR-AWP → f_AWP, L_LAR-AWP
      Lines 14-15: Gram-Volume → L_GV
      Line 20:    L_GRACE = L_task + λ_LAR · L_LAR-AWP + λ_GV · L_GV
    """
    # --- Line 5: Clean features and L_task ---
    f_ID = model.encode_image(images)
    logits_clean = 100.0 * f_ID @ model.classifier_weights
    L_task = F.cross_entropy(logits_clean, labels)

    # --- Line 6: PGD attack → x_Adv ---
    x_Adv = pgd_attack(
        model=model, images=images, labels=labels,
        epsilon=config.pgd_epsilon, step_size=config.pgd_step_size,
        num_steps=config.pgd_steps,
    )

    # --- Line 7: Compute f_Adv ---
    f_Adv = model.encode_image(x_Adv)

    # --- Lines 8-13: LAR-AWP inner maximization ---
    L_LAR_AWP, f_AWP = lar_awp_inner_max(
        model=model, rank_allocation=rank_allocation,
        x_adv=x_Adv, labels=labels, config=config,
    )

    # --- Lines 14-15: Gram-Volume alignment loss ---
    L_GV = gram_volume_loss(
        f_ID=f_ID.detach(), f_Adv=f_Adv.detach(), f_AWP=f_AWP,
        eps=config.gram_eps,
    )

    # --- Line 20: Combined loss (Eq. 4) ---
    L_GRACE = L_task + config.lambda_LAR * L_LAR_AWP + config.lambda_GV * L_GV

    metrics = {
        "L_GRACE": L_GRACE.item(),
        "L_task": L_task.item(),
        "L_LAR_AWP": L_LAR_AWP.item(),
        "L_GV": L_GV.item(),
        "clean_acc": (logits_clean.argmax(dim=-1) == labels).float().mean().item(),
    }

    return L_GRACE, metrics
