"""
GRACE Configuration — exact hyperparameters from Table 10 (p.18).

Paper: "The Geometry of Robustness" (arXiv:2603.27139)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GRACEConfig:
    # ---- Model (Section F.3-F.4) ----
    clip_model_name: str = "openai/clip-vit-base-patch32"  # HuggingFace model ID

    # ---- LoRA (Section 5.2, Eq. 5) ----
    # Table 10: r_W = 64
    lora_rank: int = 64
    lora_alpha: float = 128.0          # scaling = alpha / rank = 2.0
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj"]
    )

    # ---- LAR-AWP (Section 5.3, Eq. 6-7) ----
    # Table 10: r_AWP_max = 4
    awp_max_rank: int = 4              # maximum perturbation rank per layer
    awp_rho: float = 0.05             # perturbation budget ρ (‖Δ‖ ≤ ρ)
    awp_inner_steps: int = 1          # T_AWP: inner maximization steps
    awp_inner_lr: float = 0.01        # η for inner gradient ascent (Alg 1, line 10)

    # ---- Curvature Estimation (Appendix E.1, Eq. 28-29) ----
    curvature_ema_beta: float = 0.9   # β ∈ [0.85, 0.95] (Eq. 29)
    curvature_update_K: int = 1000    # update every K iterations (Alg 1, line 16)
    curvature_percentile: float = 0.8 # φ_AWP = 80th percentile (Table 10)

    # ---- PGD Adversarial Attack (Section 6.1) ----
    # Table 10: ε = 4/255, step_size = 1/255
    pgd_epsilon: float = 4.0 / 255.0  # L∞ perturbation radius
    pgd_step_size: float = 1.0 / 255.0  # PGD step size α
    pgd_steps: int = 10               # number of PGD steps

    # ---- Loss weights (Eq. 4) ----
    # L_GRACE = L_task + λ_LAR · L_LAR-AWP + λ_GV · L_GV
    lambda_LAR: float = 1.0           # λ_LAR
    lambda_GV: float = 1.0            # λ_GV
    gram_eps: float = 1e-6            # ε in Gram matrix (Eq. 8)

    # ---- Training (Table 10) ----
    lr: float = 2e-4                  # η₃ (learning rate)
    weight_decay: float = 0.01
    epochs: int = 50
    batch_size: int = 256
    num_workers: int = 4
    warmup_epochs: int = 1
    seed: int = 42

    # ---- Data ----
    data_root: str = "./data"

    # ---- Logging / Checkpointing ----
    log_interval: int = 50
    save_dir: str = "./checkpoints"
    wandb_project: Optional[str] = None

    # ---- Evaluation ----
    eval_adversarial: bool = True
    autoattack_eval: bool = False
