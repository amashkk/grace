"""
GRACE training loop — follows Algorithm 1 (page 16) exactly.

Algorithm 1 order per iteration:
  Line 4:      Sample minibatch
  Line 5:      Clean forward → f_ID, L_task
  Line 6:      PGD attack → x_Adv          (Step 1)
  Line 7:      Compute f_Adv
  Lines 8-12:  LAR-AWP inner max            (Step 2)
  Line 13:     f_AWP via W_pert (Eq. 6)
  Lines 14-15: Gram-Volume L_GV             (Step 3)
  Lines 16-19: Curvature update (mod K)     (Step 4)
  Line 20:     Outer update L_GRACE         (Step 5)
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from config import GRACEConfig
from model import GRACEModel
from losses import compute_grace_loss
from curvature import CurvatureEstimator
from evaluate import evaluate_id, evaluate_adversarial

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def build_optimizer(model: GRACEModel, config: GRACEConfig):
    """Table 10: lr = 2e-4, AdamW."""
    params = model.get_lora_params()
    return AdamW(params, lr=config.lr, weight_decay=config.weight_decay)


def build_scheduler(optimizer, config: GRACEConfig, steps_per_epoch: int):
    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = config.epochs * steps_per_epoch

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


def train_one_epoch(
    model: GRACEModel,
    train_loader,
    optimizer,
    scheduler,
    config: GRACEConfig,
    curvature_estimator: CurvatureEstimator,
    epoch: int,
    global_step: int,
    device: torch.device,
):
    """One epoch of GRACE training following Algorithm 1."""
    model.train()
    running_metrics = {}
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")

    for batch_idx, (images, labels) in enumerate(pbar):
        # --- Algorithm 1, line 4: Sample minibatch ---
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # --- Algorithm 1, lines 16-19: Curvature update (Step 4) ---
        # "Curvature updates are performed once every K iterations (K=1000)"
        if global_step % config.curvature_update_K == 0:
            curvature_estimator.update_curvature(model, images, labels)

        rank_allocation = curvature_estimator.get_rank_allocation()

        # --- Algorithm 1, lines 5-15: Compute L_GRACE ---
        optimizer.zero_grad()
        L_GRACE, metrics = compute_grace_loss(
            model=model,
            images=images,
            labels=labels,
            config=config,
            rank_allocation=rank_allocation,
        )

        # --- Algorithm 1, line 20: Outer update ---
        L_GRACE.backward()
        torch.nn.utils.clip_grad_norm_(model.get_lora_params(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        global_step += 1

        # Accumulate metrics
        for k, v in metrics.items():
            running_metrics[k] = running_metrics.get(k, 0) + v
        num_batches += 1

        if (batch_idx + 1) % config.log_interval == 0:
            avg = {k: v / num_batches for k, v in running_metrics.items()}
            pbar.set_postfix({
                "loss": f"{avg['L_GRACE']:.4f}",
                "acc": f"{avg['clean_acc']:.3f}",
                "awp": f"{avg['L_LAR_AWP']:.4f}",
                "gv": f"{avg['L_GV']:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            if HAS_WANDB and config.wandb_project:
                wandb.log({
                    **{f"train/{k}": v for k, v in avg.items()},
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/global_step": global_step,
                })

    avg_metrics = {k: v / num_batches for k, v in running_metrics.items()}
    return avg_metrics, global_step


def train(
    model: GRACEModel,
    train_loader,
    val_loader,
    config: GRACEConfig,
    device: torch.device,
):
    """
    Full GRACE training following Algorithm 1.

    Line 1: LoRA init + freeze (done in GRACEModel.__init__)
    Line 2: r_AWP ← 0, h_W ← 0 (done in CurvatureEstimator.__init__)
    Lines 3-21: Training loop
    """
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))

    # Algorithm 1, line 2: Initialize curvature estimator (h_W ← 0, r_AWP ← 0)
    curvature_estimator = CurvatureEstimator(config, model)

    if HAS_WANDB and config.wandb_project:
        wandb.init(project=config.wandb_project, config=vars(config))

    os.makedirs(config.save_dir, exist_ok=True)
    best_harmonic = 0.0
    global_step = 0

    # Algorithm 1, line 3: For each training iteration
    for epoch in range(config.epochs):
        t0 = time.time()

        train_metrics, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            curvature_estimator=curvature_estimator,
            epoch=epoch,
            global_step=global_step,
            device=device,
        )

        # Evaluate
        model.eval()
        id_acc = evaluate_id(model, val_loader, device)

        adv_acc = 0.0
        if config.eval_adversarial:
            adv_acc = evaluate_adversarial(model, val_loader, config, device)

        epoch_time = time.time() - t0

        harmonic = 2 * id_acc * adv_acc / (id_acc + adv_acc + 1e-8) if adv_acc > 0 else 0.0

        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"ID: {id_acc:.1f}% | Adv: {adv_acc:.1f}% | "
            f"Harmonic: {harmonic:.1f}% | Time: {epoch_time:.0f}s"
        )

        if HAS_WANDB and config.wandb_project:
            wandb.log({"val/id_acc": id_acc, "val/adv_acc": adv_acc,
                        "val/harmonic": harmonic, "epoch": epoch + 1})

        # Save best
        if harmonic > best_harmonic:
            best_harmonic = harmonic
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": {k: v for k, v in model.state_dict().items() if "lora_" in k},
                "optimizer_state_dict": optimizer.state_dict(),
                "id_acc": id_acc, "adv_acc": adv_acc, "harmonic": harmonic,
                "config": vars(config),
            }, os.path.join(config.save_dir, "best.pt"))
            print(f"  -> Saved best (harmonic={harmonic:.1f}%)")

        # Save latest
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": {k: v for k, v in model.state_dict().items() if "lora_" in k},
            "optimizer_state_dict": optimizer.state_dict(),
            "id_acc": id_acc, "adv_acc": adv_acc, "harmonic": harmonic,
        }, os.path.join(config.save_dir, "latest.pt"))

    if HAS_WANDB and config.wandb_project:
        wandb.finish()

    print(f"\nTraining complete. Best harmonic mean: {best_harmonic:.1f}%")
    return model
