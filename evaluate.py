"""
Evaluation utilities for GRACE.

Supports:
- In-Distribution (ID) accuracy on CIFAR-100 test
- Adversarial robustness via PGD and AutoAttack (APGD-CE at ε=4/255)
- Out-of-Distribution (OOD) accuracy on CIFAR-100-C corruptions
- Feature geometry analysis (cosine alignment, LID)

Reference: Section 6.1 — Testing adversarial robustness:
  "During training, adversarial samples are generated using 10-step PGD
   under L∞ radius 4/255 and step size 1/255. At test time, we employ
   AutoAttack (APGD-CE) at the same perturbation radius."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict

from config import GRACEConfig
from losses import pgd_attack


@torch.no_grad()
def evaluate_id(model: nn.Module, val_loader, device: torch.device) -> float:
    """Evaluate top-1 accuracy on in-distribution data."""
    model.eval()
    correct = 0
    total = 0

    for images, labels in tqdm(val_loader, desc="ID Eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.shape[0]

    return 100.0 * correct / total


def evaluate_adversarial(
    model: nn.Module,
    val_loader,
    config: GRACEConfig,
    device: torch.device,
    use_autoattack: bool = False,
) -> float:
    """
    Evaluate adversarial robustness.

    Section 6.1: "At test time, we employ AutoAttack (APGD-CE) at ε=4/255."
    Falls back to PGD if AutoAttack is not installed.
    """
    model.eval()

    if use_autoattack:
        return _evaluate_autoattack(model, val_loader, config, device)

    correct = 0
    total = 0

    for images, labels in tqdm(val_loader, desc="Adv Eval (PGD)", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        adv_images = pgd_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=config.pgd_epsilon,
            step_size=config.pgd_step_size,
            num_steps=config.pgd_steps,
        )

        with torch.no_grad():
            logits = model(adv_images)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]

    return 100.0 * correct / total


def _evaluate_autoattack(
    model: nn.Module,
    val_loader,
    config: GRACEConfig,
    device: torch.device,
) -> float:
    """Section 6.1: AutoAttack (APGD-CE) at ε=4/255."""
    try:
        from autoattack import AutoAttack
    except ImportError:
        print("AutoAttack not installed. Falling back to PGD.")
        return evaluate_adversarial(model, val_loader, config, device, False)

    all_images, all_labels = [], []
    for images, labels in val_loader:
        all_images.append(images)
        all_labels.append(labels)
    all_images = torch.cat(all_images, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)

    if all_images.shape[0] > 5000:
        indices = torch.randperm(all_images.shape[0])[:5000]
        all_images = all_images[indices]
        all_labels = all_labels[indices]

    adversary = AutoAttack(
        model, norm="Linf", eps=config.pgd_epsilon,
        version="standard", verbose=False,
    )
    adversary.attacks_to_run = ["apgd-ce"]

    adv_images = adversary.run_standard_evaluation(
        all_images, all_labels, bs=config.batch_size
    )

    with torch.no_grad():
        logits = model(adv_images)
        correct = (logits.argmax(dim=-1) == all_labels).sum().item()

    return 100.0 * correct / all_images.shape[0]


@torch.no_grad()
def evaluate_ood(
    model: nn.Module,
    ood_loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate accuracy on multiple OOD datasets."""
    model.eval()
    results = {}

    for name, loader in ood_loaders.items():
        correct = 0
        total = 0
        for images, labels in tqdm(loader, desc=f"OOD ({name})", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.shape[0]
        results[name] = 100.0 * correct / total

    if results:
        results["ood_avg"] = sum(results.values()) / len(results)
    return results


@torch.no_grad()
def analyze_feature_geometry(
    model: nn.Module,
    clean_loader,
    config: GRACEConfig,
    device: torch.device,
    num_samples: int = 1000,
) -> Dict[str, float]:
    """
    Feature geometry analysis from Section 4 / Table 2:
    - Cosine alignment between clean and adversarial class centroids
    - ΔLID (change in Local Intrinsic Dimensionality)
    """
    model.eval()
    clean_features, adv_features = [], []
    collected = 0

    for images, labels in clean_loader:
        if collected >= num_samples:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        f_clean = model.encode_image(images)

        model.train()
        adv_images = pgd_attack(
            model, images, labels,
            config.pgd_epsilon, config.pgd_step_size, config.pgd_steps,
        )
        model.eval()
        f_adv = model.encode_image(adv_images)

        clean_features.append(f_clean.cpu())
        adv_features.append(f_adv.cpu())
        collected += images.shape[0]

    clean_features = torch.cat(clean_features, dim=0)[:num_samples]
    adv_features = torch.cat(adv_features, dim=0)[:num_samples]

    cos_sim = F.cosine_similarity(clean_features, adv_features, dim=-1)
    mean_alignment = cos_sim.mean().item()

    lid_clean = _estimate_lid(clean_features, k=20)
    lid_adv = _estimate_lid(adv_features, k=20)
    delta_lid = abs(lid_adv - lid_clean)

    return {
        "cosine_alignment": mean_alignment,
        "lid_clean": lid_clean,
        "lid_adv": lid_adv,
        "delta_lid": delta_lid,
    }


def _estimate_lid(features: torch.Tensor, k: int = 20) -> float:
    """Estimate Local Intrinsic Dimensionality using k-NN MLE estimator."""
    dists = torch.cdist(features, features)
    knn_dists, _ = dists.topk(k + 1, dim=-1, largest=False)
    knn_dists = knn_dists[:, 1:]
    knn_dists = torch.clamp(knn_dists, min=1e-10)
    log_dists = torch.log(knn_dists)
    r_max = knn_dists[:, -1:]
    lid_per_sample = -k / (log_dists - torch.log(r_max)).sum(dim=-1)
    return lid_per_sample.mean().item()
