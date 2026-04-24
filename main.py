"""
GRACE: Gram-aligned Robustness via Adaptive Curvature Estimation

Main entry point for training and evaluation.

Paper: "The Geometry of Robustness" (arXiv:2603.27139)
All defaults match Table 10 (page 18).

Usage:
    python main.py --mode train
    python main.py --mode eval --checkpoint checkpoints/best.pt
    python main.py --mode eval_full --checkpoint checkpoints/best.pt
"""

import argparse
import os
import torch
import random
import numpy as np

from config import GRACEConfig
from model import GRACEModel
from data import build_cifar100_loaders, build_cifar100c_loader, CIFAR100C_CORRUPTIONS
from train import train
from evaluate import evaluate_id, evaluate_adversarial, evaluate_ood, analyze_feature_geometry


def parse_args():
    parser = argparse.ArgumentParser(description="GRACE: Robust CLIP Fine-tuning")

    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval", "eval_full"])

    # Data
    parser.add_argument("--data_root", type=str, default="./data")

    # Model (Section F.4)
    parser.add_argument("--clip_model_name", type=str,
                        default="openai/clip-vit-base-patch32")

    # LoRA (Table 10)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=128.0)

    # GRACE hyperparameters (Table 10)
    parser.add_argument("--lambda_LAR", type=float, default=1.0)
    parser.add_argument("--lambda_GV", type=float, default=1.0)
    parser.add_argument("--awp_rho", type=float, default=0.05)
    parser.add_argument("--awp_max_rank", type=int, default=4)
    parser.add_argument("--pgd_epsilon", type=float, default=None,
                        help="L-inf epsilon (default: 4/255)")
    parser.add_argument("--curvature_percentile", type=float, default=0.8)
    parser.add_argument("--curvature_update_K", type=int, default=1000)

    # Training (Table 10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)

    # Evaluation
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--autoattack", action="store_true")
    parser.add_argument("--corruption_severity", type=int, default=5)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--wandb_project", type=str, default=None)

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_config(args) -> GRACEConfig:
    config = GRACEConfig()
    config.clip_model_name = args.clip_model_name
    config.lora_rank = args.lora_rank
    config.lora_alpha = args.lora_alpha
    config.lambda_LAR = args.lambda_LAR
    config.lambda_GV = args.lambda_GV
    config.awp_rho = args.awp_rho
    config.awp_max_rank = args.awp_max_rank
    config.curvature_percentile = args.curvature_percentile
    config.curvature_update_K = args.curvature_update_K
    if args.pgd_epsilon is not None:
        config.pgd_epsilon = args.pgd_epsilon
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.weight_decay = args.weight_decay
    config.num_workers = args.num_workers
    config.data_root = args.data_root
    config.seed = args.seed
    config.save_dir = args.save_dir
    config.wandb_project = args.wandb_project
    config.autoattack_eval = args.autoattack
    return config


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = build_config(args)

    # Load CLIP + LoRA (Section 5.2, F.3-F.4)
    print(f"Loading {config.clip_model_name} with LoRA (rank={config.lora_rank})...")
    model = GRACEModel(config)
    model.to(device)

    # Load CIFAR-100 (auto-downloads)
    print(f"Loading CIFAR-100 from {config.data_root}...")
    train_loader, test_loader, classnames = build_cifar100_loaders(
        root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # Build zero-shot classifier (Appendix B.1, Eq. 12-14)
    print("Building zero-shot classifier from text prompts...")
    model.build_classifier(classnames, device)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded (epoch {ckpt.get('epoch', '?')})")

    if args.mode == "train":
        print("\n" + "=" * 60)
        print("GRACE Training (Algorithm 1)")
        print(f"  Epochs: {config.epochs} | Batch: {config.batch_size}")
        print(f"  LR: {config.lr} | LoRA rank: {config.lora_rank}")
        print(f"  λ_LAR: {config.lambda_LAR} | λ_GV: {config.lambda_GV}")
        print(f"  AWP ρ: {config.awp_rho} | AWP max rank: {config.awp_max_rank}")
        print(f"  PGD ε: {config.pgd_epsilon:.4f} ({config.pgd_epsilon*255:.0f}/255)")
        print(f"  Curvature K: {config.curvature_update_K} | percentile: {config.curvature_percentile}")
        print("=" * 60 + "\n")

        model = train(model, train_loader, test_loader, config, device)

    elif args.mode == "eval":
        model.eval()
        print("\n--- Evaluation (CIFAR-100) ---")

        id_acc = evaluate_id(model, test_loader, device)
        print(f"ID Accuracy: {id_acc:.2f}%")

        adv_acc = evaluate_adversarial(
            model, test_loader, config, device, use_autoattack=args.autoattack
        )
        atk = "AutoAttack" if args.autoattack else "PGD"
        print(f"Adversarial Accuracy ({atk}): {adv_acc:.2f}%")

        harmonic = 2 * id_acc * adv_acc / (id_acc + adv_acc + 1e-8)
        print(f"Harmonic Mean: {harmonic:.2f}%")

    elif args.mode == "eval_full":
        model.eval()
        print("\n--- Full Evaluation (CIFAR-100) ---")

        id_acc = evaluate_id(model, test_loader, device)
        print(f"\nID Accuracy: {id_acc:.2f}%")

        adv_acc = evaluate_adversarial(
            model, test_loader, config, device, use_autoattack=args.autoattack
        )
        atk = "AutoAttack" if args.autoattack else "PGD"
        print(f"Adversarial Accuracy ({atk}): {adv_acc:.2f}%")

        # CIFAR-100-C corruptions
        cifar100c_path = os.path.join(args.data_root, "CIFAR-100-C")
        ood_results = {}
        if os.path.exists(cifar100c_path):
            print(f"\nCIFAR-100-C (severity={args.corruption_severity}):")
            ood_loaders = {}
            for c in CIFAR100C_CORRUPTIONS:
                try:
                    ood_loaders[c] = build_cifar100c_loader(
                        args.data_root, c, args.corruption_severity,
                        config.batch_size, config.num_workers,
                    )
                except FileNotFoundError:
                    pass
            if ood_loaders:
                ood_results = evaluate_ood(model, ood_loaders, device)
                for name, acc in ood_results.items():
                    print(f"  {name}: {acc:.2f}%")
        else:
            print(f"\nSkipping CIFAR-100-C (not found at {cifar100c_path})")

        print("\nFeature Geometry:")
        geometry = analyze_feature_geometry(model, test_loader, config, device)
        for name, val in geometry.items():
            print(f"  {name}: {val:.4f}")

        ood_avg = ood_results.get("ood_avg", 0.0)
        print(f"\n--- Summary ---")
        print(f"ID: {id_acc:.1f}% | OOD: {ood_avg:.1f}% | Adv: {adv_acc:.1f}%")
        if adv_acc > 0 and id_acc > 0 and ood_avg > 0:
            h3 = 3 / (1/id_acc + 1/ood_avg + 1/adv_acc)
            print(f"3-way Harmonic Mean: {h3:.1f}%")


if __name__ == "__main__":
    main()
