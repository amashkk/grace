# GRACE: Gram-aligned Robustness via Adaptive Curvature Estimation

Implementation of ["The Geometry of Robustness"](https://arxiv.org/abs/2603.27139) (Chopra et al., 2026).

Addresses the three-way trade-off between **ID accuracy**, **OOD generalization**, and **adversarial robustness** when fine-tuning CLIP. Uses **CIFAR-100** as dataset.

## Installation

```bash
py -m pip install torch torchvision transformers peft timm tqdm numpy wandb
```

## Training

```bash
# Default settings from Table 10 (LoRA rank=64, lr=2e-4, epochs=50, batch=256)
py run.py --mode train

# Smaller batch for limited GPU memory
py run.py --mode train --batch_size 64 --epochs 10
```

## Evaluation

```bash
py run.py --mode eval --checkpoint checkpoints/best.pt
py run.py --mode eval_full --checkpoint checkpoints/best.pt
```

## Architecture

All code follows **Algorithm 1** (page 16) exactly:

| Alg 1 Line | Step | File:Function | Equation |
|------------|------|---------------|----------|
| 1 | Init LoRA, freeze backbone | `model.py:GRACEModel.__init__` | Eq. (5) |
| 2 | Init r_AWP←0, h_W←0 | `curvature.py:CurvatureEstimator.__init__` | — |
| 5 | Clean forward → f_ID, L_task | `losses.py:compute_grace_loss` | — |
| 6 | PGD → x_Adv | `losses.py:pgd_attack` | Eq. (19) |
| 7 | Compute f_Adv | `losses.py:compute_grace_loss` | — |
| 8-12 | LAR-AWP inner max | `losses.py:lar_awp_inner_max` | Eq. (6)-(7) |
| 13 | f_AWP via W_pert | `losses.py:lar_awp_inner_max` | Eq. (6) |
| 14-15 | Gram-Volume L_GV | `losses.py:gram_volume_loss` | Eq. (8)-(9) |
| 16-19 | Curvature update (mod K) | `curvature.py:update_curvature` | Eq. (28)-(29) |
| 20 | Outer update L_GRACE | `train.py:train_one_epoch` | Eq. (4) |

## Hyperparameters (Table 10)

| Parameter | Value | Source |
|-----------|-------|--------|
| LoRA rank (r_W) | 64 | Table 10 |
| Max AWP rank (r_AWP) | 4 | Table 10 |
| PGD ε | 4/255 | Table 10 |
| PGD step size | 1/255 | Table 10 |
| Learning rate | 2e-4 | Table 10 |
| Curvature percentile (φ_AWP) | 80 | Table 10 |
| Epochs | 50 | Table 10 |
| Batch size | 256 | Table 10 |
| Curvature update K | 1000 | Appendix E.1 |
| EMA β | 0.9 | Eq. (29), β∈[0.85,0.95] |

## Citation

```bibtex
@article{chopra2026geometry,
  title={The Geometry of Robustness},
  author={Chopra, Shivang and Halbe, Shaunak and Huang, Chengyue and Maneechotesuwan, Brisa and Kira, Zsolt},
  journal={arXiv:2603.27139},
  year={2026}
}
```
