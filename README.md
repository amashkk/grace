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

# GRACE 複現結果

論文:_The Geometry of Robustness_ (arXiv:2603.27139, Chopra et al., 2026)

---

## 一、實驗設定(Table 10)

| 項目 | 設定 |
|---|---|
| Backbone | CLIP ViT-B/32(OpenAI) |
| Fine tuned | LoRA r=64,only visual encoder 的 q/k/v/out_proj |
| Dataset | CIFAR-100(50k train / 10k test,resize 32→224) |
| Attack | PGD-10,L∞,**ε=4/255**,**step=1/255** |
| Epochs / Batch / LR | 50 / 256 / 2e-4(AdamW + cosine) |
| GRACE parameter | λ_LAR=1.0, λ_GV=1.0, ρ=0.05, r_AWP_max=4 |
| Curvature | K=1000, percentile=0.8, β=0.9 |

---

## 二、核心結果(`checkpoints/best.pt`)

| Indicator | Acc (%) |
|---|---:|
| **ID Accuracy** | **85.37 %** |
| **Adversarial Accuracy** | **37.10 %** |
| **OOD Average** | **55.54 %** |
| **Harmonic Mean** | **52.94 %** |

---

## 三、CIFAR-100-C 各 corruption 結果(severity=5)

| Corruption | Acc (%) | Corruption | Acc (%) |
|---|---:|---|---:|
| brightness | **76.98** | jpeg_compression | 68.10 |
| frost | 70.83 | defocus_blur | 65.38 |
| pixelate | 70.32 | motion_blur | 62.67 |
| zoom_blur | 69.60 | glass_blur | 53.54 |
| snow | 69.42 | fog | 46.86 |
| elastic_transform | 68.39 | shot_noise | 37.55 |
| | | gaussian_noise | 35.14 |
| | | impulse_noise | 20.95 |
| | | **contrast** | **17.31** |

**觀察**:
- 最容易:brightness、frost、pixelate(70%+) — 模型對 photometric / structural 擾動穩
- 最弱:contrast(17.31%)、impulse_noise(20.95%) — 對極端對比度跟脈衝雜訊明顯崩
- 整體 noise 類別(gaussian / shot / impulse)平均只有 ~31%,是 OOD 表現的主要拖油瓶

---

## 四、Feature Geometry(Section 4 / Table 2)

| Metric | Value |
|---|---:|
| Cosine alignment(clean ↔ adv) | 0.4501 |
| LID(clean) | 13.98 |
| LID(adv) | 15.86 |
| ΔLID | 1.87 |

> adv 特徵的 LID 比 clean 高 ~13%,表示對抗樣本把特徵推到較高內在維度的區域;cosine alignment 0.45 顯示對抗特徵跟乾淨特徵還是有相當偏移(完美一致為 1.0)。

---

## 五、與論文的比較(Paper vs Ours)


| Indicator | Ours  | GRACE | Δ |
|---|---:|---:|---:|
| Clean / ID Acc | _____ % | 85.37 % | _____ |
| Robust / Adv Acc (PGD-10) | _____ % | 37.10 % | _____ |
| OOD Avg (CIFAR-100-C, sev=5) | _____ % | 55.54 % | _____ |
| 3-way Harmonic Mean | _____ % | 52.94 % | _____ |


## Citation

```bibtex
@article{chopra2026geometry,
  title={The Geometry of Robustness},
  author={Chopra, Shivang and Halbe, Shaunak and Huang, Chengyue and Maneechotesuwan, Brisa and Kira, Zsolt},
  journal={arXiv:2603.27139},
  year={2026}
}
```
