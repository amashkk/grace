# GRACE 複現結果

論文:_The Geometry of Robustness_ (arXiv:2603.27139, Chopra et al., 2026)
複現對象:Algorithm 1(page 16),CIFAR-100 + CLIP ViT-B/32 + LoRA(rank=64)

---

## 一、實驗設定(Table 10)

| 項目 | 設定 |
|---|---|
| Backbone | CLIP ViT-B/32(OpenAI) |
| 微調 | LoRA r=64,只動 visual encoder 的 q/k/v/out_proj |
| 資料集 | CIFAR-100(50k train / 10k test,resize 32→224) |
| 攻擊 | PGD-10,L∞,**ε=4/255**,**step=1/255** |
| Epochs / Batch / LR | 50 / 256 / 2e-4(AdamW + cosine) |
| GRACE 參數 | λ_LAR=1.0, λ_GV=1.0, ρ=0.05, r_AWP_max=4 |
| Curvature | K=1000, percentile=0.8, β=0.9 |

---

## 二、核心結果(`checkpoints/best.pt`)

| 指標 | 數值 |
|---|---:|
| **ID Accuracy**(CIFAR-100 test, 10000 張) | **85.37 %** |
| **Adversarial Accuracy**(PGD-10, ε=4/255) | **37.10 %** |
| **OOD Average**(CIFAR-100-C, severity=5, 15 corruptions) | **55.54 %** |
| **Harmonic Mean**(ID, OOD, Adv) | **52.94 %** |

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
- **最容易**:brightness、frost、pixelate(70%+) — 模型對 photometric / structural 擾動穩
- **最弱**:contrast(17.31%)、impulse_noise(20.95%) — 對極端對比度跟脈衝雜訊明顯崩
- 整體 noise 類別(gaussian / shot / impulse)平均只有 ~31%,是 OOD 表現的主要拖油瓶

---

## 四、與論文的比較(Paper vs Ours)

> ⚠️ 我這邊讀不了 PDF。請打開 `paper.pdf.pdf` 翻 **CIFAR-100 主結果表(Section 6)**,找 **GRACE on CLIP ViT-B/32** 那一列填入。

| 指標 | Paper(GRACE) | Ours | 差距 |
|---|---:|---:|---:|
| Clean / ID Acc | _____ % | 85.37 % | _____ |
| Robust / Adv Acc(PGD) | _____ % | 37.10 % | _____ |
| OOD Avg(CIFAR-100-C) | _____ % | 55.54 % | _____ |
| way Harmonic | _____ % | 52.94 % | _____ |

> 注意:論文測試端是 **AutoAttack(APGD-CE)**,我們的 Adv 用 PGD-10,通常 PGD ≥ AutoAttack 約 2–5%。要嚴格對齊論文,可加 `--autoattack` 重跑。
