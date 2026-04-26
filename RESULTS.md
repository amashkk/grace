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

