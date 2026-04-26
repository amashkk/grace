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

| Indicator | Acc (%) |
|---|---:|
| **ID Accuracy** | **85.37 %** |
| **Adversarial Accuracy** (PGD-10, ε=4/255) | **37.10 %** |
| **OOD Average** (CIFAR-100-C, severity=5) | **55.54 %** |
| **Harmonic Mean** (3-way: ID / OOD / Adv) | **52.94 %** |

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

## 四、Feature Geometry(Section 4 / Table 2)

| Metric | Value |
|---|---:|
| Cosine alignment(clean ↔ adv 特徵餘弦相似度) | 0.4501 |
| LID(clean) | 13.98 |
| LID(adv) | 15.86 |
| ΔLID | 1.87 |

> 解讀:adv 特徵的 LID 比 clean 高 ~13%,表示對抗樣本把特徵推到較高內在維度的區域;cosine alignment 0.45 顯示對抗特徵跟乾淨特徵還是有相當偏移(完美一致為 1.0)。

---

## 五、與論文的比較(Paper vs Ours)

> ⚠️ 我這邊讀不了 PDF。請打開 `paper.pdf.pdf` 翻 **CIFAR-100 主結果表(Section 6)**,找 **GRACE on CLIP ViT-B/32** 那一列填入「Paper」欄。

| Indicator | Paper (GRACE) | Ours | Δ |
|---|---:|---:|---:|
| Clean / ID Acc | _____ % | 85.37 % | _____ |
| Robust / Adv Acc (PGD-10) | _____ % | 37.10 % | _____ |
| OOD Avg (CIFAR-100-C, sev=5) | _____ % | 55.54 % | _____ |
| 3-way Harmonic Mean | _____ % | 52.94 % | _____ |

> 注意:論文測試端是 **AutoAttack(APGD-CE)**,我們的 Adv 用 PGD-10,通常 PGD ≥ AutoAttack 約 2–5%。要嚴格對齊論文,可加 `--autoattack` 重跑。

---

## 六、PPT 一頁式摘要建議

**標題**:GRACE Reproduction — CIFAR-100 / CLIP ViT-B/32

**三點 bullet**:
- Full implementation of Algorithm 1 (PGD + LAR-AWP + Gram-Volume + Curvature-adaptive rank)
- LoRA r=64 on the visual encoder, backbone frozen
- Hits all three axes: **ID 85.37 %**, **OOD 55.54 %**, **Adv 37.10 %**, 3-way harmonic **52.94 %**

**圖建議(優先序)**:
1. `results_4metrics.png` — 4 根長條圖(ID / OOD / Adv / Harmonic),**主視覺**
2. `results_ood.png` — CIFAR-100-C 15 corruption 排序橫條圖

---

## 七、還沒做、可選補

| 項目 | 怎麼補 |
|---|---|
| AutoAttack(嚴格對齊論文) | `pip install autoattack` 後 `py run.py --mode eval --checkpoint checkpoints/best.pt --autoattack` |
| 訓練曲線圖 | `train.log` 沒寫到逐 epoch,要重跑訓練並 redirect stdout |
