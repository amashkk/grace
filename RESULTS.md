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
| 攻擊 | PGD-10,L∞,ε=4/255,step=1/255 |
| Epochs / Batch / LR | 50 / 256 / 2e-4(AdamW + cosine) |
| GRACE 參數 | λ_LAR=1.0, λ_GV=1.0, ρ=0.05, r_AWP_max=4 |
| Curvature | K=1000, percentile=0.8, β=0.9 |

---

## 二、評估結果(`checkpoints/best.pt`,CIFAR-100 test 完整 10000 張)

| 指標 | 數值 |
|---|---:|
| **ID Accuracy**(乾淨樣本) | **85.37 %** |
| **Adversarial Accuracy**(PGD-10, ε=4/255) | **37.10 %** |
| **Harmonic Mean**(2·ID·Adv / (ID+Adv)) | **51.72 %** |

---

## 三、與論文的比較(Paper vs Ours)

> ⚠️ 下表中「Paper」欄位請對照論文 Table 5 / Table 6(CIFAR-100, CLIP ViT-B/32, GRACE row)填入。
> 我這邊讀不了 PDF,需要你打開 `paper.pdf.pdf` 抄一下。

| 指標 | Paper(GRACE) | Ours | 差距 |
|---|---:|---:|---:|
| Clean / ID Acc | _____ % | 85.37 % | _____ |
| Robust / Adv Acc | _____ % | 37.10 % | _____ |
| Harmonic Mean | _____ % | 51.72 % | _____ |
| OOD(CIFAR-100-C avg, sev=5) | _____ % | _未跑_ | — |

---

## 四、PPT 一頁式摘要建議

**標題**:GRACE 複現 — CIFAR-100 / CLIP ViT-B/32

**三點 bullet**:
- 完整實作 Algorithm 1(PGD + LAR-AWP + Gram-Volume + Curvature-adaptive rank)
- LoRA r=64,只訓練 0.X M 參數(凍結 backbone)
- ID **85.37 %** / PGD-Adv **37.10 %** / Harmonic **51.72 %**

**圖**:
- 左:Algorithm 1 流程示意(從 README 那張對照表)
- 右:結果長條圖(三根:ID / Adv / Harmonic)

---

## 五、還沒做、可以補的東西

| 項目 | 怎麼補 | 估時 |
|---|---|---|
| AutoAttack 評估 | `py run.py --mode eval --checkpoint checkpoints/best.pt --autoattack` | 較久,看 GPU |
| OOD(CIFAR-100-C) | 從 https://zenodo.org/record/3555552 下載到 `data/CIFAR-100-C/`,再 `--mode eval_full` | +30 min eval |
| Feature geometry(cosine align, ΔLID) | `eval_full` 會一起算 | 含在上面 |
| 訓練曲線圖 | `train.log` 沒寫到逐 epoch,需要重跑訓練並把 stdout 全部 redirect | 重跑 ~很久 |
