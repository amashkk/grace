# GRACE Implementation Plan

Paper: "The Geometry of Robustness" (arXiv:2603.27139)
Algorithm 1 (page 16) is the authoritative reference for the training loop.

## Module → Equation/Section Mapping

### `config.py` — Hyperparameters (Table 10, p.18)
- r_W = 64 (LoRA rank)
- r_AWP_max = 4 (max AWP perturbation rank)
- ε = 4/255, step_size = 1/255 (PGD, 10 steps)
- lr = 2e-4
- φ_AWP = 80 (gradient threshold percentile)
- epochs = 50, batch_size = 256
- K = 1000 (curvature update interval, Appendix E.1)
- β ∈ [0.85, 0.95] (EMA decay, Eq. 29)
- λ_LAR, λ_GV (tradeoff params, Eq. 4)

### `model.py` — CLIP + PEFT LoRA (Section 5.2, Eq. 5, Section F.3-F.4)
- Load CLIP ViT-B/32 from OpenAI via HuggingFace transformers
- Apply LoRA via PEFT library: W(θ) = W(θ₀) + B_W · A_W (Eq. 5)
- target_modules: q_proj, k_proj, v_proj, out_proj on visual encoder
- Zero-shot classifier from text prompts (Eq. 12-14, Appendix B.1)
- Feature extraction: penultimate-layer L2-normalized embeddings

### `curvature.py` — Gauss-Newton Curvature Proxy (Appendix E.1, Eq. 28-29)
- Eq. 28: ĥ_W = n_v · (g_W ⊙ g_W) where g_W = ∇_W (1/n_v Σ L)
- Eq. 29: h_W^(t) = β · h_W^(t-1) + (1-β) · ĥ_W^(t), β ∈ [0.85, 0.95]
- Normalize: h_norm_W = (1/|W|) · Σ ĥ_W
- Rank allocation: r_AWP ∝ 1{h_norm_W ≥ τ_p}, τ_p = quantile({ĥ_W}, p=0.8)
- Sharp layers (top 20%) → highest rank; flat layers (bottom 20%) → 0 rank
- Executed every K=1000 iterations (Alg 1, lines 16-19)

### `losses.py` — PGD, LAR-AWP, Gram-Volume (Algorithm 1 Steps 1-3)
- **PGD attack** (Alg 1 line 6, Eq. 19): 10-step PGD, L∞, ε=4/255, α=1/255
- **LAR-AWP inner max** (Alg 1 lines 8-12, Eq. 6-7):
  - Eq. 6: W_pert(θ,Δ) = W(θ₀) + B_W·A_W + B_AWP·A_AWP
  - Gradient ascent on (A_AWP, B_AWP) for T_AWP steps
  - Project Δ onto ‖Δ‖ ≤ ρ with layerwise rank masks
  - L_LAR-AWP = (1/n) Σ max L(F_{W_pert}(x_adv), y) (Eq. 7)
- **Gram-Volume** (Alg 1 lines 14-15, Eq. 8-9):
  - Eq. 8: G_i = [⟨f_j, f_k⟩] for j,k ∈ {ID, Adv, AWP} + εI
  - Eq. 9: L_GV = √|det(G_i)|
- **Combined loss** (Alg 1 line 20, Eq. 4):
  - L_GRACE = L_task + λ_LAR · L_LAR-AWP + λ_GV · L_GV

### `train.py` — Training Loop (Algorithm 1 exactly)
- Line 1-2: Initialize LoRA params, freeze backbone, init AWP ranks=0, h_W=0
- Line 3-4: For each iteration, sample minibatch
- Line 5: Clean forward → f_ID, L_task
- Line 6-7: PGD attack → x_Adv, f_Adv (Step 1)
- Lines 8-13: LAR-AWP inner max → f_AWP (Step 2)
- Lines 14-15: Gram-Volume loss L_GV (Step 3)
- Lines 16-19: Periodic curvature update every K=1000 iters (Step 4)
- Line 20: Outer update: minimize L_GRACE (Step 5)

### `data.py` — CIFAR-100 Data Loading
- CIFAR-100 train/test (auto-download)
- CIFAR-100-C corruptions for OOD eval
- Resize 32→224 for CLIP compatibility

### `evaluate.py` — Evaluation
- ID accuracy (CIFAR-100 test)
- Adversarial accuracy (PGD + AutoAttack APGD-CE at ε=4/255)
- OOD accuracy (CIFAR-100-C corruptions)
- Feature geometry (cosine alignment, LID)

### `main.py` — CLI Entry Point
- argparse with all Table 10 defaults
- train / eval / eval_full modes

## Algorithm 1 Step-by-Step Cross-Check (VERIFIED)

| Alg 1 Line | Description | File | Function/Line | Eq |
|------------|-------------|------|---------------|-----|
| Require | Load CLIP ViT-B/32, apply PEFT LoRA, freeze | model.py | `GRACEModel.__init__` | Eq. 5 |
| 1 | Init LoRA Θ={A_W,B_W}, freeze backbone | model.py:62-73 | `get_peft_model()` | Eq. 5 |
| 2 | Init r_AWP^(W)←0, h_W←0 | curvature.py:51-55 | `CurvatureEstimator.__init__` | — |
| 3-4 | For each iter, sample batch | train.py:77-80 | `train_one_epoch` loop | — |
| 5 | f_ID = encode(x), L_task = CE | losses.py:220-222 | `compute_grace_loss` | — |
| 6 | PGD → x_Adv | losses.py:225-229 | `pgd_attack()` | Eq. 19 |
| 7 | f_Adv = encode(x_Adv) | losses.py:232 | `compute_grace_loss` | — |
| 8-12 | LAR-AWP inner max: grad ascent on (A_AWP,B_AWP), project ‖Δ‖≤ρ | losses.py:84-174 | `lar_awp_inner_max` | Eq. 6-7 |
| 13 | f_AWP via W_pert(θ,Δ) | losses.py:160-163 | final forward in `lar_awp_inner_max` | Eq. 6 |
| 14 | Gram matrix G_i from (f_ID, f_Adv, f_AWP) | losses.py:195-199 | `gram_volume_loss` | Eq. 8 |
| 15 | L_GV = √|det(G_i)| | losses.py:202 | `gram_volume_loss` | Eq. 9 |
| 16-17 | if iter mod K=0: estimate ĥ_W = n_v·(g_W⊙g_W) | curvature.py:68-85 | `update_curvature` | Eq. 28 |
| 17 cont | EMA: h_W = β·h_W + (1-β)·ĥ_W | curvature.py:87-90 | `update_curvature` | Eq. 29 |
| 18 | Map {h_W} → percentile → r_AWP | curvature.py:97-116 | `_update_rank_allocation` | App E.1 |
| 19 | end if | train.py:85 | condition check | — |
| 20 | Update Θ: minimize L_GRACE = L_task + λ_LAR·L_LAR-AWP + λ_GV·L_GV | train.py:95-99 | `backward()` + `optimizer.step()` | Eq. 4 |

### Variable Name Verification
- `f_ID`, `f_Adv`, `f_AWP`: ✅ losses.py lines 220, 232, 235
- `G_i`: ✅ losses.py:199
- `L_GV`: ✅ losses.py:202
- `L_task`, `L_LAR_AWP`, `L_GRACE`: ✅ losses.py:222, 235, 242
- `h_W`, `r_AWP`: ✅ curvature.py:52, 55
- `A_AWP`, `B_AWP`: ✅ losses.py:124-125
- `x_Adv`: ✅ losses.py:225
- `lambda_LAR`, `lambda_GV`: ✅ config.py:37-38
- `rho` (ρ): ✅ config.py:28, losses.py:103
