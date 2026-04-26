"""
畫 PPT 用的長條圖。請在 Windows Python 環境跑:
    py results/make_chart.py
產出 results/results_bar.png
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
r = metrics["results"]

labels = ["ID Acc\n(Clean)", "Adv Acc\n(PGD-10, 4/255)", "Harmonic\nMean"]
values = [r["id_accuracy"], r["adversarial_accuracy_pgd"], r["harmonic_mean"]]
colors = ["#4C9AFF", "#FF6B6B", "#7C5CFF"]

fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="black", linewidth=0.6)

for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.2f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_ylim(0, 100)
ax.set_title("GRACE Reproduction — CIFAR-100 / CLIP ViT-B/32 + LoRA",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
out = HERE / "results_bar.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
