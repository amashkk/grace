"""
畫 PPT 用的圖。請在 Windows Python 環境跑:
    py results/make_chart.py
產出:
  results/results_3way.png   — ID / OOD / Adv 三方長條圖
  results/results_ood.png    — CIFAR-100-C 15 corruption 橫條圖
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
r = metrics["results"]
ood = metrics["ood_per_corruption_severity5"]


# ---- Figure 1: 3-way bar chart ----
labels = ["ID Acc\n(Clean)", "OOD Avg\n(CIFAR-100-C, sev=5)", "Adv Acc\n(PGD-10, 4/255)"]
values = [r["id_accuracy"], r["ood_avg_severity5"], r["adversarial_accuracy_pgd"]]
colors = ["#4C9AFF", "#36B37E", "#FF6B6B"]

fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
bars = ax.bar(labels, values, color=colors, width=0.55,
              edgecolor="black", linewidth=0.6)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.2f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_ylim(0, 100)
ax.set_title(f"GRACE Reproduction on CIFAR-100  (3-way Harmonic Mean = {r['harmonic_mean_3way']:.2f}%)",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
out1 = HERE / "results_3way.png"
plt.savefig(out1, dpi=200, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close()


# ---- Figure 2: OOD per-corruption horizontal bar ----
items = sorted(ood.items(), key=lambda x: x[1])  # ascending — worst at top of chart
names = [k for k, _ in items]
vals = [v for _, v in items]


def color_for(v):
    if v >= 60: return "#36B37E"   # green
    if v >= 40: return "#FFAB00"   # amber
    return "#FF5630"               # red


cols = [color_for(v) for v in vals]

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
bars = ax.barh(names, vals, color=cols, edgecolor="black", linewidth=0.4)
for b, v in zip(bars, vals):
    ax.text(v + 0.7, b.get_y() + b.get_height()/2, f"{v:.1f}%",
            va="center", fontsize=10)

ax.axvline(r["ood_avg_severity5"], color="black", linestyle="--",
           linewidth=1, label=f"avg = {r['ood_avg_severity5']:.2f}%")
ax.legend(loc="lower right", fontsize=10)
ax.set_xlabel("Accuracy (%)", fontsize=12)
ax.set_xlim(0, 90)
ax.set_title("CIFAR-100-C  (severity=5,  sorted worst → best)",
             fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
out2 = HERE / "results_ood.png"
plt.savefig(out2, dpi=200, bbox_inches="tight")
print(f"Saved: {out2}")
