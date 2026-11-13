"""Revised Figure 3: training log of the submitted model plus the common loss.

Logged curves come from Code/checkpoints/training_history.json (training loss averaged
over augmented batches, selection loss and micro-averaged IoU / F1 on the 14 selection
images at thresholds 0.50 / 0.25).  Markers: common loss recomputed with the saved
checkpoints (results/common_loss_published.csv).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.titlesize": 7,
                     "axes.labelsize": 7, "legend.fontsize": 6.2, "savefig.dpi": 600})
C_CRATER, C_CRACK, C_MEAN = "#E8820C", "#7B3FA0", "#2E9E44"
C_TRAIN, C_VAL = "#1F5FA8", "#C0392B"

h = pd.DataFrame(json.load(open(RC.CKPT_DIR / "training_history.json")))
cl = pd.read_csv(RC.RESULTS / "common_loss_published.csv")
best = 125

fig, axes = plt.subplots(3, 1, figsize=(3.46, 4.6), sharex=True,
                         gridspec_kw={"hspace": 0.12, "left": 0.14, "right": 0.98,
                                      "top": 0.985, "bottom": 0.08})
ax = axes[0]
ax.plot(h.epoch, h.train_loss, color=C_TRAIN, lw=1.0, label="training loss (logged, augmented)")
ax.plot(h.epoch, h.val_loss, color=C_VAL, lw=1.0, label="selection loss (logged)")
ax.plot(cl.epoch, cl.train_common_noaug, "o", ms=3.2, mfc="white", mec=C_TRAIN, mew=0.9,
        label="common loss, 85 training images")
ax.plot(cl.epoch, cl.val_common, "s", ms=3.0, mfc="white", mec=C_VAL, mew=0.9,
        label="common loss, 14 selection images")
ax.set_ylabel("Loss")
ax.legend(loc="upper right", frameon=False, handlelength=1.8, labelspacing=0.25)

for ax, key, lab in ((axes[1], "iou", "IoU"), (axes[2], "f1", "F1")):
    ax.plot(h.epoch, h[f"crater_{key}"], color=C_CRATER, lw=1.0, label="craters")
    ax.plot(h.epoch, h[f"crack_{key}"], color=C_CRACK, lw=1.0, label="cracks")
    ax.plot(h.epoch, h[f"mean_{key}"], color=C_MEAN, lw=1.0, label="mean")
    ax.set_ylabel(f"{lab}, selection images")
    ax.set_ylim(0, 1)
axes[1].legend(loc="lower right", frameon=False, ncol=3, handlelength=1.6, columnspacing=1.0)
for ax in axes:
    ax.axvline(15.5, color="#D4AC0D", ls="--", lw=0.9)
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_xlim(0, 157)
row = h[h.epoch == best].iloc[0]
axes[1].plot(best, row.mean_iou, marker="*", ms=8, color="#E91E8C", mec="k", mew=0.3, zorder=5)
axes[2].plot(best, row.mean_f1, marker="*", ms=8, color="#E91E8C", mec="k", mew=0.3, zorder=5)
axes[0].plot(best, row.val_loss, marker="*", ms=8, color="#E91E8C", mec="k", mew=0.3, zorder=5)
axes[2].set_xlabel("Epoch")
axes[0].text(16.5, axes[0].get_ylim()[1] * 0.97, "encoder unfrozen", fontsize=5.8, va="top",
             color="#9A7D0A")
out = RC.FIGURES / "fig03_convergence.png"
fig.savefig(out, facecolor="white")
print("written", out)
