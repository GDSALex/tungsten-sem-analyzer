"""Figure 4: distributions of the out-of-fold per-image scores of the network output.

Configuration A, fixed thresholds 0.50 / 0.25, both repetitions of the edge-grouped
cross-validation pooled (two predictions per image).  Crater panels use the images with
annotated craters and crack panels the images with annotated cracks, failures kept as
zeros.  Centreline precision is undefined when nothing is predicted, so its panel has
fewer values.  20 equal-width bins in [0, 1]; dashed line = mean, dotted line = median.
Output: figures/fig04_oof_distributions.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.titlesize": 7,
                     "axes.labelsize": 7, "legend.fontsize": 6.2, "savefig.dpi": 600})
C_CRATER, C_CRACK = "#E8820C", "#7B3FA0"

cv = pd.read_csv(RC.RESULTS / "cv_records.csv")
d = cv[(cv.config == "A") & cv.scheme.isin(["edge5_r1", "edge5_r2"]) & (cv.variant == "net")]
cr = d[d.crater_ref > 0]
ck = d[d.crack_ref > 0]
PANELS = [
    (cr, "c_iou", "Crater IoU (strict)", C_CRATER),
    (cr, "c_f1", "Crater F1 (strict)", C_CRATER),
    (ck, "k_strict_iou", "Crack band IoU (strict)", C_CRACK),
    (ck, "k_cl_P_5px", "Crack centreline precision, 5 px", C_CRACK),
    (ck, "k_cl_R_5px", "Crack centreline recall, 5 px", C_CRACK),
    (ck, "k_cl_F1_5px", "Crack centreline F1, 5 px", C_CRACK),
]

fig, axes = plt.subplots(3, 2, figsize=(3.46, 4.3))
bins = np.linspace(0, 1, 21)
for ax, (df, col, title, colour) in zip(axes.ravel(), PANELS):
    x = df[col].dropna().clip(0, 1)
    ax.hist(x, bins=bins, color=colour, alpha=0.85, edgecolor="white", linewidth=0.4)
    m, md = x.mean(), x.median()
    ax.axvline(m, color="k", ls="--", lw=0.9)
    ax.axvline(md, color="k", ls=":", lw=1.1)
    ax.set_title(f"{title}\nmean {m:.2f}, median {md:.2f}, $n$ = {len(x)}", fontsize=6.2, linespacing=1.15)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=6, length=2, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)
for ax in axes[:, 0]:
    ax.set_ylabel("Predictions")
for ax in axes[-1, :]:
    ax.set_xlabel("Score")
fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.6)
out = RC.FIGURES / "fig04_oof_distributions.png"
fig.savefig(out, facecolor="white")
plt.close(fig)
print("written", out, "| crater n", len(cr), "| crack n", len(ck),
      "| precision n", int(ck["k_cl_P_5px"].notna().sum()))
