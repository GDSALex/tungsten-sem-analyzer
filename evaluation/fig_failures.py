"""Representative failures of the out-of-fold network output (reviewer specific comment 4).

Four cases chosen by fixed criteria on the first repetition of the edge-grouped scheme
(configuration A, network output, thresholds 0.50 / 0.25), never by looking at the images:
  1. the lowest crater IoU among the images with annotated craters;
  2. the longest crack length reported by the analysis stage on an image without
     annotated cracks;
  3. the longest annotated crack length among the images where the centreline F1 is zero
     and nothing is predicted;
  4. the longest predicted centreline among the images where the centreline F1 is zero
     although cracks are both annotated and predicted.
Output: figures/figF_failures.png and results/failure_cases.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402
import eval_lib as EL     # noqa: E402
import fig_examples as FX  # noqa: E402  (helpers only, its main() is guarded)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


def pick_cases():
    cv = pd.read_csv(RC.RESULTS / "cv_records.csv")
    n = cv[(cv.config == "A") & (cv.scheme == "edge5_r1") & (cv.variant == "net")].set_index("name")
    c1 = n[n.crater_ref > 0]["c_iou"].idxmin()
    c2 = n[(n.crack_ref == 0)]["m_crack_len_rev_um"].idxmax()
    miss = n[(n.crack_ref > 0) & (n.crack_pred == 0)]
    c3 = miss["r_crack_len_um"].idxmax()
    wrong = n[(n.crack_ref > 0) & (n.crack_pred > 0) & (n.k_cl_F1_5px == 0)]
    c4 = wrong["k_len_pred_px"].idxmax()
    return [(c1, "lowest crater IoU"), (c2, "longest crack reported on a crack free image"),
            (c3, "longest annotated crack entirely missed"),
            (c4, "longest predicted crack with no match")], n


def panels(name, tau=5.0):
    run, fold = FX.oof_run(name)
    glob, _ = FX.maps_for(name, str(run))
    g, roi_h, v = FX.context(name)
    um = FX.SCALE[name]
    shapes = RC.load_shapes(RC.MASK_DIR / f"{name}.json")
    ref_c, ref_k = RC.rasterise(shapes, roi_h)
    ref_c = ref_c[:v]
    ref_cl = RC.crack_centreline(shapes, roi_h)[:v]
    roi = g[:v]
    mc, mk = glob[0][:v] > 0.5, glob[1][:v] > 0.25
    sk = skeletonize(mk) if mk.any() else mk
    d_ref = ndi.distance_transform_edt(~ref_cl) if ref_cl.any() else np.full(mk.shape, np.inf)
    d_pred = ndi.distance_transform_edt(~sk) if sk.any() else np.full(mk.shape, np.inf)
    crater_rgb = FX.blend(roi, [(ref_c & mc, FX.GREEN, 0.75), (~ref_c & mc, FX.RED, 0.75),
                                (ref_c & ~mc, FX.BLUE, 0.75)])
    crack_rgb = FX.blend(roi, [(FX.thick(sk & (d_ref <= tau), 1), FX.GREEN, 1.0),
                               (FX.thick(sk & (d_ref > tau), 1), FX.RED, 1.0),
                               (FX.thick(ref_cl & (d_pred > tau), 1), FX.BLUE, 1.0)])
    return roi, crater_rgb, crack_rgb, um, fold


def main():
    cases, n = pick_cases()
    fig = plt.figure(figsize=(7.1, 7.4))
    gs = fig.add_gridspec(len(cases), 3, wspace=0.04, hspace=0.32,
                          left=0.005, right=0.995, top=0.955, bottom=0.05)
    rows = []
    for i, (name, crit) in enumerate(cases):
        roi, crgb, krgb, um, fold = panels(name)
        H, W = roi.shape
        r = n.loc[name]
        c_txt = (f"IoU {r.c_iou:.2f}" if r.crater_ref > 0 else
                 ("false alarm" if r.crater_pred > 0 else "no reference, no prediction"))
        if r.crack_ref > 0:
            k_txt = f"centreline F1 {r.k_cl_F1_5px:.2f} at 5 px"
        else:
            k_txt = "false alarm" if r.crack_pred > 0 else "no reference, no prediction"
        for j, (img, title) in enumerate(((roi, f"{chr(97 + i)}) {name}"),
                                          (crgb, f"Craters, {c_txt}"), (krgb, f"Cracks, {k_txt}"))):
            a = fig.add_subplot(gs[i, j])
            a.imshow(img, cmap="gray" if j == 0 else None, vmin=0, vmax=255)
            a.set_xticks([]); a.set_yticks([])
            a.set_title(title, fontsize=6.6)
            if j == 0:
                FX.scalebar(a, um, W, H)
        rows.append({"panel": chr(97 + i), "name": name, "criterion": crit, "fold": fold + 1,
                     "c_iou": r.c_iou, "k_cl_F1_5px": r.k_cl_F1_5px,
                     "m_crack_len_rev_um": r.m_crack_len_rev_um, "r_crack_len_um": r.r_crack_len_um,
                     "m_crater_largest_diam_um": r.m_crater_largest_diam_um,
                     "r_crater_largest_diam_um": r.r_crater_largest_diam_um})
    handles = [Patch(color=FX.GREEN, label="TP pixels / matched predicted centreline"),
               Patch(color=FX.RED, label="FP pixels / unmatched predicted centreline"),
               Patch(color=FX.BLUE, label="FN pixels / missed reference centreline")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.4,
               handlelength=1.4, columnspacing=1.2)
    out = RC.FIGURES / "figF_failures.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(RC.RESULTS / "failure_cases.csv", index=False)
    print(pd.DataFrame(rows).to_string())
    print("written", out)


if __name__ == "__main__":
    main()
