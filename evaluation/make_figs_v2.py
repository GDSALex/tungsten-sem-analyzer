"""Figures of the condensed manuscript and of its supplementary material, for the adopted network.

    python make_figs_v2.py <arch> [all|dist|examples|failures]

ALMW examples use the out-of-fold maps of the network of the first repetition that never saw
their edge (arch_runs/<arch>/edge5_r1_f<k>/probs_test); ATW examples use the final model
(arch_runs/<arch>/all/probs_atw, written by eval_final.py maps).

Outputs (figures/v2):
    fig03_oof_distributions.png                       main text, distributions of the scores
    fig04_*, fig05_*                                  main text, error figures
    fig06_* ... fig09_*                               main text, segmentation figures
    sup_err_*, sup_seg_*, sup_failures.png            supplementary material
and results/v2/example_errors.csv, example_measurements.csv, failure_cases.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC   # noqa: E402
import eval_lib as EL     # noqa: E402
import evaluate as E      # noqa: E402
import fig_hybrid as FH   # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

OUT = RC.REV / "figures" / "v2"
RES = RC.RESULTS / "v2"
RUNS = RC.REV / "arch_runs"
TEST_OUT = os.environ.get("V2_TEST_OUT")     # test runs write figures and CSVs there instead
if TEST_OUT:
    OUT = Path(TEST_OUT)
RES_OUT = OUT if TEST_OUT else RES
EDGE = ("edge5_r1", "edge5_r2")
ERR = [("fig04_", "ALMW1-3a-600MW_05"), ("fig05_", "ALMW3-1a-1200MW_01"),
       ("sup_err_", "ALMW3-2d-1200MW-HT_13"), ("sup_err_", "ALMW1-1c-450MW_04")]
SEG_ALMW = [("fig06_", "ALMW1-1a-600MW-2p_06"), ("fig07_", "ALMW2-1a-600MW_2p_02"),
            ("sup_seg_", "ALMW2-1c-450MW_2p_04")]
SEG_ATW = [("fig08_", "ATW2-3a-600MW_09"), ("fig09_", "ATW3-3d_05"),
           ("sup_seg_", "ATW3-2c_01"), ("sup_seg_", "ATW3-4d_14")]


# ------------------------------------------------------------------ distributions (Fig. 3)
def distributions(df: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.titlesize": 7,
                         "axes.labelsize": 7, "savefig.dpi": 600})
    c_crater, c_crack = "#E8820C", "#7B3FA0"
    d = df[df.scheme.isin(EDGE) & (df.variant == "ms")]
    cr, ck = d[d.crater_ref.astype(bool)], d[d.crack_ref.astype(bool)]
    panels = [(cr, "c_iou", "Crater IoU (strict)", c_crater),
              (ck, "k_cl_P_5px", "Crack centreline precision, 5 px", c_crack),
              (ck, "k_cl_R_5px", "Crack centreline recall, 5 px", c_crack),
              (ck, "k_cl_F1_5px", "Crack centreline $F_1$, 5 px", c_crack)]
    fig, axes = plt.subplots(2, 2, figsize=(3.46, 3.0))
    bins = np.linspace(0, 1, 21)
    for ax, (t, col, title, colour) in zip(axes.ravel(), panels):
        x = t[col].dropna().clip(0, 1)
        ax.hist(x, bins=bins, color=colour, alpha=0.85, edgecolor="white", linewidth=0.4)
        m, md = x.mean(), x.median()
        ax.axvline(m, color="k", ls="--", lw=0.9)
        ax.axvline(md, color="k", ls=":", lw=1.1)
        ax.set_title(f"{title}\nmean {m:.2f}, median {md:.2f}, $n$ = {len(x)}", fontsize=6.4,
                     linespacing=1.15)
        ax.set_xlim(0, 1)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelsize=6, length=2, pad=1.5)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Predictions")
    for ax in axes[-1, :]:
        ax.set_xlabel("Score")
    fig.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.6)
    out = OUT / "fig03_oof_distributions.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("written", out.name, "| crater n", len(cr), "| crack n", len(ck))


# ------------------------------------------------------------------ examples (Figs 4-9, S2-S6)
def examples(arch: str) -> None:
    errs, meas = [], []
    for pre, name in ERR:
        (glob, tile), fold = FH.oof_maps(name, arch)
        r = FH.error_figure(name, glob, tile, OUT / f"{pre}{name}.png")
        errs.append(dict(r, figure=pre, fold=fold, model=f"edge5_r1 fold {fold}"))
        print("written", f"{pre}{name}.png", {k: round(v, 3) for k, v in r.items()
                                              if k in ("c_iou", "k_cl_F1_5px")})
    sc = E.scale_table()
    for pre, name in SEG_ALMW + SEG_ATW:
        if name.startswith("ALMW"):
            (glob, tile), fold = FH.oof_maps(name, arch)
            model = f"edge5_r1 fold {fold}"
        else:
            glob, tile = FH.load_maps(RUNS / arch / "all" / "probs_atw" / f"{name}.npz")
            fold, model = -1, "final model"
        rec = FH.segmentation_figure(name, glob, tile, OUT / f"{pre}{name}.png")
        row = dict(rec, figure=pre, fold=fold, model=model, kind="measured")
        meas.append(row)
        if name.startswith("ALMW"):     # analysis stage applied to the annotated masks
            ref = E.reference_measurements(name, float(sc[name]))
            meas.append({"name": name, "figure": pre, "fold": fold, "model": "reference",
                         "kind": "reference", "um_px": float(sc[name]),
                         "n_craters": ref["rp_n_craters"],
                         "total_area_um2": ref["rp_crater_total_area_um2"],
                         "largest_area_um2": ref["rp_crater_largest_area_um2"],
                         "largest_diam_um": ref["rp_crater_largest_diam_um"],
                         "n_cracks": ref["rp_n_cracks_rev"], "n_edge": ref["rp_n_edge_rev"],
                         "n_near_only": ref["rp_n_near_only_rev"],
                         "n_internal": ref["rp_n_internal_rev"],
                         "total_len_um": ref["rp_crack_len_rev_um"],
                         "longest_um": ref["rp_crack_longest_rev_um"],
                         "regular_cv": ref["rp_regular_cv_rev"],
                         "annot_len_um": ref["r_crack_len_um"]})
        print("written", f"{pre}{name}.png")
    pd.DataFrame(errs).to_csv(RES_OUT / "example_errors.csv", index=False)
    pd.DataFrame(meas).to_csv(RES_OUT / "example_measurements.csv", index=False)


# ------------------------------------------------------------------ failures (Fig. S7)
def overlays(name: str, glob, tile, tau: int = 5):
    """Crater and centreline error overlays of the network output, as in FH.error_figure."""
    roi, v = FH.roi_and_valid(name)
    H = roi.shape[0]
    shapes = RC.load_shapes(RC.MASK_DIR / f"{name}.json")
    ref_c, _ = RC.rasterise(shapes, H)
    ref_cl = RC.crack_centreline(shapes, H)
    mc, mk = (m.copy() for m in EL.variants(glob, tile)["ms"])
    mc[v:] = False
    mk[v:] = False
    err_c = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    err_c[ref_c & mc] = FH.TP_RGB
    err_c[~ref_c & mc] = FH.FP_RGB
    err_c[ref_c & ~mc] = FH.FN_RGB
    sk = EL.crack_skeleton(mk) if mk.any() else mk
    d_ref = ndi.distance_transform_edt(~ref_cl) if ref_cl.any() else np.full(mk.shape, np.inf)
    d_prd = ndi.distance_transform_edt(~sk) if sk.any() else np.full(mk.shape, np.inf)
    err_k = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    for m, col in ((ref_cl & (d_prd > tau), FH.FN_RGB), (sk & (d_ref > tau), FH.FP_RGB),
                   (sk & (d_ref <= tau), FH.TP_RGB)):
        if m.any():
            err_k[FH.binary_dilation(m, FH.disk(3))] = col
    return roi, err_c, err_k


def failures(arch: str, df: pd.DataFrame) -> None:
    n = df[(df.scheme == "edge5_r1") & (df.variant == "ms")].set_index("name")
    n = n.assign(missed_um=n.r_crack_len_um * (1 - n.k_cl_R_5px.fillna(0)),
                 unmatched_um=n.k_len_pred_px * (1 - n.k_cl_P_5px.fillna(0)) * n.um_px)
    cpos = n[n.crater_ref.astype(bool)]
    kneg = n[~n.crack_ref.astype(bool)]
    kpos = n[n.crack_ref.astype(bool)]
    # fixed criteria, ties broken by the size of the annotation, then by name
    cases = [(cpos.assign(a=-cpos.r_crater_total_area_um2).sort_values(["c_iou", "a"]).index[0],
              "lowest crater IoU among the images with annotated craters"),
             (kneg.m_crack_len_rev_um.sort_index().idxmax(),
              "longest crack length reported on an image without annotated cracks"),
             (kpos.missed_um.sort_index().idxmax(),
              "largest annotated crack length missed by the prediction"),
             (kpos.unmatched_um.sort_index().idxmax(),
              "largest predicted crack length with no annotation within 5 px")]
    assert kneg.m_crack_len_rev_um.max() > 0, "no crack reported on crack free images"
    plt.rcParams.update({"font.family": "serif", "font.size": 36})
    fig, axes = plt.subplots(4, 3, figsize=(33, 34))
    fig.patch.set_facecolor("white")
    FH.style_axes(axes.ravel(), lw=4.0)
    rows = []
    for i, (name, crit) in enumerate(cases):
        (glob, tile), fold = FH.oof_maps(name, arch)
        roi, err_c, err_k = overlays(name, glob, tile)
        r = n.loc[name]
        c_txt = (f"IoU: {r.c_iou:.3f}" if r.crater_ref else
                 ("false alarm" if r.crater_pred else "none annotated or predicted"))
        k_txt = (f"F1: {r.k_cl_F1_5px:.3f}" if r.crack_ref else
                 ("false alarm" if r.crack_pred else "none annotated or predicted"))
        H, W = roi.shape
        axes[i, 0].imshow(roi, cmap="gray", vmin=0, vmax=255)
        FH.scalebar(axes[i, 0], float(r.um_px), W, H, fontsize=32, lw=8)
        axes[i, 0].set_title(f"({chr(97 + i)}) {name}", fontsize=38, fontweight="bold")
        axes[i, 1].imshow(err_c)
        axes[i, 1].set_title(f"Craters — {c_txt}", fontsize=38, fontweight="bold")
        axes[i, 2].imshow(err_k)
        axes[i, 2].set_title(f"Cracks — {k_txt}", fontsize=38, fontweight="bold")
        rows.append({"panel": chr(97 + i), "name": name, "criterion": crit, "fold": fold,
                     "c_iou": r.c_iou, "k_cl_F1_5px": r.k_cl_F1_5px, "k_cl_P_5px": r.k_cl_P_5px,
                     "m_crack_len_rev_um": r.m_crack_len_rev_um, "r_crack_len_um": r.r_crack_len_um,
                     "missed_um": r.missed_um, "unmatched_um": r.unmatched_um,
                     "m_crater_largest_diam_um": r.m_crater_largest_diam_um,
                     "r_crater_largest_diam_um": r.r_crater_largest_diam_um,
                     "rp_crater_largest_diam_um": r.rp_crater_largest_diam_um})
    leg = [mpatches.Patch(color=np.array(FH.TP_RGB) / 255,
                          label="True positive pixels /\nmatched predicted centreline"),
           mpatches.Patch(color=np.array(FH.FP_RGB) / 255,
                          label="False positive pixels /\nunmatched predicted centreline"),
           mpatches.Patch(color=np.array(FH.FN_RGB) / 255,
                          label="False negative pixels /\nmissed reference centreline")]
    fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False, fontsize=34,
               handlelength=1.6, handleheight=1.6, columnspacing=2.5)
    plt.tight_layout(rect=[0, 0.06, 1, 1.0])
    out = OUT / "sup_failures.png"
    plt.savefig(str(out), dpi=110, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(RES_OUT / "failure_cases.csv", index=False)
    print(pd.DataFrame(rows).to_string())
    print("written", out.name)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    arch = sys.argv[1]
    what = sys.argv[2] if len(sys.argv) > 2 else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    recs = pd.read_csv(RES / "final_records.csv")
    if what in ("all", "dist"):
        distributions(recs)
    if what in ("all", "examples"):
        examples(arch)
    if what in ("all", "failures"):
        failures(arch, recs)
