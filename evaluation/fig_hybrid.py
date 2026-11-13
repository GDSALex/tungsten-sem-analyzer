"""Example figures of the revision in the format of the first version.

The layout, fonts, colours and legends are those of the authors' code
(validation_figures.visualize_one for the error figures and
inference.fig_original_vs_segmentation for the segmentation figures).  What changes is the
content:
  * the predictions come from a network that never saw the image (ALMW, grouped
    cross-validation) or from the final model (ATW);
  * craters are scored strictly, cracks with the length-weighted centreline protocol
    (buffer 5 px), and the crack error panel shows centrelines;
  * the SEM panels carry a scale bar;
  * the measurement box gives the revised values and the size category of the largest
    crater like region instead of a severity grade;
  * no edge row line is drawn.

    python fig_hybrid.py sample <arch> <out_dir>      # style sample: one figure of each kind
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402
import eval_lib as EL    # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage.morphology import binary_dilation, disk  # noqa: E402

from tungsten_analyzer import inference as INF  # noqa: E402

RUNS = RC.REV / "arch_runs"
SCALE = pd.read_csv(RC.RESULTS / "scale_verified.csv").set_index("name")["um_px"].to_dict()
MAN = RC.load_manifest().set_index("name")
SIZE_CAT = {0: "none", 1: "< 100 µm", 2: "100-600 µm", 3: "600-950 µm", 4: "> 950 µm"}

# colours of the authors' code
TP_RGB, FP_RGB, FN_RGB = (0, 220, 0), (220, 0, 0), (0, 80, 255)      # make_error_overlay
CRATER_BGR, CRACK_BGR = INF.C_CRATER_VIS, INF.C_CRACK_VIS              # orange, red
INTERNAL_BGR = (0, 215, 255)                                           # yellow
NEAR_BGR = (217, 26, 217)                                              # magenta
GREY_BGR = (110, 110, 110)
SEG_TITLE = "U-Net Segmentation"     # changed if another network family is adopted


# ----------------------------------------------------------------------------- data
def image_path(name: str) -> Path:
    p = RC.IMG_DIR / f"{name}.jpg"
    return p if p.exists() else RC.ATW_DIR / f"{name}.jpg"


def roi_and_valid(name: str):
    g = RC.read_gray(image_path(name))
    H = g.shape[0]
    roi_h = H - RC.strip_rows(H)
    return g[:roi_h], min(roi_h, RC.bar_top_row(g))


def load_maps(path: Path):
    z = np.load(path)
    glob = z["glob"].astype(np.float32)
    tile = z["tile"].astype(np.float32) if bool(z["has_tile"]) else None
    return glob, tile


def oof_maps(name: str, arch: str, scheme: str = "edge5_r1"):
    fold = int(MAN.loc[name, f"fold_{scheme}"])
    return load_maps(RUNS / arch / f"{scheme}_f{fold}" / "probs_test" / f"{name}.npz"), fold


def fused(glob, tile):
    """Probabilities of the 'ms' inference variant (global and tiled passes fused)."""
    if tile is None:
        return glob[0], glob[1]
    return np.maximum(glob[0], 0.7 * tile[0]), np.maximum(glob[1], tile[1])


# -------------------------------------------------------------------------- helpers
def scalebar(ax, um_px: float, W: int, H: int, fontsize: float, lw: float):
    target = 0.2 * W * um_px
    nice = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    L = max([v for v in nice if v <= target] or [1])
    x0, y0 = 0.04 * W, 0.92 * H
    ax.plot([x0, x0 + L / um_px], [y0, y0], color="white", lw=lw, solid_capstyle="butt",
            path_effects=[pe.Stroke(linewidth=lw + 3, foreground="black"), pe.Normal()])
    ax.text(x0 + L / um_px / 2, y0 - 0.035 * H, f"{L:g} µm", color="white", ha="center",
            va="bottom", fontsize=fontsize, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=3, foreground="black")])


def style_axes(axes, lw=3.0):
    """Image panels with the black frame of the revised figures."""
    for ax in axes:
        ax.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("black")
            sp.set_linewidth(lw)


# ------------------------------------------------------------ error figure (Figs 5-7)
def error_figure(name: str, glob, tile, out_png: Path, tau: int = 5, variant: str = "ms"):
    """visualize_one of the authors' code with revised content."""
    roi, v = roi_and_valid(name)
    H, W = roi.shape
    um = SCALE[name]
    shapes = RC.load_shapes(RC.MASK_DIR / f"{name}.json")
    ref_c, ref_k = RC.rasterise(shapes, H)
    ref_cl = RC.crack_centreline(shapes, H)
    masks = EL.variants(glob, tile)[variant]
    pc_map, pk_map = fused(glob, tile) if variant != "net" else (glob[0], glob[1])
    mc, mk = masks[0].copy(), masks[1].copy()
    mc[v:] = False
    mk[v:] = False
    cs = EL.crater_scores(ref_c[:v], mc[:v])
    ks = EL.crack_scores(ref_k[:v], ref_cl[:v], mk[:v], taus_px=(tau,), um_px=None)

    # crater errors: pixels, as make_error_overlay
    err_c = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    err_c[ref_c & mc] = TP_RGB
    err_c[~ref_c & mc] = FP_RGB
    err_c[ref_c & ~mc] = FN_RGB
    # crack errors: centrelines (revised skeleton of the prediction against the polylines)
    sk = EL.crack_skeleton(mk) if mk.any() else mk
    d_ref = ndi.distance_transform_edt(~ref_cl) if ref_cl.any() else np.full(mk.shape, np.inf)
    d_prd = ndi.distance_transform_edt(~sk) if sk.any() else np.full(mk.shape, np.inf)
    th = disk(3)
    err_k = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    for m, col in ((ref_cl & (d_prd > tau), FN_RGB), (sk & (d_ref > tau), FP_RGB),
                   (sk & (d_ref <= tau), TP_RGB)):
        if m.any():
            err_k[binary_dilation(m, th)] = col

    plt.rcParams.update({"font.family": "serif", "font.size": 40})
    fig, axes = plt.subplots(2, 4, figsize=(44, 22))
    fig.patch.set_facecolor("white")
    style_axes(axes.ravel(), lw=4.0)

    for row in (0, 1):
        axes[row, 0].imshow(roi, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title("SEM Image", color="#222222", fontsize=46, fontweight="bold")
        scalebar(axes[row, 0], um, W, H, fontsize=40, lw=9)

    gt_c = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    gt_c[ref_c] = [255, 140, 0]
    axes[0, 1].imshow(gt_c)
    axes[0, 1].set_title("Reference — Craters", color="#CC5500", fontsize=46, fontweight="bold")
    im_c = axes[0, 2].imshow(pc_map, cmap="hot", vmin=0, vmax=1)
    cb = fig.colorbar(im_c, ax=axes[0, 2], fraction=0.046, pad=0.04, shrink=0.85)
    cb.set_label("P(crater)", fontsize=40, color="black")
    cb.ax.tick_params(labelsize=34)
    axes[0, 2].set_title("Crater Probability Map", color="#222222", fontsize=46, fontweight="bold")
    axes[0, 3].imshow(err_c)
    if cs["crater_ref"]:
        t = f"IoU: {cs['c_iou']:.3f}  |  F1: {cs['c_f1']:.3f}"
    else:
        t = "false alarm" if cs["crater_pred"] else "no crater, none predicted"
    axes[0, 3].set_title(f"Detection Errors — Craters\n{t}", color="#222222", fontsize=42,
                         fontweight="bold")

    gt_k = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB).astype(float)
    gt_k[ref_k] = 0.55 * gt_k[ref_k] + 0.45 * np.array([210, 0, 0])
    gt_k = gt_k.astype(np.uint8)
    gt_k[binary_dilation(ref_cl, disk(2))] = [210, 0, 0]
    axes[1, 1].imshow(gt_k)
    axes[1, 1].set_title("Reference — Cracks", color="#AA0000", fontsize=46, fontweight="bold")
    im_k = axes[1, 2].imshow(pk_map, cmap="hot", vmin=0, vmax=1)
    cb = fig.colorbar(im_k, ax=axes[1, 2], fraction=0.046, pad=0.04, shrink=0.85)
    cb.set_label("P(crack)", fontsize=40, color="black")
    cb.ax.tick_params(labelsize=34)
    axes[1, 2].set_title("Crack Probability Map", color="#222222", fontsize=46, fontweight="bold")
    axes[1, 3].imshow(err_k)
    if ks["crack_ref"]:
        t = (f"F1: {ks[f'k_cl_F1_{tau}px']:.3f}  |  P: {ks[f'k_cl_P_{tau}px']:.2f}"
             f"  |  R: {ks[f'k_cl_R_{tau}px']:.2f}")
    else:
        t = "false alarm" if ks["crack_pred"] else "no crack, none predicted"
    axes[1, 3].set_title(f"Detection Errors — Cracks\n{t}", color="#222222", fontsize=42,
                         fontweight="bold")

    leg = [mpatches.Patch(color=np.array(TP_RGB) / 255,
                          label="True positive pixels /\nmatched predicted centreline"),
           mpatches.Patch(color=np.array(FP_RGB) / 255,
                          label="False positive pixels /\nunmatched predicted centreline"),
           mpatches.Patch(color=np.array(FN_RGB) / 255,
                          label="False negative pixels /\nmissed reference centreline")]
    fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False, fontsize=42,
               handlelength=1.6, handleheight=1.6, columnspacing=3.0)
    plt.tight_layout(rect=[0, 0.09, 1, 1.0])
    plt.savefig(str(out_png), dpi=130, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return {"name": name, "variant": variant, **{k: cs[k] for k in ("c_iou", "c_f1")},
            **{k: ks[k] for k in ks if k.startswith("k_cl_")}}


# ---------------------------------------------------- segmentation figure (Figs 8-14)
def analysis(name: str, glob, tile):
    """Deployed inference and the analysis stage of the final code (revised crack lengths)."""
    roi, v = roi_and_valid(name)
    roi_v = roi[:v]
    um = SCALE[name]
    crater, crack = (m[:v] for m in EL.variants(glob, tile)["deployed"])
    with contextlib.redirect_stdout(io.StringIO()):
        edge_end, artifact_start = INF.compute_zones_dynamic(roi_v, bottom_pct=0.02)
        details, max_grade, labels = INF.analyze_craters(crater.copy(), roi_v, um, edge_end,
                                                         artifact_start)
    kept = np.isin(labels, [d["label"] for d in details]) if details else np.zeros_like(crater)
    ra = EL.analyze_cracks_rev(crack, um, edge_end)
    areas = [d["area_um2"] for d in details]
    rec = {
        "name": name, "um_px": um, "n_craters": len(details),
        "size_category": SIZE_CAT[max_grade if details else 0],
        "largest_diam_um": max((d["diam_um"] for d in details), default=0.0),
        "largest_area_um2": max(areas, default=0.0),
        "total_area_um2": float(sum(areas)),
        "n_cracks": ra["n"], "n_edge": ra["n_edge"], "n_near_only": ra["n_near_only"],
        "n_internal": ra["n_internal"], "n_near_edge": ra["n_near"], "total_len_um": ra["total_um"],
        "longest_um": ra["longest_um"], "regular_cv": ra["regular_cv"],
        "regular": ra["regular"],
    }
    return rec, dict(roi=roi, v=v, crater=crater, kept=kept, cracks=ra["cracks"])


def segmentation_figure(name: str, glob, tile, out_png: Path):
    """fig_original_vs_segmentation of the authors' code with revised content."""
    rec, d = analysis(name, glob, tile)
    roi, v = d["roi"], d["v"]
    H, W = roi.shape
    um = rec["um_px"]

    # overlay as build_segmentation_overlay: 35 % colour fill, contours, crack lines
    overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    kept = np.zeros((H, W), bool)
    kept[:v] = d["kept"]
    disc = np.zeros((H, W), bool)
    disc[:v] = d["crater"] & ~d["kept"]
    layer = np.zeros_like(overlay)
    layer[kept] = CRATER_BGR
    blend = cv2.addWeighted(overlay, 0.65, layer, 0.35, 0)      # retained: 35 % as in the code
    overlay[kept] = blend[kept]
    layer[disc] = GREY_BGR
    blend = cv2.addWeighted(overlay, 0.35, layer, 0.65, 0)      # discarded: darker, to be seen
    overlay[disc] = blend[disc]
    cnts, _ = cv2.findContours(kept.astype(np.uint8) * 255, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, CRATER_BGR, 2)
    for c in d["cracks"]:
        col = (CRACK_BGR if c["edge_connected"] else NEAR_BGR if c["near_edge"] else INTERNAL_BGR)
        for r, cc in np.argwhere(c["mask"]):
            cv2.circle(overlay, (int(cc), int(r)), 2, col, -1)

    # sized for a single column (about 85 mm): text of 6.5 to 8 pt once printed
    plt.rcParams.update({"font.family": "serif", "font.size": 30})
    fig = plt.figure(figsize=(18, 15.6), facecolor="white")
    gs = gridspec.GridSpec(3, 2, height_ratios=[2.6, 1.05, 2.15], hspace=0.0, wspace=0.06,
                           left=0.03, right=0.97, top=0.965, bottom=0.02)
    ax_o, ax_s = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_l, ax_i = fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :])
    style_axes([ax_o, ax_s], lw=3.0)
    ax_o.imshow(roi, cmap="gray", vmin=0, vmax=255)
    ax_o.set_title("Original SEM Image", color="black", fontsize=42, fontweight="bold", pad=14)
    scalebar(ax_o, um, W, H, fontsize=32, lw=7)
    ax_s.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax_s.set_title(SEG_TITLE, color="black", fontsize=42, fontweight="bold", pad=14)
    scalebar(ax_s, um, W, H, fontsize=32, lw=7)

    rgb = lambda bgr: (bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)  # noqa: E731
    # two columns filled top to bottom: craters on the left, the three crack labels on the right
    blank = mpatches.Patch(visible=False, label=" ")
    leg = [mpatches.Patch(facecolor=rgb(CRATER_BGR), edgecolor=rgb(CRATER_BGR), alpha=0.8,
                          label=f"Retained crater regions ({rec['n_craters']})")]
    if disc.sum() >= 500:        # specks removed as noise are not visible at this scale
        leg.append(mpatches.Patch(color=rgb(GREY_BGR), alpha=0.8, label="Discarded crater pixels"))
    leg += [blank] * (3 - len(leg))
    leg += [Line2D([], [], color=rgb(CRACK_BGR), lw=7, label=f"Edge-connected cracks ({rec['n_edge']})"),
            Line2D([], [], color=rgb(NEAR_BGR), lw=7, label=f"Near-edge cracks ({rec['n_near_only']})"),
            Line2D([], [], color=rgb(INTERNAL_BGR), lw=7, label=f"Internal cracks ({rec['n_internal']})")]
    ax_l.axis("off")
    ax_l.legend(handles=leg, loc="center", ncol=2, frameon=False, fontsize=36,
                handlelength=1.6, handleheight=0.9, columnspacing=1.4, labelspacing=0.35)

    ax_i.set_xlim(0, 1)
    ax_i.set_ylim(0, 1)
    ax_i.set_xticks([])
    ax_i.set_yticks([])
    ax_i.set_facecolor("none")
    for sp in ax_i.spines.values():
        sp.set_visible(False)
    ax_i.add_patch(FancyBboxPatch((0.005, 0.04), 0.990, 0.92, boxstyle="round,pad=0.02",
                                  facecolor="#EEF2FF", edgecolor="#8899CC", linewidth=2.0,
                                  transform=ax_i.transAxes, clip_on=False))
    ax_i.plot([0.5, 0.5], [0.10, 0.90], color="#8899CC", linewidth=1.4, transform=ax_i.transAxes)
    cv_txt = (f"{'Yes' if rec['regular'] else 'No'} (CV={rec['regular_cv']:.2f})"
              if rec["regular_cv"] is not None else "n/a (<3 at edge)")
    txt_c = (f"CRATERS\n"
             f"Scale: {um:.4f} µm/px\n"
             f"Count: {rec['n_craters']} | Size: {rec['size_category']}\n"
             f"Largest diam.: {rec['largest_diam_um']:.1f} µm\n"
             f"Largest area: {rec['largest_area_um2']:.0f} µm²\n"
             f"Total area: {rec['total_area_um2']:.0f} µm²")
    txt_k = (f"CRACKS\n"
             f"Total: {rec['n_cracks']}\n"
             f"Edge {rec['n_edge']} | Near {rec['n_near_only']} | Int. {rec['n_internal']}\n"
             f"Total length: {rec['total_len_um']:.1f} µm\n"
             f"Longest: {rec['longest_um']:.1f} µm\n"
             f"Regular: {cv_txt}")
    for x, t in ((0.25, txt_c), (0.75, txt_k)):
        ax_i.text(x, 0.52, t, transform=ax_i.transAxes, ha="center", va="center", fontsize=34,
                  color="black", fontfamily="monospace", linespacing=1.5)
    plt.savefig(str(out_png), dpi=150, bbox_inches="tight", pad_inches=0.2, facecolor="white",
                edgecolor="none")
    plt.close(fig)
    return rec


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "sample":
        arch, out = sys.argv[2], Path(sys.argv[3])
        out.mkdir(parents=True, exist_ok=True)
        for kind, name in (("seg", "ALMW1-1a-600MW-2p_06"), ("err", "ALMW1-1a-600MW-2p_06"),
                           ("err", "ALMW1-3a-600MW_05")):
            try:
                (glob, tile), fold = oof_maps(name, arch)
            except FileNotFoundError:
                print("fold not ready:", name)
                continue
            if kind == "seg":
                print(segmentation_figure(name, glob, tile, out / f"sample_seg_{name}.png"))
            else:
                print(error_figure(name, glob, tile, out / f"sample_err_{name}.png"))
