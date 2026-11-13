"""Example figures of the revision (Figures 5-15) and the table of example measurements.

* Error figures (5, 6, 7 and the undamaged field of the appendix): out-of-fold network
  output of the edge-grouped cross-validation (configuration A, repeat 1), fixed
  thresholds 0.50 / 0.25.  Craters are scored strictly; cracks with the length-weighted
  centreline protocol at a buffer of 5 px.
* Pipeline figures (8-14): output of the frozen deployed pipeline (multiscale fusion,
  ridge enhancement, hysteresis, gap bridging) followed by the analysis stage, with the
  revised length estimator.  ALMW examples use the out-of-fold models, ATW examples the
  submitted checkpoint.  The numbers are written to results/example_measurements.csv and
  to a LaTeX table instead of being printed inside the images.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402
import eval_lib as EL     # noqa: E402

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import ndimage as ndi
from skimage.morphology import binary_dilation, disk, skeletonize

from tungsten_analyzer import inference as INF

OUT = RC.FIGURES
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.titlesize": 7,
                     "legend.fontsize": 6.5, "savefig.dpi": 600})

ORANGE = np.array([0.95, 0.60, 0.05])
GREEN = np.array([0.10, 0.80, 0.10])
RED = np.array([0.90, 0.05, 0.05])
BLUE = np.array([0.10, 0.35, 1.00])
GREY = np.array([0.55, 0.55, 0.55])
MAGENTA = np.array([0.85, 0.10, 0.85])
YELLOW = np.array([1.00, 0.85, 0.05])
CYAN = "#00C8FF"

SIZE_CAT = {0: "no retained region", 1: "< 100 µm", 2: "100-600 µm", 3: "600-950 µm", 4: "> 950 µm"}

SCALE = pd.read_csv(RC.RESULTS / "scale_verified.csv").set_index("name")["um_px"].to_dict()
MAN = RC.load_manifest().set_index("name")


# -------------------------------------------------------------------------- helpers
def image_path(name):
    p = RC.IMG_DIR / f"{name}.jpg"
    return p if p.exists() else RC.ATW_DIR / f"{name}.jpg"


def context(name):
    g = RC.read_gray(image_path(name))
    H = g.shape[0]
    roi_h = H - RC.strip_rows(H)
    valid_h = min(roi_h, RC.bar_top_row(g))
    return g, roi_h, valid_h


def scalebar(ax, um_px, W, H):
    target = 0.2 * W * um_px
    nice = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    L = max([v for v in nice if v <= target] or [1])
    px = L / um_px
    x0, y0 = 0.04 * W, 0.93 * H
    ax.plot([x0, x0 + px], [y0, y0], color="white", lw=1.4, solid_capstyle="butt",
            path_effects=[pe.Stroke(linewidth=2.4, foreground="black"), pe.Normal()])
    ax.text(x0 + px / 2, y0 - 0.03 * H, f"{L:g} µm", color="white", ha="center", va="bottom",
            fontsize=4.8, path_effects=[pe.withStroke(linewidth=1.0, foreground="black")])


def blend(gray, layers):
    """gray uint8 HxW; layers list of (mask, rgb, alpha)."""
    rgb = np.repeat((gray.astype(float) / 255.0)[..., None], 3, axis=2)
    for m, col, a in layers:
        if m is None or not m.any():
            continue
        rgb[m] = (1 - a) * rgb[m] + a * col
    return np.clip(rgb, 0, 1)


def thick(m, r=1):
    return binary_dilation(m, disk(r)) if m.any() else m


def maps_for(name, source):
    """(glob, tile) probability maps: out-of-fold (CV run dir) or submitted checkpoint."""
    if source == "published":
        p = RC.RESULTS / "published_probs" / f"{name}.npz"
        if not p.exists():
            p = RC.RESULTS / "published_probs_atw" / f"{name}.npz"
            if not p.exists():
                atw_maps([name])
    else:
        p = Path(source) / "probs_test" / f"{name}.npz"
    z = np.load(p)
    return z["glob"].astype(np.float32), (z["tile"].astype(np.float32) if bool(z["has_tile"]) else None)


def atw_maps(names):
    import torch
    from tungsten_analyzer.model import build_unet
    from tungsten_analyzer.inference import UNetPredictor
    out = RC.RESULTS / "published_probs_atw"
    out.mkdir(exist_ok=True)
    dev = torch.device("cuda")
    ck = torch.load(RC.PUBLISHED_CKPT, map_location=dev, weights_only=False)
    model = build_unet("resnet34", None, 2).to(dev)
    model.load_state_dict(ck["model_state"])
    model.eval()
    pr = UNetPredictor.__new__(UNetPredictor)
    pr.model, pr.device, pr.img_size = model, dev, 512
    with torch.no_grad():
        for n in names:
            gray = RC.roi_of(RC.read_gray(image_path(n)))
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            H, W = gray.shape
            g = pr._predict_single(rgb, H, W)
            t = pr._predict_tiled(rgb, H, W)
            np.savez_compressed(out / f"{n}.npz", glob=g.astype(np.float16),
                                tile=(t if t is not None else np.zeros_like(g)).astype(np.float16),
                                has_tile=t is not None)


def oof_run(name, scheme="edge5_r1", config="A"):
    fold = int(MAN.loc[name, f"fold_{scheme}"])
    return RC.CV_RUNS / f"{config}_{scheme}_f{fold}", fold


# ------------------------------------------------------------------ error figures
def error_figure(name, out_png, tau=5.0):
    run, fold = oof_run(name)
    glob, _ = maps_for(name, str(run))
    g, roi_h, v = context(name)
    um = SCALE[name]
    shapes = RC.load_shapes(RC.MASK_DIR / f"{name}.json")
    ref_c, ref_k = RC.rasterise(shapes, roi_h)
    ref_c, ref_k = ref_c[:v], ref_k[:v]
    ref_cl = RC.crack_centreline(shapes, roi_h)[:v]
    roi = g[:v]
    pc, pk = glob[0][:v], glob[1][:v]
    mc, mk = pc > 0.5, pk > 0.25
    cs = EL.crater_scores(ref_c, mc)
    ks = EL.crack_scores(ref_k, ref_cl, mk, taus_px=(int(tau),), um_px=um)
    H, W = roi.shape

    # crack centreline matching for display
    sk = skeletonize(mk) if mk.any() else mk
    d_ref = ndi.distance_transform_edt(~ref_cl) if ref_cl.any() else np.full(mk.shape, np.inf)
    d_pred = ndi.distance_transform_edt(~sk) if sk.any() else np.full(mk.shape, np.inf)
    sk_ok, sk_bad = sk & (d_ref <= tau), sk & (d_ref > tau)
    ref_miss = ref_cl & (d_pred > tau)

    fig = plt.figure(figsize=(7.1, 3.35))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 0.035, 0.11, 1], wspace=0.06, hspace=0.36,
                          left=0.005, right=0.995, top=0.90, bottom=0.10)
    for row, cls in enumerate(("crater", "crack")):
        a0 = fig.add_subplot(gs[row, 0]); a1 = fig.add_subplot(gs[row, 1])
        a2 = fig.add_subplot(gs[row, 2]); cb = fig.add_subplot(gs[row, 3]); a3 = fig.add_subplot(gs[row, 5])
        for a in (a0, a1, a2, a3):
            a.set_xticks([]); a.set_yticks([])
        a0.imshow(roi, cmap="gray", vmin=0, vmax=255)
        scalebar(a0, um, W, H)
        a0.set_title("SEM image")
        if cls == "crater":
            a1.imshow(blend(roi, [(ref_c, ORANGE, 0.75)]))
            a1.set_title("Reference, crater polygons")
            im = a2.imshow(pc, cmap="hot", vmin=0, vmax=1)
            a2.set_title("Crater probability")
            tp, fp, fn = ref_c & mc, ~ref_c & mc, ref_c & ~mc
            a3.imshow(blend(roi, [(tp, GREEN, 0.75), (fp, RED, 0.75), (fn, BLUE, 0.75)]))
            if cs["crater_ref"]:
                a3.set_title(f"Craters, strict\nIoU {cs['c_iou']:.3f}, F1 {cs['c_f1']:.3f}", linespacing=1.1)
            else:
                a3.set_title("Craters, strict\n" + ("false alarm" if cs["crater_pred"] else "no reference, no prediction"),
                             linespacing=1.1)
        else:
            a1.imshow(blend(roi, [(ref_k, RED, 0.30), (thick(ref_cl, 1), RED, 1.0)]))
            a1.set_title("Reference, crack polylines\n(15 px band shaded)", linespacing=1.1)
            im = a2.imshow(pk, cmap="hot", vmin=0, vmax=1)
            a2.set_title("Crack probability")
            a3.imshow(blend(roi, [(thick(sk_ok, 1), GREEN, 1.0), (thick(sk_bad, 1), RED, 1.0),
                                  (thick(ref_miss, 1), BLUE, 1.0)]))
            if ks["crack_ref"]:
                t = int(tau)
                a3.set_title(f"Cracks, centreline, buffer {t} px ({tau * um:.1f} µm)\n"
                             f"P {ks[f'k_cl_P_{t}px']:.2f}, R {ks[f'k_cl_R_{t}px']:.2f}, "
                             f"F1 {ks[f'k_cl_F1_{t}px']:.2f}; band IoU {ks['k_strict_iou']:.2f}",
                             linespacing=1.1, fontsize=5.8)
            else:
                a3.set_title("Cracks\n" + ("false alarm" if ks["crack_pred"] else "no reference, no prediction"),
                             linespacing=1.1)
        c = fig.colorbar(im, cax=cb, ticks=[0, 0.5, 1])
        c.ax.tick_params(labelsize=5.2, length=1.5, pad=1)
    handles = [Patch(color=GREEN, label="TP pixels / matched predicted centreline"),
               Patch(color=RED, label="FP pixels / unmatched predicted centreline"),
               Patch(color=BLUE, label="FN pixels / missed reference centreline")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.0,
               handlelength=1.4, columnspacing=1.2)
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)
    rec = {"figure_file": Path(out_png).name, "name": name, "model": f"A edge5_r1 fold {fold + 1}",
           "um_px": um, **{k: cs[k] for k in ("c_iou", "c_f1", "crater_ref", "crater_pred")},
           **{k: ks[k] for k in ks if k.startswith("k_") or k.startswith("crack_")}}
    return rec


# --------------------------------------------------------------- pipeline figures
def pipeline_measure(name, glob, tile, um):
    g, roi_h, v = context(name)
    roi = g[:v]
    vs = EL.variants(glob, tile)
    crater, crack = (m[:v] for m in vs["deployed"])
    edge_end, artifact_start = INF.compute_zones_dynamic(roi, bottom_pct=0.02)
    with contextlib.redirect_stdout(io.StringIO()):
        details, max_grade, labels = INF.analyze_craters(crater.copy(), roi, um, edge_end, artifact_start)
        cinfo, skel, lab = INF.analyze_cracks(crack, um, edge_end)
    kept = np.isin(labels, [d["label"] for d in details]) if details else np.zeros_like(crater)
    # revised analysis stage: cracks are components of the revised centreline (eval_lib)
    ra = EL.analyze_cracks_rev(crack, um, edge_end)
    cracks = [{"len_kulpa_um": c["len_um"], "edge_connected": c["edge_connected"],
               "near_edge": c["near_edge"], "mask": c["mask"]} for c in ra["cracks"]]
    areas = [d["area_um2"] for d in details]
    diams = [d["diam_um"] for d in details]
    rec = {
        "name": name, "um_px": um, "edge_end_px": edge_end,
        "n_craters": len(details),
        "crater_total_area_um2": float(np.sum(areas)) if areas else 0.0,
        "crater_largest_area_um2": float(np.max(areas)) if areas else 0.0,
        "crater_largest_diam_um": float(np.max(diams)) if diams else 0.0,
        "size_category": SIZE_CAT[max_grade if details else 0],
        "n_cracks": len(cracks),
        "n_edge_connected": int(sum(c["edge_connected"] for c in cracks)),
        "n_near_edge_only": int(sum(c["near_edge"] and not c["edge_connected"] for c in cracks)),
        "n_internal": int(sum(not c["near_edge"] for c in cracks)),
        "crack_total_len_um": float(sum(c["len_kulpa_um"] for c in cracks)),
        "crack_longest_um": float(max((c["len_kulpa_um"] for c in cracks), default=0.0)),
        "crack_total_len_count_um": cinfo["total_length_um"],
        "n_cracks_submitted_stage": cinfo["n_total"],
        "regular_cv": ra["regular_cv"],
        "regular_cv_submitted_stage": cinfo["regular_cv"],
        "n_near_edge_for_cv": ra["n_near"],
    }
    return rec, dict(roi=roi, crater=crater, kept=kept, cracks=cracks, edge_end=edge_end)


def pipeline_figure(name, source, out_png, panel_note=""):
    glob, tile = maps_for(name, source)
    um = SCALE[name]
    rec, d = pipeline_measure(name, glob, tile, um)
    roi = d["roi"]
    H, W = roi.shape
    excl = d["crater"] & ~d["kept"]
    edge_m = np.zeros_like(roi, bool); near_m = edge_m.copy(); int_m = edge_m.copy()
    for c in d["cracks"]:
        tgt = edge_m if c["edge_connected"] else (near_m if c["near_edge"] else int_m)
        tgt |= c["mask"]
    img = blend(roi, [(excl, GREY, 0.55), (d["kept"], ORANGE, 0.45),
                      (thick(edge_m, 2), RED, 1.0), (thick(near_m, 2), MAGENTA, 1.0),
                      (thick(int_m, 2), YELLOW, 1.0)])
    kept_u8 = d["kept"].astype(np.uint8)
    cnts, _ = cv2.findContours(kept_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    fig = plt.figure(figsize=(3.46, 1.72))
    gs = fig.add_gridspec(1, 2, wspace=0.04, left=0.005, right=0.995, top=0.90, bottom=0.25)
    a0, a1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    for a in (a0, a1):
        a.set_xticks([]); a.set_yticks([])
    a0.imshow(roi, cmap="gray", vmin=0, vmax=255)
    scalebar(a0, um, W, H)
    a0.set_title("SEM image")
    a1.imshow(img)
    for cnt in cnts:
        a1.plot(cnt[:, 0, 0], cnt[:, 0, 1], color=ORANGE, lw=0.6)
    if d["edge_end"] > 0:
        a1.axhline(d["edge_end"], color=CYAN, lw=0.7, ls=(0, (4, 2)))
    scalebar(a1, um, W, H)
    a1.set_title("Pipeline output" + (f" ({panel_note})" if panel_note else ""))
    handles = [Patch(color=ORANGE, alpha=0.8, label=f"retained crater regions ({rec['n_craters']})"),
               Patch(color=GREY, alpha=0.8, label="discarded crater pixels"),
               Line2D([], [], color=RED, lw=1.6, label=f"edge-connected cracks ({rec['n_edge_connected']})"),
               Line2D([], [], color=MAGENTA, lw=1.6, label=f"near-edge cracks ({rec['n_near_edge_only']})"),
               Line2D([], [], color=YELLOW, lw=1.6, label=f"internal cracks ({rec['n_internal']})"),
               Line2D([], [], color=CYAN, lw=0.9, ls=(0, (4, 2)),
                      label="edge row" if d["edge_end"] > 0 else "edge row not found")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=6.2,
               handlelength=1.6, columnspacing=1.0, labelspacing=0.25)
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)
    rec["figure_file"] = Path(out_png).name
    rec["source"] = "submitted checkpoint" if source == "published" else Path(source).name
    return rec


EXAMPLES_ERR = [("fig05_ALMW1-3a-600MW_05.png", "ALMW1-3a-600MW_05"),
                ("fig06_ALMW3-1a-1200MW_01.png", "ALMW3-1a-1200MW_01"),
                ("fig07_ALMW3-2d-1200MW-HT_13.png", "ALMW3-2d-1200MW-HT_13"),
                ("figE_ALMW1-1c-450MW_04.png", "ALMW1-1c-450MW_04")]
EXAMPLES_PIPE = [("fig08_ALMW1-1a-600MW-2p_06.png", "ALMW1-1a-600MW-2p_06", "oof"),
                 ("fig09_ALMW2-1a-600MW_2p_02.png", "ALMW2-1a-600MW_2p_02", "oof"),
                 ("fig10_ALMW2-1c-450MW_2p_04.png", "ALMW2-1c-450MW_2p_04", "oof"),
                 ("fig11_ATW2-3a-600MW_09.png", "ATW2-3a-600MW_09", "published"),
                 ("fig12_ATW3-2c_01.png", "ATW3-2c_01", "published"),
                 ("fig13_ATW3-3d_05.png", "ATW3-3d_05", "published"),
                 ("fig14_ATW3-4d_14.png", "ATW3-4d_14", "published")]


def main():
    RC.set_low_priority_off()
    which = sys.argv[1:] or ["err", "pipe"]
    if "err" in which:
        recs = []
        for f, n in EXAMPLES_ERR:
            run, _ = oof_run(n)
            if not (run / "done.json").exists():
                print("skip (fold not finished):", n)
                continue
            recs.append(error_figure(n, OUT / f))
            print("written", f)
        if recs:
            pd.DataFrame(recs).to_csv(RC.RESULTS / "example_errors.csv", index=False)
    if "pipe" in which:
        recs = []
        for f, n, src in EXAMPLES_PIPE:
            if src == "oof":
                run, fold = oof_run(n)
                if not (run / "done.json").exists():
                    print("skip (fold not finished):", n)
                    continue
                recs.append(pipeline_figure(n, str(run), OUT / f, f"out-of-fold, fold {fold + 1}"))
            else:
                recs.append(pipeline_figure(n, "published", OUT / f, "submitted model"))
            print("written", f)
        if recs:
            df = pd.DataFrame(recs)
            df.to_csv(RC.RESULTS / "example_measurements.csv", index=False)
            with pd.option_context("display.width", 250, "display.max_columns", 40):
                print(df.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
