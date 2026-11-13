"""Evaluation library of the revision: one versioned implementation for every number.

Conventions (written into the revised manuscript, Appendix on metrics)
------------------------------------------------------------------------
* Region of interest: rows above the SEM data bar (located per image; 62-63 rows in the
  standard Zeiss export, 71-72 or 78 rows in the two CIEMAT exports) and below row 0.  Reference and
  prediction are both restricted to it.
* Craters: strict pixel counts TP, FP, FN between the filled polygons and the predicted
  mask.  IoU = TP/(TP+FP+FN), F1 = 2TP/(2TP+FP+FN), so F1 = 2 IoU/(1+IoU) holds per image.
* Cracks, several protocols, always labelled:
    - strict_band: strict IoU/F1 between the 15-px reference band (polylines rasterised
      with PIL at width 15 for a 1024-px frame) and the prediction reduced to its
      skeleton and redrawn at the same width (disk of radius 7).
    - dilated_band: ordinary IoU/F1 between the two bands after both are dilated by a
      disk of radius 5 px.
    - legacy: the score used in the submitted manuscript (metrics.tolerant_iou on the
      same bands), kept only to reconcile with the submitted Table 1.  It is NOT an IoU.
    - centreline: precision (correctness), recall (completeness) and F1 of the predicted
      centreline against the annotated polylines with a buffer tolerance tau, measured
      along the curves with geometric (Freeman) length weights.  The predicted centreline
      is the skeleton of the mask after filling pin holes (< 64 px) and pruning terminal
      branches shorter than 8 px (half the 15-px band; both scaled to a 1024-px frame).
    - clDice: Shit et al. (2021) with the reference band and the width-normalised
      prediction as volumes.
* Empty cases: a non-empty prediction against an empty reference scores 0 (a false
  positive image), an empty prediction against a non-empty reference scores 0 (a false
  negative image), and both empty is a true negative image that is counted separately
  and does not enter the overlap statistics.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import binary_dilation, disk, skeletonize, remove_small_objects, remove_small_holes

from tungsten_analyzer import metrics as M
from tungsten_analyzer import inference as INF

SQRT2 = math.sqrt(2.0)
KULPA_AXIAL, KULPA_DIAG = 0.948, 1.343      # Kulpa (1977) weights for 8-connected chains


# ---------------------------------------------------------------------------- ROI
bar_top_row = RC.bar_top_row


# ----------------------------------------------------------------- skeleton length
_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeleton_edges(skel: np.ndarray):
    """Edges of the pixel graph of an 8-connected skeleton.

    Axial neighbours are always joined.  A diagonal pair is joined only when the two
    pixels are not already linked through a common axial neighbour, which avoids
    counting the corners of L-shaped steps twice.
    Returns arrays (r0, c0, r1, c1, is_diag) with each edge listed once.
    """
    s = np.pad(skel.astype(bool), 1)
    out = []
    # axial edges: right and down
    for dr, dc in ((0, 1), (1, 0)):
        a = s[1:-1, 1:-1] & s[1 + dr:s.shape[0] - 1 + dr, 1 + dc:s.shape[1] - 1 + dc]
        r, c = np.nonzero(a)
        out.append((r, c, r + dr, c + dc, np.zeros(len(r), bool)))
    # diagonal edges: down-right and down-left
    for dc in (1, -1):
        a = s[1:-1, 1:-1] & s[2:, 1 + dc:s.shape[1] - 1 + dc]
        # common axial neighbours of (r, c) and (r+1, c+dc) are (r, c+dc) and (r+1, c)
        n1 = s[1:-1, 1 + dc:s.shape[1] - 1 + dc]
        n2 = s[2:, 1:-1]
        a = a & ~(n1 | n2)
        r, c = np.nonzero(a)
        out.append((r, c, r + 1, c + dc, np.ones(len(r), bool)))
    r0 = np.concatenate([o[0] for o in out]); c0 = np.concatenate([o[1] for o in out])
    r1 = np.concatenate([o[2] for o in out]); c1 = np.concatenate([o[3] for o in out])
    dg = np.concatenate([o[4] for o in out])
    return r0, c0, r1, c1, dg


def skeleton_length(skel: np.ndarray, method: str = "freeman") -> float:
    """Length of a skeleton in pixels.

    count   : number of skeleton pixels (the estimator of the submitted manuscript)
    freeman : 1 per axial step, sqrt(2) per diagonal step
    kulpa   : 0.948 per axial step, 1.343 per diagonal step (Kulpa 1977)
    """
    if method == "count":
        return float(np.count_nonzero(skel))
    _, _, _, _, dg = skeleton_edges(skel)
    n_d = int(dg.sum()); n_a = len(dg) - n_d
    if method == "freeman":
        return n_a + SQRT2 * n_d
    if method == "kulpa":
        return KULPA_AXIAL * n_a + KULPA_DIAG * n_d
    raise ValueError(method)


_K8 = np.ones((3, 3), int)
_K8[1, 1] = 0
SPUR_PX_1024 = 8          # half the 15-px drawing width of the annotation, 1024-px frame


def prune_spurs(skel: np.ndarray, max_len: float, n_iter: int = 3) -> np.ndarray:
    """Remove terminal branches shorter than `max_len` pixels (Freeman length).

    Skeletonising a band of finite width leaves short side branches that end on the band
    boundary; they are an artefact of the band, not crack structure.  A terminal branch
    is a connected piece of the skeleton, once the junction pixels (>= 3 neighbours) are
    removed, that contains an end point and touches a junction.  Isolated pieces are left
    to the length filter of the analysis stage.  The rule is geometric and fixed a priori
    (half the band width); it is not tuned on the data.
    """
    sk = skel.astype(bool).copy()
    for _ in range(n_iter):
        if not sk.any():
            break
        nb = ndi.convolve(sk.astype(int), _K8, mode="constant") * sk
        junction = sk & (nb >= 3)
        if not junction.any():
            break
        endpoint = sk & (nb == 1)
        branches = sk & ~junction
        lab, n = ndi.label(branches, structure=np.ones((3, 3), int))
        if n == 0:
            break
        has_end = ndi.maximum(endpoint, lab, index=np.arange(1, n + 1)).astype(bool)
        touch = ndi.maximum(ndi.binary_dilation(junction, np.ones((3, 3), bool)) & branches,
                            lab, index=np.arange(1, n + 1)).astype(bool)
        removed = False
        for i in np.nonzero(has_end & touch)[0]:
            piece = lab == (i + 1)
            if skeleton_length(piece, "freeman") + 1.0 < max_len:
                sk[piece] = False
                removed = True
        if not removed:
            break
        sk = skeletonize(sk)
    return sk


def spur_len_px(width_px: int) -> float:
    return SPUR_PX_1024 * width_px / 1024.0


HOLE_PX_1024 = 64         # pin holes left by rasterising or thresholding a band


def crack_skeleton(mask: np.ndarray) -> np.ndarray:
    """Revised centreline of a crack mask: fill pin holes, skeletonise, prune spurs."""
    m = mask.astype(bool)
    if not m.any():
        return m
    w = m.shape[1]
    m = remove_small_holes(m, max(1, int(round(HOLE_PX_1024 * (w / 1024.0) ** 2))))
    return prune_spurs(skeletonize(m), spur_len_px(w))


REGULAR_CV_MAX = 0.45     # inference.py: regular network if CV of spacings < 0.45


def analyze_cracks_rev(crack_mask: np.ndarray, um_px: float, edge_end: int) -> dict:
    """Revised analysis stage for cracks (the rules of inference.analyze_cracks applied to
    the revised centreline).

    Components of the revised skeleton (pin holes filled, spurs pruned) with at least
    max(10, int(15 um / pixel size)) pixels are cracks.  Each one is edge connected when
    some of its pixels lie above the edge row, and near edge when its topmost pixel lies
    within max(40, 0.08 H) rows below the edge row.  Its length is the Kulpa length.  The
    regularity CV is the coefficient of variation (population SD / mean) of the spacings
    between the sorted horizontal positions of the crack origins (median column of the
    skeleton pixels within 10 rows of the topmost one), over the edge connected and near
    edge cracks, when there are at least three of them.
    """
    H = crack_mask.shape[0]
    rs = crack_skeleton(crack_mask) if crack_mask.any() else np.zeros(crack_mask.shape, bool)
    min_px = max(10, int(15.0 / um_px))
    lab, n = ndi.label(rs, structure=np.ones((3, 3), int))
    margin = max(40, int(0.08 * H))
    cracks = []
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() < min_px:
            continue
        ys, xs = np.nonzero(comp)
        top = int(ys.min())
        edge_conn = bool((ys < edge_end).any())
        near = edge_conn or top < edge_end + margin
        cracks.append({"len_um": skeleton_length(comp, "kulpa") * um_px, "edge_connected": edge_conn,
                       "near_edge": near, "x_origin": float(np.median(xs[ys <= top + 10])),
                       "mask": comp})
    xs = sorted(c["x_origin"] for c in cracks if c["near_edge"])
    cv = None
    if len(xs) >= 3:
        sp = np.diff(xs)
        if sp.mean() > 1:
            cv = float(sp.std() / (sp.mean() + 1e-9))
    return {"cracks": cracks, "n": len(cracks),
            "n_edge": int(sum(c["edge_connected"] for c in cracks)),
            "n_near_only": int(sum(c["near_edge"] and not c["edge_connected"] for c in cracks)),
            "n_internal": int(sum(not c["near_edge"] for c in cracks)),
            "n_near": len(xs),
            "total_um": float(sum(c["len_um"] for c in cracks)),
            "longest_um": float(max((c["len_um"] for c in cracks), default=0.0)),
            "regular_cv": cv,
            "regular": (cv is not None and cv < REGULAR_CV_MAX)}


def pixel_length_weights(skel: np.ndarray) -> np.ndarray:
    """Per-pixel share of the Freeman length: half the length of every incident edge."""
    w = np.zeros(skel.shape, float)
    r0, c0, r1, c1, dg = skeleton_edges(skel)
    L = np.where(dg, SQRT2, 1.0) / 2.0
    np.add.at(w, (r0, c0), L)
    np.add.at(w, (r1, c1), L)
    # isolated single pixels have no edge; give them unit length so they are not lost
    iso = skel & (w == 0)
    w[iso] = 1.0
    return w


# ---------------------------------------------------------------- overlap metrics
def counts(ref: np.ndarray, pred: np.ndarray):
    ref = ref.astype(bool); pred = pred.astype(bool)
    tp = int(np.count_nonzero(ref & pred))
    fp = int(np.count_nonzero(~ref & pred))
    fn = int(np.count_nonzero(ref & ~pred))
    return tp, fp, fn


def iou_f1(tp, fp, fn):
    if tp + fp + fn == 0:
        return float("nan"), float("nan")    # both empty: true negative, no score
    return tp / (tp + fp + fn), 2 * tp / (2 * tp + fp + fn)


def width_normalise(pred: np.ndarray, width: int = 15) -> np.ndarray:
    return M.normalise_crack_width(pred, width)


def crack_scores(ref_band: np.ndarray, ref_centre: np.ndarray, pred: np.ndarray,
                 taus_px=(2, 3, 5, 10, 20), um_px: float | None = None,
                 taus_um=(2.0, 5.0, 10.0)) -> dict:
    """All crack protocols for one image.  Inputs are already restricted to the ROI."""
    out = {}
    pred = pred.astype(bool)
    has_ref, has_pred = bool(ref_band.any()), bool(pred.any())
    out["crack_ref"] = has_ref
    out["crack_pred"] = has_pred
    pred_band = width_normalise(pred)
    # strict band
    tp, fp, fn = counts(ref_band, pred_band)
    out.update(k_strict_tp=tp, k_strict_fp=fp, k_strict_fn=fn)
    out["k_strict_iou"], out["k_strict_f1"] = iou_f1(tp, fp, fn)
    # ordinary IoU after a 5-px disk dilation of both bands
    se = disk(5)
    rd = binary_dilation(ref_band, se) if has_ref else ref_band
    pdl = binary_dilation(pred_band, se) if has_pred else pred_band
    tp, fp, fn = counts(rd, pdl)
    out["k_dil5_iou"], out["k_dil5_f1"] = iou_f1(tp, fp, fn)
    # legacy score of the submitted manuscript
    if has_ref or has_pred:
        j = M.tolerant_iou(ref_band, pred_band)
        out["k_legacy_score"] = j
        out["k_legacy_f1"] = 2 * j / (1 + j) if j > 0 else 0.0
    else:
        out["k_legacy_score"] = out["k_legacy_f1"] = float("nan")
    # centreline precision / recall with length weights
    # revised centreline of the prediction: pin holes filled, spurs pruned (see crack_skeleton)
    pred_skel = crack_skeleton(pred) if has_pred else np.zeros_like(pred)
    if has_pred and not pred_skel.any():          # degenerate speck: keep its pixels
        pred_skel = pred.copy()
    w_pred = pixel_length_weights(pred_skel) if has_pred else np.zeros(pred.shape)
    w_ref = pixel_length_weights(ref_centre) if has_ref else np.zeros(pred.shape)
    L_pred, L_ref = float(w_pred.sum()), float(w_ref.sum())
    out["k_len_ref_px"], out["k_len_pred_px"] = L_ref, L_pred
    d_to_ref = ndi.distance_transform_edt(~ref_centre) if has_ref else None
    d_to_pred = ndi.distance_transform_edt(~pred_skel) if has_pred else None
    tau_list = [(f"{t}px", float(t)) for t in taus_px]
    if um_px:
        tau_list += [(f"{t:g}um", float(t) / um_px) for t in taus_um]
    for name, t in tau_list:
        if has_ref and has_pred:
            prec = float(w_pred[d_to_ref <= t].sum() / L_pred)
            rec = float(w_ref[d_to_pred <= t].sum() / L_ref)
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        elif has_ref:
            prec, rec, f1 = float("nan"), 0.0, 0.0
        elif has_pred:
            prec, rec, f1 = 0.0, float("nan"), 0.0
        else:
            prec = rec = f1 = float("nan")
        out[f"k_cl_P_{name}"], out[f"k_cl_R_{name}"], out[f"k_cl_F1_{name}"] = prec, rec, f1
    # mean distance of matched predicted skeleton to the reference (localisation error)
    if has_ref and has_pred:
        dd = d_to_ref[pred_skel]
        ww = w_pred[pred_skel]
        m = dd <= 10
        out["k_loc_err_px"] = float(np.average(dd[m], weights=ww[m])) if m.any() else float("nan")
    else:
        out["k_loc_err_px"] = float("nan")
    # clDice with the reference band and the width-normalised prediction
    if has_ref and has_pred:
        s_p = skeletonize(pred_band)
        tprec = np.count_nonzero(s_p & ref_band) / max(1, np.count_nonzero(s_p))
        tsens = np.count_nonzero(ref_centre & pred_band) / max(1, np.count_nonzero(ref_centre))
        out["k_cldice"] = 2 * tprec * tsens / (tprec + tsens) if tprec + tsens > 0 else 0.0
    elif has_ref or has_pred:
        out["k_cldice"] = 0.0
    else:
        out["k_cldice"] = float("nan")
    return out


def crater_scores(ref: np.ndarray, pred: np.ndarray) -> dict:
    tp, fp, fn = counts(ref, pred)
    i, f = iou_f1(tp, fp, fn)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"crater_ref": bool(ref.any()), "crater_pred": bool(pred.any()),
            "c_tp": tp, "c_fp": fp, "c_fn": fn, "c_iou": i, "c_f1": f,
            "c_prec": prec, "c_rec": rec}


# ------------------------------------------------------------ inference variants
def ridge(prob: np.ndarray) -> np.ndarray:
    return INF.UNetPredictor._enhance_ridges(None, prob)


def variants(glob: np.ndarray, tile: np.ndarray | None, thr_c=0.5, thr_k=0.25,
             hyst=(0.08, 0.20), fusion=0.7):
    """Masks of the inference variants compared in the revision.

    net      global 512-px pass with 4-flip TTA, fixed thresholds
    ms       + max fusion with the tiled pass (crater tile weight `fusion`)
    ms_hy    + Hessian ridge enhancement and hysteresis thresholding of cracks
    deployed + oriented gap bridging (15 px) and removal of specks < 5 px
    """
    g = glob.astype(np.float64)
    out = {"net": (g[0] > thr_c, g[1] > thr_k)}
    if tile is not None:
        t = tile.astype(np.float64)
        crater_p = np.maximum(g[0], fusion * t[0])
        crack_p = np.maximum(g[1], t[1])
    else:
        crater_p, crack_p = g[0], g[1]
    out["ms"] = (crater_p > thr_c, crack_p > thr_k)
    rp = ridge(crack_p)
    hy = INF._hysteresis_threshold(rp, low=hyst[0], high=hyst[1])
    out["ms_hy"] = (crater_p > thr_c, hy)
    out["deployed"] = (crater_p > thr_c, INF._bridge_crack_gaps(hy, max_gap=15))
    out["_ms_noridge_hy"] = (crater_p > thr_c,
                             INF._hysteresis_threshold(crack_p, low=hyst[0], high=hyst[1]))
    return out


# ---------------------------------------------------------------- measurements
def measure(crater_mask: np.ndarray, crack_mask: np.ndarray, roi_gray: np.ndarray,
            um_px: float, apply_zones: bool = True) -> dict:
    """Physical outputs of the analysis stage, with the submitted and revised estimators."""
    if apply_zones:
        edge_end, artifact_start = INF.compute_zones_dynamic(roi_gray, bottom_pct=0.02)
    else:
        edge_end, artifact_start = 0, roi_gray.shape[0]
    details, _, _ = INF.analyze_craters(crater_mask.copy(), roi_gray, um_px, edge_end, artifact_start)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        cinfo, skel_clean, labels_clean = INF.analyze_cracks(crack_mask, um_px, edge_end)
    areas = [d["area_um2"] for d in details]
    diams = [d["diam_um"] for d in details]
    out = {
        "edge_end": edge_end, "artifact_start": artifact_start,
        "m_n_craters": len(details),
        "m_crater_total_area_um2": float(np.sum(areas)) if areas else 0.0,
        "m_crater_largest_area_um2": float(np.max(areas)) if areas else 0.0,
        "m_crater_largest_diam_um": float(np.max(diams)) if diams else 0.0,
        "m_n_cracks": cinfo["n_total"],
        "m_n_cracks_edge": cinfo["n_large"],
        "m_crack_len_count_um": cinfo["total_length_um"],
        "m_crack_len_freeman_um": skeleton_length(skel_clean, "freeman") * um_px,
        "m_crack_len_kulpa_um": skeleton_length(skel_clean, "kulpa") * um_px,
        "m_regular_cv": cinfo["regular_cv"],
    }
    # submitted skeleton without the length filter, Freeman length (diagnostic)
    sk = skeletonize(crack_mask.astype(bool)) if crack_mask.any() else crack_mask.astype(bool)
    out["m_crack_len_freeman_nofilter_um"] = skeleton_length(sk, "freeman") * um_px
    # revised estimator: pin holes filled, spurs pruned, same component filter, Kulpa weights
    rs = crack_skeleton(crack_mask)
    out["m_crack_len_rev_nofilter_um"] = skeleton_length(rs, "kulpa") * um_px
    min_px = max(10, int(15.0 / um_px))
    if rs.any():
        lab, n = ndi.label(rs, structure=np.ones((3, 3), int))
        sizes = ndi.sum(rs, lab, index=np.arange(1, n + 1))
        keep = np.isin(lab, np.nonzero(sizes >= min_px)[0] + 1)
        rs = rs & keep
    out["m_crack_len_rev_um"] = skeleton_length(rs, "kulpa") * um_px
    # revised analysis stage: counts, edge labels and regularity on the revised centreline
    ra = analyze_cracks_rev(crack_mask.astype(bool), um_px, edge_end)
    out.update({"m_n_cracks_rev": ra["n"], "m_n_edge_rev": ra["n_edge"],
                "m_n_near_only_rev": ra["n_near_only"], "m_n_internal_rev": ra["n_internal"],
                "m_n_near_rev": ra["n_near"], "m_crack_longest_rev_um": ra["longest_um"],
                "m_regular_cv_rev": ra["regular_cv"] if ra["regular_cv"] is not None else np.nan})
    return out


def reference_measure(shapes: dict, roi_h: int, valid_h: int, um_px: float) -> dict:
    """The same quantities computed directly from the manual annotation."""
    crater, _ = RC.rasterise(shapes, roi_h)
    crater = crater[:valid_h]
    lab, n = ndi.label(crater)
    sizes = ndi.sum(crater, lab, index=np.arange(1, n + 1)) if n else np.array([])
    lengths = [RC.polyline_length(c, valid_h) for c in shapes["cracks"]]
    min_px = max(10, int(15.0 / um_px))
    return {
        "r_crack_len_filt_um": float(np.sum([L for L in lengths if L >= min_px])) * um_px,
        "r_n_craters": int(n),
        "r_crater_total_area_um2": float(crater.sum()) * um_px ** 2,
        "r_crater_largest_area_um2": float(sizes.max()) * um_px ** 2 if n else 0.0,
        "r_crater_largest_diam_um": float(np.sqrt(4 * sizes.max() / np.pi)) * um_px if n else 0.0,
        "r_n_cracks": int(sum(1 for L in lengths if L > 0)),
        "r_crack_len_um": float(np.sum(lengths)) * um_px,
    }
