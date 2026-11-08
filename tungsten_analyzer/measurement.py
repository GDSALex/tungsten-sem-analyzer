"""Measurement stage of the revised pipeline (revision 1 of the manuscript).

Everything here works in the pixel coordinates of the original frame.

* Scale.  The pixel size is read from the SEM data bar and accepted only when the
  printed magnification and the printed field width agree within 1 %
  (width / frame width x magnification = 111.65 um for the Zeiss exports of OLMAT).
  When the scale cannot be verified the caller must withhold every physical output,
  unless the user gives the pixel size explicitly.
* Crack centreline.  The binary crack mask is reduced to its skeleton after filling pin
  holes (< 64 px in a 1024-px frame), and terminal branches shorter than half the
  15-px training band (8 px) are pruned.
* Crack length.  Steps between 8-connected skeleton pixels are weighted 0.948 (axial)
  and 1.343 (diagonal), the Dorst-Smeulders weights often quoted after Kulpa (1977).
* Crack labels.  A crack is edge connected when some of its pixels lie above the edge
  row, and near edge when its topmost pixel lies within max(40, 0.08 H) rows of it.
  Regularity is the coefficient of variation of the spacings between the horizontal
  origins of the edge connected and near edge cracks (at least three).
"""

from __future__ import annotations

import math
import re

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import remove_small_holes, skeletonize

SQRT2 = math.sqrt(2.0)
KULPA_AXIAL, KULPA_DIAG = 0.948, 1.343
SPUR_PX_1024 = 8            # half the 15-px drawing width of the annotation, 1024-px frame
HOLE_PX_1024 = 64           # pin holes left by thresholding a band
MIN_CRACK_UM, MIN_CRACK_PX = 15.0, 10
REGULAR_CV_MAX = 0.45
MAG_WIDTH_CONSTANT_UM = 111.65   # verified on 212 images: pixel size x magnification
SCALE_TOLERANCE = 0.01

# Neutral size categories of the largest crater like region (equivalent diameter, um).
# They are size classes only: not calibrated against depth, material loss or any
# other physical endpoint.
SIZE_CATEGORIES = {
    0: "no retained region",
    1: "< 100 um",
    2: "100-600 um",
    3: "600-950 um",
    4: "> 950 um",
}


def size_category(diam_um: float | None) -> int:
    if diam_um is None or diam_um <= 0:
        return 0
    if diam_um < 100:
        return 1
    if diam_um < 600:
        return 2
    if diam_um < 950:
        return 3
    return 4


# ------------------------------------------------------------------- data bar and scale
def locate_data_bar(gray_full: np.ndarray) -> int:
    """First row of the SEM data bar (bright band at the bottom of the frame).

    62-63 rows in the standard Zeiss export, 71-72 or 78 rows in the CIEMAT exports.
    Returns the frame height when no bar is found.
    """
    H = gray_full.shape[0]
    med = np.median(gray_full, axis=1)
    white = med >= 245
    top = H
    for r in range(max(1, H - 100), H - 20):
        if white[r] and not white[r - 1] and white[r:].mean() >= 0.85:
            top = r
            break
    k = 0
    while top - 1 >= 0 and gray_full[top - 1].mean() < 40 and k < 2:   # dark rule above
        top -= 1
        k += 1
    return int(top)


_RE_MAG = re.compile(r"mag\s*[=:]?\s*([0-9]+(?:[.,][0-9]+)?)\s*(k?)\s*x", re.I)
# Tesseract often reads the micro sign as y, p, h or j
_RE_WIDTH = re.compile(r"width\s*[=:]?\s*([0-9]+(?:[.,][0-9]+)?)\s*(mm|[µμuyphj]m)", re.I)


def parse_bar_candidates(text: str):
    """All magnifications (x) and field widths (um) found in an OCR string."""
    mags = [float(m.group(1).replace(",", ".")) * (1000.0 if m.group(2) else 1.0)
            for m in _RE_MAG.finditer(text)]
    widths = [float(w.group(1).replace(",", ".")) * (1000.0 if w.group(2).lower() == "mm" else 1.0)
              for w in _RE_WIDTH.finditer(text)]
    return mags, widths


def parse_bar_text(text: str):
    """First magnification (with its unit, e.g. '1.50 kX') and field width in um, or None."""
    mags, widths = parse_bar_candidates(text)
    return (mags[0] if mags else None), (widths[0] if widths else None)


def check_scale(mag: float | None, width_um: float | None, frame_w: int,
                tol: float = SCALE_TOLERANCE):
    """Accept the pixel size only when magnification and width agree within `tol`."""
    if not mag or not width_um:
        return None, "magnification or width not read"
    um_px = width_um / frame_w
    rel = um_px * mag / MAG_WIDTH_CONSTANT_UM - 1.0
    if abs(rel) > tol:
        return None, f"magnification and width disagree ({100 * rel:+.1f} %)"
    return um_px, "verified (width and magnification agree)"


def _ocr_variants(bar: np.ndarray):
    """Pre-processings of the data bar (those used to verify the 212 images of the paper)."""
    import cv2
    up3 = cv2.resize(bar, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    up4 = cv2.resize(bar, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(up3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if otsu.mean() < 127:
        otsu = 255 - otsu
    yield otsu, "--psm 6"
    yield 255 - otsu, "--psm 6"
    yield up4, "--psm 6"
    yield otsu, "--psm 11"
    yield up4[:, up4.shape[1] // 2:], "--psm 6"


def locate_tesseract() -> str | None:
    """The Tesseract binary: the TESSERACT_CMD environment variable first, then the PATH,
    then the default Windows install location; None when none is found, in which case
    pytesseract keeps its own default."""
    import os
    import shutil
    cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if cmd:
        return cmd
    default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    return default if os.path.isfile(default) else None


def read_scale(gray_full: np.ndarray, bar_top: int | None = None) -> dict:
    """Pixel size from the data bar, accepted only when the magnification and the field
    width read agree within 1 %.

    The bar is read with five pre-processings; among all the magnification and width
    candidates the consistent pair seen most often is kept.  Returns a dict with the
    pixel size (None when no consistent pair is found), its source and the values read.
    """
    from collections import Counter
    out = {"um_per_px": None, "verified": False, "source": None, "magnification": None,
           "width_um": None, "reason": "data bar not read"}
    H, W = gray_full.shape[:2]
    top = locate_data_bar(gray_full) if bar_top is None else bar_top
    if top >= H - 5:
        out["reason"] = "data bar not found"
        return out
    try:
        import pytesseract
        cmd = locate_tesseract()
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
    except ImportError:
        out["reason"] = "pytesseract not available"
        return out
    mags, widths = Counter(), Counter()
    for img, cfg in _ocr_variants(gray_full[top:, :]):
        try:
            text = pytesseract.image_to_string(img, config=cfg)
        except Exception as exc:          # Tesseract binary missing or failing
            out["reason"] = f"OCR failed: {exc}"
            return out
        m, w = parse_bar_candidates(text.replace("\n", " "))
        mags.update(m)
        widths.update(w)
    best = None
    for mag, cm in mags.items():
        for wid, cw in widths.items():
            um_px, _ = check_scale(mag, wid, W)
            if um_px is not None and (best is None or cm + cw > best[0]):
                best = (cm + cw, mag, wid, um_px)
    if best is not None:
        out.update(um_per_px=best[3], verified=True, source="data bar", magnification=best[1],
                   width_um=best[2], reason="verified (width and magnification agree)")
    elif mags and widths:
        mag, wid = mags.most_common(1)[0][0], widths.most_common(1)[0][0]
        out.update(magnification=mag, width_um=wid, reason=check_scale(mag, wid, W)[1])
    else:
        out.update(magnification=mags.most_common(1)[0][0] if mags else None,
                   width_um=widths.most_common(1)[0][0] if widths else None,
                   reason="magnification or width not read")
    return out


# ------------------------------------------------------------------- crack centreline
def skeleton_edges(skel: np.ndarray):
    """Edges of the pixel graph of an 8-connected skeleton, each listed once.

    Diagonal pairs already linked through a common axial neighbour are not joined,
    which avoids counting the corners of L-shaped steps twice.
    """
    s = np.pad(skel.astype(bool), 1)
    out = []
    for dr, dc in ((0, 1), (1, 0)):
        a = s[1:-1, 1:-1] & s[1 + dr:s.shape[0] - 1 + dr, 1 + dc:s.shape[1] - 1 + dc]
        r, c = np.nonzero(a)
        out.append((r, c, r + dr, c + dc, np.zeros(len(r), bool)))
    for dc in (1, -1):
        a = s[1:-1, 1:-1] & s[2:, 1 + dc:s.shape[1] - 1 + dc]
        n1 = s[1:-1, 1 + dc:s.shape[1] - 1 + dc]
        n2 = s[2:, 1:-1]
        a = a & ~(n1 | n2)
        r, c = np.nonzero(a)
        out.append((r, c, r + 1, c + dc, np.ones(len(r), bool)))
    return tuple(np.concatenate([o[i] for o in out]) for i in range(5))


def skeleton_length(skel: np.ndarray, method: str = "kulpa") -> float:
    """Length of a skeleton in pixels: 'count', 'freeman' (1, sqrt 2) or 'kulpa'."""
    if method == "count":
        return float(np.count_nonzero(skel))
    dg = skeleton_edges(skel)[4]
    n_d = int(dg.sum())
    n_a = len(dg) - n_d
    if method == "freeman":
        return n_a + SQRT2 * n_d
    if method == "kulpa":
        return KULPA_AXIAL * n_a + KULPA_DIAG * n_d
    raise ValueError(method)


_K8 = np.ones((3, 3), int)
_K8[1, 1] = 0


def prune_spurs(skel: np.ndarray, max_len: float, n_iter: int = 3) -> np.ndarray:
    """Remove terminal branches shorter than `max_len` pixels (Freeman length)."""
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
        idx = np.arange(1, n + 1)
        has_end = ndi.maximum(endpoint, lab, index=idx).astype(bool)
        touch = ndi.maximum(ndi.binary_dilation(junction, np.ones((3, 3), bool)) & branches,
                            lab, index=idx).astype(bool)
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


def crack_skeleton(mask: np.ndarray) -> np.ndarray:
    """Revised centreline of a crack mask: fill pin holes, skeletonise, prune spurs."""
    m = mask.astype(bool)
    if not m.any():
        return m
    w = m.shape[1]
    m = remove_small_holes(m, max(1, int(round(HOLE_PX_1024 * (w / 1024.0) ** 2))))
    return prune_spurs(skeletonize(m), SPUR_PX_1024 * w / 1024.0)


def analyze_cracks(crack_mask: np.ndarray, um_per_px: float | None, edge_end: int) -> dict:
    """Cracks as components of the revised centreline.

    Lengths are Kulpa lengths, in pixels and, when the pixel size is known, in um.  With
    `um_per_px=None` (scale not verified) the minimum crack length is the 10-px floor only
    and every physical length is None.
    """
    H = crack_mask.shape[0]
    rs = crack_skeleton(crack_mask) if crack_mask.any() else np.zeros(crack_mask.shape, bool)
    min_px = MIN_CRACK_PX if um_per_px is None else max(MIN_CRACK_PX, int(MIN_CRACK_UM / um_per_px))
    lab, n = ndi.label(rs, structure=np.ones((3, 3), int))
    margin = max(40, int(0.08 * H))
    kept = np.zeros(rs.shape, bool)
    cracks, lengths_px = [], []
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() < min_px:
            continue
        kept |= comp
        ys, xs = np.nonzero(comp)
        top = int(ys.min())
        edge_conn = bool((ys < edge_end).any())
        near = edge_conn or top < edge_end + margin
        length_px = skeleton_length(comp, "kulpa")
        lengths_px.append(length_px)
        cracks.append({
            "id": len(cracks) + 1,
            "length_px": round(length_px, 1),
            "length_um": None if um_per_px is None else round(length_px * um_per_px, 1),
            "label": "edge connected" if edge_conn else ("near edge" if near else "internal"),
            "top_row": top,
            "x_origin": float(np.median(xs[ys <= top + 10])),
            "bbox": (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1),
        })
    xs = sorted(c["x_origin"] for c in cracks if c["label"] != "internal")
    cv = None
    if len(xs) >= 3:
        sp = np.diff(xs)
        if sp.mean() > 1:
            cv = float(sp.std() / (sp.mean() + 1e-9))
    to_um =(lambda v: None) if um_per_px is None else (lambda v: round(float(v) * um_per_px, 1))
    return {
        "n_total": len(cracks),
        "n_edge_connected": sum(c["label"] == "edge connected" for c in cracks),
        "n_near_edge": sum(c["label"] == "near edge" for c in cracks),
        "n_internal": sum(c["label"] == "internal" for c in cracks),
        "n_for_regularity": len(xs),
        "total_length_px": round(float(sum(lengths_px)), 1),
        "longest_px": round(float(max(lengths_px, default=0.0)), 1),
        "total_length_um": to_um(sum(lengths_px)),
        "longest_um": to_um(max(lengths_px, default=0.0)),
        "regular_cv": None if cv is None else round(cv, 3),
        "regular": bool(cv is not None and cv < REGULAR_CV_MAX),
        "individual_cracks": cracks,
        "skeleton": kept,
    }


# ------------------------------------------------------------- comparison with a reference
def pixel_length_weights(skel: np.ndarray) -> np.ndarray:
    """Per-pixel share of the Freeman length (half of every incident edge)."""
    w = np.zeros(skel.shape, float)
    r0, c0, r1, c1, dg = skeleton_edges(skel)
    L = np.where(dg, SQRT2, 1.0) / 2.0
    np.add.at(w, (r0, c0), L)
    np.add.at(w, (r1, c1), L)
    w[skel & (w == 0)] = 1.0
    return w


def centreline_scores(ref_centre: np.ndarray, pred_mask: np.ndarray, tau: float = 5.0) -> dict:
    """Centreline precision, recall and F1 within `tau` pixels (length weighted)."""
    ref = ref_centre.astype(bool)
    pred = crack_skeleton(pred_mask) if pred_mask.any() else np.zeros(ref.shape, bool)
    if pred_mask.any() and not pred.any():
        pred = pred_mask.astype(bool)
    if not ref.any() or not pred.any():
        p = 0.0 if pred.any() else float("nan")
        r = 0.0 if ref.any() else float("nan")
        f = 0.0 if (ref.any() or pred.any()) else float("nan")
        return {"precision": p, "recall": r, "f1": f}
    wp, wr = pixel_length_weights(pred), pixel_length_weights(ref)
    p = float(wp[ndi.distance_transform_edt(~ref) <= tau].sum() / wp.sum())
    r = float(wr[ndi.distance_transform_edt(~pred) <= tau].sum() / wr.sum())
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f}
