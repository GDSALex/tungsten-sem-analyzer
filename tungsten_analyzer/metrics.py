"""Evaluation protocol used for every number reported in the paper.

Everything about how a prediction is compared against the manual annotation lives
here, so that the classical pipeline and the network are measured by exactly the
same rules and neither is favoured by the way it is scored.

Ground truth
------------
Annotations are read from the LabelMe JSON files. Craters are closed polygons and
are filled as they are. Cracks are open polylines that follow the centreline of the
fracture, which is the only annotation that can be made consistently at these
magnifications: a thermal fatigue crack is one to two pixels across, its apparent
width shifts with the local topographic contrast of the microscope, and two
annotators would not agree on where its edges lie. The centreline, in contrast, is
unambiguous.

A pixel-wise metric still needs a mask, so each polyline is rasterised at a fixed
width of ``GT_CRACK_WIDTH`` pixels for a 1024-pixel-wide frame, scaled
proportionally for other sizes. The same rasterisation is used to build the training
targets in :mod:`tungsten_analyzer.dataset`, so annotation and targets always agree.
That width is a drawing convention and not a measurement.

Comparing crack masks
---------------------
Two steps follow from the annotation being a centreline.

First, the width of the prediction is taken out of the comparison. Whatever a method
returns on the crack channel is reduced to its skeleton and redrawn at the same
width as the annotation before anything is measured. Without this step a good part
of the comparison would be measuring line thickness rather than detection, which
matters here because the two methods return lines of very different width: the
network was trained on the 15-pixel rasterisation, while the classical detector
returns its skeleton dilated by a single pixel.

Second, a tolerance is applied. Both masks are dilated by ``TOLERANCE_PX`` pixels
before the overlap is counted, which is the centreline-tolerance criterion standard
in crack, road and vessel segmentation. Measured from the annotated centreline the
combined effect is a tolerance of roughly twenty pixels, between 4 and 40 micrometres
depending on the field of view.

The question the resulting score answers is whether the crack was found and whether
it sits where the annotator drew it. That is what matters for damage quantification,
and it is also what matters downstream, since crack length and crack density are
derived from the skeleton of the prediction and depend on position and connectivity
rather than on width.

Craters need none of this. They are annotated as closed polygons following a real,
measurable boundary and are two to three orders of magnitude larger than the
uncertainty of that boundary, so they are compared with no tolerance at all.

Aggregation
-----------
Every image is scored on its own and the paper reports the median and the mean of
those per-image scores. An overlap score is undefined against an empty reference, so
an image only enters the statistic of a class when it carries at least one annotated
region of that class. Nothing else is filtered out. In particular, images where a
method produces no overlap at all are kept and enter as zeros, since dropping them
would amount to discarding precisely the failures.

What that exclusion does leave out is the behaviour of each method on a clean
surface, which for a screening tool matters as much as the overlap. Those cases are
therefore reported separately at the level of the image, through
:func:`image_level_counts`, as a detection rate on annotated images and a false alarm
rate on images with nothing annotated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from skimage.morphology import binary_dilation, disk, skeletonize

__all__ = [
    "GT_CRACK_WIDTH",
    "TOLERANCE_PX",
    "METADATA_STRIP_FRACTION",
    "load_annotation",
    "normalise_crack_width",
    "iou",
    "dice",
    "tolerant_iou",
    "score_image",
    "summarise",
    "image_level_counts",
]

# Width, in pixels and referred to a 1024-pixel-wide frame, at which an annotated
# crack centreline is rasterised. It is scaled with the frame and it must match the
# value used to build the training targets in dataset.py.
GT_CRACK_WIDTH = 15

# Extra margin allowed on both masks when the crack overlap is counted.
TOLERANCE_PX = 5

# Fraction of the frame height occupied by the SEM metadata bar, cropped away before
# anything is measured.
METADATA_STRIP_FRACTION = 0.08

# Label vocabulary accepted in the LabelMe files. Several spellings are tolerated
# because the annotation campaign spanned more than one session; the dataset of the
# paper only uses "crater" and "crack".
CRATER_LABELS = {"crater", "cratera", "cráter", "vaporization",
                 "collapse", "crater_vaporization", "crater_collapse"}
CRACK_LABELS = {"crack", "grieta", "fracture", "fractura",
                "crack_large", "crack_small"}


# ─────────────────────────────────────────────────────────────────────────────
#  Ground truth
# ─────────────────────────────────────────────────────────────────────────────

def load_annotation(json_path: str | Path,
                    height: int,
                    width: int,
                    full_height: int | None = None,
                    full_width: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterise a LabelMe annotation into a crater mask and a crack mask.

    The masks are drawn at the size recorded in the JSON and then cropped to
    ``(height, width)``, which is the frame with the metadata bar removed. Pass
    ``full_height`` and ``full_width`` when they differ from the cropped size.

    Returns two boolean arrays, craters first.
    """
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    img_w = data.get("imageWidth", full_width or width)
    img_h = data.get("imageHeight", full_height or height)

    canvas_crater = Image.new("L", (img_w, img_h), 0)
    canvas_crack = Image.new("L", (img_w, img_h), 0)
    draw_crater = ImageDraw.Draw(canvas_crater)
    draw_crack = ImageDraw.Draw(canvas_crack)

    crack_width = max(7, int(GT_CRACK_WIDTH * max(img_w, img_h) / 1024))

    for shape in data.get("shapes", []):
        label = shape["label"].strip().lower()
        points = [(float(x), float(y)) for x, y in shape["points"]]
        kind = shape.get("shape_type", "polygon")

        if label in CRATER_LABELS:
            draw = draw_crater
        elif label in CRACK_LABELS:
            draw = draw_crack
        else:
            continue

        if kind == "polygon" and len(points) >= 3:
            draw.polygon(points, fill=255)
        elif kind in ("linestrip", "line") and len(points) >= 2:
            draw.line(points, fill=255, width=crack_width)
        elif kind == "rectangle" and len(points) >= 2:
            (x0, y0), (x1, y1) = points[0], points[-1]
            draw.rectangle([x0, y0, x1, y1], fill=255)
        elif kind == "circle" and len(points) >= 2:
            (cx, cy), (px, py) = points[0], points[1]
            r = float(np.hypot(px - cx, py - cy))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

    crater = np.array(canvas_crater)
    crack = np.array(canvas_crack)

    # Crop the rows so that the mask matches the frame with the metadata bar
    # removed, and only resample the columns in the unusual case of a width
    # mismatch between the annotation and the image on disk.
    if crater.shape[0] > height:
        crater = crater[:height, :]
        crack = crack[:height, :]
    if crater.shape[1] != width:
        import cv2
        crater = cv2.resize(crater, (width, crater.shape[0]), interpolation=cv2.INTER_NEAREST)
        crack = cv2.resize(crack, (width, crack.shape[0]), interpolation=cv2.INTER_NEAREST)

    return crater > 127, crack > 127


# ─────────────────────────────────────────────────────────────────────────────
#  Overlap measures
# ─────────────────────────────────────────────────────────────────────────────

def normalise_crack_width(mask: np.ndarray, width: int | None = None) -> np.ndarray:
    """Reduce a crack mask to its skeleton and redraw it at the annotation width.

    This is what removes line thickness from the comparison. An empty mask is
    returned untouched, since skeletonising it is meaningless.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return m
    radius = (GT_CRACK_WIDTH if width is None else width) // 2
    return binary_dilation(skeletonize(m), disk(radius))


def iou(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Intersection over union between two binary masks."""
    ref = np.asarray(reference, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    tp = float((ref & pred).sum())
    fp = float((~ref & pred).sum())
    fn = float((ref & ~pred).sum())
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0


def dice(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Dice coefficient, also known as F1, between two binary masks."""
    ref = np.asarray(reference, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    tp = float((ref & pred).sum())
    denom = 2 * tp + float((~ref & pred).sum()) + float((ref & ~pred).sum())
    return (2 * tp) / denom if denom > 0 else 0.0


def tolerant_iou(reference: np.ndarray,
                 prediction: np.ndarray,
                 tolerance_px: int = TOLERANCE_PX) -> float:
    """Centreline-tolerance IoU for thin structures.

    Both masks are dilated by ``tolerance_px`` before the overlap is counted, so that
    a prediction displaced by a few pixels is not penalised as a complete miss. A
    predicted pixel counts as a true positive when it falls within the tolerance of
    the reference, and vice versa; the two directions are averaged.

    Two empty masks are a perfect match and score 1. One empty mask against a
    non-empty one scores 0, which is what makes a silent method score zero on an
    image that does contain a crack.
    """
    ref = np.asarray(reference, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)

    if not ref.any() and not pred.any():
        return 1.0
    if not ref.any() or not pred.any():
        return 0.0

    selem = disk(tolerance_px)
    ref_d = binary_dilation(ref, selem)
    pred_d = binary_dilation(pred, selem)

    tp = float((ref_d & pred).sum() + (pred_d & ref).sum()) / 2.0
    fp = float((pred & ~ref_d).sum())
    fn = float((ref & ~pred_d).sum())
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0


def _dice_from_iou(j: float) -> float:
    """Dice implied by an IoU computed from the same counts."""
    return 2 * j / (1 + j) if j > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Per-image scoring and aggregation
# ─────────────────────────────────────────────────────────────────────────────

def score_image(gt_crater: np.ndarray,
                gt_crack: np.ndarray,
                pred_crater: np.ndarray,
                pred_crack: np.ndarray) -> Dict[str, float | bool]:
    """Score one image under the protocol of the paper.

    The returned record carries the four metrics together with the two flags that
    say whether each class was annotated, which is what decides later on whether the
    image enters each statistic.
    """
    gt_crater = np.asarray(gt_crater, dtype=bool)
    gt_crack = np.asarray(gt_crack, dtype=bool)
    pred_crater = np.asarray(pred_crater, dtype=bool)
    pred_crack = np.asarray(pred_crack, dtype=bool)

    crack_norm = normalise_crack_width(pred_crack)
    crack_iou = tolerant_iou(gt_crack, crack_norm)

    return {
        "has_crater": bool(gt_crater.any()),
        "has_crack": bool(gt_crack.any()),
        "predicted_crater": bool(pred_crater.any()),
        "predicted_crack": bool(pred_crack.any()),
        "crater_iou": iou(gt_crater, pred_crater),
        "crater_f1": dice(gt_crater, pred_crater),
        "crack_iou": crack_iou,
        "crack_f1": _dice_from_iou(crack_iou),
    }


def summarise(records: Sequence[Dict]) -> Dict[str, float | int]:
    """Median and mean of the per-image scores, class by class.

    Only images carrying an annotation of a class contribute to that class, and the
    counts that went into each figure are returned alongside it.
    """
    out: Dict[str, float | int] = {}
    for cls, flag in (("crater", "has_crater"), ("crack", "has_crack")):
        subset = [r for r in records if r[flag]]
        out[f"n_{cls}"] = len(subset)
        for metric in ("iou", "f1"):
            key = f"{cls}_{metric}"
            values = np.array([r[key] for r in subset], dtype=float)
            out[f"{key}_median"] = float(np.median(values)) if len(values) else float("nan")
            out[f"{key}_mean"] = float(values.mean()) if len(values) else float("nan")
    return out


def image_level_counts(records: Sequence[Dict]) -> Dict[str, int]:
    """Detection and false alarm counts, which the overlap statistics cannot express.

    For each class this reports how many annotated images the method found something
    on, and how many clean images it fired on. The second figure is the one the
    overlap statistics necessarily leave out, and for the classical pipeline it is
    where most of its difference with the network shows up.
    """
    out: Dict[str, int] = {}
    for cls, flag, pred in (("crater", "has_crater", "predicted_crater"),
                            ("crack", "has_crack", "predicted_crack")):
        annotated = [r for r in records if r[flag]]
        clean = [r for r in records if not r[flag]]
        out[f"{cls}_annotated"] = len(annotated)
        out[f"{cls}_detected"] = sum(1 for r in annotated if r[pred])
        out[f"{cls}_clean"] = len(clean)
        out[f"{cls}_false_alarms"] = sum(1 for r in clean if r[pred])
    return out
