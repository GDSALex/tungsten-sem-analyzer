"""Shared helpers for the first revision of the OLMAT tungsten paper (ANUCENE-D-26-01513).

Everything that touches paths, file names, annotations and image cropping lives here, so
that the manifest, the grouped cross-validation, the evaluation and the figures all read
the data in exactly the same way.

The functions mirror the conventions of the published code (tungsten-sem-analyzer,
commit a7bdb42):
  * the bottom 8 per cent of the 1024 x 768 frame (61 rows) is the SEM metadata bar
    and is cut away, which leaves a 1024 x 707 region of interest;
  * crater polygons are filled as drawn;
  * crack polylines are rasterised with PIL at a width of 15 px for a 1024-px frame.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True          # never leave __pycache__ inside the public repo
for _s in (sys.stdout, sys.stderr):     # the repo prints box-drawing characters
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
from PIL import Image, ImageDraw

# ----------------------------------------------------------------------------- paths
# This folder lives inside the repository, so the code of the paper (model, training,
# dataset, metrics, analysis) is imported from the repository root.  Everything that is
# not distributed with the repository is located with environment variables:
#   TUNGSTEN_WORK   output tree: results/, cv_runs/, arch_runs/, figures/      (default evaluation/work)
#   TUNGSTEN_DATA   folder with images/ (JPEG micrographs) and masks/ (LabelMe JSON)  (default <WORK>/data)
#   TUNGSTEN_ATW    folder with the ATW micrographs                                    (default <DATA>/atw)
#   TUNGSTEN_CKPT   folder with the checkpoint of the first submission, best_model.pth (default <WORK>/checkpoints)
# The micrographs, the annotations and the weights are available on request (see the README).
import os

EVAL = Path(__file__).resolve().parent
REPO = EVAL.parent
REV = Path(os.environ.get("TUNGSTEN_WORK", EVAL / "work"))
DATA = Path(os.environ.get("TUNGSTEN_DATA", REV / "data"))
IMG_DIR = DATA / "images"
MASK_DIR = DATA / "masks"
ATW_DIR = Path(os.environ.get("TUNGSTEN_ATW", DATA / "atw"))
CKPT_DIR = Path(os.environ.get("TUNGSTEN_CKPT", REV / "checkpoints"))
PUBLISHED_CKPT = CKPT_DIR / "best_model.pth"
RESULTS = REV / "results"
CV_RUNS = REV / "cv_runs"
FIGURES = REV / "figures"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

STRIP_FRACTION = 0.08
GT_CRACK_WIDTH = 15
CRATER_LABELS = {"crater", "cratera", "cráter", "vaporization",
                 "collapse", "crater_vaporization", "crater_collapse"}
CRACK_LABELS = {"crack", "grieta", "fracture", "fractura",
                "crack_large", "crack_small"}

# ------------------------------------------------------------------ file-name parsing
_NAME_RE = re.compile(
    r"^(?P<campaign>ALMW|ATW)(?P<spec>\d)-(?P<edge>\d)(?P<pos>[a-d]{1,2})"
    r"(?:-(?P<power>\d+)MW)?(?P<rest>.*?)_(?P<field>\d+)_?$")


def parse_name(stem: str) -> dict:
    """Split an image name such as ALMW3-4c-600MW-2p-HT_03 into its experimental fields."""
    m = _NAME_RE.match(stem)
    if not m:
        raise ValueError(f"unrecognised image name: {stem}")
    rest = m.group("rest") or ""
    pulses = re.search(r"(\d+)p", rest)
    campaign = m.group("campaign")
    spec = f"{campaign}{m.group('spec')}"
    edge = f"{spec}-{m.group('edge')}"
    pos = f"{spec}-{m.group('edge')}{m.group('pos')}"
    return {
        "name": stem,
        "campaign": campaign,
        "specimen": spec,
        "edge": edge,
        "position": pos,
        "field": int(m.group("field")),
        "power_MW_m2": int(m.group("power")) if m.group("power") else None,
        "pulses_in_name": int(pulses.group(1)) if pulses else 1,
        "preheated": "HT" in rest,
    }


def list_images(directory: Path = IMG_DIR) -> list[Path]:
    return sorted(p for p in Path(directory).iterdir()
                  if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})


# ------------------------------------------------------------------------ image I/O
def read_gray(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img


def strip_rows(full_h: int, strip_fraction: float = STRIP_FRACTION) -> int:
    return int(full_h * strip_fraction)


def roi_of(gray_full: np.ndarray, strip_fraction: float = STRIP_FRACTION) -> np.ndarray:
    h = gray_full.shape[0]
    return gray_full[:h - strip_rows(h, strip_fraction), :]


def bar_top_row(gray_full: np.ndarray) -> int:
    """First row of the SEM data bar, located from the bright band at the bottom.

    The standard Zeiss export has a 62-row bar; the CIEMAT export has a 78-row bar
    that starts with a dark rule.
    """
    H = gray_full.shape[0]
    med = np.median(gray_full, axis=1)
    white = med >= 245
    top = H
    for r in range(H - 100, H - 20):
        # first transition into a band that is at least 85 % white down to the bottom
        if white[r] and not white[r - 1] and white[r:].mean() >= 0.85:
            top = r
            break
    k = 0
    while top - 1 >= 0 and gray_full[top - 1].mean() < 40 and k < 2:   # dark rule above
        top -= 1
        k += 1
    return int(top)


# ----------------------------------------------------------------------- annotations
def load_shapes(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    craters, cracks = [], []
    for s in data.get("shapes", []):
        label = s["label"].strip().lower()
        pts = np.asarray(s["points"], dtype=float)
        kind = s.get("shape_type", "polygon")
        if label in CRATER_LABELS and kind == "polygon" and len(pts) >= 3:
            craters.append(pts)
        elif label in CRACK_LABELS and kind in ("linestrip", "line") and len(pts) >= 2:
            cracks.append(pts)
        elif label in CRATER_LABELS or label in CRACK_LABELS:
            raise ValueError(f"{json_path.name}: unexpected shape {label}/{kind}")
    return {"width": int(data.get("imageWidth")), "height": int(data.get("imageHeight")),
            "craters": craters, "cracks": cracks}


def rasterise(shapes: dict, roi_h: int, crack_width: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Crater and crack masks in ROI coordinates, drawn exactly as in metrics.load_annotation."""
    w, h = shapes["width"], shapes["height"]
    if crack_width is None:
        crack_width = max(7, int(GT_CRACK_WIDTH * max(w, h) / 1024))
    cimg = Image.new("L", (w, h), 0)
    kimg = Image.new("L", (w, h), 0)
    dc, dk = ImageDraw.Draw(cimg), ImageDraw.Draw(kimg)
    for pts in shapes["craters"]:
        dc.polygon([tuple(p) for p in pts], fill=255)
    for pts in shapes["cracks"]:
        dk.line([tuple(p) for p in pts], fill=255, width=crack_width)
    crater = np.array(cimg)[:roi_h] > 127
    crack = np.array(kimg)[:roi_h] > 127
    return crater, crack


def crack_centreline(shapes: dict, roi_h: int) -> np.ndarray:
    """One-pixel, 8-connected rendering of the annotated crack polylines (ROI coordinates)."""
    w, h = shapes["width"], shapes["height"]
    img = np.zeros((h, w), np.uint8)
    for pts in shapes["cracks"]:
        p = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [p], isClosed=False, color=1, thickness=1, lineType=cv2.LINE_8)
    return img[:roi_h].astype(bool)


def polyline_length(pts: np.ndarray, roi_h: int | None = None) -> float:
    """Euclidean length of a polyline in pixels, optionally clipped to the ROI rows."""
    pts = np.asarray(pts, float)
    if roi_h is None:
        return float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        ya, yb = a[1], b[1]
        if ya >= roi_h and yb >= roi_h:
            continue
        seg = np.hypot(*(b - a))
        if ya < roi_h and yb < roi_h:
            total += seg
        else:  # partially inside: keep the fraction above the strip
            t = (roi_h - ya) / (yb - ya)
            t = min(max(t, 0.0), 1.0)
            total += seg * (t if ya < roi_h else 1.0 - t)
    return float(total)


def polygon_area(pts: np.ndarray) -> float:
    x, y = np.asarray(pts, float).T
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


# ------------------------------------------------------------------------- folds
def load_manifest():
    import pandas as pd
    return pd.read_csv(RESULTS / "manifest_ALMW.csv")


def set_low_priority_off():
    """Remove EcoQoS power throttling from the current process (Windows, same user)."""
    try:
        import ctypes
        import ctypes.wintypes as wt

        class PPT(ctypes.Structure):
            _fields_ = [("Version", wt.ULONG), ("ControlMask", wt.ULONG), ("StateMask", wt.ULONG)]

        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentProcess.restype = wt.HANDLE
        k.SetProcessInformation.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
        h = k.GetCurrentProcess()
        s = PPT(1, 0x1 | 0x4, 0)
        k.SetProcessInformation(h, 4, ctypes.byref(s), ctypes.sizeof(s))
        k.SetPriorityClass.argtypes = [wt.HANDLE, wt.DWORD]
        k.SetPriorityClass(h, 0x8000)  # ABOVE_NORMAL
        # keep the machine awake while this process runs
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    except Exception as exc:  # pragma: no cover - best effort only
        print("  [warn] could not change process priority:", exc)


def worker_init(_worker_id: int) -> None:
    """DataLoader workers are separate processes and must lift the throttling themselves."""
    set_low_priority_off()
