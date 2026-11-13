"""Verify the pixel size of every micrograph (reviewer comment 6).

For each image the data bar is read several times with different pre-processing, and the
magnification and the field width are accepted only when they agree with each other
(width x magnification / 1024 must fall within 110.5-112.5 mm, the Zeiss reference width
divided by the frame).  Images where no consistent pair is found are flagged for visual
transcription (manual_scale.csv).  The published route (inference.auto_scale) is then
scored against the verified values: success, silent error (a wrong value returned without
warning) or failure (no value, which in the published code falls back to 0.224 um/px).
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402

import cv2
import numpy as np
import pandas as pd
import pytesseract

import os
import shutil

_cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
if _cmd:
    pytesseract.pytesseract.tesseract_cmd = _cmd

UNIT = r"(mm|[uµμyphHj]m)"


def _variants(bar: np.ndarray):
    up3 = cv2.resize(bar, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    up4 = cv2.resize(bar, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(up3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if otsu.mean() < 127:
        otsu = 255 - otsu
    yield "otsu3_psm6", otsu, "--psm 6"
    yield "inv3_psm6", 255 - otsu, "--psm 6"
    yield "gray4_psm6", up4, "--psm 6"
    yield "otsu3_psm11", otsu, "--psm 11"
    half = up4[:, up4.shape[1] // 2:]
    yield "right4_psm6", half, "--psm 6"


def _parse(text: str):
    t = text.replace(",", ".")
    mags = [float(m.group(1)) * (1000 if m.group(2) else 1)
            for m in re.finditer(r"Mag\s*[=:]?\s*([0-9]+(?:\.[0-9]+)?)\s*(K)?\s*X", t)]
    widths = []
    for m in re.finditer(r"Width\s*[=:]?\s*([0-9]+(?:\.[0-9]+)?)\s*" + UNIT, t):
        v = float(m.group(1))
        widths.append(v * 1000.0 if m.group(2) == "mm" else v)
    heights = []
    for m in re.finditer(r"Height\s*[=:]?\s*([0-9]+(?:\.[0-9]+)?)\s*" + UNIT, t):
        v = float(m.group(1))
        heights.append(v * 1000.0 if m.group(2) == "mm" else v)
    wd = re.search(r"WD\s*[=:]?\s*([0-9]+\.[0-9])\s*mm", t)
    eht = re.search(r"EHT\s*[=:]?\s*([0-9]+\.[0-9]+)\s*kV", t)
    sa = re.search(r"Signal\s*A\s*[=:]?\s*([A-Za-z]+)", t)
    sb = re.search(r"Signal\s*B\s*[=:]?\s*([A-Za-z]+)", t)
    return mags, widths, heights, (float(wd.group(1)) if wd else None), \
        (float(eht.group(1)) if eht else None), (sa.group(1) if sa else None), \
        (sb.group(1) if sb else None)


def verify(path_str: str) -> dict:
    RC.set_low_priority_off()
    p = Path(path_str)
    gray = RC.read_gray(p)
    H, W = gray.shape
    top = RC.bar_top_row(gray)
    bar = gray[top:, :]
    mags, widths, heights, wds, ehts, sas, sbs = Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    for _, img, cfg in _variants(bar):
        txt = pytesseract.image_to_string(img, config=cfg).replace("\n", " ")
        m, w, h, wd, eht, sa, sb = _parse(txt)
        mags.update(m); widths.update(w); heights.update(h)
        if wd: wds[wd] += 1
        if eht: ehts[eht] += 1
        if sa: sas[sa] += 1
        if sb: sbs[sb] += 1
    best = None
    for (mag, cm), (wid, cw) in ((a, b) for a in mags.items() for b in widths.items()):
        if mag <= 0:
            continue
        const = wid * mag / W / 1000.0 * 1000.0   # (um x X / px) -> number close to 111.6
        if 110.5 <= const <= 112.5:
            score = cm + cw
            if best is None or score > best[0]:
                best = (score, mag, wid, const)
    rec = {"name": p.stem, "dir": p.parent.name, "image_w": W, "image_h": H,
           "bar_top": top, "bar_rows": H - top,
           "export_format": "Zeiss" if H - top < 66 else ("CIEMAT-logo" if H - top < 76 else "CIEMAT-plain"),
           "mag_x": best[1] if best else None, "width_um": best[2] if best else None,
           "mag_width_const": best[3] if best else None,
           "height_um": heights.most_common(1)[0][0] if heights else None,
           "wd_mm": wds.most_common(1)[0][0] if wds else None,
           "eht_kV": ehts.most_common(1)[0][0] if ehts else None,
           "signal_a": sas.most_common(1)[0][0] if sas else None,
           "signal_b": sbs.most_common(1)[0][0] if sbs else None,
           "consistent": best is not None,
           "mag_candidates": dict(mags), "width_candidates": dict(widths),
           }
    return rec


def main():
    RC.set_low_priority_off()
    paths = [str(p) for p in RC.list_images(RC.IMG_DIR)] + [str(p) for p in RC.list_images(RC.ATW_DIR)]
    with Pool(4, initializer=RC.set_low_priority_off) as pool:
        recs = pool.map(verify, paths)
    # published route, evaluated once per image in this process (it imports torch)
    from tungsten_analyzer import inference as INF
    import contextlib, io
    for rec, path in zip(recs, paths):
        gray = RC.read_gray(Path(path))
        with contextlib.redirect_stdout(io.StringIO()):
            meta_h = INF.detect_meta_strip(gray)
            rec["repo_um_px"] = INF.auto_scale(gray, meta_h)
            rec["repo_meta_strip_rows"] = meta_h
    df = pd.DataFrame(recs)
    df.to_csv(RC.RESULTS / "ocr_verify_raw.csv", index=False)
    print(df.groupby(["dir", "export_format"]).size())
    print("consistent:", df.groupby("dir")["consistent"].mean())
    print(df.loc[~df.consistent, ["name", "mag_candidates", "width_candidates"]].to_string())


if __name__ == "__main__":
    main()
