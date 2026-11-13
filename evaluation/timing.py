"""Computational cost of the deployed pipeline (reviewer specific comment 6).

1. Stage timings per image on the 212 ALMW and ATW images with the submitted checkpoint:
   image reading, data bar detection and OCR scale reading, global pass with 4-flip TTA,
   tiled pass with 4-flip TTA, fusion + ridge + hysteresis + bridging, analysis stage.
2. End-to-end batch: the repository function analyze_image (which also writes the result
   figures) over the 99 ALMW images, wall-clock time including model loading.

Output: results/timing_stages.csv, results/summary_timing.csv
Run with the GPU otherwise idle.

    python timing.py                                   # submitted checkpoint -> results/
    python timing.py --ckpt <last_model.pth> --out <dir>
        any checkpoint that inference.UNetPredictor loads, for the condensed version the final
        model of the adopted network (arch_runs/unet_r34_ds/all/last_model.pth -> results/v2)
"""
from __future__ import annotations

import contextlib
import io
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402
import eval_lib as EL     # noqa: E402

import cv2
import numpy as np
import pandas as pd
import torch

from tungsten_analyzer import inference as INF


def image_path(name):
    p = RC.IMG_DIR / f"{name}.jpg"
    return p if p.exists() else RC.ATW_DIR / f"{name}.jpg"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(RC.PUBLISHED_CKPT))
    ap.add_argument("--out", default=str(RC.RESULTS))
    a = ap.parse_args()
    ckpt, out = a.ckpt, Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("checkpoint:", ckpt)
    RC.set_low_priority_off()
    names = sorted(pd.read_csv(RC.RESULTS / "manifest_ALMW.csv")["name"]) + \
        sorted(pd.read_csv(RC.RESULTS / "manifest_ATW.csv")["name"])
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        pred = INF.UNetPredictor(ckpt)
    sync()
    t_load = time.perf_counter() - t0
    # warm-up (cuDNN autotuning, CUDA context)
    g = RC.roi_of(RC.read_gray(image_path(names[0])))
    rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    with torch.no_grad():
        for _ in range(2):
            pred._predict_single(rgb, *g.shape)
            pred._predict_tiled(rgb, *g.shape)
    sync()
    rows = []
    for n in names:
        r = {"name": n}
        t = time.perf_counter()
        full = RC.read_gray(image_path(n))
        r["t_read"] = time.perf_counter() - t
        t = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            meta_h = INF.detect_meta_strip(full)
            um = INF.auto_scale(full, meta_h)
        r["t_ocr"] = time.perf_counter() - t
        roi = RC.roi_of(full)
        H, W = roi.shape
        rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        with torch.no_grad():
            sync(); t = time.perf_counter()
            glob = pred._predict_single(rgb, H, W)
            sync(); r["t_global"] = time.perf_counter() - t
            t = time.perf_counter()
            tile = pred._predict_tiled(rgb, H, W)
            sync(); r["t_tiled"] = time.perf_counter() - t
        t = time.perf_counter()
        crater, crack = EL.variants(glob, tile)["deployed"]
        r["t_post"] = time.perf_counter() - t
        t = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            edge_end, art = INF.compute_zones_dynamic(roi, bottom_pct=0.02)
            INF.analyze_craters(crater.copy(), roi, um or 0.224, edge_end, art)
            INF.analyze_cracks(crack, um or 0.224, edge_end)
        r["t_analysis"] = time.perf_counter() - t
        r["W"], r["H"] = W, H
        rows.append(r)
    df = pd.DataFrame(rows)
    df["t_total"] = df[[c for c in df.columns if c.startswith("t_")]].sum(axis=1)
    df.to_csv(out / "timing_stages.csv", index=False)

    # end-to-end batch with the repository function, figures included
    almw = [image_path(n) for n in sorted(pd.read_csv(RC.RESULTS / "manifest_ALMW.csv")["name"])]
    with tempfile.TemporaryDirectory(dir=str(RC.REV / "logs")) as tmp:
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            p2 = INF.UNetPredictor(ckpt)
            for p in almw:
                INF.analyze_image(p2, p, Path(tmp))
        sync()
        t_batch = time.perf_counter() - t0

    try:
        cpu = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        cpu = platform.processor()
    s = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
         "cpu": cpu, "torch": torch.__version__, "n_images": len(df), "t_model_load_s": t_load,
         "batch_almw_n": len(almw), "batch_almw_total_s": t_batch,
         "batch_almw_per_image_s": t_batch / len(almw)}
    for c in [c for c in df.columns if c.startswith("t_")]:
        s[f"{c}_median_s"] = float(df[c].median())
        s[f"{c}_p90_s"] = float(df[c].quantile(0.9))
    s["checkpoint"] = str(ckpt)
    pd.DataFrame([s]).to_csv(out / "summary_timing.csv", index=False)
    print(pd.Series(s).to_string())


if __name__ == "__main__":
    main()
