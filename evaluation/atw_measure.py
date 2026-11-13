"""Deployed pipeline and analysis stage on the whole ATW series (external application).

The submitted checkpoint is applied to the 113 ATW images with the frozen deployed
pipeline (and, for comparison, with the multiscale output before hysteresis and bridging),
and the analysis stage is run with the verified pixel size of every image.  There is no
reference annotation for these images, so the output is descriptive only.

Output: results/atw_records.csv  (one row per image and variant)
"""
from __future__ import annotations

import contextlib
import io
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402
import eval_lib as EL     # noqa: E402

import cv2
import numpy as np
import pandas as pd

from tungsten_analyzer import inference as INF

OUT = RC.RESULTS / "published_probs_atw"


def image_path(name):
    p = RC.ATW_DIR / f"{name}.jpg"
    return p if p.exists() else RC.IMG_DIR / f"{name}.jpg"


def make_maps(names):
    import torch
    from tungsten_analyzer.model import build_unet
    from tungsten_analyzer.inference import UNetPredictor
    todo = [n for n in names if not (OUT / f"{n}.npz").exists()]
    if not todo:
        return
    OUT.mkdir(exist_ok=True)
    dev = torch.device("cuda")
    ck = torch.load(RC.PUBLISHED_CKPT, map_location=dev, weights_only=False)
    model = build_unet("resnet34", None, 2).to(dev)
    model.load_state_dict(ck["model_state"])
    model.eval()
    pr = UNetPredictor.__new__(UNetPredictor)
    pr.model, pr.device, pr.img_size = model, dev, 512
    with torch.no_grad():
        for n in todo:
            gray = RC.roi_of(RC.read_gray(image_path(n)))
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            H, W = gray.shape
            g = pr._predict_single(rgb, H, W)
            t = pr._predict_tiled(rgb, H, W)
            np.savez_compressed(OUT / f"{n}.npz", glob=g.astype(np.float16),
                                tile=(t if t is not None else np.zeros_like(g)).astype(np.float16),
                                has_tile=t is not None)
    del model
    torch.cuda.empty_cache()


def worker(args):
    name, um = args
    RC.set_low_priority_off()
    g = RC.read_gray(image_path(name))
    H = g.shape[0]
    roi_h = H - RC.strip_rows(H)
    v = min(roi_h, RC.bar_top_row(g))
    z = np.load(OUT / f"{name}.npz")
    glob = z["glob"].astype(np.float32)
    tile = z["tile"].astype(np.float32) if bool(z["has_tile"]) else None
    vs = EL.variants(glob, tile)
    recs = []
    for var in ("ms", "deployed"):
        crater, crack = (m[:v].astype(bool) for m in vs[var])
        with contextlib.redirect_stdout(io.StringIO()):
            r = EL.measure(crater, crack, g[:v], um)
        r.update(name=name, variant=var, um_px=um, crater_pred=bool(crater.any()),
                 crack_pred=bool(crack.any()))
        recs.append(r)
    return recs


def main():
    RC.set_low_priority_off()
    man = pd.read_csv(RC.RESULTS / "manifest_ATW.csv")
    sc = pd.read_csv(RC.RESULTS / "scale_verified.csv").set_index("name")["um_px"]
    names = sorted(man["name"])
    make_maps(names)
    with Pool(3, initializer=RC.set_low_priority_off) as pool:
        recs = [r for rr in pool.map(worker, [(n, float(sc[n])) for n in names]) for r in rr]
    df = pd.DataFrame(recs).merge(man[["name", "specimen", "edge", "position", "power_MW_m2",
                                       "pulses_in_name"]], on="name")
    df.to_csv(RC.RESULTS / "atw_records.csv", index=False)
    d = df[df.variant == "deployed"]
    print(len(d), "images;", int((d.m_n_craters > 0).sum()), "with retained craters;",
          int((d.m_n_cracks_rev > 0).sum()), "with retained cracks;",
          int(d.m_regular_cv_rev.notna().sum()), "with >= 3 near-edge cracks")


if __name__ == "__main__":
    main()
