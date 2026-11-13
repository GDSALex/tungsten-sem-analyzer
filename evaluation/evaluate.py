"""Per-image evaluation of the published checkpoint and of the cross-validation models.

Usage
    python evaluate.py published      # maps + legacy reproduction + revised metrics, 99 images
    python evaluate.py classical      # classical pipeline, revised metrics, 99 images
    python evaluate.py cv             # every finished CV run (out-of-fold images)

Every record carries the image name, the evaluated model, the inference variant and all
metrics, so that the tables, the figures and the text of the revision are generated from
these CSV files only.
"""

from __future__ import annotations

import contextlib
import io
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402
import eval_lib as EL    # noqa: E402

import cv2
import numpy as np
import pandas as pd

from tungsten_analyzer import metrics as M
from tungsten_analyzer import inference as INF

OUT = RC.RESULTS
THR_GRID_K = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
THR_GRID_C = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def scale_table() -> dict:
    """Pixel size per image: verified table if present, else the published OCR route."""
    ver = OUT / "scale_verified.csv"
    if ver.exists():
        df = pd.read_csv(ver)
        return dict(zip(df["name"], df["um_px"]))
    man = RC.load_manifest()
    return dict(zip(man["name"], man["repo_auto_scale_um_px"].fillna(0.224)))


_CACHE: dict = {}


def image_context(name: str):
    """Everything about one image that does not depend on the prediction (cached)."""
    if name in _CACHE:
        return _CACHE[name]
    gray_full = RC.read_gray(RC.IMG_DIR / f"{name}.jpg")
    H = gray_full.shape[0]
    roi_h = H - RC.strip_rows(H)
    valid_h = min(roi_h, EL.bar_top_row(gray_full))
    roi_gray = gray_full[:roi_h]
    shapes = RC.load_shapes(RC.MASK_DIR / f"{name}.json")
    ref_c, ref_k = RC.rasterise(shapes, roi_h)
    ref_centre = RC.crack_centreline(shapes, roi_h)
    # edge row of the analysis stage (the classical pipeline analyses only the rows below it)
    with contextlib.redirect_stdout(io.StringIO()):
        edge_end, _ = INF.compute_zones_dynamic(roi_gray[:valid_h], bottom_pct=0.02)
    ctx = dict(gray_full=gray_full, roi_gray=roi_gray, roi_h=roi_h, valid_h=valid_h,
               shapes=shapes, ref_c=ref_c[:valid_h], ref_k=ref_k[:valid_h],
               ref_centre=ref_centre[:valid_h], edge_end=int(edge_end))
    _CACHE[name] = ctx
    return ctx


def reference_measurements(name: str, um_px: float) -> dict:
    ctx = image_context(name)
    v = ctx["valid_h"]
    out = EL.reference_measure(ctx["shapes"], ctx["roi_h"], v, um_px)
    # annotated centreline length above the edge row (outside the classical analysis region)
    rc = ctx["ref_centre"]
    if rc.any():
        w = EL.pixel_length_weights(rc)
        out["r_k_len_px"] = float(w.sum())
        out["r_k_len_above_edge_px"] = float(w[:ctx["edge_end"]].sum())
    else:
        out["r_k_len_px"] = out["r_k_len_above_edge_px"] = 0.0
    # the annotation passed through the same analysis stage as the predictions
    with contextlib.redirect_stdout(io.StringIO()):
        proc = EL.measure(ctx["ref_c"], ctx["ref_k"], ctx["roi_gray"][:v], um_px)
    out.update({k.replace("m_", "rp_", 1): val for k, val in proc.items()
                if k.startswith("m_")})
    return out


def score_masks(name: str, crater: np.ndarray, crack: np.ndarray, um_px: float,
                with_measure: bool) -> dict:
    ctx = image_context(name)
    v = ctx["valid_h"]
    crater, crack = crater[:v].astype(bool), crack[:v].astype(bool)
    rec = {}
    rec.update(EL.crater_scores(ctx["ref_c"], crater))
    rec.update(EL.crack_scores(ctx["ref_k"], ctx["ref_centre"], crack, um_px=um_px))
    rec["pred_bar_px"] = 0
    # the same scores inside the region analysed by both methods (rows below the edge row)
    e = ctx["edge_end"]
    cr = EL.crater_scores(ctx["ref_c"][e:], crater[e:])
    rec.update(c_iou_roi=cr["c_iou"], c_f1_roi=cr["c_f1"], crater_ref_roi=cr["crater_ref"],
               crater_pred_roi=cr["crater_pred"])
    ks = EL.crack_scores(ctx["ref_k"][e:], ctx["ref_centre"][e:], crack[e:], taus_px=(5,), um_px=None)
    rec.update(k_cl_F1_5px_roi=ks["k_cl_F1_5px"], crack_ref_roi=ks["crack_ref"],
               crack_pred_roi=ks["crack_pred"], edge_end_ctx=e)
    if with_measure:
        with contextlib.redirect_stdout(io.StringIO()):
            rec.update(EL.measure(crater, crack, ctx["roi_gray"][:v], um_px))
    return rec


def eval_maps(name: str, glob: np.ndarray, tile: np.ndarray | None, um_px: float,
              thr_c=0.5, thr_k=0.25, which=("net", "ms", "ms_hy", "deployed"),
              measure_for=("net", "deployed")) -> list[dict]:
    recs = []
    vs = EL.variants(glob, tile, thr_c=thr_c, thr_k=thr_k)
    for var in which:
        crater, crack = vs[var]
        r = score_masks(name, crater, crack, um_px, var in measure_for)
        r.update(name=name, variant=var, thr_c=thr_c, thr_k=thr_k)
        recs.append(r)
    return recs


# ======================================================================= published
def published_maps():
    """Probability maps of the published checkpoint for all 99 images (cached on disk)."""
    import torch
    from tungsten_analyzer.model import build_unet
    from tungsten_analyzer.inference import UNetPredictor
    out_dir = OUT / "published_probs"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [p.stem for p in RC.list_images()]
    todo = [n for n in names if not (out_dir / f"{n}.npz").exists()]
    if todo:
        device = torch.device("cuda")
        ck = torch.load(RC.PUBLISHED_CKPT, map_location=device, weights_only=False)
        model = build_unet("resnet34", None, 2).to(device)
        model.load_state_dict(ck["model_state"])
        model.eval()
        pred = UNetPredictor.__new__(UNetPredictor)
        pred.model, pred.device, pred.img_size = model, device, 512
        with torch.no_grad():
            for n in todo:
                gray = RC.roi_of(RC.read_gray(RC.IMG_DIR / f"{n}.jpg"))
                rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                H, W = gray.shape
                g = pred._predict_single(rgb, H, W)
                t = pred._predict_tiled(rgb, H, W)
                np.savez_compressed(out_dir / f"{n}.npz", glob=g.astype(np.float16),
                                    tile=(t if t is not None else np.zeros_like(g)).astype(np.float16),
                                    has_tile=t is not None)
    return out_dir


def load_maps(path: Path):
    z = np.load(path)
    glob = z["glob"].astype(np.float32)
    tile = z["tile"].astype(np.float32) if bool(z["has_tile"]) else None
    return glob, tile


def legacy_reproduction(maps_dir: Path) -> pd.DataFrame:
    """Recompute the submitted Table 1 exactly as reproduce_table1.py does."""
    man = RC.load_manifest().set_index("name")
    rows = []
    for name in man.index:
        glob, _ = load_maps(maps_dir / f"{name}.npz")
        img = RC.read_gray(RC.IMG_DIR / f"{name}.jpg")
        full_h, full_w = img.shape
        h = full_h - int(full_h * M.METADATA_STRIP_FRACTION)
        gt_c, gt_k = M.load_annotation(RC.MASK_DIR / f"{name}.json", h, full_w,
                                       full_height=full_h, full_width=full_w)
        for thr_k in (0.25, 0.30):
            rec = M.score_image(gt_c, gt_k, glob[0] > 0.5, glob[1] > thr_k)
            # strict width-normalised crack IoU, the 0.543 figure of the manuscript
            pk = M.normalise_crack_width(glob[1] > thr_k)
            rec["crack_iou_strict_wn"] = M.iou(gt_k, pk) if gt_k.any() else float("nan")
            rec.update(name=name, thr_k=thr_k, split=man.loc[name, "orig_split"])
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "legacy_table1_records.csv", index=False)
    summ = []
    for thr_k in (0.25, 0.30):
        for subset in ("all", "train", "validation"):
            d = df[df.thr_k == thr_k]
            if subset != "all":
                d = d[d.split == subset]
            s = M.summarise(d.to_dict("records"))
            s.update(thr_k=thr_k, subset=subset,
                     crack_iou_strict_wn_median=float(d.loc[d.has_crack, "crack_iou_strict_wn"].median()))
            summ.append(s)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(OUT / "legacy_table1_summary.csv", index=False)
    return sdf


def _published_worker(args):
    name, maps_dir, um_px = args
    RC.set_low_priority_off()
    glob, tile = load_maps(Path(maps_dir) / f"{name}.npz")
    recs = eval_maps(name, glob, tile, um_px)
    ref = reference_measurements(name, um_px)
    for r in recs:
        r.update(ref)
        r["um_px"] = um_px
    return recs


def run_published():
    maps_dir = published_maps()
    print(legacy_reproduction(maps_dir).round(3).to_string())
    man = RC.load_manifest()
    sc = scale_table()
    jobs = [(n, str(maps_dir), float(sc[n])) for n in man["name"]]
    with Pool(3, initializer=RC.set_low_priority_off) as pool:
        recs = [r for rr in pool.map(_published_worker, jobs) for r in rr]
    df = pd.DataFrame(recs).merge(man[["name", "orig_split", "specimen", "edge", "position",
                                       "in_GA_set"]], on="name")
    df["model"] = "published"
    df.to_csv(OUT / "published_records.csv", index=False)
    print("published records:", len(df))


# ======================================================================= classical
def _classical_worker(args):
    name, um_px = args
    RC.set_low_priority_off()
    sys.path.insert(0, str(RC.REPO / "scripts"))
    import reproduce_table1 as RT
    ctx = image_context(name)
    roi = ctx["roi_gray"]
    with contextlib.redirect_stdout(io.StringIO()):
        c, k = RT.classical_prediction(roi, um_px)
    r = score_masks(name, c, k, um_px, True)
    r.update(reference_measurements(name, um_px))
    r.update(name=name, variant="classical", um_px=um_px)
    return r


def run_classical():
    man = RC.load_manifest()
    sc = scale_table()
    jobs = [(n, float(sc[n])) for n in man["name"]]
    with Pool(3, initializer=RC.set_low_priority_off) as pool:
        recs = pool.map(_classical_worker, jobs)
    df = pd.DataFrame(recs).merge(man[["name", "orig_split", "specimen", "edge", "position",
                                       "in_GA_set"]], on="name")
    df["model"] = "classical"
    df.to_csv(OUT / "classical_records.csv", index=False)
    print("classical records:", len(df))


# ============================================================================== CV
def _select_thresholds(run_dir: Path, sc: dict):
    """Crater and crack thresholds of the network output chosen on the inner validation set."""
    names = json.loads((run_dir / "fold_info.json").read_text())["inner_val"]
    maps = {n: load_maps(run_dir / "probs_innerval" / f"{n}.npz") for n in names}
    best_k, best_kv = 0.25, -1
    for t in THR_GRID_K:
        vals = []
        for n, (g, _) in maps.items():
            ctx = image_context(n)
            v = ctx["valid_h"]
            pred = (g[1] > t)[:v]
            if not ctx["ref_k"].any() and not pred.any():
                continue
            s = EL.crack_scores(ctx["ref_k"], ctx["ref_centre"], pred, taus_px=(5,), um_px=None)
            vals.append(s["k_cl_F1_5px"])
        m = float(np.mean(vals)) if vals else -1
        if m > best_kv:
            best_k, best_kv = t, m
    best_c, best_cv = 0.5, -1
    for t in THR_GRID_C:
        vals = []
        for n, (g, _) in maps.items():
            ctx = image_context(n)
            v = ctx["valid_h"]
            pred = (g[0] > t)[:v]
            if not ctx["ref_c"].any() and not pred.any():
                continue
            vals.append(EL.crater_scores(ctx["ref_c"], pred)["c_f1"])
        m = float(np.mean(vals)) if vals else -1
        if m > best_cv:
            best_c, best_cv = t, m
    return best_c, best_k


def _cv_worker(run_dir: str):
    RC.set_low_priority_off()
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "fold_info.json").read_text())
    sc = scale_table()
    thr_c, thr_k = _select_thresholds(run_dir, sc)
    recs = []
    for n in info["test"]:
        glob, tile = load_maps(run_dir / "probs_test" / f"{n}.npz")
        um = float(sc[n])
        rr = eval_maps(n, glob, tile, um)
        rr += [dict(r, variant="net_sel") for r in
               eval_maps(n, glob, tile, um, thr_c=thr_c, thr_k=thr_k, which=("net",),
                         measure_for=())]
        ref = reference_measurements(n, um)
        for r in rr:
            r.update(ref)
            r.update(um_px=um, config=info["config"], scheme=info["scheme"], fold=info["fold"],
                     seed=info["seed"], sel_thr_c=thr_c, sel_thr_k=thr_k)
        recs += rr
    return recs


def run_cv():
    runs = sorted(str(p) for p in RC.CV_RUNS.iterdir() if (p / "done.json").exists())
    done_csv = OUT / "cv_records.csv"
    have = set()
    old = None
    if done_csv.exists():
        old = pd.read_csv(done_csv)
        have = set(zip(old.config, old.scheme, old.fold))
    todo = []
    for r in runs:
        info = json.loads((Path(r) / "fold_info.json").read_text())
        if (info["config"], info["scheme"], info["fold"]) not in have:
            todo.append(r)
    print(f"{len(runs)} finished runs, {len(todo)} to evaluate")
    if not todo:
        return
    with Pool(min(3, len(todo)), initializer=RC.set_low_priority_off) as pool:
        recs = [r for rr in pool.map(_cv_worker, todo) for r in rr]
    man = RC.load_manifest()
    df = pd.DataFrame(recs).merge(man[["name", "orig_split", "specimen", "edge", "position",
                                       "in_GA_set", "has_crater", "has_crack"]], on="name")
    if old is not None:
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(done_csv, index=False)
    print("cv records:", len(df))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "published"
    {"published": run_published, "classical": run_classical, "cv": run_cv}[what]()
