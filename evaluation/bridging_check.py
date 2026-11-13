"""What the gap bridging of the deployed pipeline adds (reviewer comment 5).

For every out-of-fold prediction of configuration A (both repetitions of the edge-grouped
scheme) the crack mask before bridging (multiscale fusion + ridge enhancement + hysteresis,
variant ms_hy) is compared with the deployed mask (the same plus oriented closing with
15-px lines and removal of specks).  The centreline of the deployed mask is split into the
part lying on pixels added by the bridging and the rest, and the length-weighted fraction
of each part within tau = 5 px of the annotated centreline is computed.  The analysis
stage (revised estimator) is run on both masks to measure the effect of bridging on crack
count, total length and regularity.

Output: results/bridging_records.csv, results/summary_bridging.csv
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402
import eval_lib as EL     # noqa: E402

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

TAU = 5.0


def ctx(name):
    g = RC.read_gray(RC.IMG_DIR / f"{name}.jpg")
    H = g.shape[0]
    roi_h = H - RC.strip_rows(H)
    v = min(roi_h, RC.bar_top_row(g))
    shapes = RC.load_shapes(RC.MASK_DIR / f"{name}.json")
    centre = RC.crack_centreline(shapes, roi_h)[:v]
    return g[:v], v, centre


def worker(run_dir):
    RC.set_low_priority_off()
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "fold_info.json").read_text())
    sc = pd.read_csv(RC.RESULTS / "scale_verified.csv").set_index("name")["um_px"]
    recs = []
    for n in info["test"]:
        z = np.load(run_dir / "probs_test" / f"{n}.npz")
        glob = z["glob"].astype(np.float32)
        tile = z["tile"].astype(np.float32) if bool(z["has_tile"]) else None
        roi, v, centre = ctx(n)
        um = float(sc[n])
        vs = EL.variants(glob, tile)
        hy = vs["ms_hy"][1][:v].astype(bool)
        dep = vs["deployed"][1][:v].astype(bool)
        added = dep & ~hy
        sk = EL.crack_skeleton(dep)
        w = EL.pixel_length_weights(sk) if sk.any() else np.zeros(sk.shape)
        on_added = sk & added
        d_ref = ndi.distance_transform_edt(~centre) if centre.any() else np.full(sk.shape, np.inf)
        near = d_ref <= TAU
        L_add, L_rest = float(w[on_added].sum()), float(w[sk & ~added].sum())
        r = {"name": n, "scheme": info["scheme"], "fold": info["fold"], "um_px": um,
             "ref_has_crack": bool(centre.any()),
             "hy_px": int(hy.sum()), "dep_px": int(dep.sum()), "added_px": int(added.sum()),
             "cl_len_added_px": L_add, "cl_len_rest_px": L_rest,
             "cl_len_added_near_px": float(w[on_added & near].sum()),
             "cl_len_rest_near_px": float(w[sk & ~added & near].sum())}
        with contextlib.redirect_stdout(io.StringIO()):
            edge_end, _ = EL.INF.compute_zones_dynamic(roi, bottom_pct=0.02)
        for tag, m in (("hy", hy), ("dep", dep)):
            ra = EL.analyze_cracks_rev(m, um, edge_end)
            r[f"{tag}_n_cracks"] = ra["n"]
            r[f"{tag}_len_um"] = ra["total_um"]
            r[f"{tag}_n_near"] = ra["n_near"]
            r[f"{tag}_cv"] = ra["regular_cv"] if ra["regular_cv"] is not None else np.nan
            r[f"{tag}_regular"] = ra["regular"]
        recs.append(r)
    return recs


def main():
    runs = sorted(str(p) for p in RC.CV_RUNS.glob("A_edge5_r*_f*") if (p / "done.json").exists())
    print(len(runs), "runs")
    with Pool(3, initializer=RC.set_low_priority_off) as pool:
        recs = [r for rr in pool.map(worker, runs) for r in rr]
    df = pd.DataFrame(recs)
    df.to_csv(RC.RESULTS / "bridging_records.csv", index=False)
    pos = df[df.ref_has_crack]
    s = {
        "n_pred": len(df), "n_img": df.name.nunique(),
        "frac_len_added": df.cl_len_added_px.sum() / (df.cl_len_added_px.sum() + df.cl_len_rest_px.sum()),
        "prec_added_pooled": pos.cl_len_added_near_px.sum() / max(pos.cl_len_added_px.sum(), 1e-9),
        "prec_rest_pooled": pos.cl_len_rest_near_px.sum() / max(pos.cl_len_rest_px.sum(), 1e-9),
        "frac_added_on_neg_images": df.loc[~df.ref_has_crack, "cl_len_added_px"].sum() / max(df.cl_len_added_px.sum(), 1e-9),
        "n_with_added": int((df.added_px > 0).sum()),
    }
    both = df[(df.hy_len_um > 0) & (df.dep_len_um > 0)]
    s["len_ratio_dep_hy_median"] = float((both.dep_len_um / both.hy_len_um).median())
    s["len_ratio_dep_hy_p90"] = float((both.dep_len_um / both.hy_len_um).quantile(0.9))
    s["ncr_change_median"] = float((df.dep_n_cracks - df.hy_n_cracks).median())
    s["ncr_fewer"] = int((df.dep_n_cracks < df.hy_n_cracks).sum())
    s["ncr_more"] = int((df.dep_n_cracks > df.hy_n_cracks).sum())
    ev = df[df.hy_cv.notna() | df.dep_cv.notna()]
    s["cv_evaluable_hy"] = int(df.hy_cv.notna().sum())
    s["cv_evaluable_dep"] = int(df.dep_cv.notna().sum())
    s["regular_class_changed"] = int((ev.hy_regular != ev.dep_regular).sum())
    pd.DataFrame([s]).to_csv(RC.RESULTS / "summary_bridging.csv", index=False)
    print(pd.Series(s).to_string())


if __name__ == "__main__":
    main()
