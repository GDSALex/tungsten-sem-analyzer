"""Complete per-image evaluation of the adopted network, for the condensed manuscript.

    python eval_final.py maps <arch>     # GPU: maps of the final model (scheme 'all') for the
                                         # 99 ALMW images (in sample) and the 113 ATW images
    python eval_final.py eval <arch>     # CPU: records of every out-of-fold run of <arch>
                                         # (edge5_r1, edge5_r2, loso) with measurements and the
                                         # bridging diagnostics, the in-sample records of the
                                         # final model and the analysis of the ATW images

Outputs (results/v2/):
    final_records.csv     one row per (scheme, image, inference variant), out of fold
    insample_records.csv  the same for the final model on its own training images
    bridging_records.csv  what the gap bridging adds, per out-of-fold deployed prediction
    atw_analysis.csv      analysis stage of the deployed pipeline on the ATW images
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
import eval_lib as EL    # noqa: E402
import evaluate as E     # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

RUNS = RC.REV / "arch_runs"
OUT = RC.RESULTS / "v2"
OUT.mkdir(parents=True, exist_ok=True)
SCHEMES = ("edge5_r1", "edge5_r2", "loso")
TAU = 5.0


# ------------------------------------------------------------------------------ GPU part
def cmd_maps(arch: str) -> None:
    import torch
    import arch_train as AT
    run = RUNS / arch / "all"
    device = torch.device("cuda")
    model = AT.build_network(arch).to(device)
    state = torch.load(run / "last_model.pth", map_location=device, weights_only=False)["model_state"]
    model.load_state_dict(state)
    almw = [p.stem for p in RC.list_images(RC.IMG_DIR)]
    atw = [p.stem for p in RC.list_images(RC.ATW_DIR)]
    AT.write_maps(model, almw, RC.IMG_DIR, run / "probs_insample", device)
    AT.write_maps(model, atw, RC.ATW_DIR, run / "probs_atw", device)
    print(f"maps written: {len(almw)} ALMW, {len(atw)} ATW")


# ------------------------------------------------------------------------------ CPU part
def bridging(name: str, glob, tile, um: float) -> dict:
    """Centreline length added by the gap bridging and how much of it lies on the reference."""
    ctx = E.image_context(name)
    v = ctx["valid_h"]
    vs = EL.variants(glob, tile)
    hy = vs["ms_hy"][1][:v].astype(bool)
    dep = vs["deployed"][1][:v].astype(bool)
    added = dep & ~hy
    sk = EL.crack_skeleton(dep) if dep.any() else dep
    w = EL.pixel_length_weights(sk) if sk.any() else np.zeros(sk.shape)
    centre = ctx["ref_centre"]
    d_ref = ndi.distance_transform_edt(~centre) if centre.any() else np.full(sk.shape, np.inf)
    near = d_ref <= TAU
    on_add = sk & added
    r = {"ref_has_crack": bool(centre.any()), "added_px": int(added.sum()),
         "cl_len_added_px": float(w[on_add].sum()), "cl_len_rest_px": float(w[sk & ~added].sum()),
         "cl_len_added_near_px": float(w[on_add & near].sum()),
         "cl_len_rest_near_px": float(w[sk & ~added & near].sum())}
    roi = ctx["roi_gray"][:v]
    with contextlib.redirect_stdout(io.StringIO()):
        edge_end, _ = EL.INF.compute_zones_dynamic(roi, bottom_pct=0.02)
    for tag, m in (("hy", hy), ("dep", dep)):
        ra = EL.analyze_cracks_rev(m, um, edge_end)
        r[f"{tag}_n_cracks"], r[f"{tag}_len_um"] = ra["n"], ra["total_um"]
        r[f"{tag}_cv"] = ra["regular_cv"] if ra["regular_cv"] is not None else np.nan
        r[f"{tag}_regular"] = ra["regular"]
    return r


def _run_worker(args):
    run_dir, maps_sub, scheme_tag = args
    RC.set_low_priority_off()
    run_dir = Path(run_dir)
    sc = E.scale_table()
    if maps_sub == "probs_test":
        info = json.loads((run_dir / "fold_info.json").read_text(encoding="utf-8"))
        names, key = info["test"], dict(scheme=info["scheme"], fold=info["fold"], seed=info["seed"])
    else:
        names = sorted(p.stem for p in (run_dir / maps_sub).glob("*.npz"))
        key = dict(scheme=scheme_tag, fold=-1, seed=4000)
    recs, brs = [], []
    for n in names:
        glob, tile = E.load_maps(run_dir / maps_sub / f"{n}.npz")
        um = float(sc[n])
        rr = E.eval_maps(n, glob, tile, um, measure_for=("ms", "deployed"))
        ref = E.reference_measurements(n, um)
        for r in rr:
            r.update(ref)
            r.update(um_px=um, **key)
        recs += rr
        if maps_sub == "probs_test":
            b = bridging(n, glob, tile, um)
            b.update(name=n, um_px=um, **key)
            brs.append(b)
    return recs, brs


def _atw_worker(args):
    path, um = args
    RC.set_low_priority_off()
    import fig_hybrid as FH
    glob, tile = E.load_maps(Path(path))
    rec, _ = FH.analysis(Path(path).stem, glob, tile)
    return rec


def cmd_eval(arch: str, processes: int = 4) -> None:
    man = RC.load_manifest()[["name", "specimen", "edge", "position", "has_crater", "has_crack"]]
    jobs = []
    for s in SCHEMES:
        for p in sorted((RUNS / arch).glob(f"{s}_f*")):
            if (p / "done.json").exists():
                jobs.append((str(p), "probs_test", s))
    ins = RUNS / arch / "all" / "probs_insample"
    if ins.exists():
        jobs.append((str(RUNS / arch / "all"), "probs_insample", "insample"))
    print(f"{len(jobs)} map sets to evaluate", flush=True)
    with Pool(min(processes, len(jobs)), initializer=RC.set_low_priority_off) as pool:
        out = pool.map(_run_worker, jobs)
    recs = pd.DataFrame([r for rr, _ in out for r in rr]).merge(man, on="name")
    recs["arch"] = arch
    oof = recs[recs.scheme.isin(SCHEMES)]
    oof.to_csv(OUT / "final_records.csv", index=False)
    recs[recs.scheme == "insample"].to_csv(OUT / "insample_records.csv", index=False)
    br = pd.DataFrame([b for _, bb in out for b in bb])
    br.to_csv(OUT / "bridging_records.csv", index=False)
    print("records:", len(oof), "in sample:", int((recs.scheme == "insample").sum()), "bridging:", len(br))

    atw_dir = RUNS / arch / "all" / "probs_atw"
    if atw_dir.exists():
        sc = E.scale_table()
        aj = [(str(p), float(sc[p.stem])) for p in sorted(atw_dir.glob("*.npz"))]
        with Pool(min(processes, len(aj)), initializer=RC.set_low_priority_off) as pool:
            atw = pool.map(_atw_worker, aj)
        pd.DataFrame(atw).to_csv(OUT / "atw_analysis.csv", index=False)
        print("ATW analysed:", len(atw))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd, arch = sys.argv[1], sys.argv[2]
    {"maps": cmd_maps, "eval": cmd_eval}[cmd](arch)
