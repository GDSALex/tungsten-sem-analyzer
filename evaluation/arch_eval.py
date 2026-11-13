"""Evaluation of the architecture search (arch_runs/<variant>/<scheme>_f<k>).

Decision rule, fixed before any result of the search was looked at
-------------------------------------------------------------------
* Score of a network on a scheme: S = 0.5 x median strict crater IoU + 0.5 x median crack
  centreline F1 at 5 px, over all out-of-fold images of the scheme, with the inference of the
  code at its fixed thresholds (0.5 craters, 0.25 cracks) and the global and tiled passes fused
  (variant 'ms').  The deployed variant (plus ridge filter, hysteresis and gap bridging) is
  reported next to it.
* Reference = the better, on edge5_r1, of the two U-Nets of the authors' code: the recipe of
  the submitted checkpoint (unet_r34) and the default of the final code (unet_r34_ds).
* Another network replaces the reference only if its S is higher in edge5_r1, edge5_r2 and
  LOSO, and the 95 % interval of the paired difference in S, from a bootstrap that resamples
  the 10 edges, excludes zero on the pooled out-of-fold images of the three schemes.
  If several pass, the one with the highest pooled S is adopted; differences below 0.01 go to
  the network with fewer parameters.

    python arch_eval.py            # evaluate new runs, then print and save the summary
    python arch_eval.py summary    # summary only
"""
from __future__ import annotations

import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402
import eval_lib as EL    # noqa: E402
import evaluate as E     # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUNS = RC.REV / "arch_runs"
REC_CSV = RC.RESULTS / "arch_records.csv"
SWEEP_CSV = RC.RESULTS / "arch_sweeps.csv"
SUMMARY_CSV = RC.RESULTS / "arch_summary.csv"
PAIRED_CSV = RC.RESULTS / "arch_paired_bootstrap.csv"
N_FOLDS = {"edge5_r1": 5, "edge5_r2": 5, "loso": 3}
MAIN, DEPLOYED = "ms", "deployed"
# The authors stopped the search on 2026-11-08 at 18:00, once unet_plain, the configuration with
# the highest S on edge5_r1, had been trained on the three schemes (the reviewer asked for a plain
# baseline and an ablation, not for a benchmark).  The other configurations above the reference on
# edge5_r1 were not carried to edge5_r2 and LOSO (segformer_b2 had finished 2 of its 5 edge5_r2
# folds), and the first repetition of unetpp_r34 (3 of 5 folds) and unet_convnext (no fold) was
# not completed.  The rule above is applied unchanged to the configurations that were completed.
STOPPED = {"segformer_b2", "unet_effb4", "dlv3p_r34", "unetpp_r34", "unet_convnext"}


def _worker(run_dir: str):
    RC.set_low_priority_off()
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "fold_info.json").read_text(encoding="utf-8"))
    sc = E.scale_table()
    recs, sweeps = [], []
    key = dict(arch=info["variant"], scheme=info["scheme"], fold=info["fold"], seed=info["seed"])
    for n in info["test"]:
        glob, tile = E.load_maps(run_dir / "probs_test" / f"{n}.npz")
        um = float(sc[n])
        rr = E.eval_maps(n, glob, tile, um, measure_for=())
        if tile is not None:     # the tiled pass alone (native-resolution prediction)
            rr += [dict(r, variant="tile_" + r["variant"]) for r in
                   E.eval_maps(n, tile, None, um, which=("net", "ms_hy", "deployed"),
                               measure_for=())]
        for r in rr:
            r.update(key)
        recs += rr
        # threshold sweeps of the fused maps (calibration-free comparison)
        ctx = E.image_context(n)
        v = ctx["valid_h"]
        g = glob.astype(np.float64)
        if tile is not None:
            t = tile.astype(np.float64)
            crater_p, crack_p = np.maximum(g[0], 0.7 * t[0]), np.maximum(g[1], t[1])
        else:
            crater_p, crack_p = g[0], g[1]
        for thr in E.THR_GRID_C:
            s = EL.crater_scores(ctx["ref_c"], (crater_p > thr)[:v])
            sweeps.append(dict(key, name=n, kind="crater", thr=thr, iou=s["c_iou"], f1=s["c_f1"]))
        for thr in E.THR_GRID_K:
            s = EL.crack_scores(ctx["ref_k"], ctx["ref_centre"], (crack_p > thr)[:v],
                                taus_px=(5,), um_px=None)
            sweeps.append(dict(key, name=n, kind="crack", thr=thr, f1=s["k_cl_F1_5px"],
                               iou=s["k_strict_iou"]))
    return recs, sweeps


def evaluate_new(processes: int = 4) -> None:
    runs = sorted(p for p in RUNS.glob("*/*") if (p / "done.json").exists()
                  and (p / "probs_test").exists())
    have = set()
    old_r = old_s = None
    if REC_CSV.exists():
        old_r = pd.read_csv(REC_CSV)
        old_s = pd.read_csv(SWEEP_CSV)
        have = set(zip(old_r.arch, old_r.scheme, old_r.fold))
    todo = []
    for p in runs:
        info = json.loads((p / "fold_info.json").read_text(encoding="utf-8"))
        if (info["variant"], info["scheme"], info["fold"]) not in have:
            todo.append(str(p))
    print(f"{len(runs)} finished runs, {len(todo)} to evaluate", flush=True)
    if not todo:
        return
    with Pool(min(processes, len(todo)), initializer=RC.set_low_priority_off) as pool:
        out = pool.map(_worker, todo)
    man = RC.load_manifest()[["name", "specimen", "edge", "has_crater", "has_crack"]]
    df = pd.DataFrame([r for rr, _ in out for r in rr]).merge(man, on="name")
    sw = pd.DataFrame([s for _, ss in out for s in ss]).merge(man, on="name")
    if old_r is not None:
        df = pd.concat([old_r, df], ignore_index=True)
        sw = pd.concat([old_s, sw], ignore_index=True)
    df.to_csv(REC_CSV, index=False)
    sw.to_csv(SWEEP_CSV, index=False)
    print("records:", len(df), " sweep rows:", len(sw))


def score(d: pd.DataFrame) -> float:
    return 0.5 * d["c_iou"].median() + 0.5 * d["k_cl_F1_5px"].median()


def complete(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only (network, scheme) pairs whose folds are all finished."""
    nf = df.groupby(["arch", "scheme"]).fold.nunique()
    ok = {k for k, n in nf.items() if n == N_FOLDS[k[1]]}
    return df[[(a, s) in ok for a, s in zip(df.arch, df.scheme)]]


def summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arch, scheme, var), d in df.groupby(["arch", "scheme", "variant"]):
        c, k = d["c_iou"].dropna(), d["k_cl_F1_5px"].dropna()
        rows.append(dict(arch=arch, scheme=scheme, variant=var, n_images=d.name.nunique(),
                         c_iou_median=c.median(), c_iou_q1=c.quantile(0.25),
                         c_iou_q3=c.quantile(0.75), c_iou_mean=c.mean(),
                         c_f1_median=d["c_f1"].median(),
                         k_clF1_5px_median=k.median(), k_clF1_5px_q1=k.quantile(0.25),
                         k_clF1_5px_q3=k.quantile(0.75), k_clF1_5px_mean=k.mean(),
                         k_clF1_2um_median=d["k_cl_F1_2um"].median(),
                         k_strict_iou_median=d["k_strict_iou"].median(),
                         k_dil5_iou_median=d["k_dil5_iou"].median(),
                         k_cldice_median=d["k_cldice"].median(),
                         S=score(d)))
    return pd.DataFrame(rows)


def paired_bootstrap(df: pd.DataFrame, ref: str, variant: str = MAIN, n_boot: int = 4000,
                     seed: int = 20261108) -> pd.DataFrame:
    """Difference in S between each network and the reference, edges resampled with
    replacement (the images of a resampled edge enter with all their records)."""
    rng = np.random.default_rng(seed)
    rows = []
    d0 = df[df.variant == variant]
    schemes = sorted(d0.scheme.unique()) + ["pooled"]
    for arch in sorted(d0.arch.unique()):
        if arch == ref:
            continue
        for scheme in schemes:
            sel = d0 if scheme == "pooled" else d0[d0.scheme == scheme]
            a = sel[sel.arch == arch]
            r = sel[sel.arch == ref]
            common = set(zip(a.scheme, a.name)) & set(zip(r.scheme, r.name))
            if not common:
                continue
            a = a[[k in common for k in zip(a.scheme, a.name)]]
            r = r[[k in common for k in zip(r.scheme, r.name)]]
            edges = sorted(a.edge.unique())
            arr = {e: [x[c].to_numpy(float) for x in (a[a.edge == e], r[r.edge == e])
                       for c in ("c_iou", "k_cl_F1_5px")] for e in edges}

            def s_of(pick):
                ca, ka, cr, kr = (np.concatenate([arr[e][i] for e in pick]) for i in range(4))
                return (0.5 * np.nanmedian(ca) + 0.5 * np.nanmedian(ka)
                        - 0.5 * np.nanmedian(cr) - 0.5 * np.nanmedian(kr))

            diff = score(a) - score(r)
            boots = [s_of(rng.choice(edges, size=len(edges), replace=True))
                     for _ in range(n_boot)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append(dict(arch=arch, ref=ref, scheme=scheme, variant=variant,
                             n_images=len(common), dS=diff, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def params_m(arch: str) -> float:
    done = [json.loads(p.read_text()) for p in (RUNS / arch).glob("*_f*/done.json")]
    return float(np.mean([d["params_M"] for d in done])) if done else float("nan")


def decision(df: pd.DataFrame, sm: pd.DataFrame, pb: pd.DataFrame, ref: str) -> None:
    """Applies the rule of the module docstring, unchanged, to the finished runs."""
    S = sm[sm.variant == MAIN].set_index(["arch", "scheme"])["S"]
    pooled = {a: score(d) for a, d in df[df.variant == MAIN].groupby("arch")}
    print(f"\n== decision rule (reference {ref})")
    passed, waiting, stopped = [], [], []
    for arch in sorted(df.arch.unique()):
        if arch == ref:
            continue
        higher = {s: (arch, s) in S.index and (ref, s) in S.index and S[(arch, s)] > S[(ref, s)]
                  for s in N_FOLDS}
        row = pb[(pb.arch == arch) & (pb.scheme == "pooled")]
        all3 = all((arch, s) in S.index for s in N_FOLDS)
        ci_ok = all3 and len(row) == 1 and float(row.ci_lo.iloc[0]) > 0
        ok = all(higher.values()) and ci_ok
        status = "PASSES" if ok else ("not trained in every scheme" if not all3 else "fails")
        ci = f"[{row.ci_lo.iloc[0]:+.3f}, {row.ci_hi.iloc[0]:+.3f}]" if len(row) else "--"
        print(f"  {arch:16s} higher in " + ", ".join(f"{s}={'yes' if v else 'no'}" for s, v in higher.items())
              + f"; pooled CI {ci}{' (pooled over all its schemes)' if not all3 else ''}: {status}")
        if ok:
            passed.append(arch)
        if higher["edge5_r1"] and not all3:
            (stopped if arch in STOPPED else waiting).append(arch)
    # networks of the search (the VARIANTS of arch_train.py) whose edge5_r1 is unfinished are
    # not in df, since complete() drops them
    search = re.findall(r'^    "([a-z0-9_]+)": dict\(',
                        Path(__file__).with_name("arch_train.py").read_text(encoding="utf-8"), re.M)
    r1_unfinished = [a for a in search
                     if len(list((RUNS / a).glob("edge5_r1_f*/done.json"))) < N_FOLDS["edge5_r1"]]
    r1_open = [a for a in r1_unfinished if a not in STOPPED]
    if stopped:
        print("  search stopped (see STOPPED): above the reference on edge5_r1 but not carried to "
              "the other schemes: " + ", ".join(stopped))
    if [a for a in r1_unfinished if a in STOPPED]:
        print("  search stopped (see STOPPED): edge5_r1 not completed, not reported: "
              + ", ".join(a for a in r1_unfinished if a in STOPPED))
    if waiting or r1_open:
        if waiting:
            print("DECISION PENDING: higher than the reference on edge5_r1 but not yet trained in "
                  "every scheme: " + ", ".join(waiting))
        if r1_open:
            print("DECISION PENDING: edge5_r1 not finished for " + ", ".join(r1_open))
        return
    if not passed:
        print(f"DECISION: no configuration passes; the reference {ref} is kept")
        return
    best = max(passed, key=lambda a: pooled[a])
    close = [a for a in passed if pooled[best] - pooled[a] < 0.01]
    adopted = min(close, key=params_m)
    print("  pooled S of the passing configurations: "
          + ", ".join(f"{a} {pooled[a]:.4f} ({params_m(a):.2f} M)" for a in passed))
    print(f"DECISION: {adopted} replaces the reference {ref}")


def report() -> None:
    df = complete(pd.read_csv(REC_CSV))
    if df.empty:
        print("no complete network yet")
        return
    sm = summary(df)
    sm.to_csv(SUMMARY_CSV, index=False)
    pd.set_option("display.width", 200)
    cols = ["arch", "n_images", "c_iou_median", "k_clF1_5px_median", "k_strict_iou_median", "S"]
    for scheme in sorted(sm.scheme.unique()):
        for var in (MAIN, DEPLOYED, "tile_net"):
            t = sm[(sm.scheme == scheme) & (sm.variant == var)].sort_values("S", ascending=False)
            if len(t):
                print(f"\n== {scheme} / {var}")
                print(t[cols].round(3).to_string(index=False))
    r1 = sm[(sm.scheme == "edge5_r1") & (sm.variant == MAIN)].set_index("arch")["S"]
    cand = [a for a in ("unet_r34", "unet_r34_ds") if a in r1.index]
    if cand:
        ref = max(cand, key=lambda a: r1[a])
        pb = paired_bootstrap(df, ref)
        pb.to_csv(PAIRED_CSV, index=False)
        print(f"\n== paired bootstrap over edges, reference {ref}, variant {MAIN}")
        print(pb.round(3).to_string(index=False))
        decision(df, sm, pb, ref)


if __name__ == "__main__":
    if not (len(sys.argv) > 1 and sys.argv[1] == "summary"):
        evaluate_new()
    report()
