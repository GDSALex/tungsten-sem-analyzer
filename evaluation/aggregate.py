"""Summary tables of the revision, generated only from the per-image record files.

Inputs  results/published_records.csv, classical_records.csv, cv_records.csv, manifest_ALMW.csv
Outputs results/summary_*.csv and results/tex/*.tex (table bodies included by main.tex)

Conventions
* Overlap statistics of a class are computed over the images whose reference contains
  that class (positive images).  An image with an empty reference and a non-empty
  prediction is a false-alarm image, an image with both empty a true-negative image; both
  are counted in the image-level table and do not enter the overlap statistics.
* Uncertainty: percentile intervals of a cluster bootstrap that resamples whole sample
  edges (10 clusters) with replacement, 4000 resamples.  When several out-of-fold
  predictions exist for an image (two repeats), all of them stay with their edge.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402

import numpy as np
import pandas as pd

R = RC.RESULTS
TEX = R / "tex"
TEX.mkdir(exist_ok=True)
B = 4000
RNG = np.random.default_rng(20261107)

K5 = "k_cl_F1_5px"


# ----------------------------------------------------------------------------- loading
def load():
    man = RC.load_manifest()
    pub = pd.read_csv(R / "published_records.csv")
    cla = pd.read_csv(R / "classical_records.csv") if (R / "classical_records.csv").exists() else None
    cv = pd.read_csv(R / "cv_records.csv") if (R / "cv_records.csv").exists() else None
    return man, pub, cla, cv


# ------------------------------------------------------------------------ statistics
def boot_ci(df: pd.DataFrame, col: str, stat, cluster: str = "edge"):
    d = df[[cluster, col]].dropna()
    if d.empty:
        return np.nan, np.nan
    groups = {g: v[col].to_numpy() for g, v in d.groupby(cluster)}
    keys = list(groups)
    vals = []
    for _ in range(B):
        pick = RNG.choice(len(keys), size=len(keys), replace=True)
        x = np.concatenate([groups[keys[i]] for i in pick])
        vals.append(stat(x))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def summ(df: pd.DataFrame, col: str, ci: bool = False):
    x = df[col].dropna()
    out = {"n": int(len(x)), "median": float(x.median()) if len(x) else np.nan,
           "mean": float(x.mean()) if len(x) else np.nan}
    if ci:
        out["med_lo"], out["med_hi"] = boot_ci(df, col, np.median)
        out["mean_lo"], out["mean_hi"] = boot_ci(df, col, np.mean)
    return out


def positives(df, cls):
    return df[df["crater_ref"]] if cls == "crater" else df[df["crack_ref"]]


METRICS = [("crater", "c_iou"), ("crater", "c_f1"), ("crack", "k_strict_iou"),
           ("crack", "k_strict_f1"), ("crack", K5), ("crack", "k_cl_P_5px"), ("crack", "k_cl_R_5px")]


def table_rows(label: str, df: pd.DataFrame, ci: bool) -> dict:
    row = {"row": label, "n_img": int(df["name"].nunique()), "n_rec": len(df)}
    for cls, col in METRICS:
        s = summ(positives(df, cls), col, ci)
        for k, v in s.items():
            row[f"{col}_{k}"] = v
    return row


# ------------------------------------------------------------------ image-level table
def image_level(df: pd.DataFrame, label: str) -> dict:
    out = {"row": label}
    for cls, ref, pred in (("crater", "crater_ref", "crater_pred"), ("crack", "crack_ref", "crack_pred")):
        r, p = df[ref].astype(bool), df[pred].astype(bool)
        out[f"{cls}_pos"] = int(r.sum())
        out[f"{cls}_TP_img"] = int((r & p).sum())
        out[f"{cls}_FN_img"] = int((r & ~p).sum())
        out[f"{cls}_neg"] = int((~r).sum())
        out[f"{cls}_FP_img"] = int((~r & p).sum())
        out[f"{cls}_TN_img"] = int((~r & ~p).sum())
    return out


def analysis_level(df: pd.DataFrame, label: str) -> dict:
    """Detections after the analysis stage (retained crater components, retained cracks)."""
    out = {"row": label}
    d = df.dropna(subset=["m_n_craters"])
    for cls, ref, m in (("crater", "crater_ref", "m_n_craters"), ("crack", "crack_ref", "m_n_cracks")):
        r, p = d[ref].astype(bool), d[m] > 0
        out[f"{cls}_TP_img"] = int((r & p).sum())
        out[f"{cls}_FN_img"] = int((r & ~p).sum())
        out[f"{cls}_FP_img"] = int((~r & p).sum())
        out[f"{cls}_TN_img"] = int((~r & ~p).sum())
    return out


# --------------------------------------------------------------- measurement agreement
def agreement(df: pd.DataFrame, pred_col: str, ref_col: str, label: str) -> dict:
    d = df[[pred_col, ref_col, "edge"]].dropna()
    both = d[(d[pred_col] > 0) & (d[ref_col] > 0)]
    lr = np.log(both[pred_col] / both[ref_col])
    out = {"row": label, "pred": pred_col, "ref": ref_col, "n_ref_pos": int((d[ref_col] > 0).sum()),
           "n_both": int(len(both)), "n_pred_zero_ref_pos": int(((d[pred_col] == 0) & (d[ref_col] > 0)).sum()),
           "n_pred_pos_ref_zero": int(((d[pred_col] > 0) & (d[ref_col] == 0)).sum())}
    if len(both) >= 3:
        out["gmr"] = float(np.exp(lr.mean()))
        out["loa_lo"] = float(np.exp(lr.mean() - 1.96 * lr.std(ddof=1)))
        out["loa_hi"] = float(np.exp(lr.mean() + 1.96 * lr.std(ddof=1)))
        out["median_ratio"] = float(np.exp(lr.median()))
        # empirical 5-95 % range of the per-image ratios (the log ratios are skewed, so the
        # parametric limits above can lie outside every observed ratio)
        out["p05"] = float(np.exp(lr.quantile(0.05)))
        out["p95"] = float(np.exp(lr.quantile(0.95)))
        rr = both[pred_col] / both[ref_col]
        out["f_out2"] = float(((rr < 0.5) | (rr > 2.0)).mean())      # off by a factor of 2 or more
        out["f_in20"] = float(((rr >= 0.8) & (rr <= 1.2)).mean())    # within 20 %
        out["mape"] = float(np.median(np.abs(both[pred_col] / both[ref_col] - 1)))
        out["spearman"] = float(both[[pred_col, ref_col]].corr(method="spearman").iloc[0, 1])
        lo, hi = boot_ci(both.assign(lr=lr), "lr", np.mean)
        out["gmr_lo"], out["gmr_hi"] = float(np.exp(lo)), float(np.exp(hi))
    return out


# ---------------------------------------------------------------- size categories
CAT_EDGES_UM = (1.0, 100.0, 600.0, 950.0)   # inference.py grades 0-4 by equivalent diameter


def size_category(diam_um: pd.Series) -> pd.Series:
    return pd.Series(np.searchsorted(CAT_EDGES_UM, diam_um.to_numpy(), side="right"),
                     index=diam_um.index)


def weighted_kappa(a: np.ndarray, b: np.ndarray, k: int = 5) -> float:
    """Cohen's kappa with linear weights for ordinal categories 0..k-1."""
    a = np.asarray(a, int); b = np.asarray(b, int)
    O = np.zeros((k, k))
    for i, j in zip(a, b):
        O[i, j] += 1
    if O.sum() == 0:
        return float("nan")
    W = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else float("nan")


# ----------------------------------------------------------------------------- main
def fmt(v, nd=3):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def main():
    man, pub, cla, cv = load()
    cols = ["name", "specimen", "edge", "position", "in_GA_set", "orig_split"]
    rows, img_rows, ana_rows, agr = [], [], [], []

    # ---- submitted checkpoint, in-sample / selection images (reconciliation)
    pn = pub[pub.variant == "net"]
    rows.append(table_rows("Submitted model, all 99 (85 train + 14 selection)", pn, False))
    rows.append(table_rows("Submitted model, 85 training images", pn[pn.orig_split == "train"], False))
    rows.append(table_rows("Submitted model, 14 selection images", pn[pn.orig_split == "validation"], False))
    pd_ = pub[pub.variant == "deployed"]
    rows.append(table_rows("Submitted model, deployed pipeline, all 99", pd_, False))
    for lab, d in (("Submitted model, all 99", pn), ("Submitted model, 85 training", pn[pn.orig_split == "train"]),
                   ("Submitted model, 14 selection", pn[pn.orig_split == "validation"])):
        img_rows.append(image_level(d, lab))
    ana_rows.append(analysis_level(pd_, "Submitted model, deployed, all 99"))

    # ---- classical pipeline
    if cla is not None:
        rows.append(table_rows("Classical pipeline, all 99", cla, True))
        rows.append(table_rows("Classical pipeline, 78 images used by the search", cla[cla.in_GA_set], False))
        rows.append(table_rows("Classical pipeline, 21 images not used by the search", cla[~cla.in_GA_set], True))
        img_rows.append(image_level(cla, "Classical, all 99"))
        img_rows.append(image_level(cla[~cla.in_GA_set], "Classical, 21 not in search"))

    # ---- cross-validation
    if cv is not None:
        A = cv[cv.config == "A"]
        e = A[A.scheme.isin(["edge5_r1", "edge5_r2"])]
        for var, lab in (("net", "fixed thresholds"), ("net_sel", "thresholds from inner validation"),
                         ("deployed", "deployed pipeline")):
            d = e[e.variant == var]
            if len(d):
                rows.append(table_rows(f"U-Net A, edge-grouped CV (2 x 5 folds), {lab}", d, True))
                img_rows.append(image_level(d, f"CV edge A {var}"))
            for sch in ("edge5_r1", "edge5_r2"):
                dd = A[(A.scheme == sch) & (A.variant == var)]
                if len(dd):
                    rows.append(table_rows(f"U-Net A, {sch}, {lab}", dd, False))
        lo = A[(A.scheme == "loso")]
        for var in ("net", "net_sel", "deployed"):
            d = lo[lo.variant == var]
            if len(d):
                rows.append(table_rows(f"U-Net A, leave-one-specimen-out, {var}", d, True))
                img_rows.append(image_level(d, f"LOSO A {var}"))
        ana_rows.append(analysis_level(e[e.variant == "deployed"], "CV edge A deployed"))
        ana_rows.append(analysis_level(e[e.variant == "net"], "CV edge A net"))

        # ablation and variants
        abl = []
        for cfg in sorted(cv.config.unique()):
            for sch in sorted(cv.scheme.unique()):
                d = cv[(cv.config == cfg) & (cv.scheme == sch)]
                for var in ("net", "net_sel", "ms", "ms_hy", "deployed"):
                    dv = d[d.variant == var]
                    if dv.empty:
                        continue
                    r = {"config": cfg, "scheme": sch, "variant": var, "folds": int(dv.fold.nunique()),
                         "n_img": int(dv.name.nunique())}
                    for cls, col in METRICS + [("crack", "k_cldice"), ("crack", "k_cl_F1_2px"),
                                               ("crack", "k_cl_F1_10px"), ("crack", "k_dil5_iou"),
                                               ("crack", "k_legacy_score"), ("crack", "k_loc_err_px")]:
                        pos = positives(dv, cls)
                        r[f"{col}_median"] = pos[col].median()
                        r[f"{col}_mean"] = pos[col].mean()
                        fm = pos.groupby("fold")[col].mean()
                        r[f"{col}_foldmean_sd"] = fm.std(ddof=1) if len(fm) > 1 else np.nan
                    il = image_level(dv, "")
                    r.update({k: v for k, v in il.items() if k != "row"})
                    abl.append(r)
        pd.DataFrame(abl).to_csv(R / "summary_ablation.csv", index=False)

        # tolerance sensitivity (config A, both repeats, net and deployed)
        tol = []
        for var in ("net", "deployed"):
            d = positives(e[e.variant == var], "crack")
            r = {"variant": var}
            for col in ["k_strict_iou", "k_strict_f1", "k_dil5_iou", "k_legacy_score", "k_cldice",
                        "k_loc_err_px"] + [f"k_cl_{m}_{t}" for t in ("2px", "3px", "5px", "10px", "20px",
                                                                     "2um", "5um", "10um")
                                           for m in ("P", "R", "F1")]:
                r[f"{col}_median"] = d[col].median()
                r[f"{col}_mean"] = d[col].mean()
            tol.append(r)
        pd.DataFrame(tol).to_csv(R / "summary_tolerance.csv", index=False)

        # measurement agreement, out-of-fold
        for var in ("net", "deployed"):
            d = e[e.variant == var]
            agr.append(agreement(d, "m_crater_total_area_um2", "r_crater_total_area_um2", f"{var}: crater total area vs reference"))
            agr.append(agreement(d, "m_crater_largest_diam_um", "r_crater_largest_diam_um", f"{var}: largest eq. diameter vs reference"))
            agr.append(agreement(d, "m_crack_len_kulpa_um", "r_crack_len_um", f"{var}: crack length (Kulpa) vs reference"))
            agr.append(agreement(d, "m_crack_len_count_um", "r_crack_len_um", f"{var}: crack length (pixel count) vs reference"))
            agr.append(agreement(d, "m_crack_len_freeman_nofilter_um", "r_crack_len_um", f"{var}: crack length (Freeman, no filter) vs reference"))
            agr.append(agreement(d, "m_crater_total_area_um2", "rp_crater_total_area_um2", f"{var}: crater total area vs processed reference"))
            agr.append(agreement(d, "m_crack_len_kulpa_um", "rp_crack_len_kulpa_um", f"{var}: crack length (Kulpa) vs processed reference"))
        for var in ("net", "deployed"):
            d = e[e.variant == var]
            if "m_crack_len_rev_um" in d:
                agr.append(agreement(d, "m_crack_len_rev_um", "r_crack_len_um", f"{var}: crack length (revised estimator) vs reference"))
                agr.append(agreement(d, "m_crack_len_rev_nofilter_um", "r_crack_len_um", f"{var}: crack length (revised, no filter) vs reference"))
        d = e[e.variant == "net"].drop_duplicates("name")
        if "rp_crack_len_rev_um" in d:
            agr.append(agreement(d, "rp_crack_len_rev_um", "r_crack_len_um", "analysis stage on the reference: crack length (revised estimator)"))
            agr.append(agreement(d, "rp_crack_len_rev_nofilter_um", "r_crack_len_um", "analysis stage on the reference: crack length (revised, no filter)"))
            agr.append(agreement(d, "r_crack_len_filt_um", "r_crack_len_um", "reference polylines: length kept by the component filter"))
        agr.append(agreement(d, "rp_crater_total_area_um2", "r_crater_total_area_um2", "analysis stage on the reference: crater total area"))
        agr.append(agreement(d, "rp_crater_largest_diam_um", "r_crater_largest_diam_um", "analysis stage on the reference: largest eq. diameter"))
        agr.append(agreement(d, "rp_crack_len_kulpa_um", "r_crack_len_um", "analysis stage on the reference: crack length (Kulpa)"))
        agr.append(agreement(d, "rp_crack_len_count_um", "r_crack_len_um", "analysis stage on the reference: crack length (count)"))

        # size categories of the largest crater-like region (reviewer comment 7)
        cat_rows = []
        for var in ("net", "deployed"):
            d = e[e.variant == var].dropna(subset=["m_crater_largest_diam_um"])
            pc = size_category(d["m_crater_largest_diam_um"])
            rc = size_category(d["r_crater_largest_diam_um"])
            cm = pd.crosstab(rc, pc).reindex(index=range(5), columns=range(5), fill_value=0)
            agree = float((pc.values == rc.values).mean())
            within1 = float((np.abs(pc.values - rc.values) <= 1).mean())
            cat_rows.append({"variant": var, "n": int(len(d)), "exact_agreement": agree,
                             "within_one": within1, "kappa_w_linear": weighted_kappa(rc.values, pc.values),
                             **{f"ref{i}_pred{j}": int(cm.loc[i, j]) for i in range(5) for j in range(5)}})
            # sensitivity to a +-2 % scale error (area scales with the square, diameter linearly)
            for f in (0.98, 1.02):
                pcs = size_category(d["m_crater_largest_diam_um"] * f)
                cat_rows[-1][f"changed_by_scale_{f}"] = int((pcs.values != pc.values).sum())
        pd.DataFrame(cat_rows).to_csv(R / "summary_categories.csv", index=False)

    pub_d = pub[pub.variant == "deployed"]
    agr.append(agreement(pub_d, "m_crater_total_area_um2", "r_crater_total_area_um2", "submitted model in-sample deployed: crater total area"))
    agr.append(agreement(pub_d, "m_crack_len_kulpa_um", "r_crack_len_um", "submitted model in-sample deployed: crack length (Kulpa)"))

    T = pd.DataFrame(rows)
    T.to_csv(R / "summary_table1.csv", index=False)
    pd.DataFrame(img_rows).to_csv(R / "summary_image_level.csv", index=False)
    pd.DataFrame(ana_rows).to_csv(R / "summary_analysis_level.csv", index=False)
    pd.DataFrame(agr).to_csv(R / "summary_agreement.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        show = [c for c in T.columns if c.endswith("_median") or c in ("row", "n_img")]
        print(T[show].round(3).to_string(index=False))
        print(pd.DataFrame(img_rows).to_string(index=False))
        print(pd.DataFrame(ana_rows).to_string(index=False))
        print(pd.DataFrame(agr).round(3).to_string(index=False))


def extra():
    """Additional summaries of the revision (per-repeat counts, paired differences,
    magnification groups, common region, regularity validation)."""
    man, pub, cla, cv = load()
    A = cv[cv.config == "A"]
    e = A[A.scheme.isin(["edge5_r1", "edge5_r2"])]

    # ---- image-level and analysis-level counts per repetition
    rows = []
    for sch in ("edge5_r1", "edge5_r2", "loso"):
        for var in ("net", "net_sel", "ms", "ms_hy", "deployed"):
            d = A[(A.scheme == sch) & (A.variant == var)]
            if d.empty:
                continue
            r = image_level(d, f"A {sch} {var} mask")
            r.update(scheme=sch, variant=var, level="mask", n=len(d))
            rows.append(r)
            if d["m_n_craters"].notna().any():
                a = analysis_level(d, f"A {sch} {var} analysis")
                # revised analysis stage for cracks
                dd = d.dropna(subset=["m_n_cracks_rev"])
                rr, pp = dd["crack_ref"].astype(bool), dd["m_n_cracks_rev"] > 0
                a.update(crack_rev_TP_img=int((rr & pp).sum()), crack_rev_FN_img=int((rr & ~pp).sum()),
                         crack_rev_FP_img=int((~rr & pp).sum()), crack_rev_TN_img=int((~rr & ~pp).sum()))
                a.update(scheme=sch, variant=var, level="analysis", n=len(d))
                rows.append(a)
    for lab, d in (("pub train", pub[(pub.variant == "net") & (pub.orig_split == "train")]),
                   ("pub selection", pub[(pub.variant == "net") & (pub.orig_split == "validation")]),
                   ("pub all", pub[pub.variant == "net"])):
        r = image_level(d, lab); r.update(scheme="published", variant="net", level="mask", n=len(d))
        rows.append(r)
    for lab, d in (("pub deployed train", pub[(pub.variant == "deployed") & (pub.orig_split == "train")]),
                   ("pub deployed selection", pub[(pub.variant == "deployed") & (pub.orig_split == "validation")]),
                   ("pub deployed all", pub[pub.variant == "deployed"])):
        r = image_level(d, lab + " mask"); r.update(scheme="published", variant="deployed", level="mask", n=len(d))
        rows.append(r)
        a = analysis_level(d, lab + " analysis")
        dd = d.dropna(subset=["m_n_cracks_rev"])
        rr, pp = dd["crack_ref"].astype(bool), dd["m_n_cracks_rev"] > 0
        a.update(crack_rev_TP_img=int((rr & pp).sum()), crack_rev_FN_img=int((rr & ~pp).sum()),
                 crack_rev_FP_img=int((~rr & pp).sum()), crack_rev_TN_img=int((~rr & ~pp).sum()))
        a.update(scheme="published", variant="deployed", level="analysis", n=len(d))
        rows.append(a)
    if cla is not None:
        for lab, d in (("classical all", cla), ("classical 21", cla[~cla.in_GA_set])):
            r = image_level(d, lab); r.update(scheme="classical", variant="classical", level="mask", n=len(d))
            rows.append(r)
    pd.DataFrame(rows).to_csv(R / "summary_image_level_rep.csv", index=False)

    # ---- paired differences between configurations (edge5_r1) and between repetitions
    prs = []
    cols = [("crater", "c_iou"), ("crack", K5), ("crack", "k_cl_P_5px"), ("crack", "k_cl_R_5px")]
    base = cv[(cv.config == "A") & (cv.scheme == "edge5_r1")]
    for cfg in ("B", "C", "D"):
        for var in ("net", "deployed"):
            x = cv[(cv.config == cfg) & (cv.scheme == "edge5_r1") & (cv.variant == var)]
            a = base[base.variant == var]
            if x.empty or a.empty:
                continue
            m = x.merge(a, on="name", suffixes=("_x", "_a"))
            for cls, col in cols:
                ref = "crater_ref_a" if cls == "crater" else "crack_ref_a"
                mm = m[m[ref].astype(bool)].copy()
                mm["diff"] = mm[f"{col}_x"] - mm[f"{col}_a"]
                mm["edge"] = mm["edge_a"]
                lo, hi = boot_ci(mm, "diff", np.mean)
                prs.append({"comparison": f"{cfg}-A", "variant": var, "metric": col, "n": len(mm),
                            "median_x": mm[f"{col}_x"].median(), "median_a": mm[f"{col}_a"].median(),
                            "mean_diff": mm["diff"].mean(), "median_diff": mm["diff"].median(),
                            "ci_lo": lo, "ci_hi": hi})
    for var in ("net", "deployed"):
        r1 = A[(A.scheme == "edge5_r1") & (A.variant == var)]
        r2 = A[(A.scheme == "edge5_r2") & (A.variant == var)]
        m = r2.merge(r1, on="name", suffixes=("_x", "_a"))
        for cls, col in cols:
            ref = "crater_ref_a" if cls == "crater" else "crack_ref_a"
            mm = m[m[ref].astype(bool)].copy()
            mm["diff"] = mm[f"{col}_x"] - mm[f"{col}_a"]
            mm["edge"] = mm["edge_a"]
            lo, hi = boot_ci(mm, "diff", np.mean)
            prs.append({"comparison": "A r2-r1", "variant": var, "metric": col, "n": len(mm),
                        "median_x": mm[f"{col}_x"].median(), "median_a": mm[f"{col}_a"].median(),
                        "mean_diff": mm["diff"].mean(), "median_diff": mm["diff"].median(),
                        "mean_abs_diff": mm["diff"].abs().mean(), "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(prs).to_csv(R / "summary_paired.csv", index=False)

    # ---- magnification groups (detectability)
    mg = []
    e2 = e.copy()
    e2["mag_group"] = np.where(e2["um_px"] < 0.3, "500x", "50-150x")
    for var in ("net", "deployed"):
        for gname, d in e2[e2.variant == var].groupby("mag_group"):
            r = {"variant": var, "group": gname, "n_img": d.name.nunique()}
            for cls, col in [("crater", "c_iou"), ("crack", K5), ("crack", "k_cl_P_5px"), ("crack", "k_cl_R_5px"),
                             ("crack", "k_cl_F1_2um"), ("crack", "k_cl_P_2um"), ("crack", "k_cl_R_2um"),
                             ("crack", "k_cl_F1_5um")]:
                pos = positives(d, cls)
                r[f"{col}_median"] = pos[col].median()
                r[f"{col}_n"] = int(pos[col].notna().sum())
            mg.append(r)
    pd.DataFrame(mg).to_csv(R / "summary_magnification.csv", index=False)

    # ---- common region (rows below the edge row)
    rr = []
    sets = [("A edge5 net", e[e.variant == "net"]), ("A edge5 deployed", e[e.variant == "deployed"])]
    if cla is not None:
        sets += [("classical all", cla), ("classical 21", cla[~cla.in_GA_set])]
    for lab, d in sets:
        cp = d[d["crater_ref"].astype(bool)]
        cpr = d[d["crater_ref_roi"].astype(bool)]
        kp = d[d["crack_ref"].astype(bool)]
        kpr = d[d["crack_ref_roi"].astype(bool)]
        rr.append({"set": lab, "c_iou_med": cp["c_iou"].median(), "c_iou_roi_med": cpr["c_iou_roi"].median(),
                   "k5_med": kp[K5].median(), "k5_roi_med": kpr["k_cl_F1_5px_roi"].median(),
                   "n_c": len(cp), "n_c_roi": len(cpr), "n_k": len(kp), "n_k_roi": len(kpr)})
    one = pub[pub.variant == "net"].drop_duplicates("name")
    tot, above = one["r_k_len_px"].sum(), one["r_k_len_above_edge_px"].sum()
    rr.append({"set": "annotated crack length above edge row", "frac_pooled": above / tot,
               "frac_median_img": float((one.loc[one.r_k_len_px > 0, "r_k_len_above_edge_px"] /
                                         one.loc[one.r_k_len_px > 0, "r_k_len_px"]).median())})
    pd.DataFrame(rr).to_csv(R / "summary_roi.csv", index=False)

    # ---- regularity: deployed (out of fold) against the analysis stage on the reference
    rg = []
    sets = [("A edge5 deployed", e[e.variant == "deployed"]), ("A edge5 net", e[e.variant == "net"])]
    sets += [(f"A {s} deployed", A[(A.scheme == s) & (A.variant == "deployed")]) for s in ("edge5_r1", "edge5_r2")]
    for lab, d in sets:
        if d.empty:
            continue
        ref_ev = d["rp_regular_cv_rev"].notna()
        pr_ev = d["m_regular_cv_rev"].notna()
        both = d[ref_ev & pr_ev]
        rg.append({"set": lab, "n_pred": len(d), "n_img": d.name.nunique(),
                   "ref_eval": int(d.drop_duplicates("name")["rp_regular_cv_rev"].notna().sum()),
                   "ref_regular": int((d.drop_duplicates("name")["rp_regular_cv_rev"] < 0.45).sum()),
                   "pred_eval": int(pr_ev.sum()), "pred_regular": int((d["m_regular_cv_rev"] < 0.45).sum()),
                   "both_eval": len(both),
                   "agree": int(((both["m_regular_cv_rev"] < 0.45) == (both["rp_regular_cv_rev"] < 0.45)).sum()),
                   "cv_mad": float((both["m_regular_cv_rev"] - both["rp_regular_cv_rev"]).abs().median())
                   if len(both) else np.nan,
                   "eval_pred_not_ref": int((pr_ev & ~ref_ev).sum()), "eval_ref_not_pred": int((ref_ev & ~pr_ev).sum()),
                   "cv_min_regular": float(d.loc[d["m_regular_cv_rev"] < 0.45, "m_regular_cv_rev"].min())
                   if (d["m_regular_cv_rev"] < 0.45).any() else np.nan,
                   "cv_max": float(d["m_regular_cv_rev"].max())})
    pd.DataFrame(rg).to_csv(R / "summary_regularity.csv", index=False)
    print("extra summaries written")


if __name__ == "__main__":
    main()
    extra()
