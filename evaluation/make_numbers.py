"""Every number and table body of the revised manuscript, from the result files only.

Outputs
  results/numbers.json          key -> formatted string, substituted for \\V{key} by build.py
  results/tex/<table>.tex       table bodies, substituted for %%TAB:<table>%% by build.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402

import numpy as np
import pandas as pd

R = RC.RESULTS
TEX = R / "tex"
TEX.mkdir(exist_ok=True)
N: dict[str, str] = {}
K5 = "k_cl_F1_5px"


def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def f2(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def sgn3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"$+${x:.3f}" if x >= 0 else f"$-${abs(x):.3f}")


def sgn2(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"$+${x:.2f}" if x >= 0 else f"$-${abs(x):.2f}")


def ci2(lo, hi):
    if lo is None or hi is None or np.isnan(lo) or np.isnan(hi):
        return "--"
    a = f"{lo:.2f}" if lo >= 0 else f"$-${abs(lo):.2f}"
    b = f"{hi:.2f}" if hi >= 0 else f"$-${abs(hi):.2f}"
    return f"{a} to {b}"


def of(k, n):
    return f"{int(k)} of {int(n)}"


def put(key, val):
    if key in N:
        raise KeyError(f"duplicate key {key}")
    N[key] = str(val)


# --------------------------------------------------------------------------- inputs
man = RC.load_manifest()
pub = pd.read_csv(R / "published_records.csv")
cla = pd.read_csv(R / "classical_records.csv")
cv = pd.read_csv(R / "cv_records.csv")
T1 = pd.read_csv(R / "summary_table1.csv").set_index("row")
IL = pd.read_csv(R / "summary_image_level_rep.csv").set_index("row")
TOL = pd.read_csv(R / "summary_tolerance.csv").set_index("variant")
AGR = pd.read_csv(R / "summary_agreement.csv").set_index("row")
CAT = pd.read_csv(R / "summary_categories.csv").set_index("variant")
PAIR = pd.read_csv(R / "summary_paired.csv")
MAG = pd.read_csv(R / "summary_magnification.csv")
ROI = pd.read_csv(R / "summary_roi.csv").set_index("set")
REG = pd.read_csv(R / "summary_regularity.csv").set_index("set")
A = cv[cv.config == "A"]
E = A[A.scheme.isin(["edge5_r1", "edge5_r2"])]

# --------------------------------------------------------------- Table 1 and its keys
ROWS = {
    "pubTr": "Submitted model, 85 training images",
    "pubSel": "Submitted model, 14 selection images",
    "pubAll": "Submitted model, all 99 (85 train + 14 selection)",
    "cvA_net": "U-Net A, edge-grouped CV (2 x 5 folds), fixed thresholds",
    "cvA_sel": "U-Net A, edge-grouped CV (2 x 5 folds), thresholds from inner validation",
    "cvA_dep": "U-Net A, edge-grouped CV (2 x 5 folds), deployed pipeline",
    "losoA_net": "U-Net A, leave-one-specimen-out, net",
    "losoA_dep": "U-Net A, leave-one-specimen-out, deployed",
    "cla78": "Classical pipeline, 78 images used by the search",
    "cla21": "Classical pipeline, 21 images not used by the search",
    "claAll": "Classical pipeline, all 99",
}
MET = {"c_iou": "c_iou", "c_f1": "c_f1", "k_iou": "k_strict_iou", "k_f1": "k_strict_f1",
       "k_cl5": K5, "k_clP5": "k_cl_P_5px", "k_clR5": "k_cl_R_5px"}
for rid, lab in ROWS.items():
    r = T1.loc[lab]
    for mk, col in MET.items():
        put(f"{rid}_{mk}_med", f3(r[f"{col}_median"]))
        put(f"{rid}_{mk}_mean", f3(r[f"{col}_mean"]))
        if f"{col}_med_lo" in r and not np.isnan(r[f"{col}_med_lo"]):
            put(f"{rid}_{mk}_med_ci", ci2(r[f"{col}_med_lo"], r[f"{col}_med_hi"]))
    put(f"{rid}_n_c", int(r["c_iou_n"]))
    put(f"{rid}_n_k", int(r["k_strict_iou_n"]))

TAB1 = [("In sample, submitted model, network output (reference only)", None),
        ("85 training images", "pubTr"), ("14 selection images", "pubSel"), ("All 99 images", "pubAll"),
        ("Out of fold, edge-grouped, 2 $\\times$ 5 folds", None),
        ("Network output, fixed thresholds", "cvA_net"),
        ("Network output, thresholds from inner validation", "cvA_sel"),
        ("Deployed pipeline", "cvA_dep"),
        ("Out of fold, leave one specimen out", None),
        ("Network output, fixed thresholds", "losoA_net"), ("Deployed pipeline", "losoA_dep"),
        ("Classical pipeline", None),
        ("78 images of the search", "cla78"), ("21 images outside the search", "cla21")]


def tab1_body():
    out = []
    for stat, title in (("med", "Median over images"), ("mean", "Mean over images")):
        out.append(f"\\multicolumn{{7}}{{l}}{{\\textbf{{{title}}}}} \\\\")
        for lab, rid in TAB1:
            if rid is None:
                out.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{lab}}}}} \\\\")
                continue
            r = T1.loc[ROWS[rid]]
            n = f"{int(r['c_iou_n']) if not rid.startswith(('cvA',)) else int(r['c_iou_n']) // 2}/" \
                f"{int(r['k_strict_iou_n']) if not rid.startswith(('cvA',)) else int(r['k_strict_iou_n']) // 2}"
            cells = [f3(r[f"{MET[m]}_{'median' if stat == 'med' else 'mean'}"])
                     for m in ("c_iou", "c_f1", "k_iou", "k_f1", "k_cl5")]
            out.append(f"\\quad {lab} & {n} & " + " & ".join(cells) + " \\\\")
            lo_key = f"{MET['c_iou']}_{'med' if stat == 'med' else 'mean'}_lo"
            if lo_key in r and not np.isnan(r[lo_key]):
                cis = []
                for m in ("c_iou", "c_f1", "k_iou", "k_f1", "k_cl5"):
                    lo, hi = r[f"{MET[m]}_{stat}_lo"], r[f"{MET[m]}_{stat}_hi"]
                    cis.append(f"{{\\scriptsize [{lo:.2f}, {hi:.2f}]}}")
                out.append(" & & " + " & ".join(cis) + " \\\\")
        if stat == "med":
            out.append("\\midrule")
    return "\n".join(out)


(TEX / "tab_final_metrics.tex").write_text(tab1_body() + "\n", encoding="utf-8")

# ------------------------------------------------------------ image-level counts
def il(row):
    return IL.loc[row]


for rep in ("r1", "r2"):
    sch = f"edge5_{rep}"
    for var, vk in (("net", "net"), ("net_sel", "sel"), ("deployed", "dep")):
        r = il(f"A {sch} {var} mask")
        for cls, ck in (("crater", "c"), ("crack", "k")):
            put(f"img_cvA_{vk}_{ck}_TP_{rep}", int(r[f"{cls}_TP_img"]))
            put(f"img_cvA_{vk}_{ck}_FP_{rep}", int(r[f"{cls}_FP_img"]))
    for var, vk in (("net", "net"), ("deployed", "dep")):
        a = il(f"A {sch} {var} analysis")
        put(f"ana_cvA_{vk}_k_FP_{rep}", int(a["crack_rev_FP_img"]))
        put(f"ana_cvA_{vk}_k_TP_{rep}", int(a["crack_rev_TP_img"]))
        put(f"ana_cvA_{vk}_c_FP_{rep}", int(a["crater_FP_img"]))
        put(f"ana_cvA_{vk}_c_TP_{rep}", int(a["crater_TP_img"]))
r = il("A edge5_r1 net mask")
put("img_cvA_c_pos", int(r["crater_pos"])); put("img_cvA_c_neg", int(r["crater_neg"]))
put("img_cvA_k_pos", int(r["crack_pos"])); put("img_cvA_k_neg", int(r["crack_neg"]))
r = il("classical all")
put("img_claAll_k_FP", int(r["crack_FP_img"])); put("img_claAll_k_neg", int(r["crack_neg"]))


def cnt(row, cls, kind):
    r = il(row)
    if kind == "found":
        return int(r[f"{cls}_TP_img"]), int(r[f"{cls}_pos"]) if f"{cls}_pos" in r and not pd.isna(r.get(f"{cls}_pos")) else None
    return int(r[f"{cls}_FP_img"]), int(r[f"{cls}_neg"]) if f"{cls}_neg" in r and not pd.isna(r.get(f"{cls}_neg")) else None


def il_cells(rows, analysis=False):
    cells = []
    for cls in ("crater", "crack"):
        for kind in ("found", "fa"):
            vals, den = [], None
            for row in rows:
                r = il(row)
                if analysis:
                    key = {"found": "TP", "fa": "FP"}[kind]
                    v = int(r[f"crack_rev_{key}_img"]) if cls == "crack" else int(r[f"{cls}_{key}_img"])
                    base = il(row.replace("analysis", "mask")) if "analysis" in row else r
                    d = int(base[f"{cls}_pos"]) if kind == "found" else int(base[f"{cls}_neg"])
                else:
                    v = int(r[f"{cls}_TP_img"]) if kind == "found" else int(r[f"{cls}_FP_img"])
                    d = int(r[f"{cls}_pos"]) if kind == "found" else int(r[f"{cls}_neg"])
                vals.append(v); den = d
            cells.append(", ".join(str(v) for v in vals) + f"/{den}")
    return cells


def tab_il():
    L = []
    spec = [("\\textit{In sample, submitted model}", None, False),
            ("\\quad 85 training images, network output", ["pub train"], False),
            ("\\quad 14 selection images, network output", ["pub selection"], False),
            ("\\quad 99 images, deployed, analysis level", ["pub deployed all analysis"], True),
            ("\\textit{Out of fold, edge-grouped, repetitions 1, 2}", None, False),
            ("\\quad Network output, fixed thresholds", ["A edge5_r1 net mask", "A edge5_r2 net mask"], False),
            ("\\quad Thresholds from inner validation", ["A edge5_r1 net_sel mask", "A edge5_r2 net_sel mask"], False),
            ("\\quad Network output, analysis level", ["A edge5_r1 net analysis", "A edge5_r2 net analysis"], True),
            ("\\quad Deployed, mask level", ["A edge5_r1 deployed mask", "A edge5_r2 deployed mask"], False),
            ("\\quad Deployed, analysis level", ["A edge5_r1 deployed analysis", "A edge5_r2 deployed analysis"], True),
            ("\\textit{Out of fold, leave one specimen out}", None, False),
            ("\\quad Network output, fixed thresholds", ["A loso net mask"], False),
            ("\\quad Deployed, analysis level", ["A loso deployed analysis"], True),
            ("\\textit{Classical pipeline}", None, False),
            ("\\quad All 99 images", ["classical all"], False),
            ("\\quad 21 images outside the search", ["classical 21"], False)]
    for lab, rows, ana in spec:
        if rows is None:
            L.append(f"\\multicolumn{{5}}{{l}}{{{lab}}} \\\\")
            continue
        L.append(lab + " & " + " & ".join(il_cells(rows, ana)) + " \\\\")
    return "\n".join(L)


(TEX / "tab_image_level.tex").write_text(tab_il() + "\n", encoding="utf-8")

# ----------------------------------------------------------------- tolerance table
def tol_row(lab, kind, key=None):
    cells = []
    for var in ("net", "deployed"):
        t = TOL.loc[var]
        if kind == "cl":
            cells += [f"{f2(t[f'k_cl_P_{key}_median'])} / {f2(t[f'k_cl_R_{key}_median'])}",
                      f3(t[f"k_cl_F1_{key}_median"])]
        elif kind == "one":
            cells += ["--", f3(t[f"{key}_median"])]
        elif kind == "iouf1":
            cells += ["--", f"{f3(t['k_strict_iou_median'])} / {f3(t['k_strict_f1_median'])}"]
        elif kind == "px":
            cells += ["--", f"{t[f'{key}_median']:.2f} px"]
    return lab + " & " + " & ".join(cells) + " \\\\"


TT = ["\\multicolumn{5}{l}{\\textit{Centreline, tolerance $\\tau$ in pixels (1024 px frame)}} \\\\"]
for t in ("2px", "3px", "5px", "10px", "20px"):
    TT.append(tol_row(f"\\quad $\\tau$ = {t[:-2]} px", "cl", t))
TT.append("\\multicolumn{5}{l}{\\textit{Centreline, tolerance $\\tau$ in micrometres}} \\\\")
for t in ("2um", "5um", "10um"):
    TT.append(tol_row(f"\\quad $\\tau$ = {t[:-2]}\\,$\\mu$m", "cl", t))
TT.append("\\multicolumn{5}{l}{\\textit{Band and topology scores}} \\\\")
TT.append(tol_row("\\quad Band IoU / F1, strict", "iouf1"))
TT.append(tol_row("\\quad Band IoU, both dilated by 5 px", "one", "k_dil5_iou"))
TT.append(tol_row("\\quad clDice", "one", "k_cldice"))
TT.append(tol_row("\\quad Legacy score (not an IoU)", "one", "k_legacy_score"))
TT.append(tol_row("\\quad Localisation error", "px", "k_loc_err_px"))
(TEX / "tab_tolerance.tex").write_text("\n".join(TT) + "\n", encoding="utf-8")
for var, vk in (("net", "net"), ("deployed", "dep")):
    t = TOL.loc[var]
    for tt in ("2px", "3px", "5px", "10px", "20px", "2um", "5um", "10um"):
        put(f"tol_{vk}_F1_{tt}", f3(t[f"k_cl_F1_{tt}_median"]))
        put(f"tol_{vk}_P_{tt}", f3(t[f"k_cl_P_{tt}_median"]))
        put(f"tol_{vk}_R_{tt}", f3(t[f"k_cl_R_{tt}_median"]))
    put(f"tol_{vk}_k_strict_iou", f3(t["k_strict_iou_median"]))
    put(f"tol_{vk}_k_dil5", f3(t["k_dil5_iou_median"]))
    put(f"tol_{vk}_k_legacy", f3(t["k_legacy_score_median"]))
    put(f"tol_{vk}_cldice", f3(t["k_cldice_median"]))
    put(f"tol_{vk}_locerr", f"{t['k_loc_err_px_median']:.1f}")

# ----------------------------------------------------------------- ablation table
def pos(d, cls):
    return d[d["crater_ref"].astype(bool)] if cls == "crater" else d[d["crack_ref"].astype(bool)]


def abl_stats(d):
    c, k = pos(d, "crater"), pos(d, "crack")
    fm_c = c.groupby(["scheme", "fold"])["c_iou"].mean()
    fm_k = k.groupby(["scheme", "fold"])[K5].mean()
    neg = d[~d["crack_ref"].astype(bool)]
    fa = neg.groupby("scheme")["crack_pred"].apply(lambda s: int(s.astype(bool).sum()))
    nneg = neg.groupby("scheme").size()
    return {"c_iou": c["c_iou"].median(), "c_sd": fm_c.std(ddof=1),
            "P": k["k_cl_P_5px"].median(), "R": k["k_cl_R_5px"].median(), "F1": k[K5].median(),
            "k_sd": fm_k.std(ddof=1), "fa": ", ".join(str(fa[s]) for s in sorted(fa.index)) + f"/{int(nneg.iloc[0])}"}


def abl_row(lab, d):
    s = abl_stats(d)
    return (f"{lab} & {f3(s['c_iou'])} & {f3(s['c_sd'])} & {f3(s['P'])} & {f3(s['R'])} & {f3(s['F1'])} & "
            f"{f3(s['k_sd'])} & {s['fa']} \\\\"), s


AB = []
names = {"A": "A, submitted recipe", "B": "B, plain U-Net, BCE + Dice", "C": "C, A without scSE",
         "D": "D, A + deep supervision + crack terms"}
for var, title in (("net", "Network output, fixed thresholds, repetition 1"),
                   ("deployed", "Deployed pipeline, repetition 1")):
    AB.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{title}}}}} \\\\")
    for cfg in "ABCD":
        d = cv[(cv.config == cfg) & (cv.scheme == "edge5_r1") & (cv.variant == var)]
        row, s = abl_row(f"\\quad {names[cfg]}", d)
        AB.append(row)
        put(f"abl_{cfg}_{'net' if var == 'net' else 'dep'}_c_iou", f3(s["c_iou"]))
        put(f"abl_{cfg}_{'net' if var == 'net' else 'dep'}_k_cl5", f3(s["F1"]))
        put(f"abl_{cfg}_{'net' if var == 'net' else 'dep'}_k_P", f3(s["P"]))
        put(f"abl_{cfg}_{'net' if var == 'net' else 'dep'}_k_R", f3(s["R"]))
        put(f"abl_{cfg}_{'net' if var == 'net' else 'dep'}_c_sd", f3(s["c_sd"]))
        put(f"abl_{cfg}_{'net' if var == 'net' else 'dep'}_k_sd", f3(s["k_sd"]))
AB.append("\\multicolumn{8}{l}{\\textit{Configuration A, repetition 2}} \\\\")
for var, lab in (("net", "Network output, fixed thresholds"), ("deployed", "Deployed pipeline")):
    d = A[(A.scheme == "edge5_r2") & (A.variant == var)]
    row, s = abl_row(f"\\quad {lab}", d)
    AB.append(row)
    put(f"abl_A2_{'net' if var == 'net' else 'dep'}_c_iou", f3(s["c_iou"]))
    put(f"abl_A2_{'net' if var == 'net' else 'dep'}_k_cl5", f3(s["F1"]))
AB.append("\\multicolumn{8}{l}{\\textit{Configuration A, inference steps, both repetitions}} \\\\")
for var, lab in (("net", "Global pass, fixed thresholds"), ("ms", "+ tiled pass and fusion"),
                 ("ms_hy", "+ ridge enhancement and hysteresis"), ("deployed", "+ gap bridging (deployed)")):
    d = E[E.variant == var]
    row, s = abl_row(f"\\quad {lab}", d)
    AB.append(row)
    for k in ("c_iou", "P", "R", "F1"):
        put(f"step_{var}_{k}", f3(s[k]))
    put(f"step_{var}_fa", s["fa"].split("/")[0])
(TEX / "tab_ablation.tex").write_text("\n".join(AB) + "\n", encoding="utf-8")

for _, p in PAIR.iterrows():
    cmp = p["comparison"].replace("-", "m").replace(" ", "_")
    mk = {"c_iou": "c_iou", K5: "k_cl5", "k_cl_P_5px": "k_P", "k_cl_R_5px": "k_R"}[p["metric"]]
    key = f"pd_{cmp}_{p['variant'][:3]}_{mk}"
    put(key, sgn3(p["mean_diff"]))
    put(key + "_ci", ci2(p["ci_lo"], p["ci_hi"]))
    if "mean_abs_diff" in p and not pd.isna(p.get("mean_abs_diff")):
        put(key + "_mad", f3(p["mean_abs_diff"]))

# ---------------------------------------------------------- agreement table and keys
AGR_ROWS = [
    ("\\textit{Network output, out of fold}", None),
    ("\\quad Crater total area", "net: crater total area vs reference"),
    ("\\quad Largest equivalent diameter", "net: largest eq. diameter vs reference"),
    ("\\quad Crack length, submitted estimator", "net: crack length (pixel count) vs reference"),
    ("\\quad Crack length, revised estimator", "net: crack length (revised estimator) vs reference"),
    ("\\textit{Deployed pipeline, out of fold}", None),
    ("\\quad Crater total area", "deployed: crater total area vs reference"),
    ("\\quad Largest equivalent diameter", "deployed: largest eq. diameter vs reference"),
    ("\\quad Crack length, submitted estimator", "deployed: crack length (pixel count) vs reference"),
    ("\\quad Crack length, revised estimator", "deployed: crack length (revised estimator) vs reference"),
    ("\\textit{Analysis stage applied to the reference annotation}", None),
    ("\\quad Crater total area", "analysis stage on the reference: crater total area"),
    ("\\quad Largest equivalent diameter", "analysis stage on the reference: largest eq. diameter"),
    ("\\quad Crack length, submitted estimator", "analysis stage on the reference: crack length (count)"),
    ("\\quad Crack length, revised estimator", "analysis stage on the reference: crack length (revised estimator)"),
    ("\\quad Polylines kept by the length filter", "reference polylines: length kept by the component filter"),
]


def agr_row(lab, key):
    a = AGR.loc[key]
    return (f"{lab} & {int(a['n_both'])} & {int(a['n_pred_zero_ref_pos'])} & {int(a['n_pred_pos_ref_zero'])} & "
            f"{f2(a['median_ratio'])} & {f2(a['gmr'])} {{\\scriptsize [{a['gmr_lo']:.2f}, {a['gmr_hi']:.2f}]}} & "
            f"{f2(a['p05'])}--{f2(a['p95'])} & {f2(a['mape'])} & {f2(a['spearman'])} \\\\")


AG = []
for lab, key in AGR_ROWS:
    AG.append(f"\\multicolumn{{9}}{{l}}{{{lab}}} \\\\" if key is None else agr_row(lab, key))
(TEX / "tab_agreement.tex").write_text("\n".join(AG) + "\n", encoding="utf-8")
AK = {"area": "crater total area vs reference", "diam": "largest eq. diameter vs reference",
      "lensub": "crack length (pixel count) vs reference", "lenkulpa": "crack length (Kulpa) vs reference",
      "lenrev": "crack length (revised estimator) vs reference",
      "lenrevnf": "crack length (revised, no filter) vs reference"}
for vk, var in (("net", "net"), ("dep", "deployed")):
    for k, lab in AK.items():
        a = AGR.loc[f"{var}: {lab}"]
        put(f"agr_{vk}_{k}_medratio", f2(a["median_ratio"]))
        put(f"agr_{vk}_{k}_gmr", f2(a["gmr"]))
        put(f"agr_{vk}_{k}_gmr_ci", ci2(a["gmr_lo"], a["gmr_hi"]))
        put(f"agr_{vk}_{k}_loa", f"{f2(a['loa_lo'])} to {f2(a['loa_hi'])}")
        put(f"agr_{vk}_{k}_rng", f"{f2(a['p05'])} to {f2(a['p95'])}")
        put(f"agr_{vk}_{k}_p05", f2(a["p05"]))
        put(f"agr_{vk}_{k}_p95", f2(a["p95"]))
        put(f"agr_{vk}_{k}_out2", f"{100 * a['f_out2']:.0f}")
        put(f"agr_{vk}_{k}_in20", f"{100 * a['f_in20']:.0f}")
        put(f"agr_{vk}_{k}_mape", f"{100 * a['mape']:.0f}")
        put(f"agr_{vk}_{k}_rho", f2(a["spearman"]))
        put(f"agr_{vk}_{k}_n", int(a["n_both"]))
        put(f"agr_{vk}_{k}_missed", int(a["n_pred_zero_ref_pos"]))
        put(f"agr_{vk}_{k}_spurious", int(a["n_pred_pos_ref_zero"]))
for k, lab in (("area", "crater total area"), ("diam", "largest eq. diameter"), ("lensub", "crack length (count)"),
               ("lenkulpa", "crack length (Kulpa)"), ("lenrev", "crack length (revised estimator)"),
               ("lenrevnf", "crack length (revised, no filter)")):
    a = AGR.loc[f"analysis stage on the reference: {lab}"]
    put(f"agr_rp_{k}_medratio", f2(a["median_ratio"]))
    put(f"agr_rp_{k}_loa", f"{f2(a['loa_lo'])} to {f2(a['loa_hi'])}")
    put(f"agr_rp_{k}_rng", f"{f2(a['p05'])} to {f2(a['p95'])}")
    put(f"agr_rp_{k}_mape", f"{100 * a['mape']:.0f}")
a = AGR.loc["reference polylines: length kept by the component filter"]
put("agr_filt_medratio", f2(a["median_ratio"]))
put("agr_filt_gmr", f2(a["gmr"]))
put("agr_filt_rng", f"{f2(a['p05'])} to {f2(a['p95'])}")
put("agr_filt_missed", int(a["n_pred_zero_ref_pos"]))
put("agr_filt_npos", int(a["n_ref_pos"]))

# share of annotated polyline length removed by the length filter
one = pub[pub.variant == "net"].drop_duplicates("name")
L_all, L_kept = one["r_crack_len_um"].sum(), one["r_crack_len_filt_um"].sum()
put("filt_len_removed_pct", f"{100 * (1 - L_kept / L_all):.1f}")

# ---------------------------------------------------------------- size categories
for vk, var in (("net", "net"), ("dep", "deployed")):
    c = CAT.loc[var]
    put(f"cat_{vk}_n", int(c["n"]))
    put(f"cat_{vk}_exact", f"{100 * c['exact_agreement']:.0f}")
    put(f"cat_{vk}_within1", f"{100 * c['within_one']:.0f}")
    put(f"cat_{vk}_kappa", f2(c["kappa_w_linear"]))
    put(f"cat_{vk}_chg098", int(c["changed_by_scale_0.98"]))
    put(f"cat_{vk}_chg102", int(c["changed_by_scale_1.02"]))
    put(f"cat_{vk}_zero_ref_pos", int(sum(c[f"ref{i}_pred0"] for i in range(1, 5))))
    put(f"cat_{vk}_pos_ref_zero", int(sum(c[f"ref0_pred{j}"] for j in range(1, 5))))
    put(f"cat_{vk}_under", int(sum(c[f"ref{i}_pred{j}"] for i in range(5) for j in range(5) if j < i)))
    put(f"cat_{vk}_over", int(sum(c[f"ref{i}_pred{j}"] for i in range(5) for j in range(5) if j > i)))
    put(f"cat_{vk}_top_n", int(sum(c[f"ref4_pred{j}"] for j in range(5))))
    put(f"cat_{vk}_top_hit", int(c["ref4_pred4"]))
    put(f"cat_{vk}_two_below", int(sum(c[f"ref{i}_pred{j}"] for i in range(5) for j in range(5) if j <= i - 2)))

# ------------------------------------------------------------- magnification groups
for _, g in MAG.iterrows():
    gk = "hi" if g["group"] == "500x" else "lo"
    vk = "net" if g["variant"] == "net" else "dep"
    put(f"mag_{vk}_{gk}_n", int(g["n_img"]))
    for col, k in (("c_iou", "c_iou"), (K5, "k5"), ("k_cl_P_5px", "kP5"), ("k_cl_R_5px", "kR5"),
                   ("k_cl_F1_2um", "k2um"), ("k_cl_P_2um", "kP2um"), ("k_cl_R_2um", "kR2um"),
                   ("k_cl_F1_5um", "k5um")):
        put(f"mag_{vk}_{gk}_{k}", f3(g[f"{col}_median"]))

# ---------------------------------------------------------------- common region
r = ROI.loc["A edge5 net"]
put("roi_cvA_c_iou_shift", sgn3(r["c_iou_roi_med"] - r["c_iou_med"]))
put("roi_cvA_k_cl5_shift", sgn3(r["k5_roi_med"] - r["k5_med"]))
r = ROI.loc["classical all"]
put("roi_cla_c_iou_shift", sgn3(r["c_iou_roi_med"] - r["c_iou_med"]))
put("roi_cla_k_cl5_shift", sgn3(r["k5_roi_med"] - r["k5_med"]))
r = ROI.loc["annotated crack length above edge row"]
put("roi_k_len_above_pct", f"{100 * r['frac_pooled']:.1f}")
put("roi_k_len_above_med_pct", f"{100 * r['frac_median_img']:.1f}")

# ----------------------------------------------------------------- regularity
rp = REG.loc["A edge5 deployed"]
put("reg_ref_eval", int(rp["ref_eval"])); put("reg_ref_regular", int(rp["ref_regular"]))
put("reg_dep_eval", int(rp["pred_eval"])); put("reg_n_pred", int(rp["n_pred"]))
put("reg_both_eval", int(rp["both_eval"])); put("reg_agree", int(rp["agree"]))
put("reg_cv_mad", f2(rp["cv_mad"]))
r1 = REG.loc["A edge5_r1 deployed"]
put("reg_almw_dep_eval", int(r1["pred_eval"])); put("reg_almw_dep_regular", int(r1["pred_regular"]))
atw = pd.read_csv(R / "atw_records.csv")
ad = atw[atw.variant == "deployed"]
put("reg_atw_eval", int(ad["m_regular_cv_rev"].notna().sum()))
put("reg_atw_regular", int((ad["m_regular_cv_rev"] < 0.45).sum()))
cvs = pd.concat([E.loc[(E.variant == "deployed") & (E.scheme == "edge5_r1"), "m_regular_cv_rev"],
                 ad["m_regular_cv_rev"]]).dropna()
put("reg_cv_min", f2(cvs[cvs < 0.45].min())); put("reg_cv_max", f2(cvs.max()))
put("atw_n", len(ad)); put("atw_n_crater", int((ad.m_n_craters > 0).sum()))
put("atw_n_crack", int((ad.m_n_cracks_rev > 0).sum()))

# ------------------------------------------------------------------- bridging
b = pd.read_csv(R / "summary_bridging.csv").iloc[0]
put("br_frac_added_pct", f"{100 * b['frac_len_added']:.1f}")
put("br_prec_added", f2(b["prec_added_pooled"])); put("br_prec_rest", f2(b["prec_rest_pooled"]))
put("br_added_on_neg_pct", f"{100 * b['frac_added_on_neg_images']:.0f}")
put("br_len_ratio_med", f2(b["len_ratio_dep_hy_median"])); put("br_len_ratio_p90", f2(b["len_ratio_dep_hy_p90"]))
put("br_ncr_fewer", int(b["ncr_fewer"])); put("br_ncr_more", int(b["ncr_more"]))
put("br_reg_changed", int(b["regular_class_changed"])); put("br_n_pred", int(b["n_pred"]))
put("br_cv_eval_hy", int(b["cv_evaluable_hy"])); put("br_cv_eval_dep", int(b["cv_evaluable_dep"]))

# ------------------------------------------------------------- training history
h = pd.DataFrame(json.load(open(RC.CKPT_DIR / "training_history.json"))).set_index("epoch")
for ep in (15, 125):
    put(f"hist_k_iou_ep{ep}", f2(h.loc[ep, "crack_iou"])); put(f"hist_c_iou_ep{ep}", f2(h.loc[ep, "crater_iou"]))
cl = pd.read_csv(R / "common_loss_published.csv").set_index("epoch")
for ep in (40, 125, 155):
    put(f"cl_tr_ep{ep}", f"{cl.loc[ep, 'train_common_noaug']:.3f}")
    put(f"cl_va_ep{ep}", f"{cl.loc[ep, 'val_common']:.3f}")
for ep in (40, 155):
    put(f"cl_gap_ep{ep}", f"{cl.loc[ep, 'gap']:.3f}")
put("cl_logdiff_max", f"{(cl['val_common'] - cl['history_val_loss']).abs().max():.3f}")

# --------------------------------------------------------------- CV run facts
infos, dones = [], []
for p in sorted(RC.CV_RUNS.iterdir()):
    if (p / "done.json").exists():
        infos.append(json.loads((p / "fold_info.json").read_text()))
        dones.append(json.loads((p / "done.json").read_text()))
iv = [len(i["inner_val"]) for i in infos]
be = [d["best_epoch"] for d in dones]
put("cv_inner_min", min(iv)); put("cv_inner_max", max(iv))
put("cv_bestep_min", min(be)); put("cv_bestep_max", max(be)); put("cv_bestep_med", int(np.median(be)))
put("cv_n_runs", len(dones))
put("cv_minutes_med", f"{np.median([d['minutes'] for d in dones]):.0f}")
sel = cv[(cv.config == "A") & (cv.variant == "net_sel")].drop_duplicates(["scheme", "fold"])
put("cv_selthr_k_range", f"{sel.sel_thr_k.min():.2f} to {sel.sel_thr_k.max():.2f}")
put("cv_selthr_c_range", f"{sel.sel_thr_c.min():.2f} to {sel.sel_thr_c.max():.2f}")
leg = pd.read_csv(R / "legacy_table1_summary.csv")
put("leg_k_med_025", f3(leg[(leg.thr_k == 0.25) & (leg.subset == "all")].crack_iou_median.iloc[0]))
put("leg_k_med_030", f3(leg[(leg.thr_k == 0.30) & (leg.subset == "all")].crack_iou_median.iloc[0]))
put("fig4_n_c", int(pos(E[E.variant == "net"], "crater")["c_iou"].notna().sum()))
put("fig4_n_k", int(pos(E[E.variant == "net"], "crack")[K5].notna().sum()))

# ------------------------------------------------- scale reader of the revised code
sc = pd.read_csv(R / "scale_check_revised.csv")
put("scalerev_n", len(sc))
put("scalerev_correct", int((sc.outcome == "correct").sum()))
put("scalerev_wrong", int((sc.outcome == "wrong").sum()))
put("scalerev_withheld", int((sc.outcome == "withheld").sum()))
put("scalerev_maxerr_pct", f"{100 * sc.rel_err.abs().max():.2f}")

# ------------------------------------------------- observations from the annotation
m = man.merge(pd.read_csv(R / "scale_verified.csv")[["name", "um_px"]], on="name", suffixes=("_m", ""))
rt, ph = m[~m.preheated], m[m.preheated]
surf = ["ALMW1-3a", "ALMW1-3b", "ALMW1-3c", "ALMW1-3d", "ALMW1-4a"]
corner_rt = rt[~rt.position.isin(surf)]
for P in (300, 450, 600, 800, 1000, 1200):
    d = rt[rt.log_power_MW_m2 == P]
    put(f"obs_crack{P}", of(d.has_crack.sum(), len(d)))
for P in (450, 600):
    d = corner_rt[corner_rt.log_power_MW_m2 == P]
    put(f"obs_c{P}_k", int(d.has_crater.sum())); put(f"obs_c{P}_n", len(d))
s = m[m.position.isin(surf)]
put("obs_c_surfimp", int(s.has_crater.sum())); put("obs_n_surfimp", len(s))
put("obs_rt_crack", int(rt.has_crack.sum())); put("obs_ph_crack", int(ph.has_crack.sum()))
lo_rt, lo_ph = rt[rt.um_px > 0.3], ph[ph.um_px > 0.3]
put("obs_rt_crack_low", of(lo_rt.has_crack.sum(), len(lo_rt)))
put("obs_ph_crack_low", of(lo_ph.has_crack.sum(), len(lo_ph)))
hi_rt, hi_ph = rt[rt.log_power_MW_m2 >= 1000], ph[ph.log_power_MW_m2 >= 1000]
put("obs_crack_hi_rt", of(hi_rt.has_crack.sum(), len(hi_rt)))
put("obs_crack_hi_ph", of(hi_ph.has_crack.sum(), len(hi_ph)))
put("obs_pos_multi", m[m.log_pulses > 1].position.nunique())

# ---------------------------------------------------------------- length test
lt = pd.read_csv(R / "length_test.csv")
g = lt.groupby(["method", "width", "L"]).rel_err.agg(["mean", "min", "max"]) * 100
put("lt_count_min_1px", f"{abs(g.loc[('count', 1)]['min'].min()):.0f}")
put("lt_kulpa_1px_lo", f"{g.loc[('kulpa', 1)]['min'].min():.1f}".replace("-", "$-$"))
put("lt_kulpa_1px_hi", f"{g.loc[('kulpa', 1)]['max'].max():.1f}")
for L in (50, 400):
    x = g.loc[("revised", 15, L)]
    put(f"lt_rev15_{L}_mean", f"{x['mean']:+.1f}".replace("-", "$-$").replace("+", "$+$"))
    put(f"lt_rev15_{L}_lo", f"{x['min']:.0f}".replace("-", "$-$"))
    put(f"lt_rev15_{L}_hi", f"{x['max']:.0f}")
    y = g.loc[("count", 15, L)]
    put(f"lt_cnt15_{L}_lo", f"{y['min']:.0f}".replace("-", "$-$"))
    put(f"lt_cnt15_{L}_hi", f"{y['max']:.0f}")

# ------------------------------------------------------------------- timing
tp = R / "summary_timing.csv"
if tp.exists():
    t = pd.read_csv(tp).iloc[0]
    put("time_gpu", str(t["gpu"]).replace("NVIDIA ", ""))
    put("time_cpu", str(t["cpu"]).replace("(R)", "").replace("(TM)", "").strip())
    put("time_stages_sum_med", f"{t['t_total_median_s']:.1f}")
    for k in ("t_read", "t_ocr", "t_global", "t_tiled", "t_post", "t_analysis", "t_total"):
        put(f"time_{k[2:]}_med", f"{t[f'{k}_median_s']:.2f}")
    put("time_total_p90", f"{t['t_total_p90_s']:.2f}")
    put("time_batch_total_min", f"{t['batch_almw_total_s'] / 60:.1f}")
    put("time_batch_per_img", f"{t['batch_almw_per_image_s']:.1f}")
    put("time_load", f"{t['t_model_load_s']:.1f}")

# --------------------------------------------------- example measurements (Table 7)
ex = pd.read_csv(R / "example_measurements.csv")
ee = pd.read_csv(R / "example_errors.csv")
EXR = []
FIG_LABEL = {"ALMW1-1a-600MW-2p_06": "fig:almw1_combined_06", "ALMW2-1a-600MW_2p_02": "fig:almw2_combined_1a",
             "ALMW2-1c-450MW_2p_04": "fig:almw2_combined_1c", "ATW2-3a-600MW_09": "fig:atw2_3a",
             "ATW3-2c_01": "fig:atw3_2c", "ATW3-3d_05": "fig:atw3_3d", "ATW3-4d_14": "fig:atw3_4d"}
for _, x in ex.sort_values("figure_file").iterrows():
    fig = int(x["figure_file"][3:5])
    src = "submitted" if x["source"] == "submitted checkpoint" else "out of fold"
    cvv = "--" if pd.isna(x["regular_cv"]) else f"{x['regular_cv']:.2f}"
    nm = x["name"].replace("_", "\\_")
    EXR.append(f"\\ref{{{FIG_LABEL[x['name']]}}} & {nm} & {src} & {int(x['n_craters'])} & "
               f"{x['crater_total_area_um2']:.0f} & {x['crater_largest_area_um2']:.0f} & "
               f"{x['crater_largest_diam_um']:.1f} & {int(x['n_cracks'])} "
               f"({int(x['n_edge_connected'])}/{int(x['n_near_edge_only'])}/{int(x['n_internal'])}) & "
               f"{x['crack_total_len_um']:.0f} & {x['crack_longest_um']:.0f} & {cvv} \\\\")
    k = f"ex{fig:02d}"
    put(f"{k}_ncr", int(x["n_craters"])); put(f"{k}_area", f"{x['crater_total_area_um2']:.0f}")
    put(f"{k}_larea", f"{x['crater_largest_area_um2']:.0f}"); put(f"{k}_diam", f"{x['crater_largest_diam_um']:.0f}")
    put(f"{k}_nk", int(x["n_cracks"])); put(f"{k}_nke", int(x["n_edge_connected"]))
    put(f"{k}_nkn", int(x["n_near_edge_only"])); put(f"{k}_nki", int(x["n_internal"]))
    put(f"{k}_klen", f"{x['crack_total_len_um']:.0f}"); put(f"{k}_klong", f"{x['crack_longest_um']:.0f}")
    put(f"{k}_cv", cvv); put(f"{k}_cat", x["size_category"].replace("µm", "\\,$\\mu$m"))
(TEX / "tab_examples.tex").write_text("\n".join(EXR) + "\n", encoding="utf-8")
for _, x in ee.iterrows():
    fig = x["figure_file"][3:].split("_")[0]          # '05', '06', '07' or 'E'
    k = f"err{fig}"
    put(f"{k}_c_iou", f3(x["c_iou"])); put(f"{k}_c_f1", f3(x["c_f1"]))
    for c, kk in (("k_cl_P_5px", "kP"), ("k_cl_R_5px", "kR"), ("k_cl_F1_5px", "kF1"),
                  ("k_strict_iou", "k_iou"), ("k_strict_f1", "k_f1")):
        put(f"{k}_{kk}", f3(x[c]) if c in x and not pd.isna(x[c]) else "--")
    put(f"{k}_model", x["model"].replace("_", "\\_"))

# ------------------------------------- images with no annotation, and crack false alarms
cvA = cv[cv.config == "A"]
empty = sorted(cvA[(cvA.scheme == "edge5_r1") & (cvA.variant == "net") &
                   (cvA.crater_ref == 0) & (cvA.crack_ref == 0)]["name"])
put("nd_n", len(empty))
put("nd_names", ", ".join(n.replace("_", "\\_") for n in empty))


def _cnt(df, col):
    return int((df[col] > 0).sum())


pe = pub[pub.name.isin(empty)]
pe_net = pe[(pe.variant == "net") & (pe.thr_k == 0.25)] if "thr_k" in pe else pe[pe.variant == "net"]
pe_dep = pe[pe.variant == "deployed"]
pe_dep = pe_dep[pe_dep.thr_k == 0.25] if "thr_k" in pe_dep and (pe_dep.thr_k == 0.25).any() else pe_dep
put("nd_pub_net_c", _cnt(pe_net, "crater_pred")); put("nd_pub_net_k", _cnt(pe_net, "crack_pred"))
put("nd_pub_dep_k_mask", _cnt(pe_dep, "crack_pred")); put("nd_pub_dep_k_ana", _cnt(pe_dep, "m_n_cracks_rev"))
put("nd_pub_dep_c_ana", _cnt(pe_dep, "m_n_craters"))
for rep in ("edge5_r1", "edge5_r2"):
    r = rep[-2:]
    s_net = cvA[(cvA.scheme == rep) & (cvA.variant == "net") & cvA.name.isin(empty)]
    s_dep = cvA[(cvA.scheme == rep) & (cvA.variant == "deployed") & cvA.name.isin(empty)]
    put(f"nd_cv_net_c_{r}", _cnt(s_net, "crater_pred")); put(f"nd_cv_net_k_{r}", _cnt(s_net, "crack_pred"))
    put(f"nd_cv_net_k_ana_{r}", _cnt(s_net, "m_n_cracks_rev"))
    put(f"nd_cv_dep_c_ana_{r}", _cnt(s_dep, "m_n_craters"))
    put(f"nd_cv_dep_k_mask_{r}", _cnt(s_dep, "crack_pred")); put(f"nd_cv_dep_k_ana_{r}", _cnt(s_dep, "m_n_cracks_rev"))
s_dep = cvA[cvA.scheme.isin(["edge5_r1", "edge5_r2"]) & (cvA.variant == "deployed") & cvA.name.isin(empty)]
put("nd_cv_dep_k_len_max", f"{s_dep.m_crack_len_rev_um.max():.0f}")
ce = cla[cla.name.isin(empty)]
put("nd_cla_c", _cnt(ce, "crater_pred")); put("nd_cla_k", _cnt(ce, "crack_pred"))
for v, key in (("deployed", "dep"), ("net", "net")):
    d = cvA[cvA.scheme.isin(["edge5_r1", "edge5_r2"]) & (cvA.variant == v)]
    fa = d[(d.crack_ref == 0) & (d.m_n_cracks_rev > 0)]
    put(f"fa_{key}_n", len(fa))
    put(f"fa_{key}_len_med", f"{fa.m_crack_len_rev_um.median():.0f}")
    put(f"fa_{key}_len_max", f"{fa.m_crack_len_rev_um.max():.0f}")
# images where every annotated crack is missed by the network output (out of fold)
d = cvA[cvA.scheme.isin(["edge5_r1", "edge5_r2"]) & (cvA.variant == "net")]
mm = d[(d.crack_ref > 0) & (d.crack_pred == 0)]
put("miss_k_n", len(mm))
put("miss_k_len_med", f"{mm.r_crack_len_um.median():.0f}")
put("miss_k_len_max", f"{mm.r_crack_len_um.max():.0f}")
put("all_k_len_med", f"{d[d.crack_ref > 0].r_crack_len_um.median():.0f}")

# reference values of the ALMW examples (first repetition, same rows as the figures)
for nm, k in (("ALMW1-1a-600MW-2p_06", "08"), ("ALMW2-1a-600MW_2p_02", "09"), ("ALMW2-1c-450MW_2p_04", "10"),
              ("ALMW1-3a-600MW_05", "05"), ("ALMW3-1a-1200MW_01", "06"), ("ALMW3-2d-1200MW-HT_13", "07")):
    x = cvA[(cvA.scheme == "edge5_r1") & (cvA.variant == "deployed") & (cvA.name == nm)].iloc[0]
    put(f"exref{k}_ncr", int(x["r_n_craters"]))
    put(f"exref{k}_area", f"{x['r_crater_total_area_um2']:.0f}")
    put(f"exref{k}_diam", f"{x['r_crater_largest_diam_um']:.0f}")
    put(f"exref{k}_nk", int(x["r_n_cracks"]))
    put(f"exref{k}_klen", f"{x['r_crack_len_um']:.0f}")
    put(f"exdep{k}_area", f"{x['m_crater_total_area_um2']:.0f}")
    put(f"exdep{k}_diam", f"{x['m_crater_largest_diam_um']:.0f}")
    put(f"exdep{k}_klen", f"{x['m_crack_len_rev_um']:.0f}")
    put(f"exdep{k}_nk", int(x["m_n_cracks_rev"]))

# --------------------------------------------- representative failures (fig_failures.py)
fp = R / "failure_cases.csv"
if fp.exists():
    fc = pd.read_csv(fp).set_index("panel")
    for p in fc.index:
        x = fc.loc[p]
        put(f"fc_{p}_name", x["name"].replace("_", "\\_"))
        put(f"fc_{p}_fold", int(x["fold"]))
        put(f"fc_{p}_c_iou", f2(x["c_iou"]))
        put(f"fc_{p}_k_f1", f2(x["k_cl_F1_5px"]))
        put(f"fc_{p}_mlen", f"{x['m_crack_len_rev_um']:.0f}")
        put(f"fc_{p}_rlen", f"{x['r_crack_len_um']:.0f}")
        put(f"fc_{p}_mdiam", f"{x['m_crater_largest_diam_um']:.0f}")
        put(f"fc_{p}_rdiam", f"{x['r_crater_largest_diam_um']:.0f}")

(R / "numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False), encoding="utf-8")
print(len(N), "numbers written")
