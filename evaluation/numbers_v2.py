"""Numbers and table bodies of the condensed manuscript (src_v2), from the per-image records.

Inputs
  results/v2/final_records.csv, insample_records.csv, bridging_records.csv, atw_analysis.csv
                                       (eval_final.py, adopted network)
  results/arch_summary.csv, arch_records.csv, arch_runs/*/*/done.json   (architecture search)
  results/manifest_ALMW.csv, manifest_ATW.csv, scale_verified.csv, folds.json, conditions_by_edge.csv
  results/numbers.json                 model-independent values of the long version (annotation
                                       observations, length test, scale check)
  results/v2/texts.json                hand-written sentences that depend on the results
Outputs
  results/v2/numbers.json, results/v2/tex/*.tex

    python numbers_v2.py <arch>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R = RC.RESULTS
V2 = R / "v2"
TEX = V2 / "tex"
TEX.mkdir(parents=True, exist_ok=True)
RUNS = RC.REV / "arch_runs"
B = 4000
K5 = "k_cl_F1_5px"
EDGE = ("edge5_r1", "edge5_r2")
N = {}

NAMES = {   # configuration labels of Table 2, in the order of Section 3.3
    "unet_r34": "(i) U-Net, ResNet34, scSE",
    "unet_r34_ds": "(ii) (i) + deep supervision",
    "unet_plain": "(iii) Plain U-Net, ResNet34",
    "dlv3p_r34": "(iv) DeepLabV3+, ResNet34",
    "unet_effb4": "(v) U-Net, EfficientNet-B4, scSE",
    "segformer_b2": "(vi) SegFormer, MiT-B2",
    "unet_r34_native": "(vii) (i), crops near native res.",
}
# unetpp_r34 and unet_convnext were planned but the search was stopped before their first
# repetition was complete (STOPPED in arch_eval.py); the supplementary text says so.


# ------------------------------------------------------------------------------ helpers
def put(k, v):
    N[k] = v


CLAIMS: list[str] = []      # statements of the text that the values no longer support


def claim(ok, statement: str):
    if not ok:
        CLAIMS.append(statement)


def f2(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.2f}"


def f3(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.3f}"


def ci2(lo, hi):
    return f"{lo:.2f}--{hi:.2f}"


def pct(x):
    return f"{100 * x:.0f}"


def rng_txt(lo, hi):
    return f"{lo:.2f} to {hi:.2f}" if hi < 10 else f"{lo:.2f} to {hi:.1f}"


def boot_ci(df, col, stat, seed=20261108):
    rng = np.random.default_rng(seed)
    d = df[["edge", col]].dropna()
    if d.empty:
        return np.nan, np.nan
    groups = [g[col].to_numpy() for _, g in d.groupby("edge")]
    vals = [stat(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
            for _ in range(B)]
    return tuple(np.percentile(vals, [2.5, 97.5]))


def pos(df, cls):
    return df[df["crater_ref"].astype(bool)] if cls == "crater" else df[df["crack_ref"].astype(bool)]


def agreement(df, pred, ref):
    d = df[[pred, ref, "edge"]].dropna()
    both = d[(d[pred] > 0) & (d[ref] > 0)]
    out = {"both": len(both), "missed": int(((d[pred] == 0) & (d[ref] > 0)).sum()),
           "spurious": int(((d[pred] > 0) & (d[ref] == 0)).sum())}
    if len(both) >= 3:
        lr = np.log(both[pred] / both[ref])
        rr = both[pred] / both[ref]
        out.update(median=float(np.exp(lr.median())), gmr=float(np.exp(lr.mean())),
                   p05=float(np.exp(lr.quantile(0.05))), p95=float(np.exp(lr.quantile(0.95))),
                   mard=float(np.median(np.abs(rr - 1))),
                   rho=float(both[[pred, ref]].corr(method="spearman").iloc[0, 1]),
                   in20=float(((rr >= 0.8) & (rr <= 1.2)).mean()),
                   out2=float(((rr < 0.5) | (rr > 2.0)).mean()))
        lo, hi = boot_ci(both.assign(lr=lr), "lr", np.mean)
        out["gmr_lo"], out["gmr_hi"] = float(np.exp(lo)), float(np.exp(hi))
    return out


def agr_row(label, a):
    if "median" not in a:
        return f"{label} & {a['both']} & {a['missed']} & {a['spurious']} & -- & -- & -- & -- & -- \\\\"
    return (f"{label} & {a['both']} & {a['missed']} & {a['spurious']} & {a['median']:.2f} & "
            f"{a['gmr']:.2f} [{a['gmr_lo']:.2f}, {a['gmr_hi']:.2f}] & {a['p05']:.2f}--{a['p95']:.2f} & "
            f"{a['mard']:.2f} & {a['rho']:.2f} \\\\")


CAT_EDGES_UM = (1.0, 100.0, 600.0, 950.0)


def size_category(diam):
    return np.searchsorted(CAT_EDGES_UM, np.asarray(diam, float), side="right")


def weighted_kappa(a, b, k=5):
    O = np.zeros((k, k))
    for i, j in zip(a, b):
        O[i, j] += 1
    W = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return float(1 - (W * O).sum() / (W * E).sum())


# ------------------------------------------------------------------------------ sections
def conditions():
    m = pd.read_csv(R / "manifest_ALMW.csv")
    s = pd.read_csv(R / "scale_verified.csv")
    m = m.drop(columns=[c for c in ("mag_x", "width_um") if c in m]).merge(s[["name", "mag_x"]], on="name")
    fj = json.load(open(R / "folds.json"))
    f1 = {e: k + 1 for k, g in enumerate(fj["edge5_r1"]) for e in g}
    f2_ = {e: k + 1 for k, g in enumerate(fj["edge5_r2"]) for e in g}

    def lst(v):
        return ", ".join(str(int(x)) for x in sorted(set(v.dropna())))
    lines = []
    for edge, a in m.groupby("edge"):
        t0 = r"$\approx$490\,$^\circ$C" if a.preheated.any() else "RT"
        lines.append(f"{edge} & {lst(a.log_power_MW_m2)} & {lst(a.log_pulses)} & {t0} & {len(a)} & "
                     f"{int(a.has_crater.sum())} & {int(a.has_crack.sum())} & "
                     f"{a.mag_x.min():g}--{a.mag_x.max():g} & {f1[edge]} / {f2_[edge]} \\\\")
    lines.append(r"\midrule")
    lines.append(f"ALMW total & & & & {len(m)} & {int(m.has_crater.sum())} & {int(m.has_crack.sum())} & & \\\\")
    a = pd.read_csv(R / "manifest_ATW.csv").drop(columns=["mag_x"], errors="ignore").merge(
        s[["name", "mag_x"]], on="name", how="left")
    lines.append(r"\midrule")
    for spec, g in a.groupby(a.edge.str[:4]):
        known = g.power_MW_m2.notna().any()
        P = lst(g.power_MW_m2) if known else r"\tbc{n/a}"
        pul = lst(g.pulses_in_name) if known else r"\tbc{n/a}"
        lines.append(f"{spec} ({g.edge.nunique()} edges) & {P} & {pul} & \\tbc{{RT}} & {len(g)} & -- & -- & "
                     f"{g.mag_x.min():g}--{g.mag_x.max():g} & -- \\\\")
    (TEX / "tab_conditions.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def arch_section(ref_arch: str, adopted: str):
    sm = pd.read_csv(R / "arch_summary.csv")
    sm = sm[sm.variant == "ms"]
    r1 = sm[sm.scheme == "edge5_r1"].set_index("arch")
    rows = []
    for a, lab in NAMES.items():
        if a not in r1.index:
            continue
        done = [json.load(open(p)) for p in (RUNS / a).glob("edge5_r1_f*/done.json")]
        par = np.mean([d["params_M"] for d in done]) if done else np.nan
        # training time is not tabulated: small networks shared the card two at a time
        s_other = []
        for sch in ("edge5_r2", "loso"):
            x = sm[(sm.arch == a) & (sm.scheme == sch)]
            s_other.append(f3(float(x.S.iloc[0])) if len(x) else "--")
        mark = r"$^{\ast}$" if a == adopted else ""
        rows.append(f"{lab}{mark} & {par:.1f} & {f3(r1.loc[a, 'c_iou_median'])} & "
                    f"{f3(r1.loc[a, 'k_clF1_5px_median'])} & {f3(r1.loc[a, 'S'])} & "
                    f"{s_other[0]} & {s_other[1]} \\\\")
    (TEX / "tab_arch.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    put("arch_r1_c_min", f2(r1.c_iou_median.min()))
    put("arch_r1_c_max", f2(r1.c_iou_median.max()))
    put("arch_r1_k_min", f2(r1.k_clF1_5px_median.min()))
    put("arch_r1_k_max", f2(r1.k_clF1_5px_median.max()))
    if "unet_plain" in r1.index:
        put("arch_plain_S_r1", f3(r1.loc["unet_plain", "S"]))
    put("arch_ref_S_r1", f3(r1.loc[ref_arch, "S"]))
    # every scheme of the reference, of (i) and of the plain U-Net, and the paired differences
    S = sm.set_index(["arch", "scheme"])["S"]
    for a, tag in ((ref_arch, "ref"), ("unet_plain", "plain"), ("unet_r34", "i")):
        for sch, st in (("edge5_r1", "r1"), ("edge5_r2", "r2"), ("loso", "loso")):
            if (a, sch) in S.index:
                put(f"arch_{tag}_S_{st}", f3(S[(a, sch)]))
    pb = pd.read_csv(R / "arch_paired_bootstrap.csv")
    for a, tag in (("unet_plain", "plain"), ("unet_r34", "i")):
        x = pb[(pb.arch == a) & (pb.ref == ref_arch)].set_index("scheme")
        full = all(s in x.index for s in ("edge5_r1", "edge5_r2", "loso"))
        for sch, st in (("edge5_r1", "r1"), ("edge5_r2", "r2"), ("loso", "loso"), ("pooled", "pooled")):
            if sch in x.index and (sch != "pooled" or full):
                put(f"arch_{tag}_dS_{st}", f"${x.loc[sch, 'dS']:+.3f}$")
                put(f"arch_{tag}_ci_{st}", f"[${x.loc[sch, 'ci_lo']:+.3f}$, ${x.loc[sch, 'ci_hi']:+.3f}$]")
    if "arch_plain_ci_pooled" in N:
        x = pb[(pb.arch == "unet_plain") & (pb.ref == ref_arch) & (pb.scheme == "pooled")].iloc[0]
        put("arch_plain_pooled_excl0", "yes" if x.ci_lo > 0 or x.ci_hi < 0 else "no")
        claim(x.ci_lo <= 0 <= x.ci_hi, "abstract / 3.3 / 4.1: the pooled interval of (iii) includes zero, "
              "so no configuration passes the rule")
        claim(all(float(N[f"arch_plain_S_{s}"]) > float(N[f"arch_ref_S_{s}"]) for s in ("r1", "r2", "loso")),
              "abstract / 4.1: the plain U-Net scored higher than the reference in every scheme")
    above = [a for a in r1.index if a != ref_arch and r1.loc[a, "S"] > r1.loc[ref_arch, "S"]]
    claim(set(above) == {"unet_plain", "dlv3p_r34", "unet_effb4", "segformer_b2"},
          "4.1: (iii) to (vi) are the configurations above the reference in the first repetition")
    if "unet_plain" in r1.index:
        claim(all(r1.loc[a, "S"] < r1.loc["unet_plain", "S"] for a in ("dlv3p_r34", "unet_effb4", "segformer_b2")
                  if a in r1.index),
              "Discussion: the other architectures scored between the reference and the plain U-Net in r1")
        if "unet_r34_native" in r1.index:
            claim(r1.loc["unet_r34_native", "S"] < r1.loc[ref_arch, "S"],
                  "Discussion: training near the original resolution scored lower than the reference")
    if above:
        gains = [r1.loc[a, "S"] - r1.loc[ref_arch, "S"] for a in above]
        put("arch_above_min", f3(min(gains)))
        put("arch_above_max", f3(max(gains)))
    if "unet_r34" in r1.index:
        claim(r1.loc["unet_r34", "S"] < r1.loc[ref_arch, "S"],
              "4.1: the reference (ii) is marginally above (i) in the first repetition")
    if "unet_r34_native" in r1.index:
        put("arch_native_S_r1", f3(r1.loc["unet_r34_native", "S"]))
    # the configurations of the search that were above the reference on edge5_r1 but not carried
    # further, and their gap to the plain U-Net there
    stopped = [a for a in ("dlv3p_r34", "unet_effb4", "segformer_b2") if a in r1.index]
    if stopped and "unet_plain" in r1.index:
        gaps = [r1.loc["unet_plain", "S"] - r1.loc[a, "S"] for a in stopped]
        put("arch_stop_gap_min", f3(min(gaps)))
        put("arch_stop_gap_max", f3(max(gaps)))
        claim(all(r1.loc[a, "S"] > r1.loc[ref_arch, "S"] for a in stopped),
              "3.3 / S4: (iv) to (vi) were above the reference in the first repetition")
    stopped_search(ref_arch)


def stopped_search(ref_arch: str):
    """Partial first repetition of U-Net++ when the search was stopped (Section S4)."""
    df = pd.read_csv(R / "arch_records.csv")
    d = df[(df.variant == "ms") & (df.scheme == "edge5_r1")]
    pp = d[d.arch == "unetpp_r34"]
    if pp.empty:
        return
    folds = sorted(pp.fold.unique())
    names = set(pp.name)

    def s_of(x):
        return 0.5 * x.c_iou.median() + 0.5 * x.k_cl_F1_5px.median()

    def same(a):
        return d[(d.arch == a) & d.fold.isin(folds) & d.name.isin(names)]
    put("pp_n_folds", {1: "one", 2: "two", 3: "three", 4: "four"}[len(folds)])
    put("pp_n_img", str(len(names)))
    put("pp_S", f3(s_of(pp)))
    put("pp_ref_S", f3(s_of(same(ref_arch))))
    put("pp_plain_S", f3(s_of(same("unet_plain"))))
    put("convnext_trained", "no" if not (RUNS / "unet_convnext").exists() else "partly")


def oof_section(df: pd.DataFrame):
    E = df[df.scheme.isin(EDGE)]
    L = df[df.scheme == "loso"]
    for tag, d0 in (("oof", E), ("loso", L)):
        for var, vt in (("ms", "ms"), ("deployed", "dep")):
            d = d0[d0.variant == var]
            if d.empty:
                continue
            c, k = pos(d, "crater"), pos(d, "crack")
            put(f"{tag}_{vt}_c_iou_med", f2(c.c_iou.median()))
            put(f"{tag}_{vt}_k_cl5_med", f2(k[K5].median()))
            put(f"{tag}_{vt}_k_clP5_med", f2(k.k_cl_P_5px.median()))
            put(f"{tag}_{vt}_k_clR5_med", f2(k.k_cl_R_5px.median()))
            if tag == "oof" and var == "ms":
                put("oof_ms_c_iou_ci", ci2(*boot_ci(c, "c_iou", np.median)))
                put("oof_ms_k_cl5_ci", ci2(*boot_ci(k, K5, np.median)))
    # Table 3
    lines, ci = [], {}
    for lab, d0 in (("edge-grouped", E), ("by specimen", L)):
        for var, vl in (("ms", "Network output"), ("deployed", "Deployed pipeline")):
            d = d0[d0.variant == var]
            if d.empty:
                continue
            c, k = pos(d, "crater"), pos(d, "crack")
            cl, ch = boot_ci(c, "c_iou", np.median)
            kl, kh = boot_ci(k, K5, np.median)
            ci[(lab, var)] = (round(cl, 2), round(ch, 2), round(kl, 2), round(kh, 2))
            kneg = d[~d.crack_ref.astype(bool)]
            lines.append(f"{vl}, {lab} & {c.c_iou.median():.2f} [{cl:.2f}, {ch:.2f}] & "
                         f"{k.k_cl_P_5px.median():.2f} & {k.k_cl_R_5px.median():.2f} & "
                         f"{k[K5].median():.2f} [{kl:.2f}, {kh:.2f}] & "
                         f"{int((k.m_n_cracks_rev > 0).sum())}/{len(k)} & "
                         f"{int((kneg.m_n_cracks_rev > 0).sum())}/{len(kneg)} \\\\")
    (TEX / "tab_oof.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    a, b = ci.get(("edge-grouped", "ms")), ci.get(("by specimen", "ms"))
    if a and b:
        claim(a[0] <= b[1] and b[0] <= a[1] and a[2] <= b[3] and b[2] <= a[3],
              "4.2 'the intervals of the two schemes overlap' (network output, craters and cracks)")

    ms, dep = E[E.variant == "ms"], E[E.variant == "deployed"]
    kneg_ms, kneg_dep = ms[~ms.crack_ref.astype(bool)], dep[~dep.crack_ref.astype(bool)]
    cneg_ms = ms[~ms.crater_ref.astype(bool)]
    put("oof_n_kpos", str(int(ms.crack_ref.astype(bool).sum())))
    put("oof_n_kneg", str(len(kneg_ms)))
    put("oof_n_cneg", str(len(cneg_ms)))
    put("oof_ms_fa_k", str(int((kneg_ms.m_n_cracks_rev > 0).sum())))
    put("oof_dep_fa_k", str(int((kneg_dep.m_n_cracks_rev > 0).sum())))
    put("oof_ms_fa_c", str(int((cneg_ms.m_n_craters > 0).sum())))
    put("fig3_n_c", str(len(pos(ms, "crater"))))
    put("fig3_n_k", str(len(pos(ms, "crack"))))
    put("step_ms_F1", f2(pos(ms, "crack")[K5].median()))
    put("step_dep_F1", f2(pos(dep, "crack")[K5].median()))
    # missed cracks
    k = pos(ms, "crack")
    miss = k[k.k_cl_R_5px == 0]
    put("miss_k_n", str(len(miss)))
    put("miss_k_len_med", f"{miss.r_crack_len_um.median():.0f}" if len(miss) else "--")
    put("all_k_len_med", f"{k.drop_duplicates('name').r_crack_len_um.median():.0f}")
    # magnification groups
    g = ms.assign(hi=ms.um_px < 0.3)
    for tag, d in (("lo", g[~g.hi]), ("hi", g[g.hi])):
        put(f"mag_{tag}_n", str(d.name.nunique()))
        put(f"mag_{tag}_c_iou", f2(pos(d, "crater").c_iou.median()))
        put(f"mag_{tag}_k5", f2(pos(d, "crack")[K5].median()))
    # repetition-to-repetition differences of the same image
    a = ms[ms.scheme == "edge5_r1"].set_index("name")
    b = ms[ms.scheme == "edge5_r2"].set_index("name")
    if len(a) and len(b):
        j = a.join(b, lsuffix="_1", rsuffix="_2", how="inner")
        put("rep_mad_c_iou", f2(np.median(np.abs(j.loc[j.crater_ref_1.astype(bool), "c_iou_1"]
                                                - j.loc[j.crater_ref_1.astype(bool), "c_iou_2"]))))
        put("rep_mad_k_cl5", f2(np.median(np.abs(j.loc[j.crack_ref_1.astype(bool), f"{K5}_1"]
                                                - j.loc[j.crack_ref_1.astype(bool), f"{K5}_2"]))))


def measurement_section(df: pd.DataFrame):
    E = df[df.scheme.isin(EDGE)]
    A = {}
    for var, vt in (("ms", "ms"), ("deployed", "dep")):
        d = E[E.variant == var]
        A[vt + "_area"] = agreement(d, "m_crater_total_area_um2", "r_crater_total_area_um2")
        A[vt + "_diam"] = agreement(d, "m_crater_largest_diam_um", "r_crater_largest_diam_um")
        A[vt + "_len"] = agreement(d, "m_crack_len_rev_um", "r_crack_len_um")
    one = E[E.variant == "ms"].drop_duplicates("name")
    A["rp_area"] = agreement(one, "rp_crater_total_area_um2", "r_crater_total_area_um2")
    A["rp_len"] = agreement(one, "rp_crack_len_rev_um", "r_crack_len_um")
    # the crater mask is the same for both outputs (the deployed steps act on cracks only)
    assert A["ms_area"].get("median") == A["dep_area"].get("median")
    rows = [r"\multicolumn{9}{l}{\textit{Both outputs (same crater mask)}} \\",
            agr_row(r"\quad Crater total area", A["dep_area"]),
            agr_row(r"\quad Largest equivalent diameter", A["dep_diam"]),
            r"\multicolumn{9}{l}{\textit{Total crack length}} \\",
            agr_row(r"\quad Network output", A["ms_len"]),
            agr_row(r"\quad Deployed pipeline", A["dep_len"]),
            r"\multicolumn{9}{l}{\textit{Analysis stage applied to the reference masks}} \\",
            agr_row(r"\quad Crater total area", A["rp_area"]),
            agr_row(r"\quad Total crack length", A["rp_len"])]
    (TEX / "tab_agreement.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    for vt in ("ms", "dep"):
        put(f"agr_{vt}_area_medratio", f2(A[vt + "_area"]["median"]))
        put(f"agr_{vt}_diam_medratio", f2(A[vt + "_diam"]["median"]))
        put(f"agr_{vt}_lenrev_medratio", f2(A[vt + "_len"]["median"]))
        put(f"agr_{vt}_lenrev_missed", str(A[vt + "_len"]["missed"]))
    a = A["dep_area"]
    put("agr_dep_area_rho", f2(a["rho"]))
    put("agr_dep_diam_rho", f2(A["dep_diam"]["rho"]))
    put("agr_dep_area_rng", rng_txt(a["p05"], a["p95"]))
    put("agr_dep_area_in20", pct(a["in20"]))
    put("agr_dep_area_out2", pct(a["out2"]))
    put("agr_rp_area_medratio", f2(A["rp_area"]["median"]))
    put("agr_rp_area_rng", rng_txt(A["rp_area"]["p05"], A["rp_area"]["p95"]))
    put("agr_rp_lenrev_medratio", f2(A["rp_len"]["median"]))
    put("agr_rp_lenrev_rng", rng_txt(A["rp_len"]["p05"], A["rp_len"]["p95"]))
    # the pixel count of the first submission applied to the reference bands (branches counted)
    A["rp_lensub"] = agreement(one, "rp_crack_len_count_um", "r_crack_len_um")
    put("agr_rp_lensub_medratio", f2(A["rp_lensub"]["median"]))
    dep = E[E.variant == "deployed"]
    fa = dep[(~dep.crack_ref.astype(bool)) & (dep.m_crack_len_rev_um > 0)]
    put("fa_dep_len_med", f"{fa.m_crack_len_rev_um.median():.0f}" if len(fa) else "--")
    put("fa_dep_len_max", f"{fa.m_crack_len_rev_um.max():.0f}" if len(fa) else "--")
    # component filter on the annotated polylines
    put("filt_len_removed_pct", f"{100 * (1 - one.r_crack_len_filt_um.sum() / one.r_crack_len_um.sum()):.1f}")
    kp = one[one.r_crack_len_um > 0]
    put("agr_filt_missed", str(int((kp.r_crack_len_filt_um == 0).sum())))
    put("agr_filt_npos", str(len(kp)))
    # size categories, deployed
    d = dep.dropna(subset=["m_crater_largest_diam_um"])
    pc, rc = size_category(d.m_crater_largest_diam_um), size_category(d.r_crater_largest_diam_um)
    put("cat_dep_n", str(len(d)))
    put("cat_dep_exact", pct((pc == rc).mean()))
    put("cat_dep_within1", pct((np.abs(pc - rc) <= 1).mean()))
    put("cat_dep_kappa", f2(weighted_kappa(rc, pc)))
    put("cat_dep_under", str(int((pc < rc).sum())))
    put("cat_dep_over", str(int((pc > rc).sum())))
    chg = [int((size_category(d.m_crater_largest_diam_um * f) != pc).sum()) for f in (0.98, 1.02)]
    put("cat_dep_chg_max", str(max(chg)))
    top = rc == 4                                   # largest annotated region above 950 um
    put("cat_dep_top_n", str(int(top.sum())))
    put("cat_dep_top_img", str(d[top].name.nunique()))
    put("cat_dep_top_hit", str(int((pc[top] == 4).sum())))
    # regularity
    put("reg_ref_eval", str(int(one.rp_regular_cv_rev.notna().sum())))
    both = dep[dep.rp_regular_cv_rev.notna() & dep.m_regular_cv_rev.notna()]
    put("reg_both_eval", str(len(both)))
    put("reg_agree", str(int(((both.m_regular_cv_rev < 0.45) == (both.rp_regular_cv_rev < 0.45)).sum())))


def bridging_section():
    p = V2 / "bridging_records.csv"
    if not p.exists():
        return
    br = pd.read_csv(p)
    br = br[br.scheme.isin(EDGE)].copy()
    br["ref_has_crack"] = br.ref_has_crack.astype(bool)
    q = br[br.ref_has_crack]
    put("br_prec_added", f2(q.cl_len_added_near_px.sum() / max(q.cl_len_added_px.sum(), 1e-9)))
    put("br_prec_rest", f2(q.cl_len_rest_near_px.sum() / max(q.cl_len_rest_px.sum(), 1e-9)))
    # what the bridging changes, as in bridging_check.py (deployed against ms_hy, per prediction)
    added = br.cl_len_added_px.sum()
    put("br_n_pred", str(len(br)))
    put("br_frac_added_pct", f"{100 * added / max(added + br.cl_len_rest_px.sum(), 1e-9):.1f}")
    put("br_added_on_neg_pct", f"{100 * br.loc[~br.ref_has_crack, 'cl_len_added_px'].sum() / max(added, 1e-9):.0f}")
    both = br[(br.hy_len_um > 0) & (br.dep_len_um > 0)]
    ratio = both.dep_len_um / both.hy_len_um
    put("br_len_ratio_med", f2(ratio.median()))
    put("br_len_ratio_p90", f2(ratio.quantile(0.9)))
    put("br_ncr_fewer", str(int((br.dep_n_cracks < br.hy_n_cracks).sum())))
    put("br_ncr_more", str(int((br.dep_n_cracks > br.hy_n_cracks).sum())))
    ev = br[br.hy_cv.notna() | br.dep_cv.notna()]
    put("br_cv_eval_hy", str(int(br.hy_cv.notna().sum())))
    put("br_cv_eval_dep", str(int(br.dep_cv.notna().sum())))
    # a prediction whose CV becomes or stops being evaluable counts as a change of class
    put("br_reg_changed", str(int((ev.hy_regular.astype(str) != ev.dep_regular.astype(str)).sum())))


def insample_atw():
    p = V2 / "insample_records.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[d.variant == "ms"]
        put("ins_c_iou_med", f2(pos(d, "crater").c_iou.median()))
        put("ins_k_cl5_med", f2(pos(d, "crack")[K5].median()))
    p = V2 / "atw_analysis.csv"
    if p.exists():
        a = pd.read_csv(p)
        put("atw_n_crater", str(int((a.n_craters > 0).sum())))
        put("atw_n_crack", str(int((a.n_cracks > 0).sum())))


def supp_arch_tables(ref_arch: str, adopted: str):
    """Tables S3 and S4: every candidate on edge5_r1 under three inference variants, and the
    paired differences in S with respect to the reference."""
    sm = pd.read_csv(R / "arch_summary.csv")
    r1 = sm[sm.scheme == "edge5_r1"]
    rows = []
    for a, lab in NAMES.items():
        x = r1[r1.arch == a].set_index("variant")
        if "ms" not in x.index:
            continue
        done = [json.load(open(p)) for p in (RUNS / a).glob("edge5_r1_f*/done.json")]
        par = np.mean([d["params_M"] for d in done]) if done else np.nan
        # no training times: the card was shared between two runs, and with other programs
        cells = [f"{par:.1f}"]
        for var in ("ms", "deployed", "tile_net"):
            if var in x.index:
                cells += [f3(x.loc[var, "c_iou_median"]), f3(x.loc[var, "k_clF1_5px_median"]),
                          f3(x.loc[var, "S"])]
            else:
                cells += ["--"] * 3
        mark = r"$^{\ast}$" if a == adopted else (r"$^{\dagger}$" if a == ref_arch else "")
        rows.append(f"{lab}{mark} & " + " & ".join(cells) + r" \\")
    (TEX / "stab_arch_full.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    t = r1.pivot_table(index="arch", columns="variant", values=["k_clF1_5px_median", "S"])
    archs = [a for a in NAMES if a in t.index]
    kf, sf = t["k_clF1_5px_median"].loc[archs], t["S"].loc[archs]
    for var, tag in (("ms", "ms"), ("deployed", "dep"), ("tile_net", "tile")):
        put(f"{tag}_k_min", f3(kf[var].min()))
        put(f"{tag}_k_max", f3(kf[var].max()))
    claim((kf["tile_net"] > kf["ms"]).all(),
          "S4: the tiled pass alone has a higher crack F1 than the fused output for every configuration")
    claim((kf["deployed"] < kf["ms"]).all(), "S4: the deployed pipeline lowers the crack F1 of every configuration")
    if {"unet_r34", "unet_r34_native"} <= set(archs):
        claim(sf.loc["unet_r34_native", "ms"] < sf.loc["unet_r34", "ms"]
              and sf.loc["unet_r34_native", "tile_net"] < sf.loc["unet_r34", "tile_net"],
              "S4: (vii) scores lower than (i) in the fused output and in the tiled pass alone")

    p = R / "arch_paired_bootstrap.csv"
    pb = pd.read_csv(p) if p.exists() else pd.DataFrame()
    rows = []
    for a, lab in NAMES.items():
        if a == ref_arch or pb.empty or a not in set(pb.arch):
            continue
        x = pb[(pb.arch == a) & (pb.ref == ref_arch)].set_index("scheme")
        # the pooled difference of the rule needs the three schemes; arch_eval pools whatever
        # schemes a configuration has, which for one trained only on edge5_r1 repeats that column
        full = all(s in x.index for s in ("edge5_r1", "edge5_r2", "loso"))
        cells = []
        for sch in ("edge5_r1", "edge5_r2", "loso", "pooled"):
            # difference above its interval, so that the configuration names keep their width
            cells.append(r"\begin{tabular}[t]{@{}c@{}}"
                         f"${x.loc[sch, 'dS']:+.3f}$" r"\\{}"
                         f"[${x.loc[sch, 'ci_lo']:+.3f}$, ${x.loc[sch, 'ci_hi']:+.3f}$]" r"\end{tabular}"
                         if sch in x.index and (sch != "pooled" or full) else "--")
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    (TEX / "stab_arch_paired.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    put("ref_net_label", NAMES[ref_arch].split(") ", 1)[0] + ")")


def supp_tolerance(df: pd.DataFrame):
    """Table S5: crack scores of the adopted network under other protocols (edge-grouped
    schemes pooled, images with annotated cracks)."""
    E = df[df.scheme.isin(EDGE)]
    T = {v: pos(E[E.variant == v], "crack") for v in ("ms", "deployed")}

    def med(v, c):
        return float(T[v][c].median())

    def row(lab, kind, key=None):
        cells = []
        for v in ("ms", "deployed"):
            if kind == "cl":
                cells += [f"{med(v, f'k_cl_P_{key}'):.2f} / {med(v, f'k_cl_R_{key}'):.2f}",
                          f3(med(v, f"k_cl_F1_{key}"))]
            elif kind == "one":
                cells += ["--", f3(med(v, key))]
            elif kind == "iouf1":
                cells += ["--", f"{f3(med(v, 'k_strict_iou'))} / {f3(med(v, 'k_strict_f1'))}"]
            elif kind == "px":
                cells += ["--", f"{med(v, key):.2f} px"]
        return lab + " & " + " & ".join(cells) + r" \\"

    L = [r"\multicolumn{5}{l}{\textit{Centreline, tolerance $\tau$ in pixels (1024 px frame)}} \\"]
    L += [row(rf"\quad $\tau$ = {t[:-2]} px", "cl", t) for t in ("2px", "3px", "5px", "10px", "20px")]
    L.append(r"\multicolumn{5}{l}{\textit{Centreline, tolerance $\tau$ in micrometres}} \\")
    L += [row(rf"\quad $\tau$ = {t[:-2]}\,$\mu$m", "cl", t) for t in ("2um", "5um", "10um")]
    L.append(r"\multicolumn{5}{l}{\textit{Band and topology scores}} \\")
    L += [row(r"\quad Band IoU / $F_1$, strict", "iouf1"),
          row(r"\quad Band IoU, both dilated by 5 px", "one", "k_dil5_iou"),
          row(r"\quad clDice", "one", "k_cldice"),
          row(r"\quad Legacy score (not an IoU)", "one", "k_legacy_score"),
          row(r"\quad Localisation error", "px", "k_loc_err_px")]
    (TEX / "stab_tolerance.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    for v, vk in (("ms", "ms"), ("deployed", "dep")):
        for t in ("2px", "5px", "10px"):
            put(f"tol_{vk}_F1_{t}", f2(med(v, f"k_cl_F1_{t}")))
        for t in ("5px", "10px", "20px"):
            put(f"tol_{vk}_P_{t}", f2(med(v, f"k_cl_P_{t}")))
            put(f"tol_{vk}_R_{t}", f2(med(v, f"k_cl_R_{t}")))
        put(f"tol_{vk}_strict_iou", f2(med(v, "k_strict_iou")))
        put(f"tol_{vk}_legacy", f2(med(v, "k_legacy_score")))
        put(f"tol_{vk}_locerr", f"{med(v, 'k_loc_err_px'):.1f}")
    put("tol_n_k", str(len(T["ms"])))
    # statements of Section S5.2 that depend on these values
    taus = ("2px", "3px", "5px", "10px", "20px", "2um", "5um", "10um")
    signs = {np.sign(round(med("ms", f"k_cl_F1_{t}") - med("deployed", f"k_cl_F1_{t}"), 3)) for t in taus}
    claim(len(signs) == 1, "S5.2 'the ranking of the two outputs does not change with the tolerance'")
    claim(med("ms", "k_cl_R_5px") >= 0.9 and med("ms", "k_cl_R_10px") - med("ms", "k_cl_R_5px") <= 0.06,
          "S5.2 'recall is high and nearly saturated beyond 5 pixels'")
    claim(med("ms", "k_cl_P_5px") < med("ms", "k_cl_P_10px") < med("ms", "k_cl_P_20px"),
          "S5.2 'precision keeps rising with the tolerance'")
    claim(max(med("ms", "k_loc_err_px"), med("deployed", "k_loc_err_px")) <= 2.5,
          "S5.2 'matched centreline within about two pixels of the annotation'")


def supp_magnification(df: pd.DataFrame):
    """Scores by magnification group (pixel size below or above 0.3 um)."""
    E = df[df.scheme.isin(EDGE)]
    L = []
    for v, vl in (("ms", "Network output"), ("deployed", "Deployed pipeline")):
        d = E[E.variant == v]
        assert set(d.loc[d.um_px < 0.3, "um_px"].round(2)) <= {0.22, 0.23}   # 500x only
        for tag, lab, sel in (("lo", r"50--150$\times$", d.um_px >= 0.3),
                              ("hi", r"500$\times$", d.um_px < 0.3)):
            g = d[sel]
            c, k = pos(g, "crater"), pos(g, "crack")
            if v == "ms":
                put(f"mag_{tag}_k2um", f2(k.k_cl_F1_2um.median()))
                put(f"mag_{tag}_kR5", f2(k.k_cl_R_5px.median()))
                put(f"mag_{tag}_c_diam", f"{c.drop_duplicates('name').r_crater_largest_diam_um.median():.0f}")
            L.append(f"{vl}, {lab} & {g.name.nunique()} & {f2(c.c_iou.median())} & "
                     f"{f2(k.k_cl_P_5px.median())} & {f2(k.k_cl_R_5px.median())} & {f2(k[K5].median())} & "
                     f"{f2(k.k_cl_F1_2um.median())} & {f2(k.k_cl_F1_5um.median())} \\\\")
    (TEX / "stab_magnification.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    claim(float(N["mag_lo_kR5"]) >= float(N["mag_hi_kR5"]) and float(N["mag_lo_k2um"]) < float(N["mag_hi_k2um"]),
          "4.2 / S5.3 / letter S7 'at the lower magnifications cracks are found but placed less precisely "
          "in physical terms'")
    claim(float(N["mag_lo_c_iou"]) > float(N["mag_hi_c_iou"])
          and float(N["mag_hi_c_diam"]) < 0.2 * float(N["mag_lo_c_diam"]),
          "S5.3 'crater like regions are segmented better at 50 to 150x; at 500x they are much smaller'")


def supp_steps(df: pd.DataFrame, adopted: str):
    """Table of the steps of the deployed pipeline: the global pass, the tiled pass alone, their
    fusion (network output), the ridge enhancement and hysteresis, and the gap bridging (deployed
    pipeline), out of fold, both edge-grouped repetitions pooled.  The tiled pass alone is not in
    final_records.csv; it comes from the records of the architecture search (arch_eval.py), which
    evaluated the same out-of-fold maps of the same networks with the same code."""
    E = df[df.scheme.isin(EDGE)]
    ar = pd.read_csv(R / "arch_records.csv")
    T = ar[(ar.arch == adopted) & ar.scheme.isin(EDGE)]
    # same maps, same code: the fused output of both sources must agree
    a_ms = T[T.variant == "ms"].set_index(["scheme", "name"])[K5].sort_index()
    f_ms = E[E.variant == "ms"].set_index(["scheme", "name"])[K5].sort_index()
    claim(len(a_ms) == len(f_ms) and np.allclose(a_ms.fillna(-1).to_numpy(), f_ms.fillna(-1).to_numpy()),
          "S5.4: the tiled pass alone (search records) comes from the same maps as the final records")
    steps = (("net", "net", r"Global pass"),
             ("tile_net", "tile", r"Tiled pass alone"),
             ("ms", "ms", r"Global and tiled passes fused (network output)"),
             ("ms_hy", "hy", r"+ ridge enhancement and hysteresis"),
             ("deployed", "dep", r"+ gap bridging (deployed pipeline)"))
    L = []
    for var, t, lab in steps:
        d = T[T.variant == var] if var == "tile_net" else E[E.variant == var]
        if d.empty:
            L.append(f"{lab} & -- & -- & -- & -- & -- & -- \\\\")
            continue
        c, k = pos(d, "crater"), pos(d, "crack")
        kneg = d[~d.crack_ref.astype(bool)]
        put(f"step_{t}_c_iou", f2(c.c_iou.median()))
        put(f"step_{t}_P", f2(k.k_cl_P_5px.median()))
        put(f"step_{t}_R", f2(k.k_cl_R_5px.median()))
        put(f"step_{t}_F1", f2(k[K5].median()))
        put(f"step_{t}_band_iou", f2(k.k_strict_iou.median()))
        put(f"step_{t}_R10", f2(k.k_cl_R_10px.median()))
        put(f"step_{t}_kfa", str(int(kneg.crack_pred.astype(bool).sum())))
        L.append(f"{lab} & {f2(c.c_iou.median())} & {f2(k.k_cl_P_5px.median())} & "
                 f"{f2(k.k_cl_R_5px.median())} & {f2(k[K5].median())} & {f2(k.k_strict_iou.median())} & "
                 f"{int(kneg.crack_pred.astype(bool).sum())}/{len(kneg)} \\\\")
    put("step_n_kneg", str(len(E[(E.variant == "ms") & ~E.crack_ref.astype(bool)])))
    (TEX / "stab_steps.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    if "step_tile_F1" in N:
        claim(float(N["step_tile_F1"]) > float(N["step_ms_F1"]),
              "S5.4 / Discussion: the tiled pass alone has a higher crack centreline F1 than the fused output")
        claim(float(N["step_tile_P"]) > float(N["step_ms_P"]) and float(N["step_tile_R"]) >= float(N["step_ms_R"]),
              "S5.4 / Discussion: fusing the global pass lowers both the precision and the recall of the tiled pass")
        claim(float(N["step_ms_c_iou"]) > float(N["step_tile_c_iou"]),
              "S5.4: the fusion raises the crater IoU with respect to the tiled pass alone")


def supp_undamaged(df: pd.DataFrame):
    """The four fields without any annotation, out of fold."""
    old = json.load(open(R / "numbers.json", encoding="utf-8"))
    names = [n for n in df.name.unique()
             if not df.loc[df.name == n, "crater_ref"].astype(bool).any()
             and not df.loc[df.name == n, "crack_ref"].astype(bool).any()]
    put("nd_n", str(len(names)))
    nd = df[df.name.isin(names)]
    for sch, t in (("edge5_r1", "r1"), ("edge5_r2", "r2")):
        ms = nd[(nd.scheme == sch) & (nd.variant == "ms")]
        dep = nd[(nd.scheme == sch) & (nd.variant == "deployed")]
        if ms.empty:
            continue
        put(f"nd_ms_c_{t}", str(int(ms.crater_pred.astype(bool).sum())))
        put(f"nd_ms_k_{t}", str(int(ms.crack_pred.astype(bool).sum())))
        put(f"nd_ms_k_ana_{t}", str(int((ms.m_n_cracks_rev > 0).sum())))
        put(f"nd_ms_c_ana_{t}", str(int((ms.m_n_craters > 0).sum())))
        put(f"nd_dep_k_{t}", str(int(dep.crack_pred.astype(bool).sum())))
        put(f"nd_dep_k_ana_{t}", str(int((dep.m_n_cracks_rev > 0).sum())))
    dep = nd[nd.variant == "deployed"]
    put("nd_dep_k_len_max", f"{dep.m_crack_len_rev_um.max():.0f}" if len(dep) else "--")
    for k in ("nd_names", "nd_cla_c", "nd_cla_k"):
        put(k, old[k])


EX_REF = {  # segmentation examples: figure reference as seen from the supplementary material
    "ALMW1-1a-600MW-2p_06": r"\ref{M-fig:pipe_almw1_1a}", "ALMW2-1a-600MW_2p_02": r"\ref{M-fig:pipe_almw2_1a}",
    "ATW2-3a-600MW_09": r"\ref{M-fig:pipe_atw2_3a}", "ATW3-3d_05": r"\ref{M-fig:pipe_atw3_3d}",
    "ALMW2-1c-450MW_2p_04": r"\ref{fig:sx_almw2_1c}", "ATW3-2c_01": r"\ref{fig:sx_atw3_2c}",
    "ATW3-4d_14": r"\ref{fig:sx_atw3_4d}"}
EX_KEY = {"ALMW3-2d-1200MW-HT_13": "sx07", "ALMW2-1c-450MW_2p_04": "sx10", "ATW3-2c_01": "sx12",
          "ATW3-4d_14": "sx14", "ALMW1-1c-450MW_04": "sxE", "ALMW1-3a-600MW_05": "ex04",
          "ALMW3-1a-1200MW_01": "ex05", "ALMW1-1a-600MW-2p_06": "ex06",
          "ALMW2-1a-600MW_2p_02": "ex07", "ATW2-3a-600MW_09": "ex08", "ATW3-3d_05": "ex09"}


def conditions_text(name: str) -> str:
    """Exposure and imaging conditions of an example for its caption (verified magnification)."""
    sc = pd.read_csv(R / "scale_verified.csv").set_index("name")
    mag = f"{sc.loc[name, 'mag_x']:.0f}$\\times$"
    if name.startswith("ALMW"):
        m = RC.load_manifest().set_index("name").loc[name]
        p = int(m.log_pulses)
        pul = "single pulse" if p == 1 else f"{p} pulses"
        t0 = "from the pre-heated state" if bool(m.preheated) else "at room temperature"
        return f"{m.power_MW_m2:.0f}\\,MW/m$^2$, {pul} {t0}, {mag}"
    a = pd.read_csv(R / "manifest_ATW.csv").set_index("name").loc[name]
    pw = f"{a.power_MW_m2:.0f}\\,MW/m$^2$ (nominal, from the image name), " if np.isfinite(a.power_MW_m2) else ""
    return f"{pw}{mag}"


def supp_examples():
    """Table S9 (measurements of the segmentation examples) and caption keys of the examples."""
    p = V2 / "example_measurements.csv"
    for name, k in EX_KEY.items():
        put(f"{k}_cond", conditions_text(name))
    if p.exists():
        ex = pd.read_csv(p)
        order = [n for n in EX_REF if n in set(ex.name)]
        L = []
        for n in order:
            for _, x in ex[ex.name == n].sort_values("kind").iterrows():   # measured, reference
                cv = "--" if pd.isna(x.regular_cv) else f"{x.regular_cv:.2f}"
                first = x.kind == "measured"
                def ii(v):
                    return int(v) if pd.notna(v) else 0

                def ff(v, d=0):
                    return f"{v:.{d}f}" if pd.notna(v) else "0"
                L.append(f"{EX_REF[n] if first else ''} & {n.replace('_', chr(92) + '_') if first else ''} & "
                         f"{'Measured' if first else 'Reference'} & {ii(x.n_craters)} & "
                         f"{ff(x.total_area_um2)} & {ff(x.largest_area_um2)} & {ff(x.largest_diam_um, 1)} & "
                         f"{ii(x.n_cracks)} ({ii(x.n_edge)}/{ii(x.n_near_only)}/{ii(x.n_internal)}) & "
                         f"{ff(x.total_len_um)} & {ff(x.longest_um)} & {cv} \\\\")
        (TEX / "stab_examples.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
        for n in order:
            x = ex[(ex.name == n) & (ex.kind == "measured")].iloc[0]
            if x.fold >= 0:
                put(f"{EX_KEY[n]}_fold", str(int(x.fold) + 1))
            # measurements quoted in the text and in the response letter
            for kind, t in (("measured", "m"), ("reference", "r")):
                y = ex[(ex.name == n) & (ex.kind == kind)]
                if y.empty:
                    continue
                y = y.iloc[0]
                k = f"{EX_KEY[n]}_{t}"
                put(f"{k}_nc", str(int(y.n_craters) if pd.notna(y.n_craters) else 0))
                put(f"{k}_area", f"{y.total_area_um2:.0f}" if pd.notna(y.total_area_um2) else "0")
                put(f"{k}_diam", f"{y.largest_diam_um:.0f}" if pd.notna(y.largest_diam_um) else "0")
                put(f"{k}_nk", str(int(y.n_cracks) if pd.notna(y.n_cracks) else 0))
                for col, s in (("n_edge", "e"), ("n_near_only", "n"), ("n_internal", "i")):
                    put(f"{k}_nk_{s}", str(int(y[col]) if pd.notna(y[col]) else 0))
                put(f"{k}_klen", f"{y.total_len_um:.0f}" if pd.notna(y.total_len_um) else "0")
                put(f"{k}_cv", f2(y.regular_cv) if pd.notna(y.regular_cv) else "--")
                # counts in words, for sentences that must read correctly with 0 or 1
                nc, nk = int(N[f"{k}_nc"]), int(N[f"{k}_nk"])
                put(f"{k}_craters_t", "no retained crater like region" if nc == 0 else
                    "1 retained crater like region" if nc == 1 else f"{nc} retained crater like regions")
                put(f"{k}_cracks_t", "no crack" if nk == 0 else "1 crack" if nk == 1 else f"{nk} cracks")
                kt = ("no crack" if nk == 0 else f"1 crack {N[f'{k}_klen']}\\,$\\mu$m long" if nk == 1
                      else f"{nk} cracks with a total length of {N[f'{k}_klen']}\\,$\\mu$m")
                put(f"{k}_summary", f"{N[f'{k}_craters_t']} and {kt}")
    p = V2 / "example_errors.csv"
    if p.exists():
        for _, x in pd.read_csv(p).iterrows():
            k = EX_KEY[x["name"]]
            put(f"{k}_fold", str(int(x.fold) + 1))
            put(f"{k}_c_iou", f2(x.c_iou))
            for q in ("P", "R", "F1"):
                put(f"{k}_k{q}", f2(x[f"k_cl_{q}_5px"]))


def supp_failures():
    """Section S9.2: values of the four representative failures (make_figs_v2.py failures, first
    repetition, network output), and checks that the description of each case in texts.json
    (fc_text) still applies to the case that the fixed criteria select."""
    p = V2 / "failure_cases.csv"
    if not p.exists():
        return
    fc = pd.read_csv(p).set_index("panel")
    for q, x in fc.iterrows():
        k = f"fc_{q}"
        put(f"{k}_name", x["name"].replace("_", "\\_"))
        put(f"{k}_c_iou", f2(x.c_iou))
        put(f"{k}_kF1", f2(x.k_cl_F1_5px))
        put(f"{k}_klen", f"{x.m_crack_len_rev_um:.0f}")
        put(f"{k}_rlen", f"{x.r_crack_len_um:.0f}")
        put(f"{k}_missed", f"{x.missed_um:.0f}")
        put(f"{k}_unmatched", f"{x.unmatched_um:.0f}")
    a, b, c, d = (fc.loc[q] for q in "abcd")
    claim(a["name"].startswith("ALMW1-3a-") and a.c_iou < 0.2 and a.k_cl_F1_5px > 0.7,
          "S9.2 (a): same position as Fig. 4, very low crater IoU while the cracks are found")
    claim("-4p-HT" in b["name"] and b.r_crack_len_um == 0 and b.c_iou > 0.85,
          "S9.2 (b): pre-heated four-pulse image without annotated cracks whose crater like region is segmented well")
    claim(c.missed_um > 0.3 * c.r_crack_len_um and c.c_iou < 0.6,
          "S9.2 (c): a large part of the annotated crack length is missed, crater IoU low")
    claim(d.missed_um < 0.02 * d.r_crack_len_um and d.c_iou > 0.85 and d.unmatched_um > 0.2 * d.m_crack_len_rev_um,
          "S9.2 (d): the annotated crack is traced over its whole length, much unmatched length")


def example_extras(df: pd.DataFrame):
    """Fig. S7 (pre-heated, edge at the bottom of the frame): the analysis stage applied to the
    out-of-fold deployed output and to the annotation, as quoted in sx07_text."""
    x = df[(df.name == "ALMW3-2d-1200MW-HT_13") & (df.scheme == "edge5_r1") & (df.variant == "deployed")]
    if x.empty:
        return
    x = x.iloc[0]
    put("sx07_m_diam", f"{x.m_crater_largest_diam_um:.0f}")
    put("sx07_rp_diam", f"{x.rp_crater_largest_diam_um:.0f}")
    put("sx07_r_diam", f"{x.r_crater_largest_diam_um:.0f}")
    claim(x.m_crater_largest_diam_um < x.rp_crater_largest_diam_um < x.r_crater_largest_diam_um,
          "S8 Fig. S7 text: the analysis stage truncates the region in the measurement and in the reference")
    claim(not bool(x.crack_ref), "S8 Fig. S7 text: no crack annotated in this image")


def example_claims():
    """Statements of the example texts (texts.json: err04/05, ex06 to ex09, sx07 to sx14) that
    depend on the values."""
    def g(k):
        return float(N[k])
    if "ex04_c_iou" in N:
        claim(g("ex04_c_iou") < 0.3 and g("ex04_kF1") > 0.7, "Fig. 4 text: crater IoU low, cracks mostly found")
        claim(g("ex05_c_iou") > 0.85 and g("ex05_kF1") > 0.95, "Fig. 5 text: both channels work well")
        claim(g("sx07_c_iou") < 0.75 and N["sx07_kF1"] == "0.00",
              "Fig. S7 text: centre of the region missed, crack prediction a false alarm")
    if "ex06_m_klen" in N:
        claim((N["ex06_m_nc"], N["ex06_r_nc"], N["ex06_m_nk"], N["ex06_r_nk"]) == ("3", "2", "3", "2")
              and (N["ex06_m_nk_e"], N["ex06_m_nk_n"], N["ex06_m_nk_i"], N["ex06_r_nk_n"]) == ("1", "1", "1", "2"),
              "Fig. 6 text: three regions and three cracks (1/1/1) against two and two near edge")
        claim(abs(g("ex06_m_klen") / g("ex06_r_klen") - 1) < 0.1 and abs(g("ex06_m_area") / g("ex06_r_area") - 1) < 0.2,
              "Fig. 6 text: measurements close to the reference")
        claim(N["ex07_r_nk"] == "0" and N["ex07_m_nk"] == "1" and N["ex07_m_nk_i"] == "1"
              and N["ex07_m_nc"] == "4" and N["ex07_r_nc"] == "2",
              "Fig. 7 text: four regions against two, one internal crack on an image without annotated cracks")
        claim(abs(g("ex07_m_diam") / g("ex07_r_diam") - 1) < 0.1, "Fig. 7 text: largest region close to the reference")
        claim(N["ex08_m_nc"] == "0" and N["ex08_m_nk"] == "2" and N["ex08_m_nk_e"] == "2",
              "Fig. 8 text: two cracks labelled edge connected, no crater like region")
        claim(N["ex09_m_nk"] == "4" and N["ex09_m_nk_n"] == "4" and N["ex09_m_nc"] == "2" and g("ex09_m_cv") < 0.45,
              "Fig. 9 text: four near edge cracks in the regular class, two small regions")
        claim(abs(g("sx10_m_diam") / g("sx10_r_diam") - 1) < 0.1 and abs(g("sx10_m_klen") / g("sx10_r_klen") - 1) < 0.2
              and N["sx10_m_nk_n"] == N["sx10_m_nk"] == N["sx10_r_nk_n"] == N["sx10_r_nk"] == "2",
              "Fig. S8 text: measurements agree closely with the reference, two near edge cracks")
        claim(N["sx12_m_nc"] == "4" and N["sx12_m_nk"] == "1" and N["sx12_m_nk_i"] == "1",
              "Fig. S9 text: four regions and one internal crack")
        claim(N["sx14_m_nc"] == "3" and N["sx14_m_nk"] == "1" and N["sx14_m_nk_i"] == "1",
              "Fig. S10 text: three regions and one internal crack")


def timing(adopted: str):
    """Section 4.5: time per image of the deployed pipeline with the final model of the adopted
    network (timing.py --ckpt arch_runs/<arch>/all/last_model.pth --out results/v2)."""
    p, q = V2 / "summary_timing.csv", V2 / "timing_stages.csv"
    if not (p.exists() and q.exists()):
        return
    t = pd.read_csv(p).iloc[0]
    st = pd.read_csv(q)
    put("time_gpu", str(t["gpu"]).replace("NVIDIA ", ""))
    put("time_cpu", str(t["cpu"]).replace("(R)", "").replace("(TM)", "").strip())
    put("time_total_med", f"{st.t_total.median():.2f}")
    put("time_total_p90", f"{st.t_total.quantile(0.9):.2f}")
    put("time_passes_med", f"{(st.t_global + st.t_tiled).median():.2f}")
    claim(len(st) == 212, "4.5 'over the 212 ALMW and ATW images'")
    ck = str(t.get("checkpoint", "")).replace("\\", "/")
    claim(f"/{adopted}/all/" in ck, "4.5: timing measured with the final model of the adopted network")


def carry_over():
    old = json.load(open(R / "numbers.json", encoding="utf-8"))
    for k, v in old.items():
        # pub*: network of the first submission (in sample), leg_*: its Table 1 reproduced at the
        # two crack thresholds; both are independent of the networks trained for the revision
        if k.startswith(("obs_", "lt_", "scalerev_", "cla", "hist_", "cl_", "pubAll_", "pubSel_",
                         "pubTr_", "leg_")) or \
                k in ("time_gpu", "time_cpu", "img_claAll_k_FP", "img_claAll_k_neg"):
            put(k, v)


def classical_zeros():
    """Images with cracks on which the classical pipeline places no crack centreline within
    5 px of the reference (F1 = 0), outside and inside the 78 images of its search."""
    p = R / "classical_records.csv"
    if not p.exists():
        return
    c = pd.read_csv(p)
    k = c[c.crack_ref.astype(bool)]
    for lab, d in (("cla21", k[~k.in_GA_set.astype(bool)]), ("cla78", k[k.in_GA_set.astype(bool)])):
        put(f"{lab}_k_zero", str(int((d.k_cl_F1_5px == 0).sum())))
        put(f"{lab}_k_nopred", str(int((~d.crack_pred.astype(bool)).sum())))
        claim(str(len(d)) == N.get(f"{lab}_n_k"), f"S2.5: number of {lab} images with cracks")
    claim(N.get("cla21_k_cl5_med") == N.get("cla78_k_cl5_med") == "0.000",
          "S2.5 / letter 8.2 'its median crack centreline F1 is zero on both sets'")


def _sign(a: str, b: str) -> int:
    """Direction of the change from a to b as printed (two decimals)."""
    fa, fb = float(a), float(b)
    return (fb > fa) - (fb < fa)


def _change(what: str, a: str, b: str, tail_a: str = "", tail_b: str = "", third: bool = False) -> str:
    s, e = _sign(a, b), "s" if third else ""
    if s == 0:
        return f"leave{e} {what} at {a}"
    return f"{'raise' if s > 0 else 'lower'}{e} {what} from {a}{tail_a} to {b}{tail_b}"


def phrases():
    """Clauses whose wording depends on the direction of a result.  They are written from the
    values computed above, so that the text cannot contradict the numbers it quotes."""
    Rm, Rd = N["oof_ms_k_clR5_med"], N["oof_dep_k_clR5_med"]
    Pm, Pd = N["oof_ms_k_clP5_med"], N["oof_dep_k_clP5_med"]
    sR, sP = _sign(Rm, Rd), _sign(Pm, Pd)
    if sR == sP != 0:
        put("dep_steps_change", f"{'raise' if sR > 0 else 'lower'} both the median centreline recall, "
                                f"from {Rm} to {Rd}, and the precision, from {Pm} to {Pd}")
    else:
        put("dep_steps_change", f"{_change('the median centreline recall', Rm, Rd)} and "
                                f"{_change('the precision', Pm, Pd)}")
    put("dep_steps_effect", {
        (1, -1): "trade precision for recall", (-1, 1): "trade recall for precision",
        (-1, -1): "lower both the precision and the recall", (1, 1): "raise both the precision and the recall",
        (0, -1): "lower the precision without changing the recall",
        (0, 1): "raise the precision without changing the recall",
        (1, 0): "raise the recall without changing the precision",
        (-1, 0): "lower the recall without changing the precision",
        (0, 0): "leave the precision and the recall unchanged"}[(sR, sP)])
    put("dep_abs_text", {          # abstract and conclusions
        (1, -1): "raises crack recall at the cost of precision",
        (-1, 1): "raises crack precision at the cost of recall",
        (-1, -1): "lowers both the precision and the recall of the crack centrelines",
        (1, 1): "raises both the precision and the recall of the crack centrelines",
        (0, -1): "lowers the crack precision without changing the recall",
        (0, 1): "raises the crack precision without changing the recall",
        (1, 0): "raises the crack recall without changing the precision",
        (-1, 0): "lowers the crack recall without changing the precision",
        (0, 0): "leaves the crack precision and recall unchanged"}[(sR, sP)])
    r = float(N["agr_dep_lenrev_medratio"])
    put("dep_len_abs", f"overestimates crack lengths by a median factor of {N['agr_dep_lenrev_medratio']}" if r > 1.05
        else f"underestimates crack lengths, with a median ratio of {N['agr_dep_lenrev_medratio']}" if r < 0.95
        else f"returns crack lengths with a median ratio of {N['agr_dep_lenrev_medratio']} to the reference")
    put("dep_f1_change3", _change("the median centreline $F_1$", N["step_ms_F1"], N["step_dep_F1"],
                                  " for the fused passes", " after bridging", third=True))
    # bridging alone: is the recall lost at 5 px recovered at 10 px (displaced centrelines)?
    r5h, r5d, r10h, r10d = N["step_hy_R"], N["step_dep_R"], N["step_hy_R10"], N["step_dep_R10"]
    if _sign(r5h, r5d) < 0 and abs(float(r10d) - float(r10h)) <= 0.01:
        at10 = (f"it is {r10d} both before and after it" if r10h == r10d
                else f"it is {r10h} before it and {r10d} after it")
        put("br_recall_text",
            f"The bridging lowers the median recall at 5 pixels from {r5h} to {r5d}, but at 10 pixels "
            f"{at10}, so the cracks are still found and their centrelines are displaced by a few pixels, "
            f"where the closings merge neighbouring fragments or fill concavities and so move the "
            f"skeleton of the mask.")
    elif _sign(r5h, r5d) < 0:
        put("br_recall_text", f"The bridging lowers the median recall at 5 pixels from {r5h} to {r5d}, "
                              f"and at 10 pixels from {r10h} to {r10d}.")
    else:
        put("br_recall_text", f"The bridging changes the median recall at 5 pixels from {r5h} to {r5d}.")
    # size categories
    U, O = int(N["cat_dep_under"]), int(N["cat_dep_over"])
    if U > O:
        t = f"Most disagreements are downwards, {U} predictions in a lower category against {O} in a higher one"
    elif O > U:
        t = f"Most disagreements are upwards, {O} predictions in a higher category against {U} in a lower one"
    elif U:
        t = f"The disagreements are evenly split, {U} predictions in a lower category and {O} in a higher one"
    else:
        t = "There are no disagreements"
    put("cat_dep_dir_text", t)
    n, img, hit = int(N["cat_dep_top_n"]), int(N["cat_dep_top_img"]), int(N["cat_dep_top_hit"])
    hit_t = (("neither" if n == 2 else "none") if hit == 0 else
             ("both" if n == 2 else "all") if hit == n else str(hit))
    img_t = "the image" if img == 1 else f"the {img} images"
    put("cat_dep_top_text", f"places {hit_t} of the {n} predictions of {img_t} whose largest annotated "
                            f"region exceeds 950\\,$\\mu$m in that category")
    # statements of the main text checked against the values
    if "loso_ms_c_iou_med" in N:
        claim(float(N["loso_ms_c_iou_med"]) < float(N["oof_ms_c_iou_med"])
              and float(N["loso_ms_k_cl5_med"]) >= float(N["oof_ms_k_cl5_med"]),
              "Abstract, 4.2, Conclusions 'leaving out a whole specimen lowers the median crater IoU, "
              "while the crack F1 is not lower'")
    if "miss_k_len_med" in N:
        claim(float(N["miss_k_len_med"]) < 0.5 * float(N["all_k_len_med"]),
              "4.2 'the cracks missed entirely are mostly short ones'")
    ms_r, dep_r = float(N["agr_ms_lenrev_medratio"]), float(N["agr_dep_lenrev_medratio"])
    claim(0.95 <= ms_r <= 1.05, "4.3 'the network output is right in the median where it reports cracks'")
    claim(int(N["agr_dep_lenrev_missed"]) < int(N["agr_ms_lenrev_missed"]), "4.3 'the deployed pipeline misses fewer cracks'")
    claim(dep_r > 1.05, "4.3 'the deployed pipeline overestimates the length'")
    claim(dep_r > ms_r, "4.3 'deployed length an upper estimate that includes false detections'")
    claim(int(N["oof_dep_fa_k"]) > int(N["oof_n_kneg"]) / 2,
          "Conclusions 'reports cracks on most crack free images' (now quoted with numbers)")
    claim(int(N["oof_dep_fa_k"]) > int(N["oof_ms_fa_k"]),
          "4.2 / letter 5.1: the deployed pipeline reports cracks on more crack free images than the network output")
    claim(all(0.85 <= float(N[k]) <= 1.15 for k in ("agr_dep_area_medratio", "agr_dep_diam_medratio")),
          "Discussion 'crater areas and diameters agree with the reference with medians close to one'")
    claim(int(N["agr_dep_area_out2"]) > 0, "Discussion 'a single image can be off by a factor of two or more'")
    claim(float(N["br_prec_added"]) <= 0.25, "S5.4 / letter 5.3 'the added connections rarely correspond to annotated cracks'")
    claim(int(N["reg_agree"]) / max(int(N["reg_both_eval"]), 1) < int(N["cat_dep_exact"]) / 100,
          "4.3 'the regularity of the crack network is the most fragile output'")
    claim(float(N["oof_ms_k_cl5_med"]) > float(N["cla21_k_cl5_med"])
          and float(N["oof_ms_c_iou_med"]) > float(N["cla21_c_iou_med"]),
          "the network outperforms the classical pipeline outside its search")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    adopted = sys.argv[1]
    ref_arch = sys.argv[2] if len(sys.argv) > 2 else adopted
    carry_over()
    timing(adopted)
    conditions()
    arch_section(ref_arch, adopted)
    df = pd.read_csv(V2 / "final_records.csv")
    oof_section(df)
    measurement_section(df)
    bridging_section()
    insample_atw()
    supp_arch_tables(ref_arch, adopted)
    supp_tolerance(df)
    supp_magnification(df)
    supp_steps(df, adopted)
    supp_undamaged(df)
    supp_examples()
    supp_failures()
    example_extras(df)
    example_claims()
    classical_zeros()
    phrases()
    tp = V2 / "texts.json"
    if tp.exists():
        for k, v in json.load(open(tp, encoding="utf-8")).items():
            put(k, v)
    # hand-written texts may quote computed values as \V{key}: resolve them here
    rv = re.compile(r"\\V\{([A-Za-z0-9_]+)\}")
    for _ in range(3):
        for k, v in list(N.items()):
            if isinstance(v, str) and "\\V{" in v:
                N[k] = rv.sub(lambda m: str(N[m.group(1)]) if m.group(1) in N else m.group(0), v)
    left = sorted({m for v in N.values() if isinstance(v, str) for m in rv.findall(v)})
    if left:
        print("unresolved keys inside texts:", ", ".join(left))
    (V2 / "numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False), encoding="utf-8")
    print(len(N), "values written")
    for k in sorted(N):
        if not k.startswith(("obs_", "lt_", "scalerev_")) and len(str(N[k])) < 40:
            print(f"  {k:28s} {N[k]}")
    for c in CLAIMS:
        print("CLAIM NOT SUPPORTED BY THE VALUES:", c)
    if not CLAIMS:
        print("all checked statements are supported by the values")


if __name__ == "__main__":
    main()
