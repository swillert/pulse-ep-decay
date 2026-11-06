#!/usr/bin/env python3
"""make_revision_tables.py — the single source of every number in the revision
of CMPB-D-26-02177 (plan: revision_2026-10/TABLES_PLAN_2026-10-25.md).

Reads the result files of the tests (v36-v48, arm3) and the map metadata,
builds the per-map master table (S12) and derives Table 1 (validation,
variant C), Table 2 (clinical by configuration/region), and the supplementary
tables; writes .tex (booktabs) and .csv into revision_2026-10/tables/ and a
numbers.json / numbers.md with every headline number used in the text.
Numbers from September runs (arm3) are marked as such.  Usage: --out <dir>.
"""
import argparse, json, os, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

ROOT = os.environ.get("GD_ROOT", os.path.expanduser("~/mnt/IdeaProjects"))  # GD_ROOT: root of an unpacked deposit (round 6)
MP = os.path.join(ROOT, "thesis-med/_paper/method-paper"); SVR = os.path.join(MP, "supplementary_validation_results")
SUMMARY = os.path.join(ROOT, "pulse-ultimate/decay_profiles_heat/decay_summary.csv")
P = lambda *a: os.path.join(SVR, *a)
SRC = dict(v36d=P("v36_analytic_geodesic_2026-11-06b", "v36_distance_errors.csv"), v36w=P("v36_analytic_geodesic_2026-11-06b", "v36_width_by_solver.csv"),
           v37=P("v59_r2_threshold_plateau_2026-10-26", "v59_fields.csv"), v38=P("v38_clinical_solvers_plateau_2026-10-26", "v38_per_map.csv"),
           v39=P("v39_focality_clinical_plateau_2026-10-26", "v39_per_map.csv"), v40=P("v40_mixed_model_plateau_2026-10-26", "v40_summary.txt"),
           v41=P("v41_runtime_sparsity_2026-10-21", "v41_per_mesh.csv"), v42=P("v42_range_extension_plateau_2026-10-26", "v42_summary.csv"),
           v43=P("v43_lowr2_profiles_plateau_2026-10-26", "v43_per_map.csv"), v44=P("v60_anisotropy_pkgfit_2026-10-26", "v60_rows.csv"),
           v45=P("v45_mesh_roughness_binwidth_plateau_2026-10-26", "v45_per_map.csv"), v46c=P("v46_lo_origin_plateau_2026-10-26", "v46_clinical.csv"),
           v46v=P("v46_lo_origin_plateau_2026-10-26", "v46_validation_raw.csv"), v47av=P("v47a_origin_rule_plateau_2026-10-26", "v47a_validation_raw.csv"),
           v47ac=P("v47a_origin_rule_plateau_2026-10-26", "v47a_clinical.csv"), v47b=P("v47b_structured_noise_plateau_2026-10-26", "v47b_raw.csv"),
           v47c=P("v47c_scatter_sources_plateau_2026-10-26", "v47c_summary.csv"), arm3=P("arm3_plateau_2026-10-26", "arm3_summary.csv"), study=P("map_study.csv"),
           v36b=P("v36c_width_plateau_2026-10-26", "v36c_width_raw.csv"), v49=P("v49_aniso_perm_plateau_2026-10-26", "v49_clinical.csv"),
           v50c=P("v62_focality_unified_2026-10-26", "v62_clinical.csv"), v50s=P("v62_focality_unified_2026-10-26", "v62_sphere.csv"), v50m=P("v62_focality_unified_2026-10-26", "v62_meshes.csv"),
           v56all=P("v56_peak_below_clip_all_2026-10-26", "v56_raw.csv"), v51=P("v51_argmax_ties_plateau_2026-10-26", "v51_ties.csv"), v52=P("v52_model_comparison_plateau_2026-10-26", "v52_per_map.csv"),
           occ=os.path.join(MP, "revision_2026-10/figs/_cache/origin_component_check.csv"),
           v58t=P("v58_model_diagnostics_2026-10-26", "v58_terms.csv"), v58i=P("v58_model_diagnostics_2026-10-26", "v58_inference.json"))
# R1 round 4 (26 Oct): S11a from v59 (package fitter + plateau-centre origin; v37/v53 superseded), S1b from v36c (same), S7 from v60
# (package fitter; identical to v44 within 0.0003 mm); mixed-model inference from v58 (REML at the profiled optimum, t(G-2)),
# because the statsmodels REML fit of v40 did not converge (v57).  v40 still supplies the non-model patient medians and bootstrap.
NUM = {}  # headline numbers



def tex_minus(caption):
    """Typographic minus before a number in caption text: " -0.18" -> " $-$0.18" (hyphens such as 2-30 unchanged)."""
    return re.sub(r"(?<=[ (])-(?=\d)", "$-$", caption)

def q(x, lo=.25, hi=.75): return x.quantile(lo), x.quantile(hi)
def f1(x): return "%.1f" % x
def f2(x): return "%.2f" % x
def fs(x): return ("%+.2f" % x)
def med_iqr(x, d=1): return ("%%.%df [%%.%df; %%.%df]" % (d, d, d)) % (x.median(), x.quantile(.25), x.quantile(.75))


def cell(c):
    c = str(c)
    if c.startswith("$"): return c
    c = re.sub(r"(?<![A-Za-z0-9_\\$-])-(?=\d)", r"$-$", c)  # text-mode hyphen before a digit -> math minus (not inside --)
    return c.replace("$-$0.00", "0.00").replace("-0.00", "0.00").replace("+0.00", "0.00")


def tex_table(rows, header, align, caption, label, notes="", star=False, size="small", group=None, colsep="4pt"):
    env = "table*" if star else "table"
    L = ["\\begin{%s}[t]" % env, "  \\centering", "  \\caption{%s}" % caption, "  \\label{%s}" % label, "  \\%s\\setlength{\\tabcolsep}{%s}" % (size, colsep), "  \\begin{tabular}{%s}" % align, "    \\toprule"]
    if group:  # list of (label, first_col, last_col), 1-based
        L += ["    " + " & ".join(["\\multicolumn{%d}{c}{%s}" % (b - a + 1, lab) if lab else "" for lab, a, b in group]) + " \\\\", "    " + " ".join("\\cmidrule(lr){%d-%d}" % (a, b) for lab, a, b in group if lab)]
    L += ["    " + " & ".join(header) + " \\\\", "    \\midrule"] + ["    " + " & ".join(cell(c) for c in r) + " \\\\" for r in rows] + ["    \\bottomrule", "  \\end{tabular}"]
    if notes: L.append("  \\begin{flushleft}\\footnotesize " + notes + "\\end{flushleft}")
    L.append("\\end{%s}" % env); return "\n".join(L) + "\n"


def long_table(rows, header, align, caption, label, size="footnotesize"):
    L = ["\\begin{%s}" % size, "\\setlength{\\tabcolsep}{4pt}", "\\begin{longtable}{%s}" % align, "\\caption{%s}\\label{%s}\\\\" % (caption, label), "\\toprule", " & ".join(header) + " \\\\", "\\midrule", "\\endfirsthead",
         "\\toprule", " & ".join(header) + " \\\\", "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"] + [" & ".join(cell(c) for c in r) + " \\\\" for r in rows] + ["\\end{longtable}", "\\end{%s}" % size]
    return "\n".join(L) + "\n"


def cluster_boot_rho(df, a, b, n=2000, seed=0):
    """Spearman rho with a 95 % interval from resampling patients (maps of a patient are not independent)."""
    rng = np.random.default_rng(seed); pats = df.patient.unique(); g = {p: df[df.patient == p] for p in pats}; out = []
    for _ in range(n):
        s = pd.concat([g[p] for p in rng.choice(pats, len(pats))]); out.append(spearmanr(s[a], s[b]).correlation)
    return tuple(np.nanpercentile(out, [2.5, 97.5]))


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x); m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)]); return np.percentile(m, 2.5), np.percentile(m, 97.5)


def classify(s):
    # classes of Section 5.5 (operating range, R1 plateau-centre origin): below the validated range, in range, broad, diffuse, not identifiable
    return np.select([~np.isfinite(s) | (s >= 100), s >= 50, s >= 30, s >= 2], ["n.i.", "diffuse", "broad", "in range"], default="below range")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    W = lambda name, txt: open(os.path.join(a.out, name), "w").write(txt)
    summ = pd.read_csv(SUMMARY).set_index("map_id"); study = pd.read_csv(SRC["study"]).set_index("map_id")["study_id"]

    # ------------------------------------------------------------------ S12 master
    v38all = pd.read_csv(SRC["v38"]); v38all = v38all[~v38all.fragmented]; v38 = v38all[~v38all.excluded_quality]
    v39 = pd.read_csv(SRC["v39"]).set_index("map_id"); v43 = pd.read_csv(SRC["v43"]).set_index("map_id"); v46 = pd.read_csv(SRC["v46c"]).set_index("map_id")
    M = pd.DataFrame(dict(map_id=v38.map_id.values)).set_index("map_id")
    M["patient"] = [int(study[m]) for m in M.index]; M["atrium"] = [summ.loc[m, "atrium"] for m in M.index]; M["region"] = [summ.loc[m, "part"].replace(" wall", "") for m in M.index]
    M["reference"] = ["double" if summ.loc[m, "ref_catheters"] == 2 else "single" for m in M.index]; M["n_points"] = [int(summ.loc[m, "n_points"]) for m in M.index]
    M["n_vertices"] = v38.set_index("map_id").loc[M.index, "n_largest"].values; M["sigma"] = v38.set_index("map_id").loc[M.index, "sigma_heat"].values; M["r2"] = v38.set_index("map_id").loc[M.index, "r2_heat"].values
    v50 = pd.read_csv(SRC["v50c"]).set_index("map_id").loc[M.index]
    assert np.allclose(v50.sigma.values, M.sigma.values) and np.allclose(v50.r2.values, M.r2.values)
    M["class"] = classify(M.sigma.values); M["flag_r2"] = M.r2 < 0.7; M["F"] = v50.F.values; M["F_low"] = M.F < 0.70; M["flag_F"] = M.F_low & (M.r2 >= 0.7);  # F is interpreted only when R2 >= 0.7 (Sec. 4.3)
    M["n_components"] = v50.ncomp.values
    M["F_pct"] = v39.loc[M.index, "F_edge_p95"].values; M["n_components_pct"] = v39.loc[M.index, "ncomp_edge_p95"].values; M["origin_in_largest_pct"] = pd.read_csv(os.path.join(MP, "revision_2026-10/figs/_cache/origin_component_check.csv")).set_index("map_id").loc[M.index, "origin_in_main"].astype(bool).values  # origin vertex in the largest component of the 95th-percentile set (submitted F)
    M["other_min_dist_mm"] = v50.other_min_dist_mm.values; M["other_max_score_rel"] = v50.other_max_score_rel.values; M["thr_half"] = v50.thr_half.values
    M["dAIC_gauss_minus_exp"] = v43.loc[M.index, "dAIC_gauss_minus_exp"].values; M["beta"] = v43.loc[M.index, "beta"].values; M["fwhm_modelfree"] = v43.loc[M.index, "fwhm_modelfree"].values
    M["A"] = v43.loc[M.index, "A"].values; M["B"] = v43.loc[M.index, "B"].values
    for c, n in (("sigma_exact", "sigma_exact"), ("sigma_dijkstra", "sigma_dijkstra"), ("sigma_euclid", "sigma_euclid"), ("r2_euclid", "r2_euclid")): M[n] = v38.set_index("map_id").loc[M.index, c].values
    M["marker"] = [bool(v46.loc[m, "route"] == "unique_lo_reference_candidate") if m in v46.index else False for m in M.index]
    M["offset_marker_mm"] = [v46.loc[m, "offset_geo_mm"] if m in v46.index else np.nan for m in M.index]; M["sigma_marked_origin"] = [v46.loc[m, "sigma_lo"] if m in v46.index else np.nan for m in M.index]
    M["r2_marked_origin"] = [v46.loc[m, "r2_lo"] if m in v46.index else np.nan for m in M.index]
    M.to_csv(os.path.join(a.out, "S12_per_map.csv"))
    rows = [[m, r.patient, "%s %s" % (r.atrium, r.region), r.reference, r.n_points, f1(r.sigma), f2(r.r2), r["class"], f2(r.F), int(r.n_components), f2(r.F_pct), fs(r.dAIC_gauss_minus_exp), f2(r.beta) if np.isfinite(r.beta) else "--",
             f1(r.fwhm_modelfree) if np.isfinite(r.fwhm_modelfree) else "--", f1(r.offset_marker_mm) if np.isfinite(r.offset_marker_mm) else "--"] for m, r in M.iterrows()]
    W("S12_per_map.tex", tex_table(rows, ["map", "pat.", "region", "ref.", "pts", "$\\sigma$", "$R^2$", "class", "$F$", "comp.", "$F_{\\rm pct}$", "$\\Delta$AIC", "$\\beta$", "FWHM$_{\\rm emp}$", "offset"], "rrllrrrlrrrrrrr",
                                   "Per-map results of the 55 clinical maps (Heat Method). class (width range only, not a judgement of fit quality, which the flags give): below range ($\\sigma<2$ mm), in range ($2\\le\\sigma<30$ mm), broad ($30\\le\\sigma<50$ mm), diffuse ($50\\le\\sigma<100$ mm), n.i. ($\\sigma\\ge100$ mm); $F$: focality index (share of the half-maximum set connected to the origin) and number of components of that set; $F_{\\rm pct}$: percentile index (largest component of the 95th-percentile set); $\\Delta$AIC $=$ AIC$_{\\rm Gauss}-$AIC$_{\\rm exp}$ on the binned profile; $\\beta$ generalised-Gaussian exponent; FWHM$_{\\rm emp}$ empirical half-maximum width (mm); offset: geodesic distance from the vertex of maximum score to the marked pacing site (mm; 46 maps).",
                                   "tab:S12", star=True, size="footnotesize"))
    # longtable version for the supplement (55 rows do not fit a float)
    hdr = ["map", "pat.", "region", "ref.", "pts", "$\\sigma$", "$R^2$", "class", "$F$", "comp.", "$F_{\\rm pct}$", "$\\Delta$AIC", "$\\beta$", "FWHM$_{\\rm emp}$", "offset"]
    cap = open(os.path.join(a.out, "S12_per_map.tex")).read().split("\\caption{")[1].split("}\n")[0]
    L = ["\\begin{footnotesize}", "\\setlength{\\tabcolsep}{4pt}", "\\begin{longtable}{rrllrrrlrrrrrrr}", "\\caption{" + cap + "}\\label{tab:S12}\\\\", "\\toprule", " & ".join(hdr) + " \\\\", "\\midrule", "\\endfirsthead",
         "\\toprule", " & ".join(hdr) + " \\\\", "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"] + [" & ".join(cell(c) for c in r) + " \\\\" for r in rows] + ["\\end{longtable}", "\\end{footnotesize}"]
    W("S12_per_map_long.tex", "\n".join(L) + "\n")

    # ------------------------------------------------------------------ Table 2 clinical by group
    def grp_row(name, g):
        pats = g.patient.nunique(); cls = g["class"].value_counts()
        return [name, "%d (%d)" % (len(g), pats), med_iqr(g.sigma), "%d / %d / %d / %d" % (cls.get("below range", 0), cls.get("in range", 0), cls.get("broad", 0), cls.get("diffuse", 0) + cls.get("n.i.", 0)), int(g.flag_r2.sum()), int(g.flag_F.sum())]
    rows = [grp_row("all maps", M), grp_row("double-chamber reference", M[M.reference == "double"]), grp_row("single-chamber reference", M[M.reference == "single"])]
    for atr, reg in (("LA", "posterior"), ("LA", "anterior"), ("RA", "posterior"), ("RA", "lateral")):
        rows.append(grp_row("%s %s wall" % (atr, reg), M[(M.atrium == atr) & (M.region == reg)]))
    lmm = open(SRC["v40"]).read(); blk = lmm.split("[all 55, raw sigma]")[1]
    g1 = lambda pat: re.search(pat, blk)
    L = dict(ratio=g1(r"reference\[T.single\].*?ratio exp\(coef\) ([\d.]+) \[([\d.]+); ([\d.]+)\]\s+p=([\d.]+)").groups(),
             lat=g1(r"region\[T.RA lateral\].*?ratio exp\(coef\) ([\d.]+) \[([\d.]+); ([\d.]+)\]\s+p=([\d.]+)").groups(),
             rap=g1(r"region\[T.RA posterior\].*?ratio exp\(coef\) ([\d.]+) \[([\d.]+); ([\d.]+)\]\s+p=([\d.]+)").groups(),
             icc=g1(r"ICC ([\d.]+)").group(1), lrt=g1(r"chi2 = ([\d.]+), p = ([\d.]+)").groups(),
             pm=re.search(r"patient medians: single n=(\d+) median ([\d.]+) \[[^\]]*\]; double n=(\d+) median ([\d.]+) \[[^\]]*\]; Mann-Whitney p=([\d.]+)", lmm.split("[all 55, raw sigma] patient")[1] if False else lmm).groups(),
             boot=re.search(r"ratio of map-level medians single/double = ([\d.]+); patient-cluster bootstrap 95% interval \[([\d.]+); ([\d.]+)\]", lmm).groups())
    NUM["lmm"] = L
    T58 = pd.read_csv(SRC["v58t"]); I58 = json.load(open(SRC["v58i"])); NUM["model"] = dict(terms=T58.to_dict("records"), inference=I58)
    tt = lambda an, term: T58[(T58.analysis == an) & (T58.term == term)].iloc[0]
    pf3 = lambda x: "%.3f" % x if x >= 0.001 else "<0.001"
    r_, la_, rp_ = tt("all55_raw", "reference[T.single]"), tt("all55_raw", "region[T.RA lateral]"), tt("all55_raw", "region[T.RA posterior]"); i_ = I58["all55_raw"]
    cap2 = ("Width of the 55 clinical maps by reference catheter configuration and region (Heat Method). Linear mixed model of $\\log\\sigma$ (reference and region as fixed effects, patient random intercept; 95\\,\\%% CI from an approximate $t$ reference with %d degrees of freedom): "
            "single- vs double-chamber ratio of geometric means %.2f (%.2f--%.2f); at the patient level (%d vs %d patients, two-stage analysis) %.2f (%.2f--%.2f). Diagnostics and further estimands in Supplementary Section~S15. Width classes of Section 5.5. \\textsuperscript{a}Counted only for maps with an adequate fit ($R^2 \\ge 0.7$), for which $F$ is interpreted.") % (
            int(r_.df_t), r_.ratio, r_.t_lo, r_.t_hi, i_["n_single"], i_["n_double"], i_["patient_adj"]["ratio"], i_["patient_adj"]["lo"], i_["patient_adj"]["hi"])
    W("T2_clinical_groups.tex", tex_table(rows, ["", "maps (patients)", "$\\sigma$, median [IQR] (mm)", "$<$2 / 2--30 / 30--50 / $\\ge$50 mm", "$R^2<0.7$", "$F<0.70$\\textsuperscript{a}"], "lcccrr", cap2, "tab:clinical_groups", star=True))
    NUM["clinical"] = dict(n_maps=len(M), n_patients=int(M.patient.nunique()), sigma_median=float(M.sigma.median()), sigma_q1=float(M.sigma.quantile(.25)), sigma_q3=float(M.sigma.quantile(.75)),
                           sigma_min=float(M.sigma.min()), sigma_max=float(M.sigma.max()), r2_median=float(M.r2.median()), r2_min=float(M.r2.min()), r2_max=float(M.r2.max()),
                           n_flag_r2=int(M.flag_r2.sum()), n_flag_F=int(M.flag_F.sum()), classes=M["class"].value_counts().to_dict(),
                           sigma_median_double=float(M[M.reference == "double"].sigma.median()), sigma_median_single=float(M[M.reference == "single"].sigma.median()),
                           iqr_double=list(map(float, q(M[M.reference == "double"].sigma))), iqr_single=list(map(float, q(M[M.reference == "single"].sigma))),
                           F_median=float(M.F.median()), F_iqr=list(map(float, q(M.F))), F_min=float(M.F.min()), F_flagged_maps=[int(x) for x in M[M.flag_F].index], F_pct_median=float(M.F_pct.median()), F_pct_iqr=list(map(float, q(M.F_pct))), n_flag_F_pct=int((M.F_pct < .7).sum()),
                           F_flag_sigma_median=float(M[M.flag_F].sigma.median()), F_flag_sigma_range=[float(M[M.flag_F].sigma.min()), float(M[M.flag_F].sigma.max())], F_flag_other_dist=[float(M[M.flag_F].other_min_dist_mm.min()), float(M[M.flag_F].other_min_dist_mm.max())], F_flag_other_height=[float(M[M.flag_F].other_max_score_rel.min()), float(M[M.flag_F].other_max_score_rel.max())],
                           F_flag_and_r2=int((M.flag_F & M.flag_r2).sum()), F_flag_reference=M[M.flag_F].reference.value_counts().to_dict(), F_threshold={T: int((M.F < T).sum()) for T in (0.5, 0.6, 0.7, 0.8, 0.9)}, F_pct_threshold={T: int((M.F_pct < T).sum()) for T in (0.5, 0.6, 0.7, 0.8, 0.9)},
                           n_F_eligible=int((M.r2 >= 0.7).sum()), F_low_raw=int(M.F_low.sum()), F_low_ineligible=int((M.F_low & (M.r2 < 0.7)).sum()), F_median_eligible=float(M[M.r2 >= 0.7].F.median()), F_iqr_eligible=list(map(float, q(M[M.r2 >= 0.7].F))), ncomp_1_eligible=int(((M.n_components == 1) & (M.r2 >= 0.7)).sum()), F_threshold_eligible={T: int(((M.F < T) & (M.r2 >= 0.7)).sum()) for T in (0.5, 0.6, 0.7, 0.8, 0.9)}, rho_F_sigma_eligible=float(spearmanr(M[M.r2 >= 0.7].F, M[M.r2 >= 0.7].sigma).correlation), rho_F_r2_eligible=float(spearmanr(M[M.r2 >= 0.7].F, M[M.r2 >= 0.7].r2).correlation), class_counts=M["class"].value_counts().to_dict(), class_below_range_maps=[int(x) for x in M[M["class"] == "below range"].index], n_points_range=[int(M.n_points.min()), int(M.n_points.max())], n_points_median=float(M.n_points.median()), rho_npoints_sigma=float(spearmanr(M.n_points, M.sigma).correlation), mean_edge_range=None,
                           ncomp_median=float(M.n_components.median()), ncomp_1=int((M.n_components == 1).sum()), rho_F_r2=float(spearmanr(M.F, M.r2).correlation), rho_F_sigma=float(spearmanr(M.F, M.sigma).correlation),
                           dAIC_exp_pref=int((M.dAIC_gauss_minus_exp > 2).sum()), dAIC_gauss_pref=int((M.dAIC_gauss_minus_exp < -2).sum()), dAIC_undecided=int((M.dAIC_gauss_minus_exp.abs() <= 2).sum()),
                           beta_median=float(M.beta.median()), beta_iqr=list(map(float, q(M.beta))), fwhm_ratio_median=float((M.fwhm_modelfree / (2.3548 * M.sigma)).median()),
                           fwhm_ratio_iqr=list(map(float, q(M.fwhm_modelfree / (2.3548 * M.sigma)))), rho_sigma_lambda="see v43", lowr2_sigma_median=float(M[M.flag_r2].sigma.median()), lowr2_sigma_max=float(M[M.flag_r2].sigma.max()),
                           A_median_flagged=float(M[M.flag_r2].A.median()), A_median_unflagged=float(M[~M.flag_r2].A.median()), lmm_text=lmm)

    de = M.sigma_euclid - M.sigma; foc = M[(M.r2 >= .7) & (M.sigma < 30)]; v38i = v38.set_index("map_id").loc[M.index]; f9i = v39.loc[M.index]
    NUM["clinical_extra"] = dict(euclid_gt3=int((de.abs() > 3).sum()), r2_euclid_median=float(M.r2_euclid.median()), euclid_flagged=int((M.r2_euclid < .7).sum()), focal_unflagged_n=len(foc), exact_gt1_focal=int(((foc.sigma_exact - foc.sigma).abs() > 1).sum()),
                                 largest_euclid_change=dict(map=int(de.abs().idxmax()), heat=float(M.loc[de.abs().idxmax(), "sigma"]), euclid=float(M.loc[de.abs().idxmax(), "sigma_euclid"]), region="%s %s" % tuple(M.loc[de.abs().idxmax(), ["atrium", "region"]])),
                                 group_medians={c: [float(M[M.reference == "double"][c].median()), float(M[M.reference == "single"][c].median())] for c in ("sigma", "sigma_exact", "sigma_dijkstra", "sigma_euclid")},
                                 t_median={c: float(v38i[c].median()) for c in ("t_heat_s", "t_exact_s", "t_dijkstra_s")}, t_max={c: float(v38i[c].max()) for c in ("t_heat_s", "t_exact_s", "t_dijkstra_s")},
                                 F_flagged_other_components={int(m): [round(float(M.loc[m, "other_min_dist_mm"]), 1), round(float(M.loc[m, "other_max_score_rel"]), 2), int(M.loc[m, "n_components"])] for m in M[M.flag_F].index}, both_flags=int((M.flag_r2 & M.flag_F).sum()),
                                 rho_sigma_lambda=float(spearmanr(M.sigma, v43.loc[M.index, "lam_exp"]).correlation), beta_at_or_near_bound=int(((v43.loc[M.index, "beta"] <= 0.31) | (v43.loc[M.index, "beta"] >= 3.99)).sum()), beta_bounds=[float(v43.beta.min()), float(v43.beta.max())],
                                 far_sd_flagged=float(v43.loc[M[M.flag_r2].index, "far_sd"].median()), far_sd_unflagged=float(v43.loc[M[~M.flag_r2].index, "far_sd"].median()), far_max_rel_flagged=float(v43.loc[M[M.flag_r2].index, "far_max_rel"].median()), far_max_rel_unflagged=float(v43.loc[M[~M.flag_r2].index, "far_max_rel"].median()),
                                 resid_far_frac_flagged=float(v43.loc[M[M.flag_r2].index, "resid_far_frac"].median()), withinbin_sd_pts=float(v43.loc[M.index, "withinbin_sd_pts"].median()), withinbin_sd_pct=float(v43.loc[M.index, "withinbin_sd_pct_A"].median()), withinbin_sd_pct_iqr=list(map(float, q(v43.loc[M.index, "withinbin_sd_pct_A"]))),
                                 sigma_at_bound=int((M.sigma >= 199).sum()), n_broad_or_wider=int((M.sigma >= 30).sum()), n_diffuse=int(((M.sigma >= 50)).sum()))
    # ------------------------------------------------------------------ Table 1a / S3 validation
    va = pd.read_csv(SRC["v47av"]); vk = pd.read_csv(SRC["v46v"])
    def val_rows(noise_sel, sig_sel):
        out = []
        for s0 in sig_sel:
            for eta in noise_sel:
                g = va[(va.sigma0 == s0) & (va.noise == eta)]; k = vk[(vk.sigma0 == s0) & (vk.noise == eta)]
                b = g.sigma_argmax - s0; lo, hi = boot_ci(b.values); bk = k.sigma_hat - s0
                out.append(dict(sigma0=s0, noise=eta, n=len(g), bias=b.mean(), ci_lo=lo, ci_hi=hi, bias_median=b.median(), sd=g.sigma_argmax.std(), off=g.off_argmax.median(), p95=g.sigma_argmax.quantile(.95),
                                frac_ni=(g.sigma_argmax >= 100).mean(), bias_known=bk.mean(), sd_known=k.sigma_hat.std(), bias_known_median=bk.median()))
        return pd.DataFrame(out)
    full = val_rows((0.0, 0.05, 0.10, 0.15), (3, 5, 8, 10, 15, 20, 25, 30)); full.to_csv(os.path.join(a.out, "S3_validation_grid.csv"), index=False)
    # Table 1 (reader pass 27 Oct): primary protocol = noise-free peak below the score limit (v56: A 35, B 55), 10 % noise, detected origin
    v56t = pd.read_csv(SRC["v56all"]); rows = []; NUM["table1"] = {}
    for s0 in (2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50):
        g = v56t[(v56t.sigma0 == s0) & np.isclose(v56t.noise, 0.10)]; b = g.sigma_plateau - s0; lo, hi = boot_ci(b.values)
        p5, p95 = g.sigma_plateau.quantile(.05), g.sigma_plateau.quantile(.95); tied = (g.n_tied > 1).mean()
        rows.append([s0, fs(b.mean()), "[%s; %s]" % (fs(lo), fs(hi)), fs(b.median()), f2(g.sigma_plateau.std()), "%s--%s" % (f1(p5), f1(p95)), f1(g.off_plateau.median()), "%.1f" % (100 * tied)])
        NUM["table1"]["%g" % s0] = dict(n=len(g), bias=round(float(b.mean()), 6), ci=[round(float(lo), 6), round(float(hi), 6)], bias_median=round(float(b.median()), 6), sd=round(float(g.sigma_plateau.std()), 6), p5=round(float(p5), 6), p95=round(float(p95), 6), off=round(float(g.off_plateau.median()), 6), tied=round(float(tied), 6))
    W("T1a_validation_iid.tex", tex_table(rows, ["$\\sigma_0$ (mm)", "bias (mm)", "95\\,\\% CI", "median bias (mm)", "SD (mm)", "5th--95th pct.\\ (mm)", "origin offset (mm)", "tied (\\%)"], "rccccccc",
                                          "Synthetic validation, primary protocol (noise-free peak 90): exact-geodesic ground truth on all 56 clinical meshes, noise-free peak below the upper score limit ($A$ = 35, $B$ = 55), independent vertex noise of 10\\,\\% of the amplitude, origin detected by the pipeline, $n = 1{,}120$ fits per row (56 meshes $\\times$ 20 seeds). Bias $=$ mean of $\\hat\\sigma-\\sigma_0$ with percentile-bootstrap CI (2{,}000 resamples of the fits, conditional on the 56 meshes); SD and 5th--95th percentiles of $\\hat\\sigma$; median geodesic distance of the detected origin from the true one; fraction of fields whose maximum score is shared by several vertices. Noise of 5 and 15\\,\\% in Supplementary Table~S4b; noise-free fields, the origin fixed at the true vertex, the saturating sensitivity protocol (noise-free peak at the score limit), and widths of 1--1.5 and 60--80\\,mm in Supplementary Tables~S3, S3b and S4.",
                                          "tab:validation", star=True, colsep="3pt"))
    rows = [[int(r.sigma0), "%.0f" % (100 * r.noise), fs(r.bias), fs(r.bias_median), f2(r.sd), f1(r.p95), f1(r.off), "%.3f" % r.frac_ni, fs(r.bias_known), f2(r.sd_known)] for _, r in full.iterrows()]
    W("S3_validation_grid.tex", tex_table(rows, ["$\\sigma_0$", "$\\eta$ (\\%)", "bias", "median bias", "SD", "p95 $\\hat\\sigma$", "offset", "n.i.", "bias (known)", "SD (known)"], "rrrrrrrrrr",
                                          "Validation with the noise-free peak at the upper score limit ($A$ = 35, $B$ = 65; saturating sensitivity protocol, Section S3), full grid (exact-geodesic truth, 56 meshes $\\times$ 20 seeds): detected origin (bias, median bias, SD, 95th percentile of $\\hat\\sigma$, median origin offset, fraction $\\hat\\sigma\\ge100$ mm) and known origin.", "tab:S3", star=True))
    # S3b: known origin at 40 and 50 mm (v46 part b; the detected-origin rows for these widths are in S4)
    rows = []; NUM["known_origin_wide"] = {}
    for s0 in (40, 50):
        for eta in (0.0, 0.05, 0.10, 0.15):
            k = vk[(vk.sigma0 == s0) & (vk.noise == eta)]; bk = k.sigma_hat - s0
            rows.append([s0, "%.0f" % (100 * eta), len(k), fs(bk.mean()), fs(bk.median()), f2(k.sigma_hat.std()), f1(k.sigma_hat.quantile(.95))])
            NUM["known_origin_wide"]["%d/%g" % (s0, eta)] = dict(n=len(k), bias=float(bk.mean()), bias_median=float(bk.median()), sd=float(k.sigma_hat.std()))
    W("S3b_known_origin_wide.tex", tex_table(rows, ["$\\sigma_0$ (mm)", "$\\eta$ (\\%)", "$n$", "bias", "median bias", "SD", "p95 $\\hat\\sigma$"], "rrrrrrr",
                                             "Known-origin mode (origin fixed at the generating vertex) at 40 and 50 mm, same fields and protocol as Table S3 (56 meshes $\\times$ 20 seeds); detected-origin results for these widths in Table S4.", "tab:S3b"))
    # unrounded values: numbers.md must match the tables (no double rounding)
    NUM["validation"] = {("%g/%g" % (r.sigma0, r.noise)): dict(bias=round(r.bias, 6), ci=[round(r.ci_lo, 6), round(r.ci_hi, 6)], sd=round(r.sd, 6), off=round(r.off, 6), bias_known=round(r.bias_known, 6), sd_known=round(r.sd_known, 6)) for _, r in full.iterrows()}

    # ------------------------------------------------------------------ Table 1b / S9 correlated
    vb = pd.read_csv(SRC["v47b"])
    def corr_rows(kind, amp):
        out = []; NUM.setdefault("corr", {})
        for s0 in (5.0, 10.0, 15.0, "all"):
            g = vb[(vb.kind == kind) & (vb.amp == amp)] if s0 == "all" else vb[(vb.kind == kind) & (vb.amp == amp) & (vb.sigma0 == s0)]
            h = g[g.offset_mm <= 10]; b = g.sigma_hat - g.sigma0; bh = h.sigma_hat - h.sigma0
            out.append(["all" if s0 == "all" else int(s0), "%.3f" % g.r2.median(), "%.0f" % (100 * (g.r2 < .7).mean()), "%.0f" % (100 * (g.offset_mm > 10).mean()), med_iqr(b, 2), med_iqr(bh, 2), "%.1f (%.0f\\,\\%%)" % (g.wsd_pct_A.median() * 0.35, g.wsd_pct_A.median())])
            NUM["corr"]["%s_%g_%s" % (kind, amp, s0)] = dict(r2=float(g.r2.median()), flag=float((g.r2 < .7).mean()), off10=float((g.offset_mm > 10).mean()), bias_med=float(b.median()), bias_iqr=list(map(float, q(b))), bias_sd=float(b.std()), bias_h_med=float(bh.median()), bias_h_iqr=list(map(float, q(bh))), bias_h_sd=float(bh.std()), wsd_pts=float(g.wsd_pct_A.median() * 0.35), wsd_pct=float(g.wsd_pct_A.median()), F_pct_flag=float((g.F < .7).mean()))  # v47b 'F' is the percentile index
        return out
    W("T1b_validation_correlated.tex", tex_table(corr_rows("corr10", 0.25), ["$\\sigma_0$", "median $R^2$", "$R^2<0.7$ (\\%)", "origin $>10$ mm (\\%)", "bias, all fits", "bias, origin $\\le10$ mm", "within-bin SD"], "rcccccc",
                                                 "Spatially correlated deviations (SD 0.25\\,$A$, smoothing scale 10 mm; $A$ = 35 score points) on the 56 clinical meshes, 10 seeds per mesh and width ($\\sigma_0$ and bias in mm): median $R^2$, fraction of fits flagged, fraction of fields in which the detected origin lies more than 10 mm from the true one, median [IQR] of $\\hat\\sigma-\\sigma_0$ for all fits and for fits with the origin within 10 mm, and within-bin SD of the field in the first 30 mm in score points (\\%% of $A$ in parentheses; clinical maps: %.1f points, %.0f\\,\\%% of $A$)." % (NUM["clinical_extra"]["withinbin_sd_pts"], NUM["clinical_extra"]["withinbin_sd_pct"]), "tab:validation_corr", star=True, size="footnotesize"))
    for kind, amp in (("corr10", 0.1), ("corr10", 0.4), ("corr5", 0.25), ("bumps", 0.25), ("bumps", 0.5)): corr_rows(kind, amp)
    rows = []
    for (kind, amp, s0), g in vb.groupby(["kind", "amp", "sigma0"]):
        h = g[g.offset_mm <= 10]
        rows.append([kind, "%.2f" % amp, int(s0), "%.1f (%.0f)" % (g.wsd_pct_A.median() * 0.35, g.wsd_pct_A.median()), "%.3f" % g.r2.median(), "%.0f" % (100 * (g.r2 < .7).mean()), "%.2f" % g.F.median(), "%.0f" % (100 * (g.F < .7).mean()),
                     med_iqr(g.sigma_hat - s0, 2), "%.0f" % (100 * (g.offset_mm > 10).mean()), med_iqr(h.sigma_hat - s0, 2) if len(h) else "--"])
    W("S9_structured_meshes.tex", tex_table(rows, ["deviation", "amp.", "$\\sigma_0$", "SD (\\%$A$)", "$R^2$", "flag \\%", "$F_{\\rm pct}$", "$F_{\\rm pct}<0.7$ \\%", "bias all", "orig.$>$10 \\%", "bias orig.$\\le$10"], "llrrrrrrrrr",
                                             "Structured (bumps: 30 Gaussian bumps of 8 mm, plus 10\\,\\% independent noise) and spatially correlated deviations (corr5/corr10: smoothing scale $\\ell$ = 5/10 mm) on the 56 clinical meshes; amplitude SD in units of $A$; 10 seeds per mesh and condition; within-bin SD in score points (\\% of $A$); flag: fraction of fits with $R^2<0.7$; $F_{\\rm pct}$: percentile index ($F$ on these meshes in Table~\\ref{tab:S14b}); orig.$>$10: fraction of fields with the detected origin more than 10 mm from the true one; bias as median [IQR] (mm), for all fits and for fits with the origin within 10 mm.", "tab:S9", star=True, size="footnotesize", colsep="3pt"))

    # ------------------------------------------------------------------ S4 range, S5 origin rule, S6 arm3, S7 aniso, S10, S11, S13-S18
    v42 = pd.read_csv(SRC["v42"]); rows = [[("%g" % r.sigma0), "%.0f" % (100 * r.noise), fs(r.bias_mean), fs(r.bias_median), f2(r.sd), f1(r.p5), f1(r.p95), "%.3f" % r.frac_at_bound, "%.3f" % r.frac_gt_100, f1(r.origin_offset_median)] for _, r in v42[v42.sigma0.isin([1, 1.5, 2, 2.5, 40, 50, 60, 80])].iterrows()]
    W("S4_range.tex", tex_table(rows, ["$\\sigma_0$", "$\\eta$ (\\%)", "bias", "median bias", "SD", "p5", "p95", "at bound", "$>100$ mm", "offset"], "rrrrrrrrrr", "Widths outside 3--30 mm in the saturating sensitivity protocol (noise-free peak at the upper score limit; as Table S3; exact-geodesic truth, detected origin, 56 meshes $\\times$ 20 seeds). at bound: fraction of fits with $\\hat\\sigma$ at the upper parameter bound (200 mm); $>100$ mm: fraction with $\\hat\\sigma>100$ mm (includes the former).", "tab:S4", star=True))
    vac = pd.read_csv(SRC["v47ac"]); ok = vac.dropna(subset=["offset_argmax_mm"])
    s5 = va.groupby(["sigma0", "noise"]).apply(lambda g: pd.Series(dict(bias_a=(g.sigma_argmax - g.sigma0).mean(), sd_a=g.sigma_argmax.std(), off_a=g.off_argmax.median(), bias_s=(g.sigma_smooth - g.sigma0).mean(), sd_s=g.sigma_smooth.std(), off_s=g.off_smooth.median()))).reset_index()
    rows = [[int(r.sigma0), "%.0f" % (100 * r.noise), fs(r.bias_a), f2(r.sd_a), f1(r.off_a), fs(r.bias_s), f2(r.sd_s), f1(r.off_s)] for _, r in s5[s5.noise.isin([0.10, 0.15])].iterrows()]
    notes5 = "Clinical maps (46 with a marked pacing site): distance of the detected origin to the marked site, median %.1f mm (argmax) vs %.1f mm (3-mm neighbourhood maximum); beyond 10 mm in %d vs %d of 46; narrow maps ($\\sigma<10$ mm): %d vs %d of %d." % (
        ok.offset_argmax_mm.median(), ok.offset_smooth_mm.median(), (ok.offset_argmax_mm > 10).sum(), (ok.offset_smooth_mm > 10).sum(), (ok[ok.sigma_argmax < 10].offset_argmax_mm > 10).sum(), (ok[ok.sigma_smooth < 10].offset_smooth_mm > 10).sum(), (ok.sigma_argmax < 10).sum())
    W("S5_origin_rule.tex", tex_table(rows, ["$\\sigma_0$", "$\\eta$ (\\%)", "bias", "SD", "offset", "bias", "SD", "offset"], "rrrrrrrr", "Origin rule: maximum-score vertex with the plateau-centre tie rule (left, as in the pipeline) versus vertex with the highest 3-mm-neighbourhood mean (right); same synthetic fields (exact truth, 56 meshes $\\times$ 20 seeds).", "tab:S5", notes=notes5))
    NUM["origin_rule"] = notes5
    a3 = pd.read_csv(SRC["arm3"]); rows = [[r.profile.replace("_", " "), f1(r.fwhm_true), "%.0f" % (100 * r.noise), f1(r.fwhm_hat), "%+.0f" % (100 * r.fwhm_rel_err_mean), "%.3f" % r.r2_gauss, "%.3f" % r.r2_exp, "%.0f" % (100 * r.gauss_preferred_frac)] for _, r in a3.iterrows()]
    W("S6_nongaussian.tex", tex_table(rows, ["true profile", "FWHM (mm)", "$\\eta$ (\\%)", "$2.355\\hat\\sigma$ (mm)", "error (\\%)", "$R^2_{\\rm Gauss}$", "$R^2_{\\rm exp}$", "Gauss preferred (\\%)"], "lrrrrrrr",
                                       "Non-Gaussian truth (exact geodesics of the 56 meshes from the centroid-nearest vertex, same noise, clip and mask protocol and the same pipeline, 560 fits per row): recovery of the full width at half maximum by the Gaussian width and model preference by AIC.", "tab:S6", star=True))
    v44 = pd.read_csv(SRC["v44"]); A_ = v44[v44.part == "A"]
    rows = [[("%g" % gm), ("%g" % ratio), "%.0f" % (100 * eta), f2(g.sigma_hat.median() / gm), "%.3f" % g.r2.median(), f1(g.origin_offset.median())] for (gm, ratio, eta), g in A_.groupby(["gm", "ratio", "eta"])]
    W("S7_anisotropy_sphere.tex", tex_table(rows, ["geom. mean (mm)", "axis ratio", "$\\eta$ (\\%)", "$\\hat\\sigma/\\sqrt{\\sigma_a\\sigma_b}$", "$R^2$", "offset (mm)"], "rrrrrr", "Isotropic fit to elliptical Gaussian fields on a sphere (analytic geodesics; median results over 20 trials per condition).", "tab:S7"))
    v47c = pd.read_csv(SRC["v47c"]); rows = [[("%g" % r.gm), ("%g" % r.ratio), r.kind, "%.2f" % r.amp, "%.0f" % r.wsd, "%.3f" % r.r2, f2(r.rel), f1(r.off)] for _, r in v47c.iterrows()]
    W("S10_scatter_sources.tex", tex_table(rows, ["geom. mean", "axis ratio", "noise", "amp. ($A$)", "within-bin SD (\\% $A$)", "$R^2$", "$\\hat\\sigma$/gm", "offset"], "rrlrrrrr", "Sources of within-bin scatter (sphere, 20 trials): anisotropy and correlated noise.", "tab:S10"))
    v37 = pd.read_csv(SRC["v37"]); rows = []
    for kind in ("focal", "flat", "ring", "two-peak"):
        for eta, g in v37[v37.kind == kind].groupby("noise"):
            mis = [100 * ((g.r2 < T).mean() if kind == "focal" else (g.r2 >= T).mean()) for T in (0.5, 0.6, 0.7, 0.8, 0.9)]
            rows.append([kind, "%.0f" % (100 * eta), len(g), "%.3f" % g.r2.median()] + ["%.0f" % m for m in mis])
    W("S11a_r2_threshold.tex", tex_table(rows, ["field", "$\\eta$ (\\%)", "$n$", "median $R^2$", "0.5", "0.6", "0.7", "0.8", "0.9"], "lrrrrrrrr", "Misclassification of the $R^2$ flag (\\%) by threshold: Gaussian fields rejected / flat, ring and two-peak fields accepted (sphere, 20 trials).", "tab:S11a"))
    v50s = pd.read_csv(SRC["v50s"]); rows = []; NUM["F_synthetic"] = {}
    def frate(g, op=False):  # round 6: the single denominator rule for tables AND numbers.md: fields with F available
        g = g[g.F.notna()]
        return float("nan") if len(g) == 0 else float(((g.F < .7) & (g.r2 >= .7)).mean() if op else (g.F < .7).mean())
    def fop(g):  # operational flag (F < 0.70 and R2 >= 0.7), raw F < 0.70 in parentheses where different; formats frate() (round 7)
        if g.F.notna().sum() == 0: return "--"
        op, raw = 100 * frate(g, op=True), 100 * frate(g)
        return "%.0f" % op if round(op) == round(raw) else "%.0f (%.0f)" % (op, raw)
    for (D, ratio), g in v50s[v50s.kind == "two-peak"].groupby(["D", "ratio"]):
        r_ = [int(D), "%.1f" % ratio, len(g[g.amp == 0.10])] + [fop(g[g.amp == e]) for e in (0.05, 0.10, 0.25)] + ["%.0f" % (100 * (g[g.amp == e].F_pct < .7).mean()) for e in (0.05, 0.10, 0.25)] + ["%.2f" % g[g.amp == 0.10].F.median()]
        rows.append(r_); NUM["F_synthetic"]["twopeak_D%d_r%.1f" % (D, ratio)] = dict(F_flag_raw=[frate(g[g.amp == e]) for e in (0.05, 0.10, 0.25)], F_flag_op=[frate(g[g.amp == e], op=True) for e in (0.05, 0.10, 0.25)], F_nan=[int(g[g.amp == e].F.isna().sum()) for e in (0.05, 0.10, 0.25)], F_median_10=float(g[g.amp == 0.10].F.median()))
    rows_c = []
    for kind in ("focal", "flat", "ring"):
        for (nk, amp), g in v50s[v50s.kind == kind].groupby(["noise", "amp"]):
            lab = {"focal": "single peak", "flat": "flat", "ring": "ring"}[kind] + (" (corr.\\ noise)" if nk != "iid" else "")
            rows_c.append([lab, "%.2f" % amp, len(g), "%.0f" % (100 * g.F.isna().mean()), fop(g), "%.0f" % (100 * (g.F_pct < .7).mean()), "%.2f" % g.F.median()]); NUM["F_synthetic"]["%s_%s_%.2f" % (kind, nk, amp)] = dict(F_flag_raw=frate(g), F_flag_op=frate(g, op=True), F_pct_flag=float((g.F_pct < .7).mean()), F_median=float(g.F.median()), F_nan=float(g.F.isna().mean()), F_available=int(g.F.notna().sum()), n=len(g))
    W("S11b_focality_detection.tex", tex_table(rows, ["separation (mm)", "height ratio / noise", "$n$", "5\\,\\%", "10\\,\\%", "25\\,\\%", "5\\,\\%", "10\\,\\%", "25\\,\\%", "$F$ median (10\\,\\%)"], "rrrrrrrrrr",
                                                "Focality index on the sphere, two-peak fields ($n$ fields per noise level; two lobes of $\\sigma$ = 7 mm combined by their pointwise maximum, not their sum; should be flagged when the second peak is comparably high and separated). Columns 4--6: fraction of fields flagged by $F$ (half-maximum set, component of the origin) as in the pipeline, i.e.\\ $F<0.70$ with $R^2\\ge0.7$, at 5, 10 and 25\\,\\% vertex noise (raw fraction with $F<0.70$ in parentheses where different; denominators: fields with $F$ available; flat fields are caught by the $R^2$ flag instead); columns 7--9: the same for the percentile index $F_{\\rm pct}$ (95th-percentile set, largest component).", "tab:S11b", star=True, group=[("", 1, 3), ("$F$ flag (\\%)", 4, 6), ("$F_{\\rm pct}<0.70$ (\\%)", 7, 9), ("", 10, 10)]))
    W("S11c_focality_other.tex", tex_table(rows_c, ["field", "noise ($A$)", "$n$", "$F$ n.a. (\\%)", "$F$ flag (\\%)", "$F_{\\rm pct}<0.70$ (\\%)", "$F$ median"], "lrrrrrr",
                                             "Focality index on the sphere, single-peak, flat and ring fields (single peak: $\\sigma_0$ = 5, 10 and 15 mm with 20 trials each, 60 fields per row; flat and ring: 20 fields per row; noise as a fraction of $A$, independent unless marked correlated). $F$ n.a.: fields in which $F$ is unavailable (non-positive fitted amplitude, or fewer than three vertices in the half-maximum set); the $F$ rates are fractions of the fields with $F$ available, i.e.\\ such fields are excluded from their denominator. $F$ flag: $F<0.70$ with $R^2\\ge0.7$, as in the pipeline (raw fraction with $F<0.70$ in parentheses where different; flat fields are caught by the $R^2$ flag instead); $F_{\\rm pct}$: percentile index (95th-percentile set, largest component).", "tab:S11c"))
    v50m = pd.read_csv(SRC["v50m"]); rows = []; NUM["F_meshes"] = {}
    for (kind, amp), g in v50m.groupby(["kind", "amp"]):
        rows.append([{"iid": "independent noise", "bumps": "bumps $+$ 10\\,\\% noise", "corr10": "correlated noise (10 mm)"}[kind], "%.2f" % amp] + [fop(g[g.sigma0 == s0]) for s0 in (5, 10, 15)] + ["%.0f" % (100 * (g[g.sigma0 == s0].F_pct < .7).mean()) for s0 in (5, 10, 15)] + ["%.0f" % (100 * (g.offset_mm > 10).mean())])
        NUM["F_meshes"]["%s_%.2f" % (kind, amp)] = dict(F_flag_raw=[frate(g[g.sigma0 == s0]) for s0 in (5, 10, 15)], F_flag_op=[frate(g[g.sigma0 == s0], op=True) for s0 in (5, 10, 15)], F_available=[int(g[g.sigma0 == s0].F.notna().sum()) for s0 in (5, 10, 15)], n=[int((g.sigma0 == s0).sum()) for s0 in (5, 10, 15)], F_pct_flag=[float((g[g.sigma0 == s0].F_pct < .7).mean()) for s0 in (5, 10, 15)])
    KLAB = {"iid": "independent noise", "bumps": "bumps", "corr10": "correlated noise"}; na = v50m[v50m.F.isna()]
    NUM["F_meshes_unavailable"] = [dict(map_id=int(r.map_id), kind=r.kind, amp=float(r.amp), sigma0=float(r.sigma0), trial=int(r.trial), A=float(r.A), r2=float(r.r2)) for _, r in na.iterrows()]
    why = lambda r: "non-positive fitted amplitude" if r.A <= 0 else ("fitted amplitude far above the score range, half-maximum threshold %s, empty set" % "{:,.0f}".format(r.thr).replace(",", "{,}") if r.thr > 100 else "fewer than three vertices in the half-maximum set")
    na_txt = "; ".join("%s %.2f\\,$A$, $\\sigma_0$ = %g mm: %s, $R^2$ = %.2f" % (KLAB[r.kind], r.amp, r.sigma0, why(r), r.r2) for _, r in na.iterrows())
    W("S14b_focality_meshes.tex", tex_table(rows, ["deviation", "amp. ($A$)", "5", "10", "15", "5", "10", "15", "origin $>10$ mm (\\%)"], "lrrrrrrrr",
                                             "Focality index on the 56 clinical meshes with synthetic single-peak fields ($\\sigma_0$ = 5, 10, 15 mm; exact-geodesic truth; 3 seeds per mesh and condition, i.e.\\ 168 fields per width-specific cell; the last column is pooled over the three widths, %d fields per row): fraction of fields flagged by $F$ as in the pipeline ($F<0.70$ with $R^2\\ge0.7$; raw fraction with $F<0.70$ in parentheses where different), fraction with $F_{\\rm pct}<0.70$ for the percentile index, and fraction of fields with the detected origin more than 10 mm from the true one.  $F$ rates are fractions of the fields with $F$ available; $F$ was unavailable in %d of the %s fields (%s), excluded from the denominator of %s." % (int(v50m.groupby(["kind", "amp"]).size().iloc[0]), len(na), "{:,}".format(len(v50m)).replace(",", "{,}"), na_txt, "its cell" if len(na) == 1 else "their cells"), "tab:S14b", group=[("", 1, 2), ("$F$ flag (\\%) at $\\sigma_0$ (mm)", 3, 5), ("$F_{\\rm pct}<0.70$ (\\%)", 6, 8), ("", 9, 9)]))
    c = v38.set_index("map_id").loc[M.index]
    rows = []
    for name in ("heat", "exact", "dijkstra", "euclid"):
        s = c["sigma_" + name]; d = s - c.sigma_heat
        rows.append([name, med_iqr(s, 2), "%.3f" % c["r2_" + name].median(), int((c["r2_" + name] < .7).sum()), int((s >= 30).sum()), "--" if name == "heat" else med_iqr(d, 2), "--" if name == "heat" else int((d.abs() > 1).sum()), "--" if name == "heat" else f2(spearmanr(s, c.sigma_heat).correlation),
                     "--" if name == "exact" else "%+.1f" % (100 * c["relerr_%s_30" % name].median()), "--" if name == "exact" else "%.1f" % (100 * c["p95_%s_30" % name].median())])
    W("S13_distance_definitions.tex", tex_table(rows, ["distance", "$\\sigma$ median [IQR]", "$R^2$", "$R^2<0.7$", "$\\sigma\\ge30$", "$\\Delta\\sigma$ vs Heat", "$|\\Delta\\sigma|>1$ mm", "$\\rho$", "rel. error $\\le30$ mm (\\%)", "p95 (\\%)"], "lcccccccrr",
                                                "Width of the 55 clinical maps with four distance definitions from the same origin; distance error relative to the exact geodesic within 30 mm (map-wise medians).", "tab:S13", star=True, size="footnotesize", colsep="2.5pt"))
    NUM["distance"] = {n: dict(dsig_median=float((c["sigma_" + n] - c.sigma_heat).median()), dsig_iqr=list(map(float, q(c["sigma_" + n] - c.sigma_heat))), gt1=int(((c["sigma_" + n] - c.sigma_heat).abs() > 1).sum()), rho=float(spearmanr(c["sigma_" + n], c.sigma_heat).correlation)) for n in ("exact", "dijkstra", "euclid")}
    NUM["distance"]["err30"] = {n: dict(median=float(c["relerr_%s_30" % n].median()), iqr=list(map(float, q(c["relerr_%s_30" % n]))), p95=float(c["p95_%s_30" % n].median())) for n in ("heat", "dijkstra", "euclid")}
    NUM["distance"]["errall"] = {n: dict(median=float(c["relerr_%s_all" % n].median()), p95=float(c["p95_%s_all" % n].median())) for n in ("heat", "dijkstra", "euclid")}
    f9 = v39.loc[M.index]; E = M[M.r2 >= 0.7]
    rows = [["$F$, $R^2\\ge0.7$ ($n$ = %d)" % len(E), med_iqr(E.F, 2), f2(E.F.min())] + [int((E.F < T).sum()) for T in (0.5, 0.6, 0.7, 0.8, 0.9)] + [int(E.n_components.median()), int((E.n_components == 1).sum())],
            ["$F$, all 55 (descriptive)", med_iqr(M.F, 2), f2(M.F.min())] + [int((M.F < T).sum()) for T in (0.5, 0.6, 0.7, 0.8, 0.9)] + [int(M.n_components.median()), int((M.n_components == 1).sum())]]
    for pct in (90, 95, 98, 99):
        f = f9["F_edge_p%d" % pct]; rows.append(["$F_{\\rm pct}$, %dth pct.%s" % (pct, " (as in $F_{\\rm pct}$)" if pct == 95 else ""), med_iqr(f, 2), f2(f.min())] + [int((f < T).sum()) for T in (0.5, 0.6, 0.7, 0.8, 0.9)] + [int(f9["ncomp_edge_p%d" % pct].median()), int((f9["ncomp_edge_p%d" % pct] == 1).sum())])
    v50c = pd.read_csv(SRC["v50c"]).set_index("map_id").loc[M.index]
    W("S14_focality_clinical.tex", tex_table(rows, ["index", "median [IQR]", "min", "$<0.5$", "$<0.6$", "$<0.7$", "$<0.8$", "$<0.9$", "comp. (median)", "one comp."], "lccrrrrrrr",
                                             "Focality index of the clinical maps: distribution, number of maps below each threshold, median number of connected components of the high-score set and number of maps with a single component. $F$: set of vertices at or above the half maximum of the fitted profile ($\\hat B+\\hat A/2$), share of the component containing the origin; $F$ is interpreted only when $R^2\\ge0.7$ (first row); in these maps Spearman $\\rho(F,R^2)$ = %.2f, $\\rho(F,\\sigma)$ = %.2f. Percentile index $F_{\\rm pct}$: largest component of the percentile set; with the 95th percentile the vertex of maximum score lies outside the largest component in %d of 55 maps (median share of its own component %.2f), because the percentile set of a narrow peak extends to %.0f\\,\\%% of the peak height (median; IQR %.0f--%.0f\\,\\%%) and is then dominated by broad regions elsewhere." % (
                                                 spearmanr(E.F, E.r2).correlation, spearmanr(E.F, E.sigma).correlation, int((pd.read_csv(os.path.join(MP, "revision_2026-10/figs/_cache/origin_component_check.csv")).set_index("map_id").loc[M.index].origin_in_main == False).sum()), float(pd.read_csv(os.path.join(MP, "revision_2026-10/figs/_cache/origin_component_check.csv")).set_index("map_id").loc[M.index].share_o.median()),
                                                 100 * float(pd.read_csv(os.path.join(MP, "revision_2026-10/figs/_cache/origin_component_check.csv")).set_index("map_id").loc[M.index].thr95_rel.median()), *[100 * float(x) for x in q(pd.read_csv(os.path.join(MP, "revision_2026-10/figs/_cache/origin_component_check.csv")).set_index("map_id").loc[M.index].thr95_rel)]), "tab:S14", star=True, size="footnotesize"))
    occ = pd.read_csv(os.path.join(MP, "revision_2026-10/figs/_cache/origin_component_check.csv")).set_index("map_id").loc[M.index]
    NUM["F_pct_origin"] = dict(origin_not_in_largest=int((occ.origin_in_main == False).sum()), share_median=float(occ.share_o.median()), thr95_rel_median=float(occ.thr95_rel.median()), thr95_rel_iqr=list(map(float, q(occ.thr95_rel))), largest_elsewhere_dist_median=float(occ[occ.origin_in_main == False].main_min_dist.median()), largest_elsewhere_height_median=float(occ[occ.origin_in_main == False].main_max_score_rel.median()))
    W("S15_mixed_model.txt", open(P("v58_model_diagnostics_2026-10-26", "v58_summary.txt")).read() + "\n\n--- v40 (model block SUPERSEDED by v58; patient medians and bootstrap current) ---\n" + lmm)
    pmb = {}
    for name, body in re.findall(r"\[([^\]\n]*)\] patient medians(.*?)(?=\n\n|\Z)", lmm, re.S):
        pm = re.search(r"single n=(\d+) median ([\d.]+) \[[^\]]*\]; double n=(\d+) median ([\d.]+) \[[^\]]*\]; Mann-Whitney p=([\d.]+)", body).groups()
        bt = re.search(r"ratio of map-level medians single/double = ([\d.]+); patient-cluster bootstrap 95% interval \[([\d.]+); ([\d.]+)\]", body).groups(); pmb[name] = (pm, bt)
    keymap = [("all55_raw", "all 55, raw sigma", "all 55 maps"), ("all55_clip30", "all 55, sigma censored at 30 mm", "all 55, $\\sigma$ clipped at 30 mm"), ("r2ge07_raw", "R2>=0.7 (n=42), raw sigma", "42 maps with $R^2 \\ge 0.7$")]
    pf = lambda x: "$p<0.001$" if x < 0.001 else "$p$ = %.3f" % x
    ci = lambda r: "%.2f (%.2f--%.2f), %s" % (r.ratio, r.t_lo, r.t_hi, pf(r.p_t))
    cols = []
    for an, v40name, lab in keymap:
        i_ = I58[an]; pm, bt = pmb[v40name]; tr = lambda term: T58[(T58.analysis == an) & (T58.term == term)].iloc[0]
        cols.append([lab + " (%d patients: %d single, %d double)" % (i_["n_patients"], i_["n_single"], i_["n_double"]), ci(tr("reference[T.single]")), ci(tr("region[T.LA anterior]")), ci(tr("region[T.RA posterior]")), ci(tr("region[T.RA lateral]")),
                     "%.2f" % i_["icc"], "%.2f, %s" % (i_["lrt_chi2"], pf(i_["lrt_p"])), "%.2f (%.2f--%.2f), %s" % (i_["cr2"]["ratio"], i_["cr2"]["lo"], i_["cr2"]["hi"], pf(i_["cr2"]["p"])), pf(i_["wild_p"]),
                     "%.2f (%.2f--%.2f), %s" % (i_["patient_adj"]["ratio"], i_["patient_adj"]["lo"], i_["patient_adj"]["hi"], pf(i_["patient_adj"]["p"])), "%s vs %s, $p$ = %s" % (pm[1], pm[3], pm[4]), "%s (%s--%s)" % bt])
    T = list(zip(*cols)); labels = ["single- vs double-chamber reference", "LA anterior vs LA posterior", "RA posterior vs LA posterior", "RA lateral vs LA posterior", "patient ICC", "likelihood-ratio test, reference ($\\chi^2$, $p$; ML)", "OLS, CR2 patient-cluster SE, Satterthwaite df", "wild cluster bootstrap $p$ (Webb, 9{,}999)", "patient level, two-stage: $\\exp$(difference of mean region residuals), Welch", "patient-level medians single vs double (mm), Mann--Whitney $p$", "ratio of map-level medians (patient-cluster bootstrap 95\\,\\% interval)"]
    rows_t = [[lab] + list(T[k + 1]) for k, lab in enumerate(labels)]
    W("S15_mixed_model.tex", tex_table(rows_t, ["term"] + list(T[0]), "p{4.6cm}p{3.4cm}p{3.4cm}p{3.4cm}",
                                        "Reference configuration and region: linear mixed model of $\\log\\sigma$ with reference and region as fixed effects and a patient random intercept, REML at the profiled optimum of the variance ratio for every column (closed-form GLS; the default optimiser of statsmodels did not converge for all 55 maps and reported convergence with warnings and a large score for the clipped analysis), ratios $\\exp(\\mathrm{coef})$ with 95\\,\\% CI and $p$ from the model-based covariance (inverse GLS information at the profiled variance ratio) with an approximate $t$ reference on (patients $-$ 2) degrees of freedom, applied to all terms (no Kenward--Roger or Satterthwaite adjustment), relative to the double-chamber reference and the LA posterior wall; intraclass correlation of the patient; likelihood-ratio test of the reference term (ML fits, both converged); small-sample alternatives without the random effect (ordinary least squares with CR2 patient-cluster-robust SE; wild cluster bootstrap); two-stage patient-level sensitivity analysis (residuals of $\\log\\sigma$ on region, averaged per patient, Welch test; $\\exp$ of the difference of the group means; the first-stage uncertainty is not propagated); ratio of the map-level group medians with the patient-cluster bootstrap interval (2{,}000 resamples).", "tab:S15", star=True, size="footnotesize"))
    v41 = pd.read_csv(SRC["v41"]); fitl = lambda col: np.polyfit(np.log(v41.n_vertices), np.log(v41[col]), 1)[0]
    rows = [[int(r.map_id), int(r.n_vertices), f2(r.mean_edge_mm), "%.3f" % r.t_factor_s, "%.4f" % r.t_query_s, f1(r.nnz_A_per_row), f1(r.nnz_L_per_row)] for _, r in v41.iterrows()]
    W("S16_runtime_long.tex", long_table(rows, ["map", "$N$", "edge (mm)", "construction (s)", "query (s)", "nnz($A$)/$N$", "nnz($L$)/$N$"], "rrrrrrr", "Heat Method solver construction and query time and fill-in per mesh (single core); log-log exponents: construction $N^{%.2f}$, query $N^{%.2f}$, nnz($L$) $N^{%.2f}$." % (fitl("t_factor_s"), fitl("t_query_s"), fitl("nnz_L")), "tab:S16"))
    W("S16_runtime.tex", tex_table(rows, ["map", "$N$", "edge (mm)", "construction (s)", "query (s)", "nnz($A$)/$N$", "nnz($L$)/$N$"], "rrrrrrr", "Heat Method solver construction and query time and fill-in per mesh (single core); log-log exponents: construction $N^{%.2f}$, query $N^{%.2f}$, nnz($L$) $N^{%.2f}$." % (fitl("t_factor_s"), fitl("t_query_s"), fitl("nnz_L")), "tab:S16"))
    NUM["runtime"] = dict(exp_factor=float(fitl("t_factor_s")), exp_query=float(fitl("t_query_s")), exp_nnz=float(fitl("nnz_L")), nnzA=float(v41.nnz_A_per_row.median()), nnzL=float(v41.nnz_L_per_row.median()), nnzL_range=[float(v41.nnz_L_per_row.min()), float(v41.nnz_L_per_row.max())], t_max=float(v41.t_factor_s.max()), q_max=float(v41.t_query_s.max()))
    v45 = pd.read_csv(SRC["v45"]); rows = []
    for tag, lab in (("1mm", "$\\Delta d = 1$ mm"), ("4mm", "$\\Delta d = 4$ mm"), ("1p5h", "$\\Delta d = 1.5\\,\\bar h$"), ("smooth10", "Taubin 10 it."), ("smooth30", "Taubin 30 it.")):
        d = v45["sigma_" + tag] - v45.sigma_2mm; rows.append([lab, med_iqr(d, 2), int((d.abs() > 1).sum()), f2(spearmanr(v45["sigma_" + tag], v45.sigma_2mm).correlation), "%.3f" % v45["r2_" + tag].median()])
    W("S17_binwidth_smoothing.tex", tex_table(rows, ["variant", "$\\Delta\\sigma$ vs 2 mm, median [IQR]", "$|\\Delta\\sigma|>1$ mm", "$\\rho$", "median $R^2$"], "lccrr", "Bin width and mesh smoothing sensitivities on the 55 clinical maps (reference: $\\Delta d = 2$ mm, median $R^2$ %.3f)." % v45.r2_2mm.median(), "tab:S17"))
    mk = M[M.marker]; rows = []
    for lab, m in (("all", np.ones(len(mk), bool)), ("$\\sigma<10$ mm", mk.sigma < 10), ("$10\\le\\sigma<20$ mm", (mk.sigma >= 10) & (mk.sigma < 20)), ("$\\sigma\\ge20$ mm", mk.sigma >= 20), ("$R^2\\ge0.7$", ~mk.flag_r2), ("$R^2<0.7$", mk.flag_r2), ("double-chamber", mk.reference == "double"), ("single-chamber", mk.reference == "single")):
        g = mk[m]; rows.append([lab, len(g), med_iqr(g.offset_marker_mm), int((g.offset_marker_mm <= 5).sum()), int((g.offset_marker_mm > 10).sum()), int((g.offset_marker_mm > 20).sum())])
    W("S18_marker_check.tex", tex_table(rows, ["subset", "$n$", "offset, median [IQR] (mm)", "$\\le5$ mm", "$>10$ mm", "$>20$ mm"], "lrcrrr", "Geodesic distance from the vertex of maximum score to the pacing site marked during acquisition (46 maps). Spearman $\\rho$(offset, $\\sigma$) = %.2f (patient-cluster bootstrap 95\\,\\%% interval %.2f to %.2f, 2{,}000 resamples of patients); $\\rho$(offset, $R^2$) = %.2f. Offsets are measured to the mesh vertex nearest to the marked position." % (spearmanr(mk.offset_marker_mm, mk.sigma).correlation, *cluster_boot_rho(mk, "offset_marker_mm", "sigma"), spearmanr(mk.offset_marker_mm, mk.r2).correlation), "tab:S18"))
    # ------------------------------------------------------------------ S4b peak below the score limit (v56, all 56 meshes)
    v56 = pd.read_csv(SRC["v56all"]); rows = []; NUM["v56"] = {}
    for (s0, eta), g in v56.groupby(["sigma0", "noise"]):
        b = g.sigma_plateau - s0
        rows.append([("%g" % s0), "%.0f" % (100 * eta), len(g), fs(b.mean()), fs(b.median()), f2(g.sigma_plateau.std()), f1(g.sigma_plateau.quantile(.05)), f1(g.sigma_plateau.quantile(.95)), "%.1f" % (100 * (g.sigma_plateau >= 100).mean()), f1(g.off_plateau.median()), "%.0f" % (100 * (g.n_tied > 1).mean())])
        NUM["v56"]["%g_%.2f" % (s0, eta)] = dict(n=len(g), bias=float(b.mean()), bias_median=float(b.median()), sd=float(g.sigma_plateau.std()), off=float(g.off_plateau.median()), frac_ni=float((g.sigma_plateau >= 100).mean()), tied=float((g.n_tied > 1).mean()))
    W("S4b_peak_below_limit.tex", tex_table(rows, ["$\\sigma_0$", "$\\eta$ (\\%)", "$n$", "bias", "median bias", "SD", "p5", "p95", "$>100$ mm (\\%)", "offset", "tied (\\%)"], "rrrrrrrrrrr",
                                            "Primary protocol of Table 1, full grid: noise-free peak below the upper score limit ($A$ = 35, $B$ = 55, peak 90), exact-geodesic truth on the 56 meshes, clamping to [50, 100], 5\\,\\% masked, 20 seeds. Bias, SD, 5th/95th percentiles of $\\hat\\sigma$ (mm); offset: median distance of the detected origin from the true source (mm); tied: fraction of fields whose maximum score is shared by several vertices.", "tab:S4b", star=True, size="footnotesize", colsep="3pt"))
    # ------------------------------------------------------------------ S19 ties of the maximum score (v51b: plateau-centre origin, first index as sensitivity)
    t51 = pd.read_csv(SRC["v51"]).set_index("map_id").loc[M.index]; tt = t51[t51.n_tied > 1]
    foc51 = tt[(tt.r2 >= 0.7) & (tt.sigma < 30)]
    rows = [[int(m), "%.0f" % r.max_score, int(r.n_tied), int(r.n_tied_components), f1(r.spread_mm), f1(r.origin_shift_mm), f2(r.sigma), f2(r.sigma_alt), f2(r.r2), f2(r.r2_alt)] for m, r in tt.iterrows()]
    W("S19_argmax_ties.tex", tex_table(rows, ["map", "max. score", "tied vertices", "regions", "spread (mm)", "shift (mm)", "$\\sigma$ (mm)", "$\\sigma_{\\rm first}$ (mm)", "$R^2$", "$R^2_{\\rm first}$"], "rrrrrrrrrr",
                                          "Maps in which several vertices share the maximum score. The pipeline takes as origin the vertex nearest to the centre of the largest connected tied region on the mesh edge graph ($\\sigma$, $R^2$; Section 3.4 of the manuscript). Sensitivity: the first tied vertex in vertex order ($\\sigma_{\\rm first}$, $R^2_{\\rm first}$); regions: connected tied regions; spread: largest geodesic distance from the origin to a tied vertex; shift: distance between the two origins.", "tab:S19", star=True, size="footnotesize", colsep="3pt"))
    alt = M.copy(); alt.loc[tt.index, "sigma"] = tt.sigma_alt; alt.loc[tt.index, "r2"] = tt.r2_alt; alt["class"] = classify(alt.sigma.values)
    chg_cls = [int(m) for m in tt.index if alt.loc[m, "class"] != M.loc[m, "class"]]; chg_r2 = [int(m) for m in tt.index if (alt.loc[m, "r2"] < .7) != (M.loc[m, "r2"] < .7)]
    NUM["ties_effect"] = dict(class_changed=chg_cls, r2flag_changed=chg_r2, sigma_median_all=[float(M.sigma.median()), float(alt.sigma.median())],
                              median_double=[float(M[M.reference == "double"].sigma.median()), float(alt[alt.reference == "double"].sigma.median())], median_single=[float(M[M.reference == "single"].sigma.median()), float(alt[alt.reference == "single"].sigma.median())],
                              n_r2flag=[int((M.r2 < .7).sum()), int((alt.r2 < .7).sum())], classes_alt=alt["class"].value_counts().to_dict())
    NUM["r2_threshold_clinical"] = {"%.1f" % T: dict(n_flagged=int((M.r2 < T).sum()), sigma_median_rest=float(M[M.r2 >= T].sigma.median())) for T in (0.6, 0.7, 0.8)}
    _foc = (M.r2 >= .7) & (M.sigma < 30); NUM["distance_extra"] = dict(exact_abs_median=float((M.sigma_exact - M.sigma).abs().median()), exact_within1=int(((M.sigma_exact - M.sigma).abs() <= 1).sum()), euclid_gt3=int(((M.sigma_euclid - M.sigma).abs() > 3).sum()), euclid_gt3_focal=int(((M.sigma_euclid - M.sigma).abs()[_foc] > 3).sum()), euclid_gt1_focal=int(((M.sigma_euclid - M.sigma).abs()[_foc] > 1).sum()), euclid_within1=int(((M.sigma_euclid - M.sigma).abs() <= 1).sum()), euclid_within10pct=int(((M.sigma_euclid - M.sigma).abs() <= 0.1 * M.sigma).sum()), n_focal=int(_foc.sum()))
    NUM["max_score"] = dict(median=float(t51.max_score.median()), iqr=list(map(float, q(t51.max_score))), min=float(t51.max_score.min()), max=float(t51.max_score.max()), n_at_100=int((t51.max_score >= 99.999).sum()), n=len(t51))
    NUM["ties"] = dict(n_maps=len(tt), n_tied_range=[int(tt.n_tied.min()), int(tt.n_tied.max())], spread_range=[float(tt.spread_mm.min()), float(tt.spread_mm.max())], dsig_gt1=int(((tt.sigma_alt - tt.sigma).abs() > 1).sum()), dsig_gt1_focal=int(((foc51.sigma_alt - foc51.sigma).abs() > 1).sum()), n_focal=len(foc51), maps_gt1=[int(m) for m in tt[(tt.sigma_alt - tt.sigma).abs() > 1].index])
    # ------------------------------------------------------------------ S20 clinical model comparison (v52 test)
    v52 = pd.read_csv(SRC["v52"]).set_index("map_id").loc[M.index]; rows = []
    for lab, sel in (("all 55 maps", np.ones(len(v52), bool)), ("$R^2_{\\rm Gauss}\\ge0.7$", (v52.r2_gauss >= 0.7).values), ("$R^2_{\\rm Gauss}<0.7$", (v52.r2_gauss < 0.7).values)):
        g = v52[sel]; rows.append([lab, len(g), med_iqr(g.r2_gauss, 2), med_iqr(g.r2_exp, 2), int((g.r2_exp > g.r2_gauss).sum()), med_iqr(g.rmse_gauss, 1), med_iqr(g.rmse_exp, 1), med_iqr(g.rmse_near_gauss, 1), med_iqr(g.rmse_near_exp, 1), "%d / %d / %d" % ((g.dAIC > 2).sum(), (g.dAIC.abs() <= 2).sum(), (g.dAIC < -2).sum())])
    W("S20_model_comparison.tex", tex_table(rows, ["maps", "$n$", "$R^2_{\\rm G}$", "$R^2_{\\rm E}$", "E $>$ G", "RMSE$_{\\rm G}$", "RMSE$_{\\rm E}$", "RMSE$_{\\rm G}^{\\le3\\sigma}$", "RMSE$_{\\rm E}^{\\le3\\sigma}$", "$\\Delta$AIC E/u/G"], "lrcccccccc",
                                          "Gaussian (G) versus exponential (E) fit of the binned clinical profiles (three parameters each, same start values, bounds and bins; fitted over the full geodesic range, median %.0f mm): median [IQR] of $R^2$, of the root-mean-square residual of the bin means (score points) over the whole range and within $3\\hat\\sigma$ of the origin, and the number of maps in which $\\Delta$AIC $=$ AIC$_{\\rm Gauss}-$AIC$_{\\rm exp}$ favours the exponential ($>2$), is undecided (u) or favours the Gaussian ($<-2$)." % v52.fit_range_mm.median(), "tab:S20", star=True, size="scriptsize", colsep="2pt"))
    NUM["model_comparison"] = dict(r2_gauss=float(v52.r2_gauss.median()), r2_exp=float(v52.r2_exp.median()), exp_higher=int((v52.r2_exp > v52.r2_gauss).sum()), rmse_gauss=float(v52.rmse_gauss.median()), rmse_exp=float(v52.rmse_exp.median()), rmse_near_gauss=float(v52.rmse_near_gauss.median()), rmse_near_exp=float(v52.rmse_near_exp.median()), fit_range_median=float(v52.fit_range_mm.median()))
    nar = mk[mk.sigma < 10]; agree5 = mk[mk.offset_marker_mm <= 5]; far = mk[mk.offset_marker_mm > 10]
    NUM["marker"] = dict(n=len(mk), offset_median=float(mk.offset_marker_mm.median()), offset_iqr=list(map(float, q(mk.offset_marker_mm))), le5=int((mk.offset_marker_mm <= 5).sum()), gt10=int((mk.offset_marker_mm > 10).sum()), gt20=int((mk.offset_marker_mm > 20).sum()),
                         narrow_n=len(nar), narrow_median=float(nar.offset_marker_mm.median()), narrow_iqr=list(map(float, q(nar.offset_marker_mm))), narrow_within10=int((nar.offset_marker_mm <= 10).sum()), narrow_gt10_maps=[int(x) for x in nar[nar.offset_marker_mm > 10].index],
                         mid_n=int(((mk.sigma >= 10) & (mk.sigma < 20)).sum()), mid_median=float(mk[(mk.sigma >= 10) & (mk.sigma < 20)].offset_marker_mm.median()), broad_n=int((mk.sigma >= 20).sum()), broad_median=float(mk[mk.sigma >= 20].offset_marker_mm.median()), broad_iqr=list(map(float, q(mk[mk.sigma >= 20].offset_marker_mm))),
                         rho_sigma=float(spearmanr(mk.offset_marker_mm, mk.sigma).correlation), p_sigma=float(spearmanr(mk.offset_marker_mm, mk.sigma).pvalue), rho_sigma_cluster_ci=list(map(float, cluster_boot_rho(mk, "offset_marker_mm", "sigma"))), n_patients_marker=int(mk.patient.nunique()), rho_r2=float(spearmanr(mk.offset_marker_mm, mk.r2).correlation),
                         snap_median=float(v46.loc[mk.index, "snap_mm"].median()), snap_max=float(v46.loc[mk.index, "snap_mm"].max()),
                         sigma_lo_vs_argmax_within5=dict(n=len(agree5), median=float((agree5.sigma_marked_origin - agree5.sigma).median()), iqr=list(map(float, q(agree5.sigma_marked_origin - agree5.sigma)))),
                         sigma_lo_vs_argmax_far=dict(n=len(far), median=float((far.sigma_marked_origin - far.sigma).median()), at_bound=int((far.sigma_marked_origin >= 199).sum())),
                         r2_lo_median=float(mk.r2_marked_origin.median()), r2_lo_lt07=int((mk.r2_marked_origin < .7).sum()), flagged_among_far_narrow=dict(R2=int(nar[nar.offset_marker_mm > 10].flag_r2.sum()), F=int(nar[nar.offset_marker_mm > 10].flag_F.sum())))
    # v36 S1
    d36 = pd.read_csv(SRC["v36d"]); rows = [[s.replace("_", " "), sv, int(g.n_vertices.iloc[0]), f2(g.mean_edge_mm.iloc[0]), "%+.2f" % (100 * g.rel_err_mean.mean()), "%.2f" % (100 * g.rel_err_p95.mean()), f2(g.abs_err_mean_mm.mean())] for (s, sv), g in d36.groupby(["surface", "solver"])]
    W("S1_analytic_surfaces.tex", tex_table(rows, ["surface", "solver", "$N$", "edge (mm)", "signed rel. error (\\%)", "p95 $|$rel.$|$ (\\%)", "mean $|$error$|$ (mm)"], "llrrrrr", "Distance error against analytic geodesics on spheres ($R$ = 25 mm) and open cylinders ($R$ = 20 mm; j0.3: vertices jittered by 0.3 of the edge length); mean over 5 sources per surface. Width recovery on the same surfaces in Table~\\ref{tab:S1b}.", "tab:S1"))
    wb = pd.read_csv(SRC["v36b"]); piv = wb.pivot_table(index=["surface", "src", "sigma0", "noise", "trial"], columns="solver", values="sigma_hat").reset_index()
    rows = []; NUM["v36b"] = {}
    for surf, g in piv.groupby("surface"):
        el = wb[wb.surface == surf].mean_edge_mm.iloc[0]; n = int(wb[wb.surface == surf].n_vertices.iloc[0]); r = [surf.replace("_", " "), n, f2(el)]
        for sv in ("heat", "exact", "dijkstra"):
            d = g[sv] - g.analytic; d1 = d[g.noise == 0.1]; r += [fs(d1.median()), f2(d1.abs().max())]
            NUM["v36b"]["%s_%s" % (surf, sv)] = dict(median=float(d1.median()), maxabs=float(d1.abs().max()), maxabs_noisefree=float(d[g.noise == 0].abs().max()))
        ga = g[g.noise == 0.1]; r += [fs((ga.analytic - ga.sigma0).median())]; rows.append(r)
    W("S1b_width_by_solver.tex", tex_table(rows, ["surface", "$N$", "edge (mm)", "Heat", "max", "exact", "max", "Dijkstra", "max", "analytic bias"], "lrrrrrrrrr",
                                           "Width recovered with each distance definition minus the width recovered with the analytic geodesic distance of the same field (same detected origin; 10\\,\\% noise, 5 sources $\\times$ 10 trials $\\times$ 4 widths per surface): median and maximum absolute difference (mm); last column: median $\\hat\\sigma-\\sigma_0$ with analytic distances.", "tab:S1b", star=True))
    v49 = pd.read_csv(SRC["v49"]).set_index("map_id").loc[M.index]; v49["reference"] = M.reference; v49["atrium"] = M.atrium; v49["region"] = M.region; v49["sigma_pipeline"] = M.sigma
    foc = v49[v49.sigma_iso < 30]; bnd = (foc.sigma_major >= 199) | (foc.sigma_minor <= 0.11); fo = foc[~bnd]
    rows = []
    for lab, g in (("all focal maps ($\\sigma_{\\rm iso}<30$ mm)", foc), ("~~without principal width at a bound", fo), ("double-chamber reference", fo[fo.reference == "double"]), ("single-chamber reference", fo[fo.reference == "single"]),
                   ("LA posterior", fo[(fo.atrium == "LA") & (fo.region == "posterior")]), ("LA anterior", fo[(fo.atrium == "LA") & (fo.region == "anterior")]), ("RA posterior", fo[(fo.atrium == "RA") & (fo.region == "posterior")]), ("RA lateral", fo[(fo.atrium == "RA") & (fo.region == "lateral")])):
        rows.append([lab, len(g), med_iqr(g.ratio), int((g.ratio >= 2).sum()), int((g.p_perm < .05).sum()), int((g.delta_aic > 0).sum()), med_iqr(g.sigma_minor), med_iqr(g.sigma_major)])
    W("S8_anisotropy_clinical.tex", tex_table(rows, ["subset", "$n$", "ratio, median [IQR]", "ratio $\\ge2$", "$p<0.05$", "$\\Delta$AIC $>0$", "$\\sigma_{\\rm minor}$ (mm)", "$\\sigma_{\\rm major}$ (mm)"], "lrcrrrcc",
                                              "Exploratory anisotropy of the clinical maps: elliptical Gaussian fitted in geodesic polar coordinates (logarithmic map, Vector Heat Method) around the detected origin; ratio $=\\sigma_{\\rm major}/\\sigma_{\\rm minor}$; $p$ from 100 permutations of the angular position within 2-mm rings (minimum attainable $p$ = 0.01); $\\Delta$AIC $>0$: elliptical model preferred over the isotropic one. Orientation is not reported (coordinate convention of the export not verified). %d of %d focal maps have a principal width at a fit bound and are excluded from the subsets." % (int(bnd.sum()), len(foc)), "tab:S8", star=True, size="footnotesize", colsep="3pt"))
    NUM["aniso"] = dict(n_focal=len(foc), n_bound=int(bnd.sum()), ratio_median_all=float(foc.ratio.median()), ratio_iqr_all=list(map(float, q(foc.ratio))), p05_all=int((foc.p_perm < .05).sum()), ratio_median=float(fo.ratio.median()), ratio_iqr=list(map(float, q(fo.ratio))), p05=int((fo.p_perm < .05).sum()), n=len(fo), ge2=int((fo.ratio >= 2).sum()), daic=int((fo.delta_aic > 0).sum()),
                        rho_ratio_sigma=float(spearmanr(fo.ratio, fo.sigma_pipeline).correlation), rho_ratio_r2=float(spearmanr(fo.ratio, M.loc[fo.index, "r2"]).correlation), sigma_iso_vs_pipeline_median_diff=float((v49.sigma_iso - v49.sigma_pipeline).median()))
    rows = [[int(r.map_id), int(r.n_largest), f2(r.mean_edge_mm), "%+.2f" % (100 * r.relerr_heat_all), "%.2f" % (100 * r.p95_heat_all), f2(r.abserr_mm_heat_all), "%+.2f" % (100 * r.relerr_heat_30), "%.2f" % (100 * r.p95_heat_30), f1(r.max_exact_mm)] for _, r in v38all.iterrows()]
    v38 = v38all
    cap2 = tex_minus("Heat Method versus exact polyhedral geodesics from the clinical argmax origin on the 56 meshes (whole mesh, and between 2 and 30 mm from the origin (subscript 2--30); rel. err.: mean signed relative error; p95: 95th percentile of the absolute relative error; $|$err.$|$: mean absolute error; max $d$: largest exact distance; map 7 is included here because the comparison concerns the geometry only). Medians over maps: %+.2f\\,\\%% (IQR %+.2f to %+.2f), p95 %.2f\\,\\%%; between 2 and 30 mm from the origin %+.2f\\,\\%% (IQR %+.2f to %+.2f), p95 %.2f\\,\\%%." % (100 * v38.relerr_heat_all.median(), 100 * v38.relerr_heat_all.quantile(.25), 100 * v38.relerr_heat_all.quantile(.75), 100 * v38.p95_heat_all.median(), 100 * v38.relerr_heat_30.median(), 100 * v38.relerr_heat_30.quantile(.25), 100 * v38.relerr_heat_30.quantile(.75), 100 * v38.p95_heat_30.median()))
    W("S2_heat_vs_exact_meshes_long.tex", long_table(rows, ["map", "$N$", "edge (mm)", "rel. err. (\\%)", "p95 (\\%)", "$|$err.$|$ (mm)", "rel. err.$_{2-30}$ (\\%)", "p95$_{2-30}$ (\\%)", "max $d$ (mm)"], "rrrrrrrrr", cap2, "tab:S2"))
    W("S2_heat_vs_exact_meshes.tex", tex_table(rows, ["map", "$N$", "edge (mm)", "rel. err. (\\%)", "p95 (\\%)", "$|$err.$|$ (mm)", "rel. err.$_{2-30}$ (\\%)", "p95$_{2-30}$ (\\%)", "max $d$ (mm)"], "rrrrrrrrr", tex_minus("Heat Method versus exact polyhedral geodesics from the clinical argmax origin on the 56 meshes (whole mesh, and between 2 and 30 mm from the origin (subscript 2--30); rel. err.: mean signed relative error; p95: 95th percentile of the absolute relative error; $|$err.$|$: mean absolute error; max $d$: largest exact distance; map 7 is included here because the comparison concerns the geometry only). Medians over maps: %+.2f\\,\\%% (IQR %+.2f to %+.2f), p95 %.2f\\,\\%%; between 2 and 30 mm from the origin %+.2f\\,\\%% (IQR %+.2f to %+.2f), p95 %.2f\\,\\%%." % (100 * v38.relerr_heat_all.median(), 100 * v38.relerr_heat_all.quantile(.25), 100 * v38.relerr_heat_all.quantile(.75), 100 * v38.p95_heat_all.median(), 100 * v38.relerr_heat_30.median(), 100 * v38.relerr_heat_30.quantile(.25), 100 * v38.relerr_heat_30.quantile(.75), 100 * v38.p95_heat_30.median())), "tab:S2", star=True))
    NUM["heat_vs_exact_all"] = dict(median=float(v38.relerr_heat_all.median()), iqr=list(map(float, q(v38.relerr_heat_all))), p95_median=float(v38.p95_heat_all.median()), abs_median=float(v38.abserr_mm_heat_all.median()), abs_max=float(v38.abserr_mm_heat_all.max()), per_mesh_min=float(v38.relerr_heat_all.min()), per_mesh_max=float(v38.relerr_heat_all.max()))
    # numbers.md
    json.dump(NUM, open(os.path.join(a.out, "numbers.json"), "w"), indent=1, default=str)
    md = ["# Headline numbers (from make_revision_tables.py)", ""]
    cl = NUM["clinical"]; md += ["## Clinical (55 maps, %d patients)" % cl["n_patients"], "- sigma median %.2f [%.2f; %.2f], range %.1f-%.1f" % (cl["sigma_median"], cl["sigma_q1"], cl["sigma_q3"], cl["sigma_min"], cl["sigma_max"]),
                                 "- R2 median %.3f, range %.2f-%.2f; flagged %d; F flagged %d; classes %s" % (cl["r2_median"], cl["r2_min"], cl["r2_max"], cl["n_flag_r2"], cl["n_flag_F"], cl["classes"]),
                                 "- double %.2f [%.2f; %.2f] vs single %.2f [%.2f; %.2f]" % (cl["sigma_median_double"], *cl["iqr_double"], cl["sigma_median_single"], *cl["iqr_single"]),
                                 "- F, ALL 55 MAPS (descriptive): median %.2f [%.2f; %.2f] min %.2f; operational flags (F<0.70 and R2>=0.7) %s; sigma of flagged median %.1f (%.1f-%.1f); other comp. %.0f-%.0f mm, height %.2f-%.2f of peak; also R2-flagged %d; by threshold %s; ncomp median %.0f, single comp. %d; rho(F,R2) %.2f rho(F,sigma) %.2f (all 55) | F_pct median %.2f [%.2f; %.2f], flagged %d, by threshold %s" % (
                                     cl["F_median"], *cl["F_iqr"], cl["F_min"], cl["F_flagged_maps"], cl["F_flag_sigma_median"], *cl["F_flag_sigma_range"], *cl["F_flag_other_dist"], *cl["F_flag_other_height"], cl["F_flag_and_r2"], cl["F_threshold"], cl["ncomp_median"], cl["ncomp_1"], cl["rho_F_r2"], cl["rho_F_sigma"], cl["F_pct_median"], *cl["F_pct_iqr"], cl["n_flag_F_pct"], cl["F_pct_threshold"]), "- dAIC: exp preferred %d, Gauss %d, undecided %d; beta median %.2f [%.2f; %.2f]; FWHM_mf/2.355sigma median %.2f [%.2f; %.2f]" % (cl["dAIC_exp_pref"], cl["dAIC_gauss_pref"], cl["dAIC_undecided"], cl["beta_median"], *cl["beta_iqr"], cl["fwhm_ratio_median"], *cl["fwhm_ratio_iqr"]),
                                 "- low-R2 maps: sigma median %.1f, max %.1f; A median flagged %.1f vs %.1f" % (cl["lowr2_sigma_median"], cl["lowr2_sigma_max"], cl["A_median_flagged"], cl["A_median_unflagged"]), ""]
    md += ["## Clinical extra: %s" % NUM["clinical_extra"], "", "## Validation (detected origin / known origin), eta = 10 %"] + ["- sigma0 %s: bias %+.2f [%+.2f; %+.2f], SD %.2f, offset %.1f | known: bias %+.2f, SD %.2f" % (k.split("/")[0], v["bias"], *v["ci"], v["sd"], v["off"], v["bias_known"], v["sd_known"]) for k, v in NUM["validation"].items() if k.endswith("/0.1")]
    md += ["", "## F, ELIGIBLE MAPS (R2 >= 0.7; as reported in the manuscript): n %d, median %.2f [%.2f; %.2f], flagged %d, by threshold %s, single comp. %d, rho(F,sigma) %.2f, rho(F,R2) %.2f; F<0.70 in maps with R2<0.7 (not interpreted): %d" % (NUM["clinical"]["n_F_eligible"], NUM["clinical"]["F_median_eligible"], *NUM["clinical"]["F_iqr_eligible"], NUM["clinical"]["n_flag_F"], NUM["clinical"]["F_threshold_eligible"], NUM["clinical"]["ncomp_1_eligible"], NUM["clinical"]["rho_F_sigma_eligible"], NUM["clinical"]["rho_F_r2_eligible"], NUM["clinical"]["F_low_ineligible"])]
    md += ["", "## PACING SITES vs WIDTH (55 maps; added 29 Oct 2026): Spearman rho(n_points, sigma) %.3f" % NUM["clinical"]["rho_npoints_sigma"]]
    md += ["", "## Table 1, primary protocol (v56, peak 90, 10 %% noise; bootstrap CI): %s" % NUM["table1"]]
    md += ["", "## Peak below the score limit (v56): %s" % NUM["v56"], "", "## Clinical maximum vertex score: %s" % NUM["max_score"], "", "## R2 threshold on the clinical maps: %s" % NUM["r2_threshold_clinical"], "", "## Distance definitions, extra: %s" % NUM["distance_extra"]]
    md += ["", "## Ties of the maximum score (v51): %s" % NUM["ties"], "", "## Ties, effect of the plateau origin on classes/flags/summaries: %s" % NUM["ties_effect"]]
    md += ["", "## Marker check (%d maps)" % NUM["marker"]["n"]] + ["- %s: %s" % (k, v) for k, v in NUM["marker"].items() if k != "n"]
    md += ["", "## Distance definitions"] + ["- %s: %s" % (k, v) for k, v in NUM["distance"].items()]
    md += ["", "## Correlated/structured (v47b; pooled over sigma0 unless given)"] + ["- %s: %s" % (k, {kk: (round(vv, 3) if isinstance(vv, float) else [round(x, 3) for x in vv]) for kk, vv in v.items()}) for k, v in NUM["corr"].items()]
    md += ["", "## v36b width by solver minus analytic (10 % noise)"] + ["- %s: %s" % (k, {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in NUM["v36b"].items()]
    md += ["", "## Anisotropy clinical (v49): %s" % NUM["aniso"], "", "## Mixed model (v58, current: REML at the profiled optimum, approximate t(G-2)): %s" % json.dumps(NUM["model"]["inference"]), "", "## Mixed-model terms (v58): %s" % NUM["model"]["terms"], "", "## v40 (SUPERSEDED for model estimates: non-converged statsmodels REML; still the source of patient medians and the bootstrap only): %s" % NUM["lmm"], "", "## F percentile-set property (origin check): %s" % NUM["F_pct_origin"], "", "## F synthetic (v62 sphere; F rates among fields with F available)"] + ["- %s: %s" % (k, v) for k, v in NUM["F_synthetic"].items()] + ["", "## F meshes (v62; F rates among fields with F available; F_available / n per sigma0 = 5, 10, 15)"] + ["- %s: %s" % (k, v) for k, v in NUM["F_meshes"].items()]
    md += ["", "## Heat vs exact (whole mesh): %s" % NUM["heat_vs_exact_all"], "", "## Runtime: %s" % NUM["runtime"], "", "## Origin rule: %s" % NUM["origin_rule"], "", "## v40 summary text (SUPERSEDED model block; patient medians/bootstrap current):", "```", lmm, "```"]
    W("numbers.md", "\n".join(md)); write_manifest(a.out); print("\n".join(md[:40]))


# source-to-output manifest (round 4/5, Q13): every generated table -> SRC keys and denominator; the script is identified by the result folder prefix
CLIN = "55 clinical maps of 16 patients (59 maps minus 3 fragmented meshes and map 7, quality exclusion)"
SYN56 = "1,120 fits per row = 56 meshes x 20 seeds"
TABLE_SRC = {
    "T1a_validation_iid": (["v56all"], SYN56), "T1b_validation_correlated": (["v47b", "v43"], "560 fields per row = 56 meshes x 10 seeds; clinical scatter: 55 maps"),
    "T2_clinical_groups": (["v38", "v50c", "v58t", "v58i", "v40"], CLIN + "; model: 55 maps / 16 patients; F flags among maps with R2 >= 0.7"),
    "S1_analytic_surfaces": (["v36d"], "per surface and solver: 5 sources"), "S1b_width_by_solver": (["v36b"], "per surface: 5 sources x (1 noise-free + 10 noisy trials) x 4 widths"),
    "S2_heat_vs_exact_meshes": (["v38"], "56 non-fragmented meshes (incl. map 7)"), "S3_validation_grid": (["v47av", "v46v"], SYN56), "S3b_known_origin_wide": (["v46v"], SYN56),
    "S4_range": (["v42"], SYN56), "S4b_peak_below_limit": (["v56all"], SYN56), "S5_origin_rule": (["v47av", "v47ac"], SYN56 + "; clinical: 46 maps with a marked site"),
    "S6_nongaussian": (["arm3"], "560 fits per row = 56 meshes x 10 seeds"), "S7_anisotropy_sphere": (["v44"], "20 trials per row (sphere)"), "S8_anisotropy_clinical": (["v49"], "focal maps (sigma_iso < 30 mm) of the 55; subsets as labelled"),
    "S9_structured_meshes": (["v47b"], "560 fields per row = 56 meshes x 10 seeds"), "S10_scatter_sources": (["v47c"], "20 trials per row (sphere)"),
    "S11a_r2_threshold": (["v37"], "sphere: 60 fields (Gaussian, 3 widths x 20), 20 (flat, ring), 180 (two-peak) per noise level"),
    "S11b_focality_detection": (["v50s"], "sphere: n = 20 fields per row and noise level; F rates among fields with F available"),
    "S11c_focality_other": (["v50s"], "sphere: n per row as in the table (60 single peak, 20 flat/ring); F rates among fields with F available"),
    "S12_per_map": (["v38", "v39", "v43", "v46c", "v50c", "study", "occ"], CLIN), "S13_distance_definitions": (["v38"], CLIN), "S14_focality_clinical": (["v39", "v50c", "occ"], CLIN + "; F row 1: 42 maps with R2 >= 0.7"),
    "S14b_focality_meshes": (["v50m"], "56 meshes x 3 seeds per sigma0 and condition; F rates among fields with F available"), "S15_mixed_model": (["v58t", "v58i", "v40"], "55 maps / 16 patients (8 vs 8); R2 subset 42 maps / 13 patients (6 vs 7)"),
    "S16_runtime": (["v41"], "59 clinical meshes"), "S17_binwidth_smoothing": (["v45"], CLIN), "S18_marker_check": (["v46c"], "46 maps with a marked pacing site; subsets as labelled"),
    "S19_argmax_ties": (["v51"], "14 maps with a tied maximum score"), "S20_model_comparison": (["v52"], CLIN + "; subsets by R2"),
}


HAND_TABLES = {  # tables typed in the manuscript source, not generated (round 6)
    "tab:comparison": "literature-based: published approaches and values (checked on 26 Oct 2026 against the sources in method-paper/literature/); no simulation or clinical denominator, no result file",
}
REV = "thesis-med/_paper/method-paper/revision_2026-10"; SV = "thesis-med/_paper/method-paper/supplementary_validation_results"
CLIN_FIG = "yes, from the per-map cache of the maps shown (mesh, scores, distances, marker vertex; built from pulse-ep/data/validation_meshes); every sigma/R2/F shown is asserted against S12_per_map.csv, F also against pulse_ep_decay.analysis.focality_index"
CACHE_F = lambda *ms: [REV + "/figs/_cache/map_%03d.npz" % m for m in ms] + [REV + "/tables/S12_per_map.csv"]
FIGURE_SRC = {  # figure file -> (generator function of make_revision_figures.py, source files relative to ROOT, reproducible from the deposited result files?)
    "fig_workflow": ("fig_workflow", CACHE_F(28), CLIN_FIG),
    "fig_validation": ("fig_validation", [SV + "/v56_peak_below_clip_all_2026-10-26/v56_raw.csv"], "yes"),
    "fig_heatmap": ("fig_heatmap", CACHE_F(75, 78), CLIN_FIG),
    "fig_clinical_example": ("fig_example", CACHE_F(1, 25), CLIN_FIG),
    "figS_decay_profile": ("fig_profiles", CACHE_F(25, 54, 32), CLIN_FIG),  # main-text Fig. 3 until 29 Oct 2026
    "figS_boxplot_sigma": ("fig_boxplot", [REV + "/tables/S12_per_map.csv"], "yes"),
    "figS_analytic_surfaces": ("fig_supp", [SV + "/v36_analytic_geodesic_2026-11-06b/v36_distance_errors.csv"], "yes"),
    "figS_anisotropy_sphere": ("fig_supp", [SV + "/v60_anisotropy_pkgfit_2026-10-26/v60_rows.csv"], "yes"),
    "figS_structured_deviations": ("fig_supp", [SV + "/v47b_structured_noise_plateau_2026-10-26/v47b_summary.csv"], "yes"),
    "figS_focality_synthetic": ("fig_supp", [SV + "/v62_focality_unified_2026-10-26/v62_sphere.csv"], "yes"),
    "figS_runtime": ("fig_supp", [SV + "/v41_runtime_sparsity_2026-10-21/v41_per_mesh.csv"], "yes"),
    "fig_marker_check": ("fig_supp", [SV + "/v46_lo_origin_plateau_2026-10-26/v46_clinical.csv"], "yes"),
}


def visible_items():
    """Tables and figures as numbered in the PDFs, parsed from the LaTeX sources (round 6: the check covers what the reader sees)."""
    tex = lambda f: open(os.path.join(MP, "revision_2026-10", f)).read()
    if not all(os.path.exists(os.path.join(MP, "revision_2026-10", f)) for f in ("paper_rev1.tex", "supplement_rev1.tex")): return None, None  # deposit without manuscript sources
    paper, supp = tex("paper_rev1.tex"), tex("supplement_rev1.tex")
    paper = re.sub(r"(?m)(?<!\\)%.*$", "", paper); supp = re.sub(r"(?m)(?<!\\)%.*$", "", supp)
    tabs = []
    for k, m in enumerate(re.finditer(r"\\input\{tables/([^}]+)\}|\\begin\{table\*?\}(?:\[[^\]]*\])?(.*?)\\end\{table\*?\}", paper, re.S), 1):
        if m.group(1): tabs.append(("Table %d" % k, m.group(1), None))
        else: lab = re.search(r"\\label\{([^}]+)\}", m.group(2)); tabs.append(("Table %d" % k, None, lab.group(1) if lab else "(unlabelled)"))
    assert not re.search(r"\\begin\{table", supp), "hand-written table in the supplement: add it to HAND_TABLES"
    tabs += [("Table " + m.group(1), m.group(2), None) for m in re.finditer(r"\\stab\{([^}]+)\}\\input\{tables_supp/([^}]+)\}", supp)]
    figs = [("Fig. %d" % k, os.path.splitext(m.group(1))[0]) for k, m in enumerate(re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", paper), 1)]
    figs += [("Fig. S%d" % k, os.path.splitext(m.group(1))[0]) for k, m in enumerate(re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", supp), 1)]
    return tabs, figs


def write_manifest(out):
    import hashlib, glob as _g
    scripts = os.path.join(MP, "scripts")
    def script_for(path):
        d = os.path.basename(os.path.dirname(path)); tag = d.split("_")[0]
        c = sorted(_g.glob(os.path.join(scripts, tag + "_*.py"))) or sorted(_g.glob(os.path.join(ROOT, "geodesic-decay/scripts", tag + "*.py")))
        return os.path.relpath(c[0], ROOT) if c else "(see folder README)"
    sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()[:12] if os.path.exists(f) else "MISSING"
    made = {os.path.splitext(f)[0] for f in os.listdir(out) if f.endswith(".tex")}
    missing = sorted(t for t in made if t.replace("_long", "") not in TABLE_SRC)
    assert not missing, "tables without manifest entry: %s" % missing
    tabs, figs = visible_items(); gen_of = lambda f: f.replace("_long", "")
    if tabs is None:  # deposit without the manuscript sources: list the generated tables and all figure entries, without visible labels
        tabs = [("(%s)" % t, t, None) for t in TABLE_SRC] + [("(typed)", None, lab) for lab in HAND_TABLES]; figs = [("(%s)" % f, f) for f in FIGURE_SRC]
    unmapped = [v for v, f, lab in tabs if (f and gen_of(f) not in TABLE_SRC) or (lab and lab not in HAND_TABLES)]
    assert not unmapped, "visible tables without manifest entry: %s" % unmapped
    nofig = [v for v, f in figs if f not in FIGURE_SRC]; assert not nofig, "figures without manifest entry: %s" % nofig
    L = ["# Source-to-output manifest (generated by make_revision_tables.py)", "",
         "Every table and figure of the manuscript and the supplement as numbered in the PDFs (parsed from paper_rev1.tex and supplement_rev1.tex;",
         "the generator stops if a visible table or figure has no entry).  Result paths relative to ~/IdeaProjects (or GD_ROOT of an unpacked deposit).", "",
         "## Visible tables", "", "| visible label | generated file (or source) | denominator |", "|---|---|---|"]
    for v, f, lab in tabs:
        if f:
            exp = "T" + v[6:] if v[6:].isdigit() else v[6:]
            L.append("| %s | %s%s | %s |" % (v, gen_of(f), "" if gen_of(f).split("_")[0] == exp else " (generated under this name, shown as %s)" % v, TABLE_SRC[gen_of(f)][1]))
        else: L.append("| %s | typed in the manuscript (\\label{%s}) | %s |" % (v, lab, HAND_TABLES[lab]))
    L += ["", "## Generated tables: sources", "", "| table | denominator | source key | result file | sha256 (12) | script |", "|---|---|---|---|---|---|"]
    for t, (keys, den) in TABLE_SRC.items():
        for i, k in enumerate(keys):
            f = SRC[k]; L.append("| %s | %s | %s | %s | %s | %s |" % (t if i == 0 else "", den if i == 0 else "", k, os.path.relpath(f, ROOT), sha(f), script_for(f)))
    L += ["", "## Figures", "", "| visible label | file | generator (make_revision_figures.py) | source file | sha256 (12) | rebuilt from the listed files |", "|---|---|---|---|---|---|"]
    for v, f in figs:
        gen, srcs, rep = FIGURE_SRC[f]
        for i, sf in enumerate(srcs):
            L.append("| %s | %s | %s | %s | %s | %s |" % (v if i == 0 else "", f if i == 0 else "", gen if i == 0 else "", sf, sha(os.path.join(ROOT, sf)), rep if i == 0 else ""))
    L += ["", "Mixed model (Table 2, S15): v58 (REML at the profiled optimum, approximate t(G-2)); v40 supplies patient medians and the bootstrap only.",
          "F (S11b/c, S12, S14, S14b, %s): single implementation pulse_ep_decay.analysis.focality_index (v62); v50 superseded." % dict((f, v) for v, f in figs)["figS_focality_synthetic"],
          "Anisotropy (S7, %s): v60 (package fitter); v44 superseded." % dict((f, v) for v, f in figs)["figS_anisotropy_sphere"],
          "Estimator of all end-to-end arms: plateau-centre origin (pulse_ep_decay.profile.find_origin) and the package fit (amplitude unconstrained);",
          "component experiments that do not select an origin from scores (v36 distance errors, v41 runtime) are not affected."]
    open(os.path.join(out, "MANIFEST.md"), "w").write("\n".join(L) + "\n")
    json.dump(dict(tables=[dict(visible=v, file=f, label=lab) for v, f, lab in tabs], figures=[dict(visible=v, file=f, generator=FIGURE_SRC[f][0], sources=FIGURE_SRC[f][1]) for v, f in figs]),
              open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
