#!/usr/bin/env python3
"""v40 — Determinants of sigma with the patient as the unit of correlation
(reviewer 3, point 5; statistics question of reviewers 1/3).

Data: per-map width of the April 2026 design (v38_per_map.csv, Heat
Method), reference configuration (single = decapolar CS catheter; double =
CS + RA catheter), region (atrium x wall), patient (map_study.csv).
Note: in this cohort the reference configuration is a BETWEEN-patient
factor (8 patients single, 8 double; no patient has both), region a
WITHIN-patient factor (up to 4 maps per patient).

Analyses (all on log sigma, sigma censored at 30 mm as a sensitivity):
  1. linear mixed model  log sigma ~ reference + region + (1 | patient), REML
  2. patient-level summary: median sigma per patient, Mann-Whitney 8 vs 8
  3. cluster bootstrap over patients (2000 resamples) for the ratio of
     group medians single / double
  4. the same restricted to maps with R2 >= 0.7
Outputs: v40_summary.txt, v40_patient_medians.csv.  TEST only.
"""
import argparse, os, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.environ.get("GD_ROOT", os.path.expanduser("~/mnt/IdeaProjects")), "thesis-med/_paper/method-paper/supplementary_validation_results")


def lmm(df, label, L):
    df = df.copy(); df["logs"] = np.log(df.sigma_used); df["patient"] = df.study_id.astype(str)
    df["region"] = pd.Categorical(df.atrium + " " + df.region.str.replace(" wall", ""), ["LA posterior", "LA anterior", "RA posterior", "RA lateral"])
    df["reference"] = pd.Categorical(df.reference, ["double", "single"])
    try:
        m = smf.mixedlm("logs ~ reference + region", df, groups=df["patient"]).fit(reml=True)
        L.append("  [%s] LMM log sigma ~ reference + region + (1|patient), n=%d maps, %d patients" % (label, len(df), df.patient.nunique()))
        for k in m.params.index:
            if k == "Group Var": continue
            ci = m.conf_int().loc[k]
            L.append("    %-26s coef %+.3f  [%+.3f; %+.3f]  ratio exp(coef) %.2f [%.2f; %.2f]  p=%.3f" % (k, m.params[k], ci[0], ci[1], np.exp(m.params[k]), np.exp(ci[0]), np.exp(ci[1]), m.pvalues[k]))
        vp = float(m.cov_re.iloc[0, 0]); ve = float(m.scale)
        L.append("    patient variance %.3f, residual variance %.3f, ICC %.2f" % (vp, ve, vp / (vp + ve)))
        m0 = smf.mixedlm("logs ~ region", df, groups=df["patient"]).fit(reml=False)
        m1 = smf.mixedlm("logs ~ reference + region", df, groups=df["patient"]).fit(reml=False)
        from scipy.stats import chi2
        lr = 2 * (m1.llf - m0.llf); L.append("    likelihood-ratio test for reference (ML fits): chi2 = %.2f, p = %.3f" % (lr, chi2.sf(lr, 1)))
    except Exception as e:
        L.append("  [%s] LMM failed: %r" % (label, e))


def patient_level(df, label, L, rng):
    pm = df.groupby(["study_id", "reference"]).sigma_used.median().reset_index()
    s = pm[pm.reference == "single"].sigma_used.values; d = pm[pm.reference == "double"].sigma_used.values
    u = mannwhitneyu(s, d, alternative="two-sided")
    L.append("  [%s] patient medians: single n=%d median %.2f [%.2f; %.2f]; double n=%d median %.2f [%.2f; %.2f]; Mann-Whitney p=%.3f" % (
        label, len(s), np.median(s), np.percentile(s, 25), np.percentile(s, 75), len(d), np.median(d), np.percentile(d, 25), np.percentile(d, 75), u.pvalue))
    # cluster bootstrap over patients for the ratio of map-level group medians
    pats = {r: df[df.reference == r].study_id.unique() for r in ("single", "double")}
    ratios = []
    for _ in range(2000):
        parts = []
        for r in ("single", "double"):
            pick = rng.choice(pats[r], size=len(pats[r]), replace=True)
            parts.append(pd.concat([df[(df.study_id == p) & (df.reference == r)] for p in pick]))
        ratios.append(parts[0].sigma_used.median() / parts[1].sigma_used.median())
    obs = df[df.reference == "single"].sigma_used.median() / df[df.reference == "double"].sigma_used.median()
    L.append("    ratio of map-level medians single/double = %.2f; patient-cluster bootstrap 95%% interval [%.2f; %.2f] (2000 resamples)" % (obs, np.percentile(ratios, 2.5), np.percentile(ratios, 97.5)))
    return pm


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(os.path.join(ROOT, "v38_clinical_solvers_plateau_2026-10-26/v38_per_map.csv"))
    df = df[(~df.fragmented) & (~df.excluded_quality)].copy()
    rng = np.random.default_rng(20261021)
    L = ["v40 determinants of sigma with patient as the unit of correlation — summary", "=" * 70,
         "n = %d maps, %d patients; maps per patient: %s; reference is between-patient (no patient has both)" % (
             len(df), df.study_id.nunique(), df.groupby("study_id").size().value_counts().sort_index().to_dict()), ""]
    L.append("Map-level descriptives (Heat Method, April design): single median %.2f [%.2f; %.2f] (n=%d), double %.2f [%.2f; %.2f] (n=%d); map-level Mann-Whitney p=%.4f (NOT valid: maps are not independent)" % (
        df[df.reference == "single"].sigma_heat.median(), df[df.reference == "single"].sigma_heat.quantile(.25), df[df.reference == "single"].sigma_heat.quantile(.75), (df.reference == "single").sum(),
        df[df.reference == "double"].sigma_heat.median(), df[df.reference == "double"].sigma_heat.quantile(.25), df[df.reference == "double"].sigma_heat.quantile(.75), (df.reference == "double").sum(),
        mannwhitneyu(df[df.reference == "single"].sigma_heat, df[df.reference == "double"].sigma_heat).pvalue))
    out_pm = []
    for label, sub, cens in (("all 55, raw sigma", df, None), ("all 55, sigma censored at 30 mm", df, 30.0), ("R2>=0.7 (n=%d), raw sigma" % (df.r2_heat >= .7).sum(), df[df.r2_heat >= .7], None)):
        s = sub.copy(); s["sigma_used"] = s.sigma_heat if cens is None else s.sigma_heat.clip(upper=cens)
        L.append(""); lmm(s, label, L); pm = patient_level(s, label, L, rng); pm["analysis"] = label; out_pm.append(pm)
    pd.concat(out_pm).to_csv(os.path.join(a.out, "v40_patient_medians.csv"), index=False)
    txt = "\n".join(L); print(txt); open(os.path.join(a.out, "v40_summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
