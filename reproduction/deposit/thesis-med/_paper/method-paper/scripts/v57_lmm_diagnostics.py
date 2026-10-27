"""v57 — TEST (26 Oct 2026): diagnostics of the mixed model of v40 (all 55 maps, raw sigma) after the R1 tie rule.
Reports convergence, warnings and the random-intercept variance of the REML fit, and as a sensitivity an OLS fit of
log sigma ~ reference + region with patient-cluster-robust standard errors (16 clusters).  Input: v38 plateau per-map
widths and the study map (same data preparation as v40).  Nothing in the manuscript is changed by this script."""
import os, warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf
SVR = os.path.expanduser("~/mnt/IdeaProjects/thesis-med/_paper/method-paper/supplementary_validation_results")
OUT = os.path.join(SVR, "v57_lmm_diagnostics_2026-10-26"); os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(os.path.join(SVR, "v38_clinical_solvers_plateau_2026-10-26/v38_per_map.csv"))
df = df[~df.fragmented & ~df.excluded_quality].copy()
df["logs"] = np.log(df.sigma_heat); df["patient"] = df.study_id.astype(str)
df["region"] = pd.Categorical(df.atrium + " " + df.region.str.replace(" wall", ""), ["LA posterior", "LA anterior", "RA posterior", "RA lateral"])
df["reference"] = pd.Categorical(df.reference, ["double", "single"])
L = ["v57 LMM diagnostics (n=%d maps, %d patients)" % (len(df), df.patient.nunique())]
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    m = smf.mixedlm("logs ~ reference + region", df, groups=df["patient"]).fit(reml=True)
L.append("REML converged: %s; random-intercept variance %.4g; warnings: %s" % (m.converged, float(m.cov_re.iloc[0, 0]), "; ".join(sorted(set(str(x.message)[:120] for x in w))) or "none"))
k = "reference[T.single]"; ci = m.conf_int().loc[k]
L.append("LMM ratio %.2f [%.2f; %.2f]" % (np.exp(m.params[k]), np.exp(ci[0]), np.exp(ci[1])))
o = smf.ols("logs ~ reference + region", df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.patient)[0]})
ci = o.conf_int().loc[k]
L.append("OLS, patient-cluster-robust SE: ratio %.2f [%.2f; %.2f], p = %.3f" % (np.exp(o.params[k]), np.exp(ci[0]), np.exp(ci[1]), o.pvalues[k]))
open(os.path.join(OUT, "v57_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
