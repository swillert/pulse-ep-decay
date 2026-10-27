"""v58 — TEST (26 Oct 2026, review round 4, Q1): convergence and stability of every model behind Table 2 / S15.
For the three analyses of v40 (all 55 raw; all 55 clipped at 30 mm; 42 maps R2>=0.7) this script records
  (a) convergence, warnings, gradient norm and random-intercept variance of the REML fit and of the full/reduced ML fits
      used for the likelihood-ratio test (statsmodels MixedLM, default optimiser chain as in v40);
  (b) the REML and ML profile log-likelihood of the random-intercept variance ratio gamma = tau^2/sigma^2 on a grid,
      computed in closed form (own GLS code), and the location of its maximum;
  (c) small-sample alternatives that do not rely on the random-effect fit:
      OLS of log sigma ~ reference + region with CR2 (Bell-McCaffrey) patient-cluster-robust SE and Satterthwaite df;
      a wild cluster bootstrap (Webb 6-point weights, restricted, 9,999 draws) p value for the reference term;
      a patient-level analysis: mean log sigma per patient (region-adjusted residual mean and raw mean), Welch t, 8 vs 8.
Input: v38 plateau per-map widths (same preparation as v40/v57).  Nothing in the manuscript is changed by this script."""
import os, warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
SVR = os.path.expanduser("~/mnt/IdeaProjects/thesis-med/_paper/method-paper/supplementary_validation_results")
OUT = os.path.join(SVR, "v58_model_diagnostics_2026-10-26"); os.makedirs(OUT, exist_ok=True)
base = pd.read_csv(os.path.join(SVR, "v38_clinical_solvers_plateau_2026-10-26/v38_per_map.csv"))
base = base[~base.fragmented & ~base.excluded_quality].copy()
base["patient"] = base.study_id.astype(str)
base["region"] = pd.Categorical(base.atrium + " " + base.region.str.replace(" wall", ""), ["LA posterior", "LA anterior", "RA posterior", "RA lateral"])
base["reference"] = pd.Categorical(base.reference, ["double", "single"])
K = "reference[T.single]"; rng = np.random.default_rng(20261026)

def fit(formula, df, reml):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = smf.mixedlm(formula, df, groups=df["patient"]).fit(reml=reml)
    msg = sorted(set(str(x.message).split(".")[0][:70] for x in w))
    g = np.nan
    try: g = float(np.max(np.abs(m.model.score(m.params_object))))
    except Exception: pass
    return m, msg, g

def design(df, formula):
    import patsy
    y, X = patsy.dmatrices(formula, df, return_type="dataframe"); return y.values.ravel(), X

def profile(y, X, groups, gammas, reml):
    Xv = X.values; n, p = Xv.shape; out = []
    G = pd.factorize(groups)[0]; Z = np.zeros((n, G.max() + 1)); Z[np.arange(n), G] = 1
    for g in gammas:
        V = np.eye(n) + g * Z @ Z.T; Vi = np.linalg.inv(V); XtViX = Xv.T @ Vi @ Xv
        b = np.linalg.solve(XtViX, Xv.T @ Vi @ y); r = y - Xv @ b; q = r @ Vi @ r
        _, ld = np.linalg.slogdet(V)
        if reml:
            s2 = q / (n - p); _, ldx = np.linalg.slogdet(XtViX)
            ll = -0.5 * ((n - p) * np.log(2 * np.pi * s2) + ld + ldx + (n - p))
        else:
            s2 = q / n; ll = -0.5 * (n * np.log(2 * np.pi * s2) + ld + n)
        out.append(ll)
    return np.array(out)

def cr2(y, X, cl):
    Xv = X.values; n, p = Xv.shape; XtXi = np.linalg.inv(Xv.T @ Xv); b = XtXi @ Xv.T @ y; e = y - Xv @ b
    H = Xv @ XtXi @ Xv.T; meat = np.zeros((p, p)); A = {}
    for c in np.unique(cl):
        i = np.where(cl == c)[0]; Ic = np.eye(len(i)) - H[np.ix_(i, i)]
        w, U = np.linalg.eigh(Ic); Ac = U @ np.diag(1 / np.sqrt(np.clip(w, 1e-12, None))) @ U.T; A[c] = (i, Ac)
        u = Xv[i].T @ Ac @ e[i]; meat += np.outer(u, u)
    Vb = XtXi @ meat @ XtXi; j = list(X.columns).index(K); se = np.sqrt(Vb[j, j])
    # Bell-McCaffrey / Imbens-Kolesar Satterthwaite df for contrast e_j (working model: independence)
    ell = XtXi[:, j]; Gm = []
    for c, (i, Ac) in A.items():
        M = np.zeros(n); M[i] = Ac @ Xv[i] @ ell; g = M - H @ M  # (I-H)' column for cluster c
        Gm.append(g)
    Gm = np.array(Gm).T; lam = np.linalg.eigvalsh(Gm.T @ Gm); df_ = lam.sum() ** 2 / (lam ** 2).sum()
    return b[j], se, df_

def wild_boot(y, X, cl, B=9999):
    Xv = X.values; j = list(X.columns).index(K); Xr = np.delete(Xv, j, 1)
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]; yr = Xr @ br; er = y - yr
    def tstat(yy):
        XtXi = np.linalg.inv(Xv.T @ Xv); b = XtXi @ Xv.T @ yy; e = yy - Xv @ b; meat = np.zeros((Xv.shape[1],) * 2)
        for c in np.unique(cl):
            i = cl == c; u = Xv[i].T @ e[i]; meat += np.outer(u, u)
        G = len(np.unique(cl)); V = XtXi @ meat @ XtXi * G / (G - 1); return b[j] / np.sqrt(V[j, j])
    t0 = tstat(y); webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
    cls = np.unique(cl); idx = np.searchsorted(cls, cl); ts = np.empty(B)
    for bb in range(B):
        w = rng.choice(webb, len(cls)); ts[bb] = tstat(yr + er * w[idx])
    return t0, (1 + np.sum(np.abs(ts) >= abs(t0))) / (B + 1)

L = ["v58 model diagnostics (review round 4, Q1) — TEST, %s" % pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "=" * 78]
rows = []
for label, sub, clip in (("all55_raw", base, None), ("all55_clip30", base, 30.0), ("r2ge07_raw", base[base.r2_heat >= .7], None)):
    df = sub.copy(); df["logs"] = np.log(df.sigma_heat if clip is None else df.sigma_heat.clip(upper=clip))
    L += ["", "[%s] n = %d maps, %d patients (single %d / double %d patients)" % (label, len(df), df.patient.nunique(),
          df[df.reference == "single"].patient.nunique(), df[df.reference == "double"].patient.nunique())]
    for name, f, reml in (("REML full", "logs ~ reference + region", True), ("ML full", "logs ~ reference + region", False), ("ML reduced", "logs ~ region", False)):
        m, msg, g = fit(f, df, reml)
        L.append("  %-10s converged=%s  RE var=%.5f  resid var=%.4f  |score|max=%.3g  llf=%.4f  warnings: %s" % (name, m.converged, float(m.cov_re.iloc[0, 0]), m.scale, g, m.llf, "; ".join(msg) or "none"))
        rows.append(dict(analysis=label, model=name, converged=m.converged, re_var=float(m.cov_re.iloc[0, 0]), resid_var=m.scale, score_max=g, llf=m.llf, warnings=" | ".join(msg)))
    y, X = design(df, "logs ~ reference + region"); gam = np.concatenate([[0], np.geomspace(1e-4, 2, 60)])
    for reml in (True, False):
        ll = profile(y, X, df.patient.values, gam, reml); k = int(np.argmax(ll))
        L.append("  profile %s log-lik of gamma=tau2/sigma2: max at gamma=%.4g (ll %.4f); ll(0)=%.4f; ll(0.05)=%.4f; ll(0.2)=%.4f; ll(1)=%.4f" % (
            "REML" if reml else "ML  ", gam[k], ll[k], ll[0], np.interp(.05, gam, ll), np.interp(.2, gam, ll), np.interp(1, gam, ll)))
        pd.DataFrame(dict(gamma=gam, loglik=ll)).to_csv(os.path.join(OUT, "v58_profile_%s_%s.csv" % (label, "reml" if reml else "ml")), index=False)
    # ML LRT from closed-form profile maxima (robust to optimiser)
    yr, Xr = design(df, "logs ~ region")
    llf = profile(y, X, df.patient.values, gam, False).max(); llr = profile(yr, Xr, df.patient.values, gam, False).max()
    lr = 2 * (llf - llr); L.append("  LRT reference from profiled ML maxima: chi2=%.3f, p=%.4f" % (lr, stats.chi2.sf(lr, 1)))
    o = smf.ols("logs ~ reference + region", df).fit()
    b, se, dfs = cr2(y, X, df.patient.values); tq = stats.t.ppf(.975, dfs)
    L.append("  OLS coef %.4f (ratio %.2f); CR2 SE %.4f, Satterthwaite df %.2f -> ratio 95%% CI [%.2f; %.2f], p=%.4f" % (
        b, np.exp(b), se, dfs, np.exp(b - tq * se), np.exp(b + tq * se), 2 * stats.t.sf(abs(b / se), dfs)))
    oc = smf.ols("logs ~ reference + region", df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.patient)[0]})
    ci = oc.conf_int().loc[K]; L.append("  OLS CR1 (statsmodels default, normal ref): ratio %.2f [%.2f; %.2f], p=%.4f" % (np.exp(oc.params[K]), np.exp(ci[0]), np.exp(ci[1]), oc.pvalues[K]))
    t0, pw = wild_boot(y, X, df.patient.values); L.append("  wild cluster bootstrap (restricted, Webb, 9,999): t=%.3f, p=%.4f" % (t0, pw))
    # patient level
    df["res"] = smf.ols("logs ~ region", df).fit().resid
    pl = df.groupby(["patient", "reference"], observed=True).agg(m_raw=("logs", "mean"), m_adj=("res", "mean"), n=("logs", "size")).reset_index()
    for col, nm in (("m_raw", "raw"), ("m_adj", "region-adjusted")):
        s = pl[pl.reference == "single"][col]; d = pl[pl.reference == "double"][col]; t = stats.ttest_ind(s, d, equal_var=False)
        dd = s.mean() - d.mean(); se_ = np.sqrt(s.var() / len(s) + d.var() / len(d))
        nu = se_ ** 4 / ((s.var() / len(s)) ** 2 / (len(s) - 1) + (d.var() / len(d)) ** 2 / (len(d) - 1)); q = stats.t.ppf(.975, nu)
        L.append("  patient-level mean log sigma (%s), %d vs %d: ratio of geometric means %.2f [%.2f; %.2f], Welch df %.1f, p=%.4f" % (
            nm, len(s), len(d), np.exp(dd), np.exp(dd - q * se_), np.exp(dd + q * se_), nu, t.pvalue))
    pl.to_csv(os.path.join(OUT, "v58_patient_level_%s.csv" % label), index=False)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "v58_fits.csv"), index=False)
open(os.path.join(OUT, "v58_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))

# (d) all fixed effects at the profiled REML optimum (closed form; replaces the non-converged statsmodels REML estimate),
#     model-based GLS SE with normal and t(G - 2) reference (G = number of patients; reference is a between-patient factor).
#     Outputs v58_terms.csv (one row per analysis x term) and v58_inference.json (ICC, LRT, CR2, wild bootstrap, patient level).
import json
from scipy.optimize import minimize_scalar
L2 = ["", "(d) fixed effects at the profiled REML optimum, t(G-2) reference"]; terms = []; INF = {}
for label, sub, clip in (("all55_raw", base, None), ("all55_clip30", base, 30.0), ("r2ge07_raw", base[base.r2_heat >= .7], None)):
    df = sub.copy(); df["logs"] = np.log(df.sigma_heat if clip is None else df.sigma_heat.clip(upper=clip))
    y, X = design(df, "logs ~ reference + region"); grp = df.patient.values
    r = minimize_scalar(lambda lg: -profile(y, X, grp, [np.exp(lg)], True)[0], bounds=(np.log(1e-6), np.log(5)), method="bounded")
    g = np.exp(r.x); g = 0.0 if profile(y, X, grp, [0], True)[0] >= -r.fun else g
    Xv = X.values; n, p = Xv.shape; G = pd.factorize(grp)[0]; Z = np.zeros((n, G.max() + 1)); Z[np.arange(n), G] = 1
    Vi = np.linalg.inv(np.eye(n) + g * Z @ Z.T); A_ = np.linalg.inv(Xv.T @ Vi @ Xv); b = A_ @ Xv.T @ Vi @ y
    s2 = (y - Xv @ b) @ Vi @ (y - Xv @ b) / (n - p); nc = len(np.unique(grp)); qt = stats.t.ppf(.975, nc - 2)
    for j, name in enumerate(X.columns):
        se = np.sqrt(s2 * A_[j, j]); tt = b[j] / se
        terms.append(dict(analysis=label, term=name, coef=b[j], se=se, ratio=np.exp(b[j]), z_lo=np.exp(b[j] - 1.96 * se), z_hi=np.exp(b[j] + 1.96 * se), p_z=2 * stats.norm.sf(abs(tt)),
                          df_t=nc - 2, t_lo=np.exp(b[j] - qt * se), t_hi=np.exp(b[j] + qt * se), p_t=2 * stats.t.sf(abs(tt), nc - 2)))
    k = [t for t in terms if t["analysis"] == label and t["term"] == K][0]
    yr, Xr = design(df, "logs ~ region"); gam = np.concatenate([[0], np.geomspace(1e-4, 2, 200)])
    lr = 2 * (profile(y, X, grp, gam, False).max() - profile(yr, Xr, grp, gam, False).max())
    bq, se_c, dfs = cr2(y, X, grp); tq = stats.t.ppf(.975, dfs); _, pw = wild_boot(y, X, grp)
    df["res"] = smf.ols("logs ~ region", df).fit().resid
    pl = df.groupby(["patient", "reference"], observed=True).res.mean().reset_index(); sp = pl[pl.reference == "single"].res; dp = pl[pl.reference == "double"].res
    dd = sp.mean() - dp.mean(); se_ = np.sqrt(sp.var() / len(sp) + dp.var() / len(dp)); nu = se_ ** 4 / ((sp.var() / len(sp)) ** 2 / (len(sp) - 1) + (dp.var() / len(dp)) ** 2 / (len(dp) - 1)); qq = stats.t.ppf(.975, nu)
    INF[label] = dict(n_maps=int(n), n_patients=int(nc), n_single=int(df[df.reference == "single"].patient.nunique()), n_double=int(df[df.reference == "double"].patient.nunique()),
                      gamma=float(g), tau2=float(g * s2), sigma2=float(s2), icc=float(g / (1 + g)), lrt_chi2=float(lr), lrt_p=float(stats.chi2.sf(lr, 1)),
                      cr2=dict(ratio=float(np.exp(bq)), lo=float(np.exp(bq - tq * se_c)), hi=float(np.exp(bq + tq * se_c)), df=float(dfs), p=float(2 * stats.t.sf(abs(bq / se_c), dfs))), wild_p=float(pw),
                      patient_adj=dict(ratio=float(np.exp(dd)), lo=float(np.exp(dd - qq * se_)), hi=float(np.exp(dd + qq * se_)), df=float(nu), p=float(stats.ttest_ind(sp, dp, equal_var=False).pvalue)))
    L2.append("  [%s] ICC %.3f; reference ratio %.2f, t(%d) [%.2f; %.2f] p=%.4f (z [%.2f; %.2f] p=%.4f); LRT chi2 %.2f p=%.4f; CR2 [%.2f; %.2f] p=%.4f; wild p=%.4f; patient-level adj %.2f [%.2f; %.2f] p=%.4f" % (
        label, g / (1 + g), k["ratio"], nc - 2, k["t_lo"], k["t_hi"], k["p_t"], k["z_lo"], k["z_hi"], k["p_z"], lr, INF[label]["lrt_p"], INF[label]["cr2"]["lo"], INF[label]["cr2"]["hi"], INF[label]["cr2"]["p"], pw,
        INF[label]["patient_adj"]["ratio"], INF[label]["patient_adj"]["lo"], INF[label]["patient_adj"]["hi"], INF[label]["patient_adj"]["p"]))
pd.DataFrame(terms).to_csv(os.path.join(OUT, "v58_terms.csv"), index=False); json.dump(INF, open(os.path.join(OUT, "v58_inference.json"), "w"), indent=1)
open(os.path.join(OUT, "v58_summary.txt"), "a").write("\n".join(L2) + "\n"); print("\n".join(L2))
