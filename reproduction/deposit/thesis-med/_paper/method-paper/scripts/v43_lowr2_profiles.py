#!/usr/bin/env python3
"""v43 — What the maps with R^2 < 0.7 look like (reviewer 3, point 3;
reviewer 2, point 6; reviewer 1, limitation 2).

For each of the 55 clinical maps (April design, Heat Method, argmax):
fitted A, B, sigma, R^2; peak contrast A/(bin-mean range); the radial
profile beyond 3 sigma (and beyond 20 mm): its SD, its maximum relative to
the peak (secondary bump), the fraction of the binned variance that lies
beyond 3 sigma; the within-bin SD of vertex scores in the first 30 mm
(noise level as used in the manuscript, in % of A).  Also the
generalised-Gaussian exponent beta and the exponential fit (Delta AIC),
re-computed with the package functions, to confirm clinical_metrics.csv.
Outputs: v43_per_map.csv, v43_summary.txt.  TEST only.
"""
import argparse, os
import numpy as np, pandas as pd
from pulse_ep_decay.mesh import HeatSolver, largest_component
from pulse_ep_decay.profile import bin_profile, fit_gaussian, fit_exponential, fit_gengauss, find_origin, gauss_model, fwhm_from

ROOT = os.environ.get("GD_ROOT", os.path.expanduser("~/mnt/IdeaProjects"))
MESHES = os.path.join(ROOT, "pulse-ep/data/validation_meshes")
SUMMARY = os.path.join(ROOT, "pulse-ultimate/decay_profiles_heat/decay_summary.csv")
STUDY = os.path.join(ROOT, "thesis-med/_paper/method-paper/supplementary_validation_results/map_study.csv")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    summ = pd.read_csv(SUMMARY).set_index("map_id"); study = pd.read_csv(STUDY).set_index("map_id")["study_id"]
    rows = []
    for mid in sorted(summ.index):
        if summ.loc[mid, "max_geodesic_dist"] < 5.0 or mid == 7: continue
        z = np.load(os.path.join(MESHES, "map_%03d.npz" % mid), allow_pickle=True)
        V, F, idx = largest_component(z["vertices"], z["triangles"].astype(int)); S = z["scores"].astype(float)[idx >= 0]
        o = find_origin(S, V, F); d = HeatSolver(V, F)(o)
        c, m, cnt = bin_profile(d, S); g = fit_gaussian(c, m); e = fit_exponential(c, m)
        try:
            gg = fit_gengauss(c, m); beta = gg.get("beta", np.nan); fwhm_gg = fwhm_from(gg["alpha"], gg["beta"]) if np.isfinite(gg.get("alpha", np.nan)) else np.nan
        except Exception:
            beta, fwhm_gg = np.nan, np.nan
        s = g["sigma"]; yhat = gauss_model(c, g["A"], s, g["B"])
        far = c > max(3 * s, 6.0); near = c <= 30
        ss_tot = ((m - m.mean()) ** 2).sum()
        # within-bin SD of vertex scores (first 30 mm), in % of A
        k = np.floor(np.clip(d, 0, None) / 2.0).astype(int); ok = np.isfinite(S) & (d <= 30)
        wsd = np.nanmean([S[ok & (k == kk)].std() for kk in np.unique(k[ok]) if (ok & (k == kk)).sum() >= 3])
        # model-free FWHM from the binned profile: first crossing of B + A/2 (B = far-field median)
        Bmf = np.median(m[far]) if far.sum() >= 3 else m.min(); half = Bmf + (m.max() - Bmf) / 2
        below = np.where(m < half)[0]; fwhm_mf = 2 * c[below[0]] if len(below) else np.nan
        rows.append(dict(map_id=mid, study_id=int(study.get(mid, -1)), reference=("double" if summ.loc[mid, "ref_catheters"] == 2 else "single"),
                         region=summ.loc[mid, "atrium"] + " " + summ.loc[mid, "part"].replace(" wall", ""), n_points=int(summ.loc[mid, "n_points"]),
                         sigma=s, r2=g["r2"], A=g["A"], B=g["B"], peak_bin=m.max(), range_bins=m.max() - m.min(), n_bins=len(c), dmax_bin=c.max(),
                         far_sd=m[far].std() if far.sum() >= 2 else np.nan, far_max_rel=(m[far].max() - g["B"]) / g["A"] if far.sum() else np.nan,
                         far_var_frac=((m[far] - m.mean()) ** 2).sum() / ss_tot if ss_tot > 0 else np.nan,
                         resid_far_frac=((m[far] - yhat[far]) ** 2).sum() / max(((m - yhat) ** 2).sum(), 1e-12),
                         withinbin_sd_pct_A=100 * wsd / g["A"] if g["A"] > 0 else np.nan, withinbin_sd_pts=wsd,
                         lam_exp=e["lam"], r2_exp=e["r2"], dAIC_gauss_minus_exp=g["aic"] - e["aic"], beta=beta, fwhm_gg=fwhm_gg, fwhm_gauss=2.3548 * s, fwhm_modelfree=fwhm_mf))
        print("map %3d sigma %5.1f R2 %.2f A %4.1f B %4.1f far_sd %4.1f far_max_rel %+.2f resid_far %.2f wsd %.0f%% beta %.2f dAIC %+.1f" % (
            mid, s, g["r2"], g["A"], g["B"], rows[-1]["far_sd"], rows[-1]["far_max_rel"], rows[-1]["resid_far_frac"], rows[-1]["withinbin_sd_pct_A"], beta, rows[-1]["dAIC_gauss_minus_exp"]), flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "v43_per_map.csv"), index=False)
    lo = df[df.r2 < 0.7]; hi = df[df.r2 >= 0.7]
    L = ["v43 profiles of the maps with R2 < 0.7 — summary", "=" * 50, "n = %d maps; R2 < 0.7 in %d" % (len(df), len(lo)), ""]
    L.append("A. The low-R2 maps: sigma median %.1f [%.1f; %.1f] (max %.1f); n(sigma>20) = %d; n_points median %d; A median %.1f pts (high-R2 maps: sigma %.1f [%.1f; %.1f], A median %.1f)" % (
        lo.sigma.median(), lo.sigma.quantile(.25), lo.sigma.quantile(.75), lo.sigma.max(), (lo.sigma > 20).sum(), lo.n_points.median(), lo.A.median(), hi.sigma.median(), hi.sigma.quantile(.25), hi.sigma.quantile(.75), hi.A.median()))
    L.append("   fraction of the squared residual beyond 3 sigma: low-R2 median %.2f, high-R2 median %.2f" % (lo.resid_far_frac.median(), hi.resid_far_frac.median()))
    L.append("   far-field SD of bin means (beyond 3 sigma): low-R2 median %.1f pts, high-R2 %.1f pts; far maximum relative to A: low %.2f, high %.2f" % (lo.far_sd.median(), hi.far_sd.median(), lo.far_max_rel.median(), hi.far_max_rel.median()))
    L.append("   within-bin SD of vertex scores (first 30 mm): all maps median %.0f%% of A [IQR %.0f; %.0f] = %.1f score points; low-R2 %.0f%%, high-R2 %.0f%%" % (
        df.withinbin_sd_pct_A.median(), df.withinbin_sd_pct_A.quantile(.25), df.withinbin_sd_pct_A.quantile(.75), df.withinbin_sd_pts.median(), lo.withinbin_sd_pct_A.median(), hi.withinbin_sd_pct_A.median()))
    L += ["", "B. Per map (R2 < 0.7):", "   map ref    region        pts sigma   R2    A     B  far_sd far_max_rel resid_far  wsd%A  beta  dAIC(G-E)"]
    for _, r in lo.sort_values("r2").iterrows():
        L.append("   %3d %-6s %-13s %3d %5.1f %.2f %5.1f %5.1f %6.1f %+10.2f %9.2f %6.0f %5.2f %+8.1f" % (r.map_id, r.reference, r.region, r.n_points, r.sigma, r.r2, r.A, r.B, r.far_sd, r.far_max_rel, r.resid_far_frac, r.withinbin_sd_pct_A, r.beta, r.dAIC_gauss_minus_exp))
    L += ["", "C. Model comparison on all 55 maps (binned profiles): dAIC = AIC_gauss - AIC_exp; Gaussian preferred (dAIC < -2) in %d, exponential (dAIC > 2) in %d, undecided in %d" % (
        (df.dAIC_gauss_minus_exp < -2).sum(), (df.dAIC_gauss_minus_exp > 2).sum(), (df.dAIC_gauss_minus_exp.abs() <= 2).sum())]
    L.append("   generalised Gaussian exponent beta: median %.2f [%.2f; %.2f] (n with beta at a bound: %d); R2>=0.7 subset: beta median %.2f" % (
        df.beta.median(), df.beta.quantile(.25), df.beta.quantile(.75), ((df.beta <= 0.31) | (df.beta >= 3.99)).sum(), hi.beta.median()))
    r = (df.fwhm_modelfree / df.fwhm_gauss)
    L.append("   model-free FWHM / Gaussian FWHM (2.355 sigma): median %.2f [%.2f; %.2f] (all); R2>=0.7: %.2f [%.2f; %.2f]; Spearman rho(sigma, lambda_exp) = %.3f" % (
        r.median(), r.quantile(.25), r.quantile(.75), r[df.r2 >= .7].median(), r[df.r2 >= .7].quantile(.25), r[df.r2 >= .7].quantile(.75), df[["sigma", "lam_exp"]].corr("spearman").iloc[0, 1]))
    txt = "\n".join(L); print(txt); open(os.path.join(a.out, "v43_summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
