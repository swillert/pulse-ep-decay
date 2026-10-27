#!/usr/bin/env python3
"""v46 — The pacing site marked during acquisition ("Location Only", LO) as
an independent reference for the origin (decision 21 Oct 2026, Sven).

Part A (clinical, April vertex field, Heat Method, 2-mm bins >= 3 vertices,
Gaussian fit): for the 46 maps with a unique LO marker
(lo_selection_2026-09-14/selection.json, route 'unique_lo_reference_candidate';
coordinates from the raw CARTO export, verified 14 Sept): nearest mesh vertex
to the marked site (snap distance); sigma and R2 with that vertex as origin
(sigma_lo); sigma and R2 with the argmax origin (as submitted); geodesic
distance argmax -> marked site (localisation offset of the displayed
maximum); vertex score at the marked site relative to the maximum.  The nine
maps without a unique marker are listed with the argmax result only.

Part B (validation, 56 meshes, exact-geodesic truth as in arm 1 / v42):
origin FIXED at the generating vertex v0 (known-source mode), Heat-Method
distances from v0, manuscript noise/clip/mask protocol, sigma0 in
{3,5,8,10,15,20,25,30,40,50} mm, eta in {0,5,10,15} %, 20 trials; resumable
per mesh (parts/).

Outputs: v46_clinical.csv, v46_validation_raw.csv, v46_validation_summary.csv,
v46_summary.txt.  TEST only; nothing is written into the manuscript.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from pulse_ep_decay.mesh import HeatSolver, ExactSolver, largest_component, centroid_vertex, mean_edge_length
from pulse_ep_decay.profile import bin_profile, fit_gaussian, find_origin

ROOT = os.path.expanduser("~/mnt/IdeaProjects")
MESHES = os.path.join(ROOT, "pulse-ep/data/validation_meshes")
SVR = os.path.join(ROOT, "thesis-med/_paper/method-paper/supplementary_validation_results")
SUMMARY = os.path.join(ROOT, "pulse-ultimate/decay_profiles_heat/decay_summary.csv")
SELECTION = os.path.join(SVR, "lo_selection_2026-09-14/selection.json")
FRAGMENTED = {87, 102, 103}; EXCLUDE_QUALITY = {7}
A, B = 35.0, 65.0
SIGMAS = (3, 5, 8, 10, 15, 20, 25, 30, 40, 50); NOISES = (0.0, 0.05, 0.10, 0.15); TRIALS = 20


def fit(d, S):
    c, m, _ = bin_profile(d, S); g = fit_gaussian(c, m); return g["sigma"], g["r2"], len(c)


def part_a(out):
    summ = pd.read_csv(SUMMARY).set_index("map_id")
    sel = {r["map_id"]: r for r in json.load(open(SELECTION))}
    rows = []
    for mid in sorted(summ.index):
        if mid in FRAGMENTED or mid in EXCLUDE_QUALITY: continue
        z = np.load(os.path.join(MESHES, "map_%03d.npz" % mid), allow_pickle=True)
        V, F, idx = largest_component(z["vertices"], z["triangles"].astype(int)); S = z["scores"].astype(float)[idx >= 0]
        heat = HeatSolver(V, F); o = find_origin(S, V, F); d_arg = heat(o)
        r = dict(map_id=mid, study_id=sel[mid]["study_id"], reference=("double" if summ.loc[mid, "ref_catheters"] == 2 else "single"),
                 region=summ.loc[mid, "atrium"] + " " + summ.loc[mid, "part"].replace(" wall", ""), n_points=int(summ.loc[mid, "n_points"]),
                 n_vertices=len(V), mean_edge_mm=mean_edge_length(V, F), argmax=o, score_max=float(S[o]), n_lo=sel[mid]["n_lo"], route=sel[mid]["route"])
        r["sigma_argmax"], r["r2_argmax"], _ = fit(d_arg, S)
        if sel[mid]["route"] != "unique_lo_reference_candidate":
            rows.append(r); print("map %3d: no unique marker (%s)" % (mid, sel[mid]["route"])); continue
        xyz = np.asarray(sel[mid]["selected_xyz_mm"], float); r["lo_point_id"] = sel[mid]["selected_point_id"]
        v_lo = int(np.argmin(np.linalg.norm(V - xyz, axis=1))); r["lo_vertex"] = v_lo; r["snap_mm"] = float(np.linalg.norm(V[v_lo] - xyz))
        d_lo = heat(v_lo); r["sigma_lo"], r["r2_lo"], _ = fit(d_lo, S)
        r["offset_geo_mm"] = float(d_lo[o]); r["offset_eucl_mm"] = float(np.linalg.norm(V[o] - V[v_lo]))
        r["score_at_lo"] = float(S[v_lo]); r["score_at_lo_rel"] = float(S[v_lo] / S[o])
        rows.append(r)
        print("map %3d %-6s sigma argmax %5.1f (R2 %.2f) | LO %5.1f (R2 %.2f) | offset %5.1f mm | score at LO %.1f of %.1f | snap %.1f" % (
            mid, r["reference"], r["sigma_argmax"], r["r2_argmax"], r["sigma_lo"], r["r2_lo"], r["offset_geo_mm"], r["score_at_lo"], r["score_max"], r["snap_mm"]), flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "v46_clinical.csv"), index=False); return df


def part_b_mesh(f, out):
    z = np.load(f, allow_pickle=True); mid = int(z["map_id"])
    if mid in FRAGMENTED or os.path.exists(os.path.join(out, "parts", "%03d.csv" % mid)): return
    V, T, _ = largest_component(z["vertices"], z["triangles"].astype(int))
    v0 = centroid_vertex(V); d0 = ExactSolver(V, T)(v0); d0 = np.where(np.isfinite(d0), d0, np.nan); dh = HeatSolver(V, T)(v0)
    rows = []
    for s0 in SIGMAS:
        for eta in NOISES:
            for t in range(TRIALS):
                rng = np.random.default_rng(1_000_000 * mid + int(1000 * s0) + int(100 * eta * 100) + t)
                S = A * np.exp(-np.nan_to_num(d0) ** 2 / (2 * s0 ** 2)) + B
                if eta > 0: S = S + rng.normal(0, eta * A, len(S))
                S = np.clip(S, 50, 100); S[np.isnan(d0)] = np.nan; S[rng.random(len(S)) < 0.05] = np.nan
                sh, r2, nb = fit(dh, S)
                rows.append(dict(map_id=mid, sigma0=s0, noise=eta, trial=t, sigma_hat=sh, r2=r2, n_bins=nb))
    pd.DataFrame(rows).to_csv(os.path.join(out, "parts", "%03d.csv" % mid), index=False); print("validation: map %3d done" % mid, flush=True)


def summarise(out):
    L = ["v46 marked pacing site as origin reference — summary", "=" * 55]
    ca = os.path.join(out, "v46_clinical.csv")
    if os.path.exists(ca):
        df = pd.read_csv(ca); ok = df.dropna(subset=["sigma_lo"]); d = ok.sigma_lo - ok.sigma_argmax
        L += ["", "A. Clinical (April vertex field): %d maps with a unique marker of %d; 9 without (6 none, 3 two markers) listed with argmax only" % (len(ok), len(df))]
        L.append("  sigma_LO median %.2f [%.2f; %.2f]; sigma_argmax (same 46) %.2f [%.2f; %.2f]; LO - argmax median %+.2f [%+.2f; %+.2f]; |d| > 1 mm in %d, > 3 mm in %d; rho %.3f" % (
            ok.sigma_lo.median(), ok.sigma_lo.quantile(.25), ok.sigma_lo.quantile(.75), ok.sigma_argmax.median(), ok.sigma_argmax.quantile(.25), ok.sigma_argmax.quantile(.75),
            d.median(), d.quantile(.25), d.quantile(.75), (d.abs() > 1).sum(), (d.abs() > 3).sum(), spearmanr(ok.sigma_lo, ok.sigma_argmax).correlation))
        L.append("  R2_LO median %.3f (n < 0.7: %d); R2_argmax %.3f (n < 0.7: %d)" % (ok.r2_lo.median(), (ok.r2_lo < .7).sum(), ok.r2_argmax.median(), (ok.r2_argmax < .7).sum()))
        o = ok.offset_geo_mm
        L.append("  offset argmax -> marked site (geodesic): median %.1f mm [%.1f; %.1f], max %.1f; <= 5 mm in %d, > 10 mm in %d, > 20 mm in %d of %d; snap distance median %.1f mm (max %.1f); score at marked site / max: median %.2f, min %.2f" % (
            o.median(), o.quantile(.25), o.quantile(.75), o.max(), (o <= 5).sum(), (o > 10).sum(), (o > 20).sum(), len(ok), ok.snap_mm.median(), ok.snap_mm.max(), ok.score_at_lo_rel.median(), ok.score_at_lo_rel.min()))
        L.append("  offset vs sigma_argmax: Spearman rho %.3f (p %.3f); offset vs R2_argmax: rho %.3f; offset vs n_points: rho %.3f" % (
            spearmanr(o, ok.sigma_argmax).correlation, spearmanr(o, ok.sigma_argmax).pvalue, spearmanr(o, ok.r2_argmax).correlation, spearmanr(o, ok.n_points).correlation))
        for lab, m in (("sigma_argmax < 10 mm", ok.sigma_argmax < 10), ("10-20 mm", (ok.sigma_argmax >= 10) & (ok.sigma_argmax < 20)), (">= 20 mm", ok.sigma_argmax >= 20),
                       ("R2_argmax >= 0.7", ok.r2_argmax >= .7), ("R2_argmax < 0.7", ok.r2_argmax < .7)):
            g = ok[m]
            if len(g): L.append("    %-20s n=%2d: offset median %.1f mm [%.1f; %.1f], > 10 mm in %d, max %.1f" % (lab, len(g), g.offset_geo_mm.median(), g.offset_geo_mm.quantile(.25), g.offset_geo_mm.quantile(.75), (g.offset_geo_mm > 10).sum(), g.offset_geo_mm.max()))
        for ref, g in ok.groupby("reference"):
            L.append("  %-6s n=%2d: sigma_LO median %.2f [%.2f; %.2f], sigma_argmax %.2f, offset median %.1f mm (> 10 mm: %d), n(R2_LO<0.7)=%d" % (
                ref, len(g), g.sigma_lo.median(), g.sigma_lo.quantile(.25), g.sigma_lo.quantile(.75), g.sigma_argmax.median(), g.offset_geo_mm.median(), (g.offset_geo_mm > 10).sum(), (g.r2_lo < .7).sum()))
        L += ["  per map (offset > 10 mm):"]
        for _, r in ok[ok.offset_geo_mm > 10].sort_values("offset_geo_mm", ascending=False).iterrows():
            L.append("    map %3d %-6s %-13s pts %3d  sigma argmax %5.1f (R2 %.2f)  LO %5.1f (R2 %.2f)  offset %5.1f mm  score at LO %.2f of max" % (
                r.map_id, r.reference, r.region, r.n_points, r.sigma_argmax, r.r2_argmax, r.sigma_lo, r.r2_lo, r.offset_geo_mm, r.score_at_lo_rel))
    parts = sorted(glob.glob(os.path.join(out, "parts", "*.csv")))
    if parts:
        raw = pd.concat([pd.read_csv(p) for p in parts]); raw.to_csv(os.path.join(out, "v46_validation_raw.csv"), index=False)
        s = raw.groupby(["sigma0", "noise"]).apply(lambda g: pd.Series(dict(n=len(g), n_meshes=g.map_id.nunique(), bias_mean=(g.sigma_hat - g.sigma0).mean(), bias_median=(g.sigma_hat - g.sigma0).median(),
            sd=g.sigma_hat.std(), p5=g.sigma_hat.quantile(.05), p95=g.sigma_hat.quantile(.95), frac_gt_100=(g.sigma_hat > 100).mean(), r2_mean=g.r2.mean()))).reset_index()
        s.to_csv(os.path.join(out, "v46_validation_summary.csv"), index=False)
        L += ["", "B. Validation, known-source mode (origin fixed at v0; exact-geodesic truth; Heat Method from v0): %d fits on %d meshes" % (len(raw), raw.map_id.nunique())]
        for _, r in s.iterrows():
            L.append("  sigma0 %5.1f noise %.2f n %4d  bias mean %+6.2f median %+6.2f  SD %6.2f  p5 %6.2f p95 %6.2f  >100 mm %.3f  R2 %.3f" % (
                r.sigma0, r.noise, r.n, r.bias_mean, r.bias_median, r.sd, r.p5, r.p95, r.frac_gt_100, r.r2_mean))
    txt = "\n".join(L); print(txt); open(os.path.join(out, "v46_summary.txt"), "w").write(txt + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--part", choices=["a", "b", "summary"], required=True)
    a = ap.parse_args(); os.makedirs(os.path.join(a.out, "parts"), exist_ok=True)
    if a.part == "a": part_a(a.out)
    elif a.part == "b":
        for f in sorted(glob.glob(os.path.join(MESHES, "map_*.npz"))): part_b_mesh(f, a.out)
    summarise(a.out)


if __name__ == "__main__":
    main()
