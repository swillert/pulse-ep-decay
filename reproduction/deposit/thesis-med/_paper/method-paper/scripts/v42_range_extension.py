#!/usr/bin/env python3
"""v42 — Operating range of the sigma estimator outside 3-30 mm
(reviewer 2, point 2) with the independent design of validation arm 1:
synthetic Gaussian fields generated on EXACT polyhedral geodesics
(pygeodesic) from the centroid-nearest vertex of each of the 56 clinical
meshes, evaluated with the submitted pipeline (argmax origin -> Heat
Method distances from the detected origin -> 2-mm bins (>= 3 vertices) ->
three-parameter Gaussian, sigma in [0.1, 200] mm).  Noise/clip/mask
protocol of the manuscript (A = 35, B = 65, N(0, (eta A)^2), clip
[50, 100], 5 % of vertices masked).

sigma0 in {1, 1.5, 2, 2.5, 3, 30, 40, 50, 60, 80} mm, eta in {0, 5, 10, 15} %,
20 trials -> 56 x 10 x 4 x 20 = 44,800 fits.  Reported per (sigma0, eta):
mean and median bias, SD, 5th/95th percentile of sigma_hat, fraction at the
upper bound (>= 199 mm), fraction with sigma_hat > 100 mm ("not
identifiable"), median origin offset (exact distance of the detected
argmax from the true origin), mean R^2, median number of bins.

Resumable per mesh: --maps <ids> writes parts/<id>.csv; --assemble builds
v42_raw.csv, v42_summary.csv, v42_summary.txt.  TEST only.
"""
import argparse, os, glob
import numpy as np, pandas as pd
from pulse_ep_decay.mesh import HeatSolver, ExactSolver, largest_component, centroid_vertex, mean_edge_length
from pulse_ep_decay.profile import bin_profile, fit_gaussian, find_origin

ROOT = os.environ.get("GD_ROOT", os.path.expanduser("~/mnt/IdeaProjects"))
MESHES = os.path.join(ROOT, "pulse-ep/data/validation_meshes")
SIGMAS = [1.0, 1.5, 2.0, 2.5, 3.0, 30.0, 40.0, 50.0, 60.0, 80.0]
NOISES = [0.0, 0.05, 0.10, 0.15]
TRIALS = 20
A, B = 35.0, 65.0


def run_mesh(f, out):
    z = np.load(f, allow_pickle=True); mid = int(z["map_id"])
    V, T, _ = largest_component(z["vertices"], z["triangles"].astype(int))
    if len(V) < 50 or np.isnan(z["scores"]).all():
        return
    v0 = centroid_vertex(V); d0 = ExactSolver(V, T)(v0); d0 = np.where(np.isfinite(d0), d0, np.nan)
    heat = HeatSolver(V, T); h = mean_edge_length(V, T)
    rows = []
    for s0 in SIGMAS:
        for eta in NOISES:
            for t in range(TRIALS):
                rng = np.random.default_rng(1_000_000 * mid + int(1000 * s0) + int(100 * eta * 100) + t)
                S = A * np.exp(-np.nan_to_num(d0) ** 2 / (2 * s0 ** 2)) + B
                if eta > 0: S = S + rng.normal(0, eta * A, len(S))
                S = np.clip(S, 50, 100); S[np.isnan(d0)] = np.nan
                S[rng.random(len(S)) < 0.05] = np.nan
                o = find_origin(S, V, T); d = heat(o)
                c, m, _ = bin_profile(d, S); g = fit_gaussian(c, m)
                rows.append(dict(map_id=mid, n_vertices=len(V), mean_edge_mm=h, sigma0=s0, noise=eta, trial=t, sigma_hat=g["sigma"], r2=g["r2"],
                                 n_bins=len(c), origin_offset_mm=float(d0[o]), converged=g["converged"]))
    pd.DataFrame(rows).to_csv(os.path.join(out, "parts", "%03d.csv" % mid), index=False)
    print("map %3d done (%d vertices)" % (mid, len(V)), flush=True)


FRAGMENTED = {87, 102, 103}  # excluded from the validation set of the submission (56 meshes)


def assemble(out):
    df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(out, "parts", "*.csv")))])
    df = df[~df.map_id.isin(FRAGMENTED)]
    df.to_csv(os.path.join(out, "v42_raw.csv"), index=False)
    rows = []
    for (s0, eta), g in df.groupby(["sigma0", "noise"]):
        sh = g.sigma_hat; ok = sh.notna()
        rows.append(dict(sigma0=s0, noise=eta, n=len(g), n_meshes=g.map_id.nunique(), sigma_hat_mean=sh[ok].mean(), sigma_hat_median=sh[ok].median(),
                         bias_mean=(sh[ok] - s0).mean(), bias_median=(sh[ok] - s0).median(), sd=sh[ok].std(), p5=sh[ok].quantile(.05), p95=sh[ok].quantile(.95),
                         frac_at_bound=(sh >= 199).mean(), frac_gt_100=(sh > 100).mean(), frac_failed=(~ok).mean(), origin_offset_median=g.origin_offset_mm.median(),
                         origin_offset_p90=g.origin_offset_mm.quantile(.9), r2_mean=g.r2.mean(), n_bins_median=g.n_bins.median()))
    s = pd.DataFrame(rows); s.to_csv(os.path.join(out, "v42_summary.csv"), index=False)
    L = ["v42 operating range outside 3-30 mm (exact-geodesic truth, argmax + Heat Method pipeline) — summary", "=" * 90,
         "%d fits on %d meshes; %d trials per (mesh, sigma0, noise)" % (len(df), df.map_id.nunique(), TRIALS), "",
         "  sigma0  noise      n  mean sigma_hat  median  bias(mean)  bias(median)      SD     p5    p95  at bound  >100 mm  offset med/p90 (mm)   R2   bins"]
    for _, r in s.iterrows():
        L.append("  %6.1f  %5.2f  %5d  %14.2f  %6.2f  %10.2f  %12.2f  %6.2f  %5.2f  %5.2f  %8.3f  %7.3f  %7.1f / %5.1f     %.3f  %4.0f" % (
            r.sigma0, r.noise, r.n, r.sigma_hat_mean, r.sigma_hat_median, r.bias_mean, r.bias_median, r.sd, r.p5, r.p95, r.frac_at_bound, r.frac_gt_100,
            r.origin_offset_median, r.origin_offset_p90, r.r2_mean, r.n_bins_median))
    txt = "\n".join(L); print(txt); open(os.path.join(out, "v42_summary.txt"), "w").write(txt + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--maps", type=int, nargs="*"); ap.add_argument("--assemble", action="store_true")
    a = ap.parse_args(); os.makedirs(os.path.join(a.out, "parts"), exist_ok=True)
    if a.assemble:
        assemble(a.out); return
    files = sorted(glob.glob(os.path.join(MESHES, "map_*.npz")))
    for f in files:
        mid = int(os.path.basename(f)[4:7])
        if a.maps and mid not in a.maps: continue
        if os.path.exists(os.path.join(a.out, "parts", "%03d.csv" % mid)): continue
        run_mesh(f, a.out)


if __name__ == "__main__":
    main()
