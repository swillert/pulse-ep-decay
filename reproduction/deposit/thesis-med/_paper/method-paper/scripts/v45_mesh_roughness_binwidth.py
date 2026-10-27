#!/usr/bin/env python3
"""v45 — Two sensitivities for the limitations (reviewer 1, limitations 1
and 3): (a) mesh roughness — the clinical mesh is smoothed (Taubin
lambda/mu smoothing, 10 and 30 iterations; vertex scores unchanged) and
the width is recomputed with Heat Method distances from the same argmax
vertex; reported: change of geodesic distances within 30 mm and of sigma.
(b) bin width — sigma with Delta d = 2 mm (submission), 1 mm, 4 mm, and
Delta d = 1.5 x mean edge length of the map (bins with >= 3 vertices).
55 clinical maps, April design.  Outputs: v45_per_map.csv, v45_summary.txt.
TEST only.
"""
import argparse, os
import numpy as np, pandas as pd, trimesh
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from pulse_ep_decay.mesh import HeatSolver, largest_component, mean_edge_length
from pulse_ep_decay.profile import bin_profile, fit_gaussian, find_origin, gauss_model

ROOT = os.path.expanduser("~/mnt/IdeaProjects")
MESHES = os.path.join(ROOT, "pulse-ep/data/validation_meshes")
SUMMARY = os.path.join(ROOT, "pulse-ultimate/decay_profiles_heat/decay_summary.csv")


def fit_bw(d, S, bw):
    ok = np.isfinite(d) & np.isfinite(S); d, S = np.clip(d[ok], 0, None), S[ok]
    k = np.floor(d / bw).astype(int); cnt = np.bincount(k); sm = np.bincount(k, weights=S)
    v = cnt >= 3; c = (np.arange(len(cnt)) + .5) * bw; c, m = c[v], sm[v] / cnt[v]
    g = fit_gaussian(c, m); return g["sigma"], g["r2"], len(c)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    summ = pd.read_csv(SUMMARY).set_index("map_id"); rows = []
    for mid in sorted(summ.index):
        if summ.loc[mid, "max_geodesic_dist"] < 5.0 or mid == 7: continue
        z = np.load(os.path.join(MESHES, "map_%03d.npz" % mid), allow_pickle=True)
        V, F, idx = largest_component(z["vertices"], z["triangles"].astype(int)); S = z["scores"].astype(float)[idx >= 0]
        o = find_origin(S, V, F); h = mean_edge_length(V, F); d0 = HeatSolver(V, F)(o)
        r = dict(map_id=mid, n_vertices=len(V), mean_edge_mm=h, reference=("double" if summ.loc[mid, "ref_catheters"] == 2 else "single"))
        r["sigma_2mm"], r["r2_2mm"], _ = fit_bw(d0, S, 2.0)
        for it in (10, 30):
            m = trimesh.Trimesh(V.copy(), F.copy(), process=False); trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=0.53, iterations=it)
            Vs = np.asarray(m.vertices); ds = HeatSolver(Vs, F)(o)
            near = (d0 <= 30) & (d0 > 2)
            r["drel_%d" % it] = float(np.median((ds[near] - d0[near]) / d0[near])); r["drel_p05_%d" % it] = float(np.percentile((ds[near] - d0[near]) / d0[near], 5))
            r["sigma_smooth%d" % it], r["r2_smooth%d" % it], _ = fit_bw(ds, S, 2.0)
            r["disp_mm_%d" % it] = float(np.linalg.norm(Vs - V, axis=1).mean())
        for bw, tag in ((1.0, "1mm"), (4.0, "4mm"), (1.5 * h, "1p5h")):
            r["sigma_" + tag], r["r2_" + tag], r["nbins_" + tag] = fit_bw(d0, S, bw)
        rows.append(r)
        print("map %3d h %.2f sigma 2mm %5.1f | smooth10 %5.1f (d %+.3f) smooth30 %5.1f (d %+.3f) | 1mm %5.1f 4mm %5.1f 1.5h(%.1f) %5.1f" % (
            mid, h, r["sigma_2mm"], r["sigma_smooth10"], r["drel_10"], r["sigma_smooth30"], r["drel_30"], r["sigma_1mm"], r["sigma_4mm"], 1.5 * h, r["sigma_1p5h"]), flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "v45_per_map.csv"), index=False)
    foc = df[(df.r2_2mm >= .7) & (df.sigma_2mm < 30)]
    L = ["v45 mesh roughness and bin width — summary (55 clinical maps)", "=" * 60, ""]
    for it in (10, 30):
        d = df["sigma_smooth%d" % it] - df.sigma_2mm; df_ = foc["sigma_smooth%d" % it] - foc.sigma_2mm
        L.append("Taubin smoothing %2d it.: mean vertex displacement median %.2f mm; geodesic distances within 30 mm change by median %+.1f%% (per-map medians, IQR %+.1f .. %+.1f%%); sigma change median %+.2f mm [IQR %+.2f; %+.2f], |d|>1 mm in %d/55; focal subset (n=%d): %+.2f [%+.2f; %+.2f]" % (
            it, df["disp_mm_%d" % it].median(), 100 * df["drel_%d" % it].median(), 100 * df["drel_%d" % it].quantile(.25), 100 * df["drel_%d" % it].quantile(.75), d.median(), d.quantile(.25), d.quantile(.75), (d.abs() > 1).sum(), len(foc), df_.median(), df_.quantile(.25), df_.quantile(.75)))
    L.append("")
    for tag, lab in (("1mm", "Delta d = 1 mm"), ("4mm", "Delta d = 4 mm"), ("1p5h", "Delta d = 1.5 x mean edge (%.1f-%.1f mm)" % (1.5 * df.mean_edge_mm.min(), 1.5 * df.mean_edge_mm.max()))):
        d = df["sigma_" + tag] - df.sigma_2mm; df_ = foc["sigma_" + tag] - foc.sigma_2mm
        L.append("%-40s: sigma change vs 2 mm median %+.2f mm [IQR %+.2f; %+.2f], mean %+.2f, |d|>1 mm in %d/55; focal subset %+.2f [%+.2f; %+.2f]; rho %.3f; R2 median %.3f (2 mm: %.3f)" % (
            lab, d.median(), d.quantile(.25), d.quantile(.75), d.mean(), (d.abs() > 1).sum(), df_.median(), df_.quantile(.25), df_.quantile(.75), spearmanr(df["sigma_" + tag], df.sigma_2mm).correlation, df["r2_" + tag].median(), df.r2_2mm.median()))
    txt = "\n".join(L); print(txt); open(os.path.join(a.out, "v45_summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
