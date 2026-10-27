#!/usr/bin/env python3
"""v38 — Width of the 55 clinical maps under four distance definitions
(reviewer 3, point 1, last sentence; reviewer 1, statistics point 3;
reviewer 2, point 5).

Design of the April 2026 submission (vertex field of the displayed map,
argmax origin, 2-mm bins with >= 3 vertices, three-parameter Gaussian,
sigma in [0.1, 200] mm) run with the distances of
  heat     : Heat Method (potpourri3d; the manuscript's solver)
  exact    : exact polyhedral geodesics (pygeodesic, MMP)
  dijkstra : shortest paths on the mesh edge graph (scipy)
  euclid   : straight-line distance in R^3
from the SAME argmax origin, so that only the distance definition differs.
Also reported per map: the error of heat / dijkstra / euclid against the
exact geodesic from that origin, within 30 mm and over the whole mesh.

Inputs: pulse-ep/data/validation_meshes/map_<id>.npz (vertices, triangles,
scores), pulse-ultimate/decay_profiles_heat/decay_summary.csv (reference
catheters, region, fragment flag via max_geodesic_dist), map_study.csv.
Outputs: v38_per_map.csv, v38_summary.txt.  TEST only; nothing is written
into the manuscript.
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr
from pulse_ep_decay.mesh import HeatSolver, ExactSolver, largest_component, mean_edge_length
from pulse_ep_decay.profile import bin_profile, fit_gaussian, find_origin

ROOT = os.path.expanduser("~/mnt/IdeaProjects")
MESHES = os.path.join(ROOT, "pulse-ep/data/validation_meshes")
SUMMARY = os.path.join(ROOT, "pulse-ultimate/decay_profiles_heat/decay_summary.csv")
STUDY = os.path.join(ROOT, "thesis-med/_paper/method-paper/supplementary_validation_results/map_study.csv")
EXCLUDE_QUALITY = {7}  # low-quality map excluded in the submission (12 points, max score 68 %)


def edge_graph(V, F):
    e = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    w = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    n = len(V)
    return coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()


def fit(d, S):
    c, m, _ = bin_profile(d, S)
    g = fit_gaussian(c, m)
    return g["sigma"], g["r2"], len(c)


def err(d, truth, lim=None):
    ok = np.isfinite(truth) & np.isfinite(d) & (truth > 2.0)
    if lim is not None:
        ok &= truth <= lim
    rel = (d[ok] - truth[ok]) / truth[ok]
    return float(rel.mean()), float(np.percentile(np.abs(rel), 95)), float(np.abs(d[ok] - truth[ok]).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summ = pd.read_csv(SUMMARY).set_index("map_id")
    study = pd.read_csv(STUDY).set_index("map_id")["study_id"]
    rows = []
    for mid in sorted(summ.index):
        f = os.path.join(MESHES, "map_%03d.npz" % mid)
        if not os.path.exists(f):
            print("missing", f); continue
        z = np.load(f, allow_pickle=True)
        V0, F0, S0 = z["vertices"], z["triangles"].astype(int), z["scores"].astype(float)
        V, F, idx = largest_component(V0, F0)
        S = S0[idx >= 0]
        r = dict(map_id=mid, study_id=int(study.get(mid, -1)), atrium=summ.loc[mid, "atrium"], region=summ.loc[mid, "part"],
                 reference=("double" if summ.loc[mid, "ref_catheters"] == 2 else "single"), n_points=int(summ.loc[mid, "n_points"]),
                 n_vertices=len(V0), n_largest=len(V), mean_edge_mm=mean_edge_length(V, F),
                 fragmented=bool(summ.loc[mid, "max_geodesic_dist"] < 5.0), excluded_quality=mid in EXCLUDE_QUALITY)
        if r["fragmented"]:
            rows.append(r); print("map %3d fragmented, skipped" % mid); continue
        o = find_origin(S, V, F); r["origin"] = o; r["origin_score"] = float(S[o])
        t = time.time(); dh = HeatSolver(V, F)(o); r["t_heat_s"] = time.time() - t
        t = time.time(); de = ExactSolver(V, F)(o); r["t_exact_s"] = time.time() - t
        t = time.time(); dd = dijkstra(edge_graph(V, F), directed=False, indices=o); r["t_dijkstra_s"] = time.time() - t
        du = np.linalg.norm(V - V[o], axis=1)
        for name, d in (("heat", dh), ("exact", de), ("dijkstra", dd), ("euclid", du)):
            s, r2, nb = fit(d, S)
            r["sigma_" + name], r["r2_" + name], r["nbins_" + name] = s, r2, nb
            if name != "exact":
                for lim, tag in ((30.0, "30"), (None, "all")):
                    m, p95, ab = err(d, de, lim)
                    r["relerr_%s_%s" % (name, tag)], r["p95_%s_%s" % (name, tag)], r["abserr_mm_%s_%s" % (name, tag)] = m, p95, ab
        r["max_exact_mm"] = float(np.nanmax(de[np.isfinite(de)]))
        rows.append(r)
        print("map %3d n=%6d sigma heat %6.2f exact %6.2f dijkstra %6.2f euclid %6.2f | r2 heat %.2f | heat rel err(30) %+.4f dijkstra %+.4f euclid %+.4f" % (
            mid, len(V), r["sigma_heat"], r["sigma_exact"], r["sigma_dijkstra"], r["sigma_euclid"], r["r2_heat"],
            r["relerr_heat_30"], r["relerr_dijkstra_30"], r["relerr_euclid_30"]), flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "v38_per_map.csv"), index=False)

    c = df[(~df.fragmented) & (~df.excluded_quality)].copy()
    L = ["v38 clinical maps under four distance definitions — summary", "=" * 60,
         "cohort: %d maps (59 minus 3 fragmented minus map 7), %d patients" % (len(c), c.study_id.nunique()), ""]
    L.append("A. Distance error against the exact geodesic from the argmax origin (per-map values, median [IQR] over maps)")
    for name in ("heat", "dijkstra", "euclid"):
        for tag in ("30", "all"):
            m = c["relerr_%s_%s" % (name, tag)]; p = c["p95_%s_%s" % (name, tag)]; ab = c["abserr_mm_%s_%s" % (name, tag)]
            L.append("  %-8s within %-3s: signed rel err median %+.4f [%+.4f; %+.4f]  p95|rel| median %.4f  mean|err| median %.2f mm (max %.2f)" % (
                name, tag, m.median(), m.quantile(.25), m.quantile(.75), p.median(), ab.median(), ab.max()))
    L += ["", "B. Width by distance definition (55 maps): median [IQR]; paired difference to heat"]
    for name in ("heat", "exact", "dijkstra", "euclid"):
        s = c["sigma_" + name]; L.append("  %-8s sigma median %.2f [%.2f; %.2f]  R2 median %.3f  n(R2<0.7)=%d  n(sigma>=30)=%d" % (
            name, s.median(), s.quantile(.25), s.quantile(.75), c["r2_" + name].median(), (c["r2_" + name] < 0.7).sum(), (s >= 30).sum()))
    for name in ("exact", "dijkstra", "euclid"):
        d = c["sigma_" + name] - c["sigma_heat"]; rel = d / c["sigma_heat"]
        rho = spearmanr(c["sigma_" + name], c["sigma_heat"]).correlation
        L.append("  %-8s - heat: dsigma median %+.2f [%+.2f; %+.2f] mm; |dsigma|>1 mm in %d, >10%% in %d of %d; rho %.3f" % (
            name, d.median(), d.quantile(.25), d.quantile(.75), (d.abs() > 1).sum(), (rel.abs() > .1).sum(), len(c), rho))
    foc = c[(c.r2_heat >= 0.7) & (c.sigma_heat < 30)]
    L += ["", "C. Same, restricted to maps with R2_heat >= 0.7 and sigma_heat < 30 mm (n=%d)" % len(foc)]
    for name in ("exact", "dijkstra", "euclid"):
        d = foc["sigma_" + name] - foc["sigma_heat"]; rel = d / foc["sigma_heat"]
        L.append("  %-8s - heat: dsigma median %+.2f [%+.2f; %+.2f] mm; |dsigma|>1 mm in %d, >10%% in %d of %d; rho %.3f" % (
            name, d.median(), d.quantile(.25), d.quantile(.75), (d.abs() > 1).sum(), (rel.abs() > .1).sum(), len(foc),
            spearmanr(foc["sigma_" + name], foc["sigma_heat"]).correlation))
    L += ["", "D. Group medians by reference (heat / exact / dijkstra / euclid)"]
    for ref, g in c.groupby("reference"):
        L.append("  %-6s n=%2d: %s" % (ref, len(g), "  ".join("%s %.2f" % (n, g["sigma_" + n].median()) for n in ("heat", "exact", "dijkstra", "euclid"))))
    L += ["", "E. Timing per map (median): heat factor+query %.3f s, exact %.2f s, dijkstra %.3f s" % (c.t_heat_s.median(), c.t_exact_s.median(), c.t_dijkstra_s.median())]
    big = c[c.abs()["sigma_euclid"].sub(c["sigma_heat"]).abs() > 3] if False else c[(c.sigma_euclid - c.sigma_heat).abs() > 3]
    L += ["", "F. Maps with |sigma_euclid - sigma_heat| > 3 mm:"]
    for _, r in big.iterrows():
        L.append("  map %3d %-6s %-14s sigma heat %6.1f euclid %6.1f exact %6.1f R2 heat %.2f euclid %.2f" % (r.map_id, r.reference, r.region, r.sigma_heat, r.sigma_euclid, r.sigma_exact, r.r2_heat, r.r2_euclid))
    txt = "\n".join(L); print(txt); open(os.path.join(a.out, "v38_summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
