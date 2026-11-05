#!/usr/bin/env python3
"""v39 — The focality index F on the 55 clinical maps of the April 2026
submission (reviewer 2, point 4; reviewer 3, points 2 and 4) and its
relation to R^2 and sigma.

F = |largest connected component of H| / |H|, H = vertices with score >=
the 95th percentile of the vertex field (Eq. 7 of the submission).  Two
connectivity rules are computed: 'face' (two top vertices are connected
when they share a triangle whose three vertices are all in H; this is the
rule of the platform code that produced the submission's focality values,
pulse-ultimate/spatial_decay.py::compute_focality) and 'edge' (connected
through any mesh edge, the wording of Eq. 7).  Sensitivity: percentile 90 /
95 / 98 / 99; thresholds F < 0.5 ... 0.9.  Also: number of components,
geodesic distance (Heat Method) from the argmax origin to the farthest
vertex of H and to the farthest component ('secondary region' > 15 mm).

Outputs: v39_per_map.csv, v39_summary.txt.  TEST only.
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from pulse_ep_decay.mesh import HeatSolver, largest_component
from pulse_ep_decay.profile import bin_profile, fit_gaussian, find_origin

ROOT = os.environ.get("GD_ROOT", os.path.expanduser("~/mnt/IdeaProjects"))
MESHES = os.path.join(ROOT, "pulse-ep/data/validation_meshes")
SUMMARY = os.path.join(ROOT, "pulse-ultimate/decay_profiles_heat/decay_summary.csv")
STUDY = os.path.join(ROOT, "thesis-med/_paper/method-paper/supplementary_validation_results/map_study.csv")
EXCLUDE_QUALITY = {7}


def focality(S, F, pct, rule, d_origin):
    thr = np.percentile(np.where(np.isnan(S), 0.0, S), pct)
    top = np.where(np.isnan(S), 0.0, S) >= thr
    H = np.where(top)[0]; nH = len(H)
    if nH < 3:
        return dict(F=np.nan, n_comp=0, nH=nH, far_top_mm=np.nan, far_comp_mm=np.nan, secondary=False)
    pos = -np.ones(len(S), int); pos[H] = np.arange(nH)
    if rule == "face":
        keep = top[F].all(axis=1); Fk = F[keep]
        e = np.vstack([Fk[:, [0, 1]], Fk[:, [1, 2]], Fk[:, [2, 0]]])
    else:
        e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); e = e[top[e].all(axis=1)]
    e = pos[e]
    G = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(nH, nH)).tocsr()
    nc, lab = connected_components(G, directed=False)
    sizes = np.bincount(lab)
    dH = d_origin[H]
    # farthest component: min distance of its vertices to the origin, max over components
    comp_min = np.array([dH[lab == k].min() for k in range(nc)])
    return dict(F=float(sizes.max() / nH), n_comp=int(nc), nH=int(nH), far_top_mm=float(np.nanmax(dH)),
                far_comp_mm=float(comp_min.max()), secondary=bool((comp_min > 15.0).any()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summ = pd.read_csv(SUMMARY).set_index("map_id"); study = pd.read_csv(STUDY).set_index("map_id")["study_id"]
    rows = []
    for mid in sorted(summ.index):
        if summ.loc[mid, "max_geodesic_dist"] < 5.0 or mid in EXCLUDE_QUALITY:
            continue
        z = np.load(os.path.join(MESHES, "map_%03d.npz" % mid), allow_pickle=True)
        V, F, idx = largest_component(z["vertices"], z["triangles"].astype(int)); S = z["scores"].astype(float)[idx >= 0]
        o = find_origin(S, V, F); heat = HeatSolver(V, F); d = heat(o)
        c, m, _ = bin_profile(d, S); g = fit_gaussian(c, m)
        r = dict(map_id=mid, study_id=int(study.get(mid, -1)), region=summ.loc[mid, "part"], atrium=summ.loc[mid, "atrium"],
                 reference=("double" if summ.loc[mid, "ref_catheters"] == 2 else "single"), n_points=int(summ.loc[mid, "n_points"]),
                 n_vertices=len(V), sigma=g["sigma"], r2=g["r2"], origin_score=float(S[o]),
                 F_platform_pct=float(summ.loc[mid, "focality_pct"]), platform_multifocal=bool(summ.loc[mid, "is_multifocal"]))
        for pct in (90, 95, 98, 99):
            for rule in ("face", "edge"):
                f = focality(S, F, pct, rule, d)
                r["F_%s_p%d" % (rule, pct)] = f["F"]; r["ncomp_%s_p%d" % (rule, pct)] = f["n_comp"]
                if rule == "edge":
                    r["nH_p%d" % pct] = f["nH"]; r["far_top_mm_p%d" % pct] = f["far_top_mm"]; r["far_comp_mm_p%d" % pct] = f["far_comp_mm"]; r["secondary_p%d" % pct] = f["secondary"]
        rows.append(r)
        print("map %3d F face95 %.2f edge95 %.2f (platform %.1f%%) ncomp %d sigma %.1f R2 %.2f far_comp %.1f mm" % (
            mid, r["F_face_p95"], r["F_edge_p95"], r["F_platform_pct"], r["ncomp_edge_p95"], r["sigma"], r["r2"], r["far_comp_mm_p95"]), flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "v39_per_map.csv"), index=False)
    L = ["v39 focality index on the 55 clinical maps — summary", "=" * 55, "n = %d maps, %d patients" % (len(df), df.study_id.nunique()), ""]
    L.append("A. Reproduction of the platform value (face rule, p95): max |F_face_p95*100 - F_platform_pct| = %.2f; n multifocal (F<0.70): face %d, edge %d, platform %d" % (
        (df.F_face_p95 * 100 - df.F_platform_pct).abs().max(), (df.F_face_p95 < .7).sum(), (df.F_edge_p95 < .7).sum(), df.platform_multifocal.sum()))
    L += ["", "B. Distribution of F (edge rule) by percentile: median [IQR], min; n(F<0.70); n(F<0.5)"]
    for pct in (90, 95, 98, 99):
        f = df["F_edge_p%d" % pct]
        L.append("  p%d: F median %.2f [%.2f; %.2f] min %.2f  n(F<0.70)=%d  n(F<0.5)=%d  n(F<0.9)=%d  components median %d max %d  |H| median %d" % (
            pct, f.median(), f.quantile(.25), f.quantile(.75), f.min(), (f < .7).sum(), (f < .5).sum(), (f < .9).sum(),
            df["ncomp_edge_p%d" % pct].median(), df["ncomp_edge_p%d" % pct].max(), df["nH_p%d" % pct].median()))
    L += ["", "C. Threshold sensitivity (edge rule, p95): n flagged at F < T"]
    f = df.F_edge_p95
    L.append("  " + "  ".join("T=%.2f: %d" % (T, (f < T).sum()) for T in (0.5, 0.6, 0.7, 0.8, 0.9)))
    L += ["", "D. Relation to R2 and sigma (edge rule, p95)"]
    L.append("  Spearman rho(F, R2) = %.3f (p=%.3f); rho(F, sigma) = %.3f (p=%.3f); rho(F, n_points) = %.3f" % (
        spearmanr(f, df.r2).correlation, spearmanr(f, df.r2).pvalue, spearmanr(f, df.sigma).correlation, spearmanr(f, df.sigma).pvalue, spearmanr(f, df.n_points).correlation))
    cross = pd.crosstab(df.r2 >= 0.7, f >= 0.7); L.append("  crosstab rows R2>=0.7 (False/True) x cols F>=0.70 (False/True):\n" + cross.to_string())
    L += ["", "E. Maps flagged multifocal (F_edge_p95 < 0.70):"]
    for _, r in df[f < .7].sort_values("F_edge_p95").iterrows():
        L.append("  map %3d %-6s %-14s F %.2f (%d comps) sigma %5.1f R2 %.2f farthest comp %.1f mm secondary(>15 mm) %s n_points %d" % (
            r.map_id, r.reference, r.region, r.F_edge_p95, r.ncomp_edge_p95, r.sigma, r.r2, r.far_comp_mm_p95, r.secondary_p95, r.n_points))
    L += ["", "F. Secondary high-score region > 15 mm (geodesic) from the origin, p95: n = %d of %d; among them F<0.70: %d" % (
        df.secondary_p95.sum(), len(df), (df.secondary_p95 & (f < .7)).sum())]
    L += ["", "G. By reference (edge p95): median F, n(F<0.70), n(secondary)"]
    for ref, g in df.groupby("reference"):
        L.append("  %-6s n=%2d median F %.2f  n(F<0.70)=%d  n(secondary)=%d  n(R2<0.7)=%d" % (ref, len(g), g.F_edge_p95.median(), (g.F_edge_p95 < .7).sum(), g.secondary_p95.sum(), (g.r2 < .7).sum()))
    txt = "\n".join(L); print(txt); open(os.path.join(a.out, "v39_summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
