#!/usr/bin/env python3
"""v47a — Origin rule: single maximum vertex (submission) versus the vertex
with the highest mean score in its 3-mm neighbourhood ("smoothed argmax",
pulse_ep_decay.support.smoothed_argmax, Euclidean ball; on the clinical
meshes the edge-graph and Euclidean 3-mm neighbourhoods coincide).

Part A (clinical, 55 maps, April vertex field, Heat Method, 2-mm bins >= 3
vertices, Gaussian fit): origin, sigma, R2 under both rules; shift between
the two origins; for the 46 maps with a unique marker (v46 selection) the
geodesic distance of each origin to the marked site.
Part B (validation, 56 meshes, exact-geodesic truth from the centroid-nearest
vertex, manuscript noise/clip/mask protocol, sigma0 3-30 mm, eta 0/5/10/15 %,
20 trials = 35,840 fields): both rules on the SAME fields; bias, SD, origin
offset per rule.  Resumable per mesh (parts/).
Outputs: v47a_clinical.csv, v47a_validation_raw.csv, v47a_validation_summary.csv,
v47a_summary.txt.  TEST only.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from pulse_ep_decay.mesh import HeatSolver, ExactSolver, largest_component, centroid_vertex, mean_edge_length
from pulse_ep_decay.profile import bin_profile, fit_gaussian, find_origin
from pulse_ep_decay.support import smoothed_argmax

ROOT = os.path.expanduser("~/mnt/IdeaProjects")
MESHES = os.path.join(ROOT, "pulse-ep/data/validation_meshes")
SVR = os.path.join(ROOT, "thesis-med/_paper/method-paper/supplementary_validation_results")
SUMMARY = os.path.join(ROOT, "pulse-ultimate/decay_profiles_heat/decay_summary.csv")
SELECTION = os.path.join(SVR, "lo_selection_2026-09-14/selection.json")
FRAGMENTED = {87, 102, 103}; EXCLUDE_QUALITY = {7}; RADIUS = 3.0
A, B = 35.0, 65.0; SIGMAS = (3, 5, 8, 10, 15, 20, 25, 30); NOISES = (0.0, 0.05, 0.10, 0.15); TRIALS = 20


def fit(d, S):
    c, m, _ = bin_profile(d, S); g = fit_gaussian(c, m); return g["sigma"], g["r2"]


class Smoother:
    """vectorised 3-mm Euclidean-ball mean (identical to smoothed_argmax without triangles)."""
    def __init__(self, V):
        nb = cKDTree(V).query_ball_point(V, RADIUS)
        rows = np.repeat(np.arange(len(V)), [len(n) for n in nb]); cols = np.concatenate([np.asarray(n) for n in nb])
        self.W = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(V), len(V))).tocsr()
    def origin(self, S):
        fin = np.isfinite(S); num = self.W @ np.where(fin, S, 0.0); den = self.W @ fin.astype(float)
        means = np.where((den > 0) & fin, num / np.maximum(den, 1), -np.inf)
        return int(np.argmax(means))


def part_a(out):
    summ = pd.read_csv(SUMMARY).set_index("map_id"); sel = {r["map_id"]: r for r in json.load(open(SELECTION))}; rows = []
    for mid in sorted(summ.index):
        if mid in FRAGMENTED or mid in EXCLUDE_QUALITY: continue
        z = np.load(os.path.join(MESHES, "map_%03d.npz" % mid), allow_pickle=True)
        V, F, idx = largest_component(z["vertices"], z["triangles"].astype(int)); S = z["scores"].astype(float)[idx >= 0]
        heat = HeatSolver(V, F); o1 = find_origin(S, V, F); o2 = smoothed_argmax(V, S, RADIUS); o2v = Smoother(V).origin(S)
        d1, d2 = heat(o1), heat(o2)
        r = dict(map_id=mid, reference=("double" if summ.loc[mid, "ref_catheters"] == 2 else "single"), region=summ.loc[mid, "atrium"] + " " + summ.loc[mid, "part"].replace(" wall", ""),
                 n_points=int(summ.loc[mid, "n_points"]), origin_argmax=o1, origin_smooth=o2, smooth_vectorised_equal=(o2 == o2v), shift_mm=float(d1[o2]),
                 score_max=float(S[o1]), score_at_smooth=float(S[o2]))
        r["sigma_argmax"], r["r2_argmax"] = fit(d1, S); r["sigma_smooth"], r["r2_smooth"] = fit(d2, S)
        if sel[mid]["route"] == "unique_lo_reference_candidate":
            xyz = np.asarray(sel[mid]["selected_xyz_mm"], float); v_lo = int(np.argmin(np.linalg.norm(V - xyz, axis=1))); d_lo = heat(v_lo)
            r["offset_argmax_mm"], r["offset_smooth_mm"] = float(d_lo[o1]), float(d_lo[o2])
        rows.append(r)
        print("map %3d: argmax sigma %5.1f R2 %.2f | smooth sigma %5.1f R2 %.2f | shift %5.1f mm | offsets %s / %s" % (
            mid, r["sigma_argmax"], r["r2_argmax"], r["sigma_smooth"], r["r2_smooth"], r["shift_mm"],
            ("%.1f" % r["offset_argmax_mm"]) if "offset_argmax_mm" in r else "-", ("%.1f" % r["offset_smooth_mm"]) if "offset_smooth_mm" in r else "-"), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "v47a_clinical.csv"), index=False)


def part_b_mesh(f, out):
    z = np.load(f, allow_pickle=True); mid = int(z["map_id"])
    if mid in FRAGMENTED or os.path.exists(os.path.join(out, "parts", "%03d.csv" % mid)): return
    V, T, _ = largest_component(z["vertices"], z["triangles"].astype(int))
    v0 = centroid_vertex(V); d0 = ExactSolver(V, T)(v0); d0 = np.where(np.isfinite(d0), d0, np.nan); heat = HeatSolver(V, T); sm = Smoother(V)
    rows = []
    for s0 in SIGMAS:
        for eta in NOISES:
            for t in range(TRIALS):
                rng = np.random.default_rng(1_000_000 * mid + int(1000 * s0) + int(100 * eta * 100) + t)
                S = A * np.exp(-np.nan_to_num(d0) ** 2 / (2 * s0 ** 2)) + B
                if eta > 0: S = S + rng.normal(0, eta * A, len(S))
                S = np.clip(S, 50, 100); S[np.isnan(d0)] = np.nan; S[rng.random(len(S)) < 0.05] = np.nan
                o1 = find_origin(S, V, T); o2 = sm.origin(S)
                s1, r1 = fit(heat(o1), S); s2, r2 = fit(heat(o2), S) if o2 != o1 else (s1, r1)
                rows.append(dict(map_id=mid, sigma0=s0, noise=eta, trial=t, sigma_argmax=s1, r2_argmax=r1, off_argmax=float(d0[o1]), sigma_smooth=s2, r2_smooth=r2, off_smooth=float(d0[o2])))
    pd.DataFrame(rows).to_csv(os.path.join(out, "parts", "%03d.csv" % mid), index=False); print("validation: map %3d done" % mid, flush=True)


def summarise(out):
    L = ["v47a origin rule: single maximum vs 3-mm neighbourhood maximum — summary", "=" * 72]
    ca = os.path.join(out, "v47a_clinical.csv")
    if os.path.exists(ca):
        df = pd.read_csv(ca); L += ["", "A. Clinical, %d maps (vectorised smoother identical to package function in %d/%d)" % (len(df), df.smooth_vectorised_equal.sum(), len(df))]
        L.append("  origin shift argmax -> smoothed: median %.1f mm, > 3 mm in %d, > 10 mm in %d, > 20 mm in %d" % (df.shift_mm.median(), (df.shift_mm > 3).sum(), (df.shift_mm > 10).sum(), (df.shift_mm > 20).sum()))
        d = df.sigma_smooth - df.sigma_argmax
        L.append("  sigma smooth - argmax: median %+.2f [%+.2f; %+.2f], |d| > 1 mm in %d, > 3 mm in %d; R2 median %.3f vs %.3f; n(R2<0.7) %d vs %d" % (
            d.median(), d.quantile(.25), d.quantile(.75), (d.abs() > 1).sum(), (d.abs() > 3).sum(), df.r2_smooth.median(), df.r2_argmax.median(), (df.r2_smooth < .7).sum(), (df.r2_argmax < .7).sum()))
        ok = df.dropna(subset=["offset_argmax_mm"])
        for rule in ("argmax", "smooth"):
            o = ok["offset_%s_mm" % rule]; s = ok["sigma_%s" % rule]; nar = ok[s < 10]
            L.append("  offset to marker (%s, n=%d): median %.1f [%.1f; %.1f]; > 10 mm in %d, > 20 mm in %d; narrow maps (sigma<10, n=%d): median %.1f, > 10 mm in %d" % (
                rule, len(ok), o.median(), o.quantile(.25), o.quantile(.75), (o > 10).sum(), (o > 20).sum(), len(nar), nar["offset_%s_mm" % rule].median(), (nar["offset_%s_mm" % rule] > 10).sum()))
        L += ["  maps where the two rules differ by > 3 mm:"]
        for _, r in df[df.shift_mm > 3].sort_values("shift_mm", ascending=False).iterrows():
            L.append("    map %3d %-6s pts %3d shift %5.1f mm | argmax sigma %5.1f R2 %.2f offset %s | smooth sigma %5.1f R2 %.2f offset %s | score max %.1f, at smooth origin %.1f" % (
                r.map_id, r.reference, r.n_points, r.shift_mm, r.sigma_argmax, r.r2_argmax, ("%.1f" % r.offset_argmax_mm) if pd.notna(r.get("offset_argmax_mm", np.nan)) else "-",
                r.sigma_smooth, r.r2_smooth, ("%.1f" % r.offset_smooth_mm) if pd.notna(r.get("offset_smooth_mm", np.nan)) else "-", r.score_max, r.score_at_smooth))
    parts = sorted(glob.glob(os.path.join(out, "parts", "*.csv")))
    if parts:
        raw = pd.concat([pd.read_csv(p) for p in parts]); raw.to_csv(os.path.join(out, "v47a_validation_raw.csv"), index=False)
        g = raw.groupby(["sigma0", "noise"])
        s = pd.DataFrame(dict(n=g.size(), bias_argmax=g.apply(lambda x: (x.sigma_argmax - x.sigma0).mean()), sd_argmax=g.sigma_argmax.std(), off_argmax=g.off_argmax.median(),
                              bias_smooth=g.apply(lambda x: (x.sigma_smooth - x.sigma0).mean()), sd_smooth=g.sigma_smooth.std(), off_smooth=g.off_smooth.median(),
                              bias_med_argmax=g.apply(lambda x: (x.sigma_argmax - x.sigma0).median()), bias_med_smooth=g.apply(lambda x: (x.sigma_smooth - x.sigma0).median()))).reset_index()
        s.to_csv(os.path.join(out, "v47a_validation_summary.csv"), index=False)
        L += ["", "B. Validation on the same %d fields (%d meshes): argmax vs 3-mm neighbourhood maximum" % (len(raw), raw.map_id.nunique()),
              "  sigma0 noise     n | argmax: bias(mean/median)   SD   offset | smooth: bias(mean/median)   SD   offset"]
        for _, r in s.iterrows():
            L.append("  %5.1f  %.2f  %4d | %+6.2f / %+6.2f  %6.2f  %5.1f | %+6.2f / %+6.2f  %6.2f  %5.1f" % (
                r.sigma0, r.noise, r.n, r.bias_argmax, r.bias_med_argmax, r.sd_argmax, r.off_argmax, r.bias_smooth, r.bias_med_smooth, r.sd_smooth, r.off_smooth))
    txt = "\n".join(L); print(txt); open(os.path.join(out, "v47a_summary.txt"), "w").write(txt + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--part", choices=["a", "b", "summary"], required=True)
    a = ap.parse_args(); os.makedirs(os.path.join(a.out, "parts"), exist_ok=True)
    if a.part == "a": part_a(a.out)
    elif a.part == "b":
        for f in sorted(glob.glob(os.path.join(MESHES, "map_*.npz"))): part_b_mesh(f, a.out)
    summarise(a.out)


if __name__ == "__main__":
    main()
