#!/usr/bin/env python3
"""v48 — Two candidates for a better secondary-region flag (reviewer 2 point 4,
reviewer 3 point 4; the five narrow maps of v46 that no flag caught):
  F_raw    : focality index of the submission (95th percentile, edge components)
  F_smooth : the same index computed on the 3-mm-neighbourhood mean field
  rho2     : secondary-maximum ratio = (S~_2 - B) / (S~_1 - B), where S~ is the
             3-mm mean field, S~_1 its value at the detected origin (raw argmax),
             S~_2 its maximum beyond D = max(15 mm, 3 sigma_hat) geodesic from the
             origin, B the fitted baseline.  rho2 > 1 means a region elsewhere is
             on average higher than the origin's neighbourhood.
Part S (sphere, analytic geodesics, 2,562 vertices, 20 trials): single peaks
(sigma0 5/10/15; iid noise 5/10/15/25 % and correlated noise 0.25 A, l = 10 mm),
flat, ring, two peaks (separation 10/15/25/40 mm, height ratio 0.3/0.5/0.7/0.8/1.0,
sigma 7 mm each; iid 5/10/25 %).  Reported: detection rates at thresholds.
Part C (clinical, 55 maps, April field): F_raw, F_smooth, rho2 (D rule and fixed
15/20 mm), joined with R2, sigma, the v46 marker offset.
Outputs: v48_sphere.csv, v48_clinical.csv, v48_summary.txt.  TEST only.
"""
import argparse, json, os
import numpy as np, pandas as pd, trimesh
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from pulse_ep_decay.mesh import HeatSolver, largest_component
from pulse_ep_decay.profile import bin_profile, fit_gaussian, find_origin

ROOT = os.environ.get("GD_ROOT", os.path.expanduser("~/mnt/IdeaProjects")); MESHES = os.path.join(ROOT, "pulse-ep/data/validation_meshes")
SVR = os.path.join(ROOT, "thesis-med/_paper/method-paper/supplementary_validation_results")
SUMMARY = os.path.join(ROOT, "pulse-ultimate/decay_profiles_heat/decay_summary.csv")
RNG = np.random.default_rng(20261025); R = 25.0; A, B0 = 35.0, 65.0


def gauss(d, A_, s, B_): return A_ * np.exp(-d ** 2 / (2 * s ** 2)) + B_


def smoother(V):
    nb = cKDTree(V).query_ball_point(V, 3.0); rows = np.repeat(np.arange(len(V)), [len(n) for n in nb]); cols = np.concatenate([np.asarray(n) for n in nb])
    return coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(V), len(V))).tocsr()


def smooth(W, S):
    fin = np.isfinite(S); num = W @ np.where(fin, S, 0.0); den = W @ fin.astype(float)
    out = np.full(len(S), np.nan); ok = den > 0; out[ok] = num[ok] / den[ok]; return out


def edge_graph(F, n):
    e = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    return coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()


def focality(S, G):
    idx = np.where(np.isfinite(S))[0]; thr = np.percentile(S[idx], 95); H = idx[S[idx] >= thr]
    if len(H) < 3: return np.nan
    _, lab = connected_components(G[H][:, H], directed=False); return float(np.bincount(lab).max() / len(H))


def rho2(St, d, origin, sigma, Bfit, Dfix=None):
    D = Dfix if Dfix is not None else max(15.0, 3.0 * sigma)
    far = np.isfinite(St) & (d > D)
    if not far.any() or not np.isfinite(St[origin]): return np.nan
    s1 = St[origin] - Bfit; s2 = np.nanmax(St[far]) - Bfit
    return float(s2 / s1) if s1 > 1e-9 else np.nan


def fit_sphere(d, S):
    c, m, _ = bin_profile(d, S); g = fit_gaussian(c, m); return g["sigma"], g["r2"], g["B"]


def part_s(out):
    m = trimesh.creation.icosphere(subdivisions=4, radius=R); V, F = np.asarray(m.vertices), np.asarray(m.faces); U = V / R; n = len(V)
    W = smoother(V); G = edge_graph(F, n)
    # correlated-noise operator
    I, J, Wt = [], [], []
    for k in range(3):
        i, j, l = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]; u, w = V[i] - V[l], V[j] - V[l]
        cot = (u * w).sum(1) / np.linalg.norm(np.cross(u, w), axis=1); I += [i, j]; J += [j, i]; Wt += [0.5 * cot, 0.5 * cot]
    Wm = coo_matrix((np.concatenate(Wt), (np.concatenate(I), np.concatenate(J))), shape=(n, n)).tocsr(); Lp = diags(np.asarray(Wm.sum(1)).ravel()) - Wm
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1); M = np.zeros(n); np.add.at(M, F.ravel(), np.repeat(area / 3, 3))
    lu = splu((diags(M) + 100.0 * Lp).tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    rows = []
    for t in range(20):
        v0 = int(RNG.integers(n)); d0 = R * np.arccos(np.clip(U @ U[v0], -1, 1))
        fields = [("focal", dict(sigma0=s0), gauss(d0, A, s0, B0)) for s0 in (5, 10, 15)]
        fields.append(("flat", {}, np.full(n, B0))); fields.append(("ring", {}, B0 + A * np.exp(-(d0 - 15) ** 2 / (2 * 4.0 ** 2))))
        for D in (10, 15, 25, 40):
            cand = np.where(np.abs(d0 - D) < 1.0)[0]; v1 = int(RNG.choice(cand)); d1 = R * np.arccos(np.clip(U @ U[v1], -1, 1))
            for ratio in (0.3, 0.5, 0.7, 0.8, 1.0):
                fields.append(("two-peak", dict(D=D, ratio=ratio), B0 + A * np.maximum(np.exp(-d0 ** 2 / 98), ratio * np.exp(-d1 ** 2 / 98))))
        for kind, par, base in fields:
            conds = [("iid", e) for e in ((0.05, 0.10, 0.15, 0.25) if kind == "focal" else (0.05, 0.10, 0.25))] + ([("corr10", 0.25)] if kind == "focal" else [])
            for nk, amp in conds:
                dev = RNG.normal(0, amp * A, n) if nk == "iid" else (lambda u: u / u.std() * amp * A)(lu.solve(M * RNG.normal(0, 1, n)))
                S = np.clip(base + dev, 50, 100); S[RNG.random(n) < .05] = np.nan
                o = find_origin(S, V, F); d = R * np.arccos(np.clip(U @ U[o], -1, 1)); sh, r2, Bf = fit_sphere(d, S); St = smooth(W, S)
                rows.append(dict(trial=t, kind=kind, noise=nk, amp=amp, sigma_hat=sh, r2=r2, F_raw=focality(S, G), F_smooth=focality(St, G),
                                 rho2=rho2(St, d, o, sh, Bf), rho2_15=rho2(St, d, o, sh, Bf, 15.0), origin_offset=float(d0[o]), **{k: par.get(k, "") for k in ("sigma0", "D", "ratio")}))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "v48_sphere.csv"), index=False); return df


def part_c(out):
    summ = pd.read_csv(SUMMARY).set_index("map_id"); v46 = pd.read_csv(os.path.join(SVR, "v46_lo_origin_plateau_2026-10-26/v46_clinical.csv")).set_index("map_id"); rows = []
    for mid in sorted(summ.index):
        if mid in (87, 102, 103, 7): continue
        z = np.load(os.path.join(MESHES, "map_%03d.npz" % mid), allow_pickle=True)
        V, F, idx = largest_component(z["vertices"], z["triangles"].astype(int)); S = z["scores"].astype(float)[idx >= 0]; n = len(V)
        o = find_origin(S, V, F); d = HeatSolver(V, F)(o); c, m_, _ = bin_profile(d, S); g = fit_gaussian(c, m_); St = smooth(smoother(V), S); G = edge_graph(F, n)
        rows.append(dict(map_id=mid, reference=("double" if summ.loc[mid, "ref_catheters"] == 2 else "single"), n_points=int(summ.loc[mid, "n_points"]), sigma=g["sigma"], r2=g["r2"],
                         F_raw=focality(S, G), F_smooth=focality(St, G), rho2=rho2(St, d, o, g["sigma"], g["B"]), rho2_15=rho2(St, d, o, g["sigma"], g["B"], 15.0), rho2_20=rho2(St, d, o, g["sigma"], g["B"], 20.0),
                         offset_marker_mm=v46.offset_geo_mm.get(mid, np.nan), score_at_lo_rel=v46.score_at_lo_rel.get(mid, np.nan)))
        print("map %3d sigma %5.1f R2 %.2f F %.2f/%.2f rho2 %.2f (15: %.2f) offset %s" % (mid, g["sigma"], g["r2"], rows[-1]["F_raw"], rows[-1]["F_smooth"], rows[-1]["rho2"], rows[-1]["rho2_15"], rows[-1]["offset_marker_mm"]), flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "v48_clinical.csv"), index=False); return df


def summarise(out, s, c):
    L = ["v48 secondary-region flags: F_raw, F_smooth, rho2 — summary", "=" * 60, "", "S. Sphere: fraction flagged (F < 0.70 | rho2 > 0.5 / 0.7 / 0.8), n per row"]
    def rate(g): return "F_raw %3.0f%%  F_smooth %3.0f%% | rho2>0.5 %3.0f%%  >0.7 %3.0f%%  >0.8 %3.0f%%  (rho2 median %.2f)" % (
        100 * (g.F_raw < .7).mean(), 100 * (g.F_smooth < .7).mean(), 100 * (g.rho2 > .5).mean(), 100 * (g.rho2 > .7).mean(), 100 * (g.rho2 > .8).mean(), g.rho2.median())
    for kind in ("focal", "flat", "ring"):
        for (nk, amp), g in s[s.kind == kind].groupby(["noise", "amp"]):
            L.append("  %-8s %-6s %.2f n=%3d: %s" % (kind, nk, amp, len(g), rate(g)))
    L.append("  two-peak fields (should be flagged):")
    for (D, ratio), g0 in s[s.kind == "two-peak"].groupby(["D", "ratio"]):
        for amp, g in g0.groupby("amp"):
            L.append("  D=%2d ratio %.1f noise %.2f n=%2d: %s" % (D, ratio, amp, len(g), rate(g)))
    L += ["", "C. Clinical (55 maps)"]
    L.append("  F_raw < 0.70: %d; F_smooth < 0.70: %d; rho2 > 0.5: %d, > 0.7: %d, > 0.8: %d (rho2 n/a in %d diffuse maps); rho2_15 > 0.7: %d" % (
        (c.F_raw < .7).sum(), (c.F_smooth < .7).sum(), (c.rho2 > .5).sum(), (c.rho2 > .7).sum(), (c.rho2 > .8).sum(), c.rho2.isna().sum(), (c.rho2_15 > .7).sum()))
    L.append("  F_smooth vs F_raw: median |diff| %.3f, max %.3f; maps flagged by one and not the other: %s" % (
        (c.F_smooth - c.F_raw).abs().median(), (c.F_smooth - c.F_raw).abs().max(), sorted(c[(c.F_raw < .7) != (c.F_smooth < .7)].map_id.tolist())))
    ok = c.dropna(subset=["offset_marker_mm"]); nar = ok[ok.sigma < 10]
    L.append("  narrow maps with marker (n=%d): offset > 10 mm in %d; of these rho2 > 0.7 in %d, F_raw < 0.7 in %d, R2 < 0.7 in %d; among offset <= 10 mm (n=%d): rho2 > 0.7 in %d" % (
        len(nar), (nar.offset_marker_mm > 10).sum(), ((nar.offset_marker_mm > 10) & (nar.rho2 > .7)).sum(), ((nar.offset_marker_mm > 10) & (nar.F_raw < .7)).sum(), ((nar.offset_marker_mm > 10) & (nar.r2 < .7)).sum(),
        (nar.offset_marker_mm <= 10).sum(), ((nar.offset_marker_mm <= 10) & (nar.rho2 > .7)).sum()))
    L.append("  per map (rho2 > 0.7 or F_raw < 0.7 or offset > 10 mm):")
    for _, r in c[(c.rho2 > .7) | (c.F_raw < .7) | (c.offset_marker_mm > 10)].sort_values("rho2", ascending=False).iterrows():
        L.append("    map %3d %-6s pts %3d sigma %5.1f R2 %.2f  F %.2f/%.2f  rho2 %5.2f (15 mm: %5.2f)  offset %s  score@marker %s" % (
            r.map_id, r.reference, r.n_points, r.sigma, r.r2, r.F_raw, r.F_smooth, r.rho2, r.rho2_15, "%.1f" % r.offset_marker_mm if pd.notna(r.offset_marker_mm) else "-", "%.2f" % r.score_at_lo_rel if pd.notna(r.score_at_lo_rel) else "-"))
    txt = "\n".join(L); print(txt); open(os.path.join(out, "v48_summary.txt"), "w").write(txt + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--part", choices=["s", "c", "summary"], required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.part == "s": part_s(a.out)
    elif a.part == "c": part_c(a.out)
    else:
        summarise(a.out, pd.read_csv(os.path.join(a.out, "v48_sphere.csv")), pd.read_csv(os.path.join(a.out, "v48_clinical.csv")))


if __name__ == "__main__":
    main()
