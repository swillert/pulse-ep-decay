#!/usr/bin/env python3
"""v50 — focality index on the half-maximum set (25 Oct 2026).

Submitted definition (F_pct): share of the LARGEST connected component of the 95th-percentile set of vertex
scores.  Finding during the revision (make_revision_figures / origin_component_check): in 27 of 55 clinical maps
the vertex of maximum score is not in that largest component, because the percentile set of a narrow peak
extends far below the peak height and is dominated by broad regions of moderately high score elsewhere.
Revised definition (F): the high-score set is H = {v : S_v >= B + A/2}, the half maximum of the fitted radial
profile (A, B from the same Gaussian fit as sigma), connected components by mesh edges, and
F = |component containing the origin| / |H|.  Flag F < 0.70 as before.
Parts:  s  sphere (R 25 mm, icosphere 4; same field families as v48: focal 5/10/15 mm with iid 5-25 % and
           correlated noise, flat, ring, two peaks at 10/15/25/40 mm with height ratio 0.3-1.0; 20 trials)
        m  meshes (56, exact-geodesic truth from the centroid vertex, sigma0 5/10/15; iid 10 %, bumps 0.25/0.50 A,
           corr10 0.25/0.40 A; 3 seeds; resumable per mesh)  — reduced version of v47b for the F columns only
        c  clinical 55 maps (F_pct and F for comparison)
Usage: v50_focality_halfmax.py s|m|c|summary [budget_s]
"""
import sys, os, glob, time
import numpy as np, pandas as pd, trimesh
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import connected_components
from pulse_ep_decay.mesh import HeatSolver, ExactSolver, largest_component, centroid_vertex
from pulse_ep_decay.profile import bin_profile, fit_gaussian, find_origin

ROOT = os.environ.get("GD_ROOT", os.path.expanduser("~/mnt/IdeaProjects")); MESHES = os.path.join(ROOT, "pulse-ep/data/validation_meshes")
SVR = os.path.join(ROOT, "thesis-med/_paper/method-paper/supplementary_validation_results"); OUT = os.path.join(SVR, "v50_focality_halfmax_plateau_2026-10-26")
SUMMARY = os.path.join(ROOT, "pulse-ultimate/decay_profiles_heat/decay_summary.csv")
R = 25.0; A, B0 = 35.0, 65.0; FRAGMENTED = {87, 102, 103}


def gauss(d, A_, s, B_): return A_ * np.exp(-d ** 2 / (2 * s ** 2)) + B_


def edge_graph(F, n):
    e = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    return coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()


def components(S, G, mask):
    idx = np.where(mask)[0]
    if len(idx) < 3: return idx, None, None
    _, lab = connected_components(G[idx][:, idx], directed=False); return idx, lab, np.bincount(lab)


def F_pct(S, G):
    fin = np.isfinite(S); idx = np.where(fin)[0]; thr = np.percentile(S[idx], 95)
    H, lab, sz = components(S, G, fin & (S >= thr))
    return (float(sz.max() / len(H)), int(len(sz))) if lab is not None else (np.nan, 0)


def F_half(S, G, o, Afit, Bfit):
    thr = Bfit + 0.5 * Afit; fin = np.isfinite(S); H, lab, sz = components(S, G, fin & (S >= thr))
    if lab is None: return np.nan, 0, np.nan, thr
    pos = np.where(H == o)[0]
    if len(pos) == 0:   # origin below the threshold (fit peak above the maximum vertex) — use the component of the highest vertex in H
        pos = [int(np.argmax(S[H]))]
    k = lab[pos[0]]; return float(sz[k] / len(H)), int(len(sz)), float(sz.max() / len(H)), thr


def cotan_laplacian(V, F):
    n = len(V); I, J, W = [], [], []
    for k in range(3):
        i, j, l = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]; u, w = V[i] - V[l], V[j] - V[l]
        cot = (u * w).sum(1) / np.maximum(np.linalg.norm(np.cross(u, w), axis=1), 1e-12); I += [i, j]; J += [j, i]; W += [0.5 * cot, 0.5 * cot]
    Wm = coo_matrix((np.concatenate(W), (np.concatenate(I), np.concatenate(J))), shape=(n, n)).tocsr(); L = diags(np.asarray(Wm.sum(1)).ravel()) - Wm
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1); M = np.zeros(n); np.add.at(M, F.ravel(), np.repeat(area / 3, 3))
    return L.tocsc(), M


def evaluate(S, d_fn, G, V, F):
    """Pipeline on a synthetic field: argmax origin, distances d_fn(o), binned Gaussian fit, both indices."""
    o = find_origin(S, V, F); d = d_fn(o); c, m, _ = bin_profile(d, S); g = fit_gaussian(c, m)
    fp, ncp = F_pct(S, G); fh, nch, fh_largest, thr = F_half(S, G, o, g["A"], g["B"])
    return o, g, dict(sigma_hat=g["sigma"], r2=g["r2"], A=g["A"], B=g["B"], F_pct=fp, ncomp_pct=ncp, F=fh, ncomp=nch, F_largest=fh_largest, thr=thr)


def part_s():
    rng = np.random.default_rng(20261025 + 50)
    m = trimesh.creation.icosphere(subdivisions=4, radius=R); V, F = np.asarray(m.vertices), np.asarray(m.faces); U = V / R; n = len(V); G = edge_graph(F, n)
    L, M = cotan_laplacian(V, F); lu = splu((diags(M) + 100.0 * L).tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    rows = []
    for t in range(20):
        v0 = int(rng.integers(n)); d0 = R * np.arccos(np.clip(U @ U[v0], -1, 1))
        fields = [("focal", dict(sigma0=s0), gauss(d0, A, s0, B0)) for s0 in (5, 10, 15)]
        fields.append(("flat", {}, np.full(n, B0))); fields.append(("ring", {}, B0 + A * np.exp(-(d0 - 15) ** 2 / (2 * 4.0 ** 2))))
        for D in (10, 15, 25, 40):
            cand = np.where(np.abs(d0 - D) < 1.0)[0]; v1 = int(rng.choice(cand)); d1 = R * np.arccos(np.clip(U @ U[v1], -1, 1))
            for ratio in (0.3, 0.5, 0.7, 0.8, 1.0):
                fields.append(("two-peak", dict(D=D, ratio=ratio), B0 + A * np.maximum(np.exp(-d0 ** 2 / 98), ratio * np.exp(-d1 ** 2 / 98))))
        for kind, par, base in fields:
            conds = [("iid", e) for e in ((0.05, 0.10, 0.15, 0.25) if kind == "focal" else (0.05, 0.10, 0.25))] + ([("corr10", 0.10), ("corr10", 0.25)] if kind == "focal" else [])
            for nk, amp in conds:
                dev = rng.normal(0, amp * A, n) if nk == "iid" else (lambda u: u / u.std() * amp * A)(lu.solve(M * rng.normal(0, 1, n)))
                S = np.clip(base + dev, 50, 100); S[rng.random(n) < .05] = np.nan
                o, g, res = evaluate(S, lambda o: R * np.arccos(np.clip(U @ U[o], -1, 1)), G, V, F)
                rows.append(dict(trial=t, kind=kind, noise=nk, amp=amp, origin_offset=float(d0[o]), **{k: par.get(k, "") for k in ("sigma0", "D", "ratio")}, **res))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "v50_sphere.csv"), index=False); print("sphere rows", len(df))


CONDS_M = [("iid", 0.10), ("bumps", 0.25), ("bumps", 0.50), ("corr10", 0.25), ("corr10", 0.40)]
def part_m(budget):
    os.makedirs(os.path.join(OUT, "parts"), exist_ok=True); t0 = time.time()
    for f in sorted(glob.glob(os.path.join(MESHES, "map_*.npz"))):
        z = np.load(f, allow_pickle=True); mid = int(z["map_id"]); pf = os.path.join(OUT, "parts", "%03d.csv" % mid)
        if mid in FRAGMENTED or os.path.exists(pf): continue
        if time.time() - t0 > budget: print("budget reached before", mid); return
        V, T, _ = largest_component(z["vertices"], z["triangles"].astype(int)); n = len(V)
        v0 = centroid_vertex(V); d0 = ExactSolver(V, T)(v0); d0 = np.where(np.isfinite(d0), d0, np.nan); heat = HeatSolver(V, T); G = edge_graph(T, n)
        L, M = cotan_laplacian(V, T); lu = splu((diags(M) + 100.0 * L).tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
        rows = []
        for s0 in (5.0, 10.0, 15.0):
            base = A * np.exp(-np.nan_to_num(d0) ** 2 / (2 * s0 ** 2)) + B0
            for kind, amp in CONDS_M:
                for t in range(3):
                    rng = np.random.default_rng(50_000_000 + 10_000 * mid + int(100 * s0) + int(100 * amp) * 3 + (0 if kind == "iid" else 1 if kind == "bumps" else 2) * 1000 + t)
                    if kind == "iid": dev = rng.normal(0, amp * A, n)
                    elif kind == "bumps":
                        c = V[rng.integers(n, size=30)]; w = rng.normal(0, amp * A, 30); dd = np.linalg.norm(V[:, None, :] - c[None, :, :], axis=2)
                        dev = (np.exp(-dd ** 2 / (2 * 8.0 ** 2)) * w).sum(1) + rng.normal(0, 0.10 * A, n)
                    else: u = lu.solve(M * rng.normal(0, 1, n)); dev = u / u.std() * amp * A
                    S = np.clip(base + dev, 50, 100); S[np.isnan(d0)] = np.nan; S[rng.random(n) < 0.05] = np.nan
                    o, g, res = evaluate(S, heat, G, V, T)
                    rows.append(dict(map_id=mid, sigma0=s0, kind=kind, amp=amp, trial=t, offset_mm=float(d0[o]), **res))
        pd.DataFrame(rows).to_csv(pf, index=False); print("map %3d done (%d vertices, %.0f s)" % (mid, n, time.time() - t0), flush=True)
    print("all meshes done")


def part_c():
    summ = pd.read_csv(SUMMARY).set_index("map_id"); rows = []
    for mid in sorted(summ.index):
        if mid in FRAGMENTED or mid == 7: continue
        z = np.load(os.path.join(MESHES, "map_%03d.npz" % mid), allow_pickle=True)
        V, F, idx = largest_component(z["vertices"], z["triangles"].astype(int)); S = z["scores"].astype(float)[idx >= 0]; n = len(V)
        o = find_origin(S, V, F); heat = HeatSolver(V, F); d = heat(o); c, m, _ = bin_profile(d, S); g = fit_gaussian(c, m); G = edge_graph(F, n)
        fp, ncp = F_pct(S, G); fh, nch, fl, thr = F_half(S, G, o, g["A"], g["B"])
        # geometry of the other components above half maximum: nearest geodesic distance and highest score
        fin = np.isfinite(S); H, lab, sz = components(S, G, fin & (S >= thr)); ko = lab[np.where(H == o)[0][0]] if (H == o).any() else -1
        other = H[lab != ko] if lab is not None else np.array([], int)
        rows.append(dict(map_id=mid, sigma=g["sigma"], r2=g["r2"], A=g["A"], B=g["B"], thr_half=thr, F_pct=fp, ncomp_pct=ncp, F=fh, ncomp=nch, F_largest=fl, nH=int(len(H)),
                         other_min_dist_mm=float(d[other].min()) if len(other) else np.nan, other_max_score_rel=float((np.nanmax(S[other]) - g["B"]) / g["A"]) if len(other) else np.nan, other_share=float(len(other) / len(H))))
        print("map %3d sigma %5.1f R2 %.2f  F_pct %.2f (%2d comp)  F %.2f (%2d comp)  other: %.1f mm, %.2f of peak" % (mid, g["sigma"], g["r2"], fp, ncp, fh, nch, rows[-1]["other_min_dist_mm"] or np.nan, rows[-1]["other_max_score_rel"] or np.nan), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "v50_clinical.csv"), index=False)


def summary():
    L = ["v50 focality index on the half-maximum set — summary", "=" * 56]
    s = pd.read_csv(os.path.join(OUT, "v50_sphere.csv"))
    L += ["", "A. Sphere (20 trials). Flag rate F < 0.70 (%), F_pct for comparison"]
    for (kind, nk, amp), g in s[s.kind == "focal"].groupby(["kind", "noise", "amp"]): L.append("  single peak %-6s %.2f: F<0.7 %3.0f %%  F_pct<0.7 %3.0f %%  F median %.3f  ncomp median %d" % (nk, amp, 100 * (g.F < .7).mean(), 100 * (g.F_pct < .7).mean(), g.F.median(), g.ncomp.median()))
    for kind in ("flat", "ring"):
        for (nk, amp), g in s[s.kind == kind].groupby(["noise", "amp"]): L.append("  %-5s %s %.2f: F<0.7 %3.0f %%  F_pct<0.7 %3.0f %%  F median %.3f" % (kind, nk, amp, 100 * (g.F < .7).mean(), 100 * (g.F_pct < .7).mean(), g.F.median()))
    L.append("  two peaks (should be flagged): rows D, cols ratio -> F<0.7 % at 5/10/25 % noise")
    for D, gD in s[s.kind == "two-peak"].groupby("D"):
        L.append("    D=%2d: " % D + "  ".join("r%.1f: %3.0f/%3.0f/%3.0f" % (ratio, *[100 * (gg[gg.amp == e].F < .7).mean() for e in (0.05, 0.10, 0.25)]) for ratio, gg in gD.groupby("ratio")))
    L.append("    origin captured by the second peak (offset > 5 mm), ratio 1.0: %.0f %%" % (100 * (s[(s.kind == "two-peak") & (s.ratio == 1.0)].origin_offset > 5).mean()))
    parts = sorted(glob.glob(os.path.join(OUT, "parts", "*.csv")))
    if parts:
        m = pd.concat([pd.read_csv(p) for p in parts]); m.to_csv(os.path.join(OUT, "v50_meshes.csv"), index=False)
        L += ["", "B. Meshes (%d, 3 seeds): F<0.7 %% (F_pct<0.7 %%) by condition and sigma0" % m.map_id.nunique()]
        for (kind, amp), g in m.groupby(["kind", "amp"]): L.append("  %-6s %.2f: " % (kind, amp) + "  ".join("s0=%2d: %3.0f (%3.0f)" % (s0, 100 * (gg.F < .7).mean(), 100 * (gg.F_pct < .7).mean()) for s0, gg in g.groupby("sigma0")) + "   origin>10mm %.0f %%" % (100 * (g.offset_mm > 10).mean()))
    c = pd.read_csv(os.path.join(OUT, "v50_clinical.csv"))
    L += ["", "C. Clinical (55 maps): F median %.2f [%.2f; %.2f]; F<0.70 in %d (maps %s); F_pct<0.70 in %d; origin's component = largest in %d; ncomp median %d" % (
        c.F.median(), c.F.quantile(.25), c.F.quantile(.75), (c.F < .7).sum(), ", ".join(str(x) for x in c[c.F < .7].map_id), (c.F_pct < .7).sum(), (abs(c.F - c.F_largest) < 1e-9).sum(), c.ncomp.median())]
    x = c[c.F < .7]; L.append("   flagged maps: sigma median %.1f (%.1f-%.1f); other components %.0f-%.0f mm from the origin, highest score %.2f-%.2f of the peak height; R2<0.7 in %d of them" % (x.sigma.median(), x.sigma.min(), x.sigma.max(), x.other_min_dist_mm.min(), x.other_min_dist_mm.max(), x.other_max_score_rel.min(), x.other_max_score_rel.max(), (x.r2 < .7).sum()))
    for T in (0.5, 0.6, 0.7, 0.8, 0.9): L.append("   threshold %.1f: %d maps flagged" % (T, (c.F < T).sum()))
    open(os.path.join(OUT, "v50_summary.txt"), "w").write("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); cmd = sys.argv[1]
    {"s": part_s, "c": part_c, "summary": summary}.get(cmd, lambda: part_m(float(sys.argv[2]) if len(sys.argv) > 2 else 150))()
