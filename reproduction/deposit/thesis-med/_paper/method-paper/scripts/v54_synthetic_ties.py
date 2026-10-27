"""v54 — TEST (26 Oct 2026): tied maxima in the synthetic validation (clip at 100) and the tie rule.
Same field protocol as v47a/v46 (exact-geodesic truth from the centroid-nearest vertex, A = 35, B = 65, iid noise,
clip to [50, 100], 5 % masked).  Subset: every 4th of the 56 meshes, sigma0 5/10/20/30 mm, eta 5/10/15 %, 20 trials.
Per field: number of vertices at the maximum; origin by (a) first tied vertex (pipeline, np.nanargmax),
(b) centre of the largest tied region (vertex nearest to its centroid), (c) a random tied vertex; offset from the
true source and sigma for each.  Resumable per mesh (parts/).  Nothing in the manuscript is changed by this script."""
import argparse, glob, os
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pulse_ep_decay.mesh import HeatSolver, ExactSolver, largest_component, centroid_vertex
from pulse_ep_decay.profile import bin_profile, fit_gaussian
ROOT = os.path.expanduser("~/mnt/IdeaProjects"); MESHES = os.path.join(ROOT, "pulse-ep/data/validation_meshes")
OUT = os.path.join(ROOT, "thesis-med/_paper/method-paper/supplementary_validation_results/v54_synthetic_ties_2026-10-26")
FRAGMENTED = {87, 102, 103}; A, B = 35.0, 65.0; SIGMAS = (5, 10, 20, 30); NOISES = (0.05, 0.10, 0.15); TRIALS = 20


def fit(d, S):
    c, m, _ = bin_profile(d, S); g = fit_gaussian(c, m); return g["sigma"]


def plateau_centre(V, T, tied):
    if len(tied) == 1: return int(tied[0])
    e = np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]); keep = np.isin(e[:, 0], tied) & np.isin(e[:, 1], tied)
    pos = -np.ones(len(V), int); pos[tied] = np.arange(len(tied)); ee = pos[e[keep]]
    Adj = coo_matrix((np.ones(len(ee)), (ee[:, 0], ee[:, 1])), shape=(len(tied), len(tied)))
    _, lab = connected_components(Adj, directed=False); big = tied[lab == np.bincount(lab).argmax()]
    return int(big[np.argmin(np.linalg.norm(V[big] - V[big].mean(0), axis=1))])


def run_mesh(f):
    z = np.load(f, allow_pickle=True); mid = int(z["map_id"]); pf = os.path.join(OUT, "parts", "%03d.csv" % mid)
    if mid in FRAGMENTED or os.path.exists(pf): return
    V, T, _ = largest_component(z["vertices"], z["triangles"].astype(int))
    v0 = centroid_vertex(V); d0 = ExactSolver(V, T)(v0); d0 = np.where(np.isfinite(d0), d0, np.nan); heat = HeatSolver(V, T); rows = []
    for s0 in SIGMAS:
        for eta in NOISES:
            for t in range(TRIALS):
                rng = np.random.default_rng(1_000_000 * mid + int(1000 * s0) + int(100 * eta * 100) + t)
                S = A * np.exp(-np.nan_to_num(d0) ** 2 / (2 * s0 ** 2)) + B + rng.normal(0, eta * A, len(V))
                S = np.clip(S, 50, 100); S[np.isnan(d0)] = np.nan; S[rng.random(len(S)) < 0.05] = np.nan
                mx = np.nanmax(S); tied = np.where(S == mx)[0]
                oa = int(np.nanargmax(S)); ob = plateau_centre(V, T, tied); oc = int(rng.choice(tied))
                sa = fit(heat(oa), S); sb = sa if ob == oa else fit(heat(ob), S); sc = sa if oc == oa else (sb if oc == ob else fit(heat(oc), S))
                rows.append(dict(map_id=mid, sigma0=s0, noise=eta, trial=t, n_tied=len(tied), max_is_100=bool(mx >= 100), spread_tied=float(np.nanmax(d0[tied]) - np.nanmin(d0[tied])),
                                 off_first=float(d0[oa]), off_plateau=float(d0[ob]), off_random=float(d0[oc]), sigma_first=sa, sigma_plateau=sb, sigma_random=sc))
    pd.DataFrame(rows).to_csv(pf, index=False); print("mesh %d done" % mid, flush=True)


def summary():
    d = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(OUT, "parts", "*.csv")))]); d.to_csv(os.path.join(OUT, "v54_raw.csv"), index=False)
    L = ["v54 synthetic ties (%d meshes, %d fields)" % (d.map_id.nunique(), len(d))]
    for (s0, eta), g in d.groupby(["sigma0", "noise"]):
        L.append("sigma0 %2d eta %.2f: tied max in %3.0f %% (max = 100 in %3.0f %%), n_tied median %.0f; offset median first/plateau/random %.1f/%.1f/%.1f mm; bias mean first/plateau/random %+.2f/%+.2f/%+.2f; SD %.2f/%.2f/%.2f" % (
            s0, eta, 100 * (g.n_tied > 1).mean(), 100 * g.max_is_100.mean(), g.n_tied.median(), g.off_first.median(), g.off_plateau.median(), g.off_random.median(),
            (g.sigma_first - s0).mean(), (g.sigma_plateau - s0).mean(), (g.sigma_random - s0).mean(), g.sigma_first.std(), g.sigma_plateau.std(), g.sigma_random.std()))
    open(os.path.join(OUT, "v54_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--summary", action="store_true"); a = ap.parse_args()
    if a.summary: summary()
    else:
        fs = [f for f in sorted(glob.glob(os.path.join(MESHES, "map_*.npz"))) if int(np.load(f, allow_pickle=True)["map_id"]) not in FRAGMENTED][::4]
        for f in fs: run_mesh(f)
