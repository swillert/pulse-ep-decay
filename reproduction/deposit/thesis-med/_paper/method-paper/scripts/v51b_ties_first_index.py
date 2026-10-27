"""v51b — R1 rerun (26 Oct 2026) of v51 after adopting the plateau-centre tie rule (pulse_ep_decay.profile.find_origin).
Now the cached origin IS the plateau centre; the alternative reported is the submitted rule (first tied vertex in
index order, np.nanargmax), i.e. the sensitivity is reversed.  Columns as in v51 (origin_alt / sigma_alt = first index).

Original description (v51):  ties of the maximum score (integer scores) and the argmax origin.

For each of the 55 clinical maps (vertex field, April design; cached by make_revision_figures.py):
  * n_tied: number of vertices sharing the maximum score; spread: largest geodesic distance from the
    current origin (first vertex in index order, np.nanargmax) to another tied vertex;
  * tied components on the mesh edge graph; size of the largest one;
  * alternative tie-aware origin: the vertex of the largest tied component nearest to that component's
    Euclidean centroid (plateau centre); sigma / R2 refitted from it with the unchanged pipeline.
Nothing in the manuscript is changed by this script.
"""
import os, glob, sys
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pulse_ep_decay.mesh import HeatSolver
from pulse_ep_decay.profile import bin_profile, fit_gaussian

MP = os.path.expanduser("~/mnt/IdeaProjects/thesis-med/_paper/method-paper")
CACHE = os.path.join(MP, "revision_2026-10/figs/_cache"); OUT = os.path.join(MP, "supplementary_validation_results/v51_argmax_ties_plateau_2026-10-26"); os.makedirs(OUT, exist_ok=True)


def fit(d, S):
    c, m, n = bin_profile(d, S); r = fit_gaussian(c, m); return r


rows = []
for f in sorted(glob.glob(os.path.join(CACHE, "map_*.npz"))):
    z = np.load(f); mid = int(z["map_id"]); V, F, S, d0, o = z["V"], z["F"], z["S"], z["d"], int(z["origin"])
    mx = np.nanmax(S); tied = np.where(S == mx)[0]
    r = dict(map_id=mid, max_score=float(mx), n_tied=len(tied), n_vertices=len(S), spread_mm=float(d0[tied].max()), sigma=float(z["sigma"]), r2=float(z["r2"]))
    if len(tied) > 1:
        e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); keep = np.isin(e[:, 0], tied) & np.isin(e[:, 1], tied)
        idx = {v: i for i, v in enumerate(tied)}; ee = np.array([[idx[a], idx[b]] for a, b in e[keep]]) if keep.any() else np.zeros((0, 2), int)
        A = coo_matrix((np.ones(len(ee)), (ee[:, 0], ee[:, 1])), shape=(len(tied), len(tied))) if len(ee) else coo_matrix((len(tied), len(tied)))
        nc, lab = connected_components(A, directed=False); sizes = np.bincount(lab); big = tied[lab == sizes.argmax()]
        cen = V[big].mean(0); assert o == int(big[np.argmin(np.linalg.norm(V[big] - cen, axis=1))])  # cache origin = plateau centre
        o2 = int(np.nanargmax(S))  # submitted rule: first tied vertex
        d2 = HeatSolver(V, F)(o2); g = fit(d2, S)
        r.update(n_tied_components=int(nc), largest_tied_component=int(sizes.max()), origin_alt=o2, origin_shift_mm=float(d0[o2]),
                 sigma_alt=float(g["sigma"]), r2_alt=float(g["r2"]), d_sigma=float(g["sigma"]) - r["sigma"])
    rows.append(r); print(mid, r.get("n_tied"), round(r["spread_mm"], 1), round(r.get("d_sigma", 0.0), 2), flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "v51_ties.csv"), index=False)
t = df[df.n_tied > 1]; foc = t[(t.r2 >= 0.7) & (t.sigma < 30)]
lines = ["v51b argmax ties, R1 plateau-centre origin (55 maps)", "maps with tied maximum: %d; of these adequately fitted (R2>=0.7) and sigma<30: %d" % (len(t), len(foc)),
         "n_tied median %s, range %d-%d; spread from the plateau-centre origin median %.1f mm, range %.1f-%.1f" % (t.n_tied.median(), t.n_tied.min(), t.n_tied.max(), t.spread_mm.median(), t.spread_mm.min(), t.spread_mm.max()),
         "submitted rule (first tied vertex) instead of the plateau centre: |d sigma| median %.2f mm; >1 mm in %d maps (%d of them R2>=0.7 & sigma<30)" % (t.d_sigma.abs().median(), (t.d_sigma.abs() > 1).sum(), (foc.d_sigma.abs() > 1).sum()), "",
         t[["map_id", "max_score", "n_tied", "n_tied_components", "largest_tied_component", "spread_mm", "origin_shift_mm", "sigma", "sigma_alt", "r2", "r2_alt"]].round(2).to_string(index=False)]
open(os.path.join(OUT, "v51_summary.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
