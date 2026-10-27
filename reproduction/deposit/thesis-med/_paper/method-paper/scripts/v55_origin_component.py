"""v55 — origin vs. largest component of the 95th-percentile set (submitted focality definition), per clinical map.
Reproduces column origin_in_main of revision_2026-10/figs/_cache/origin_component_check.csv (ad-hoc check of 25 Oct)
from the per-map cache of make_revision_figures.py, so that it can be regenerated after the R1 tie rule.
Usage: v55_origin_component.py <out.csv>"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

MP = os.path.expanduser("~/mnt/IdeaProjects/thesis-med/_paper/method-paper"); CACHE = sys.argv[2] if len(sys.argv) > 2 else os.path.join(MP, "revision_2026-10/figs/_cache")
rows = []
for f in sorted(glob.glob(os.path.join(CACHE, "map_*.npz"))):
    z = np.load(f, allow_pickle=True); S, F, o = z["S"].astype(float), z["F"].astype(int), int(z["origin"])
    S0 = np.where(np.isnan(S), 0.0, S); top = S0 >= np.percentile(S0, 95); H = np.where(top)[0]
    pos = -np.ones(len(S), int); pos[H] = np.arange(len(H))
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); e = pos[e[top[e].all(axis=1)]]
    G = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(H), len(H)))
    nc, lab = connected_components(G, directed=False); sizes = np.bincount(lab)
    k_main = sizes.argmax(); inmain = bool(top[o] and lab[pos[o]] == k_main)
    share_o = float(sizes[lab[pos[o]]] / len(H)) if top[o] else 0.0
    A, B, d = float(z["A"]), float(z["B"]), z["d"]; main = H[lab == k_main]
    rows.append(dict(map_id=int(z["map_id"]), sigma=float(z["sigma"]), r2=float(z["r2"]), F=float(sizes.max() / len(H)), origin_in_main=inmain, share_o=share_o,
                     main_min_dist=float(np.nanmin(d[main])), main_max_score_rel=float((np.nanmax(S[main]) - B) / A), thr95_rel=float((np.percentile(S0, 95) - B) / A)))
pd.DataFrame(rows).to_csv(sys.argv[1], index=False); print(len(rows), "maps; origin not in largest component:", sum(not r["origin_in_main"] for r in rows))
