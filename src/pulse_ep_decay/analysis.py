"""Algorithm 1 of the method paper as one public entry point.

:func:`analyse_map` takes an exported score field on a triangulated surface
(one score per vertex, NaN where no score is available) and returns, for
every map, the quantities reported in the paper: origin, Gaussian width and
fit, exponential fit and the AIC difference, the focality index F on the
half-maximum set, the width class and the two flags, plus a status that
states why a quantity is unavailable.  No per-map uncertainty interval is
returned: the precision figures of the paper are conditional simulation
results, not intervals for an individual map.

Usage::

    from pulse_ep_decay import analyse_map
    res = analyse_map(scores, vertices, faces)
    res["sigma"], res["width_class"], res["flag_r2"], res["flag_F"], res["status"]
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pulse_ep_decay.profile import bin_profile, find_origin, fit_exponential, fit_gaussian

R2_THRESHOLD = 0.7
F_THRESHOLD = 0.70
MIN_H_VERTICES = 3
# width classes of Sec. 5.5 (width range only, not a judgement of fit quality)
CLASS_EDGES_MM = (2.0, 30.0, 50.0, 100.0)


def width_class(sigma: float) -> str:
    """'below range' (< 2 mm), 'in range' (2-30), 'broad' (30-50), 'diffuse' (50-100),
    'n.i.' (>= 100 mm or no width)."""
    if sigma is None or not np.isfinite(sigma) or sigma >= CLASS_EDGES_MM[3]:
        return "n.i."
    if sigma >= CLASS_EDGES_MM[2]:
        return "diffuse"
    if sigma >= CLASS_EDGES_MM[1]:
        return "broad"
    if sigma >= CLASS_EDGES_MM[0]:
        return "in range"
    return "below range"


def edge_graph(faces: np.ndarray, n: int):
    T = np.asarray(faces, int)
    e = np.unique(np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1), axis=0)
    return coo_matrix(
        (np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)
    ).tocsr()


def focality_index(scores, faces, origin, A, B, graph=None):
    """F (Eq. 10): share of the half-maximum set H = {v : s_v >= B + A/2} (observed scores,
    vertex counts, mesh edge graph) that lies in the connected component containing the
    origin.  If the origin itself lies below the threshold (fitted peak above the observed
    maximum), the component of the highest vertex in H is used.
    F is unavailable (NaN, with a status) when the fit failed, the fitted amplitude is not
    positive, or H contains fewer than three vertices (MIN_H_VERTICES).  This function is the
    single implementation used by the public API and by the study scripts (v62).
    Returns (F, n_components, status)."""
    s = np.asarray(scores, float)
    if not (np.isfinite(A) and np.isfinite(B)):
        return np.nan, 0, "F unavailable: fit failed"
    if A <= 0:
        return np.nan, 0, "F unavailable: non-positive amplitude"
    G = edge_graph(faces, len(s)) if graph is None else graph
    H = np.flatnonzero(np.isfinite(s) & (s >= B + 0.5 * A))
    if len(H) < MIN_H_VERTICES:
        return np.nan, 0, "F unavailable: fewer than %d vertices in the half-maximum set" % MIN_H_VERTICES
    _, lab = connected_components(G[H][:, H], directed=False)
    sz = np.bincount(lab)
    pos = np.flatnonzero(H == origin)
    k = lab[pos[0]] if len(pos) else lab[int(np.argmax(s[H]))]
    return float(sz[k] / len(H)), int(len(sz)), "ok"


def analyse_map(scores, vertices, faces, dist_solver=None, origin: int | None = None) -> dict:
    """Algorithm 1: origin (plateau-centre rule) -> geodesic distances (Heat Method) ->
    2-mm binning -> Gaussian fit (and exponential fit for Delta AIC) -> F -> class and flags.

    scores   : (n,) score per vertex, NaN where unavailable (vertices of the largest component)
    vertices : (n, 3) mm;  faces : (m, 3) int
    dist_solver : callable origin -> (n,) distances; default HeatSolver(vertices, faces)
    origin   : optional fixed origin vertex (known-source mode)

    Returns a dict with origin, sigma, A, B, r2, n_bins, aic_gauss, aic_exp, dAIC
    (= AIC_Gauss - AIC_exp; > 2 favours the exponential), lam_exp, F, n_components,
    width_class, flag_r2 (R2 < 0.7), flag_F (F < 0.70 with R2 >= 0.7; F is not
    interpreted when R2 < 0.7), and status ('ok' or the reason a quantity is unavailable)."""
    s = np.asarray(scores, float)
    V = np.asarray(vertices, float)
    T = np.asarray(faces, int)
    if dist_solver is None:
        from pulse_ep_decay.mesh import HeatSolver

        dist_solver = HeatSolver(V, T)
    if origin is None:
        origin = find_origin(s, V, T)
    d = np.asarray(dist_solver(origin), float)
    c, m, _ = bin_profile(d, s)
    out = dict(origin=int(origin), n_bins=int(len(c)))
    status = []
    if len(c) < 4:
        g = dict(A=np.nan, sigma=np.nan, B=np.nan, r2=np.nan, aic=np.nan, converged=False)
        e = dict(lam=np.nan, aic=np.nan)
        status.append("fit failed: fewer than 4 bins")
    else:
        g, e = fit_gaussian(c, m), fit_exponential(c, m)
        if not g["converged"]:
            status.append("fit failed: Gaussian did not converge")
        if not e["converged"]:
            status.append("exponential fit failed: lam_exp and dAIC unavailable")
    out.update(sigma=float(g["sigma"]), A=float(g["A"]), B=float(g["B"]), r2=float(g["r2"]),
               aic_gauss=float(g["aic"]), aic_exp=float(e["aic"]), lam_exp=float(e["lam"]))
    out["dAIC"] = out["aic_gauss"] - out["aic_exp"]
    F, ncomp, fstat = focality_index(s, T, origin, out["A"], out["B"])
    out.update(F=F, n_components=ncomp)
    if fstat != "ok":
        status.append(fstat)
    out["width_class"] = width_class(out["sigma"]) if g["converged"] else "n.i."
    out["flag_r2"] = bool(not np.isfinite(out["r2"]) or out["r2"] < R2_THRESHOLD)
    out["flag_F"] = bool(np.isfinite(F) and F < F_THRESHOLD and not out["flag_r2"])
    out["status"] = "ok" if not status else "; ".join(status)
    return out
