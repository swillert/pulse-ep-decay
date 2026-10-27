"""Measured support, origin rule and high-score regions .

The CARTO export colours *every* vertex (fill radius effectively unlimited:
100 % of vertices carry a value, up to 60 mm from the nearest measured point,
the paper's Section 2.2).  The platform's area report masks vertices
farther than 5 mm from the nearest measured point; this module applies the
same mask before the fit.

Origin = best-scoring vertex *on the support* (the nearest vertex of the
best measured point, or a vertex within `radius` of it).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

SUPPORT_RADIUS_MM = 5.0  # platform default for the area report
SUPPORT_RADII_SENS = (5.0, 10.0)


def support_mask(vertices: np.ndarray, points_xyz: np.ndarray, radius: float = SUPPORT_RADIUS_MM):
    """Boolean mask of vertices within `radius` (Euclidean) of the projection
    of any measured point onto the mesh — identical to the platform's area
    masking (points are first snapped to their nearest vertex)."""
    tree_v = cKDTree(vertices)
    _, snap = tree_v.query(points_xyz)
    proj = vertices[np.unique(snap)]
    d, _ = cKDTree(proj).query(vertices)
    return d <= radius


def masked_scores(scores: np.ndarray, mask: np.ndarray):
    s = np.array(scores, dtype=float, copy=True)
    s[~mask] = np.nan
    return s


def component_areas(vertices, triangles, sel_vertex_mask):
    """Areas (cm²) of connected components of triangles whose three vertices
    are all selected. Returns sorted descending list."""
    tri_sel = sel_vertex_mask[triangles].all(axis=1)
    T = triangles[tri_sel]
    if len(T) == 0:
        return []
    a = (
        0.5
        * np.linalg.norm(
            np.cross(vertices[T[:, 1]] - vertices[T[:, 0]], vertices[T[:, 2]] - vertices[T[:, 0]]),
            axis=1,
        )
        / 100.0
    )
    # union-find over shared vertices
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for t in T:
        r0 = find(int(t[0]))
        for v in t[1:]:
            r = find(int(v))
            if r != r0:
                parent[r] = r0
    comp = {}
    for i, t in enumerate(T):
        r = find(int(t[0]))
        comp[r] = comp.get(r, 0.0) + a[i]
    return sorted(comp.values(), reverse=True)


def top_region(scores, frac=0.95):
    theta = np.nanpercentile(scores, frac * 100)
    return np.nan_to_num(scores, nan=-np.inf) >= theta


ORIGIN_SMOOTH_RADIUS_MM = 3.0


def edge_graph(vertices, triangles):
    """Sparse graph of the mesh edges weighted by edge length (each edge once)."""
    from scipy.sparse import csr_matrix

    T = np.asarray(triangles)
    e = np.unique(np.sort(np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1), axis=0)
    w = np.linalg.norm(vertices[e[:, 0]] - vertices[e[:, 1]], axis=1)
    n = len(vertices)
    return csr_matrix(
        (np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)
    )


def smoothed_argmax(
    vertices, scores, radius: float = ORIGIN_SMOOTH_RADIUS_MM, triangles=None
) -> int:
    """Origin (Eq. 1): the vertex whose `radius`-neighbourhood has the highest
    mean score (NaN ignored).  Averaging before the argmax removes
    single-vertex noise; on the synthetic support validation it reduces the
    origin offset from ~sigma/2 to ~sigma/8 and the sigma bias for
    sigma <= 15 mm.

    With `triangles` the neighbourhood is geodesic — vertices within `radius`
    along the mesh edges (Dijkstra on the edge graph), so that a fold or the
    opposite wall cannot enter it; without, it is the Euclidean ball.  On the
    clinical meshes the two neighbourhoods coincide within the support
    (graph/Euclidean distance ratio median 1.00-1.02 at 3 mm, no vertex of a
    Euclidean ball unreachable along the surface).
    """
    finite = np.isfinite(scores)
    cand = np.where(finite)[0]
    sc = scores[cand]
    if triangles is not None:
        from scipy.sparse.csgraph import dijkstra

        # edge graph restricted to the candidate vertices (paths through vertices
        # without a score, i.e. outside the field, are not needed within 3 mm)
        sel = np.zeros(len(vertices), bool)
        sel[cand] = True
        T = np.asarray(triangles)
        T = T[sel[T].all(axis=1)]
        local = -np.ones(len(vertices), int)
        local[cand] = np.arange(len(cand))
        G = edge_graph(vertices[cand], local[T])
        means = np.empty(len(cand))
        chunk = 500  # dijkstra returns a dense (sources x candidates) block; keep it small
        for a in range(0, len(cand), chunk):
            D = dijkstra(G, indices=np.arange(a, min(a + chunk, len(cand))), limit=radius)
            means[a : a + chunk] = [sc[np.isfinite(row)].mean() for row in D]
    else:
        tree = cKDTree(vertices[cand])
        nb = tree.query_ball_point(vertices[cand], radius)
        means = np.array([sc[n].mean() for n in nb])
    return int(cand[int(np.argmax(means))])
