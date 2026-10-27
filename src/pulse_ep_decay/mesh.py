"""Mesh geometry and geodesic distance solvers for the decay analysis.

The estimator of the method paper (Willert et al., geodesic decay analysis).  :class:`HeatSolver` wraps
potpourri3d's Heat Method (the solver used for every number in the paper);
:class:`ExactSolver` wraps pygeodesic's exact polyhedral geodesics, used as an
independent ground truth in the validation.  Both are optional dependencies
(potpourri3d is a hard dependency, pygeodesic the optional extra ``exact``);
:func:`heat_solver` falls back to :mod:`pulse_ep.core.geodesic` when
potpourri3d cannot be imported, with a warning:
the two implementations agree to about one percent in distance, which
leaves narrow widths unchanged but can shift wide, poorly determined fits
by tens of percent (map 35 of the paper: 24.6 vs 20.0 mm).
"""

from __future__ import annotations

import warnings

import numpy as np

try:
    import potpourri3d as pp3d
except ImportError:  # pragma: no cover
    pp3d = None

try:
    from pygeodesic import geodesic as _pygeo
except ImportError:  # pragma: no cover
    _pygeo = None


def largest_component(vertices: np.ndarray, triangles: np.ndarray):
    """Restrict mesh to its largest connected component (as in the pipeline's
    fragmented-mesh exclusion). Returns (vertices, triangles, index_map)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(vertices)
    i = np.r_[triangles[:, 0], triangles[:, 1], triangles[:, 2]]
    j = np.r_[triangles[:, 1], triangles[:, 2], triangles[:, 0]]
    adj = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    keep_label = np.bincount(labels).argmax()
    keep = labels == keep_label
    new_index = -np.ones(n, dtype=int)
    new_index[keep] = np.arange(keep.sum())
    tri_keep = keep[triangles].all(axis=1)
    return vertices[keep], new_index[triangles[tri_keep]], new_index


def mean_edge_length(vertices: np.ndarray, triangles: np.ndarray) -> float:
    e = np.vstack(
        [
            vertices[triangles[:, 0]] - vertices[triangles[:, 1]],
            vertices[triangles[:, 1]] - vertices[triangles[:, 2]],
            vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
        ]
    )
    return float(np.linalg.norm(e, axis=1).mean())


def centroid_vertex(vertices: np.ndarray) -> int:
    c = vertices.mean(axis=0)
    return int(np.argmin(np.linalg.norm(vertices - c, axis=1)))


def nearest_vertex(vertices: np.ndarray, point: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(vertices - np.asarray(point), axis=1)))


class HeatSolver:
    """Heat-method geodesic distances (as used in the manuscript)."""

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        if pp3d is None:
            raise ImportError("HeatSolver needs potpourri3d: pip install potpourri3d")
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        self.triangles = np.ascontiguousarray(triangles, dtype=np.int64)
        self._solver = pp3d.MeshHeatMethodDistanceSolver(self.vertices, self.triangles)
        self._cache: dict[int, np.ndarray] = {}

    def __call__(self, source: int) -> np.ndarray:
        source = int(source)
        if source not in self._cache:  # argmax origins repeat across trials
            self._cache[source] = np.asarray(self._solver.compute_distance(source))
        return self._cache[source]


class ExactSolver:
    """Exact polyhedral geodesics (Mitchell-Mount-Papadimitriou, Kirsanov
    implementation via pygeodesic). Independent of the Heat Method."""

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        if _pygeo is None:
            raise ImportError("ExactSolver needs pygeodesic: pip install 'pulse-ep-decay[exact]'")
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        self.triangles = np.ascontiguousarray(triangles, dtype=np.int64)
        self._alg = _pygeo.PyGeodesicAlgorithmExact(self.vertices, self.triangles)

    def __call__(self, source: int) -> np.ndarray:
        d, _ = self._alg.geodesicDistances(np.array([int(source)]), None)
        d = np.asarray(d, dtype=float)
        d[~np.isfinite(d)] = np.inf
        return d


class CoreHeatSolver:
    """Heat-Method distances from :mod:`pulse_ep.core.geodesic` (pure scipy).

    Same interface as :class:`HeatSolver`; distances differ from potpourri3d's
    at the percent level (different time step and linear solver), so the
    paper's numbers are reproduced exactly only with :class:`HeatSolver`.
    """

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        from pulse_ep.core.geodesic import heat_geodesic

        self.vertices = self._V = np.ascontiguousarray(vertices, dtype=float)
        self.triangles = self._T = np.ascontiguousarray(triangles, dtype=int)
        self._fn = heat_geodesic
        self._cache: dict[int, np.ndarray] = {}

    def __call__(self, source: int) -> np.ndarray:
        source = int(source)
        if source not in self._cache:
            self._cache[source] = np.asarray(self._fn(self._V, self._T, [source]), dtype=float)
        return self._cache[source]


def heat_solver(vertices: np.ndarray, triangles: np.ndarray):
    """The paper's solver when potpourri3d is installed, else the core fallback (with a warning)."""
    if pp3d is not None:
        return HeatSolver(vertices, triangles)
    warnings.warn(
        "potpourri3d not installed: using pulse_ep.core.geodesic (about 1 % different distances)",
        stacklevel=2,
    )
    return CoreHeatSolver(vertices, triangles)
