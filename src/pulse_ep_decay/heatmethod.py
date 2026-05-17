"""Geodesic distance on a triangle mesh via the Heat Method.

Wraps :mod:`potpourri3d` (geometry-central by Crane et al.). The Heat
Method approximates the true surface geodesic by solving two linear
systems — the heat equation for a short time step, then a Poisson
equation against the normalised heat-gradient — and is the recommended
default for accurate σ-extraction on cardiac surface meshes.

Reference: Crane, Weischedel, Wardetzky.
*The Heat Method for Distance Computation.* Comm. ACM 60(11), 2017.
https://doi.org/10.1145/3131280
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "heat_distances",
    "MeshHeatSolver",
]


def heat_distances(
    vertices: np.ndarray,
    triangles: np.ndarray,
    origin_idx: int,
) -> np.ndarray:
    """Per-vertex geodesic distance from a single origin using the Heat Method.

    Convenience one-shot wrapper. For repeated queries on the same mesh,
    use :class:`MeshHeatSolver` directly — it factorises the operators
    once per mesh.

    Parameters
    ----------
    vertices : (N, 3) array
    triangles : (M, 3) integer array
    origin_idx : int
        Index of the source vertex.

    Returns
    -------
    (N,) array
        Distances in millimetres.

    Raises
    ------
    ImportError
        If ``potpourri3d`` is not installed.
    """
    try:
        import potpourri3d as pp3d
    except ImportError as exc:
        raise ImportError(
            "potpourri3d is required for the Heat Method. "
            "Install with: pip install potpourri3d"
        ) from exc

    verts = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    solver = pp3d.MeshHeatMethodDistanceSolver(verts, faces)
    return np.asarray(solver.compute_distance(int(origin_idx)))


class MeshHeatSolver:
    """Reusable Heat-Method solver for repeated distance queries on one mesh.

    Use this when you need geodesic distances from many origin vertices on
    the same triangulation — the underlying operator factorisations are
    cached.

    Example
    -------
    >>> solver = MeshHeatSolver(vertices, triangles)
    >>> d_from_v0 = solver.distances(0)
    >>> d_from_v42 = solver.distances(42)
    """

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray) -> None:
        try:
            import potpourri3d as pp3d
        except ImportError as exc:
            raise ImportError(
                "potpourri3d is required for the Heat Method. "
                "Install with: pip install potpourri3d"
            ) from exc
        self._solver = pp3d.MeshHeatMethodDistanceSolver(
            np.asarray(vertices, dtype=np.float64),
            np.asarray(triangles, dtype=np.int64),
        )

    def distances(self, origin_idx: int) -> np.ndarray:
        """Per-vertex geodesic distances from ``origin_idx``."""
        return np.asarray(self._solver.compute_distance(int(origin_idx)))
