"""Baseline geodesic distance on a triangle mesh via Dijkstra on the edge graph.

The Dijkstra estimate is a strict upper bound for the true surface
geodesic and biases σ-fits high relative to the Heat-Method ground truth.
Kept here both as a reference implementation and as a comparison baseline
for :mod:`pulse_ep_decay.heatmethod`.

All functions take plain NumPy arrays and have no dependency on the
pulse-ep platform.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import shortest_path

__all__ = [
    "build_adjacency_matrix",
    "dijkstra_distances",
]


def build_adjacency_matrix(vertices: np.ndarray, triangles: np.ndarray) -> csr_matrix:
    """Sparse adjacency matrix with Euclidean edge weights.

    Each undirected edge of the triangulation is stored with the
    Euclidean length between its two endpoints. Edges shared by multiple
    triangles keep the minimum encountered length (defensive against
    duplicate triangles).

    Parameters
    ----------
    vertices : (N, 3) array
        Triangle-mesh vertex positions, in millimetres.
    triangles : (M, 3) integer array
        Per-triangle vertex indices (0-based).

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric (N, N) weighted adjacency.
    """
    n_verts = len(vertices)
    adj = lil_matrix((n_verts, n_verts), dtype=np.float64)

    for tri in triangles:
        for i in range(3):
            v0 = tri[i]
            v1 = tri[(i + 1) % 3]
            dist = float(np.linalg.norm(vertices[v0] - vertices[v1]))
            if adj[v0, v1] == 0 or dist < adj[v0, v1]:
                adj[v0, v1] = dist
                adj[v1, v0] = dist

    return adj.tocsr()


def dijkstra_distances(
    vertices: np.ndarray,
    triangles: np.ndarray,
    origin_idx: int,
    *,
    adjacency: csr_matrix | None = None,
) -> np.ndarray:
    """Per-vertex geodesic distance from a single origin using Dijkstra.

    Parameters
    ----------
    vertices : (N, 3) array
    triangles : (M, 3) integer array
    origin_idx : int
        Index of the source vertex.
    adjacency : optional
        Precomputed adjacency from :func:`build_adjacency_matrix`. Useful
        when computing distances from many origins on the same mesh
        (avoids the O(M) rebuild each time).

    Returns
    -------
    (N,) array
        Distances in millimetres; vertices unreachable from the source
        contain ``np.inf``.
    """
    if adjacency is None:
        adjacency = build_adjacency_matrix(vertices, triangles)
    return shortest_path(adjacency, method="D", directed=False, indices=origin_idx)
