"""Mesh edge graph (length-weighted), shared by the study scripts and :mod:`legacy`."""

from __future__ import annotations

import numpy as np


def edge_graph(vertices, triangles):
    """Sparse graph of the mesh edges weighted by edge length (each edge once)."""
    from scipy.sparse import csr_matrix

    T = np.asarray(triangles)
    e = np.unique(np.sort(np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1), axis=0)
    w = np.linalg.norm(vertices[e[:, 0]] - vertices[e[:, 1]], axis=1)
    n = len(vertices)
    return csr_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))


# kept importable from here for the study scripts (v47a); see pulse_ep_decay.legacy
from pulse_ep_decay.legacy import ORIGIN_SMOOTH_RADIUS_MM, smoothed_argmax  # noqa: E402, F401
