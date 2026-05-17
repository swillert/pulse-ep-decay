"""Dijkstra-baseline geodesic distance tests."""

from __future__ import annotations

import numpy as np

from pulse_ep_decay import build_adjacency_matrix, dijkstra_distances


def test_single_triangle_distances(unit_triangle) -> None:
    vertices, triangles = unit_triangle
    d = dijkstra_distances(vertices, triangles, origin_idx=0)
    # vertex 0 -> 0 is zero; -> 1 is 1.0; -> 2 is 1.0
    assert d[0] == 0.0
    assert d[1] == 1.0
    assert d[2] == 1.0


def test_unit_square_diagonal(small_strip) -> None:
    """On a unit square split into two triangles along the diagonal
    [0, 2], the shortest *edge* path 0 -> 2 has length sqrt(2)."""
    vertices, triangles = small_strip
    d = dijkstra_distances(vertices, triangles, origin_idx=0)
    assert np.isclose(d[0], 0.0)
    assert np.isclose(d[1], 1.0)
    assert np.isclose(d[2], np.sqrt(2.0))
    assert np.isclose(d[3], 1.0)


def test_adjacency_is_symmetric(small_strip) -> None:
    vertices, triangles = small_strip
    adj = build_adjacency_matrix(vertices, triangles)
    dense = adj.toarray()
    np.testing.assert_allclose(dense, dense.T)


def test_dijkstra_unreachable_returns_inf() -> None:
    """A mesh with two disconnected components: distances to the other
    component must be inf."""
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [10, 10, 10],
            [11, 10, 10],
            [10, 11, 10],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    d = dijkstra_distances(vertices, triangles, origin_idx=0)
    assert np.isfinite(d[:3]).all()
    assert np.isinf(d[3:]).all()
