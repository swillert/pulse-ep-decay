"""Shared fixtures: synthetic meshes for the test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def unit_triangle() -> tuple[np.ndarray, np.ndarray]:
    """A single right-angle triangle in the xy-plane."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    return vertices, triangles


@pytest.fixture
def small_strip() -> tuple[np.ndarray, np.ndarray]:
    """Two triangles sharing one edge — a unit square in the xy-plane."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return vertices, triangles


@pytest.fixture
def synthetic_sphere() -> tuple[np.ndarray, np.ndarray]:
    """A modest icosphere-like sphere built without pyvista.

    Returns ``(vertices, triangles)`` of an icosphere with ~162 vertices —
    enough resolution to make the Heat-Method and σ-recovery tests
    meaningful while keeping the suite fast.
    """
    # Icosahedron base
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )

    # Subdivide twice (icosphere)
    cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int, verts: list[np.ndarray]) -> int:
        key = (min(a, b), max(a, b))
        if key in cache:
            return cache[key]
        mid = (verts[a] + verts[b]) / 2.0
        mid = mid / np.linalg.norm(mid)  # project onto sphere
        verts.append(mid)
        cache[key] = len(verts) - 1
        return cache[key]

    vs = [v / np.linalg.norm(v) for v in verts]  # normalise
    for _ in range(2):
        new_faces = []
        for a, b, c in faces:
            ab = midpoint(int(a), int(b), vs)
            bc = midpoint(int(b), int(c), vs)
            ca = midpoint(int(c), int(a), vs)
            new_faces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        faces = np.array(new_faces, dtype=np.int64)
        cache.clear()

    vertices = np.asarray(vs, dtype=np.float64) * 25.0  # 25 mm radius
    return vertices, faces
