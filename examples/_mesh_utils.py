"""Shared mesh helper for the example scripts.

Generates a deterministic ~642-vertex icosphere without depending on
``pyvista`` (which has heavy native deps and is overkill for a demo).
Equivalent to the ``icosphere`` helper of the test suite —
duplicated here so the examples stand on their own.
"""

from __future__ import annotations

import numpy as np

__all__ = ["icosphere"]


def icosphere(*, subdivisions: int = 3, radius_mm: float = 25.0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vertices, triangles)`` for a unit icosphere scaled to ``radius_mm``.

    ``subdivisions=3`` yields 642 vertices and 1280 triangles, which is
    a good demo size: large enough for Heat-Method to produce smooth
    geodesics, small enough to run in seconds without a GPU.
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int64,
    )

    cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int, vs: list[np.ndarray]) -> int:
        key = (min(a, b), max(a, b))
        if key in cache:
            return cache[key]
        mid = (vs[a] + vs[b]) / 2.0
        mid = mid / np.linalg.norm(mid)  # back onto unit sphere
        vs.append(mid)
        cache[key] = len(vs) - 1
        return cache[key]

    vs = [v / np.linalg.norm(v) for v in verts]
    for _ in range(subdivisions):
        new_faces = []
        for a, b, c in faces:
            ab = midpoint(int(a), int(b), vs)
            bc = midpoint(int(b), int(c), vs)
            ca = midpoint(int(c), int(a), vs)
            new_faces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        faces = np.array(new_faces, dtype=np.int64)
        cache.clear()

    vertices = np.asarray(vs, dtype=np.float64) * float(radius_mm)
    return vertices, faces
