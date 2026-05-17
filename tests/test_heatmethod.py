"""Heat-Method geodesic distance tests."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("potpourri3d")  # whole module is optional in offline dev installs

from pulse_ep_decay import MeshHeatSolver, heat_distances


def test_heat_distances_on_sphere(synthetic_sphere) -> None:
    """On a 25 mm sphere, geodesic distance to the antipode is ~π·R = 78.5 mm.

    The Heat Method is an approximation, so we allow a 5 % tolerance.
    """
    vertices, triangles = synthetic_sphere
    R = 25.0  # fixture radius
    origin_idx = 0
    antipode = int(
        np.argmin(np.linalg.norm(vertices - (-vertices[origin_idx]), axis=1))
    )
    d = heat_distances(vertices, triangles, origin_idx=origin_idx)
    expected = np.pi * R
    assert d[origin_idx] == pytest.approx(0.0, abs=1e-6)
    assert d[antipode] == pytest.approx(expected, rel=0.05)


def test_solver_reuse(synthetic_sphere) -> None:
    """The cached solver returns the same distances as the one-shot helper."""
    vertices, triangles = synthetic_sphere
    solver = MeshHeatSolver(vertices, triangles)
    d_oneshot = heat_distances(vertices, triangles, origin_idx=10)
    d_solver = solver.distances(10)
    np.testing.assert_allclose(d_oneshot, d_solver, rtol=1e-6)
