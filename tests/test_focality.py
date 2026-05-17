"""Focality / cluster analysis tests."""

from __future__ import annotations

import numpy as np
import pytest

from pulse_ep_decay import compute_focality, generate_synthetic_field
from pulse_ep_decay.geodesic import build_adjacency_matrix, dijkstra_distances

pytest.importorskip("potpourri3d")  # we use the sphere fixture + dijkstra is enough


def test_single_focal_blob_is_concentrated(synthetic_sphere) -> None:
    """One Gaussian peak on a sphere is one focal cluster."""
    vertices, triangles = synthetic_sphere
    d = dijkstra_distances(
        vertices, triangles, origin_idx=0,
        adjacency=build_adjacency_matrix(vertices, triangles),
    )
    scores = generate_synthetic_field(d, sigma_true_mm=5.0, noise_level=0.0)

    result = compute_focality(triangles, scores, percentile=95.0)
    assert result.n_clusters == 1
    assert result.largest_cluster_pct == pytest.approx(100.0)
    assert result.is_multifocal is False


def test_two_distant_peaks_are_multifocal(synthetic_sphere) -> None:
    """Two Gaussians on opposite sides → at least 2 clusters above the percentile."""
    vertices, triangles = synthetic_sphere
    adj = build_adjacency_matrix(vertices, triangles)
    # Find antipodes: vertex 0 and the one farthest from it
    d0 = dijkstra_distances(vertices, triangles, 0, adjacency=adj)
    antipode = int(np.argmax(d0[np.isfinite(d0)]))
    d_anti = dijkstra_distances(vertices, triangles, antipode, adjacency=adj)

    s0 = generate_synthetic_field(d0, sigma_true_mm=4.0, noise_level=0.0)
    s1 = generate_synthetic_field(d_anti, sigma_true_mm=4.0, noise_level=0.0)
    scores = np.nanmax(np.vstack([s0, s1]), axis=0)

    result = compute_focality(triangles, scores, percentile=97.0)
    assert result.n_clusters >= 2
    assert result.is_multifocal is True
