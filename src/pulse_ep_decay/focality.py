"""Focality / cluster analysis of the high-score region.

Pace-mapping with multiple foci (e.g. fractionated or fragmented score
patterns) violates the single-Gaussian assumption of the decay fit. This
module quantifies how concentrated the top-percentile region is by
counting connected components in the sub-graph induced on those
vertices.

A ``largest_cluster_pct`` close to 100 % means the top region is one
focal patch; values well below ~70 % indicate a multifocal map that
should be excluded from σ-extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components

__all__ = [
    "FocalityResult",
    "compute_focality",
]


@dataclass(frozen=True)
class FocalityResult:
    """Outcome of the focality analysis.

    Attributes
    ----------
    n_top_vertices : int
        Vertices above the chosen percentile threshold.
    n_clusters : int
        Connected components in the top-vertex sub-graph.
    largest_cluster_pct : float
        Share (0-100) of top vertices that belong to the largest cluster.
    threshold : float
        Score threshold corresponding to the chosen percentile.
    """

    n_top_vertices: int
    n_clusters: int
    largest_cluster_pct: float
    threshold: float

    @property
    def is_multifocal(self) -> bool:
        """``True`` when the largest cluster contains < 70 % of top vertices."""
        return self.largest_cluster_pct < 70.0


def compute_focality(
    triangles: np.ndarray,
    scores: np.ndarray,
    *,
    percentile: float = 95.0,
    min_cluster_size: int = 3,
) -> FocalityResult:
    """Cluster the top-percentile vertices via the mesh edge graph.

    Parameters
    ----------
    triangles : (M, 3) integer array
        Per-triangle vertex indices.
    scores : (N,) array
        Per-vertex scalar field. ``nan`` is treated as 0.
    percentile : float, default 95.0
        Threshold percentile.
    min_cluster_size : int, default 3
        Below this number of top vertices, focality is undefined; the
        result reports zero clusters.
    """
    s = np.where(np.isnan(scores), 0.0, scores)
    threshold = float(np.percentile(s, percentile))
    top_mask = s >= threshold
    top_indices = set(np.where(top_mask)[0])
    n_top = len(top_indices)

    if n_top < min_cluster_size:
        return FocalityResult(
            n_top_vertices=n_top,
            n_clusters=0,
            largest_cluster_pct=0.0,
            threshold=threshold,
        )

    idx_map = {v: i for i, v in enumerate(sorted(top_indices))}
    n_sub = len(idx_map)
    adj = lil_matrix((n_sub, n_sub), dtype=np.float64)

    for tri in triangles:
        if tri[0] in top_indices and tri[1] in top_indices and tri[2] in top_indices:
            for i in range(3):
                v0, v1 = tri[i], tri[(i + 1) % 3]
                i0, i1 = idx_map[int(v0)], idx_map[int(v1)]
                adj[i0, i1] = 1
                adj[i1, i0] = 1

    n_components, labels = connected_components(adj.tocsr(), directed=False)
    cluster_sizes = np.bincount(labels)
    largest_pct = float(cluster_sizes.max() / n_sub * 100.0) if n_components > 0 else 0.0

    return FocalityResult(
        n_top_vertices=n_top,
        n_clusters=int(n_components),
        largest_cluster_pct=largest_pct,
        threshold=threshold,
    )
