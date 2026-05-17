"""End-to-end σ-recovery experiment on triangulated meshes.

Given a mesh, sweeps over prescribed σ₀ values × noise levels × trial
seeds, regenerates a synthetic Gaussian score field for each
combination, re-runs the analysis pipeline (origin detection → geodesic
distances from the *found* origin → Gaussian fit) and reports the bias
σ̂ − σ₀.

This is the experiment described in the method paper and the main
acceptance test for the decay package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal

import numpy as np

from pulse_ep_decay.decay_fit import DecayFitResult, fit_gaussian_decay
from pulse_ep_decay.geodesic import build_adjacency_matrix, dijkstra_distances
from pulse_ep_decay.synthetic import (
    DEFAULT_AMPLITUDE,
    DEFAULT_BASELINE,
    generate_synthetic_field,
)

__all__ = [
    "GeodesicMethod",
    "RecoveryResult",
    "recover_sigma",
    "run_sigma_recovery",
]

GeodesicMethod = Literal["heat", "dijkstra"]


@dataclass(frozen=True)
class RecoveryResult:
    """Outcome of a single σ-recovery trial."""

    sigma_true_mm: float
    sigma_hat_mm: float | None
    bias_mm: float | None
    relative_bias_pct: float | None
    noise_level: float
    missing_fraction: float
    trial: int
    true_origin: int
    found_origin: int
    origin_offset_mm: float | None
    fit: DecayFitResult | None


def _geodesic_callable(
    method: GeodesicMethod,
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> Callable[[int], np.ndarray]:
    """Return a function ``origin_idx -> distances`` for the requested method."""
    if method == "heat":
        from pulse_ep_decay.heatmethod import MeshHeatSolver

        solver = MeshHeatSolver(vertices, triangles)
        return solver.distances
    if method == "dijkstra":
        adjacency = build_adjacency_matrix(vertices, triangles)
        return lambda origin_idx: dijkstra_distances(
            vertices, triangles, origin_idx, adjacency=adjacency
        )
    raise ValueError(f"Unknown geodesic method: {method!r}")


def recover_sigma(
    vertices: np.ndarray,
    triangles: np.ndarray,
    scores: np.ndarray,
    *,
    method: GeodesicMethod = "heat",
    bin_width_mm: float = 2.0,
) -> DecayFitResult | None:
    """End-to-end σ-extraction on a real score field.

    1. Pick the vertex with the highest (NaN-cleaned) score as the origin.
    2. Compute geodesic distances from it.
    3. Fit ``A·exp(-d²/(2σ²)) + B`` to the binned radial profile.

    Returns the :class:`DecayFitResult`, or ``None`` if the fit fails.
    """
    s_clean = np.where(np.isnan(scores), 0.0, scores)
    origin_idx = int(np.argmax(s_clean))
    distances_fn = _geodesic_callable(method, vertices, triangles)
    distances = distances_fn(origin_idx)
    return fit_gaussian_decay(distances, scores, bin_width_mm=bin_width_mm)


def run_sigma_recovery(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    sigma_true_mm: Iterable[float] = (3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0),
    noise_levels: Iterable[float] = (0.0, 0.05, 0.10, 0.15),
    missing_fraction: float = 0.05,
    n_trials: int = 20,
    method: GeodesicMethod = "heat",
    amplitude: float = DEFAULT_AMPLITUDE,
    baseline: float = DEFAULT_BASELINE,
    seed: int = 42,
    bin_width_mm: float = 2.0,
    true_origin: int | None = None,
) -> list[RecoveryResult]:
    """Sweep σ-recovery on one mesh.

    For each (σ₀, noise, trial) combination:

    1. Generate a synthetic field with known σ₀ from the *true* origin.
    2. Re-detect the origin as ``argmax(score)`` (matches the clinical
       pipeline; will move under noise → captures origin-misplacement bias).
    3. Compute geodesic distances from the *detected* origin.
    4. Fit and record σ̂, bias, relative bias.

    Parameters
    ----------
    vertices, triangles
        Mesh arrays.
    sigma_true_mm, noise_levels
        Iterables of parameters to sweep over.
    missing_fraction
        Fraction of vertices to mark as missing per trial.
    n_trials
        Independent random draws per (σ₀, noise) combination.
    method
        ``"heat"`` (default, accurate) or ``"dijkstra"`` (baseline).
    seed
        Reproducible base seed for the trial RNG.
    true_origin
        Optional vertex index used as the synthetic origin. If ``None``,
        the vertex closest to the mesh centroid is used.
    """
    rng = np.random.default_rng(seed)

    if true_origin is None:
        centroid = vertices.mean(axis=0)
        true_origin = int(np.argmin(np.linalg.norm(vertices - centroid, axis=1)))

    distances_fn = _geodesic_callable(method, vertices, triangles)
    geo_dist_true = distances_fn(true_origin)
    n_reachable = int(np.sum(np.isfinite(geo_dist_true)))
    if n_reachable < 100:
        raise RuntimeError(
            f"Too few reachable vertices from origin {true_origin}: {n_reachable}/{len(vertices)}"
        )

    results: list[RecoveryResult] = []

    for sigma0 in sigma_true_mm:
        for noise in noise_levels:
            for trial in range(n_trials):
                scores = generate_synthetic_field(
                    geo_dist_true,
                    sigma0,
                    noise_level=noise,
                    missing_fraction=missing_fraction,
                    amplitude=amplitude,
                    baseline=baseline,
                    rng=rng,
                )
                s_clean = np.where(np.isnan(scores), 0.0, scores)
                found_origin = int(np.argmax(s_clean))
                origin_offset = (
                    float(geo_dist_true[found_origin])
                    if np.isfinite(geo_dist_true[found_origin])
                    else None
                )

                if found_origin == true_origin:
                    geo_dist_found = geo_dist_true
                else:
                    geo_dist_found = distances_fn(found_origin)

                fit = fit_gaussian_decay(geo_dist_found, scores, bin_width_mm=bin_width_mm)

                if fit is not None:
                    bias = fit.sigma_mm - sigma0
                    rel_bias = bias / sigma0 * 100.0
                    sigma_hat = fit.sigma_mm
                else:
                    bias = rel_bias = sigma_hat = None

                results.append(
                    RecoveryResult(
                        sigma_true_mm=float(sigma0),
                        sigma_hat_mm=sigma_hat,
                        bias_mm=bias,
                        relative_bias_pct=rel_bias,
                        noise_level=float(noise),
                        missing_fraction=float(missing_fraction),
                        trial=int(trial),
                        true_origin=true_origin,
                        found_origin=found_origin,
                        origin_offset_mm=origin_offset,
                        fit=fit,
                    )
                )

    return results
