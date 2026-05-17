"""``pulse-ep-decay-compare-geodesic`` — Dijkstra vs Heat Method on the same mesh.

Reports per-vertex relative error of Dijkstra (the upper-bound graph
approximation) against the Heat-Method ground truth, and the resulting
σ̂ difference when both are fed into the Gaussian decay fit.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pulse_ep_decay import fit_gaussian_decay, generate_synthetic_field
from pulse_ep_decay.geodesic import build_adjacency_matrix, dijkstra_distances
from pulse_ep_decay.heatmethod import heat_distances


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pulse-ep-decay-compare-geodesic",
        description="Dijkstra vs Heat-Method geodesics on a mesh + σ̂ difference",
    )
    parser.add_argument("--mesh", type=Path, required=True, help="NPZ with vertices/triangles")
    parser.add_argument(
        "--origin",
        type=int,
        default=None,
        help="Origin vertex index (default: vertex closest to centroid)",
    )
    parser.add_argument(
        "--sigma-true",
        type=float,
        default=8.0,
        help="σ₀ for the synthetic Gaussian field used to drive the σ-fit comparison",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("./decay_profiles"))
    args = parser.parse_args(argv)

    data = np.load(args.mesh)
    vertices = data["vertices"]
    triangles = data["triangles"]

    if args.origin is None:
        centroid = vertices.mean(axis=0)
        args.origin = int(np.argmin(np.linalg.norm(vertices - centroid, axis=1)))

    print(f"Mesh: {len(vertices)} vertices, {len(triangles)} triangles")
    print(f"Origin vertex: {args.origin}")

    d_heat = heat_distances(vertices, triangles, args.origin)
    adj = build_adjacency_matrix(vertices, triangles)
    d_dijk = dijkstra_distances(vertices, triangles, args.origin, adjacency=adj)

    finite = np.isfinite(d_heat) & np.isfinite(d_dijk) & (d_heat > 0)
    rel_err = (d_dijk[finite] - d_heat[finite]) / d_heat[finite]
    print(f"Reachable common vertices: {int(finite.sum())}/{len(vertices)}")
    print(
        "Dijkstra/Heat relative error:  "
        f"mean={rel_err.mean():+.4f}  median={float(np.median(rel_err)):+.4f}  "
        f"p95={float(np.quantile(rel_err, 0.95)):+.4f}  max={rel_err.max():+.4f}"
    )

    # Drive the σ-fit on a synthetic field generated from the Heat geodesics
    rng = np.random.default_rng(args.seed)
    scores = generate_synthetic_field(d_heat, args.sigma_true, noise_level=0.05, rng=rng)
    fit_heat = fit_gaussian_decay(d_heat, scores)
    fit_dijk = fit_gaussian_decay(d_dijk, scores)
    print(
        f"\nσ̂ fitted on Heat distances:     "
        f"{fit_heat.sigma_mm:.2f} mm  (R²={fit_heat.r_squared:.3f})"
        if fit_heat
        else "Heat fit failed"
    )
    print(
        f"σ̂ fitted on Dijkstra distances: {fit_dijk.sigma_mm:.2f} mm  (R²={fit_dijk.r_squared:.3f})"
        if fit_dijk
        else "Dijkstra fit failed"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"geodesic_comparison_{args.mesh.stem}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vertex_idx", "d_heat_mm", "d_dijkstra_mm", "rel_err"])
        for vi in range(len(vertices)):
            if finite[vi]:
                w.writerow(
                    [
                        vi,
                        f"{d_heat[vi]:.4f}",
                        f"{d_dijk[vi]:.4f}",
                        f"{(d_dijk[vi] - d_heat[vi]) / d_heat[vi]:+.6f}",
                    ]
                )
    print(f"Per-vertex CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
