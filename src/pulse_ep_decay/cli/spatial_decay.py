"""``pulse-ep-decay-spatial`` — batch geodesic decay on one or many meshes.

Two input modes:

* ``--mesh path.npz``: a single mesh file with arrays
  ``vertices`` (N, 3), ``triangles`` (M, 3), ``scores`` (N,).
* ``--from-pulse-ep``: iterate over all pace-map studies in the
  connected pulse-ep PostgreSQL database (requires the ``pulse-ep`` extra).

For each map, writes a per-vertex CSV (``decay_map_<id>.csv``) and
appends one row to ``decay_summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from pulse_ep_decay import (
    DecayFitResult,
    compute_focality,
    fit_gaussian_decay,
)
from pulse_ep_decay.validation import _geodesic_callable


def _process_one(
    *,
    map_id: int | str,
    vertices: np.ndarray,
    triangles: np.ndarray,
    scores: np.ndarray,
    method: str,
    output_dir: Path,
    write_per_vertex: bool = True,
) -> dict:
    n_vertices = len(vertices)
    score_clean = np.where(np.isnan(scores), 0.0, scores)
    origin_idx = int(np.argmax(score_clean))
    origin_score = float(score_clean[origin_idx])
    origin_xyz = vertices[origin_idx]

    focality = compute_focality(triangles, scores)

    distances_fn = _geodesic_callable(method, vertices, triangles)  # type: ignore[arg-type]
    geo_dist = distances_fn(origin_idx)
    finite = np.isfinite(geo_dist)
    max_geodesic = float(np.nanmax(geo_dist[finite])) if finite.any() else 0.0

    fit: DecayFitResult | None = fit_gaussian_decay(geo_dist, scores)

    if write_per_vertex:
        per_vertex_path = output_dir / f"decay_map_{map_id}.csv"
        with per_vertex_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["vertex_idx", "geodesic_dist_mm", "score", "x", "y", "z"])
            for vi in range(n_vertices):
                gd = geo_dist[vi] if np.isfinite(geo_dist[vi]) else ""
                sc = scores[vi] if not np.isnan(scores[vi]) else ""
                writer.writerow(
                    [
                        vi,
                        f"{gd:.2f}" if isinstance(gd, float) else gd,
                        f"{sc:.2f}" if isinstance(sc, float) else sc,
                        f"{vertices[vi, 0]:.4f}",
                        f"{vertices[vi, 1]:.4f}",
                        f"{vertices[vi, 2]:.4f}",
                    ]
                )
        print(f"  -> {per_vertex_path}")

    return {
        "map_id": map_id,
        "geodesic_method": method,
        "n_vertices": n_vertices,
        "origin_idx": origin_idx,
        "origin_score": f"{origin_score:.2f}",
        "origin_x": f"{origin_xyz[0]:.2f}",
        "origin_y": f"{origin_xyz[1]:.2f}",
        "origin_z": f"{origin_xyz[2]:.2f}",
        "focality_pct": f"{focality.largest_cluster_pct:.1f}",
        "n_clusters": focality.n_clusters,
        "is_multifocal": focality.is_multifocal,
        "max_geodesic_dist": f"{max_geodesic:.1f}",
        "sigma_mm": f"{fit.sigma_mm:.2f}" if fit else "",
        "fit_amplitude": f"{fit.amplitude:.2f}" if fit else "",
        "fit_baseline": f"{fit.baseline:.2f}" if fit else "",
        "fit_r_squared": f"{fit.r_squared:.4f}" if fit else "",
        "fit_n_bins": fit.n_bins if fit else "",
    }


def _write_summary(rows: list[dict], output_dir: Path) -> Path:
    fieldnames = list(rows[0].keys()) if rows else []
    summary_path = output_dir / "decay_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pulse-ep-decay-spatial",
        description="Compute geodesic decay profiles + Gaussian σ-fit on mesh+score input.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--mesh", type=Path, help="NPZ file with vertices/triangles/scores arrays")
    src.add_argument(
        "--from-pulse-ep",
        action="store_true",
        help="Iterate over all pace-maps in the pulse-ep database (requires pulse-ep extra)",
    )
    parser.add_argument(
        "--method",
        choices=["heat", "dijkstra"],
        default="heat",
        help="Geodesic computation method (default: heat)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./decay_profiles"), help="Where to write CSVs"
    )
    parser.add_argument(
        "--no-per-vertex",
        action="store_true",
        help="Skip the per-map per-vertex CSV; only write the summary",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    if args.mesh is not None:
        if not args.mesh.exists():
            parser.error(f"mesh file not found: {args.mesh}")
        data = np.load(args.mesh)
        for required in ("vertices", "triangles", "scores"):
            if required not in data.files:
                parser.error(f"{args.mesh} is missing required array '{required}'")
        print(f"Processing {args.mesh}")
        row = _process_one(
            map_id=args.mesh.stem,
            vertices=data["vertices"],
            triangles=data["triangles"],
            scores=data["scores"],
            method=args.method,
            output_dir=args.output_dir,
            write_per_vertex=not args.no_per_vertex,
        )
        rows.append(row)
    else:
        try:
            from pulse_ep_decay.integration.pulse_ep import iter_pacemaps
        except ImportError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        for epmap, _attrs, mesh in iter_pacemaps():
            print(f"Processing map {epmap.id}: {epmap.map_name}")
            row = _process_one(
                map_id=f"{epmap.id:03d}",
                vertices=mesh.vertices,
                triangles=mesh.triangles,
                scores=mesh.scores,
                method=args.method,
                output_dir=args.output_dir,
                write_per_vertex=not args.no_per_vertex,
            )
            rows.append(row)

    if not rows:
        print("No maps processed.")
        return 1

    summary_path = _write_summary(rows, args.output_dir)
    print(f"\nSummary: {summary_path} ({len(rows)} map(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
