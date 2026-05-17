"""Python helper for the R σ-fit demo.

Builds the example icosphere, injects a synthetic Gaussian field with a
known σ₀, computes Heat-Method geodesic distances from the detected
score-maximum (matching the published pipeline), and writes a CSV with
the per-vertex ``(geodesic_distance_mm, score)`` cloud.

The CSV starts with a YAML-front-matter-style comment block recording
the synthesis parameters so the R script can quote the bias directly::

    # sigma_true_mm: 8.0
    # noise_level: 0.05
    # missing_fraction: 0.05
    # found_origin: 123
    # n_vertices: 642
    geodesic_distance_mm,score

Run::

    python export_radial_profile.py --sigma-true 8.0 --noise 0.05 \\
        --output radial_profile.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Allow running this script from anywhere — make _mesh_utils importable.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from _mesh_utils import icosphere  # noqa: E402

from pulse_ep_decay import MeshHeatSolver, generate_synthetic_field  # noqa: E402


def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Export the per-vertex (geodesic distance, score) cloud of "
            "a synthetic σ-recovery experiment to CSV, so an external "
            "stack (e.g. R) can re-fit the Gaussian decay."
        )
    )
    p.add_argument("--sigma-true", type=float, default=8.0, help="Injected σ₀ in mm.")
    p.add_argument("--noise", type=float, default=0.05, help="Noise level (fraction of amplitude).")
    p.add_argument(
        "--missing-fraction",
        type=float,
        default=0.05,
        help="Fraction of vertices marked as missing.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("radial_profile.csv"),
        help="Destination CSV path.",
    )
    p.add_argument(
        "--subdivisions",
        type=int,
        default=2,
        help="Icosphere subdivision level (default: 2 → 642 vertices).",
    )
    p.add_argument("--radius-mm", type=float, default=25.0, help="Icosphere radius in mm.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _argparser().parse_args(argv)

    vertices, triangles = icosphere(subdivisions=args.subdivisions, radius_mm=args.radius_mm)
    solver = MeshHeatSolver(vertices, triangles)

    # True origin: vertex closest to +x pole — same convention as the notebook.
    true_origin = int(np.argmax(vertices[:, 0]))
    geo_dist_true = solver.distances(true_origin)

    rng = np.random.default_rng(args.seed)
    scores = generate_synthetic_field(
        geo_dist_true,
        sigma_true_mm=args.sigma_true,
        noise_level=args.noise,
        missing_fraction=args.missing_fraction,
        rng=rng,
    )

    # Detected origin = argmax(score); matches `recover_sigma`.
    s_clean = np.where(np.isnan(scores), 0.0, scores)
    found_origin = int(np.argmax(s_clean))
    if found_origin == true_origin:
        geo_dist_found = geo_dist_true
    else:
        geo_dist_found = solver.distances(found_origin)

    out: Path = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        fh.write(f"# sigma_true_mm: {args.sigma_true}\n")
        fh.write(f"# noise_level: {args.noise}\n")
        fh.write(f"# missing_fraction: {args.missing_fraction}\n")
        fh.write(f"# found_origin: {found_origin}\n")
        fh.write(f"# true_origin: {true_origin}\n")
        fh.write(f"# n_vertices: {len(vertices)}\n")
        writer = csv.writer(fh)
        writer.writerow(["geodesic_distance_mm", "score"])
        for d, s in zip(geo_dist_found, scores, strict=True):
            writer.writerow([f"{d:.6f}", "" if np.isnan(s) else f"{s:.6f}"])

    n_valid = int(np.sum(np.isfinite(geo_dist_found) & ~np.isnan(scores)))
    print(f"Wrote {out}")
    print(f"  σ₀ = {args.sigma_true} mm, noise = {args.noise}")
    print(f"  {n_valid}/{len(vertices)} vertices with finite (d, s); ready for R fit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
