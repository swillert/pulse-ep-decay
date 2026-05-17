"""``pulse-ep-decay-validate-sigma`` — synthetic σ-recovery experiment.

Generates synthetic Gaussian score fields with known σ₀ on a given
mesh, re-runs the analysis pipeline, and writes per-trial and
aggregated-bias CSVs. This is the experiment that backs the method paper.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from pulse_ep_decay import run_sigma_recovery


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pulse-ep-decay-validate-sigma",
        description="Synthetic σ-recovery experiment on a triangle mesh.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--mesh", type=Path, help="NPZ with arrays 'vertices' (N,3) and 'triangles' (M,3)"
    )
    src.add_argument(
        "--pulse-ep-map-id",
        type=int,
        help="Map id to pull from the pulse-ep database (requires pulse-ep extra)",
    )

    parser.add_argument("--method", choices=["heat", "dijkstra"], default="heat")
    parser.add_argument(
        "--sigma",
        type=float,
        nargs="+",
        default=[3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0],
        help="σ₀ values to sweep, mm",
    )
    parser.add_argument(
        "--noise",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.10, 0.15],
        help="Relative noise levels to sweep",
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--missing-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("./decay_profiles"))

    args = parser.parse_args(argv)

    # Load mesh
    if args.mesh is not None:
        if not args.mesh.exists():
            parser.error(f"mesh file not found: {args.mesh}")
        data = np.load(args.mesh)
        vertices = data["vertices"]
        triangles = data["triangles"]
        source_label = args.mesh.stem
    else:
        try:
            from pulse_ep.core.database import get_db_session
            from pulse_ep.core.models import EPMapModel
        except ImportError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        with get_db_session() as s:
            epmap = s.query(EPMapModel).filter_by(id=args.pulse_ep_map_id).first()
            if epmap is None:
                print(f"Map {args.pulse_ep_map_id} not found", file=sys.stderr)
                return 1
            vertices = np.array(epmap.vertices).reshape(-1, 3)
            triangles = np.array(epmap.triangles).reshape(-1, 3)
            source_label = f"map_{args.pulse_ep_map_id}"

    print(f"Mesh '{source_label}': {len(vertices)} vertices, {len(triangles)} triangles")
    print(f"Geodesic method: {args.method}")
    print(f"σ-sweep: {args.sigma} mm")
    print(f"Noise-sweep: {args.noise}")
    print(f"Trials per (σ, noise): {args.trials}")

    results = run_sigma_recovery(
        vertices,
        triangles,
        sigma_true_mm=args.sigma,
        noise_levels=args.noise,
        n_trials=args.trials,
        missing_fraction=args.missing_fraction,
        method=args.method,
        seed=args.seed,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    detail_path = args.out / f"validation_{source_label}.csv"
    with detail_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sigma_true_mm",
                "noise_level",
                "trial",
                "sigma_hat_mm",
                "bias_mm",
                "relative_bias_pct",
                "true_origin",
                "found_origin",
                "origin_offset_mm",
                "r_squared",
                "n_bins",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.sigma_true_mm,
                    r.noise_level,
                    r.trial,
                    f"{r.sigma_hat_mm:.3f}" if r.sigma_hat_mm is not None else "",
                    f"{r.bias_mm:+.3f}" if r.bias_mm is not None else "",
                    f"{r.relative_bias_pct:+.2f}" if r.relative_bias_pct is not None else "",
                    r.true_origin,
                    r.found_origin,
                    f"{r.origin_offset_mm:.3f}" if r.origin_offset_mm is not None else "",
                    f"{r.fit.r_squared:.4f}" if r.fit else "",
                    r.fit.n_bins if r.fit else "",
                ]
            )

    # Aggregated bias table
    agg_path = args.out / f"validation_bias_{source_label}.csv"
    biases: dict[tuple[float, float], list[float]] = {}
    for r in results:
        if r.bias_mm is None:
            continue
        biases.setdefault((r.sigma_true_mm, r.noise_level), []).append(r.bias_mm)
    with agg_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sigma_true_mm", "noise_level", "n_valid", "mean_bias_mm", "median_bias_mm", "std_bias_mm"])
        for (sigma0, noise), arr in sorted(biases.items()):
            a = np.array(arr)
            writer.writerow(
                [
                    sigma0,
                    noise,
                    len(a),
                    f"{a.mean():+.3f}",
                    f"{float(np.median(a)):+.3f}",
                    f"{a.std(ddof=1):.3f}" if len(a) > 1 else "",
                ]
            )

    print(f"\nResults: {detail_path}")
    print(f"Bias summary: {agg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
