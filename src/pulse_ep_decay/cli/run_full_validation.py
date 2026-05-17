"""``pulse-ep-decay-full-validation`` — orchestrator for the full validation suite.

Runs the synthetic σ-recovery experiment over an entire list of meshes
(supplied as a directory of NPZ files or pulled from the connected
pulse-ep database), aggregates the per-mesh CSVs, and writes a global
bias summary across all meshes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pulse_ep_decay import run_sigma_recovery


def _iter_meshes(args):
    if args.npz_dir is not None:
        for path in sorted(args.npz_dir.glob("*.npz")):
            data = np.load(path)
            yield path.stem, data["vertices"], data["triangles"]
        return
    # pulse-ep DB path
    try:
        from pulse_ep.core.database import get_db_session
        from pulse_ep.core.models import EPMapModel
    except ImportError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    with get_db_session() as s:
        rows = s.query(EPMapModel).all()
        for r in rows:
            if r.vertices is None or r.triangles is None:
                continue
            yield (
                f"map_{r.id:03d}",
                np.array(r.vertices).reshape(-1, 3),
                np.array(r.triangles).reshape(-1, 3),
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pulse-ep-decay-full-validation",
        description="Run σ-recovery across many meshes and produce a global bias summary.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz-dir", type=Path, help="Directory of mesh NPZ files")
    src.add_argument(
        "--from-pulse-ep",
        action="store_true",
        help="Pull all maps from the pulse-ep database (requires pulse-ep extra)",
    )

    parser.add_argument("--method", choices=["heat", "dijkstra"], default="heat")
    parser.add_argument(
        "--sigma", type=float, nargs="+", default=[3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    )
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("./validation_full"))
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    bias_buckets: dict[tuple[float, float], list[float]] = {}

    for label, vertices, triangles in _iter_meshes(args):
        print(f"\n=== {label}: {len(vertices)} vertices ===")
        try:
            results = run_sigma_recovery(
                vertices,
                triangles,
                sigma_true_mm=args.sigma,
                noise_levels=args.noise,
                n_trials=args.trials,
                method=args.method,
                seed=args.seed,
            )
        except RuntimeError as e:
            print(f"  skipped: {e}")
            continue

        for r in results:
            if r.bias_mm is not None:
                bias_buckets.setdefault((r.sigma_true_mm, r.noise_level), []).append(r.bias_mm)

        summary_rows.append(
            {
                "mesh": label,
                "n_vertices": len(vertices),
                "n_results": len(results),
                "n_valid_fits": sum(1 for r in results if r.sigma_hat_mm is not None),
            }
        )

    # Per-mesh summary
    summary_path = args.out / "mesh_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mesh", "n_vertices", "n_results", "n_valid_fits"])
        writer.writeheader()
        writer.writerows(summary_rows)

    # Aggregated bias across meshes
    bias_path = args.out / "bias_summary.csv"
    with bias_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sigma_true_mm",
                "noise_level",
                "n_valid",
                "mean_bias_mm",
                "median_bias_mm",
                "std_bias_mm",
            ]
        )
        for (sigma0, noise), arr in sorted(bias_buckets.items()):
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

    print(f"\nPer-mesh summary: {summary_path}")
    print(f"Bias summary:     {bias_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
