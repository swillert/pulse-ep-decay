"""Export the per-vertex (geodesic distance, score) cloud of a synthetic
sigma-recovery experiment to CSV, so that another stack (here: R) can re-fit
the Gaussian decay independently.

The Python side is used only for what R cannot do easily -- the Heat-Method
geodesic distances on the mesh. The field follows the manuscript protocol
(A = 35, B = 65, noise, clipping to [50, 100], 5 % unscored vertices) and the
origin is detected as in the pipeline (maximum, plateau-centre tie rule). The
header records the injected sigma_0 and the width the Python pipeline
recovers, so the R script can quote both.

    python export_radial_profile.py --sigma-true 8.0 --noise 0.05 --output radial_profile.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mesh_utils import icosphere  # noqa: E402

from pulse_ep_decay import GAUSS_FWHM_FACTOR, HeatSolver, analyse_map, synth_field  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sigma-true", type=float, default=8.0, help="injected sigma_0 in mm")
    p.add_argument("--noise", type=float, default=0.05, help="noise SD as a fraction of the amplitude")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subdivisions", type=int, default=3, help="icosphere subdivisions (3 -> 642 vertices)")
    p.add_argument("--radius-mm", type=float, default=25.0)
    p.add_argument("--output", type=Path, default=Path("radial_profile.csv"))
    a = p.parse_args(argv)

    V, T = icosphere(subdivisions=a.subdivisions, radius_mm=a.radius_mm)
    solver = HeatSolver(V, T)
    true_origin = int(np.argmax(V[:, 0]))
    rng = np.random.default_rng(a.seed)
    scores = synth_field(solver(true_origin), "gaussian", GAUSS_FWHM_FACTOR * a.sigma_true, a.noise, rng)
    res = analyse_map(scores, V, T, dist_solver=solver)
    d = solver(res["origin"])

    out = a.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        fh.write(f"# sigma_true_mm: {a.sigma_true}\n")
        fh.write(f"# noise_level: {a.noise}\n")
        fh.write(f"# true_origin: {true_origin}\n")
        fh.write(f"# found_origin: {res['origin']}\n")
        fh.write(f"# n_vertices: {len(V)}\n")
        fh.write(f"# python_sigma_mm: {res['sigma']:.6f}\n")
        fh.write(f"# python_r2: {res['r2']:.6f}\n")
        w = csv.writer(fh)
        w.writerow(["geodesic_distance_mm", "score"])
        for di, si in zip(d, scores, strict=True):
            w.writerow([f"{di:.6f}", "" if np.isnan(si) else f"{si:.6f}"])

    n_valid = int(np.sum(np.isfinite(d) & np.isfinite(scores)))
    print(
        f"wrote {out}: {n_valid}/{len(V)} vertices with (d, s); sigma_0 = {a.sigma_true} mm, "
        f"Python pipeline sigma = {res['sigma']:.3f} mm"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
