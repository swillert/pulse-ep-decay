"""End-to-end sigma recovery on a synthetic icosphere (Python).

Inject a Gaussian score field of known width sigma_0 (manuscript protocol:
A = 35, B = 65, noise, clipping to [50, 100], 5 % unscored vertices), run the
published pipeline (origin -> Heat-Method geodesics -> 2-mm bins -> Gaussian
fit) and report the recovered width, class, R^2 and focality index.

    python examples/sigma_recovery.py --sigma-true 8 --noise 0.05
    python examples/sigma_recovery.py --sigma-true 8 --plot sigma_recovery.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mesh_utils import icosphere  # noqa: E402

from pulse_ep_decay import GAUSS_FWHM_FACTOR, HeatSolver, analyse_map, bin_profile, gauss_model, synth_field  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sigma-true", type=float, default=8.0, help="injected width sigma_0 in mm")
    p.add_argument("--noise", type=float, default=0.05, help="noise SD as a fraction of the amplitude A")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subdivisions", type=int, default=3, help="icosphere subdivisions (3 -> 642 vertices)")
    p.add_argument("--radius-mm", type=float, default=25.0)
    p.add_argument(
        "--plot", type=Path, help="write the binned profile and the fit to this PNG (needs matplotlib)"
    )
    a = p.parse_args(argv)

    V, T = icosphere(subdivisions=a.subdivisions, radius_mm=a.radius_mm)
    solver = HeatSolver(V, T)
    true_origin = int(np.argmax(V[:, 0]))
    rng = np.random.default_rng(a.seed)
    scores = synth_field(solver(true_origin), "gaussian", GAUSS_FWHM_FACTOR * a.sigma_true, a.noise, rng)

    res = analyse_map(scores, V, T, dist_solver=solver)
    print(f"mesh            {len(V)} vertices, {len(T)} triangles, radius {a.radius_mm:g} mm")
    print(
        f"injected        sigma_0 = {a.sigma_true:.2f} mm, noise {a.noise:.2f}, origin vertex {true_origin}"
    )
    print(f"recovered       sigma   = {res['sigma']:.2f} mm  (bias {res['sigma'] - a.sigma_true:+.2f} mm)")
    print(
        f"                origin vertex {res['origin']}, class {res['width_class']}, R2 {res['r2']:.3f}, F {res['F']:.2f}"
    )
    print(
        f"                flags: r2={res['flag_r2']} F={res['flag_F']}; dAIC {res['dAIC']:+.1f}; status {res['status']}"
    )

    if a.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        d = solver(res["origin"])
        c, m, _ = bin_profile(d, scores)
        grid = np.linspace(0, c.max(), 200)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(d, scores, s=3, color="0.8", label="vertices")
        ax.plot(c, m, "o", color="#1f77b4", label="2-mm bin means")
        ax.plot(
            grid,
            gauss_model(grid, res["A"], res["sigma"], res["B"]),
            color="#e34a33",
            lw=2,
            label=f"fit: sigma = {res['sigma']:.2f} mm",
        )
        ax.axvline(a.sigma_true, color="grey", ls="--", label=f"sigma_0 = {a.sigma_true:g} mm")
        ax.set_xlabel("geodesic distance from origin (mm)")
        ax.set_ylabel("score")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=150)
        print(f"wrote {a.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
