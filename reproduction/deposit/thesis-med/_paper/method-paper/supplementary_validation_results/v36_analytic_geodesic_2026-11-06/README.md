# v36 rerun of 6 Nov 2026 — source vertex excluded from the relative error

Finding (external check, 6 Nov 2026): `analytic_sphere()` returns a self-distance of ~3.7e-7 mm instead of 0 for some
sources (u·u rounds to 1 − ε, arccos → ~1.5e-8 rad × 25 mm), the mask `truth > 1e-9` in `errs()` then keeps the source,
and the solvers' exact 0 enters the signed relative error as −100 % (one vertex of N). Affected: 5 of the 15 sphere
sources (sub3: 627, 591; sub4: 422, 1321; sub5: 1805); cylinders unaffected (self-distance exactly 0).

Fix in `scripts/v36_analytic_geodesic.py`: `d[src] = 0.0` in `analytic_sphere()`, and `errs(d, truth, src)` excludes
the source index. Rerun into this folder (Linux VM, Python 3.10, numpy/scipy/potpourri3d/pygeodesic from pip, 6 Nov 2026).

Effect (`v36_source_check.out`, archived vs corrected per affected row; Table S1 means over 5 sources):
- sphere sub3: dijkstra +6.33 → +6.40 %, exact −0.28 → −0.22 %, heat −1.73 → −1.67 %; p95 12.73 → 12.70, 3.41 → 3.40
- sphere sub4: dijkstra +6.52 → +6.54, exact −0.07 → −0.06, heat −1.05 → −1.04
- sphere sub5: exact −0.02 → −0.01; all other cells unchanged at 2 decimals
- width study (`v36_width_by_solver.csv`): differences ≤ 3e-8 mm in sigma_hat (floating point), Table S1b is from v36c anyway
- No number in the manuscript, supplement text or response letter quotes an affected value ("6–18 %" for Dijkstra holds).

Archived results used by the R1 upload set of 6 Nov: `../v36_analytic_geodesic_2026-10-21/` (unchanged).
