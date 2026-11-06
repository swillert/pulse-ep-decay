# v36 rerun of 6 Nov 2026 (b) — source vertex excluded by index, atan2 great-circle distance

Finding (external check, 6 Nov 2026): `analytic_sphere()` returned a self-distance of ~3.7e-7 mm instead of 0 for some
sources (u·u rounds to 1 − ε, arccos → ~1.5e-8 rad × 25 mm), the mask `truth > 1e-9` in `errs()` then kept the source,
and the solvers' exact 0 entered the signed relative error as −100 % (one vertex of N). Affected: 5 of the 15 sphere
sources (sub3: 627, 591; sub4: 422, 1321; sub5: 1805); cylinders unaffected (self-distance exactly 0).

Fix in `scripts/v36_analytic_geodesic.py`:
- the great-circle distance is `R·atan2(|u × u_src|, u · u_src)`, well conditioned at θ = 0 and θ = π (arccos of the
  dot product loses ~8 digits at both ends; the icosphere has exact antipodal pairs), and `d[src] = 0` by definition;
- `errs(d, truth, src)` masks the source by index, not by a threshold on the truth.

Rerun into this folder (Linux VM, Python 3.10, numpy 2.2.6 / scipy 1.15.3 / potpourri3d 1.4.0 / pygeodesic, 6 Nov 2026).
`v36_source_check.py` reproduces the archived (21 Oct) values with the old mask and shows the corrected ones.

Effect (Table S1, means over 5 sources): sphere sub3 dijkstra +6.33 → +6.40 %, exact −0.28 → −0.22 %, heat −1.73 →
−1.67 % (p95 12.73 → 12.70, 3.41 → 3.40); sub4 dijkstra +6.52 → +6.54, exact −0.07 → −0.06, heat −1.05 → −1.04; sub5
exact −0.02 → −0.01; all other cells unchanged at 2 decimals. Against the first rerun of 6 Nov (arccos with `d[src] = 0`,
archived in release v1.1.2 of the deposit) the atan2 form changes the mean signed relative error by ≤ 8e-12 and the
summary not at all; the width study (part B) differs by ≤ 6e-4 mm in individual fits (curve-fit tolerance), Table S1b
is generated from v36c in any case. No sentence of the manuscript, supplement or response letter quotes an affected
value.

The folder of 21 Oct (`../v36_analytic_geodesic_2026-10-21/`) is kept for the record.
