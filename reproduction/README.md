# Reproduction of the tables and figures

This folder rebuilds the data-derived tables and figures of the method paper
from archived result files. It contains no clinical meshes or score maps.

`deposit/` keeps the directory layout of the original working tree, because
the scripts address files relative to a root directory (`GD_ROOT`):

- `thesis-med/_paper/method-paper/scripts/`: table and figure generators and
  the validation and analysis scripts v36–v62;
- `thesis-med/_paper/method-paper/supplementary_validation_results/`: their
  result files (summaries and per-map results; full raw outputs omitted);
- `thesis-med/_paper/method-paper/revision_2026-10/`: the manuscript and
  supplement source (the generators read the table and figure list from it);
- `pulse-ultimate/decay_profiles_heat/decay_summary.csv`: per-map summary of
  the clinical import.

Map and study numbers are pseudonymous.

## Run

From the repository root:

```bash
pip install -e ".[repro]"          # or: pip install -r reproduction/requirements.txt
cd reproduction/deposit
export GD_ROOT=$PWD MPLBACKEND=Agg SOURCE_DATE_EPOCH=1793059200
M=thesis-med/_paper/method-paper
python $M/scripts/make_revision_tables.py --out $M/revision_2026-10/tables
python $M/scripts/make_revision_figures.py data
```

The first command writes every generated table, `numbers.md/json` and
`MANIFEST.md/json` (source file, SHA-256 and script for each table). The
second draws every figure that can be built from result files alone: Fig. 2
(validation), Fig. 5 (marker check) and Supplementary Figs. S1–S6.

Check (6 Nov 2026, manuscript state of the R1 upload set; Linux arm64, Python
3.10, versions of `requirements.txt`): all 38 LaTeX tables and number files
were byte-identical to the manuscript build, and all 16 figure files (8 PDF,
8 PNG) byte-identical. `MANIFEST.md` differs only where it must: the clinical
per-map caches of Figs. 1, 3, 4 and S7 are marked `MISSING`, and the script
column of two entries whose scripts are in
[geodesic-decay](https://github.com/swillert/geodesic-decay) (`arm3`,
anisotropy) reads "(see folder README)".

Correction of 6 Nov 2026 (versions 1.1.2 and 1.1.3): in `v36_analytic_geodesic.py`
the analytic self-distance of the source vertex could round to ~4e-7 mm
instead of 0 and then entered the signed relative error as −100 % for 5 of the
15 sphere sources. Since 1.1.3 the great-circle distance is computed as
`R·atan2(|u × u_src|, u · u_src)` (well conditioned at θ = 0 and θ = π, where
`arccos` of the dot product loses digits), the source distance is zero by
definition and the source is masked by index. The rerun is in
`supplementary_validation_results/v36_analytic_geodesic_2026-11-06b/` (README
there), which Table S1 and Fig. S1 use. Seven cells of Table S1 moved by
0.01–0.07 percentage points against the 21 Oct results; no other table, figure
or sentence is affected. The folder of 21 Oct is kept for the record; the
intermediate rerun of 1.1.2 (`arccos` with the source set to zero, identical at
reporting precision) is archived in that release.

## Your own maps

`make_revision_figures.py` draws the per-map panels of the clinical figures
(score surface with the origin, geodesic distance, radial profile with the
Gaussian fit) from a mesh file in the same `.npz` layout that `pulse-ep-decay
run --npz` reads (`vertices`, `triangles`, `scores`). To draw them for a map of
your own, save it as `pulse-ep/data/validation_meshes/map_<id>.npz` under
`GD_ROOT` and run

```bash
python $M/scripts/make_revision_figures.py test <id>
```

which writes `revision_2026-10/figs/_test_map<id>.pdf/.png`. The marked pacing
site of the clinical maps is read from a selection file that is not part of
the deposit; without it no marker is drawn. The panels are laid out for atrial
maps (42-mm window, contours at 1 and 2 mm); adjust `WINDOW` and the contour
levels in the script for other geometries.

## Not reproducible from this folder

- Figs. 1, 3, 4 and Supplementary Fig. S7 show clinical maps and need the
  clinical meshes and scores, which are not public (data protection).
- Re-running the validation scripts v36–v62 needs the 56 clinical meshes
  (`pulse-ep/data/validation_meshes`) and `pygeodesic`; v49 additionally
  needs `geodesic-decay/scripts/anisotropy.py`.

## Keeping the deposit in step with the working tree

The scripts here are copies of `thesis-med/_paper/method-paper/scripts/` with
the package import renamed (`geodesic_decay`, the development name, →
`pulse_ep_decay`) and developer paths made relative. `sync_from_worktree.py`
refreshes every file the deposit already holds and reports the changes;
`--check` only reports:

```bash
python reproduction/sync_from_worktree.py --check
```

New scripts or result folders are added by hand (never clinical data: no
`.npz`, no `figs/_cache/map_*.npz`; see `.gitignore`).
