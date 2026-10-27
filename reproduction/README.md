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
  supplement source (the generators read the table and figure list from it),
  the hash inventory of the 40 distinct mesh reconstructions and the
  re-aggregation of the earlier validation;
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
second draws every figure that can be built from result files alone: Fig. 4
(validation), Fig. 7 (marker check) and Supplementary Figs. S1–S6.

Check (27 Oct 2026, Python 3.10, versions of `requirements.txt`): all LaTeX
tables and number files were byte-identical to the submitted manuscript, and
the eight figure previews pixel-identical. `MANIFEST.md` differs only in the
script column of two entries whose scripts are in
[geodesic-decay](https://github.com/swillert/geodesic-decay) (`arm3`,
anisotropy).

## Not reproducible from this folder

- Figs. 1, 3, 5 and 6 show clinical maps and need the clinical meshes and
  scores, which are not public (data protection).
- Re-running the validation scripts v36–v62 needs the 56 clinical meshes
  (`pulse-ep/data/validation_meshes`) and `pygeodesic`; v49 additionally
  needs `geodesic-decay/scripts/anisotropy.py`.
