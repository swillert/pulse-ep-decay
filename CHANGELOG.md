# Changelog

## 1.1.3 — 2026-11-06

Analytic sphere distance in the benchmark computed in a well-conditioned
form; no change to the package.

- `v36_analytic_geodesic.py`: great-circle distance as
  `R·atan2(|u × u_src|, u · u_src)` instead of `R·arccos(u · u_src)`, which
  loses ~8 digits where the dot product rounds to ±1 (the source itself and
  exact antipodes of the icosphere); the source is masked by index in the
  error statistics. Rerun archived as
  `supplementary_validation_results/v36_analytic_geodesic_2026-11-06b/`
  (replaces the 1.1.2 folder `…_2026-11-06`, which is kept in that release);
  Table S1 is unchanged against 1.1.2 (differences ≤ 8e-12 in the means), the
  manifest hashes change.

## 1.1.2 — 2026-11-06

Zenodo: <https://doi.org/10.5281/zenodo.24022767>.

Correction of the analytic sphere benchmark in the reproduction deposit; no
change to the package.

- `v36_analytic_geodesic.py`: the analytic self-distance of the source vertex
  could round to ~4e-7 mm instead of 0 (`arccos` of a dot product that rounds
  to 1 − ε) and then passed the `truth > 1e-9` mask, so the solvers' exact 0
  entered the signed relative error as −100 % for 5 of the 15 sphere sources.
  The source distance is now set to zero and the source is excluded from the
  relative error. Rerun archived as
  `supplementary_validation_results/v36_analytic_geodesic_2026-11-06/`; Table
  S1 and Fig. S1 are generated from it (seven cells of Table S1 move by
  0.01–0.07 percentage points; nothing else changes). The folder of 21 Oct is
  kept for the record.
- `make_revision_figures.py` no longer requires the pacing-site selection file
  when it builds a per-map cache from a mesh file (no marker is drawn without
  it); `reproduction/README.md` describes how to draw the per-map panels for a
  map of your own.
- Clean-room check repeated (6 Nov 2026): 38/38 tables and number files and
  16/16 figure files byte-identical to the manuscript build.

## 1.1.1 — 2026-11-06

Zenodo: <https://doi.org/10.5281/zenodo.23991054>.

Reproduction deposit cleaned for publication; no change to the package.

- `reproduction/deposit/.../paper_rev1.tex` and `supplement_rev1.tex` are now
  synced without LaTeX comments (working notes of the authors stayed in the
  v1.1.0 archive by mistake); `sync_from_worktree.py` strips them.
- Removed `revision_2026-10/validation_reaggregation_2026-10-22/`, a working
  note on the superseded pre-correction validation that no deposit script
  reads.
- Manuscript source synced with the corrections of 5/6 Nov 2026 (sourced
  double-chamber widths, SD under correlated deviations also in Supplementary
  Section S9, S6 range −10 to +1 %, S21 text, PDF metadata); caption of
  Supplementary Table S2 with typographic minus.

## 1.1.0 — 2026-11-05

Zenodo: <https://doi.org/10.5281/zenodo.23990505>. GitHub: <https://github.com/swillert/pulse-ep-decay>.

Code of the revised manuscript (CMPB-D-26-02177 R1).

- pulse-ep adapter reads the score from the vendor-neutral scalar field
  `pacemap_score` (CARTO 3 and EnSite X imports) and falls back to the legacy
  CARTO array `act_bip`; `from_epmap` accepts `EPMapModel` rows and `EPMap`
  objects; new `analyse_epmap`, `is_pacemap`, `score_field`; `iter_pacemaps`
  takes `map_ids`, `pacemaps_only` and a session, no longer requires the
  `atrium`/`part` tags by default.
- Command line `pulse-ep-decay`: `run --db … --out widths.csv` (every pace map
  of a pulse-ep database, columns of Supplementary Table S12), `run --npz DIR`,
  `demo`, `seed-synthetic`.
- `integration.synthetic`: eight synthetic pace maps as pulse-ep objects with
  known truth (focal, broad, two peaks, noise, unscored holes, detached mesh
  fragment, legacy format); `seed_database` writes them into a pulse-ep study.
- Tests: algorithm level (plateau tie rule, focality fallback and status
  strings, ΔAIC sign, class edges, bin-count status, fixed origin and custom
  solver, exact solver), adapter and synthetic scenarios, command line; the
  database round trip runs with `PULSE_EP_TEST_DATABASE_URL`.
- Clean-up: superseded estimators moved to `legacy` (first-index tie rule,
  smoothed maximum, generalised-Gaussian fit; `profile` and `support`
  re-export them for the study scripts); unused development code removed
  (origin refinement, bootstrap, support masks, component areas); the public
  API (`__all__`) reduced to Algorithm 1, its steps and parameters.
  `analysis`: the unweighted adjacency used for connected components is now
  private (`_adjacency`); the length-weighted graph remains
  `support.edge_graph`.
- `examples/`: Python σ-recovery script and the R re-fit (`export_radial_profile.py`,
  `fit_decay.R`), ported from the May 2026 demos to the 1.x API; covered by tests.
- `docs/GUIDE.md`: user guide (export → pulse-ep → `widths.csv`, column and
  flag semantics, API reference).
- Study scripts: `ROOT` honours `GD_ROOT` (deposit and working tree).
- Reproduction folder synced to the manuscript of 29 Oct 2026 (figure
  renumbering: five main-text figures, decay profiles as Fig. S7; new number
  ρ(n_points, σ)).

## 1.0.0 — 2026-10-27

Package of the method paper as resubmitted: array-first core (`mesh`,
`profile`, `analysis`, `support`), optional pulse-ep adapter, reproduction
folder with the scripts v36–v62 and the archived result files that rebuild
every data-derived table and figure.

## 0.1.0-may — 2026-05

Pre-submission state (tag `v0.1.0-may`): the pipeline of the April 2026
submission without the plateau-centre origin rule, the half-maximum focality
index and the width classes. Kept for reference only.
