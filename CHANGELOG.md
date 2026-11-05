# Changelog

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
