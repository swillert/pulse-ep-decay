# pulse-ep-decay — user guide

From a mapping-system export to a table of decay widths, and how to read that
table. Commands assume a Python ≥ 3.10 environment.

## 1. What the analysis does

A pace-mapping score map assigns every mesh vertex a similarity (0–100 %)
between the paced beat at that site and the template beat. pulse-ep-decay
measures how sharply the score falls off around its maximum, along the
surface:

| Step | What | Parameter |
|---|---|---|
| 0 | keep the largest connected mesh component | — |
| 1 | origin = vertex of maximum score; on a plateau of equal maxima, the plateau vertex nearest its centre | — |
| 2 | geodesic distance from the origin to every vertex (Heat Method) | — |
| 3 | mean score per 2-mm distance bin (bins with fewer than 3 vertices dropped) | 2 mm |
| 4 | Gaussian fit `A·exp(−d²/2σ²) + B` → **σ in mm** | σ ∈ [0.1, 200] mm |
| 5 | exponential fit for comparison → ΔAIC | — |
| 6 | focality index F = share of the half-maximum set connected to the origin | — |
| 7 | width class and flags | see §4 |

σ is a property of the score map — how far from the maximum the match
stays high — not the localisation error of the mapping system.

## 2. Install

```bash
# with the pulse-ep adapter (reads maps from a pulse-ep database):
pip install "pulse-ep-decay[pulse-ep] @ git+https://github.com/swillert/pulse-ep-decay"

# core only (NumPy arrays or .npz files):
pip install "pulse-ep-decay @ git+https://github.com/swillert/pulse-ep-decay"
```

Check: `pulse-ep-decay --version`. Try it without any data (needs the
`pulse-ep` extra): `pulse-ep-decay demo` analyses eight synthetic maps with
known widths and prints one line per map.

## 3. Run

### 3a. Maps in a pulse-ep database

pulse-ep imports CARTO 3 and EnSite X exports into one PostgreSQL database
([pulse-ep README](https://github.com/swillert/pulse-ep)):

```bash
docker compose up -d            # in the pulse-ep checkout: PostgreSQL
pulse-ep-init                   # .env, schema, colormaps, admin user
pulse-ep-import-carto  -i /path/to/carto-export      # or
pulse-ep-import-ensite -i /path/to/ensite-export.zip
pulse-ep-tag-maps --apply       # optional: tags maps (pacemap, atrium, part) from their names
```

Then, in the same environment (pulse-ep's `.env` or `PULSE_EP_DATABASE_URL`
names the database):

```bash
pulse-ep-decay run --out widths.csv              # every pace map
pulse-ep-decay run --map-ids 12 13 --out widths.csv
pulse-ep-decay run --db postgresql://user:pw@host:5432/pulse --out widths.csv
pulse-ep-decay run --all --out widths.csv        # every map that carries a score field
```

A map counts as a pace map when its attributes say `pacemap: true`
(`pulse-ep-tag-maps`) or when it carries a scalar field of kind
`pacemap_score`, which both importers register for pace-mapping exports. The
score is read from that field; maps imported by pulse-ep versions before the
scalar fields existed are read from the legacy CARTO array `act_bip`
(column `score_source` says which).

Progress goes to stderr (one line per map), the table to `--out` (or stdout).
`--format json` writes JSON instead of CSV.

### 3b. Arrays without pulse-ep

```bash
pulse-ep-decay run --npz meshes/ --out widths.csv
```

Each `.npz` holds `vertices` (N × 3, mm), `triangles` (M × 3, 0-based vertex
indices) and `scores` (N, NaN where a vertex has no score). Files without a
`scores` array are skipped with a note.

From Python:

```python
from pulse_ep_decay import analyse_map, largest_component

V, T, idx = largest_component(vertices, triangles)
res = analyse_map(scores[idx >= 0], V, T)
res["sigma"], res["width_class"], res["r2"], res["flag_r2"], res["F"], res["flag_F"], res["status"]
```

`analyse_map(..., origin=i)` fixes the origin to vertex `i` (a known pacing
site) instead of searching the maximum.

## 4. Reading `widths.csv`

| Column | Meaning |
|---|---|
| `map_id`, `map_name`, `study_name` | pulse-ep ids and names (file name for `--npz`) |
| `n_vertices` | vertices of the largest mesh component |
| `n_scored` | of those, vertices with a score |
| `origin` | index of the origin vertex (in the largest component) |
| `sigma` | **decay width σ in mm** |
| `width_class` | `below range` (< 2 mm), `in range` (2–30 mm), `broad` (30–50 mm), `diffuse` (50–100 mm), `n.i.` (≥ 100 mm or no fit) |
| `r2` | R² of the Gaussian fit to the binned profile |
| `flag_r2` | `True` when R² < 0.7: the profile is not Gaussian-shaped; σ should not be interpreted |
| `F` | focality index: share of the half-maximum set (vertices with score ≥ B + A/2) in the connected component that contains the origin |
| `n_components` | number of connected components of the half-maximum set |
| `flag_F` | `True` when F < 0.70 and R² ≥ 0.7: a second high-score region exists |
| `dAIC` | AIC(Gaussian) − AIC(exponential); positive values favour the exponential shape |
| `A`, `B` | fitted amplitude and baseline (score points) |
| `n_bins` | number of 2-mm bins in the fit |
| `score_source` | `scalar_fields:pacemap_score`, `act_bip` or `npz:scores` |
| `status` | `ok`, or why a quantity is unavailable |

How to read a row:

1. `status` ≠ `ok` → something is missing (fewer than 4 bins, fit did not
   converge, F undefined). The row is reported, not interpreted.
2. `flag_r2` → the score does not fall off like a single peak. Typical
   causes: a flat map, a map whose maximum is a single noisy vertex, a
   profile that rises again. σ is printed but means little; `dAIC` and the
   profile itself tell more.
3. `flag_F` (with R² ≥ 0.7) → the fit is fine but part of the high-score
   region is not connected to the origin: two candidate sites, or a wrapped
   surface. σ describes the region around the maximum only.
4. No flag → σ is the width of a single, well-fitted peak. Compare maps on
   this scale; the operating range validated on synthetic surfaces is
   2–30 mm (bias within ±0.25 mm; precision 0.2–1.8 mm for 3–30 mm).
5. `width_class` is a coarse label of the same number; `below range` usually
   means the maximum's own bin carries the fit (noise, a single vertex).

σ is **not** a per-map confidence interval and **not** the distance between
the score maximum and the true pacing site; the method paper evaluates that
distance separately against marked pacing sites.

## 5. Synthetic test data in pulse-ep

To exercise the whole chain on a pulse-ep installation without patient data:

```bash
pulse-ep-decay seed-synthetic --db postgresql://...   # writes a study "pulse-ep-decay synthetic"
pulse-ep-decay run --db postgresql://... --out widths.csv
```

The eight maps and their expected results (`decay_truth` in the map
attributes):

| map | expectation |
|---|---|
| `focal_6`, `focal_12` | σ ≈ 6 / 12 mm, `in range`, no flags |
| `broad_40` | σ ≈ 40 mm, `broad` |
| `two_peaks` | second source 45 mm away → `flag_F`, F ≈ 0.5, 2 components |
| `flat_noise` | no peak → `below range` or `flag_r2`, never an unflagged in-range map |
| `nan_holes` | σ ≈ 8 mm despite an unscored disc 25 mm away |
| `two_components` | σ ≈ 8 mm; the detached fragment is dropped (`n_vertices` < total) |
| `legacy_act_bip` | σ ≈ 10 mm, `score_source = act_bip` |

## 6. Public API

| Name | Role |
|---|---|
| `analyse_map(scores, vertices, faces, dist_solver=None, origin=None) -> dict` | Algorithm 1; keys as in §4 plus `aic_gauss`, `aic_exp`, `lam_exp` |
| `largest_component(vertices, triangles) -> (V, T, index_map)` | step 0; `index_map[i]` is the new index of old vertex `i`, −1 if dropped |
| `find_origin(scores, vertices, faces) -> int` | step 1 with the plateau-centre tie rule |
| `HeatSolver(V, T)(origin) -> distances`; `ExactSolver` (pygeodesic); `heat_solver()` | step 2 |
| `bin_profile(distances, scores) -> (centres, means, counts)` | step 3 |
| `fit_gaussian(centres, means)`, `fit_exponential(centres, means)` | steps 4–5; dicts with parameters, `r2`, `aic`, `converged` |
| `focality_index(scores, faces, origin, A, B) -> (F, n_components, status)` | step 6 |
| `width_class(sigma) -> str` | step 7 |
| `synth_field(distances, profile, fwhm, noise, rng)` | synthetic fields of the validation (`gaussian`, `exponential`, `lorentzian`, `raised_cosine`) |
| `R2_THRESHOLD = 0.7`, `F_THRESHOLD = 0.70`, `BIN_WIDTH_MM = 2.0`, `SIGMA_BOUNDS_MM = (0.1, 200.0)`, `GAUSS_FWHM_FACTOR` | parameters |
| `integration.pulse_ep`: `from_epmap`, `analyse_epmap`, `iter_pacemaps`, `is_pacemap`, `score_field` | pulse-ep adapter |
| `integration.synthetic`: `make_synthetic_pacemaps`, `seed_database`, `SCENARIOS` | synthetic pulse-ep maps |
| `legacy` | superseded estimators used only by the supplement's sensitivity analyses |

## 7. Reproducing the paper

`reproduction/` rebuilds every data-derived table and figure of the method
paper from archived result files; see `reproduction/README.md`. Clinical
meshes and scores are not included.
