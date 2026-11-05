# pulse-ep-decay

Geodesic decay analysis of pace-mapping score maps, a module for
[pulse-ep](https://github.com/swillert/pulse-ep).

The package measures how fast a pace-mapping score falls off with geodesic
distance from its maximum on a triangulated atrial surface:

1. **Origin**: the vertex with the highest score; on a plateau of equal maxima,
   the plateau vertex closest to its centre.
2. **Geodesic distance** from the origin to every vertex with the Heat Method
   (potpourri3d / geometry-central). Exact geodesics (pygeodesic, MMP) are
   available for validation.
3. **Radial profile**: mean score in 2-mm distance bins.
4. **Gaussian fit** `A · exp(−d² / 2σ²) + B`. The width σ (mm) is the result.
5. **Flags**: `R² < 0.7` marks an inadequate fit; the focality index
   `F < 0.70` (share of the half-maximum set connected to the origin) marks a
   second high-score region and is interpreted only when `R² ≥ 0.7`.
   Width classes: below range (< 2 mm), in range (2–30 mm), broad (30–50 mm),
   diffuse (50–100 mm), not identifiable (≥ 100 mm).

This is the code of the method paper *Geodesic Decay Analysis: Quantifying the
Spatial Resolution of Intracardiac Pace-Mapping Score Maps on Triangulated
Atrial Surfaces* (Willert, Maslova, Zaman, Frank, Lian; Computer Methods and
Programs in Biomedicine, under revision).

**Guide:** [docs/GUIDE.md](docs/GUIDE.md) — from export to `widths.csv`, how to read the columns and flags, API reference.

## Installation

```bash
pip install "pulse-ep-decay @ git+https://github.com/swillert/pulse-ep-decay"
# extras: [pulse-ep] database adapter and synthetic maps, [exact] pygeodesic,
#         [repro] reproduction scripts, [all] everything
```

## Use with pulse-ep

[pulse-ep](https://github.com/swillert/pulse-ep) imports CARTO 3 and EnSite X
studies into one database. With the `pulse-ep` extra installed, every pace map
in that database can be analysed in one command (the database is found through
pulse-ep's own configuration, `PULSE_EP_DATABASE_URL` or `.env`; `--db`
overrides it):

```bash
pip install "pulse-ep-decay[pulse-ep] @ git+https://github.com/swillert/pulse-ep-decay"
pulse-ep-decay run --out widths.csv                   # all pace maps
pulse-ep-decay run --map-ids 12 13 --out widths.csv   # selected maps
```

`widths.csv` has one row per map: width σ and class, R² and its flag, F with
the number of half-maximum components and its flag, ΔAIC (Gaussian minus
exponential), fitted amplitude and baseline, number of bins, where the score
came from, and a status. A map counts as a pace map when it is tagged
`pacemap` (`pulse-ep-tag-maps`) or carries a scalar field of kind
`pacemap_score`; `--all` analyses every map with a score field.

The same from Python:

```python
from pulse_ep.core.database import get_db_session
from pulse_ep_decay.integration.pulse_ep import analyse_epmap, iter_pacemaps

with get_db_session() as s:
    for epmap, attrs, arrays in iter_pacemaps(session=s):
        res = analyse_epmap(epmap)
        print(epmap.id, res["sigma"], res["width_class"], res["flag_r2"], res["flag_F"])
```

The adapter reads the score from the vendor-neutral scalar field
`pacemap_score` that both pulse-ep importers register, and falls back to the
legacy CARTO array `act_bip` for maps imported by older pulse-ep versions.

**Try it without data.** `pulse-ep-decay demo` builds eight synthetic pace
maps on pulse-ep's ellipsoidal pseudo-atrium (known widths, two peaks, pure
noise, unscored holes, a detached mesh fragment, a legacy-format map) and
analyses them; `pulse-ep-decay seed-synthetic --db …` writes the same maps
into a pulse-ep database as a test study.

## Use without pulse-ep

The core takes plain NumPy arrays:

```python
from pulse_ep_decay import analyse_map, largest_component

V, T, idx = largest_component(vertices, triangles)   # mm, (n, 3) and (m, 3)
res = analyse_map(scores[idx >= 0], V, T)            # NaN where no score
res["sigma"], res["width_class"], res["r2"], res["flag_r2"], res["F"], res["flag_F"]
```

`analyse_map` also returns the fitted amplitude and baseline, ΔAIC against an
exponential fit, and a `status` string that names any unavailable quantity.
On the command line, `pulse-ep-decay run --npz DIR` analyses `.npz` files with
arrays `vertices`, `triangles` and `scores`.

## Modules

| Module | Content |
|---|---|
| `mesh` | largest component, Heat-Method and exact geodesic solvers |
| `profile` | origin rule, 2-mm binning, Gaussian, exponential and generalised fits, synthetic fields |
| `analysis` | `analyse_map`, `width_class`, `focality_index` |
| `support` | length-weighted mesh edge graph |
| `legacy` | superseded estimators, only for the supplement's sensitivity analyses |
| `cli` | `pulse-ep-decay run | demo | seed-synthetic` |
| `integration.pulse_ep` | pulse-ep adapter: `from_epmap`, `analyse_epmap`, `iter_pacemaps` |
| `integration.synthetic` | synthetic pace maps as pulse-ep objects (`make_synthetic_pacemaps`, `seed_database`) |

## Reproduction

`reproduction/` contains the scripts behind every data-derived table and
figure of the paper and its supplement, and the archived result files needed
to regenerate all data-derived tables, Figs. 2 and 5, and Supplementary
Figs. S1–S6. See
[reproduction/README.md](reproduction/README.md). The clinical meshes and
score maps are not public (data protection); figures of clinical maps
(Figs. 1, 3, 4 and S7) cannot be regenerated from this repository.

## Examples

`examples/sigma_recovery.py` recovers an injected width on a synthetic
icosphere in Python; `examples/r/` exports the geodesic profile and re-fits
the Gaussian with base-R `nls()` — an independent reproduction of σ in another
language. See [examples/README.md](examples/README.md).

## Tests

```bash
pip install -e ".[all]"
pytest            # synthetic surfaces only; no clinical data needed
```

The database round trip (`seed-synthetic` → `run`) runs when
`PULSE_EP_TEST_DATABASE_URL` names a PostgreSQL pulse-ep database.

## Related

- pulse-ep: Willert S, Frank D, Lian E. pulse-ep: A Multivendor Database and
  Analysis Platform for Electroanatomical Mapping Research and AI-Assisted
  Workflows. SSRN preprint, 2026. https://doi.org/10.2139/ssrn.7499672
- pulse-ep-timeref: activation-timing reference analysis (separate package).

## Licence

MIT. See [LICENSE](LICENSE). Citation metadata: [CITATION.cff](CITATION.cff).
Changes: [CHANGELOG.md](CHANGELOG.md).
