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

## Installation

```bash
pip install "pulse-ep-decay @ git+https://github.com/swillert/pulse-ep-decay"
# extras: [exact] pygeodesic, [pulse-ep] database adapter, [repro] reproduction scripts
```

## Usage

The core takes plain NumPy arrays and does not need pulse-ep:

```python
from pulse_ep_decay import analyse_map, largest_component

V, T, idx = largest_component(vertices, triangles)   # mm, (n, 3) and (m, 3)
res = analyse_map(scores[idx >= 0], V, T)            # NaN where no score
res["sigma"], res["width_class"], res["r2"], res["flag_r2"], res["F"], res["flag_F"]
```

`analyse_map` also returns the fitted amplitude and baseline, ΔAIC against an
exponential fit, and a `status` string that names any unavailable quantity.

With pulse-ep installed, `pulse_ep_decay.integration.pulse_ep` reads meshes
and scores from a pulse-ep database (`from_epmap`, `iter_pacemaps`).

## Modules

| Module | Content |
|---|---|
| `mesh` | largest component, Heat-Method and exact geodesic solvers |
| `profile` | origin rule, 2-mm binning, Gaussian, exponential and generalised fits, synthetic fields |
| `analysis` | `analyse_map`, `width_class`, `focality_index` |
| `support` | edge graph, connected components, smoothed maximum |
| `integration.pulse_ep` | optional pulse-ep adapter |

## Reproduction

`reproduction/` contains the scripts and archived result files behind every
data-derived table and figure of the paper and its supplement. See
[reproduction/README.md](reproduction/README.md). The clinical meshes and
score maps are not public (data protection); figures of clinical maps cannot
be regenerated from this repository.

## Related

- pulse-ep: Willert S, Frank D, Lian E. pulse-ep: A Multivendor Database and
  Analysis Platform for Electroanatomical Mapping Research and AI-Assisted
  Workflows. SSRN preprint, 2026. https://doi.org/10.2139/ssrn.7499672
- pulse-ep-timeref: activation-timing reference analysis (separate package).

## Licence

MIT. See [LICENSE](LICENSE). Citation metadata: [CITATION.cff](CITATION.cff).
