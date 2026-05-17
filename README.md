# pulse-ep-decay

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Geodesic spatial-resolution analysis for intracardiac pace-mapping — the
scientific extension to [pulse-ep](https://gitlab.willert.net/sw/pulse-ep).

`pulse-ep-decay` quantifies the spatial resolution of a scalar field on a
triangulated surface mesh. Given a per-vertex score (e.g. pace-mapping
similarity), it

1. computes geodesic distances from the score's maximum (the **origin**) to
   every other vertex using the **Heat Method**
   ([Crane et al. 2017](https://doi.org/10.1145/3131280)) — exact on the
   triangle manifold, much cheaper than repeated Dijkstra calls,
2. fits a **Gaussian decay model** `f(d) = a · exp(−d²/(2σ²)) + c` to the
   radial profile, and
3. reports σ̂ as a **physical-unit (mm) resolution measure**.

A built-in **synthetic validation framework** generates score fields with
known σ₀ on real atrial meshes and recovers σ̂ to characterise bias and
variance — this is what backs the accompanying method paper.

## Design principle: array-first

The scientific kernel takes **plain NumPy arrays**:

```python
import numpy as np
from pulse_ep_decay import recover_sigma

vertices  : np.ndarray  # (N, 3)  triangle-mesh vertices
triangles : np.ndarray  # (M, 3)  vertex indices
scores    : np.ndarray  # (N,)    per-vertex scalar field

result = recover_sigma(vertices, triangles, scores)
print(result.sigma_mm, result.r_squared)
```

That means you can apply `pulse-ep-decay` to **any** triangulated mesh and
scalar field — no CARTO, no Postgres, no `pulse-ep` required.

If you already use `pulse-ep` to ingest CARTO studies, the optional adapter
pulls the same arrays out of an `EPMapModel`:

```python
# pip install "pulse-ep-decay[pulse-ep]"
from pulse_ep_decay.integration.pulse_ep import from_epmap
from pulse_ep_decay import recover_sigma

vertices, triangles, scores = from_epmap(epmap_model)
result = recover_sigma(vertices, triangles, scores)
```

## Installation

```bash
# scientific core only
pip install pulse-ep-decay

# with pulse-ep adapter for CARTO workflows
pip install "pulse-ep-decay[pulse-ep]"

# all extras (adapter, figures, dev)
pip install "pulse-ep-decay[all]"
```

## Quickstart

```bash
# synthetic σ-recovery on a Gaussian field over a built-in sphere mesh
pulse-ep-decay-validate-sigma --sigma 6.0 --noise 0.05

# Heat Method vs. Dijkstra comparison on your own meshes
pulse-ep-decay-compare-geodesic --mesh path/to/atrium.npz
```

See `docs/` for the full walkthrough including the validation experiments
from the method paper.

## Citation

If you use `pulse-ep-decay` in academic work, please cite the
[CITATION.cff](CITATION.cff) entry. Until the method paper is accepted,
the canonical citation is the software itself; once accepted, please
prefer the journal article.

## License

MIT — see [LICENSE](LICENSE).

## Related

- [pulse-ep](https://gitlab.willert.net/sw/pulse-ep) — the open-source
  CARTO platform that `pulse-ep-decay` integrates with.
- [pulse-ultimate](https://gitlab.willert.net/sw/pulse-ultimate) —
  the predecessor monorepo (frozen archive).
