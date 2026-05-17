"""pulse-ep-decay — geodesic spatial-resolution analysis for cardiac mesh data.

Public API:

    from pulse_ep_decay import (
        geodesic_distances_dijkstra,   # baseline edge-graph distances
        geodesic_distances_heat,       # exact distances via Heat Method
        fit_gaussian_decay,            # σ-fit on a radial score profile
        DecayFitResult,                # dataclass with σ̂, R², residuals
        generate_synthetic_field,      # for validation
        recover_sigma,                 # end-to-end σ-recovery on a mesh
    )

The optional adapter to the pulse-ep platform lives at
``pulse_ep_decay.integration.pulse_ep`` and is only importable when
``pulse-ep`` is installed (``pip install pulse-ep-decay[pulse-ep]``).

See the README and ``pulse_ep_decay/validation/`` for the σ-recovery
experiments that back the accompanying method paper.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__: list[str] = [
    # Filled in once the modules are implemented.
]
