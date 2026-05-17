"""pulse-ep-decay — geodesic spatial-resolution analysis for cardiac meshes.

Companion to the CMPB method paper. The scientific kernel takes plain
NumPy arrays and has no dependency on the pulse-ep platform; the
:mod:`pulse_ep_decay.integration.pulse_ep` adapter is an optional bridge
to the CARTO database (install with ``pip install pulse-ep-decay[pulse-ep]``).

Public API::

    from pulse_ep_decay import (
        # geodesic distances on a triangle mesh
        build_adjacency_matrix,
        dijkstra_distances,
        heat_distances,
        MeshHeatSolver,

        # Gaussian decay model
        DecayFitResult,
        fit_gaussian_decay,
        gaussian_decay,

        # synthetic field generation
        generate_synthetic_field,

        # focality / multifocality check
        FocalityResult,
        compute_focality,

        # end-to-end σ-extraction and validation
        recover_sigma,
        run_sigma_recovery,
        RecoveryResult,
    )
"""

from __future__ import annotations

__version__ = "0.1.0"

from pulse_ep_decay.decay_fit import (
    DecayFitResult,
    fit_gaussian_decay,
    gaussian_decay,
)
from pulse_ep_decay.focality import FocalityResult, compute_focality
from pulse_ep_decay.geodesic import build_adjacency_matrix, dijkstra_distances
from pulse_ep_decay.heatmethod import MeshHeatSolver, heat_distances
from pulse_ep_decay.synthetic import (
    DEFAULT_AMPLITUDE,
    DEFAULT_BASELINE,
    generate_synthetic_field,
)
from pulse_ep_decay.validation import (
    GeodesicMethod,
    RecoveryResult,
    recover_sigma,
    run_sigma_recovery,
)

__all__ = [
    "__version__",
    # geodesic
    "build_adjacency_matrix",
    "dijkstra_distances",
    "heat_distances",
    "MeshHeatSolver",
    # decay fit
    "DecayFitResult",
    "fit_gaussian_decay",
    "gaussian_decay",
    # synthetic
    "generate_synthetic_field",
    "DEFAULT_AMPLITUDE",
    "DEFAULT_BASELINE",
    # focality
    "FocalityResult",
    "compute_focality",
    # validation
    "GeodesicMethod",
    "RecoveryResult",
    "recover_sigma",
    "run_sigma_recovery",
]
