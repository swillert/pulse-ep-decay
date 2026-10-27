"""pulse-ep-decay: geodesic decay analysis of pace-mapping score maps.

A module for `pulse-ep <https://github.com/swillert/pulse-ep>`_ and the code of
the method paper *Geodesic Decay Analysis: Quantifying the Spatial Resolution of
Intracardiac Pace-Mapping Score Maps on Triangulated Atrial Surfaces*
(Willert et al., Computer Methods and Programs in Biomedicine, under revision).

- :mod:`pulse_ep_decay.mesh` — largest component, Heat-Method (potpourri3d) and
  exact (pygeodesic, optional) geodesic solvers;
- :mod:`pulse_ep_decay.profile` — origin (maximum with plateau-centre tie rule),
  radial binning (2 mm), Gaussian and alternative fits, synthetic fields;
- :mod:`pulse_ep_decay.analysis` — :func:`analyse_map` (Algorithm 1 of the
  paper): width, class, R² and focality flags, ΔAIC, availability status;
- :mod:`pulse_ep_decay.support` — mesh-graph helpers (edge graph, components,
  smoothed maximum);
- :mod:`pulse_ep_decay.integration.pulse_ep` — optional adapter that reads
  meshes and score fields from a pulse-ep database.

Usage::

    from pulse_ep_decay import analyse_map, largest_component
    V, T, idx = largest_component(vertices, triangles)
    res = analyse_map(scores[idx >= 0], V, T)   # sigma, width_class, r2, flag_r2, F, flag_F, dAIC, status
"""

__version__ = "1.0.0"

from pulse_ep_decay.analysis import analyse_map, focality_index, width_class
from pulse_ep_decay.mesh import (
    CoreHeatSolver,
    ExactSolver,
    HeatSolver,
    heat_solver,
    largest_component,
    mean_edge_length,
    nearest_vertex,
)
from pulse_ep_decay.profile import (
    BIN_WIDTH_MM,
    SIGMA_BOUNDS_MM,
    bin_profile,
    fit_exponential,
    fit_gaussian,
    fit_gengauss,
    gauss_model,
    run_pipeline,
    synth_field,
)
from pulse_ep_decay.support import (
    ORIGIN_SMOOTH_RADIUS_MM,
    SUPPORT_RADIUS_MM,
    component_areas,
    masked_scores,
    smoothed_argmax,
    support_mask,
    top_region,
)

__all__ = [
    "analyse_map",
    "focality_index",
    "width_class",
    "BIN_WIDTH_MM",
    "CoreHeatSolver",
    "ExactSolver",
    "HeatSolver",
    "ORIGIN_SMOOTH_RADIUS_MM",
    "SIGMA_BOUNDS_MM",
    "SUPPORT_RADIUS_MM",
    "bin_profile",
    "component_areas",
    "fit_exponential",
    "fit_gaussian",
    "fit_gengauss",
    "gauss_model",
    "heat_solver",
    "largest_component",
    "masked_scores",
    "mean_edge_length",
    "nearest_vertex",
    "run_pipeline",
    "smoothed_argmax",
    "support_mask",
    "synth_field",
    "top_region",
]
