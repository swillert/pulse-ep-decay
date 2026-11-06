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
- :mod:`pulse_ep_decay.support` — length-weighted mesh edge graph;
- :mod:`pulse_ep_decay.legacy` — superseded estimators kept only for the
  sensitivity analyses of the supplement (first-index tie rule, smoothed
  maximum, generalised-Gaussian fit);
- :mod:`pulse_ep_decay.integration.pulse_ep` — optional adapter that reads
  meshes and score fields from a pulse-ep database (``analyse_epmap``,
  ``iter_pacemaps``);
- :mod:`pulse_ep_decay.integration.synthetic` — synthetic pace maps as pulse-ep
  objects with known truth;
- :mod:`pulse_ep_decay.cli` — ``pulse-ep-decay run | demo | seed-synthetic``.

Usage::

    from pulse_ep_decay import analyse_map, largest_component
    V, T, idx = largest_component(vertices, triangles)
    res = analyse_map(scores[idx >= 0], V, T)   # sigma, width_class, r2, flag_r2, F, flag_F, dAIC, status
"""

__version__ = "1.1.3"

from pulse_ep_decay.analysis import F_THRESHOLD, R2_THRESHOLD, analyse_map, focality_index, width_class
from pulse_ep_decay.mesh import ExactSolver, HeatSolver, heat_solver, largest_component, mean_edge_length
from pulse_ep_decay.profile import (
    BIN_WIDTH_MM,
    GAUSS_FWHM_FACTOR,
    SIGMA_BOUNDS_MM,
    bin_profile,
    find_origin,
    fit_exponential,
    fit_gaussian,
    gauss_model,
    synth_field,
)

__all__ = [
    # Algorithm 1 and its parameters
    "analyse_map",
    "focality_index",
    "width_class",
    "R2_THRESHOLD",
    "F_THRESHOLD",
    "BIN_WIDTH_MM",
    "SIGMA_BOUNDS_MM",
    "GAUSS_FWHM_FACTOR",
    # steps
    "largest_component",
    "mean_edge_length",
    "HeatSolver",
    "ExactSolver",
    "heat_solver",
    "find_origin",
    "bin_profile",
    "fit_gaussian",
    "fit_exponential",
    "gauss_model",
    # validation
    "synth_field",
]
