"""Synthetic pace maps as pulse-ep objects (test fixtures, demo data).

Builds :class:`pulse_ep.EPMap` domain objects on the ellipsoidal pseudo-atrium
of ``pulse_ep.examples.demo_synthetic`` and paints score fields with known
decay width -- defined in **geodesic** distance (Heat Method), so the width
the analysis recovers can be checked against the truth.  The score protocol
is the manuscript's (:func:`pulse_ep_decay.profile.synth_field`: A = 35,
B = 65, clipped to [50, 100], 5 % of vertices unscored).

Scenarios (:data:`SCENARIOS`)::

    focal_6         sigma = 6 mm, in range; score field + legacy act_bip (modern CARTO import)
    focal_12        sigma = 12 mm, in range; score field only (EnSite-X-like import)
    broad_40        sigma = 40 mm, class 'broad'
    two_peaks       two sources 45 mm apart -> F < 0.70 (focality flag)
    flat_noise      no peak -> never an unflagged in-range map: width below range (< 2 mm) or R^2 < 0.7
    nan_holes       sigma = 8 mm with an unscored disc 25 mm from the origin
    two_components  sigma = 8 mm plus a detached mesh fragment (dropped by largest_component)
    legacy_act_bip  sigma = 10 mm, act_bip only, no scalar field (pre-2026 import)

The same objects can be written to a pulse-ep database with
:func:`seed_database` (PostgreSQL; the schema uses native arrays) -- the
``pulse-ep-decay seed-synthetic`` command does that.  Requires the
``pulse-ep`` extra (and pyvista through pulse-ep).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from pulse_ep import EPMap
    from pulse_ep.core.models import EPMapAttributes, EPMapModel, StudyModel
    from pulse_ep.examples.demo_synthetic import make_synthetic_atrium, triangle_areas
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pulse_ep_decay.integration.synthetic requires the 'pulse-ep' package. "
        "Install with: pip install 'pulse-ep-decay[pulse-ep]'"
    ) from exc

from pulse_ep_decay.mesh import HeatSolver
from pulse_ep_decay.profile import GAUSS_FWHM_FACTOR, SYNTH_A, SYNTH_B, SYNTH_CLIP, synth_field

__all__ = ["SCENARIOS", "SyntheticMap", "make_synthetic_pacemaps", "to_models", "seed_database"]

SCENARIOS = (
    "focal_6",
    "focal_12",
    "broad_40",
    "two_peaks",
    "flat_noise",
    "nan_holes",
    "two_components",
    "legacy_act_bip",
)
DEFAULT_RESOLUTION = 64  # ~6,000 vertices, mean edge ~2 mm (clinical range 0.95-2.78 mm)
STUDY_NAME = "pulse-ep-decay synthetic"


@dataclass
class SyntheticMap:
    """One synthetic pace map and what the analysis should find on it."""

    epmap: EPMap
    truth: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.epmap.map_name


def _origin_vertex(vertices: np.ndarray) -> int:
    """A vertex on the long side of the ellipsoid (away from the poles)."""
    return int(np.argmin(np.linalg.norm(vertices - np.array([25.0, 0.0, 0.0]), axis=1)))


def _epmap(name: str, V, T, scores, *, with_scalar=True, with_act_bip=False, source="synthetic") -> EPMap:
    kw = {}
    if with_act_bip:
        kw["act_bip"] = np.column_stack(
            [np.where(np.isfinite(scores), scores, np.nan), np.zeros(len(scores))]
        )
    m = EPMap(
        map_name=name,
        study_name=STUDY_NAME,
        map_number_of_points=int(np.isfinite(scores).sum()),
        triangles=np.asarray(T, int),
        vertices=np.asarray(V, float),
        triangle_areas=triangle_areas(V, T),
        **kw,
    )
    if with_scalar:
        m.register_scalar("pacemap_score", scores, kind="pacemap_score", unit="%", source=source)
    m.attributes = {"pacemap": True, "type": "synthetic", "atrium": "synthetic", "part": name}
    return m


def make_synthetic_pacemaps(
    scenarios: tuple[str, ...] | list[str] = SCENARIOS,
    resolution: int = DEFAULT_RESOLUTION,
    seed: int = 0,
    noise: float = 0.05,
) -> list[SyntheticMap]:
    """Build the requested scenarios (see module docstring)."""
    rng = np.random.default_rng(seed)
    V, T = make_synthetic_atrium(resolution=resolution)
    V = np.ascontiguousarray(V, dtype=float)
    o = _origin_vertex(V)
    solver = HeatSolver(V, T)
    d0 = solver(o)
    out: list[SyntheticMap] = []

    def gauss(sig, d=d0):
        return synth_field(d, "gaussian", GAUSS_FWHM_FACTOR * sig, noise, rng)

    for sc in scenarios:
        if sc not in SCENARIOS:
            raise ValueError(f"unknown scenario {sc!r}; choose from {SCENARIOS}")
        truth = dict(
            scenario=sc,
            origin=o,
            sigma_true=None,
            expect_class=None,
            expect_flag_r2=False,
            expect_flag_F=False,
        )
        if sc == "focal_6":
            s = gauss(6.0)
            m = _epmap(sc, V, T, s, with_scalar=True, with_act_bip=True, source="synthetic:carto-like")
            truth.update(sigma_true=6.0, expect_class="in range")
        elif sc == "focal_12":
            s = gauss(12.0)
            m = _epmap(sc, V, T, s, source="synthetic:ensite-like")
            truth.update(sigma_true=12.0, expect_class="in range")
        elif sc == "broad_40":
            s = gauss(40.0)
            m = _epmap(sc, V, T, s)
            truth.update(sigma_true=40.0, expect_class="broad")
        elif sc == "two_peaks":
            # second source 45 mm away on the same latitude band (comparable vertex density)
            band = np.abs(V[:, 2]) < 5.0
            far = int(np.flatnonzero(band)[np.argmin(np.abs(d0[band] - 45.0))])
            d1 = solver(far)
            a = SYNTH_A * np.exp(-(d0**2) / (2 * 7.0**2))
            b = SYNTH_A * np.exp(-(d1**2) / (2 * 7.0**2))
            s = np.clip(SYNTH_B + np.maximum(a, b) + rng.normal(0, noise * SYNTH_A, len(V)), *SYNTH_CLIP)
            s[rng.random(len(s)) < 0.05] = np.nan
            m = _epmap(sc, V, T, s)
            truth.update(sigma_true=7.0, second_origin=far, expect_flag_F=True)
        elif sc == "flat_noise":
            s = np.clip(SYNTH_B + rng.normal(0, noise * SYNTH_A, len(V)), *SYNTH_CLIP)
            s[rng.random(len(s)) < 0.05] = np.nan
            m = _epmap(sc, V, T, s)
            truth.update(expect_flagged=True)  # below-range width or R^2 flag, seed-dependent
        elif sc == "nan_holes":
            s = gauss(8.0)
            hole = int(np.argmin(np.abs(d0 - 25.0)))
            s[solver(hole) < 8.0] = np.nan
            m = _epmap(sc, V, T, s)
            truth.update(sigma_true=8.0, expect_class="in range", hole_vertex=hole)
        elif sc == "two_components":
            s = gauss(8.0)
            Vb, Tb = make_synthetic_atrium(resolution=10, radius_xyz=(5.0, 5.0, 5.0))
            Vb = np.asarray(Vb, float) + np.array([0.0, 0.0, 70.0])  # detached blob above the north pole
            V2 = np.vstack([V, Vb])
            T2 = np.vstack([T, np.asarray(Tb, int) + len(V)])
            s2 = np.r_[s, np.full(len(Vb), 60.0)]
            m = _epmap(sc, V2, T2, s2)
            truth.update(
                sigma_true=8.0, expect_class="in range", n_vertices_main=len(V), n_vertices_total=len(V2)
            )
        elif sc == "legacy_act_bip":
            s = gauss(10.0)
            m = _epmap(sc, V, T, s, with_scalar=False, with_act_bip=True)
            truth.update(sigma_true=10.0, expect_class="in range", expect_score_source="act_bip")
        out.append(SyntheticMap(epmap=m, truth=truth))
    return out


def to_models(maps: list[SyntheticMap], study: StudyModel) -> list[tuple[EPMapModel, EPMapAttributes]]:
    """ORM rows for the maps (``study.id`` must be set, i.e. the study flushed)."""
    rows = []
    for sm in maps:
        model = EPMapModel.from_epmap(sm.epmap, study_id=study.id)
        attrs = EPMapAttributes()
        attrs.attributes = dict(sm.epmap.attributes, decay_truth=_jsonable(sm.truth))
        rows.append((model, attrs))
    return rows


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = None if not np.isfinite(v) else float(v)
        else:
            out[k] = v
    return out


def seed_database(session, maps: list[SyntheticMap] | None = None, study_name: str = STUDY_NAME) -> list[int]:
    """Write the synthetic maps into a pulse-ep database; returns the new map ids.

    Creates a study ``study_name`` (vendor ``synthetic``); maps are tagged
    ``pacemap = True`` and carry the truth under ``decay_truth`` in their
    attributes.  Needs a PostgreSQL pulse-ep database with the schema in place
    (``pulse-ep-init``).
    """
    if maps is None:
        maps = make_synthetic_pacemaps()
    study = StudyModel(
        name=study_name, vendor="synthetic", provenance={"generator": "pulse-ep-decay.integration.synthetic"}
    )
    session.add(study)
    session.flush()
    ids = []
    for model, attrs in to_models(maps, study):
        session.add(model)
        session.flush()
        attrs.map_id = model.id
        session.add(attrs)
        ids.append(int(model.id))
    session.commit()
    return ids
