"""Adapter from pulse-ep to ``pulse-ep-decay`` arrays.

This module is **optional**: it requires the ``pulse-ep`` package to be
installed alongside ``pulse-ep-decay``::

    pip install "pulse-ep-decay[pulse-ep]"

pulse-ep imports CARTO 3 and EnSite X studies into one data structure
(:class:`pulse_ep.core.models.EPMapModel`, with the domain object
:class:`pulse_ep.EPMap`).  The functions here pull the three arrays the
decay analysis needs out of either object -- vertices, triangles and one
pace-mapping score per vertex -- so that :func:`pulse_ep_decay.analyse_map`
runs on any pace map pulse-ep holds, whatever system acquired it.

Score field
-----------
The score is taken from the vendor-neutral scalar field of kind
``pacemap_score`` (both importers register it; EnSite X maps carry nothing
else).  Maps imported by older pulse-ep versions have only the legacy CARTO
array ``act_bip``; there the score is ``abs(act_bip[:, 0])``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

try:
    from pulse_ep.core.database import get_db_session
    from pulse_ep.core.models import EPMapAttributes, EPMapModel, EPMapPoint
except ImportError as exc:  # pragma: no cover -- guard against missing extra
    raise ImportError(
        "pulse_ep_decay.integration.pulse_ep requires the 'pulse-ep' package. "
        "Install with: pip install 'pulse-ep-decay[pulse-ep]'"
    ) from exc

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

__all__ = [
    "PACEMAP_SCORE_KIND",
    "MeshArrays",
    "score_field",
    "from_epmap",
    "is_pacemap",
    "analyse_epmap",
    "iter_pacemaps",
    "compute_cs_distance",
]

#: kind under which both pulse-ep importers register the pace-mapping score
PACEMAP_SCORE_KIND = "pacemap_score"


class MeshArrays(NamedTuple):
    """Plain array bundle extracted from an EPMap (model or domain object)."""

    vertices: np.ndarray  # (N, 3) mm
    triangles: np.ndarray  # (M, 3) int
    scores: np.ndarray  # (N,) %, NaN where no score
    score_source: str  # 'scalar_fields:<name>' or 'act_bip'


def _scalar_fields(epmap: Any) -> dict:
    """``{name: (values, kind)}`` from a model (dicts) or a domain object (ScalarField)."""
    fields = getattr(epmap, "scalar_fields", None) or {}
    out = {}
    for name, f in fields.items():
        if isinstance(f, dict):
            out[name] = (np.asarray(f["values"], dtype=float), f.get("kind"))
        else:
            out[name] = (np.asarray(f.values, dtype=float), getattr(f, "kind", None))
    return out


def score_field(epmap: Any, n_vertices: int | None = None) -> tuple[np.ndarray, str] | None:
    """The pace-mapping score per vertex and where it came from, or ``None``.

    Order: a scalar field named ``pacemap_score``; else the first field of
    that kind; else ``abs(act_bip[:, 0])``.  A field whose length does not
    match ``n_vertices`` (a per-point field) is skipped.
    """
    fields = _scalar_fields(epmap)
    candidates = []
    if PACEMAP_SCORE_KIND in fields:
        candidates.append(PACEMAP_SCORE_KIND)
    candidates += [n for n, (_, k) in fields.items() if k == PACEMAP_SCORE_KIND and n != PACEMAP_SCORE_KIND]
    for name in candidates:
        values = fields[name][0].ravel()
        if n_vertices is None or len(values) == n_vertices:
            return values, "scalar_fields:" + name
    act_bip = getattr(epmap, "act_bip", None)
    if act_bip is not None and len(act_bip) > 0:
        a = np.asarray(act_bip, dtype=float)
        a = a.reshape(-1, 2) if a.ndim == 1 else a
        values = np.abs(a[:, 0])
        if n_vertices is None or len(values) == n_vertices:
            return values, "act_bip"
    return None


def from_epmap(epmap: Any) -> MeshArrays:
    """Extract ``(vertices, triangles, scores, score_source)`` from an EPMap.

    ``epmap`` is an :class:`EPMapModel` row or a :class:`pulse_ep.EPMap`.
    NaN scores are preserved (vertices without a score).

    Raises
    ------
    ValueError
        If the map has no mesh or no per-vertex pace-mapping score.
    """
    ident = getattr(epmap, "id", None) or getattr(epmap, "map_name", "?")
    if epmap.vertices is None or epmap.triangles is None or len(epmap.vertices) == 0:
        raise ValueError(f"EPMap {ident} has no mesh data attached")
    vertices = np.asarray(epmap.vertices, dtype=float).reshape(-1, 3)
    triangles = np.asarray(epmap.triangles, dtype=int).reshape(-1, 3)
    found = score_field(epmap, n_vertices=len(vertices))
    if found is None:
        raise ValueError(
            f"EPMap {ident} has no per-vertex pace-mapping score (scalar field '{PACEMAP_SCORE_KIND}' or act_bip)"
        )
    scores, source = found
    return MeshArrays(vertices=vertices, triangles=triangles, scores=scores, score_source=source)


def is_pacemap(epmap: Any, attributes: dict | None = None) -> bool:
    """True when the map is tagged ``pacemap`` or carries a pace-mapping score field.

    ``attributes`` is the dict of :class:`EPMapAttributes` (``pulse-ep-tag-maps``
    sets ``pacemap``); a map without a tag counts when it registers a scalar
    field of kind ``pacemap_score`` (the importers' own declaration).
    """
    a = attributes if attributes is not None else getattr(epmap, "attributes", None) or {}
    if isinstance(a, dict) and "pacemap" in a and a["pacemap"] is not None:
        return bool(a["pacemap"])
    return any(k == PACEMAP_SCORE_KIND for _, k in _scalar_fields(epmap).values())


def analyse_epmap(epmap: Any, **kwargs) -> dict:
    """Run the decay analysis on one pulse-ep map (largest mesh component).

    Returns the dict of :func:`pulse_ep_decay.analyse_map` plus ``map_id``,
    ``map_name``, ``study_name``, ``n_vertices`` (largest component),
    ``n_vertices_total``, ``n_scored`` and ``score_source``.  ``kwargs`` go
    to :func:`analyse_map` (e.g. ``origin=`` for a known pacing site, given
    as an index into the full mesh).
    """
    from pulse_ep_decay.analysis import analyse_map
    from pulse_ep_decay.mesh import largest_component

    arrays = from_epmap(epmap)
    V, T, idx = largest_component(arrays.vertices, arrays.triangles)
    s = arrays.scores[idx >= 0]
    if "origin" in kwargs and kwargs["origin"] is not None:
        o = int(idx[int(kwargs["origin"])])
        if o < 0:
            raise ValueError("origin vertex lies outside the largest mesh component")
        kwargs["origin"] = o
    res = analyse_map(s, V, T, **kwargs)
    res.update(
        map_id=getattr(epmap, "id", None),
        map_name=getattr(epmap, "map_name", None),
        study_name=getattr(epmap, "study_name", None),
        n_vertices=int(len(V)),
        n_vertices_total=int(len(arrays.vertices)),
        n_scored=int(np.isfinite(s).sum()),
        score_source=arrays.score_source,
    )
    return res


def iter_pacemaps(
    *,
    session: Session | None = None,
    map_ids: list[int] | None = None,
    pacemaps_only: bool = True,
    require_atrium: bool = False,
    require_part: bool = False,
) -> Iterator[tuple[EPMapModel, dict, MeshArrays]]:
    """Iterate over the pace maps of a pulse-ep database.

    Yields ``(epmap_row, attributes_dict, mesh_arrays)``.  With
    ``pacemaps_only`` (default) a map is included when :func:`is_pacemap`
    holds; ``map_ids`` restricts to the given ids (and ignores the pace-map
    test).  ``require_atrium`` / ``require_part`` additionally demand those
    tags (the study-specific filter of the method paper).  Maps without a
    usable mesh or score are skipped.

    A pulse-ep ``get_db_session()`` (configured through ``PULSE_EP_DATABASE_URL``
    etc.) is opened unless a ``session`` is supplied.
    """

    def _scan(s: Session) -> Iterator[tuple[EPMapModel, dict, MeshArrays]]:
        q = s.query(EPMapModel.id).order_by(EPMapModel.id)
        if map_ids is not None:
            q = q.filter(EPMapModel.id.in_(list(map_ids)))
        for (mid,) in q.all():
            epmap = s.get(EPMapModel, mid)
            attrs_row = s.get(EPMapAttributes, mid)
            attrs = (attrs_row.attributes if attrs_row is not None else None) or {}
            if map_ids is None and pacemaps_only and not is_pacemap(epmap, attrs):
                continue
            if require_atrium and not attrs.get("atrium"):
                continue
            if require_part and not attrs.get("part"):
                continue
            try:
                mesh = from_epmap(epmap)
            except ValueError:
                continue
            yield epmap, attrs, mesh
            s.expunge(epmap)  # free the arrays

    if session is not None:
        yield from _scan(session)
        return
    with get_db_session() as managed:
        yield from _scan(managed)


def compute_cs_distance(
    session: Session,
    map_id: int,
    point_xyz: np.ndarray,
) -> dict | None:
    """Coronary-sinus catheter distance from the mapping region centroid.

    Only valid for **single-chamber reference** maps: all 10 CS
    electrodes per measurement point are assumed to belong to one
    physical catheter in the coronary sinus. For double-chamber
    references the positions are ambiguous (pin-box) and this analysis
    is not applicable.

    Returns
    -------
    dict or None
        ``{'region_centroid', 'cs_centroid', 'cs_distance_mm',
        'n_points_with_cs'}`` — or ``None`` when no CS data was found.
    """
    cs_rows = (
        session.query(EPMapPoint.cs_positions)
        .filter(EPMapPoint.map_id == map_id, EPMapPoint.cs_positions.isnot(None))
        .all()
    )
    if not cs_rows or point_xyz is None or len(point_xyz) == 0:
        return None

    electrodes: list[np.ndarray] = []
    for (cs_pos,) in cs_rows:
        arr = np.array(cs_pos).reshape(-1, 3)
        if len(arr) >= 10:
            electrodes.append(arr[:10])
    if not electrodes:
        return None

    cs_centroid = np.vstack(electrodes).mean(axis=0)
    region_centroid = point_xyz.mean(axis=0)
    return {
        "region_centroid": region_centroid,
        "cs_centroid": cs_centroid,
        "cs_distance_mm": float(np.linalg.norm(cs_centroid - region_centroid)),
        "n_points_with_cs": len(electrodes),
    }
