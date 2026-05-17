"""Adapter from the pulse-ep CARTO platform to ``pulse-ep-decay`` arrays.

This module is **optional**: it requires the ``pulse-ep`` package to be
installed alongside ``pulse-ep-decay``::

    pip install "pulse-ep-decay[pulse-ep]"

The functions here are thin extractors that pull NumPy arrays out of the
SQLAlchemy ORM models defined in :mod:`pulse_ep.core.models` so that the
array-only scientific kernel can analyse pulse-ep-managed CARTO studies
without further glue code.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, NamedTuple

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
    "MeshArrays",
    "from_epmap",
    "iter_pacemaps",
    "compute_cs_distance",
]


class MeshArrays(NamedTuple):
    """Plain array bundle extracted from an :class:`EPMapModel`."""

    vertices: np.ndarray   # (N, 3) mm
    triangles: np.ndarray  # (M, 3) int
    scores: np.ndarray     # (N,) %  — abs(act_bip[:, 0]) by pulse-ep convention


def from_epmap(epmap: EPMapModel) -> MeshArrays:
    """Extract ``(vertices, triangles, scores)`` from an :class:`EPMapModel`.

    Score convention follows the pulse-ep / CARTO workflow:
    ``abs(act_bip[:, 0])`` is the pace-mapping matching score in
    percent. NaN entries are preserved.

    Raises
    ------
    ValueError
        If the EPMap has no mesh data (vertices, triangles or act_bip
        are ``None`` / empty).
    """
    if epmap.vertices is None or epmap.triangles is None or epmap.act_bip is None:
        raise ValueError(f"EPMap {epmap.id} has no mesh data attached")

    vertices = np.array(epmap.vertices).reshape(-1, 3)
    triangles = np.array(epmap.triangles).reshape(-1, 3)
    act_bip = np.array(epmap.act_bip).reshape(-1, 2)
    scores = np.abs(act_bip[:, 0])
    return MeshArrays(vertices=vertices, triangles=triangles, scores=scores)


def iter_pacemaps(
    *,
    require_atrium: bool = True,
    require_part: bool = True,
    session: "Session | None" = None,
) -> Iterator[tuple[EPMapModel, EPMapAttributes, MeshArrays]]:
    """Iterate over all pace-mapping EPMaps in the connected pulse-ep DB.

    Yields ``(epmap_row, attributes_row, mesh_arrays)`` for each map
    whose ``attributes['pacemap']`` flag is ``True``. Filters can be
    relaxed via ``require_atrium=False`` / ``require_part=False``.

    A pulse-ep ``get_db_session()`` is opened automatically unless a
    caller-managed ``session`` is supplied.

    Examples
    --------
    >>> from pulse_ep_decay.integration.pulse_ep import iter_pacemaps
    >>> for epmap, attrs, mesh in iter_pacemaps():
    ...     print(epmap.id, mesh.vertices.shape, mesh.scores.max())
    """
    from sqlalchemy import Boolean

    def _scan(s: "Session") -> Iterator[tuple[EPMapModel, EPMapAttributes, MeshArrays]]:
        q = (
            s.query(EPMapModel, EPMapAttributes)
            .join(EPMapAttributes, EPMapAttributes.map_id == EPMapModel.id)
            .filter(EPMapAttributes.attributes["pacemap"].astext.cast(Boolean) == True)  # noqa: E712
        )
        for epmap, attrs in q.all():
            a = attrs.attributes or {}
            if require_atrium and not a.get("atrium"):
                continue
            if require_part and not a.get("part"):
                continue
            try:
                mesh = from_epmap(epmap)
            except ValueError:
                continue
            yield epmap, attrs, mesh

    if session is not None:
        yield from _scan(session)
        return

    with get_db_session() as managed:
        yield from _scan(managed)


def compute_cs_distance(
    session: "Session",
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
