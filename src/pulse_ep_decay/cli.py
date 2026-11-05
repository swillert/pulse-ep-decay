"""Command line: ``pulse-ep-decay``.

::

    pulse-ep-decay run  --db postgresql://... --out widths.csv   # every pace map of a pulse-ep database
    pulse-ep-decay run  --db ... --map-ids 12 13 --out widths.csv
    pulse-ep-decay run  --npz meshes/ --out widths.csv           # .npz files: vertices, triangles, scores
    pulse-ep-decay demo --out widths.csv                         # synthetic maps, no database needed
    pulse-ep-decay seed-synthetic --db ...                       # write the synthetic maps into pulse-ep

One row per map with the quantities of Supplementary Table S12 of the method
paper: width sigma (mm) and class, R^2 and its flag, focality index F with the
number of half-maximum components and its flag, Delta AIC (Gaussian minus
exponential), fitted amplitude and baseline, number of bins, and a status.
``--db`` overrides pulse-ep's own configuration (``PULSE_EP_DATABASE_URL``
or ``.env``); the ``run --db`` and ``seed-synthetic`` commands need the
``pulse-ep`` extra.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections.abc import Iterable

import numpy as np

from pulse_ep_decay import __version__
from pulse_ep_decay.analysis import analyse_map
from pulse_ep_decay.mesh import largest_component

COLUMNS = [
    "map_id",
    "map_name",
    "study_name",
    "n_vertices",
    "n_scored",
    "origin",
    "sigma",
    "width_class",
    "r2",
    "flag_r2",
    "F",
    "n_components",
    "flag_F",
    "dAIC",
    "A",
    "B",
    "n_bins",
    "score_source",
    "status",
]


def _row(res: dict) -> dict:
    out = {}
    for k in COLUMNS:
        v = res.get(k)
        if isinstance(v, float):
            v = "" if not np.isfinite(v) else round(v, 4)
        out[k] = "" if v is None else v
    return out


def write_rows(rows: Iterable[dict], out: str | None, fmt: str = "csv") -> int:
    rows = list(rows)
    fh = open(out, "w", newline="") if out else sys.stdout
    try:
        if fmt == "json":
            json.dump(rows, fh, indent=1)
            fh.write("\n")
        else:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
    finally:
        if out:
            fh.close()
    return len(rows)


def _say(msg: str) -> None:
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
def _session(url: str | None):
    """A SQLAlchemy session on a pulse-ep database (``url`` or pulse-ep's config)."""
    try:
        from pulse_ep.core.database import create_db_engine, get_db_session
    except ImportError as exc:
        raise SystemExit("pulse-ep is not installed: pip install 'pulse-ep-decay[pulse-ep]'") from exc
    if url is None:
        return get_db_session()
    from contextlib import contextmanager

    from sqlalchemy.orm import sessionmaker

    engine = create_db_engine(url)
    Session = sessionmaker(engine)

    @contextmanager
    def managed():
        s = Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    return managed()


def run_db(url: str | None, map_ids: list[int] | None, all_maps: bool) -> list[dict]:
    from pulse_ep_decay.integration.pulse_ep import analyse_epmap, iter_pacemaps

    rows = []
    with _session(url) as s:
        for epmap, _attrs, _mesh in iter_pacemaps(session=s, map_ids=map_ids, pacemaps_only=not all_maps):
            res = analyse_epmap(epmap)
            _say(
                f"map {res['map_id']:>5} {str(res['map_name'])[:28]:28s} sigma {res['sigma']:7.2f} mm  {res['width_class']:11s} R2 {res['r2']:.2f}  F {res['F']:.2f}  {res['status']}"
            )
            rows.append(_row(res))
    return rows


def run_npz(folder: str) -> list[dict]:
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz files in {folder}")
    rows = []
    for f in files:
        z = np.load(f)
        missing = [k for k in ("vertices", "triangles", "scores") if k not in z]
        if missing:
            _say(f"{os.path.basename(f)}: skipped, missing {missing}")
            continue
        V0 = np.asarray(z["vertices"], float).reshape(-1, 3)
        T0 = np.asarray(z["triangles"], int).reshape(-1, 3)
        s0 = np.asarray(z["scores"], float).ravel()
        V, T, idx = largest_component(V0, T0)
        s = s0[idx >= 0]
        res = analyse_map(s, V, T)
        res.update(
            map_id=os.path.splitext(os.path.basename(f))[0],
            map_name=os.path.basename(f),
            study_name="",
            n_vertices=len(V),
            n_scored=int(np.isfinite(s).sum()),
            score_source="npz:scores",
        )
        _say(
            f"{res['map_name']:28s} sigma {res['sigma']:7.2f} mm  {res['width_class']:11s} R2 {res['r2']:.2f}  F {res['F']:.2f}  {res['status']}"
        )
        rows.append(_row(res))
    return rows


def _need_pulse_ep(what: str):
    try:
        import pulse_ep  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"{what} needs pulse-ep: pip install 'pulse-ep-decay[pulse-ep]'") from exc


def run_demo(scenarios: list[str] | None, resolution: int, seed: int) -> list[dict]:
    _need_pulse_ep("the synthetic maps")
    from pulse_ep_decay.integration.pulse_ep import analyse_epmap
    from pulse_ep_decay.integration.synthetic import SCENARIOS, make_synthetic_pacemaps

    rows = []
    for sm in make_synthetic_pacemaps(
        tuple(scenarios) if scenarios else SCENARIOS, resolution=resolution, seed=seed
    ):
        res = analyse_epmap(sm.epmap)
        res["map_id"] = sm.name
        t = sm.truth
        truth = f"true sigma {t['sigma_true']}" if t.get("sigma_true") is not None else "no peak"
        _say(
            f"{sm.name:15s} sigma {res['sigma']:7.2f} mm ({truth})  {res['width_class']:11s} R2 {res['r2']:.2f}  F {res['F']:.2f}  flags r2={res['flag_r2']} F={res['flag_F']}  {res['status']}"
        )
        rows.append(_row(res))
    return rows


def seed_synthetic(
    url: str | None, scenarios: list[str] | None, resolution: int, seed: int, study_name: str | None
) -> list[int]:
    _need_pulse_ep("seed-synthetic")
    from pulse_ep_decay.integration.synthetic import (
        SCENARIOS,
        STUDY_NAME,
        make_synthetic_pacemaps,
        seed_database,
    )

    maps = make_synthetic_pacemaps(
        tuple(scenarios) if scenarios else SCENARIOS, resolution=resolution, seed=seed
    )
    with _session(url) as s:
        ids = seed_database(s, maps, study_name=study_name or STUDY_NAME)
    for sm, i in zip(maps, ids, strict=True):
        _say(f"map {i:>5}  {sm.name}")
    return ids


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pulse-ep-decay", description="Geodesic decay analysis of pace-mapping score maps."
    )
    p.add_argument("--version", action="version", version=f"pulse-ep-decay {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="analyse the pace maps of a pulse-ep database or a folder of .npz files")
    src = r.add_mutually_exclusive_group()
    src.add_argument(
        "--db",
        metavar="URL",
        help="SQLAlchemy URL of the pulse-ep database (default: pulse-ep's own configuration)",
    )
    src.add_argument(
        "--npz",
        metavar="DIR",
        help="folder of .npz files with arrays vertices (N,3), triangles (M,3), scores (N,)",
    )
    r.add_argument("--map-ids", type=int, nargs="+", metavar="ID", help="only these map ids")
    r.add_argument(
        "--all",
        action="store_true",
        help="every map with a score field, not only maps tagged/declared as pace maps",
    )
    r.add_argument("--out", metavar="FILE", help="write the table here (default: stdout)")
    r.add_argument("--format", choices=("csv", "json"), default="csv")

    d = sub.add_parser("demo", help="analyse the built-in synthetic pace maps (needs pulse-ep, no database)")
    d.add_argument("--scenarios", nargs="+", metavar="NAME")
    d.add_argument("--resolution", type=int, default=64)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", metavar="FILE")
    d.add_argument("--format", choices=("csv", "json"), default="csv")

    s = sub.add_parser("seed-synthetic", help="write the synthetic pace maps into a pulse-ep database")
    s.add_argument("--db", metavar="URL")
    s.add_argument("--scenarios", nargs="+", metavar="NAME")
    s.add_argument("--resolution", type=int, default=64)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--study-name")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "run":
        rows = run_npz(a.npz) if a.npz else run_db(a.db, a.map_ids, a.all)
        n = write_rows(rows, a.out, a.format)
        _say(f"{n} maps" + (f" -> {a.out}" if a.out else ""))
        return 0
    if a.cmd == "demo":
        n = write_rows(run_demo(a.scenarios, a.resolution, a.seed), a.out, a.format)
        _say(f"{n} synthetic maps" + (f" -> {a.out}" if a.out else ""))
        return 0
    if a.cmd == "seed-synthetic":
        ids = seed_synthetic(a.db, a.scenarios, a.resolution, a.seed, a.study_name)
        _say(f"{len(ids)} maps written")
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
