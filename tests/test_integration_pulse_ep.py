"""Adapter and synthetic fixtures against pulse-ep objects (no database needed).

The synthetic maps are pulse-ep ``EPMap`` objects converted to ``EPMapModel``
rows exactly as pulse-ep stores them; the adapter must read the score from
the vendor-neutral scalar field (EnSite-X-like maps have nothing else) and
fall back to the legacy CARTO ``act_bip`` array, and the analysis must
recover the known widths and raise the right flags.

The database round trip (``seed_database`` -> ``iter_pacemaps`` -> CLI) runs
only when ``PULSE_EP_TEST_DATABASE_URL`` names a PostgreSQL pulse-ep database
whose schema exists (``pulse-ep-init``); the schema uses native arrays, so
SQLite cannot stand in.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("pulse_ep")

from pulse_ep.core.models import EPMapModel  # noqa: E402

from pulse_ep_decay.integration.pulse_ep import (  # noqa: E402
    PACEMAP_SCORE_KIND,
    analyse_epmap,
    from_epmap,
    is_pacemap,
    score_field,
)
from pulse_ep_decay.integration.synthetic import SCENARIOS, make_synthetic_pacemaps  # noqa: E402


@pytest.fixture(scope="module")
def maps():
    return {sm.name: sm for sm in make_synthetic_pacemaps(seed=0)}


@pytest.fixture(scope="module")
def models(maps):
    """ORM rows as pulse-ep would store them (nested lists, JSON scalar fields)."""
    return {name: EPMapModel.from_epmap(sm.epmap, study_id=1) for name, sm in maps.items()}


def test_scenarios_complete(maps):
    assert set(maps) == set(SCENARIOS)


def test_score_source_modern_and_legacy(models):
    assert from_epmap(models["focal_12"]).score_source == "scalar_fields:" + PACEMAP_SCORE_KIND  # EnSite-like
    assert (
        from_epmap(models["focal_6"]).score_source == "scalar_fields:" + PACEMAP_SCORE_KIND
    )  # CARTO: field preferred
    assert from_epmap(models["legacy_act_bip"]).score_source == "act_bip"
    assert models["legacy_act_bip"].scalar_fields in (None, {})


def test_model_roundtrip_preserves_nan(maps, models):
    s_dom = maps["focal_12"].epmap.get_scalar("pacemap_score")
    s_row = from_epmap(models["focal_12"]).scores
    assert s_row.shape == s_dom.shape
    assert np.array_equal(np.isnan(s_row), np.isnan(s_dom))
    assert np.allclose(s_row[np.isfinite(s_row)], s_dom[np.isfinite(s_dom)])


def test_arrays_shapes(models):
    a = from_epmap(models["focal_6"])
    assert a.vertices.ndim == 2 and a.vertices.shape[1] == 3
    assert a.triangles.ndim == 2 and a.triangles.shape[1] == 3
    assert a.scores.shape == (len(a.vertices),)


def test_no_score_raises(maps):
    m = EPMapModel.from_epmap(maps["focal_12"].epmap, study_id=1)
    m.scalar_fields = None
    m.act_bip = None
    with pytest.raises(ValueError, match="pace-mapping score"):
        from_epmap(m)
    assert score_field(m) is None


def test_is_pacemap(maps, models):
    assert is_pacemap(models["focal_12"], {"pacemap": True})
    assert not is_pacemap(models["focal_12"], {"pacemap": False})
    assert is_pacemap(models["focal_12"], {})  # declared by the score field
    assert not is_pacemap(models["legacy_act_bip"], {})  # legacy array alone does not declare a pace map
    assert is_pacemap(maps["focal_12"].epmap)  # domain object: attributes carry the tag


@pytest.mark.parametrize(
    "name", ["focal_6", "focal_12", "broad_40", "nan_holes", "two_components", "legacy_act_bip"]
)
def test_width_recovered(models, maps, name):
    r = analyse_epmap(models[name])
    t = maps[name].truth
    assert abs(r["sigma"] - t["sigma_true"]) < 0.05 * t["sigma_true"] + 0.3
    assert r["width_class"] == t["expect_class"]
    assert not r["flag_r2"] and not r["flag_F"]
    assert r["status"] == "ok"


def test_two_components_dropped(models, maps):
    r = analyse_epmap(models["two_components"])
    t = maps["two_components"].truth
    assert r["n_vertices"] == t["n_vertices_main"]
    assert r["n_vertices_total"] == t["n_vertices_total"]


def test_two_peaks_flagged(models):
    r = analyse_epmap(models["two_peaks"])
    assert r["n_components"] == 2
    assert r["F"] < 0.70 and r["flag_F"]
    assert not r["flag_r2"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_noise_never_passes(seed):
    (sm,) = make_synthetic_pacemaps(("flat_noise",), seed=seed)
    r = analyse_epmap(sm.epmap)
    assert r["width_class"] == "below range" or r["flag_r2"]


def test_known_origin_mode(models, maps):
    t = maps["focal_6"].truth
    r = analyse_epmap(models["focal_6"], origin=t["origin"])
    assert r["origin"] == t["origin"]  # full mesh == largest component here
    assert abs(r["sigma"] - 6.0) < 0.6


def test_metadata(models):
    r = analyse_epmap(models["focal_12"])
    assert r["map_name"] == "focal_12" and r["study_name"].startswith("pulse-ep-decay")
    assert 0 < r["n_scored"] <= r["n_vertices"]


# ---------------------------------------------------------------------------
DB_URL = os.environ.get("PULSE_EP_TEST_DATABASE_URL")


@pytest.mark.skipif(not DB_URL, reason="PULSE_EP_TEST_DATABASE_URL not set (PostgreSQL pulse-ep database)")
def test_database_roundtrip_and_cli(tmp_path):
    import csv

    from pulse_ep.core.database import create_db_engine
    from pulse_ep.core.models import EPMapAttributes, EPMapModel, StudyModel
    from sqlalchemy.orm import sessionmaker

    from pulse_ep_decay.cli import main
    from pulse_ep_decay.integration.pulse_ep import iter_pacemaps
    from pulse_ep_decay.integration.synthetic import seed_database

    Session = sessionmaker(create_db_engine(DB_URL))
    s = Session()
    study_name = "pulse-ep-decay test " + os.urandom(3).hex()
    ids: list[int] = []
    try:
        maps = make_synthetic_pacemaps(("focal_12", "two_peaks", "legacy_act_bip"), seed=0)
        ids = seed_database(s, maps, study_name=study_name)
        got = {e.id: (attrs, mesh) for e, attrs, mesh in iter_pacemaps(session=s, map_ids=ids)}
        assert set(got) == set(ids)
        assert got[ids[2]][1].score_source == "act_bip"
        assert got[ids[0]][1].score_source == "scalar_fields:pacemap_score"
        assert got[ids[0]][0]["decay_truth"]["sigma_true"] == 12.0
        out = tmp_path / "widths.csv"
        assert main(["run", "--db", DB_URL, "--map-ids", *map(str, ids), "--out", str(out)]) == 0
        rows = list(csv.DictReader(open(out)))
        assert [int(r["map_id"]) for r in rows] == ids
        assert abs(float(rows[0]["sigma"]) - 12.0) < 0.9
        assert rows[1]["flag_F"] == "True"
        assert rows[2]["score_source"] == "act_bip"
    finally:
        # the seed commits; remove the test study again
        s.rollback()
        if ids:
            s.query(EPMapAttributes).filter(EPMapAttributes.map_id.in_(ids)).delete(synchronize_session=False)
            s.query(EPMapModel).filter(EPMapModel.id.in_(ids)).delete(synchronize_session=False)
        s.query(StudyModel).filter(StudyModel.name == study_name).delete(synchronize_session=False)
        s.commit()
        s.close()
