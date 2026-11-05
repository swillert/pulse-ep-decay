"""Command line on .npz folders and on the synthetic maps."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from pulse_ep_decay.cli import COLUMNS, main
from pulse_ep_decay.profile import GAUSS_FWHM_FACTOR, synth_field

from .test_pulse_ep_decay import great_circle, icosphere


@pytest.fixture
def npz_dir(tmp_path):
    V, T = icosphere(subdiv=3)
    rng = np.random.default_rng(0)
    d = great_circle(V, 0)
    for i, sig in enumerate((5.0, 12.0)):
        np.savez(
            tmp_path / f"mesh_{i:03d}.npz",
            vertices=V,
            triangles=T,
            scores=synth_field(d, "gaussian", GAUSS_FWHM_FACTOR * sig, 0.05, rng),
        )
    np.savez(tmp_path / "mesh_noscores.npz", vertices=V, triangles=T)  # skipped
    return tmp_path


def test_run_npz_csv(npz_dir, tmp_path, capsys):
    out = tmp_path / "widths.csv"
    assert main(["run", "--npz", str(npz_dir), "--out", str(out)]) == 0
    rows = list(csv.DictReader(open(out)))
    assert [r["map_id"] for r in rows] == ["mesh_000", "mesh_001"]
    assert list(rows[0]) == COLUMNS
    assert abs(float(rows[0]["sigma"]) - 5.0) < 0.6 and abs(float(rows[1]["sigma"]) - 12.0) < 1.3
    assert rows[0]["width_class"] == "in range" and rows[0]["status"] == "ok"
    assert "missing" in capsys.readouterr().err


def test_run_npz_json_stdout(npz_dir, capsys):
    assert main(["run", "--npz", str(npz_dir), "--format", "json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 2 and rows[0]["score_source"] == "npz:scores"


def test_run_npz_empty(tmp_path):
    with pytest.raises(SystemExit):
        main(["run", "--npz", str(tmp_path)])


def test_demo(tmp_path):
    pytest.importorskip("pulse_ep")
    out = tmp_path / "demo.csv"
    assert main(["demo", "--scenarios", "focal_6", "two_peaks", "--out", str(out)]) == 0
    rows = list(csv.DictReader(open(out)))
    assert [r["map_id"] for r in rows] == ["focal_6", "two_peaks"]
    assert rows[1]["flag_F"] == "True"
