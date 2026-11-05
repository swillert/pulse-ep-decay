"""The example scripts run and recover the injected width."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _run(script: Path, argv: list[str], capsys) -> str:
    old = sys.argv
    sys.argv = [str(script), *argv]
    try:
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as e:
            assert e.code in (0, None)
    finally:
        sys.argv = old
    return capsys.readouterr().out


def test_sigma_recovery(capsys):
    out = _run(EXAMPLES / "sigma_recovery.py", ["--sigma-true", "8", "--seed", "1"], capsys)
    line = next(ln for ln in out.splitlines() if "recovered" in ln)
    sigma = float(line.split("sigma   =")[1].split("mm")[0])
    assert abs(sigma - 8.0) < 0.8


def test_export_radial_profile(tmp_path, capsys):
    out = tmp_path / "radial_profile.csv"
    _run(EXAMPLES / "r" / "export_radial_profile.py", ["--sigma-true", "6", "--output", str(out)], capsys)
    lines = out.read_text().splitlines()
    header = {ln[2:].split(":")[0]: ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("#")}
    assert header["sigma_true_mm"] == "6.0" and abs(float(header["python_sigma_mm"]) - 6.0) < 0.6
    assert lines[len(header)] == "geodesic_distance_mm,score"
    assert len(lines) - len(header) - 1 == int(header["n_vertices"])
