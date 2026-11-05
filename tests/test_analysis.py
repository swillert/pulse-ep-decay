"""Algorithm-level tests of the public entry point and its building blocks
(plateau tie rule, focality fallback, Delta AIC sign, status strings, classes)."""

from __future__ import annotations

import numpy as np
import pytest

from pulse_ep_decay import HeatSolver, analyse_map, focality_index, largest_component, width_class
from pulse_ep_decay.analysis import CLASS_EDGES_MM, F_THRESHOLD, R2_THRESHOLD
from pulse_ep_decay.legacy import find_origin_first
from pulse_ep_decay.profile import GAUSS_FWHM_FACTOR, find_origin, synth_field

from .test_pulse_ep_decay import R, great_circle, icosphere


@pytest.fixture(scope="module")
def sphere():
    V, T = icosphere()
    V, T, _ = largest_component(V, T)
    return V, T


def test_width_class_edges_exact():
    lo, hi_in, hi_broad, hi_diff = CLASS_EDGES_MM
    assert width_class(lo - 1e-9) == "below range" and width_class(lo) == "in range"
    assert width_class(hi_in - 1e-9) == "in range" and width_class(hi_in) == "broad"
    assert width_class(hi_broad - 1e-9) == "broad" and width_class(hi_broad) == "diffuse"
    assert width_class(hi_diff - 1e-9) == "diffuse" and width_class(hi_diff) == "n.i."
    assert width_class(np.nan) == "n.i." and width_class(None) == "n.i."


def test_plateau_tie_rule(sphere):
    V, T = sphere
    d = great_circle(V, 0)
    s = np.clip(100.0 - 2.0 * d, 50, 100)
    s[d < 6.0] = 100.0  # plateau of equal maxima around vertex 0
    tied = np.flatnonzero(s == 100.0)
    assert len(tied) > 10
    o = find_origin(s, V, T)
    assert o in tied
    # the plateau centre is the tied vertex closest to the plateau centroid, not the first index
    centroid = V[tied].mean(0)
    assert o == int(tied[np.argmin(np.linalg.norm(V[tied] - centroid, axis=1))])
    assert find_origin_first(s) == int(tied[0])


def test_plateau_largest_region_wins(sphere):
    V, T = sphere
    d0, d1 = great_circle(V, 0), great_circle(V, int(np.argmin(np.abs(great_circle(V, 0) - 50.0))))
    s = np.full(len(V), 60.0)
    s[d0 < 6.0] = 100.0  # big plateau
    s[d1 < 2.5] = 100.0  # small plateau far away
    o = find_origin(s, V, T)
    assert d0[o] < 6.0


def test_focality_fallback_origin_below_half_maximum(sphere):
    V, T = sphere
    d = great_circle(V, 0)
    s = 65.0 + 35.0 * np.exp(-(d**2) / (2 * 8.0**2))
    far = int(np.argmax(d))  # an origin whose score lies below B + A/2
    assert s[far] < 65.0 + 17.5
    F, n, status = focality_index(s, T, origin=far, A=35.0, B=65.0)
    assert status == "ok" and n == 1 and F == 1.0  # component of the highest vertex in H is used
    F0, n0, st0 = focality_index(s, T, origin=0, A=35.0, B=65.0)
    assert (F0, n0, st0) == (F, n, status)


def test_focality_unavailable_reasons(sphere):
    V, T = sphere
    s = np.full(len(V), 70.0)
    assert focality_index(s, T, 0, np.nan, 65.0)[2].startswith("F unavailable: fit failed")
    assert focality_index(s, T, 0, -1.0, 65.0)[2].startswith("F unavailable: non-positive")
    s2 = np.full(len(V), 60.0)
    s2[0] = 100.0
    assert "fewer than" in focality_index(s2, T, 0, 35.0, 65.0)[2]


def test_daic_sign_and_keys(sphere):
    V, T = sphere
    rng = np.random.default_rng(3)
    d = great_circle(V, 0)
    gauss = synth_field(d, "gaussian", GAUSS_FWHM_FACTOR * 8.0, 0.03, rng)
    expo = synth_field(d, "exponential", 2.0 * 8.0, 0.03, rng)
    rg, re = analyse_map(gauss, V, T), analyse_map(expo, V, T)
    for r in (rg, re):
        assert np.isclose(r["dAIC"], r["aic_gauss"] - r["aic_exp"])
        assert set(r) >= {
            "origin",
            "sigma",
            "A",
            "B",
            "r2",
            "n_bins",
            "aic_gauss",
            "aic_exp",
            "dAIC",
            "lam_exp",
            "F",
            "n_components",
            "width_class",
            "flag_r2",
            "flag_F",
            "status",
        }
    assert rg["dAIC"] < 0  # Gaussian truth: Gaussian fits better (lower AIC)
    assert re["dAIC"] > 0  # exponential truth: exponential fits better


def test_flags_follow_thresholds(sphere):
    V, T = sphere
    rng = np.random.default_rng(4)
    r = analyse_map(synth_field(great_circle(V, 0), "gaussian", GAUSS_FWHM_FACTOR * 10.0, 0.05, rng), V, T)
    assert r["flag_r2"] == (r["r2"] < R2_THRESHOLD)
    assert r["flag_F"] == (np.isfinite(r["F"]) and r["F"] < F_THRESHOLD and not r["flag_r2"])


def test_too_few_bins_status(sphere):
    V, T = sphere
    s = np.full(len(V), np.nan)
    d = great_circle(V, 0)
    s[d < 5.0] = 90.0  # scores only within 5 mm -> fewer than 4 bins
    r = analyse_map(s, V, T)
    assert "fewer than 4 bins" in r["status"]
    assert r["width_class"] == "n.i." and r["flag_r2"]
    assert not np.isfinite(r["sigma"])


def test_custom_solver_and_fixed_origin(sphere):
    V, T = sphere
    rng = np.random.default_rng(5)
    s = synth_field(great_circle(V, 0), "gaussian", GAUSS_FWHM_FACTOR * 6.0, 0.05, rng)
    r_heat = analyse_map(s, V, T, dist_solver=HeatSolver(V, T), origin=0)
    r_true = analyse_map(s, V, T, dist_solver=lambda i: great_circle(V, i), origin=0)
    assert r_heat["origin"] == 0 and r_true["origin"] == 0
    assert abs(r_heat["sigma"] - r_true["sigma"]) < 0.3


def test_exact_solver_matches_great_circle(sphere):
    pytest.importorskip("pygeodesic")
    from pulse_ep_decay import ExactSolver

    V, T = sphere
    d = ExactSolver(V, T)(0)
    t = great_circle(V, 0)
    sel = (t > 2) & (t < 0.9 * np.pi * R)
    assert np.median(np.abs(d[sel] - t[sel]) / t[sel]) < 0.01
