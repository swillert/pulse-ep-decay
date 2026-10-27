"""Tests on synthetic surfaces (no clinical data needed).

An icosphere of radius 25 mm carries Gaussian score fields with known width;
the pipeline must recover the width, assign the class and flags, and the
focality index must separate one peak from two well-separated peaks.
"""

import numpy as np
import pytest

from pulse_ep_decay import HeatSolver, analyse_map, largest_component, mean_edge_length, width_class
from pulse_ep_decay.profile import GAUSS_FWHM_FACTOR, synth_field

R = 25.0


def icosphere(subdiv: int = 4, radius: float = R):
    t = (1 + 5**0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(subdiv):
        cache, nf = {}, []

        def mid(a, b, cache=cache):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(verts) * radius, np.array(f, int)


def great_circle(V, i):
    u = V / np.linalg.norm(V, axis=1, keepdims=True)
    return R * np.arccos(np.clip(u @ u[i], -1, 1))


@pytest.fixture(scope="module")
def sphere():
    V, T = icosphere()
    V, T, _ = largest_component(V, T)
    return V, T


def test_mesh(sphere):
    V, T = sphere
    assert len(V) == 2562
    assert 1.6 < mean_edge_length(V, T) < 2.2


def test_heat_method_close_to_great_circle(sphere):
    V, T = sphere
    d_heat = HeatSolver(V, T)(0)
    d_true = great_circle(V, 0)
    sel = (d_true > 2) & (d_true < 30)
    rel = np.median((d_heat[sel] - d_true[sel]) / d_true[sel])
    assert abs(rel) < 0.03


@pytest.mark.parametrize("sigma0", [5.0, 10.0])
def test_width_recovered(sphere, sigma0):
    V, T = sphere
    rng = np.random.default_rng(1)
    s = synth_field(great_circle(V, 0), "gaussian", GAUSS_FWHM_FACTOR * sigma0, 0.05, rng)
    res = analyse_map(s, V, T)
    assert abs(res["sigma"] - sigma0) < 0.1 * sigma0
    assert res["width_class"] == "in range"
    assert not res["flag_r2"]


def test_width_class_edges():
    assert width_class(1.5) == "below range"
    assert width_class(10.0) == "in range"
    assert width_class(40.0) == "broad"
    assert width_class(60.0) == "diffuse"


def test_focality_one_vs_two_peaks(sphere):
    V, T = sphere
    rng = np.random.default_rng(2)
    d0 = great_circle(V, 0)
    far = int(np.argmin(np.abs(d0 - 40.0)))  # second source 40 mm away
    one = synth_field(d0, "gaussian", GAUSS_FWHM_FACTOR * 7.0, 0.05, rng)
    a = 35.0 * np.exp(-(d0**2) / (2 * 7.0**2))
    b = 35.0 * np.exp(-(great_circle(V, far) ** 2) / (2 * 7.0**2))
    two = np.clip(65.0 + np.maximum(a, b) + rng.normal(0, 1.75, len(V)), 50, 100)
    r1, r2 = analyse_map(one, V, T), analyse_map(two, V, T)
    assert r1["F"] > 0.9 and not r1["flag_F"]
    assert r2["F"] < 0.7
