#!/usr/bin/env python3
"""v53 (26 Oct 2026) = v37 with the shared package fitter (amplitude unconstrained) instead of the local fit with A >= 0.\nv37 — Synthetic test of the two classification thresholds of the submitted
manuscript (R^2 >= 0.7 "focal", F >= 0.70 "unifocal") on a sphere with
analytic geodesics (reviewer 2, points 4 and 6; reviewer 3, point 4).

Vertex-field design (as in the April 2026 submission, NOT the site-sampled
design of v7): fields on an icosphere (R = 25 mm, subdivision 4, 2,562
vertices, mean edge 1.9 mm):

  focal      : Gaussian, sigma0 in {5, 10, 15} mm, A = 35, B = 65
  flat       : constant B (+ noise)                      -> should be R^2 < 0.7
  two-peak   : Gaussian at v0 plus a second Gaussian at geodesic distance
               D in {15, 25, 40} mm with amplitude ratio a2/a1 in
               {0.5, 0.8, 1.0}, same sigma0 = 7 mm         -> F should fall
  ring       : raised ring (annulus) around v0            -> non-focal shape

Noise: iid vertex noise eta in {5, 10, 15, 25} % of A; 5 % of vertices masked.
Per field: argmax origin, exact analytic distances from that origin, 2 mm
bins, Gaussian fit -> R^2; focality F = |largest connected component| /
|vertices >= 95th percentile| on the mesh graph.

Outputs: v37_fields.csv (one row per field), v37_summary.txt with
misclassification rates of R^2 at thresholds 0.5 ... 0.9 and the response of
F to the two-peak configurations.

TEST only; nothing is written into the manuscript.
"""
import argparse, os, csv
import numpy as np, trimesh
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

RNG = np.random.default_rng(20261021)
R = 25.0


def gauss(d, A, s, B):
    return A * np.exp(-d ** 2 / (2 * s ** 2)) + B


def sph_dist(U, src):
    return R * np.arccos(np.clip(U @ U[src], -1, 1))


def decay_fit(d, s, mask, bw=2.0):
    # v53: the shared package fit (pulse_ep_decay.profile: bin_profile + fit_gaussian, amplitude unconstrained)
    from pulse_ep_decay.profile import bin_profile, fit_gaussian
    dd = np.where(mask, d, np.nan); ss = np.where(mask, s, np.nan)
    c, mm, n = bin_profile(dd, ss); g = fit_gaussian(c, mm)
    return float(g["sigma"]), float(g["r2"]), float(g["A"])


def focality(S, mask, G):
    idx = np.where(mask)[0]
    thr = np.percentile(S[idx], 95)
    H = idx[S[idx] >= thr]
    sub = G[H][:, H]
    _, lab = connected_components(sub, directed=False)
    return float(np.bincount(lab).max() / len(H)), len(H)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--trials", type=int, default=10)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    m = trimesh.creation.icosphere(subdivisions=4, radius=R)
    V, F = np.asarray(m.vertices), np.asarray(m.faces); U = V / R; n = len(V)
    e = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    G = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    A, B = 35.0, 65.0
    rows = []
    for t in range(a.trials):
        v0 = int(RNG.integers(n)); d0 = sph_dist(U, v0)
        fields = []
        for s0 in (5, 10, 15):
            fields.append(("focal", dict(sigma0=s0), gauss(d0, A, s0, B)))
        fields.append(("flat", {}, np.full(n, B)))
        fields.append(("ring", {}, B + A * np.exp(-(d0 - 15) ** 2 / (2 * 4.0 ** 2))))
        for D in (15, 25, 40):
            # second origin at geodesic distance D from v0
            cand = np.where(np.abs(d0 - D) < 1.0)[0]; v1 = int(RNG.choice(cand)); d1 = sph_dist(U, v1)
            for ratio in (0.5, 0.8, 1.0):
                fields.append(("two-peak", dict(D=D, ratio=ratio), B + A * np.maximum(np.exp(-d0 ** 2 / 98), ratio * np.exp(-d1 ** 2 / 98))))
        for kind, par, base in fields:
            for eta in (0.05, 0.10, 0.15, 0.25):
                S = np.clip(base + RNG.normal(0, eta * A, n), 50, 100)
                mask = RNG.random(n) >= 0.05
                origin = int(np.argmax(np.where(mask, S, -np.inf)))
                sh, r2, Ahat = decay_fit(sph_dist(U, origin), S, mask)
                Fi, nH = focality(S, mask, G)
                rows.append(dict(trial=t, kind=kind, noise=eta, sigma_hat=sh, r2=r2, A_hat=Ahat, F=Fi, nH=nH, origin_offset_mm=float(d0[origin]), **{k: par.get(k, "") for k in ("sigma0", "D", "ratio")}))
    with open(os.path.join(a.out, "v53_fields.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    L = ["v37 threshold / focality synthetic test (icosphere R=25 mm, %d vertices, %d trials)" % (n, a.trials), ""]
    L.append("A. R^2 by field kind and noise (median, 5th pct) and misclassification against threshold T")
    for kind in ("focal", "flat", "ring", "two-peak"):
        for eta in (0.05, 0.10, 0.15, 0.25):
            rr = np.array([r["r2"] for r in rows if r["kind"] == kind and r["noise"] == eta and np.isfinite(r["r2"])])
            mis = " ".join("T=%.1f:%3.0f%%" % (T, 100 * (np.mean(rr < T) if kind == "focal" else np.mean(rr >= T))) for T in (0.5, 0.6, 0.7, 0.8, 0.9))
            L.append("  %-9s noise %.2f  R2 median %.3f p5 %.3f  n=%d  misclass %s" % (kind, eta, np.median(rr), np.percentile(rr, 5), len(rr), mis))
    L += ["", "  (focal: misclassified = R2 < T; others: misclassified = R2 >= T, i.e. accepted as focal)", "",
          "B. Focality index F for two-peak fields (median, p90 over trials) and fraction F < 0.70"]
    for D in (15, 25, 40):
        for ratio in (0.5, 0.8, 1.0):
            for eta in (0.05, 0.10, 0.25):
                rr = np.array([r["F"] for r in rows if r["kind"] == "two-peak" and r["D"] == D and r["ratio"] == ratio and r["noise"] == eta])
                r2 = np.array([r["r2"] for r in rows if r["kind"] == "two-peak" and r["D"] == D and r["ratio"] == ratio and r["noise"] == eta])
                L.append("  D=%2d mm ratio %.1f noise %.2f  F median %.2f p10 %.2f  F<0.70 in %3.0f%%   R2 median %.2f" % (D, ratio, eta, np.median(rr), np.percentile(rr, 10), 100 * np.mean(rr < .7), np.median(r2)))
    L += ["", "C. F for single-peak focal and flat fields (median, p10)"]
    for kind in ("focal", "flat", "ring"):
        for eta in (0.05, 0.10, 0.25):
            rr = np.array([r["F"] for r in rows if r["kind"] == kind and r["noise"] == eta])
            L.append("  %-6s noise %.2f  F median %.2f p10 %.2f  F<0.70 in %3.0f%%" % (kind, eta, np.median(rr), np.percentile(rr, 10), 100 * np.mean(rr < .7)))
    txt = "\n".join(L); print(txt); open(os.path.join(a.out, "v53_summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
