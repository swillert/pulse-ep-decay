#!/usr/bin/env python3
"""v44 — What the isotropic sigma returns when the truth is anisotropic,
and what structured (spatially correlated) deviations do to R^2 and sigma
(reviewer 3, point 3; reviewer 1, limitation 2; reviewer 2, point 1 failure
modes).  Icosphere R = 25 mm (subdivision 4, 2,562 vertices, mean edge
1.9 mm), analytic geodesics, pipeline of the April submission (argmax
origin, 2-mm bins >= 3 vertices, three-parameter Gaussian).

A. Anisotropic truth: elliptical Gaussian in geodesic polar coordinates
   (r, phi) about v0: S = A exp(-r^2/2 (cos^2 phi / sa^2 + sin^2 phi / sb^2)) + B,
   with geometric mean sqrt(sa sb) in {5, 10} mm and ratio sa/sb in
   {1, 1.5, 2, 3, 4}; iid vertex noise 0 and 10 % of A.  Reported: sigma_hat
   vs sa, sb, sqrt(sa sb), (sa+sb)/2; R^2.
B. Structured deviations: isotropic Gaussian (sigma0 in {5, 10}) plus a
   smooth random field made of K = 30 Gaussian bumps (width 8 mm) with
   amplitude SD in {0, 0.1, 0.25, 0.5} x A, plus iid noise 10 %.  Reported:
   R^2 distribution, fraction R^2 < 0.7, sigma_hat bias (median, IQR) overall
   and in the R^2 < 0.7 subset, origin offset.
20 trials each.  Outputs: v44_rows.csv, v44_summary.txt.  TEST only.
"""
import argparse, os
import numpy as np, pandas as pd, trimesh
from scipy.optimize import curve_fit
from pulse_ep_decay.profile import find_origin  # plateau-centre tie rule (R1)

RNG = np.random.default_rng(20261021); R = 25.0; A, B = 35.0, 65.0


def gauss(d, A_, s, B_): return A_ * np.exp(-d ** 2 / (2 * s ** 2)) + B_


def polar(U, v0):
    """geodesic polar coordinates on the unit sphere about U[v0]: r (mm), phi."""
    n0 = U[v0]; c = np.clip(U @ n0, -1, 1); r = R * np.arccos(c)
    e1 = np.cross(n0, [0, 0, 1.0]);
    if np.linalg.norm(e1) < 1e-6: e1 = np.cross(n0, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(n0, e1)
    t = U - np.outer(c, n0)  # tangent components
    return r, np.arctan2(t @ e2, t @ e1)


def fit(d, S):
    ok = np.isfinite(S); d, S = d[ok], S[ok]; k = np.floor(d / 2.0).astype(int)
    xs, ys = [], []
    for kk in np.unique(k):
        m = k == kk
        if m.sum() >= 3: xs.append((kk + .5) * 2.0); ys.append(S[m].mean())
    xs, ys = np.asarray(xs), np.asarray(ys)
    try:
        p, _ = curve_fit(gauss, xs, ys, p0=[ys.max() - ys.min(), 10, ys.min()], bounds=([0, .1, -np.inf], [np.inf, 200, np.inf]), maxfev=20000)
    except Exception:
        return np.nan, np.nan
    return float(p[1]), float(1 - ((ys - gauss(xs, *p)) ** 2).sum() / ((ys - ys.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--trials", type=int, default=20); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    m = trimesh.creation.icosphere(subdivisions=4, radius=R); V, F = np.asarray(m.vertices), np.asarray(m.faces); U = V / R; n = len(V)
    rows = []
    for t in range(a.trials):
        v0 = int(RNG.integers(n)); r, phi = polar(U, v0)
        # A. anisotropy
        for gm in (5.0, 10.0):
            for ratio in (1.0, 1.5, 2.0, 3.0, 4.0):
                sa, sb = gm * np.sqrt(ratio), gm / np.sqrt(ratio)
                base = A * np.exp(-r ** 2 / 2 * (np.cos(phi) ** 2 / sa ** 2 + np.sin(phi) ** 2 / sb ** 2)) + B
                for eta in (0.0, 0.10):
                    S = np.clip(base + RNG.normal(0, eta * A, n), 50, 100); S[RNG.random(n) < .05] = np.nan
                    o = find_origin(S, V, F); d = R * np.arccos(np.clip(U @ U[o], -1, 1)); sh, r2 = fit(d, S)
                    rows.append(dict(part="A", trial=t, gm=gm, ratio=ratio, sa=sa, sb=sb, eta=eta, sigma_hat=sh, r2=r2, origin_offset=float(r[o])))
        # B. structured deviations
        for s0 in (5.0, 10.0):
            for amp in (0.0, 0.1, 0.25, 0.5):
                centres = U[RNG.integers(n, size=30)]; w = RNG.normal(0, amp * A, 30)
                dd = R * np.arccos(np.clip(U @ centres.T, -1, 1)); bumps = (np.exp(-dd ** 2 / (2 * 8.0 ** 2)) * w).sum(1)
                base = A * np.exp(-r ** 2 / (2 * s0 ** 2)) + B + bumps
                S = np.clip(base + RNG.normal(0, 0.10 * A, n), 50, 100); S[RNG.random(n) < .05] = np.nan
                o = find_origin(S, V, F); d = R * np.arccos(np.clip(U @ U[o], -1, 1)); sh, r2 = fit(d, S)
                rows.append(dict(part="B", trial=t, sigma0=s0, amp=amp, eta=0.10, sigma_hat=sh, r2=r2, origin_offset=float(r[o])))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "v44_rows.csv"), index=False)
    L = ["v44 anisotropic truth and structured deviations (icosphere R = 25 mm, analytic geodesics, %d trials) — summary" % a.trials, "=" * 100, "",
         "A. Isotropic fit to an elliptical Gaussian truth: sigma_hat (median over trials) vs sa (major), sb (minor), sqrt(sa sb), (sa+sb)/2"]
    for gm in (5.0, 10.0):
        for ratio in (1.0, 1.5, 2.0, 3.0, 4.0):
            for eta in (0.0, 0.10):
                g = df[(df.part == "A") & (df.gm == gm) & (df.ratio == ratio) & (df.eta == eta)]
                sa, sb = g.sa.iloc[0], g.sb.iloc[0]
                L.append("  gm=%4.1f ratio=%.1f (sa %5.2f, sb %5.2f, mean %5.2f) noise %.2f: sigma_hat median %5.2f [%5.2f; %5.2f]  R2 median %.3f min %.3f  offset median %.1f mm" % (
                    gm, ratio, sa, sb, (sa + sb) / 2, eta, g.sigma_hat.median(), g.sigma_hat.quantile(.25), g.sigma_hat.quantile(.75), g.r2.median(), g.r2.min(), g.origin_offset.median()))
    L += ["", "B. Isotropic Gaussian + smooth random field (30 bumps of 8 mm) + 10 % iid noise: R2 and sigma_hat"]
    for s0 in (5.0, 10.0):
        for amp in (0.0, 0.1, 0.25, 0.5):
            g = df[(df.part == "B") & (df.sigma0 == s0) & (df.amp == amp)]; lo = g[g.r2 < .7]
            L.append("  sigma0=%4.1f bump amp %.2f A: R2 median %.3f [%.3f; %.3f]  frac R2<0.7 %.2f  bias median %+.2f [%+.2f; %+.2f]  bias in R2<0.7 subset %s  offset median %.1f mm (p90 %.1f)" % (
                s0, amp, g.r2.median(), g.r2.quantile(.25), g.r2.quantile(.75), (g.r2 < .7).mean(), (g.sigma_hat - s0).median(), (g.sigma_hat - s0).quantile(.25), (g.sigma_hat - s0).quantile(.75),
                ("%+.2f (n=%d)" % ((lo.sigma_hat - s0).median(), len(lo))) if len(lo) else "n/a", g.origin_offset.median(), g.origin_offset.quantile(.9)))
    txt = "\n".join(L); print(txt); open(os.path.join(a.out, "v44_summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
