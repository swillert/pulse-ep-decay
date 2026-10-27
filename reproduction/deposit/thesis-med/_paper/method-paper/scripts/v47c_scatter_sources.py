#!/usr/bin/env python3
"""v47c — Which deviations produce the clinical within-bin scatter (37 % of A,
v43) without destroying the radial fit?  Icosphere R = 25 mm (subdivision 4,
2,562 vertices), analytic geodesics, pipeline as submitted.  Conditions:
  anisotropic truth (axis ratio 1, 1.5, 2, 3, 4; geometric mean 5 and 10 mm)
  with 10 % iid noise; and axis ratio 1 or 3 combined with spatially
  correlated noise (heat-kernel-smoothed white noise, l = 10 mm) of SD
  0.10 / 0.25 A.  Per field: within-bin SD in % of A (first 30 mm), R2,
  sigma_hat / geometric mean, origin offset.  20 trials.  TEST only.
"""
import argparse, os
import numpy as np, pandas as pd, trimesh
from scipy.optimize import curve_fit
from pulse_ep_decay.profile import find_origin  # plateau-centre tie rule (R1)
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu

RNG = np.random.default_rng(20261025); R = 25.0; A, B = 35.0, 65.0


def gauss(d, A_, s, B_): return A_ * np.exp(-d ** 2 / (2 * s ** 2)) + B_


def polar(U, v0):
    n0 = U[v0]; c = np.clip(U @ n0, -1, 1); r = R * np.arccos(c)
    e1 = np.cross(n0, [0, 0, 1.0]); e1 = np.cross(n0, [1.0, 0, 0]) if np.linalg.norm(e1) < 1e-6 else e1; e1 /= np.linalg.norm(e1); e2 = np.cross(n0, e1)
    t = U - np.outer(c, n0); return r, np.arctan2(t @ e2, t @ e1)


def fit(d, S):
    ok = np.isfinite(S); d, S = d[ok], S[ok]; k = np.floor(d / 2.0).astype(int); xs, ys = [], []
    for kk in np.unique(k):
        m = k == kk
        if m.sum() >= 3: xs.append((kk + .5) * 2.0); ys.append(S[m].mean())
    xs, ys = np.asarray(xs), np.asarray(ys)
    try:
        p, _ = curve_fit(gauss, xs, ys, p0=[ys.max() - ys.min(), 10, ys.min()], bounds=([0, .1, -np.inf], [np.inf, 200, np.inf]), maxfev=20000)
    except Exception:
        return np.nan, np.nan
    return float(p[1]), float(1 - ((ys - gauss(xs, *p)) ** 2).sum() / ((ys - ys.mean()) ** 2).sum())


def wsd(d, S):
    ok = np.isfinite(S) & (d <= 30); k = np.floor(d / 2.0).astype(int)
    v = [S[ok & (k == kk)].std() for kk in np.unique(k[ok]) if (ok & (k == kk)).sum() >= 3]; return 100 * float(np.mean(v)) / A


def cotan(V, F):
    n = len(V); I, J, W = [], [], []
    for k in range(3):
        i, j, l = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]; u, w = V[i] - V[l], V[j] - V[l]
        cot = (u * w).sum(1) / np.linalg.norm(np.cross(u, w), axis=1); I += [i, j]; J += [j, i]; W += [0.5 * cot, 0.5 * cot]
    Wm = coo_matrix((np.concatenate(W), (np.concatenate(I), np.concatenate(J))), shape=(n, n)).tocsr(); L = diags(np.asarray(Wm.sum(1)).ravel()) - Wm
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1); M = np.zeros(n); np.add.at(M, F.ravel(), np.repeat(area / 3, 3))
    return L.tocsc(), M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--trials", type=int, default=20); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    m = trimesh.creation.icosphere(subdivisions=4, radius=R); V, F = np.asarray(m.vertices), np.asarray(m.faces); U = V / R; n = len(V)
    L, M = cotan(V, F); lu = splu((diags(M) + 100.0 * L).tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    rows = []
    for t in range(a.trials):
        v0 = int(RNG.integers(n)); r, phi = polar(U, v0)
        for gm in (5.0, 10.0):
            conds = [(ratio, "iid", 0.10) for ratio in (1.0, 1.5, 2.0, 3.0, 4.0)] + [(ratio, "corr10", amp) for ratio in (1.0, 3.0) for amp in (0.10, 0.25)]
            for ratio, kind, amp in conds:
                sa, sb = gm * np.sqrt(ratio), gm / np.sqrt(ratio)
                base = A * np.exp(-r ** 2 / 2 * (np.cos(phi) ** 2 / sa ** 2 + np.sin(phi) ** 2 / sb ** 2)) + B
                dev = RNG.normal(0, amp * A, n) if kind == "iid" else (lambda u: u / u.std() * amp * A)(lu.solve(M * RNG.normal(0, 1, n)))
                S = np.clip(base + dev, 50, 100); S[RNG.random(n) < .05] = np.nan
                o = find_origin(S, V, F); d = R * np.arccos(np.clip(U @ U[o], -1, 1)); sh, r2 = fit(d, S)
                rows.append(dict(trial=t, gm=gm, ratio=ratio, kind=kind, amp=amp, wsd=wsd(d, S), sigma_hat=sh, r2=r2, offset=float(r[o])))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "v47c_rows.csv"), index=False)
    g = df.groupby(["gm", "ratio", "kind", "amp"])
    s = pd.DataFrame(dict(wsd=g.wsd.median(), r2=g.r2.median(), r2_min=g.r2.min(), rel=g.apply(lambda x: (x.sigma_hat / x.gm).median()), off=g.offset.median(), off_p90=g.offset.quantile(.9))).reset_index()
    s.to_csv(os.path.join(a.out, "v47c_summary.csv"), index=False)
    Lns = ["v47c sources of within-bin scatter (sphere, analytic geodesics, %d trials) — clinical reference: 37 %% of A [27; 52], R2 0.82" % a.trials, "=" * 90,
           "  gm   ratio  noise   amp | within-bin SD %A | R2 med  min | sigma_hat/gm | offset med p90"]
    for _, r in s.iterrows():
        Lns.append("  %4.1f  %4.1f  %-6s %5.2f | %6.0f           | %.3f %.3f | %6.2f       | %5.1f %5.1f" % (r.gm, r.ratio, r.kind, r.amp, r.wsd, r.r2, r.r2_min, r.rel, r.off, r.off_p90))
    txt = "\n".join(Lns); print(txt); open(os.path.join(a.out, "v47c_summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
