#!/usr/bin/env python3
"""v36 — Independent validation of the geodesic computation on surfaces with
analytic ground truth (reviewer 3, point 1 of CMPB-D-26-02177).

Two parts:

A. Distance accuracy. On a sphere (great-circle distance) and an open
   cylinder (unrolled-plane distance) the Heat Method (potpourri3d),
   Dijkstra on the edge graph (scipy) and the exact polyhedral solver
   (pygeodesic, MMP) are compared with the analytic distance from one
   source vertex. Reported: mean signed relative error, 95th percentile of
   the absolute relative error, mean absolute error in mm.

B. Width recovery. A Gaussian field S = A exp(-d_analytic^2 / 2 sigma0^2) + B
   with vertex noise is generated with the ANALYTIC distance (not with any
   solver), and the simple decay pipeline of the submitted manuscript
   (argmax origin, 2-mm radial bins, three-parameter Gaussian fit) is run
   with each solver's distances. Reported: median error sigma_hat - sigma0
   per solver, so that the solver-induced part of the width error is
   isolated from the generative model.

This script is self-contained (it does not import pulse_ep_decay) and is a
TEST in the sense of the project rules: results go to a dated results
directory and a TEST note; nothing is written into the manuscript.

Usage: python3 v36_analytic_geodesic.py --out <results_dir>
"""
import argparse, json, os, time
import numpy as np
import trimesh
import potpourri3d as pp3d
import pygeodesic.geodesic as pgeo
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import curve_fit

RNG = np.random.default_rng(20261021)


# ---------------------------------------------------------------- meshes
def sphere(radius_mm, subdiv):
    m = trimesh.creation.icosphere(subdivisions=subdiv, radius=radius_mm)
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


def cylinder(radius_mm, height_mm, n_theta, n_z, jitter=0.0):
    """Open cylinder (no caps), regular grid, optional in-surface jitter so
    that the triangulation is not perfectly aligned with the geodesics."""
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z = np.linspace(0, height_mm, n_z)
    TH, Z = np.meshgrid(th, z, indexing="ij")
    if jitter > 0:
        TH = TH + RNG.uniform(-jitter, jitter, TH.shape) * (2 * np.pi / n_theta)
        Z[:, 1:-1] = Z[:, 1:-1] + RNG.uniform(-jitter, jitter, Z[:, 1:-1].shape) * (height_mm / (n_z - 1))
    V = np.stack([radius_mm * np.cos(TH), radius_mm * np.sin(TH), Z], -1).reshape(-1, 3)
    idx = np.arange(n_theta * n_z).reshape(n_theta, n_z)
    F = []
    for i in range(n_theta):
        i2 = (i + 1) % n_theta
        for j in range(n_z - 1):
            a, b, c, d = idx[i, j], idx[i2, j], idx[i2, j + 1], idx[i, j + 1]
            F.append([a, b, c]); F.append([a, c, d])
    return V, np.asarray(F, np.int64), (TH.reshape(-1), Z.reshape(-1))


def analytic_sphere(V, src, R):
    """Great-circle distance R*theta with theta = atan2(|u x u_src|, u . u_src): well conditioned
    at theta = 0 and theta = pi, where arccos(u . u_src) loses ~8 digits (the dot product rounds
    to 1 - eps or -1 + eps; 6/7 Nov 2026).  The source is at distance 0 by definition."""
    u = V / np.linalg.norm(V, axis=1, keepdims=True)
    d = R * np.arctan2(np.linalg.norm(np.cross(u, u[src]), axis=1), u @ u[src])
    d[src] = 0.0
    return d


def analytic_cylinder(th, z, src, R):
    dth = np.abs(np.angle(np.exp(1j * (th - th[src]))))
    return np.sqrt((R * dth) ** 2 + (z - z[src]) ** 2)


# ---------------------------------------------------------------- solvers
def heat(V, F, src):
    t0 = time.perf_counter()
    solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
    t1 = time.perf_counter()
    d = solver.compute_distance(src)
    return d, t1 - t0, time.perf_counter() - t1


def dijk(V, F, src):
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)  # each edge once (coo would sum duplicates)
    w = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    n = len(V)
    G = coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    t0 = time.perf_counter()
    d = dijkstra(G, directed=False, indices=src)
    return d, time.perf_counter() - t0


def exact(V, F, src):
    t0 = time.perf_counter()
    alg = pgeo.PyGeodesicAlgorithmExact(V, F)
    d, _ = alg.geodesicDistances(np.array([src]))
    return d, time.perf_counter() - t0


def errs(d, truth, src):
    m = np.ones(len(truth), dtype=bool); m[src] = False  # the source has no relative error; masked by index, not by a threshold on
    # the truth (6 Nov 2026: a rounded analytic self-distance of ~4e-7 mm on 5 of the 15 sphere sources passed the former
    # `truth > 1e-9` mask and entered the signed mean as -100 %)
    assert truth[m].min() > 0
    rel = (d[m] - truth[m]) / truth[m]
    return dict(rel_err_mean=float(rel.mean()), rel_err_median=float(np.median(rel)),
                rel_err_p95=float(np.percentile(np.abs(rel), 95)),
                abs_err_mean_mm=float(np.abs(d[m] - truth[m]).mean()),
                abs_err_max_mm=float(np.abs(d[m] - truth[m]).max()))


# ---------------------------------------------------------------- pipeline
def gauss(d, A, s, B):
    return A * np.exp(-d ** 2 / (2 * s ** 2)) + B


def decay_fit(d, s, bw=2.0, min_n=3):
    k = np.floor(d / bw).astype(int)
    xs, ys = [], []
    for kk in np.unique(k):
        m = k == kk
        if m.sum() >= min_n:
            xs.append((kk + 0.5) * bw); ys.append(s[m].mean())
    xs, ys = np.asarray(xs), np.asarray(ys)
    p0 = [ys.max() - ys.min(), 10.0, ys.min()]
    try:
        p, _ = curve_fit(gauss, xs, ys, p0=p0, bounds=([0, 0.1, -np.inf], [np.inf, 200, np.inf]), maxfev=20000)
    except Exception:
        return np.nan, np.nan
    r2 = 1 - ((ys - gauss(xs, *p)) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    return float(p[1]), float(r2)


def width_study(V, F, truth_fn, src_candidates, sigmas, noises, trials, solvers):
    out = []
    for src in src_candidates:
        truth = truth_fn(src)
        dsol = {name: fn(V, F, src)[0] for name, fn in solvers.items()}
        dsol["analytic"] = truth
        for s0 in sigmas:
            for eta in noises:
                for t in range(trials):
                    A, B = 35.0, 65.0
                    S = gauss(truth, A, s0, B) + RNG.normal(0, eta * A, len(V))
                    S = np.clip(S, 50, 100)
                    origin = int(np.argmax(S))  # argmax rule of the submitted manuscript
                    for name in dsol:
                        # distances from the DETECTED origin, as in the pipeline
                        if name == "analytic":
                            d = truth_fn(origin)
                        else:
                            d = solvers[name](V, F, origin)[0]
                        sh, r2 = decay_fit(d, S)
                        out.append(dict(src=src, sigma0=s0, noise=eta, trial=t, solver=name,
                                        sigma_hat=sh, r2=r2, origin_offset_mm=float(truth[origin])))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    solvers = {"heat": heat, "dijkstra": dijk, "exact": exact}
    dist_rows, width_rows = [], []

    # --- surfaces: radius 25 mm sphere (area ~ 79 cm2, atrial order of magnitude)
    surfaces = []
    for sub in ([3, 4] if a.quick else [3, 4, 5]):
        V, F = sphere(25.0, sub)
        el = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1).mean()
        surfaces.append(("sphere_sub%d" % sub, V, F, lambda s, V=V: analytic_sphere(V, s, 25.0), el))
    for (nt, nz, jit) in ([(96, 40, 0.0), (96, 40, 0.3)] if a.quick else [(64, 30, 0.0), (128, 50, 0.0), (128, 50, 0.3)]):
        V, F, (th, z) = cylinder(20.0, 80.0, nt, nz, jit)
        el = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1).mean()
        surfaces.append(("cylinder_%dx%d_j%.1f" % (nt, nz, jit), V, F, lambda s, th=th, z=z: analytic_cylinder(th, z, s, 20.0), el))

    for name, V, F, tf, el in surfaces:
        n = len(V)
        srcs = RNG.choice(n, size=(3 if a.quick else 5), replace=False)
        for src in srcs:
            truth = tf(src)
            dh, t_fac, t_q = heat(V, F, src); dd, t_d = dijk(V, F, src); de, t_e = exact(V, F, src)
            for sname, d, tt in (("heat", dh, t_q), ("dijkstra", dd, t_d), ("exact", de, t_e)):
                r = errs(d, truth, src); r.update(surface=name, n_vertices=n, mean_edge_mm=float(el), src=int(src), solver=sname, time_s=tt)
                if sname == "heat": r["factor_time_s"] = t_fac
                dist_rows.append(r)
        # width study on a subset of sources (cost: exact solver per detected origin)
        width_rows += [dict(r, surface=name, n_vertices=n, mean_edge_mm=float(el)) for r in
                       width_study(V, F, tf, [int(s) for s in srcs[:2]], [3.0, 5.0, 10.0, 15.0], [0.0, 0.10],
                                   1 if a.quick else 3, solvers)]
        print("done", name, n, "vertices, mean edge %.2f mm" % el, flush=True)

    import csv
    with open(os.path.join(a.out, "v36_distance_errors.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in dist_rows for k in r})); w.writeheader(); w.writerows(dist_rows)
    with open(os.path.join(a.out, "v36_width_by_solver.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in width_rows for k in r})); w.writeheader(); w.writerows(width_rows)

    # summary
    lines = ["v36 analytic geodesic validation — summary", "=" * 44, ""]
    lines.append("A. Distance error vs analytic truth (per surface, solver; mean over sources)")
    for name, *_ in surfaces:
        for sname in ("heat", "dijkstra", "exact"):
            rr = [r for r in dist_rows if r["surface"] == name and r["solver"] == sname]
            lines.append("  %-22s %-8s n=%6d edge=%.2f mm  signed rel err mean %+.4f  p95 |rel| %.4f  mean |err| %.2f mm  max |err| %.2f mm  t=%.3fs" % (
                name, sname, rr[0]["n_vertices"], rr[0]["mean_edge_mm"], np.mean([r["rel_err_mean"] for r in rr]),
                np.mean([r["rel_err_p95"] for r in rr]), np.mean([r["abs_err_mean_mm"] for r in rr]), np.max([r["abs_err_max_mm"] for r in rr]),
                np.mean([r["time_s"] for r in rr])))
    lines += ["", "B. Width error sigma_hat - sigma0 (median over sources/trials) by solver, field generated with the ANALYTIC distance"]
    for name, *_ in surfaces:
        for s0 in (3.0, 5.0, 10.0, 15.0):
            for eta in (0.0, 0.10):
                parts = []
                for sname in ("analytic", "exact", "heat", "dijkstra"):
                    rr = [r["sigma_hat"] - s0 for r in width_rows if r["surface"] == name and r["sigma0"] == s0 and r["noise"] == eta and r["solver"] == sname and np.isfinite(r["sigma_hat"])]
                    parts.append("%s %+.2f" % (sname, np.median(rr)) if rr else "%s n/a" % sname)
                lines.append("  %-22s sigma0=%4.1f noise=%.2f  " % (name, s0, eta) + "  ".join(parts))
    txt = "\n".join(lines); print(txt)
    open(os.path.join(a.out, "v36_summary.txt"), "w").write(txt + "\n")
    json.dump(dict(script="v36_analytic_geodesic.py", seed=20261021, note="self-contained; argmax origin, 2 mm bins, Gaussian fit as in the April 2026 submission"),
              open(os.path.join(a.out, "v36_manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
