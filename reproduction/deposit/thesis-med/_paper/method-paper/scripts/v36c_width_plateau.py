#!/usr/bin/env python3
"""v36c (26 Oct 2026, review round 4, Q3) = v36b with the R1 origin rule (pulse_ep_decay.profile.find_origin, plateau
centre) instead of the first-index argmax, and the shared package fit (bin_profile + fit_gaussian, amplitude
unconstrained) instead of the local fit with A >= 0; everything else (surfaces, sources, seeds, fit) identical.
v36b — part B of v36 (width recovery on analytic surfaces by solver) with 5 sources x 10 trials.
Same surfaces, sources (from v36_distance_errors.csv), field model (A 35, B 65, clip [50,100]), argmax
origin and 2-mm/>=3 fit as v36; noise-free fields are deterministic and therefore run once per source.
Distances for the fit are taken from the DETECTED origin with each solver (analytic, heat, dijkstra, exact);
exact/heat/dijkstra solutions are cached per origin vertex. Seeds: (surface index, src, sigma0, trial).
Resumable per (surface, source) part. Usage: v36b_width_rerun.py run [budget_s] | assemble
"""
import sys, os, glob, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_analytic_geodesic as v36
from pulse_ep_decay.profile import find_origin, bin_profile, fit_gaussian
def pkg_fit(d, S):
    c, m, _ = bin_profile(np.asarray(d, float), np.asarray(S, float)); g = fit_gaussian(c, m); return float(g['sigma']), float(g['r2'])
MP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(MP, "supplementary_validation_results/v36c_width_plateau_2026-10-26"); PARTS = os.path.join(OUT, "parts")
DIST = pd.read_csv(os.path.join(MP, "supplementary_validation_results/v36_analytic_geodesic_2026-10-21/v36_distance_errors.csv"))
SIGMAS, NOISES, TRIALS = [3.0, 5.0, 10.0, 15.0], [0.0, 0.10], 10


def surfaces():
    out = []
    for sub in [3, 4, 5]:
        V, F = v36.sphere(25.0, sub); out.append(("sphere_sub%d" % sub, V, F, lambda s, V=V: v36.analytic_sphere(V, s, 25.0)))
    for (nt, nz, jit) in [(64, 30, 0.0), (128, 50, 0.0), (128, 50, 0.3)]:
        V, F, (th, z) = v36.cylinder(20.0, 80.0, nt, nz, jit); out.append(("cylinder_%dx%d_j%.1f" % (nt, nz, jit), V, F, lambda s, th=th, z=z: v36.analytic_cylinder(th, z, s, 20.0)))
    return out


def run(budget):
    os.makedirs(PARTS, exist_ok=True); t0 = time.time(); solvers = {"heat": v36.heat, "dijkstra": v36.dijk, "exact": v36.exact}
    for si, (name, V, F, tf) in enumerate(surfaces()):
        srcs = sorted(DIST[DIST.surface == name].src.unique().tolist())
        el = float(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1).mean())
        for src in srcs:
            pf = os.path.join(PARTS, "%s_%d.json" % (name, src))
            if os.path.exists(pf): continue
            if time.time() - t0 > budget: print("budget reached before", name, src); return
            truth = tf(src); cache = {k: {} for k in solvers}; rows = []
            def dist(sname, origin):
                if origin not in cache[sname]: cache[sname][origin] = solvers[sname](V, F, origin)[0]
                return cache[sname][origin]
            for s0 in SIGMAS:
                for eta in NOISES:
                    for t in range(1 if eta == 0 else TRIALS):
                        rng = np.random.default_rng(1_000_000 * si + 1000 * src + int(10 * s0) * 10 + t)
                        A, B = 35.0, 65.0; S = np.clip(v36.gauss(truth, A, s0, B) + rng.normal(0, eta * A, len(V)), 50, 100)
                        origin = find_origin(S, V, F)
                        for sname in ("analytic", "heat", "dijkstra", "exact"):
                            d = tf(origin) if sname == "analytic" else dist(sname, origin)
                            sh, r2 = pkg_fit(d, S)
                            rows.append(dict(surface=name, n_vertices=len(V), mean_edge_mm=el, src=int(src), sigma0=s0, noise=eta, trial=t, solver=sname, sigma_hat=sh, r2=r2, origin_offset_mm=float(truth[origin])))
            json.dump(rows, open(pf, "w")); print(name, src, "%d rows, %.0f s, exact solves %d" % (len(rows), time.time() - t0, len(cache["exact"])), flush=True)
    print("all parts done")


def assemble():
    rows = [r for p in sorted(glob.glob(os.path.join(PARTS, "*.json"))) for r in json.load(open(p))]
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "v36c_width_raw.csv"), index=False)
    df["err"] = df.sigma_hat - df.sigma0
    s = df.groupby(["surface", "noise", "sigma0", "solver"]).agg(n=("err", "size"), bias_median=("err", "median"), bias_mean=("err", "mean"), sd=("sigma_hat", "std"), r2=("r2", "median")).reset_index()
    s.to_csv(os.path.join(OUT, "v36c_width_summary.csv"), index=False)
    # solver-vs-analytic difference per field
    piv = df.pivot_table(index=["surface", "src", "sigma0", "noise", "trial"], columns="solver", values="sigma_hat")
    for k in ("heat", "dijkstra", "exact"): piv["d_" + k] = piv[k] - piv["analytic"]
    dd = piv.reset_index().groupby(["surface", "noise"])[["d_heat", "d_dijkstra", "d_exact"]].agg(["median", lambda x: x.abs().max()])
    dd.to_csv(os.path.join(OUT, "v36c_solver_minus_analytic.csv")); print(dd.round(3).to_string())
    print(s[s.noise == 0.1].pivot_table(index=["surface", "sigma0"], columns="solver", values="bias_median").round(2).to_string())


if __name__ == "__main__":
    run(float(sys.argv[2]) if len(sys.argv) > 2 else 150) if sys.argv[1] == "run" else assemble()
