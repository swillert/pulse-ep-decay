"""v56 — TEST (26 Oct 2026): synthetic validation with the peak below the clip.
In the manuscript protocol (A = 35, B = 65, clip [50, 100]) the noise-free peak equals the upper clip, so noisy fields
saturate in a plateau centred on the source, which the R1 plateau-centre tie rule exploits.  Clinical maxima mostly lie
below 100.  Here B = 55 (peak 90), otherwise identical (exact-geodesic truth from the centroid-nearest vertex, iid noise,
clip [50, 100], 5 % masked), every 4th of the 56 meshes (TEST run, sigma0 3-40 mm) or all 56 (--all, R1 supplement, sigma0 2-50 mm), eta 5/10/15 %, 20 trials.  Per field: origin by
the plateau-centre rule (package find_origin) and by the first tied vertex; offset and sigma for each.
Nothing in the manuscript is changed by this script."""
import argparse, glob, os
import numpy as np, pandas as pd
from pulse_ep_decay.mesh import HeatSolver, ExactSolver, largest_component, centroid_vertex
from pulse_ep_decay.profile import bin_profile, fit_gaussian, find_origin
ROOT = os.path.expanduser("~/mnt/IdeaProjects"); MESHES = os.path.join(ROOT, "pulse-ep/data/validation_meshes")
SVR = os.path.join(ROOT, "thesis-med/_paper/method-paper/supplementary_validation_results"); OUT = os.path.join(SVR, "v56_peak_below_clip_2026-10-26")
FRAGMENTED = {87, 102, 103}; A, B = 35.0, 55.0; SIGMAS = (2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50); NOISES = (0.05, 0.10, 0.15); TRIALS = 20


def fit(d, S):
    c, m, _ = bin_profile(d, S); return fit_gaussian(c, m)["sigma"]


def run_mesh(f):
    z = np.load(f, allow_pickle=True); mid = int(z["map_id"]); pf = os.path.join(OUT, "parts", "%03d.csv" % mid)
    if os.path.exists(pf): return
    V, T, _ = largest_component(z["vertices"], z["triangles"].astype(int))
    v0 = centroid_vertex(V); d0 = ExactSolver(V, T)(v0); d0 = np.where(np.isfinite(d0), d0, np.nan); heat = HeatSolver(V, T); rows = []
    for s0 in SIGMAS:
        for eta in NOISES:
            for t in range(TRIALS):
                rng = np.random.default_rng(2_000_000 * mid + int(1000 * s0) + int(100 * eta * 100) + t)
                S = A * np.exp(-np.nan_to_num(d0) ** 2 / (2 * s0 ** 2)) + B + rng.normal(0, eta * A, len(V))
                S = np.clip(S, 50, 100); S[np.isnan(d0)] = np.nan; S[rng.random(len(S)) < 0.05] = np.nan
                n_tied = int((S == np.nanmax(S)).sum()); op = find_origin(S, V, T); of = int(np.nanargmax(S))
                sp = fit(heat(op), S); sf = sp if of == op else fit(heat(of), S)
                rows.append(dict(map_id=mid, sigma0=s0, noise=eta, trial=t, n_tied=n_tied, off_plateau=float(d0[op]), off_first=float(d0[of]), sigma_plateau=sp, sigma_first=sf))
    pd.DataFrame(rows).to_csv(pf, index=False); print("mesh %d done" % mid, flush=True)


def summary():
    d = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(OUT, "parts", "*.csv")))]); d.to_csv(os.path.join(OUT, "v56_raw.csv"), index=False)
    L = ["v56 peak below the clip (A 35, B 55; %d meshes, %d fields); plateau-centre rule (package) / first tied vertex" % (d.map_id.nunique(), len(d))]
    for (s0, eta), g in d.groupby(["sigma0", "noise"]):
        L.append("sigma0 %2d eta %.2f: tied max in %3.0f %%; offset median %.1f/%.1f mm; bias mean %+.2f/%+.2f, median %+.2f; SD %.2f/%.2f; >100 mm %.1f %%" % (
            s0, eta, 100 * (g.n_tied > 1).mean(), g.off_plateau.median(), g.off_first.median(), (g.sigma_plateau - s0).mean(), (g.sigma_first - s0).mean(),
            (g.sigma_plateau - s0).median(), g.sigma_plateau.std(), g.sigma_first.std(), 100 * (g.sigma_plateau >= 100).mean()))
    open(os.path.join(OUT, "v56_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--summary", action="store_true"); ap.add_argument("--all", action="store_true", help="all 56 meshes (default: every 4th)"); a = ap.parse_args()
    if a.all: OUT = os.path.join(SVR, "v56_peak_below_clip_all_2026-10-26")
    os.makedirs(os.path.join(OUT, "parts"), exist_ok=True)
    if not a.summary:
        fs = [f for f in sorted(glob.glob(os.path.join(MESHES, "map_*.npz"))) if int(np.load(f, allow_pickle=True)["map_id"]) not in FRAGMENTED][::(1 if a.all else 4)]
        for f in fs: run_mesh(f)
    summary()
