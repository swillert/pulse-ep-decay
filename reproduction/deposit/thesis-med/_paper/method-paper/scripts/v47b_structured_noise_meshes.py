#!/usr/bin/env python3
"""v47b — Structured and spatially correlated deviations on the 56 clinical
meshes (reviewer 1, limitation 2; reviewer 2, point 1 failure modes; extends
v44 from the sphere to the real geometries).

Truth: Gaussian field on exact polyhedral geodesics from the centroid-nearest
vertex v0 (sigma0 5, 10, 15 mm; A = 35, B = 65; clip [50,100]; 5 % masked).
Deviations:
  bumps   : 10 % iid vertex noise + 30 Gaussian bumps (8 mm width, Euclidean,
            random centres) with amplitude SD 0.10 / 0.25 / 0.50 A
  corr    : spatially correlated noise instead of iid noise: white noise
            smoothed with the heat kernel (M + l^2 L) u = M eps, l = 5 / 10 mm,
            scaled to SD 0.10 / 0.25 / 0.40 A
Pipeline as submitted (argmax origin, Heat Method from the detected origin,
2-mm bins >= 3 vertices, Gaussian fit); focality index F (95th percentile,
edge connectivity).  Per field: sigma_hat, R2, F, origin offset (exact
distance of the detected origin from v0), within-bin SD of the field in the
first 30 mm in % of A (the quantity that is 37 % of A in the clinical maps,
v43).  10 trials per (mesh, sigma0, condition).  Resumable per mesh.
Outputs: v47b_raw.csv, v47b_summary.csv, v47b_summary.txt.  TEST only.
"""
import argparse, glob, os
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import connected_components
from pulse_ep_decay.mesh import HeatSolver, ExactSolver, largest_component, centroid_vertex
from pulse_ep_decay.profile import bin_profile, fit_gaussian, find_origin

ROOT = os.environ.get("GD_ROOT", os.path.expanduser("~/mnt/IdeaProjects")); MESHES = os.path.join(ROOT, "pulse-ep/data/validation_meshes")
FRAGMENTED = {87, 102, 103}; A, B = 35.0, 65.0; SIGMAS = (5.0, 10.0, 15.0); TRIALS = 10
CONDS = [("bumps", 0.10), ("bumps", 0.25), ("bumps", 0.50), ("corr5", 0.10), ("corr5", 0.25), ("corr5", 0.40), ("corr10", 0.10), ("corr10", 0.25), ("corr10", 0.40)]


def cotan_laplacian(V, F):
    n = len(V); I, J, W = [], [], []
    for k in range(3):
        i, j, l = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        u, w = V[i] - V[l], V[j] - V[l]
        cot = (u * w).sum(1) / np.maximum(np.linalg.norm(np.cross(u, w), axis=1), 1e-12)
        I += [i, j]; J += [j, i]; W += [0.5 * cot, 0.5 * cot]
    Wm = coo_matrix((np.concatenate(W), (np.concatenate(I), np.concatenate(J))), shape=(n, n)).tocsr()
    L = diags(np.asarray(Wm.sum(1)).ravel()) - Wm
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    M = np.zeros(n); np.add.at(M, F.ravel(), np.repeat(area / 3, 3))
    return L.tocsc(), M


def edge_graph(F, n):
    e = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    return coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()


def focality(S, G):
    fin = np.isfinite(S); idx = np.where(fin)[0]; thr = np.percentile(S[idx], 95); H = idx[S[idx] >= thr]
    if len(H) < 3: return np.nan
    sub = G[H][:, H]; _, lab = connected_components(sub, directed=False); return float(np.bincount(lab).max() / len(H))


def within_bin_sd(d, S):
    ok = np.isfinite(S) & (d <= 30); k = np.floor(np.clip(d, 0, None) / 2.0).astype(int)
    sds = [S[ok & (k == kk)].std() for kk in np.unique(k[ok]) if (ok & (k == kk)).sum() >= 3]
    return 100 * float(np.mean(sds)) / A if sds else np.nan


def run_mesh(f, out):
    z = np.load(f, allow_pickle=True); mid = int(z["map_id"])
    if mid in FRAGMENTED or os.path.exists(os.path.join(out, "parts", "%03d.csv" % mid)): return
    V, T, _ = largest_component(z["vertices"], z["triangles"].astype(int)); n = len(V)
    v0 = centroid_vertex(V); d0 = ExactSolver(V, T)(v0); d0 = np.where(np.isfinite(d0), d0, np.nan); heat = HeatSolver(V, T)
    L, M = cotan_laplacian(V, T); G = edge_graph(T, n)
    lu = {l: splu((diags(M) + (l * l) * L).tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True)) for l in (5.0, 10.0)}
    rows = []
    for s0 in SIGMAS:
        base = A * np.exp(-np.nan_to_num(d0) ** 2 / (2 * s0 ** 2)) + B
        for kind, amp in CONDS:
            for t in range(TRIALS):
                rng = np.random.default_rng(10_000_000 * mid + int(1000 * s0) + int(100 * amp) * 7 + (0 if kind == "bumps" else (5 if kind == "corr5" else 10)) * 1000 + t)
                if kind == "bumps":
                    c = V[rng.integers(n, size=30)]; w = rng.normal(0, amp * A, 30)
                    dd = np.linalg.norm(V[:, None, :] - c[None, :, :], axis=2); dev = (np.exp(-dd ** 2 / (2 * 8.0 ** 2)) * w).sum(1) + rng.normal(0, 0.10 * A, n)
                else:
                    u = lu[5.0 if kind == "corr5" else 10.0].solve(M * rng.normal(0, 1, n)); dev = u / u.std() * amp * A
                S = np.clip(base + dev, 50, 100); S[np.isnan(d0)] = np.nan; S[rng.random(n) < 0.05] = np.nan
                o = find_origin(S, V, T); d = heat(o); c_, m_, _ = bin_profile(d, S); g = fit_gaussian(c_, m_)
                rows.append(dict(map_id=mid, sigma0=s0, kind=kind, amp=amp, trial=t, sigma_hat=g["sigma"], r2=g["r2"], F=focality(S, G), offset_mm=float(d0[o]), wsd_pct_A=within_bin_sd(d, S)))
    pd.DataFrame(rows).to_csv(os.path.join(out, "parts", "%03d.csv" % mid), index=False); print("map %3d done (%d vertices)" % (mid, n), flush=True)


def summarise(out):
    raw = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(out, "parts", "*.csv")))]); raw.to_csv(os.path.join(out, "v47b_raw.csv"), index=False)
    g = raw.groupby(["kind", "amp", "sigma0"])
    s = pd.DataFrame(dict(n=g.size(), wsd=g.wsd_pct_A.median(), r2_med=g.r2.median(), r2_q1=g.r2.quantile(.25), flag_r2=g.apply(lambda x: (x.r2 < .7).mean()),
                          F_med=g.F.median(), flag_F=g.apply(lambda x: (x.F < .7).mean()), bias_med=g.apply(lambda x: (x.sigma_hat - x.sigma0).median()),
                          bias_q1=g.apply(lambda x: (x.sigma_hat - x.sigma0).quantile(.25)), bias_q3=g.apply(lambda x: (x.sigma_hat - x.sigma0).quantile(.75)),
                          sd=g.sigma_hat.std(), off_med=g.offset_mm.median(), off_p90=g.offset_mm.quantile(.9), frac_off_gt10=g.apply(lambda x: (x.offset_mm > 10).mean()))).reset_index()
    s.to_csv(os.path.join(out, "v47b_summary.csv"), index=False)
    L = ["v47b structured / correlated deviations on %d clinical meshes — summary (%d fields)" % (raw.map_id.nunique(), len(raw)), "=" * 90,
         "clinical reference (v43): within-bin SD 37 %% of A [27; 52]; R2 median 0.82, 13/55 < 0.7; F median 0.97, 6/55 < 0.70", "",
         "  cond    amp  sigma0     n  wsd%A | R2 med  Q1   flag<0.7 | F med flag<0.7 | bias med [Q1; Q3]      SD | offset med p90  >10mm"]
    for _, r in s.iterrows():
        L.append("  %-6s %5.2f  %5.1f  %5d  %4.0f | %.3f %.3f  %5.2f   | %.2f  %5.2f   | %+5.2f [%+5.2f; %+5.2f] %5.2f | %5.1f %5.1f  %4.2f" % (
            r.kind, r.amp, r.sigma0, r.n, r.wsd, r.r2_med, r.r2_q1, r.flag_r2, r.F_med, r.flag_F, r.bias_med, r.bias_q1, r.bias_q3, r.sd, r.off_med, r.off_p90, r.frac_off_gt10))
    txt = "\n".join(L); print(txt); open(os.path.join(out, "v47b_summary.txt"), "w").write(txt + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--summary", action="store_true"); a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "parts"), exist_ok=True)
    if not a.summary:
        for f in sorted(glob.glob(os.path.join(MESHES, "map_*.npz"))): run_mesh(f, a.out)
    summarise(a.out)


if __name__ == "__main__":
    main()
