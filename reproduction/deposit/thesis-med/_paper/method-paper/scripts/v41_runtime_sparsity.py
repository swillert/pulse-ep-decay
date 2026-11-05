#!/usr/bin/env python3
"""v41 — Empirical complexity and fill-in of the Heat Method on the 59
clinical meshes (reviewer 1, statistics points 1 and 2).

For every mesh (largest component, as in the pipeline):
  * wall time of potpourri3d's MeshHeatMethodDistanceSolver construction
    (Laplacian assembly + sparse Cholesky factorisation of the two systems;
    defaults t_coef = 1, use_robust = True) and of one single-source query
    from the argmax vertex; minimum of 3 repeats.
  * nnz of the cotangent Laplacian (N + 2E) and nnz of the Cholesky factor
    of M - t L computed with SuperLU in symmetric mode (minimum-degree
    ordering MMD_AT_PLUS_A, no pivoting): for a symmetric positive-definite
    matrix this L has the structure of the Cholesky factor under that
    ordering, so nnz(L)/nnz(A) is the fill ratio.  (potpourri3d /
    geometry-central factorise with Eigen's SimplicialLDLT or SuiteSparse
    CHOLMOD, AMD ordering; the fill with AMD is of the same order.)
  * log-log fits: factor time ~ N^a, query time ~ N^b, nnz(L) ~ N^c.
Outputs: v41_per_mesh.csv, v41_summary.txt.  TEST only.
"""
import argparse, os, time, glob
import numpy as np, pandas as pd, potpourri3d as pp3d
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from pulse_ep_decay.mesh import largest_component, mean_edge_length

ROOT = os.environ.get("GD_ROOT", os.path.expanduser("~/mnt/IdeaProjects"))
MESHES = os.path.join(ROOT, "pulse-ep/data/validation_meshes")


def cotan_laplacian(V, F):
    n = len(V); I, J, W = [], [], []
    for k in range(3):
        i, j, l = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        u, w = V[i] - V[l], V[j] - V[l]
        cot = (u * w).sum(1) / np.linalg.norm(np.cross(u, w), axis=1)
        I += [i, j]; J += [j, i]; W += [0.5 * cot, 0.5 * cot]
    Wm = coo_matrix((np.concatenate(W), (np.concatenate(I), np.concatenate(J))), shape=(n, n)).tocsr()
    L = diags(np.asarray(Wm.sum(1)).ravel()) - Wm  # positive semi-definite
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    M = np.zeros(n); np.add.at(M, F.ravel(), np.repeat(area / 3, 3))
    return L.tocsc(), diags(M).tocsc()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(MESHES, "map_*.npz"))):
        z = np.load(f, allow_pickle=True); mid = int(z["map_id"])
        V, F, idx = largest_component(z["vertices"], z["triangles"].astype(int)); S = z["scores"][idx >= 0]
        if len(V) < 50: continue
        n = len(V); h = mean_edge_length(V, F); o = int(np.nanargmax(S))
        tf, tq = [], []
        for _ in range(3):
            t = time.perf_counter(); s = pp3d.MeshHeatMethodDistanceSolver(V, F); tf.append(time.perf_counter() - t)
            t = time.perf_counter(); s.compute_distance(o); tq.append(time.perf_counter() - t)
            t = time.perf_counter(); s.compute_distance((o + 1) % n); tq.append(time.perf_counter() - t)
        L, M = cotan_laplacian(V, F); A = (M + (h * h) * L).tocsc()  # heat system with t = h^2 (t_coef = 1)
        nnzA = A.nnz
        t = time.perf_counter(); lu = splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True)); tc = time.perf_counter() - t
        nnzL = lu.L.nnz
        rows.append(dict(map_id=mid, n_vertices=n, n_triangles=len(F), mean_edge_mm=h, t_factor_s=min(tf), t_query_s=min(tq), nnz_A=nnzA, nnz_L=nnzL,
                         fill_ratio=nnzL / nnzA, nnz_A_per_row=nnzA / n, nnz_L_per_row=nnzL / n, t_splu_s=tc))
        print("map %3d N=%6d factor %.3fs query %.4fs nnz(A)/N %.1f nnz(L)/N %.1f fill %.1f" % (mid, n, min(tf), min(tq), nnzA / n, nnzL / n, nnzL / nnzA), flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "v41_per_mesh.csv"), index=False)
    L_ = ["v41 runtime and fill-in of the Heat Method on %d clinical meshes — summary" % len(df), "=" * 70,
          "potpourri3d %s (geometry-central), defaults t_coef = 1 (t = h^2), use_robust = True (intrinsic-Delaunay / tufted Laplacian); single core, Linux VM on Apple silicon" % getattr(pp3d, "__version__", "1.4.0"), ""]
    for col, lab in (("t_factor_s", "solver construction (assembly + factorisation)"), ("t_query_s", "single-source query"), ("nnz_L", "nnz of Cholesky factor"), ("t_splu_s", "SuperLU symmetric factorisation (reference)")):
        x, y = np.log(df.n_vertices), np.log(df[col]); b, a0 = np.polyfit(x, y, 1); r = np.corrcoef(x, y)[0, 1]
        L_.append("  %-48s ~ N^%.2f  (log-log fit, r = %.3f); range %.4g .. %.4g" % (lab, b, r, df[col].min(), df[col].max()))
    L_ += ["", "  N range %d .. %d; nnz(A)/N median %.2f; nnz(L)/N median %.1f [%.1f; %.1f]; fill ratio nnz(L)/nnz(A) median %.2f [%.2f; %.2f], max %.2f at N=%d" % (
        df.n_vertices.min(), df.n_vertices.max(), df.nnz_A_per_row.median(), df.nnz_L_per_row.median(), df.nnz_L_per_row.min(), df.nnz_L_per_row.max(),
        df.fill_ratio.median(), df.fill_ratio.min(), df.fill_ratio.max(), df.fill_ratio.max(), df.loc[df.fill_ratio.idxmax(), "n_vertices"])]
    L_ += ["  largest mesh (N=%d): construction %.3f s, query %.4f s; smallest (N=%d): %.4f s / %.5f s" % (
        df.n_vertices.max(), df.loc[df.n_vertices.idxmax(), "t_factor_s"], df.loc[df.n_vertices.idxmax(), "t_query_s"], df.n_vertices.min(), df.loc[df.n_vertices.idxmin(), "t_factor_s"], df.loc[df.n_vertices.idxmin(), "t_query_s"])]
    txt = "\n".join(L_); print(txt); open(os.path.join(a.out, "v41_summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
