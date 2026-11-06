import sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); import v36_analytic_geodesic as v
arch = pd.read_csv(sys.argv[2])
surfaces = []
for sub in [3, 4, 5]:
    V, F = v.sphere(25.0, sub); surfaces.append(("sphere_sub%d" % sub, V, F, lambda s, V=V: v.analytic_sphere(V, s, 25.0)))
for (nt, nz, jit) in [(64, 30, 0.0), (128, 50, 0.0), (128, 50, 0.3)]:
    V, F, (th, z) = v.cylinder(20.0, 80.0, nt, nz, jit); surfaces.append(("cylinder_%dx%d_j%.1f" % (nt, nz, jit), V, F, lambda s, th=th, z=z: v.analytic_cylinder(th, z, s, 20.0)))
for name, V, F, tf in surfaces:
    srcs = [int(s) for s in arch[arch.surface == name].src.unique()]
    a_src = arch[arch.surface == name].src.unique()

    for src in srcs:
        truth = tf(src); t0 = truth[src]; included = t0 > 1e-9
        dh = v.heat(V, F, src)[0]; dd = v.dijk(V, F, src)[0]; de = v.exact(V, F, src)[0]
        print("%-22s src %5d truth[src]=%.3e included=%s  d[src]: heat %.2e dijk %.2e exact %.2e" % (name, src, t0, included, dh[src], dd[src], de[src]))
        if included:
            for sv, d in (("heat", dh), ("dijkstra", dd), ("exact", de)):
                row = arch[(arch.surface == name) & (arch.src == src) & (arch.solver == sv)].iloc[0]
                r_old = v.errs(d, truth); m = truth > 1e-9; m[src] = False
                rel = (d[m] - truth[m]) / truth[m]
                print("      %-8s archived %.6f  recomputed(old mask) %.6f  corrected %.6f  (n masked old %d new %d)  p95 old %.6f new %.6f" % (
                    sv, row.rel_err_mean, r_old["rel_err_mean"], rel.mean(), int((truth > 1e-9).sum()), int(m.sum()), row.rel_err_p95, np.percentile(np.abs(rel), 95)))
