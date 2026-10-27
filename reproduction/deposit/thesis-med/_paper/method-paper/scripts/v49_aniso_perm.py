#!/usr/bin/env python3
"""v49 — clinical anisotropy ratio with a 100-permutation null (D.5; supplement S8, ratio only).
Re-runs geodesic-decay/scripts/anisotropy.py::analyse + permutation_null per map (resumable parts),
on the 55-map cohort (59 meshes minus fragmented 87/102/103 and map 7). Seeds: 11_000 + map_id.
Usage: v49_aniso_perm.py run --budget 160 ; v49_aniso_perm.py assemble
"""
import sys, os, glob, json, time, ast
import numpy as np, pandas as pd
ROOT = os.path.expanduser("~/mnt/IdeaProjects")
sys.path.insert(0, os.path.join(ROOT, "geodesic-decay/scripts")); sys.path.insert(0, os.path.join(ROOT, "geodesic-decay/src"))
import anisotropy as an
from decay_pipeline import largest_component
OUT = os.path.join(ROOT, "thesis-med/_paper/method-paper/supplementary_validation_results/v49_aniso_perm_plateau_2026-10-26"); PARTS = os.path.join(OUT, "parts")
MESHES = os.path.join(ROOT, "pulse-ep/data/validation_meshes"); EXCL = {7, 87, 102, 103}; NPERM = 100


def run(budget):
    os.makedirs(PARTS, exist_ok=True); t0 = time.time()
    for f in sorted(glob.glob(os.path.join(MESHES, "map_*.npz"))):
        z = np.load(f, allow_pickle=True); mid = int(z["map_id"])
        if mid in EXCL: continue
        pf = os.path.join(PARTS, "map_%d.json" % mid)
        if os.path.exists(pf): continue
        if time.time() - t0 > budget: print("budget reached; stopping before map", mid); return
        V, T, idx = largest_component(z["vertices"], z["triangles"].astype(int))
        sc = np.full(len(V), np.nan); k = idx >= 0; sc[idx[k]] = z["scores"][k]
        t1 = time.time(); res, lm = an.analyse(V, sc, faces=T); row = dict(map_id=mid, n_vertices=int(len(V)))
        if res.get("converged"):
            rng = np.random.default_rng(11_000 + mid); null = an.permutation_null(lm, sc, res["fit_window_mm"], res["sigma_radial"], NPERM, rng)
            res["null_ratio_median"] = float(np.median(null)) if len(null) else np.nan; res["null_ratio_p95"] = float(np.percentile(null, 95)) if len(null) else np.nan
            res["p_perm"] = float((np.sum(null >= res["ratio"]) + 1) / (len(null) + 1)) if len(null) else np.nan; res["n_null"] = int(len(null))
        row.update({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in res.items() if k not in ("origin",)})
        row["seconds"] = time.time() - t1; json.dump(row, open(pf, "w"), default=str); print(mid, "ratio %.2f p %.3f (%.0f s)" % (res.get("ratio", np.nan), res.get("p_perm", np.nan), row["seconds"]), flush=True)
    print("all parts done")


def assemble():
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(PARTS, "*.json")))]
    df = pd.DataFrame(rows).sort_values("map_id"); df.to_csv(os.path.join(OUT, "v49_clinical.csv"), index=False)
    ok = df[df.converged == True]; print("maps", len(df), "converged", len(ok))
    foc = ok[ok.sigma_iso < 30]; print("focal (sigma_iso<30):", len(foc), "ratio median %.2f [%.2f; %.2f]" % tuple(foc.ratio.quantile([.5, .25, .75])), "p<0.05:", int((foc.p_perm < .05).sum()), "p<0.01:", int((foc.p_perm < .01).sum()))


if __name__ == "__main__":
    if sys.argv[1] == "run": run(float(sys.argv[3]) if len(sys.argv) > 3 else 150)
    else: assemble()
