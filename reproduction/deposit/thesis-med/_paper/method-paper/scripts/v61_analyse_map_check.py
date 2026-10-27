"""v61 — TEST (26 Oct 2026, review round 4, Q4): the public entry point pulse_ep_decay.analyse_map (Algorithm 1)
run on the 55 clinical maps; every reported per-map quantity compared with Supplementary Table S12
(revision_2026-10/tables/S12_per_map.csv, produced by the study scripts).  Output: v61_check.csv, v61_summary.txt."""
import os, numpy as np, pandas as pd
from pulse_ep_decay import analyse_map, largest_component
ROOT = os.path.expanduser("~/mnt/IdeaProjects"); MESHES = os.path.join(ROOT, "pulse-ep/data/validation_meshes")
MP = os.path.join(ROOT, "thesis-med/_paper/method-paper"); OUT = os.path.join(MP, "supplementary_validation_results/v61_analyse_map_check_2026-10-26"); os.makedirs(OUT, exist_ok=True)
S12 = pd.read_csv(os.path.join(MP, "revision_2026-10/tables/S12_per_map.csv")).set_index("map_id"); rows = []
for mid in S12.index:
    z = np.load(os.path.join(MESHES, "map_%03d.npz" % mid), allow_pickle=True)
    V, F, idx = largest_component(z["vertices"], z["triangles"].astype(int)); S = z["scores"].astype(float)[idx >= 0]
    r = analyse_map(S, V, F); r["map_id"] = mid; rows.append(r)
R = pd.DataFrame(rows).set_index("map_id"); R.to_csv(os.path.join(OUT, "v61_analyse_map.csv"))
L = ["v61 analyse_map vs S12 (n=%d maps)" % len(R)]
for a, b in (("sigma", "sigma"), ("r2", "r2"), ("F", "F"), ("dAIC", "dAIC_gauss_minus_exp"), ("A", "A"), ("B", "B")):
    d = (R[a] - S12[b]).abs(); L.append("  %-6s max |diff| %.3g  (n>1e-6: %d)" % (a, d.max(), (d > 1e-6).sum()))
for a, b in (("width_class", "class"), ("flag_r2", "flag_r2"), ("flag_F", "flag_F")):
    L.append("  %-11s identical in %d / %d" % (a, (R[a].astype(str) == S12[b].astype(str)).sum(), len(R)))
L.append("  n_components identical in %d / %d" % ((R.n_components == S12.n_components).sum(), len(R)))
L.append("  status: %s" % R.status.value_counts().to_dict())
open(os.path.join(OUT, "v61_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
