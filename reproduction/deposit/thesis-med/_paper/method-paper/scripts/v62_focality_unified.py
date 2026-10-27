"""v62 — TEST (26 Oct 2026, review round 5): v50 with the single F implementation of the package
(pulse_ep_decay.analysis.focality_index: F unavailable when the fit failed, the fitted amplitude is not positive, or the
half-maximum set has fewer than three vertices).  v50 had the < 3 rule but no amplitude check.
Parts s (sphere) and c (clinical) are rerun with identical fields and seeds (v50.part_s / part_c, F_half replaced);
part m (56 meshes, slow) is not rerun: the two rules differ only where A <= 0, and v50_meshes.csv has no such row
(checked here and written unchanged as v62_meshes.csv).  Output: v62_focality_unified_2026-10-26/."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v50_focality_halfmax as v50
from pulse_ep_decay.analysis import focality_index

OUT = os.path.join(v50.SVR, "v62_focality_unified_2026-10-26"); os.makedirs(OUT, exist_ok=True)
OLD = v50.OUT


def F_half(S, G, o, Afit, Bfit):
    thr = Bfit + 0.5 * Afit
    F, ncomp, _ = focality_index(S, None, o, Afit, Bfit, graph=G)
    _, lab, sz = v50.components(S, G, np.isfinite(S) & (S >= thr))
    return F, ncomp, (float(sz.max() / sz.sum()) if lab is not None and np.isfinite(F) else np.nan), thr


v50.F_half = F_half; v50.OUT = OUT
if __name__ == "__main__":
    v50.part_s(); v50.part_c()
    m = pd.read_csv(os.path.join(OLD, "v50_meshes.csv")); assert not (m.A <= 0).any(), "meshes contain A <= 0 rows: rerun part m"
    m.to_csv(os.path.join(OUT, "v62_meshes.csv"), index=False)
    os.rename(os.path.join(OUT, "v50_sphere.csv"), os.path.join(OUT, "v62_sphere.csv")); os.rename(os.path.join(OUT, "v50_clinical.csv"), os.path.join(OUT, "v62_clinical.csv"))
    L = ["v62 focality index with the package helper (unified contract) — comparison with v50"]
    for part in ("sphere", "clinical"):
        a = pd.read_csv(os.path.join(OLD, "v50_%s.csv" % part)); b = pd.read_csv(os.path.join(OUT, "v62_%s.csv" % part))
        same = np.isclose(a.F, b.F, equal_nan=True); L.append("  %s: rows %d; F identical (incl. NaN) in %d; changed %d; F NaN v50 %d -> v62 %d; sigma max |diff| %.2g" % (
            part, len(a), same.sum(), (~same).sum(), a.F.isna().sum(), b.F.isna().sum(), (a.sigma_hat - b.sigma_hat).abs().max() if "sigma_hat" in a else (a.sigma - b.sigma).abs().max()))
        if (~same).any(): L.append("    changed rows: " + "; ".join("%s %s amp %.2f A=%.2f F %.3f->%s" % (r.kind, r.noise, r.amp, r.A, a.F[i], b.F[i]) for i, r in b[~same].iterrows()) if part == "sphere" else "")
    L.append("  meshes: %d rows, no A <= 0 -> unchanged (F NaN %d)" % (len(m), m.F.isna().sum()))
    open(os.path.join(OUT, "v62_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
