"""v52 — TEST (26 Oct 2026): clinical Gaussian vs exponential fit on the binned profiles (R2, RMSE, AIC, residual pattern).
Uses the binned profiles cached by make_revision_figures.py (April design, same bins as the pipeline) and the package fits
(pulse_ep_decay.profile.fit_gaussian / fit_exponential; 3 parameters each, same start values and bounds). Checks the
Gaussian sigma/R2 against Table S12 and the exponential R2 against v43. Nothing in the manuscript is changed by this script."""
import os, glob
import numpy as np, pandas as pd
from pulse_ep_decay.profile import fit_gaussian, fit_exponential, gauss_model, exp_model
MP = os.path.expanduser("~/mnt/IdeaProjects/thesis-med/_paper/method-paper")
CACHE = os.path.join(MP, "revision_2026-10/figs/_cache"); OUT = os.path.join(MP, "supplementary_validation_results/v52_model_comparison_plateau_2026-10-26")
S12 = pd.read_csv(os.path.join(MP, "revision_2026-10/tables/S12_per_map.csv")).set_index("map_id")
v43 = pd.read_csv(os.path.join(MP, "supplementary_validation_results/v43_lowr2_profiles_plateau_2026-10-26/v43_per_map.csv")).set_index("map_id")
os.makedirs(OUT, exist_ok=True); rows = []
for f in sorted(glob.glob(os.path.join(CACHE, "map_*.npz"))):
    z = np.load(f); m = int(z["map_id"]); c, y = z["centres"], z["means"]
    g = fit_gaussian(c, y); e = fit_exponential(c, y)
    rg = y - gauss_model(c, g["A"], g["sigma"], g["B"]); re_ = y - exp_model(c, e["A"], e["lam"], e["B"])
    near = c <= 3 * g["sigma"]
    rows.append(dict(map_id=m, n_bins=len(c), fit_range_mm=float(c.max()), sigma=g["sigma"], r2_gauss=g["r2"], r2_exp=e["r2"], lam=e["lam"],
                     rmse_gauss=float(np.sqrt(np.mean(rg ** 2))), rmse_exp=float(np.sqrt(np.mean(re_ ** 2))), dAIC=g["aic"] - e["aic"],
                     rmse_near_gauss=float(np.sqrt(np.mean(rg[near] ** 2))), rmse_near_exp=float(np.sqrt(np.mean(re_[near] ** 2))),
                     rmse_far_gauss=float(np.sqrt(np.mean(rg[~near] ** 2))) if (~near).any() else np.nan, rmse_far_exp=float(np.sqrt(np.mean(re_[~near] ** 2))) if (~near).any() else np.nan,
                     check_sigma=abs(g["sigma"] - S12.loc[m, "sigma"]), check_r2exp=abs(e["r2"] - v43.loc[m, "r2_exp"])))
d = pd.DataFrame(rows).set_index("map_id"); d.to_csv(os.path.join(OUT, "v52_per_map.csv"))
q = lambda x: "%.3f [%.3f; %.3f]" % (x.median(), x.quantile(.25), x.quantile(.75))
L = ["v52 clinical model comparison (55 maps, binned profiles, 3-parameter fits)",
     "max |sigma - S12| = %.2e mm; max |R2_exp - v43| = %.2e" % (d.check_sigma.max(), d.check_r2exp.max()),
     "R2 Gaussian %s; R2 exponential %s; exp higher in %d maps" % (q(d.r2_gauss), q(d.r2_exp), int((d.r2_exp > d.r2_gauss).sum())),
     "RMSE Gaussian %s; RMSE exponential %s score points" % (q(d.rmse_gauss), q(d.rmse_exp)),
     "near field (d <= 3 sigma): RMSE Gaussian %s, exponential %s" % (q(d.rmse_near_gauss), q(d.rmse_near_exp)),
     "far field (d > 3 sigma): RMSE Gaussian %s, exponential %s" % (q(d.rmse_far_gauss.dropna()), q(d.rmse_far_exp.dropna())),
     "dAIC (Gauss - exp) > 2: %d, < -2: %d, undecided %d" % ((d.dAIC > 2).sum(), (d.dAIC < -2).sum(), (d.dAIC.abs() <= 2).sum()),
     "R2>=0.7 maps (n=%d): R2 Gaussian %s, exponential %s" % ((d.r2_gauss >= .7).sum(), q(d[d.r2_gauss >= .7].r2_gauss), q(d[d.r2_gauss >= .7].r2_exp)),
     "fitted range: %s mm (bins %s)" % (q(d.fit_range_mm), q(d.n_bins))]
open(os.path.join(OUT, "v52_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
