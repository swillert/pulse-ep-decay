# σ recovery in R

1. **Export** — `export_radial_profile.py` builds the icosphere, injects a
   Gaussian field of width σ₀, detects the origin and computes Heat-Method
   geodesic distances with `pulse_ep_decay`, and writes one row per vertex
   (`geodesic_distance_mm`, `score`; empty where unscored). Header comments
   record σ₀, the origin and the width the Python pipeline recovers.
2. **Fit** — `fit_decay.R` reads the CSV, bins the distances into 2-mm rings
   with at least three vertices exactly as `pulse_ep_decay.profile.bin_profile`
   does, fits `A·exp(−d²/2σ²) + B` with base-R `nls()` (σ ∈ [0.1, 200] mm),
   prints σ̂ with its bias against σ₀ and against the Python value, and writes
   `<csv>_fit.png`.

```bash
cd examples/r
python export_radial_profile.py --sigma-true 8.0 --noise 0.05 --output radial_profile.csv
Rscript fit_decay.R radial_profile.csv
```

R packages: `install.packages(c("ggplot2", "tibble"))`.
