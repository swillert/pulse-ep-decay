# σ-recovery in R

Two-step demo that re-implements the σ-fit in R while reusing the
Python pipeline for the geodesic-distance computation (Heat-Method on
a sparse Laplacian — not trivial in R).

1. **Export** — `export_radial_profile.py` builds the icosphere, runs
   `MeshHeatSolver` and `generate_synthetic_field`, and writes a CSV
   with the per-vertex `(geodesic_distance_mm, score)` cloud plus a
   header line recording the injected σ₀.
2. **Fit** — `fit_decay.R` reads that CSV, bins distances into 2 mm
   rings exactly like `pulse_ep_decay.decay_fit.fit_gaussian_decay`,
   and fits `A·exp(-d²/(2σ²)) + B` with base-R `nls()`.

The recovered σ̂ should agree with the Python pipeline (which itself
recovers σ̂ ≈ σ₀ — see [`../sigma_fit_synthetic.ipynb`](../sigma_fit_synthetic.ipynb)).

## Required R packages

```r
install.packages(c("ggplot2", "tibble", "readr"))
```

## Run

```bash
# from the repo root, with the package installed (pip install -e ".[dev]")
cd examples/r

# 1) Python helper: writes radial_profile.csv next to itself
python export_radial_profile.py --sigma-true 8.0 --noise 0.05 --output radial_profile.csv

# 2) R: read CSV, fit, plot
Rscript fit_decay.R radial_profile.csv
```

Override `--sigma-true` / `--noise` to try other parameter combinations.
Use `--seed` for fully deterministic runs.

## Reading the output

`fit_decay.R` prints σ̂ and the recovered amplitude/baseline, and
writes `radial_profile_fit.png` with the binned profile and the fitted
curve. The header of `radial_profile.csv` records the injected σ₀ so
the R script can quote the bias directly.
