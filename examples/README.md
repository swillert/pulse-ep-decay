# Examples

Small, self-contained demonstrations that the published width σ can be
reproduced on a synthetic surface — in Python, and independently in R.
They use a 642-vertex icosphere of 25 mm radius (`_mesh_utils.py`, no
pyvista needed) and the manuscript's synthetic protocol (`synth_field`:
A = 35, B = 65, noise, clipping to [50, 100], 5 % unscored vertices).

| File | What it shows |
|---|---|
| `sigma_recovery.py` | End-to-end in Python: inject σ₀, run the pipeline (`analyse_map`), print σ̂, class, R², F and flags; `--plot` draws the binned profile with the fit. |
| `r/export_radial_profile.py` + `r/fit_decay.R` | Python computes only the geodesic distances (Heat Method) and writes a CSV; base-R `nls()` re-fits the same Gaussian on the same 2-mm bins. σ̂ from R should agree with the Python pipeline to within the noise. |

```bash
pip install -e ".[dev]"                        # from the repository root
python examples/sigma_recovery.py --sigma-true 8 --noise 0.05 --plot sigma_recovery.png
python examples/r/export_radial_profile.py --sigma-true 8 --output radial_profile.csv
Rscript examples/r/fit_decay.R radial_profile.csv   # needs R with ggplot2, tibble
```

Why an R re-fit: a reviewer who asks whether the σ values depend on our
code gets a chain in which only the geodesic distances come from this
package and the fit is done by a different optimiser in a different
language. Recovering σ̂ ≈ σ₀ there is a stronger statement than re-running
the Python script.

For the real data path (pace maps in a pulse-ep database) see
[`docs/GUIDE.md`](../docs/GUIDE.md) and `pulse-ep-decay run`.
