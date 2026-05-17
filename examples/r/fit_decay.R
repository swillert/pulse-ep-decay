# fit_decay.R — Re-fit a σ-recovery experiment in R.
#
# Reads a CSV produced by `export_radial_profile.py` and fits
#   s(d) = A * exp(-d^2 / (2 sigma^2)) + B
# to the binned radial profile, using base-R nls(). Bin width matches
# the Python pipeline (2 mm) so σ̂ is comparable across stacks.
#
# Usage:
#   Rscript fit_decay.R radial_profile.csv
#
# Required packages:
#   install.packages(c("ggplot2", "tibble", "readr"))

suppressPackageStartupMessages({
  library(ggplot2)
  library(tibble)
})

args <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) >= 1) args[1] else "radial_profile.csv"
if (!file.exists(csv_path)) {
  stop("CSV not found: ", csv_path,
       "\nRun export_radial_profile.py first.", call. = FALSE)
}

# --- Parse header comments to recover σ₀ etc. ----------------------------
read_meta <- function(path) {
  lines <- readLines(path, n = 20)
  meta <- list()
  for (ln in lines) {
    if (!startsWith(ln, "#")) break
    kv <- sub("^# *", "", ln)
    parts <- strsplit(kv, ":", fixed = TRUE)[[1]]
    if (length(parts) >= 2) {
      key <- trimws(parts[1])
      val <- trimws(paste(parts[-1], collapse = ":"))
      num <- suppressWarnings(as.numeric(val))
      meta[[key]] <- if (is.na(num)) val else num
    }
  }
  meta
}

meta <- read_meta(csv_path)
cat(sprintf("σ₀  = %.3f mm  (injected)\n",      meta$sigma_true_mm))
cat(sprintf("noise = %.2f                       \n", meta$noise_level))
cat(sprintf("n     = %d vertices                \n", meta$n_vertices))

# --- Load profile, drop NA/Inf, bin into 2 mm rings ---------------------
df <- read.csv(csv_path, comment.char = "#", stringsAsFactors = FALSE)
df <- df[is.finite(df$geodesic_distance_mm) & !is.na(df$score), , drop = FALSE]
cat(sprintf("kept  = %d vertices with finite (d, s)\n", nrow(df)))

BIN_WIDTH_MM <- 2.0
MIN_VERTICES_PER_BIN <- 3
MIN_BINS_FOR_FIT <- 4

df$bin <- floor(df$geodesic_distance_mm / BIN_WIDTH_MM)
agg <- aggregate(score ~ bin, data = df,
                 FUN = function(x) c(mean = mean(x), n = length(x)))
agg <- do.call(data.frame, agg)
names(agg) <- c("bin", "score_mean", "n")
agg <- agg[agg$n >= MIN_VERTICES_PER_BIN, ]
agg$d_centre <- (agg$bin + 0.5) * BIN_WIDTH_MM

if (nrow(agg) < MIN_BINS_FOR_FIT) {
  stop(sprintf("Only %d valid bins (need ≥ %d).",
               nrow(agg), MIN_BINS_FOR_FIT), call. = FALSE)
}

# --- Gaussian decay fit -------------------------------------------------
start_A <- max(agg$score_mean) - min(agg$score_mean)
start_B <- min(agg$score_mean)
start_sigma <- 10.0

fit <- nls(
  score_mean ~ A * exp(-d_centre^2 / (2 * sigma^2)) + B,
  data    = agg,
  start   = list(A = start_A, sigma = start_sigma, B = start_B),
  lower   = c(A = 0,        sigma = 0.1, B = -Inf),
  upper   = c(A = Inf,      sigma = 200, B = Inf),
  algorithm = "port",
  control = nls.control(maxiter = 200, warnOnly = TRUE)
)
co <- coef(fit)
sigma_hat <- abs(unname(co["sigma"]))

bias_mm <- sigma_hat - meta$sigma_true_mm
rel_bias <- bias_mm / meta$sigma_true_mm * 100
cat("\n--- R nls() fit ---\n")
cat(sprintf("A         = %.3f\n", co["A"]))
cat(sprintf("σ̂         = %.3f mm\n", sigma_hat))
cat(sprintf("B         = %.3f\n", co["B"]))
cat(sprintf("bias σ̂−σ₀ = %+.3f mm (%+.1f %%)\n", bias_mm, rel_bias))

# --- Plot ---------------------------------------------------------------
grid <- tibble(d = seq(0, max(agg$d_centre), length.out = 200))
grid$s_hat <- co["A"] * exp(-grid$d^2 / (2 * sigma_hat^2)) + co["B"]

p <- ggplot() +
  geom_point(data = agg,
             aes(x = d_centre, y = score_mean),
             size = 2, colour = "#1f77b4") +
  geom_line(data = grid, aes(x = d, y = s_hat),
            colour = "#e34a33", linewidth = 1.2) +
  geom_vline(xintercept = meta$sigma_true_mm,
             colour = "grey", linetype = "dashed") +
  labs(title = sprintf("σ-recovery in R — σ̂ = %.2f mm (σ₀ = %.2f mm)",
                       sigma_hat, meta$sigma_true_mm),
       subtitle = sprintf("bias = %+.2f mm (%+.1f %%)", bias_mm, rel_bias),
       x = "Geodesic distance from detected origin [mm]  (binned, 2 mm)",
       y = "Score (binned mean)") +
  theme_minimal()

out_png <- sub("\\.csv$", "_fit.png", csv_path)
ggsave(out_png, p, width = 6, height = 4, dpi = 150)
cat(sprintf("\nWrote %s\n", out_png))
