#!/usr/bin/env python3
"""Reaggregate existing CMPB validation CSVs; never run new simulations.

Default input paths describe the existing local workspace. Run with pulse-ep's
Python environment or any Python with numpy/pandas. Outputs stay beside this
script unless --out is explicitly supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import pandas as pd


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_hash(path):
    """Exact ordered V/F equality, preserving dtype/shape; not isomorphism."""
    with np.load(path) as data:
        h = hashlib.sha256()
        for key in ("vertices", "triangles"):
            array = np.ascontiguousarray(data[key])
            h.update(key.encode())
            h.update(str(array.dtype).encode())
            h.update(json.dumps(array.shape).encode())
            h.update(array.tobytes())
        return h.hexdigest()


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z*z/n
    centre = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return max(0., centre-half), min(1., centre+half)


def error_summary(g):
    b = g.bias_mm.to_numpy()
    return dict(n_rows=len(g), n_maps=g.map_id.nunique(),
                n_geometries=g.geometry_hash.nunique(),
                bias_mean_mm=float(b.mean()), sd_error_mm=float(b.std(ddof=1)),
                mae_mm=float(np.abs(b).mean()),
                p95_abs_error_mm=float(np.quantile(np.abs(b), .95)),
                bias_median_mm=float(np.median(b)))


def fmt_rate(row, name):
    return (f"{int(row[name+'_n'])}/{int(row.n)} "
            f"({100*row[name+'_rate']:.0f}%; 95% Wilson "
            f"{100*row[name+'_ci95_lo']:.1f}–{100*row[name+'_ci95_hi']:.1f}%)")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path,
                    default=here.parent.parent/'supplementary_validation_results')
    ap.add_argument('--meshes', type=Path,
                    default=Path('pulse-ep/data/validation_meshes'))
    ap.add_argument('--out', type=Path, default=here)
    ap.add_argument('--bootstrap', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=20261022)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    arm_path = args.results/'arm1_raw.csv'
    v37_path = args.results/'v37_threshold_focality_2026-10-21'/'v37_fields.csv'
    arm = pd.read_csv(arm_path)
    v37 = pd.read_csv(v37_path)
    assert not arm[['sigma_hat', 'bias_mm']].isna().any().any()
    assert np.allclose(arm.sigma_hat-arm.sigma_true, arm.bias_mm)
    assert not v37[['r2', 'F']].isna().any().any()

    geo_rows = []
    for mid in sorted(arm.map_id.unique()):
        path = args.meshes/f'map_{mid:03d}.npz'
        geo_rows.append(dict(map_id=int(mid), geometry_hash=geometry_hash(path),
                             input_file=str(path), input_sha256=file_hash(path)))
    geo = pd.DataFrame(geo_rows)
    geo['representative_map_id'] = geo.groupby('geometry_hash').map_id.transform('min')
    geo['geometry_multiplicity'] = geo.groupby('geometry_hash').map_id.transform('size')
    geo.to_csv(args.out/'geometry_inventory.csv', index=False)
    arm = arm.merge(geo[['map_id', 'geometry_hash', 'representative_map_id']],
                    on='map_id', validate='many_to_one')
    # Check all duplicate geometry rows are numerically identical before removal.
    check_keys = ['geometry_hash', 'sigma_true', 'noise', 'trial']
    duplicate_range = arm.groupby(check_keys)[['sigma_hat', 'r2', 'origin_shift_mm']].agg(
        lambda x: float(x.max()-x.min()))
    duplicate_max = float(duplicate_range.to_numpy().max())
    assert duplicate_max < 1e-9, duplicate_max
    unique = arm[arm.map_id == arm.representative_map_id].copy()
    assert not unique.duplicated(check_keys).any()
    rng = np.random.default_rng(args.seed)
    arm_rows = []
    cluster_rows = []
    for (s0, noise), g in arm.groupby(['sigma_true', 'noise'], sort=True):
        u = unique[(unique.sigma_true == s0) & (unique.noise == noise)]
        cluster_mean = u.groupby('geometry_hash', sort=True).bias_mm.mean()
        cluster_n = u.groupby('geometry_hash').size()
        assert cluster_n.nunique() == 1  # equal 20 trials; equal geometry weights
        n_geo = len(cluster_mean)
        draws = rng.integers(0, n_geo, size=(args.bootstrap, n_geo))
        boots = cluster_mean.to_numpy()[draws].mean(axis=1)
        lo, hi = np.quantile(boots, [.025, .975])
        for weighting, rows in [('map_weighted', g), ('unique_geometry_weighted', u)]:
            summary = dict(sigma_true_mm=s0, noise_fraction=noise, weighting=weighting,
                           **error_summary(rows),
                           geometry_bootstrap_bias_ci95_lo_mm=(lo if weighting == 'unique_geometry_weighted' else np.nan),
                           geometry_bootstrap_bias_ci95_hi_mm=(hi if weighting == 'unique_geometry_weighted' else np.nan))
            arm_rows.append(summary)
        for h, gm in cluster_mean.items():
            cluster_rows.append(dict(sigma_true_mm=s0, noise_fraction=noise,
                                     geometry_hash=h, n_trials=int(cluster_n[h]),
                                     mean_bias_mm=gm))
    arms = pd.DataFrame(arm_rows)
    arms.to_csv(args.out/'arm1_error_summary.csv', index=False)
    pd.DataFrame(cluster_rows).to_csv(args.out/'arm1_geometry_means.csv', index=False)

    # Represent every generator parameter, including fixed widths omitted from
    # the original CSV. The two-peak field is max(component1, component2).
    v37['true_width_mm'] = v37.sigma0
    v37.loc[v37.kind == 'two-peak', 'true_width_mm'] = 7.
    v37.loc[v37.kind == 'ring', 'true_width_mm'] = 4.
    v37['ring_radius_mm'] = np.where(v37.kind == 'ring', 15., np.nan)
    keys = ['kind', 'true_width_mm', 'noise', 'D', 'ratio', 'ring_radius_mm']
    flag_rows = []
    for values, g in v37.groupby(keys, dropna=False, sort=True):
        kind, width, noise, distance, ratio, ring_radius = values
        assert len(g) == 20 and g.trial.nunique() == 20
        for tr in (.6, .7, .8, .9):
            for tf in (.6, .7, .8, .9):
                rflag = g.r2.to_numpy() < tr
                fflag = g.F.to_numpy() < tf
                row = dict(kind=kind, true_width_mm=width,
                           width_definition=('radial_sd_about_ring_radius' if kind == 'ring'
                                             else 'component_gaussian_sigma' if kind != 'flat'
                                             else 'not_applicable'),
                           noise_fraction=noise, noise_sd_score=35*noise,
                           amplitude_score=35., baseline_score=65.,
                           nominal_separation_mm=distance, amplitude_ratio=ratio,
                           ring_radius_mm=ring_radius,
                           R2_threshold=tr, F_threshold=tf, n=len(g))
                metrics = {'R2_flag': rflag, 'F_flag': fflag,
                           'either_flag': rflag | fflag, 'both_flags': rflag & fflag,
                           'joint_pass': ~(rflag | fflag)}
                for name, indicator in metrics.items():
                    k = int(indicator.sum()); lo, hi = wilson(k, len(g))
                    row.update({name+'_n': k, name+'_rate': k/len(g),
                                name+'_ci95_lo': lo, name+'_ci95_hi': hi})
                flag_rows.append(row)
    flags = pd.DataFrame(flag_rows)
    flags.to_csv(args.out/'v37_threshold_grid.csv', index=False)
    default = flags[(flags.R2_threshold == .7) & (flags.F_threshold == .7)]
    default.to_csv(args.out/'v37_threshold_070.csv', index=False)
    assert len(flags) == 56*16 and len(default) == 56
    assert (flags.either_flag_n+flags.joint_pass_n == flags.n).all()
    assert (flags.R2_flag_n+flags.F_flag_n-flags.both_flags_n == flags.either_flag_n).all()

    lines = [
        '# Nachaggregation bestehender Validierungsdaten — 22.10.2026', '',
        '**Keine neuen Simulationen.** Die folgenden Resultate wurden ausschließlich aus vorhandenen arm1-/v37-CSV neu aggregiert.', '',
        '## Arm1: gleiche Gewichtung jeder eindeutigen Geometrie', '',
        f'{arm.map_id.nunique()} Map-Dateien entsprechen {geo.geometry_hash.nunique()} exakt unterschiedlichen geordneten Vertex-/Dreieck-Arrays. '
        f'Doppelte Geometrien liefern identische σ/R²/Ursprungsversatz-Zeilen (maximale Differenz {duplicate_max:g}). '
        'Für die eindeutige Auswertung bleibt jeweils die kleinste Map-ID erhalten. Pro σ/Rauschzelle bleiben 40 Geometrien × 20 Trials = 800 Zeilen statt 56×20 = 1.120.', '',
        'SD ist die Stichproben-SD der einzelnen Fehler; MAE und p95 beziehen sich auf deren Absolutbeträge. '
        'Die 95%-Intervalle des mittleren Bias entstehen aus 10.000 Bootstrap-Ziehungen von 40 vollständigen Geometrie-Clustern mit Zurücklegen; alle 20 Trials eines gezogenen Clusters bleiben zusammen. '
        'Sie ersetzen die bisherige Einzelzeilen-Resampling-Annahme für diese Auswertung.', '',
        '**Geltungsbereich:** bedingter numerischer Benchmark der beobachteten Geometrien, Ursprünge und vorhandenen Zufallsrealisierungen; keine Patienten-/Populations- oder klinischen Genauigkeitsintervalle. '
        'Die ursprüngliche Simulation benutzt denselben Trial-Seed auf allen Geometrien. Diese gemeinsamen Zufallszahlen können Resultate zwischen Geometrien korrelieren; das Geometrie-Bootstrap ist ausdrücklich bedingt auf den vorhandenen 20 Realisierungen und quantifiziert deren Monte-Carlo-Unsicherheit nicht. '
        'Auch die 40 Geometrien sind nicht als unabhängige Patienten interpretiert. Neue unabhängige Seeds wären für eine zusätzliche unbedingte Simulationsunsicherheitsanalyse nötig.', '',
        'Bei 10 % iid-Vertexrauschen (Einheiten mm):', '',
        '| σ₀ | Gewichtung | Bias | SD | MAE | p95 absolut | Geometrie-Bootstrap 95% Bias |',
        '|---:|---|---:|---:|---:|---:|---|'
    ]
    for _, r in arms[np.isclose(arms.noise_fraction, .1)].iterrows():
        ci = ('—' if r.weighting == 'map_weighted' else
              f'[{r.geometry_bootstrap_bias_ci95_lo_mm:.3f}; {r.geometry_bootstrap_bias_ci95_hi_mm:.3f}]')
        lines.append(f'| {r.sigma_true_mm:g} | {r.weighting} | {r.bias_mean_mm:+.3f} | {r.sd_error_mm:.3f} | {r.mae_mm:.3f} | {r.p95_abs_error_mm:.3f} | {ci} |')
    lines += ['', '## v37: vollständiges Raster und endliche Stichproben', '',
              'Die Tabelle `v37_threshold_grid.csv` enthält 56 konkrete Generatorbedingungen × 16 Schwellenpaare (R²/F jeweils 0,6/0,7/0,8/0,9), '
              'jeweils n=20. Angegeben sind Anzahl, Rate und zweiseitiges 95%-Wilson-Intervall für R²-Flag, F-Flag, mindestens einen Flag, beide Flags und gemeinsames Bestehen. '
              'Flags gelten strikt unter der Schwelle, Bestehen bei ≥. `v37_threshold_070.csv` ist die vollständige 0,70/0,70-Auswahl. '
              'Die Raten sind bedingte synthetische Flag-Häufigkeiten; sie werden nicht als klinische Sensitivität oder Spezifität bezeichnet.', '',
              'Wilson-Intervalle gelten je Zelle unter der Annahme unabhängiger Simulationswiederholungen. '
              'Felder innerhalb derselben Trial-ID teilen den Ausgangspunkt und teilweise den zweiten Gipfel; die Zellen sind daher nicht voneinander unabhängig. '
              'Die Intervalle sind weder simultan über alle Zellen gültig noch für die Auswahl einer optimalen Schwelle adjustiert.', '',
              'Ausgewählte Resultate bei R²/F=0,70:', '']
    for width in (5, 10, 15):
        r = default[(default.kind == 'focal') & (default.true_width_mm == width) &
                    np.isclose(default.noise_fraction, .1)].iloc[0]
        lines.append(f'- Einzelgipfel σ₀={width} mm, 10 % Rauschen: F-Flag {fmt_rate(r, "F_flag")}.')
    r = default[(default.kind == 'ring') & np.isclose(default.noise_fraction, .05)].iloc[0]
    lines.append(f'- Ring, 5 % Rauschen: gemeinsames Bestehen {fmt_rate(r, "joint_pass")}.')
    for noise in (.05, .1, .15, .25):
        r = default[(default.kind == 'flat') & np.isclose(default.noise_fraction, noise)].iloc[0]
        lines.append(f'- Flaches Feld, {100*noise:g} % Rauschen: R²-Flag {fmt_rate(r, "R2_flag")}.')
    lines += ['', 'Die v37-Definition der Zweigipfelfelder ist das Maximum zweier Gaußkomponenten (σ=7 mm), nicht deren Summe. '
              'Die Ringspalte `true_width_mm=4` bezeichnet die radiale Standardabweichung um den Ringradius 15 mm; beim flachen Feld ist eine Breite nicht definiert. '
              '`amplitude_score=35` ist bei flachen Feldern die Rauschnormierung, kein vorhandener Gipfel. '
              'Diese Charakterisierung validiert weder die Isotropie noch die klinische Lokalisationsgenauigkeit.', '',
              '## Reproduktion', '', '```sh',
              'pulse-ep/.venv/bin/python reaggregate.py', '```', '',
              'Aus diesem Ordner ausführen. `--results`, `--meshes`, `--out`, `--bootstrap` und `--seed` sind explizit überschreibbar. '
              'Eingabe-Hashes, Versionen und Parameter stehen in `manifest.json`; die geometrischen Zuordnungen in `geometry_inventory.csv`.']
    (args.out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    manifest = dict(date='2026-10-22', activity='reaggregation_only_no_new_simulations',
                    script_sha256=file_hash(__file__), python=platform.python_version(),
                    numpy=np.__version__, pandas=pd.__version__,
                    bootstrap_replicates=args.bootstrap, bootstrap_seed=args.seed,
                    bootstrap_unit='unique geometry, retaining all 20 existing trials',
                    inputs=[dict(path=str(p.resolve()), sha256=file_hash(p)) for p in [arm_path, v37_path]],
                    n_maps=int(arm.map_id.nunique()), n_unique_geometries=int(geo.geometry_hash.nunique()),
                    n_arm1_rows=len(arm), n_unique_geometry_rows=len(unique),
                    max_duplicate_geometry_outcome_difference=duplicate_max,
                    n_v37_rows=len(v37), n_v37_threshold_combinations=len(flags),
                    outputs=['geometry_inventory.csv','arm1_error_summary.csv','arm1_geometry_means.csv',
                             'v37_threshold_grid.csv','v37_threshold_070.csv','SUMMARY.md'])
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
