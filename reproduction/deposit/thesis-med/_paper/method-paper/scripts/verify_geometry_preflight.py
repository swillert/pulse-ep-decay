"""Validate saved geometry-only output and emit patient/intersection summaries."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from geometry_preflight import OUT, self_check

self_check()
maps = pd.read_csv(OUT / 'maps.csv')
origins = pd.read_csv(OUT / 'origins.csv')
summary = pd.read_csv(OUT / 'summary.csv')
manifest = json.loads((OUT / 'manifest.json').read_text())
for name, expected in manifest['inputs'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
assert len(maps) == 55 and maps.study_id.nunique() == 16
assert maps.n_positions.sum() == 2886 and maps.n_finite_sites.sum() == 2836
assert maps.n_origins.sum() == 219 and len(origins) == 219 * 6 * 3
assert not origins.isna().any().any()
assert origins.halfheight_coverage.between(-1e-10, 1 + 1e-10).all()
assert origins.origin_site_distance_mm.between(2.5, 5, inclusive='right').all()
assert (origins.origin_boundary_vertex_distance_mm >= 3).all()
assert (origins.oracle_evaluable == (origins.origin_inside & origins.coverage_pass
        & origins.site_count_pass & origins.angular_pass)).all()
for _, group in origins.groupby(['map_id', 'origin_vertex', 'shape', 'scale_mm']):
    assert (np.diff(group.sort_values('radius_mm').halfheight_coverage) >= -1e-10).all()
for row in summary.itertuples():
    g = origins[(origins.radius_mm == row.radius_mm) & (origins['shape'] == row.shape)
                & (origins.scale_mm == row.scale_mm)]
    good = g[g.oracle_evaluable]
    assert len(good) == row.n_evaluable
    assert good.map_id.nunique() == row.n_maps and good.study_id.nunique() == row.n_patients
    assert bool(row.meets_12_patients) == (row.n_patients >= 12)
patients = origins.groupby(['study_id', 'radius_mm', 'shape', 'scale_mm']).agg(
    n_selected_origins=('origin_vertex', 'size'),
    n_evaluable_origins=('oracle_evaluable', 'sum')).reset_index()
patients.to_csv(OUT / 'patients.csv', index=False)
intersections = []
for (shape, scale), g in origins.groupby(['shape', 'scale_mm']):
    wide = g.pivot(index=['study_id', 'map_id', 'origin_vertex'],
                   columns='radius_mm', values='oracle_evaluable')
    keep = wide.all(axis=1)
    good = wide[keep].reset_index()
    intersections.append(dict(shape=shape, scale_mm=scale, n_evaluable=int(keep.sum()),
        n_maps=good.map_id.nunique(), n_patients=good.study_id.nunique()))
pd.DataFrame(intersections).to_csv(OUT / 'radius_intersection.csv', index=False)
print('PASS: analytic and Monte Carlo clipping checks; all input hashes; cohort counts; '
      '219 origins; 3,942 rows; coverage bounds and radius monotonicity; oracle rules; '
      'patient/map summaries. No method outcomes were computed.')
