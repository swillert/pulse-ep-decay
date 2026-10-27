"""Geometry-only feasibility check; no noise, fitted origins or method outcomes.

Run with pulse-ep Python and geodesic-decay/src plus pygeodesic on PYTHONPATH.
The report records provisional geometric choices; this does not lock the protocol.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from pygeodesic import geodesic
from pulse_ep_decay.mesh import largest_component
from pulse_ep_decay.support import edge_graph

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('pulse-ep/data')
OUT = ROOT / 'supplementary_validation_results/geometry_preflight_2026-09-10'


def clip(poly, values):
    """Clip a barycentric polygon by a linear inequality values >= 0."""
    result = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        fa, fb = a @ values, b @ values
        if fa >= 0:
            result.append(a)
        if (fa >= 0) != (fb >= 0):
            result.append(a + fa / (fa - fb) * (b - a))
    return np.asarray(result).reshape(-1, 3)


def fraction(poly):
    if len(poly) < 3:
        return 0.0
    xy = poly[:, 1:]
    return abs(float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1)
                            - xy[:, 1] * np.roll(xy[:, 0], -1))))


def clipped_area(triangles, areas, *fields):
    """Exact intersection area of piecewise-linear nonnegative vertex fields."""
    arrays = [f[triangles] for f in fields]
    possible = np.logical_and.reduce([np.max(a, axis=1) >= 0 for a in arrays])
    full = np.logical_and.reduce([np.min(a, axis=1) >= 0 for a in arrays])
    total = float(areas[full].sum())
    for i in np.flatnonzero(possible & ~full):
        poly = np.eye(3)
        for a in arrays:
            poly = clip(poly, a[i])
            if len(poly) < 3:
                break
        total += areas[i] * fraction(poly)
    return total


def self_check():
    t = np.array([[0, 1, 2]])
    a = np.array([1.0])
    assert np.isclose(clipped_area(t, a, np.array([-.5, -.5, .5])), .25)
    assert np.isclose(clipped_area(t, a, np.array([-.5, .5, .5])), .75)
    assert np.isclose(clipped_area(t, a, np.array([.5, -.5, .5]),
                                 np.array([.5, .5, -.5])), .5)
    rng = np.random.default_rng(20260910)
    bary = rng.dirichlet(np.ones(3), size=200000)
    f, g = np.array([-.3, .8, .2]), np.array([.7, -.2, .4])
    exact = clipped_area(t, a, f, g)
    assert abs(exact - np.mean((bary @ f >= 0) & (bary @ g >= 0))) < .004
    assert np.isclose(exact, clipped_area(t, a, g, f))


def run_map(map_id, patient):
    mp = DATA / 'validation_meshes' / f'map_{map_id:03d}.npz'
    pp = DATA / 'decay_points_pointscores' / f'decay_points_{map_id:03d}.csv'
    mesh = np.load(mp)
    v, t, _ = largest_component(mesh['vertices'], mesh['triangles'].astype(int))
    points = pd.read_csv(pp, usecols=['x', 'y', 'z', 'point_score'])
    finite = np.isfinite(points.point_score.to_numpy())
    # Actual score magnitudes are never used.
    xyz = points.loc[finite, ['x', 'y', 'z']].to_numpy()
    sites = np.unique(cKDTree(v).query(xyz)[1])
    nearest = cKDTree(v[sites]).query(v)[0]
    areas = .5 * np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]],
                                        v[t[:, 2]] - v[t[:, 0]]), axis=1)
    weights = np.bincount(t.ravel(), weights=np.repeat(areas / 3, 3), minlength=len(v))
    solver = geodesic.PyGeodesicAlgorithmExact(v.astype(np.float64), t.astype(np.int32))
    def distance(origin):
        return np.asarray(solver.geodesicDistances(np.array([origin], dtype=np.int32), None)[0])
    edges, counts = np.unique(np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]],
                                                     t[:, [2, 0]]]), axis=1),
                              axis=0, return_counts=True)
    boundary = np.unique(edges[counts == 1])
    max_boundary_edge = (np.linalg.norm(v[edges[counts == 1, 0]] - v[edges[counts == 1, 1]], axis=1).max()
                         if len(boundary) else 0.0)
    bd = (np.asarray(solver.geodesicDistances(boundary.astype(np.int32), None)[0])
          if len(boundary) else np.full(len(v), np.inf))
    candidates = np.flatnonzero((nearest > 2.5) & (nearest <= 5) & (bd >= 3))
    origins, distances = [], []
    if len(candidates):
        centre = np.average(v[candidates], weights=weights[candidates], axis=0)
        first = int(candidates[np.argmin(np.linalg.norm(v[candidates] - centre, axis=1))])
        origins.append(first)
        distances.append(distance(first))
        while len(origins) < 4:
            separation = np.min(np.stack(distances)[:, candidates], axis=0)
            index = int(np.argmax(separation))
            if separation[index] < 10:
                break
            origins.append(int(candidates[index]))
            distances.append(distance(origins[-1]))
    coverage = {r: clipped_area(t, areas, r - nearest) / areas.sum() for r in (3, 5, 10)}
    metadata = dict(map_id=map_id, study_id=patient, n_positions=len(points),
                    n_finite_sites=int(finite.sum()), n_unique_sites=len(sites),
                    n_vertices=len(v), n_candidates=len(candidates), n_origins=len(origins),
                    n_boundary_vertices=len(boundary), max_boundary_edge_mm=float(max_boundary_edge),
                    mesh_area_cm2=float(areas.sum()/100),
                    **{f'area_coverage_r{r}':coverage[r] for r in coverage})
    rows = []
    graph = edge_graph(v, t)
    for number, (origin, d) in enumerate(zip(origins, distances), 1):
        local = np.flatnonzero(d <= 10)
        centre = np.average(v[local], axis=0, weights=weights[local])
        centred = v[local] - centre
        covariance = (centred * weights[local, None]).T @ centred / weights[local].sum()
        _, axes = np.linalg.eigh(covariance)
        tangent = axes[:, 1:]
        for scale in (5, 10, 15):
            near_sites = sites[d[sites] <= 2 * scale]
            offsets = (v[near_sites] - v[origin]) @ tangent
            norms = np.linalg.norm(offsets, axis=1)
            angles = np.sort(np.arctan2(offsets[norms > 1e-8, 1], offsets[norms > 1e-8, 0]))
            gap = (float(np.degrees(np.diff(np.r_[angles, angles[0] + 2*np.pi]).max()))
                   if len(angles) else 360.0)
            for shape in ('gaussian', 'exponential'):
                field = (np.exp(-d*d/(2*scale*scale)) if shape == 'gaussian'
                         else np.exp(-d/(scale*np.sqrt(2/np.log(2))))) - .5
                active = np.flatnonzero(field >= 0)
                _, labels = connected_components(graph[active][:, active], directed=False)
                label = labels[np.searchsorted(active, origin)]
                in_component = np.zeros(len(v), dtype=bool)
                in_component[active[labels == label]] = True
                keep = in_component[t].any(axis=1)
                tt, aa = t[keep], areas[keep]
                full = clipped_area(tt, aa, field)
                for radius in (3, 5, 10):
                    supported = clipped_area(tt, aa, field, radius-nearest)
                    ratio = supported/full
                    inside = bool(nearest[origin] <= radius)
                    evaluable = inside and ratio >= .8 and len(near_sites) >= 3 and gap <= 180
                    rows.append(dict(map_id=map_id, study_id=patient, origin_number=number,
                        origin_vertex=origin, shape=shape, scale_mm=scale, radius_mm=radius,
                        origin_site_distance_mm=float(nearest[origin]),
                        origin_boundary_vertex_distance_mm=float(bd[origin]),
                        boundary_rule_conservative_pass=bool(bd[origin]-max_boundary_edge/2 >= 3),
                        true_area_cm2=full/100, supported_area_cm2=supported/100,
                        halfheight_coverage=ratio, n_nearby_sites=len(near_sites),
                        n_tangent_directions=len(angles), max_angular_gap_deg=gap,
                        origin_inside=inside, coverage_pass=ratio >= .8,
                        site_count_pass=len(near_sites)>=3, angular_pass=gap<=180,
                        oracle_evaluable=evaluable))
    return metadata, rows, [mp, pp]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    self_check()
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'supplementary_validation_results'
    cohort = pd.read_csv(source/'v9_display_vs_support.csv')[['map_id']].merge(
        pd.read_csv(source/'map_study.csv'), on='map_id', validate='one_to_one')
    if args.limit:
        cohort = cohort.head(args.limit)
    all_rows, maps, inputs = [], [], [Path(__file__), source/'map_study.csv',
        source/'v9_display_vs_support.csv', ROOT/'ANALYSIS_PLAN_map_support_2026-09-09.md']
    started = time.time()
    for row in cohort.itertuples():
        meta, rows, files = run_map(int(row.map_id), int(row.study_id))
        maps.append(meta); all_rows.extend(rows); inputs.extend(files)
        print(f'Map {row.map_id}: {meta["n_candidates"]} candidates, {meta["n_origins"]} origins; '
              f'{time.time()-started:.1f}s', flush=True)
    details = pd.DataFrame(all_rows)
    pd.DataFrame(maps).to_csv(OUT/'maps.csv', index=False)
    details.to_csv(OUT/'origins.csv', index=False)
    summary = []
    for key, group in details.groupby(['radius_mm','shape','scale_mm']):
        good = group[group.oracle_evaluable]
        summary.append(dict(radius_mm=key[0], shape=key[1], scale_mm=key[2],
            n_origins=len(group), n_evaluable=len(good), n_maps=good.map_id.nunique(),
            n_patients=good.study_id.nunique(), median_coverage=group.halfheight_coverage.median(),
            coverage_q25=group.halfheight_coverage.quantile(.25),
            coverage_q75=group.halfheight_coverage.quantile(.75),
            coverage_pass=int(group.coverage_pass.sum()),
            site_count_pass=int(group.site_count_pass.sum()), angular_pass=int(group.angular_pass.sum()),
            meets_12_patients=good.study_id.nunique()>=12))
    pd.DataFrame(summary).to_csv(OUT/'summary.csv', index=False)
    manifest = {'scope':'geometry only; no method estimates or noise',
        'elapsed_seconds':time.time()-started, 'maps':len(maps),
        'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'choices':{'tangent':'area-weighted PCA of vertices within exact 10-mm geodesic radius',
                   'boundary':'exact distance to boundary vertices; conservative endpoint bound also reported',
                   'nearby_sites':'exact geodesic radius 2 times Gaussian-equivalent scale for both shapes',
                   'truth_area':'piecewise-linear truth clipped at half amplitude, origin component only',
                   'selection':'exact geodesic farthest point; deterministic vertex-index tie break'}}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
