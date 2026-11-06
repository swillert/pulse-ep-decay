#!/usr/bin/env python3
"""make_revision_figures.py — figures of the revision of CMPB-D-26-02177 (Oct 2026).

Every sigma / R2 / F shown in a figure is computed here with the pulse_ep_decay package on the archived
meshes (pulse-ep/data/validation_meshes/map_<id>.npz), i.e. the code path of v38/v39, and asserted against
revision_2026-10/tables/S12_per_map.csv (the output of make_revision_tables.py).  The surface renderer is
adapted from geodesic-decay/scripts/make_fig_surface_v6.py (7 Sept 2026): a matplotlib painter's algorithm
with vertex shading, geodesic iso-lines by marching triangles and a projected-neighbourhood depth test;
no OpenGL, no database.  The marked pacing site comes from lo_selection_2026-09-14/selection.json
(route unique_lo_reference_candidate, nearest vertex), as in v46.

Usage: make_revision_figures.py cache 25 78 ... | heatmap | profiles | validation | boxplot | workflow | example | supp | all
       data = validation, boxplot, supp (only deposited result files; the other figures need the clinical meshes or the per-map cache).
GD_ROOT (environment) replaces ~/mnt/IdeaProjects as the root of all paths, e.g. for an unpacked deposit.
Outputs: revision_2026-10/figs/*.pdf (+ .png previews); per-map cache in revision_2026-10/figs/_cache/.
"""
import sys, os, json, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib import cm, colors as mcolors
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pulse_ep_decay.mesh import HeatSolver, largest_component, mean_edge_length
from pulse_ep_decay.profile import bin_profile, fit_gaussian, find_origin
from pulse_ep_decay.analysis import focality_index

ROOT = os.environ.get("GD_ROOT", os.path.expanduser("~/mnt/IdeaProjects"))  # GD_ROOT: root of an unpacked deposit (round 6)
MP = os.path.join(ROOT, "thesis-med/_paper/method-paper")
MESHES = os.path.join(ROOT, "pulse-ep/data/validation_meshes"); SVR = os.path.join(MP, "supplementary_validation_results")
TAB = os.path.join(MP, "revision_2026-10/tables"); FIGS = os.path.join(MP, "revision_2026-10/figs"); CACHE = os.path.join(FIGS, "_cache")
SELECTION = os.path.join(SVR, "lo_selection_2026-09-14/selection.json")
WINDOW = 42.0; CMAP = "viridis"
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})
C_SC, C_DC = "#d95f02", "#1b9e77"   # single / double chamber (as in the v3-v7 figures)


def gauss(d, A, s, B): return A * np.exp(-d ** 2 / (2 * s ** 2)) + B


# ----------------------------------------------------------------------------- pipeline (v38/v39 code path)
def focality(S, F, d, pct=95):
    thr = np.percentile(np.where(np.isnan(S), 0.0, S), pct); top = np.where(np.isnan(S), 0.0, S) >= thr
    H = np.where(top)[0]; nH = len(H); pos = -np.ones(len(S), int); pos[H] = np.arange(nH)
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); e = e[top[e].all(axis=1)]; e = pos[e]
    G = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(nH, nH)).tocsr()
    nc, lab = connected_components(G, directed=False); sizes = np.bincount(lab)
    labels = -np.ones(len(S), int); labels[H] = lab
    main = int(np.argmax(sizes))
    return float(sizes.max() / nH), int(nc), labels, main


def analyse(mid):
    """Load, run the April pipeline, return a dict (cached as npz)."""
    pf = os.path.join(CACHE, "map_%03d.npz" % mid)
    if os.path.exists(pf):
        z = np.load(pf, allow_pickle=True); return {k: z[k] for k in z.files}
    z = np.load(os.path.join(MESHES, "map_%03d.npz" % mid), allow_pickle=True)
    V0, F0, S0 = z["vertices"], z["triangles"].astype(int), z["scores"].astype(float)
    V, F, idx = largest_component(V0, F0); S = S0[idx >= 0]
    o = find_origin(S, V, F); d = HeatSolver(V, F)(o); c, m, n = bin_profile(d, S); g = fit_gaussian(c, m)
    Fv, ncomp, labels, main = focality(S, F, d)
    sel = {int(s["map_id"]): s for s in json.load(open(SELECTION))} if os.path.exists(SELECTION) else {}  # no marker file (e.g. a map of one's own in the deposit layout): no marker drawn
    mk = -1; snap = np.nan
    if mid in sel and sel[mid]["route"] == "unique_lo_reference_candidate":
        xyz = np.asarray(sel[mid]["selected_xyz_mm"], float); mk = int(np.argmin(np.linalg.norm(V - xyz, axis=1))); snap = float(np.linalg.norm(V[mk] - xyz))
    r = dict(map_id=mid, V=V, F=F, S=S, origin=o, d=d, centres=c, means=m, counts=n, sigma=g["sigma"], r2=g["r2"], A=g["A"], B=g["B"],
             Fidx=Fv, ncomp=ncomp, labels=labels, main_comp=main, marker=mk, snap=snap, edge=mean_edge_length(V, F))
    np.savez_compressed(pf, **r); return r


def halfmax_components(r):
    """Revised focality index (v50): set S >= B + A/2, component of the origin.  Returns F, n_comp, labels (-1 outside the set), origin label."""
    S, F, o, A, B = r["S"], r["F"], int(r["origin"]), float(r["A"]), float(r["B"])
    top = np.where(np.isnan(S), -np.inf, S) >= B + 0.5 * A; H = np.where(top)[0]; nH = len(H); pos = -np.ones(len(S), int); pos[H] = np.arange(nH)
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); e = e[top[e].all(axis=1)]; e = pos[e]
    G = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(nH, nH)).tocsr()
    nc, lab = connected_components(G, directed=False); sizes = np.bincount(lab); labels = -np.ones(len(S), int); labels[H] = lab
    ko = lab[pos[o]] if pos[o] >= 0 else lab[int(np.argmax(S[H]))]
    return float(sizes[ko] / nH), int(nc), labels, int(ko)


def check_against_table(r):
    M = pd.read_csv(os.path.join(TAB, "S12_per_map.csv")).set_index("map_id")
    if int(r["map_id"]) in M.index:
        row = M.loc[int(r["map_id"])]; Fh, nc, _, _ = halfmax_components(r)
        Fp, ncp, _ = focality_index(r["S"], r["F"], int(r["origin"]), float(r["A"]), float(r["B"])); assert abs(Fp - Fh) < 1e-12 and ncp == nc, (r["map_id"], Fp, Fh)  # drawing helper == package
        assert abs(row.sigma - float(r["sigma"])) < 1e-6 and abs(row.r2 - float(r["r2"])) < 1e-6 and abs(row.F_pct - float(r["Fidx"])) < 1e-6 and abs(row.F - Fh) < 1e-6 and int(row.n_components) == nc, (r["map_id"], row.sigma, r["sigma"], row.r2, r["r2"], row.F_pct, r["Fidx"], row.F, Fh)
        return row
    return None


# ----------------------------------------------------------------------------- renderer (after make_fig_surface_v6.py)
def unit(v): return v / np.linalg.norm(v)


def normals(V, T):
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]); fc = V[T].mean(1)
    if np.mean(np.einsum("ij,ij->i", fn, fc - V.mean(0)) > 0) < 0.5: fn = -fn
    vn = np.zeros_like(V)
    for k in range(3): np.add.at(vn, T[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12; fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    return vn, fn


def camera(V, T, o, fn, radius=8.0):
    fc = V[T].mean(1); near = np.linalg.norm(fc - V[o], axis=1) < radius; e3 = unit(fn[near].sum(0))
    up = np.array([0, 0, 1.0]) if abs(e3[2]) < 0.9 else np.array([0, 1.0, 0]); e1 = unit(np.cross(up, e3)); e2 = np.cross(e3, e1)
    return np.stack([e1, e2, e3], 1)


def contours(V, T, d, levels):
    segs, lev, face = [], [], []; dT = d[T]; ok = np.where(np.all(np.isfinite(dT), 1))[0]
    for L in levels:
        for f in ok:
            tri, dv = T[f], dT[f]; pts = []
            for a, b in ((0, 1), (1, 2), (2, 0)):
                da, db = dv[a] - L, dv[b] - L
                if (da < 0) != (db < 0): t = da / (da - db); pts.append(V[tri[a]] + t * (V[tri[b]] - V[tri[a]]))
            if len(pts) == 2: segs.append(pts); lev.append(L); face.append(f)
    return np.array(segs), np.array(lev), np.array(face)


def visible(X, tree, Q, radius=2.5, tol=2.5):
    out = np.zeros(len(Q), bool)
    for i, q in enumerate(Q):
        nb = tree.query_ball_point(q[:2], radius); out[i] = (not nb) or (q[2] >= X[nb, 2].max() - tol)
    return out


def draw_surface(ax, r, field="score", norm=None, cmap=CMAP, levels=(1, 2), window=WINDOW, label=None, show_marker=True, show_components=False, contour_labels=True, scalebar=True):
    """field: 'score' (vertex scores), 'distance' (geodesic distance from the origin), 'grey', or 'components' (high-score set)."""
    V, T, S, o, d, sig = r["V"], r["F"], r["S"], int(r["origin"]), r["d"], float(r["sigma"])
    vn, fn = normals(V, T); R = camera(V, T, o, fn); X = (V - V[o]) @ R; XT = X[T]; order = np.argsort(XT[:, :, 2].mean(1))
    light = unit(np.array([-0.35, 0.45, 1.0])); vshade = 0.40 + 0.60 * np.clip((vn @ R) @ light, 0, 1); shade = vshade[T].mean(1)
    col = np.tile([0.90, 0.90, 0.90, 1.0], (len(T), 1)); col[:, :3] *= shade[:, None]
    if field in ("score", "distance"):
        val = S if field == "score" else d; okf = np.all(np.isfinite(val[T]), 1)
        nm = norm or (mcolors.Normalize(50, 100) if field == "score" else mcolors.Normalize(0, 50))
        cmo = matplotlib.colormaps[cmap if field == "score" else "Blues"]
        sc = cmo(nm(val[T][okf].mean(1))); sc[:, :3] *= (0.88 + 0.12 * shade[okf])[:, None]; col[okf] = sc
    elif field == "components":   # half-maximum set: red = component of the origin, blue = other components
        _, _, lab, main = halfmax_components(r); inH = lab[T].min(1) >= 0
        cc = np.where(lab[T].max(1) == main, 0, 1)
        pal = np.array([[0.85, 0.20, 0.10, 1.0], [0.10, 0.35, 0.85, 1.0]])
        sc = pal[cc[inH]]; sc[:, :3] *= (0.75 + 0.25 * shade[inH])[:, None]; col[inH] = sc
    ax.add_collection(PolyCollection(XT[order][:, :, :2], facecolors=col[order], edgecolors=col[order], linewidths=0.15, rasterized=True))
    tree = cKDTree(X[:, :2])
    if levels:
        L = [k * sig for k in levels] if field != "distance" else list(levels)
        segs, lev, face = contours(V, T, d, L)
        if len(segs):
            Sg = (segs - V[o]) @ R; mid = Sg.mean(1); vis = visible(X, tree, mid) & ((fn[face] @ R)[:, 2] > 0.2)
            for k, Lk in zip(levels, L):
                sel = vis & (lev == Lk)
                ax.add_collection(LineCollection(Sg[sel][:, :, :2], colors="white", linewidths=1.5, zorder=3))
                ax.add_collection(LineCollection(Sg[sel][:, :, :2], colors="black", linewidths=0.55, zorder=4))
                m = mid[sel]
                if contour_labels and len(m):
                    inside = (np.abs(m[:, 0]) < window - 5) & (np.abs(m[:, 1]) < window - 5)
                    if inside.any():
                        th = np.deg2rad(20 + 110 * list(levels).index(k)); dirv = np.array([np.cos(th), np.sin(th)])   # labels spread around the origin
                        j = np.argmax(np.where(inside, m[:, :2] @ dirv, -np.inf)); txt = (fr"{k if k > 1 else ''}$\sigma$" if field != "distance" else f"{int(Lk)} mm"); right = dirv[0] >= 0
                        ax.text(m[j, 0] + (1.0 if right else -1.0), m[j, 1], txt, fontsize=6.5, va="center", ha="left" if right else "right", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85), zorder=6)
    # marked pacing site (triangle) and detected origin (star)
    mk = int(r["marker"])
    if show_marker and mk >= 0:
        Q = X[[mk]] + np.array([0, 0, 0.5]); vis = visible(X, tree, Q)[0]
        ax.plot(Q[0, 0], Q[0, 1], marker="^", ms=7, mfc="white" if vis else "none", mec="black", mew=0.9, ls="none", zorder=7)
    ax.plot(0, 0, marker="*", ms=9, mfc="white", mec="black", mew=0.8, ls="none", zorder=8)
    ax.set_xlim(-window, window); ax.set_ylim(-window, window); ax.set_aspect("equal"); ax.set_axis_off()
    if scalebar:
        x0, y0 = -window + 4, -window + 4
        ax.plot([x0, x0 + 10], [y0, y0], color="black", lw=1.4, zorder=8); ax.text(x0 + 5, y0 + 1.5, "10 mm", ha="center", fontsize=6.5, zorder=8)
    if label: ax.set_title(label, loc="left", fontsize=8)
    return R, X


def profile_panel(ax, r, dmax=None, show_vertices=True, annotate=True):
    d, S, c, m, sig, A, B, r2 = r["d"], r["S"], r["centres"], r["means"], float(r["sigma"]), float(r["A"]), float(r["B"]), float(r["r2"])
    dmax = dmax or min(np.nanmax(d), max(40.0, 3.5 * sig))
    if show_vertices:
        ok = np.isfinite(d) & np.isfinite(S) & (d <= dmax)
        ax.scatter(d[ok], S[ok], s=1.2, c="#bbbbbb", linewidths=0, rasterized=True, zorder=1)
    ax.plot(c[c <= dmax], m[c <= dmax], "o", ms=3.2, color="#1f4e79", zorder=3)
    x = np.linspace(0, dmax, 300); ax.plot(x, gauss(x, A, sig, B), color="#c0392b" if r2 >= 0.7 else "#7f8c8d", lw=1.4, ls="-" if r2 >= 0.7 else "--", zorder=2)
    ax.axvline(sig, color="#c0392b", lw=0.6, ls=":", zorder=0)
    ax.set_xlim(0, dmax); ax.set_xlabel("geodesic distance from the origin (mm)"); ax.set_ylabel("matching score")
    if annotate:
        ax.text(0.97, 0.95, fr"$\sigma$ = {sig:.1f} mm" "\n" fr"$R^2$ = {r2:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.5))


def full_range_inset(ax, r, loc=(0.52, 0.30, 0.45, 0.36)):
    """Inset with the binned profile and the fit over the full geodesic range (far-field explanation of a low R2)."""
    c, m, sig, A, B, r2 = r["centres"], r["means"], float(r["sigma"]), float(r["A"]), float(r["B"]), float(r["r2"])
    ia = ax.inset_axes(loc); ia.plot(c, m, "o", ms=1.6, color="#1f4e79"); x = np.linspace(0, c.max(), 300)
    ia.plot(x, gauss(x, A, sig, B), color="#c0392b" if r2 >= 0.7 else "#7f8c8d", lw=0.9, ls="-" if r2 >= 0.7 else "--")
    ia.axvspan(0, 3 * sig, color="#f3d9d6", lw=0, zorder=0); ia.set_xlim(0, c.max()); ia.tick_params(labelsize=5, length=2, pad=1)
    ia.set_title("full range (%.0f mm); shaded: $\\leq 3\\sigma$" % c.max(), fontsize=5.5, pad=2); return ia


def save(fig, name, png=True):
    os.makedirs(FIGS, exist_ok=True); fig.savefig(os.path.join(FIGS, name + ".pdf"), dpi=300)
    if png: fig.savefig(os.path.join(FIGS, name + ".png"), dpi=150)
    plt.close(fig); print("saved", name, flush=True)


# ----------------------------------------------------------------------------- figures
def fig_test(mid):
    r = analyse(mid); row = check_against_table(r)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
    draw_surface(ax[0], r, "score", label=f"map {mid}: score, $\\sigma$ = {float(r['sigma']):.1f} mm")
    draw_surface(ax[1], r, "distance", levels=(10, 20, 30), label="geodesic distance (mm)")
    profile_panel(ax[2], r)
    save(fig, "_test_map%03d" % mid)


META = None
def meta(mid):
    global META
    if META is None: META = pd.read_csv(os.path.join(TAB, "S12_per_map.csv")).set_index("map_id")
    row = META.loc[mid]; return dict(region="%s %s" % (row.atrium, row.region), ref="%s-chamber reference" % row.reference, patient=int(row.patient), sigma=row.sigma, r2=row.r2, F=row.F, n=int(row.n_points))


def fig_heatmap(maps=(75, 78)):  # 29 Oct: map 25 dropped (shown in the clinical-example figure)
    NAME = "fig_heatmap"
    """Fig. 4 (revised): vertical stack, single-hue sequential colour map, marker, origin, 1-sigma and 2-sigma geodesic contours."""
    rs = [analyse(m) for m in maps]; [check_against_table(r) for r in rs]
    norm = mcolors.Normalize(50, 100)
    fig = plt.figure(figsize=(3.4, 0.6 + 2.8 * len(maps))); gs = fig.add_gridspec(len(maps), 1, left=0.02, right=0.80, top=1 - 0.3 / (0.6 + 2.8 * len(maps)), bottom=0.62 / (0.6 + 2.8 * len(maps)), hspace=0.18)  # panels are width-limited (equal aspect); 8.4 in keeps them square
    lines = []
    for k, (m, r) in enumerate(zip(maps, rs)):
        ax = fig.add_subplot(gs[k, 0]); md = meta(m)
        lab = "%s  %s, %s\n$\sigma$ = %.1f mm, $R^2$ = %.2f%s" % ("ABC"[k], md["region"], md["ref"].replace(" reference", ""), md["sigma"], md["r2"], "  (diffuse)" if md["sigma"] >= 50 else ("  (broad)" if md["sigma"] >= 30 else ""))
        draw_surface(ax, r, "score", norm=norm, label=lab, window=max(WINDOW, min(70.0, 1.6 * md["sigma"])))
        lines.append("Fig. 4 %s: map %d, patient %d, %s, %s, n_points %d, sigma %.2f, R2 %.3f, F %.2f, marker offset %s" % ("ABC"[k], m, md["patient"], md["region"], md["ref"], md["n"], md["sigma"], md["r2"], md["F"], "%.1f mm" % META.loc[m].offset_marker_mm if np.isfinite(META.loc[m].offset_marker_mm) else "none"))
    cax = fig.add_axes([0.84, 0.32, 0.03, 0.36]); cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax); cb.set_label("matching score (points)", fontsize=7); cb.ax.tick_params(labelsize=7)
    h = [plt.Line2D([], [], marker="*", ms=8, mfc="white", mec="black", ls="none", label="vertex of maximum score (origin)"), plt.Line2D([], [], marker="^", ms=6, mfc="white", mec="black", ls="none", label="pacing site marked during acquisition"),
         plt.Line2D([], [], color="black", lw=0.6, label="geodesic distance $\sigma$ and $2\sigma$ from the origin")]
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=1, fontsize=6.5, frameon=False)
    save(fig, "fig_heatmap"); open(os.path.join(FIGS, "captions_" + NAME + ".txt"), "w").write("\n".join(lines) + "\n")


def fig_profiles(maps=(25, 54, 32)):
    NAME = "fig_decay_profile"
    """Fig. 2 (revised): radial decay profiles of the three maps of the submission, vertical stack."""
    rs = [analyse(m) for m in maps]; [check_against_table(r) for r in rs]
    fig, axs = plt.subplots(len(maps), 1, figsize=(3.4, 7.4), sharex=False)
    lines = []
    for k, (m, r, ax) in enumerate(zip(maps, rs, axs)):
        md = meta(m); profile_panel(ax, r, dmax=40.0)
        if md["r2"] < 0.7: full_range_inset(ax, r)
        kind = ("flagged, $R^2 < 0.7$" if md["r2"] < .7 else "not identifiable" if md["sigma"] >= 100 else "diffuse" if md["sigma"] >= 50 else "broad" if md["sigma"] >= 30
                else "narrow" if md["sigma"] < 10 else "in range")  # labels follow the width classes (2-30 in range, 30-50 broad, 50-100 diffuse)
        ax.set_title("%s  %s, %s: %s" % ("ABC"[k], md["region"], md["ref"].replace(" reference", ""), kind), loc="left", fontsize=8)
        if k < len(maps) - 1: ax.set_xlabel("")
        lines.append("Fig. 2 %s: map %d, patient %d, %s, %s, sigma %.2f, R2 %.3f, A %.1f, B %.1f, n_bins %d" % ("ABC"[k], m, md["patient"], md["region"], md["ref"], md["sigma"], md["r2"], float(r["A"]), float(r["B"]), len(r["centres"])))
    fig.tight_layout(h_pad=1.0); save(fig, "figS_decay_profile"); open(os.path.join(FIGS, "captions_" + NAME + ".txt"), "w").write("\n".join(lines) + "\n")


def fig_example(maps=(1, 25)):  # 29 Oct: map 28 dropped (it is the workflow map of Fig. 1)
    NAME = "fig_clinical_example"
    """Clinical example (new): three maps x four columns: score map | geodesic distance | profile with fit | high-score set and F."""
    rs = [analyse(m) for m in maps]; [check_against_table(r) for r in rs]
    norm = mcolors.Normalize(50, 100)
    H = 1.15 + 2.15 * len(maps); fig = plt.figure(figsize=(7.2, H)); gs = fig.add_gridspec(len(maps), 4, width_ratios=[1, 1, 1.25, 1], left=0.01, right=0.99, top=1 - 0.5 / H, bottom=1.05 / H, wspace=0.16, hspace=0.22)
    lines = []
    for j, h in enumerate(["score map", "geodesic distance", "radial profile, Gaussian fit", "half-maximum set, $F$"]):
        fig.text([0.11, 0.33, 0.60, 0.88][j], 1 - 0.36 / H, h, ha="center", va="bottom", fontsize=8, fontweight="bold")
    for k, (m, r) in enumerate(zip(maps, rs)):
        md = meta(m); L = "ABC"[k]
        a0 = fig.add_subplot(gs[k, 0]); draw_surface(a0, r, "score", norm=norm, label="%s  %s, %s" % (L, md["region"], md["ref"].replace(" reference", "")))
        a1 = fig.add_subplot(gs[k, 1]); draw_surface(a1, r, "distance", levels=(10, 20), show_marker=False, scalebar=False)
        a2 = fig.add_subplot(gs[k, 2]); profile_panel(a2, r, dmax=40.0 if md["sigma"] < 15 else 80.0, annotate=False)
        if md["r2"] < 0.7: full_range_inset(a2, r, loc=(0.45, 0.42, 0.52, 0.5))
        a2.set_ylabel("score", labelpad=1); a2.set_xlabel("geodesic distance (mm)" if k == len(maps) - 1 else "")
        a3 = fig.add_subplot(gs[k, 3]); draw_surface(a3, r, "components", levels=(), show_marker=False, scalebar=False)
        flag = []
        if md["r2"] < 0.7: flag.append("$R^2 < 0.7$")
        if md["F"] < 0.70 and md["r2"] >= 0.7: flag.append("$F < 0.70$")  # F is interpreted only when R2 >= 0.7
        ftxt = "$F$ = %.2f (%d comp.)%s" % (md["F"], int(META.loc[m].n_components), "" if md["r2"] >= 0.7 else ", not interpreted")
        txt = "$\sigma$ = %.1f mm%s\n$R^2$ = %.2f\n%s%s" % (md["sigma"], " (diffuse)" if md["sigma"] >= 50 else (" (broad)" if md["sigma"] >= 30 else ""), md["r2"], ftxt, ("\nflag: " + ", ".join(flag)) if flag else "\nno flag")
        a3.text(0.02, 0.02, txt, transform=a3.transAxes, fontsize=6.5, va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.5), zorder=9)
        lines.append("Fig. example %s: map %d, patient %d, %s, %s, sigma %.2f, R2 %.3f, F %.2f (half-max set, %d comp.; F_pct %.2f), marker offset %s" % (L, m, md["patient"], md["region"], md["ref"], md["sigma"], md["r2"], md["F"], int(META.loc[m].n_components), META.loc[m].F_pct, "%.1f mm" % META.loc[m].offset_marker_mm if np.isfinite(META.loc[m].offset_marker_mm) else "none"))
    cax = fig.add_axes([0.035, 0.72 / H, 0.16, 0.085 / H]); cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal"); cb.set_label("matching score (points)", fontsize=6.5); cb.ax.tick_params(labelsize=6.5)
    cax2 = fig.add_axes([0.26, 0.72 / H, 0.16, 0.085 / H]); cb2 = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(0, 50), cmap="Blues"), cax=cax2, orientation="horizontal"); cb2.set_label("geodesic distance (mm)", fontsize=6.5); cb2.ax.tick_params(labelsize=6.5)
    h = [plt.Line2D([], [], marker="*", ms=8, mfc="white", mec="black", ls="none", label="origin"), plt.Line2D([], [], marker="^", ms=6, mfc="white", mec="black", ls="none", label="marked pacing site"),
         plt.Line2D([], [], color="black", lw=0.6, label="$\sigma$, $2\sigma$ contours"), plt.Line2D([], [], marker="o", ms=3.5, color="#1f4e79", ls="none", label="2-mm bin means"), plt.Line2D([], [], color="#c0392b", lw=1.4, label="Gaussian fit"),
         plt.Rectangle((0, 0), 1, 1, fc=(0.85, 0.20, 0.10), label="half-maximum set, component of the origin"), plt.Rectangle((0, 0), 1, 1, fc=(0.10, 0.35, 0.85), label="other components")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=6.5, frameon=False, columnspacing=1.2)
    save(fig, "fig_clinical_example"); open(os.path.join(FIGS, "captions_" + NAME + ".txt"), "w").write("\n".join(lines) + "\n")


def fig_workflow(mid=28):
    """Workflow (new Fig. 1): six panels on one clinical map."""
    r = analyse(mid); md = meta(mid); norm = mcolors.Normalize(50, 100)
    V, T, S, o, d = r["V"], r["F"], r["S"], int(r["origin"]), r["d"]
    fig = plt.figure(figsize=(7.2, 4.6)); gs = fig.add_gridspec(2, 3, left=0.04, right=0.97, top=0.93, bottom=0.10, wspace=0.30, hspace=0.42)
    # (a) mesh with vertex score field
    a = fig.add_subplot(gs[0, 0]); draw_surface(a, r, "score", norm=norm, levels=(), label="a  triangulated surface with vertex scores", show_marker=False, window=50)
    # (b) origin: zoom with wireframe
    b = fig.add_subplot(gs[0, 1]); R, X = draw_surface(b, r, "score", norm=norm, levels=(), label="b  origin: vertex of maximum score", show_marker=True, window=8, scalebar=False)
    XT = X[T]; near = np.all(np.abs(XT[:, :, :2]) < 9, axis=(1, 2)); vn, fn = normals(V, T); front = (fn @ R)[:, 2] > 0
    b.add_collection(LineCollection(np.concatenate([XT[near & front][:, [0, 1], :2], XT[near & front][:, [1, 2], :2], XT[near & front][:, [2, 0], :2]]), colors="black", linewidths=0.25, alpha=0.6, zorder=2))
    b.plot([-7, -2], [-7, -7], color="black", lw=1.4); b.text(-4.5, -6.4, "5 mm", ha="center", fontsize=6.5)
    # (c) geodesic distance field
    c = fig.add_subplot(gs[0, 2]); draw_surface(c, r, "distance", levels=(10, 20, 30), label="c  geodesic distance from the origin (Heat Method)", show_marker=False, window=50)
    # (d) radial binning
    dd = fig.add_subplot(gs[1, 0]); ok = np.isfinite(d) & (d <= 40)
    dd.scatter(d[ok], S[ok], s=1.0, c="#bbbbbb", linewidths=0, rasterized=True); cc, mm = r["centres"], r["means"]; sel = cc <= 40
    for x0 in np.arange(0, 42, 2): dd.axvline(x0, color="#eeeeee", lw=0.4, zorder=0)
    dd.plot(cc[sel], mm[sel], "o", ms=3, color="#1f4e79"); dd.set_xlim(0, 40); dd.set_xlabel("geodesic distance (mm)"); dd.set_ylabel("matching score"); dd.set_title("d  radial binning (2 mm, $\geq$ 3 vertices)", loc="left", fontsize=8)
    # (e) Gaussian fit
    e = fig.add_subplot(gs[1, 1]); profile_panel(e, r, dmax=40.0, show_vertices=False, annotate=False); e.set_title("e  three-parameter Gaussian fit", loc="left", fontsize=8); e.set_xlabel("geodesic distance (mm)")
    A, B, sg = float(r["A"]), float(r["B"]), float(r["sigma"])
    e.annotate("", xy=(sg, gauss(sg, A, sg, B)), xytext=(sg, B), arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.6)); e.text(sg + 0.8, B + 0.3 * A, "$\sigma$ = %.1f mm" % sg, fontsize=7, color="#c0392b")
    e.annotate("", xy=(30, B + A), xytext=(30, B), arrowprops=dict(arrowstyle="<->", color="#555555", lw=0.6)); e.text(30.8, B + A / 2, "$A$", fontsize=7, color="#555555"); e.text(36, B + 0.6, "$B$", fontsize=7, color="#555555")
    # (f) outputs and flags: high-score set
    f = fig.add_subplot(gs[1, 2]); draw_surface(f, r, "components", levels=(), label="f  half-maximum set; outputs $\sigma$, $R^2$, $F$", show_marker=False, scalebar=False)
    f.text(0.02, 0.02, "$\sigma$ = %.1f mm\n$R^2$ = %.2f (flag if < 0.7)\n$F$ = %.2f (flag if < 0.70)" % (md["sigma"], md["r2"], md["F"]), transform=f.transAxes, fontsize=6.5, va="bottom", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.5), zorder=9)
    cax = a.inset_axes([1.0, 0.25, 0.04, 0.5]); cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax); cb.set_label("score", fontsize=6); cb.ax.tick_params(labelsize=6)
    cax2 = c.inset_axes([1.0, 0.25, 0.04, 0.5]); cb2 = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(0, 50), cmap="Blues"), cax=cax2); cb2.set_label("mm", fontsize=6); cb2.ax.tick_params(labelsize=6)
    save(fig, "fig_workflow"); open(os.path.join(FIGS, "captions_fig_workflow.txt"), "w").write("Fig. workflow: map %d, patient %d, %s, %s, sigma %.2f, R2 %.3f, F %.2f, A %.1f, B %.1f, N %d vertices, edge %.2f mm\n" % (mid, md["patient"], md["region"], md["ref"], md["sigma"], md["r2"], md["F"], float(r["A"]), float(r["B"]), len(V), float(r["edge"])))


NOISE_C = {0.0: "#1f4e79", 0.05: "#1b9e77", 0.10: "#d95f02", 0.15: "#c0392b"}


def fig_validation():
    """Fig. 4 (reader pass 27 Oct): primary protocol (v56, noise-free peak 90, detected origin): bias and SD vs sigma0; range 2-50 mm at 10 % noise."""
    v = pd.read_csv(os.path.join(SVR, "v56_peak_below_clip_all_2026-10-26/v56_raw.csv")); rng = np.random.default_rng(0); out = []
    for (s0, eta), g in v.groupby(["sigma0", "noise"]):
        b = (g.sigma_plateau - s0).values; m = np.array([rng.choice(b, len(b)).mean() for _ in range(2000)])
        out.append(dict(sigma0=s0, noise=round(eta, 2), bias=b.mean(), lo=np.percentile(m, 2.5), hi=np.percentile(m, 97.5), sd=g.sigma_plateau.std(), med=np.median(b), p5=np.percentile(g.sigma_plateau, 5) - s0, p95=np.percentile(g.sigma_plateau, 95) - s0))
    S = pd.DataFrame(out); fig, ax = plt.subplots(3, 1, figsize=(3.4, 7.1))
    for eta in (0.05, 0.10, 0.15):
        a = S[(S.noise == eta) & (S.sigma0 >= 3) & (S.sigma0 <= 30)].sort_values("sigma0"); c = NOISE_C[eta]
        ax[0].fill_between(a.sigma0, a.lo, a.hi, color=c, alpha=0.15, lw=0); ax[0].plot(a.sigma0, a.bias, "o-", color=c, ms=3.5, lw=1.2, label="%.0f %%" % (100 * eta))
        ax[1].plot(a.sigma0, a.sd, "o-", color=c, ms=3.5, lw=1.2, label="%.0f %%" % (100 * eta))
    ax[0].axhline(0, color="#999999", lw=0.8); ax[0].set_ylabel("bias $\\hat\\sigma-\\sigma_0$ (mm)"); ax[0].set_title("A  bias (mean, 95 % bootstrap CI)", loc="left", fontsize=8)
    ax[0].legend(title="vertex noise", fontsize=6.5, title_fontsize=6.5, loc="upper left", frameon=False, ncol=3)
    ax[1].set_ylabel("SD of $\\hat\\sigma$ (mm)"); ax[1].set_title("B  precision", loc="left", fontsize=8); ax[1].set_yscale("log"); ax[1].set_yticks([0.2, 0.5, 1, 2]); ax[1].set_yticklabels(["0.2", "0.5", "1", "2"]); ax[1].minorticks_off()
    for a_ in ax[:2]: a_.set_xlim(0, 32); a_.set_xticks([3, 5, 8, 10, 15, 20, 25, 30]); a_.set_xlabel("true width $\\sigma_0$ (mm)")
    r = S[S.noise == 0.10].sort_values("sigma0"); ax[2].fill_between(r.sigma0, r.p5, r.p95, color=NOISE_C[0.10], alpha=0.15, lw=0)
    ax[2].plot(r.sigma0, r.med, "o-", color=NOISE_C[0.10], ms=3.5, lw=1.2, label="median bias, 5th–95th percentile")
    ax[2].axhline(0, color="#999999", lw=0.8); ax[2].axvspan(2, 30, color="#1b9e77", alpha=0.08, lw=0); ax[2].axvspan(30, 50, color="#d95f02", alpha=0.08, lw=0)
    ax[2].text(7.7, 11, "in range\n2–30 mm", fontsize=6.5, ha="center", color="#1b9e77"); ax[2].text(38.5, 11, "broad\n30–50", fontsize=6.5, ha="center", color="#d95f02")
    ax[2].set_xscale("log"); ax[2].set_xlim(1.8, 55); ax[2].set_xticks([2, 3, 5, 10, 20, 30, 50]); ax[2].set_xticklabels(["2", "3", "5", "10", "20", "30", "50"]); ax[2].set_ylim(-16, 16)
    ax[2].set_xlabel("true width $\\sigma_0$ (mm)"); ax[2].set_ylabel("$\\hat\\sigma-\\sigma_0$ (mm)"); ax[2].set_title("C  operating range at 10 % noise", loc="left", fontsize=8); ax[2].legend(fontsize=6, loc="lower left", frameon=False, handlelength=1.6)
    fig.tight_layout(h_pad=1.2); save(fig, "fig_validation")


def fig_boxplot():
    """Fig. 5 (revised): width by region and reference; maps as points, patient medians as diamonds; 10-mm line."""
    M = pd.read_csv(os.path.join(TAB, "S12_per_map.csv")); M["grp"] = M.atrium + " " + M.region
    order = ["LA posterior", "LA anterior", "RA posterior", "RA lateral"]; fig, ax = plt.subplots(figsize=(3.4, 3.2)); rng = np.random.default_rng(1)
    for i, g in enumerate(order):
        for j, (ref, c) in enumerate((("double", C_DC), ("single", C_SC))):
            d = M[(M.grp == g) & (M.reference == ref)]; x = i + (-0.2 if ref == "double" else 0.2)
            if len(d) == 0: continue
            ax.boxplot([d.sigma.values], positions=[x], widths=0.3, showfliers=False, patch_artist=True, boxprops=dict(facecolor=c, alpha=0.25, lw=0.6), medianprops=dict(color=c, lw=1.4), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
            ax.scatter(x + rng.uniform(-0.09, 0.09, len(d)), d.sigma, s=7, color=c, alpha=0.8, linewidths=0, zorder=3)
            pm = d.groupby("patient").sigma.median(); ax.scatter(np.full(len(pm), x), pm, marker="D", s=14, facecolor="white", edgecolor=c, linewidths=0.8, zorder=4)
    ax.axhline(10, color="#999999", lw=0.8, ls="--"); ax.set_yscale("log"); ax.set_yticks([2, 5, 10, 20, 50]); ax.set_yticklabels(["2", "5", "10", "20", "50"]); ax.set_ylabel("width $\\sigma$ (mm)")
    ax.set_xticks(range(4)); ax.set_xticklabels(order, fontsize=7); ax.set_xlim(-0.6, 3.6)
    h = [plt.Line2D([], [], marker="o", ls="none", color=C_DC, ms=4, label="double-chamber reference"), plt.Line2D([], [], marker="o", ls="none", color=C_SC, ms=4, label="single-chamber reference"), plt.Line2D([], [], marker="D", ls="none", mfc="white", mec="#555555", ms=4, label="patient median")]
    ax.legend(handles=h, fontsize=6.5, loc="upper left", frameon=False); fig.tight_layout(); save(fig, "figS_boxplot_sigma")


def fig_supp():
    # Fig. S1: analytic surfaces
    d = pd.read_csv(os.path.join(SVR, "v36_analytic_geodesic_2026-11-06b/v36_distance_errors.csv")); surf = sorted(d.surface.unique(), key=lambda x: (x.startswith("cyl"), x))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8)); col = {"heat": "#d95f02", "exact": "#1b9e77", "dijkstra": "#7570b3"}
    for j, (metric, lab) in enumerate((("rel_err_mean", "signed relative error (%)"), ("rel_err_p95", "95th percentile of |relative error| (%)"))):
        for k, sv in enumerate(("heat", "exact", "dijkstra")):
            g = d[d.solver == sv].groupby("surface")[metric].mean().reindex(surf); ax[j].bar(np.arange(len(surf)) + (k - 1) * 0.27, 100 * g.values, 0.27, color=col[sv], label=sv)
        ax[j].set_xticks(range(len(surf))); ax[j].set_xticklabels([x.replace("_", " ").replace("sphere sub", "sphere ") for x in surf], rotation=30, ha="right", fontsize=6.5); ax[j].set_ylabel(lab); ax[j].axhline(0, color="#999999", lw=0.6)
    ax[0].legend(fontsize=6.5, frameon=False); fig.tight_layout(); save(fig, "figS_analytic_surfaces")
    # Fig. S3: structured deviations
    vb = pd.read_csv(os.path.join(SVR, "v47b_structured_noise_plateau_2026-10-26/v47b_summary.csv")); fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6)); ls = {5.0: ":", 10.0: "-", 15.0: "--"}; kc = {"bumps": "#7570b3", "corr5": "#1b9e77", "corr10": "#d95f02"}
    for (kind, s0), g in vb.groupby(["kind", "sigma0"]):
        g = g.sort_values("amp"); ax[0].plot(g.amp, g.r2_med, ls[s0], color=kc[kind], marker="o", ms=3); ax[1].plot(g.amp, 100 * g.flag_r2, ls[s0], color=kc[kind], marker="o", ms=3); ax[2].plot(g.amp, 100 * g.frac_off_gt10, ls[s0], color=kc[kind], marker="o", ms=3)
    r2_clin = float(pd.read_csv(os.path.join(TAB, "S12_per_map.csv")).r2.median()); ax[0].axhline(r2_clin, color="#555555", lw=0.8); ax[0].text(0.11, r2_clin + 0.006, "clinical median", fontsize=6.5); ax[1].axhline(100 * 13 / 55, color="#555555", lw=0.8); ax[1].text(0.11, 25, "clinical", fontsize=6.5)
    for a_, yl in zip(ax, ("median $R^2$", "fits with $R^2<0.7$ (%)", "origin displaced $>$10 mm (%)")): a_.set_xlabel("amplitude of the deviation (units of $A$)"); a_.set_ylabel(yl)
    h = [plt.Line2D([], [], color=kc[k], label={"bumps": "bumps + 10 % noise", "corr5": "correlated, 5 mm", "corr10": "correlated, 10 mm"}[k]) for k in ("bumps", "corr5", "corr10")] + [plt.Line2D([], [], color="#555555", ls=ls[s0], label="$\\sigma_0$ = %g mm" % s0) for s0 in (5.0, 10.0, 15.0)]
    ax[2].legend(handles=h, fontsize=6, frameon=False, loc="lower right"); fig.tight_layout(); save(fig, "figS_structured_deviations")
    # Fig. S5: runtime
    v41 = pd.read_csv(os.path.join(SVR, "v41_runtime_sparsity_2026-10-21/v41_per_mesh.csv")); fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for col_, lab, c in (("t_factor_s", "solver construction", "#d95f02"), ("t_query_s", "single-source query", "#1b9e77")):
        p = np.polyfit(np.log(v41.n_vertices), np.log(v41[col_]), 1); x = np.array([1000, 50000]); ax.loglog(v41.n_vertices, v41[col_], "o", ms=3, color=c, label="%s ($N^{%.2f}$)" % (lab, p[0])); ax.loglog(x, np.exp(p[1]) * x ** p[0], "-", color=c, lw=0.8)
    ax.set_xlabel("vertices $N$"); ax.set_ylabel("time (s)"); ax.legend(fontsize=6.5, frameon=False); fig.tight_layout(); save(fig, "figS_runtime")
    # Fig. 7 (main text since 27 Oct): marker check, linear distance axis so that zero offsets are visible
    v46 = pd.read_csv(os.path.join(SVR, "v46_lo_origin_plateau_2026-10-26/v46_clinical.csv")); v46 = v46[v46.route == "unique_lo_reference_candidate"]; fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.axhline(10, color="#999999", lw=0.8, ls="--"); ax.axvline(10, color="#999999", lw=0.8, ls=":")
    for flag, c, lab in ((False, "#1f4e79", "$R^2 \\geq 0.7$"), (True, "#c0392b", "$R^2 < 0.7$")):
        g = v46[(v46.r2_argmax < 0.7) == flag]; ax.scatter(g.sigma_argmax, g.offset_geo_mm, s=16, color=c, alpha=0.85, linewidths=0, label=lab, zorder=3, clip_on=False)
    for _, r_ in v46[(v46.sigma_argmax < 10) & (v46.offset_geo_mm > 10)].iterrows(): ax.annotate(str(int(r_.map_id)), (r_.sigma_argmax, r_.offset_geo_mm), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xticks([2, 5, 10, 20, 50]); ax.set_xticklabels(["2", "5", "10", "20", "50"]); ax.set_ylim(0, None)
    # group summaries written into the figure (29 Oct 2026), computed from the same rows as the points
    import matplotlib.transforms as mtrans; tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    for lo_, hi_, xc, ha_ in ((0, 10, 3.6, "center"), (10, 20, 14.2, "center"), (20, np.inf, 68.0, "right")):
        g = v46[(v46.sigma_argmax >= lo_) & (v46.sigma_argmax < hi_)]
        txt = ("%d of %d\nwithin 10 mm" % (int((g.offset_geo_mm <= 10).sum()), len(g))) if hi_ == 10 else ("%d maps\nmedian %.0f mm" % (len(g), g.offset_geo_mm.median()) if lo_ == 20 else "%d maps\nmedian %.1f mm" % (len(g), g.offset_geo_mm.median()))
        ax.text(xc, 0.985, txt, transform=tr, ha=ha_, va="top", fontsize=6.2, color="#333333")
    ax.set_ylim(0, 1.2 * float(v46.offset_geo_mm.max()))
    ax.set_xlabel("width $\\sigma$ (mm)"); ax.set_ylabel("score maximum to marked pacing site (mm)"); ax.legend(fontsize=6.5, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62)); fig.tight_layout(); save(fig, "fig_marker_check")
    # Fig. S2: anisotropy on the sphere
    v44 = pd.read_csv(os.path.join(SVR, "v60_anisotropy_pkgfit_2026-10-26/v60_rows.csv")); A_ = v44[v44.part == "A"]; fig, ax = plt.subplots(figsize=(3.4, 2.8))  # v60 = source of Table S7 (round 6)
    for (gm, eta), g in A_.groupby(["gm", "eta"]):
        q = g.groupby("ratio").sigma_hat.agg(["median", lambda x: x.quantile(.05), lambda x: x.quantile(.95)]); q.columns = ["med", "lo", "hi"]
        c = "#1f4e79" if gm == 5 else "#d95f02"; ax.errorbar(q.index, q.med / gm, yerr=[(q.med - q.lo) / gm, (q.hi - q.med) / gm], fmt="o" + ("-" if eta == 0 else "--"), color=c, ms=3.5, mfc="white" if eta > 0 else c, lw=1, capsize=2, label="$\\sqrt{\\sigma_a\\sigma_b}$ = %g mm, %.0f %% noise" % (gm, 100 * eta))
    ax.axhline(1, color="#999999", lw=0.8); ax.set_xlabel("axis ratio $\\sigma_a/\\sigma_b$"); ax.set_ylabel("$\\hat\\sigma / \\sqrt{\\sigma_a\\sigma_b}$"); ax.legend(fontsize=6, frameon=False); fig.tight_layout(); save(fig, "figS_anisotropy_sphere")
    # Fig. S4: focality synthetic (v62, as Table S11b): two-peak detection matrices at 10 % noise; F: flag as in the pipeline (F < 0.70 with R2 >= 0.7) among
    # fields with F available; F_pct: F_pct < 0.70
    v50 = pd.read_csv(os.path.join(SVR, "v62_focality_unified_2026-10-26/v62_sphere.csv")); tp = v50[(v50.kind == "two-peak") & (v50.amp == 0.10)]; fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for j, (col_, lab) in enumerate((("F", "$F$ (half-maximum set)"), ("F_pct", "percentile index $F_{\\rm pct}$ (95th-percentile set)"))):
        m = (tp[tp.F.notna()].assign(fl=lambda d: 100.0 * ((d.F < .7) & (d.r2 >= .7))).groupby(["D", "ratio"]).fl.mean() if col_ == "F" else tp.groupby(["D", "ratio"])[col_].apply(lambda x: 100 * (x < .7).mean())).unstack(); im = ax[j].imshow(m.values, cmap="Reds", vmin=0, vmax=100, aspect="auto", origin="lower")
        ax[j].set_xticks(range(len(m.columns))); ax[j].set_xticklabels(["%.1f" % x for x in m.columns]); ax[j].set_yticks(range(len(m.index))); ax[j].set_yticklabels(["%d" % x for x in m.index]); ax[j].set_xlabel("amplitude ratio of the second peak"); ax[j].set_ylabel("peak separation (mm)"); ax[j].set_title(lab, loc="left", fontsize=8)
        for (ii, jj), v in np.ndenumerate(m.values): ax[j].text(jj, ii, "%.0f" % v, ha="center", va="center", fontsize=6.5, color="white" if v > 60 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("fields flagged (%)", fontsize=7); cb.ax.tick_params(labelsize=6.5); save(fig, "figS_focality_synthetic")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cache":
        for m in sys.argv[2:]: r = analyse(int(m)); print(m, "sigma %.2f R2 %.3f F %.2f marker %d" % (float(r["sigma"]), float(r["r2"]), float(r["Fidx"]), int(r["marker"])), flush=True)
    elif cmd == "test": fig_test(int(sys.argv[2]))
    elif cmd == "heatmap": fig_heatmap()
    elif cmd == "profiles": fig_profiles()
    elif cmd == "example": fig_example()
    elif cmd == "workflow": fig_workflow()
    elif cmd == "validation": fig_validation()
    elif cmd == "boxplot": fig_boxplot()
    elif cmd == "supp": fig_supp()
    elif cmd == "data":  # figures rebuilt from deposited result files only (no clinical meshes): Fig. 4, 6, S1-S6 (round 6)
        for f in (fig_validation, fig_boxplot, fig_supp): f()
    elif cmd == "all":
        for f in (fig_heatmap, fig_profiles, fig_example, fig_workflow, fig_validation, fig_boxplot, fig_supp): f()
