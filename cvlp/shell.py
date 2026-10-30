"""Radial geometry of the capsid and the T number, measured on the I1 map.

Generalises the copia section A/B measurements (teact find_shell_gap,
site_profiles) to capsids of any size and to capsids with no inner layer, and
adds the lattice test that picks T (and the hand of a skewed lattice) when it
is not given.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from . import lattice as L
from . import cutout as cut, geometry as geo, mapio


def find_layers(path, min_radius=50.0, prominence_frac=0.10, min_depth=0.25, gap_radius=None):
    """Inner layer, gap and outer (capsomer) layer radii.

    With an inner layer (copia's RNP) this is teact.find_shell_gap: the gap is
    the valley between the two outermost peaks. A peak inside the capsid only
    counts as an inner layer if the valley drops at least `min_depth` of the
    way from it to the solvent level; ripples inside one thick shell do not.
    Without an inner layer, the gap is put 10 A inside the capsid's inner
    half-maximum, so the mask's 20 A inner ramp ends where the capsid starts.
    `gap_radius` overrides the measurement.
    """
    radii, mean = cut.radial_profile(path)
    pk, _ = find_peaks(mean, prominence=prominence_frac * np.ptp(mean))
    pk = pk[radii[pk] > min_radius]
    if len(pk) == 0:
        raise RuntimeError("no capsid layer in the radial profile")
    outer = pk[np.argmax(radii[pk])]
    solvent = float(np.median(mean[radii > radii[outer] + 0.5 * (radii[-1] - radii[outer])]))
    res = dict(outer=float(radii[outer]), radii=radii, profile=mean, solvent=solvent,
               inner=None, has_inner=False)
    for inner in pk[radii[pk] < radii[outer]][::-1]:
        v = inner + int(np.argmin(mean[inner:outer]))
        if (mean[inner] - mean[v]) >= min_depth * (mean[inner] - solvent):
            res.update(inner=float(radii[inner]), gap=float(radii[v]), has_inner=True)
            break
    if not res["has_inner"]:
        lvl = solvent + 0.5 * (mean[outer] - solvent)
        j = outer
        while j > 0 and mean[j] > lvl:
            j -= 1
        res["gap"] = float(radii[j] - 10.0)
        res["filled"] = j == 0                  # density all the way in: nothing to cut
    if gap_radius is not None:
        res["gap"] = float(gap_radius)
    return res


# (name, where the density maxima should be, blur / spacing, peak neighbourhood / theta, tolerance / theta)
MORPHOLOGIES = (
    ("capsomers", lambda m: m["directions"], 0.2, 0.5, 0.25),    # copia, papilloma, most capsids
    ("dimers", L.edge_midpoints, 0.1, 0.25, 0.125),              # HBV, TCV, e15: spikes on the 2-folds
    ("trimers", L.holes, 0.1, 0.25, 0.125),                      # birnavirus: trimers in the lattice gaps
)


SURFACE_ORDERS = (6, 10, 12)       # the icosahedral harmonics: l=6 is the vertex-up/face-down mode


def ico_harmonic(dirs, l):
    """The (unique) degree-l icosahedral invariant: a sum of Legendre P_l
    centred on the 12 vertices. Zero mean over the sphere for l > 0."""
    from scipy.special import eval_legendre
    return eval_legendre(l, dirs @ geo.icosahedron()[0].T).sum(1)


def surface_model(path, r_lo, n_dirs=3000, orders=SURFACE_ORDERS):
    """The capsid's inner surface as a smooth icosahedral shape r_in(u).

    Faceted capsids are icosahedra, not spheres: the T7 phage head (EMD-1810)
    sits at ~300 A on the 5-fold vertices and ~270 A in the face centres, much
    more than its 10 A shell is thick. A band at one radius then sees only the
    vertices. Per direction, the inner edge of the capsid layer is where the
    radial profile first reaches half its maximum above the solvent; that edge
    is fitted with the lowest icosahedral harmonics only, so the facets come
    through but single capsomers do not (they protrude outwards, and the inner
    edge is the side they do not move). -> dict(coef, orders, mean, vertex, face).
    """
    vol, apix, c = mapio.load_map(path)
    dirs = geo.sphere_points(n_dirs)
    rs = np.arange(max(r_lo, apix), 0.95 * (vol.shape[0] - 1) / 2 * apix, apix / 2)
    prof = np.stack([cut.sample(vol, apix, c, dirs * r) for r in rs], 1)           # (n_dirs, n_r)
    base = np.median(prof[:, -max(3, len(rs) // 7):], 1)                            # solvent, per direction
    r_in = rs[(prof - base[:, None] >= 0.5 * (prof.max(1) - base)[:, None]).argmax(1)]
    B = np.column_stack([np.ones(n_dirs)] + [ico_harmonic(dirs, l) for l in orders])
    keep = np.ones(n_dirs, bool)
    for _ in range(2):                                   # refit without the outliers (pores, spikes)
        coef = np.linalg.lstsq(B[keep], r_in[keep], rcond=None)[0]
        res = r_in - B @ coef
        keep = np.abs(res) < 3 * 1.4826 * np.median(np.abs(res[keep])) + apix
    model = dict(coef=coef, orders=orders, mean=float(coef[0]))
    ax = geo.symmetry_axes()
    model["vertex"] = float(surface_radius(model, ax["5-fold"][:1])[0])
    model["face"] = float(surface_radius(model, ax["3-fold"][:1])[0])
    return model


FACETED_FRAC = 0.05


def is_faceted(model, frac=FACETED_FRAC):
    """Vertex and face centre more than `frac` of the radius apart (T7: 11%,
    TCV: 10%; copia, HPV45, DHBV, HAV, PCV3: <= 3%). Below that the fitted
    "facets" are mostly capsomer relief - on small capsids the capsomer pattern
    lives in the same low harmonics (at T=1 the pentamers ARE the vertices) -
    and following them cancels the very contrast the lattice test looks for,
    so the capsid is read as a sphere."""
    return abs(model["vertex"] - model["face"]) > frac * model["mean"]


def surface_radius(model, dirs):
    B = np.column_stack([np.ones(len(dirs))] + [ico_harmonic(dirs, l) for l in model["orders"]])
    return B @ model["coef"]


MIN_SPACING = 60.0      # A; capsomers are 76-150 A apart in every capsid tested (PCV3 76, copia 82, VZV 150)


def plausible(cands, r_outer, min_spacing=MIN_SPACING):
    """The candidate lattices whose capsomers would be at least min_spacing
    apart at the capsid's radius. Finer lattices are not capsomer lattices: at
    their scale the test sees subunits, which can line up with a superlattice
    (DHBV's T=4 dimer spikes sit on T=16 sites, 42 A apart)."""
    return [hk for hk in cands if L.capsid_model(*hk)["nn_angle"] * r_outer >= min_spacing] or [(1, 0)]


def lattice_scores(path, r_outer, cands, surface=None, n_dirs=20000, top=0.5, log=None,
                   morphologies=MORPHOLOGIES):
    """Does the surface of the map have the pattern of lattice (h, k), and nothing else?

    Three morphologies are tried: a blob on every capsomer site (10T + 2); for
    capsids clustered in dimers, a blob on every lattice edge (30T), their
    capsomer centres being pores; and for capsids clustered in trimers, a blob
    in every gap between three sites (20T, birnavirus T=13: 260 VP2 trimers).
    (Testing for pores on the sites instead does not work: a dimer-clustered
    T=3 has pores on its sites AND its triangle centres, which together are the
    T=9 sites.)

    Per candidate, at that lattice's own scale (theta = its neighbour angle,
    spacing = theta * r_outer):
      - blur the map with a Gaussian of sigma = blur x spacing, which fills
        hexamer pores so each capsomer is one blob
      - average the capsomer tops, r_outer .. r_outer + top x spacing (the whole
        layer puts the peaks on subunits around the pores instead). With a
        `surface` (surface_model), the band follows the capsid's facets: it
        starts at r_in(u) + (r_outer - mean r_in) in each direction u
      - find the local maxima of that angular map
      - recall = predicted positions with a maximum within tolerance,
        precision = maxima within tolerance of a predicted position
    A lattice coarser than the truth passes too (its blur merges the real
    capsomers onto its own sites); a finer one or the wrong hand does not.
    So the answer is the finest lattice that passes: see choose_lattice.
    """
    from scipy.ndimage import gaussian_filter, zoom
    vol, apix, c = mapio.load_map(path)
    models = {hk: L.capsid_model(*hk) for hk in cands}
    sig_min = min(b for _, _, b, _, _ in morphologies) * min(m["nn_angle"] for m in models.values()) * r_outer
    f = max(1.0, (sig_min / 2) / apix)                     # bin while sigma stays >= 2 px
    if f > 1:
        n = vol.shape[0]
        vol = zoom(gaussian_filter(vol, f / 2), 1 / f, order=1)
        apix = apix * (n - 1) / (vol.shape[0] - 1)
        c = np.full(3, (vol.shape[0] - 1) / 2)
    dirs = geo.sphere_points(n_dirs)
    tree = cKDTree(dirs)
    start = (np.full(n_dirs, float(r_outer)) if surface is None else
             surface_radius(surface, dirs) + (r_outer - surface["mean"]))
    out = []
    for hk, m in models.items():
        th = m["nn_angle"]
        sp = th * r_outer
        ts = np.arange(0, top * sp + 1e-6, apix / 2)
        row, amaps = [], {}
        for name, where, blur, nbhd, tol in morphologies:
            if blur not in amaps:                          # dimers and trimers share one blur
                v = gaussian_filter(vol, blur * sp / apix)
                amaps[blur] = np.mean([cut.sample(v, apix, c, dirs * (start + t)[:, None]) for t in ts], 0)
            a = amaps[blur]
            nb = tree.query_ball_point(dirs, 2 * np.sin(nbhd * th / 2))
            pk = np.array([i for i in range(n_dirs) if a[i] >= a[nb[i]].max()])
            pos = where(m)
            chord = 2 * np.sin(tol * th / 2)
            rec = float((cKDTree(dirs[pk]).query(pos)[0] < chord).mean())
            prec = float((cKDTree(pos).query(dirs[pk])[0] < chord).mean())
            out.append(dict(h=hk[0], k=hk[1], T=m["T"], label=L.label(*hk), morphology=name,
                            n_expected=len(pos), n_peaks=len(pk),
                            recall=rec, precision=prec, score=min(rec, prec)))
            row.append(f"{name} {len(pk):4d}/{len(pos):4d} rec {rec:.2f} prec {prec:.2f}")
        if log:
            log(f"    {L.label(*hk):20s} " + " | ".join(row))
    return out


def passes(s, pass_at=0.8, partial=(0.95, 0.5)):
    """'full': recall and precision >= pass_at. 'partial': nearly every maximum
    is on the lattice (precision >= 0.95) and at least half the positions have
    one - some capsomer kinds are too flat to make a maximum (Sputnik T=27:
    140 maxima, all on its 272 sites). The true capsomers alone cover at most a
    third of a CK superlattice; what else lines up with one is sub-capsomer
    detail, which `plausible` keeps out by the capsomer spacing."""
    if s["score"] >= pass_at:
        return "full"
    if s["precision"] >= partial[0] and s["recall"] >= partial[1]:
        return "partial"
    return None


def choose_lattice(scores):
    """The finest lattice with T >= 3 that passes (full before partial; of the
    two hands of one T, the better). T=1 is not evidence: its 12 vertices
    stand out on nearly every icosahedral map, so it only wins when nothing
    else passes. -> (score dict, status): status is 'pass', 'only_T1' or 'none'."""
    for s in scores:
        s["pass"] = passes(s)
    if any(s["pass"] == "full" and s["T"] > 1 for s in scores):
        # a clear lattice is there; partial matches of finer ones are what it looks like at a
        # finer scale (e15: T=7 dimers on the edges are T=28 sites)
        for s in scores:
            s["pass"] = s["pass"] if s["pass"] == "full" else None
    ok = [s for s in scores if s["pass"]]
    big = [s for s in ok if s["T"] > 1]
    if big:
        t = max(s["T"] for s in big)
        return max((s for s in big if s["T"] == t), key=lambda s: (s["pass"] == "full", s["score"])), "pass"
    if ok:
        return max(ok, key=lambda s: s["score"]), "only_T1"
    return max(scores, key=lambda s: s["score"]), "none"


def site_profiles(path, model, r_lo, half_angle=None, n_dirs=300):
    """Radial density profile on each orbit's sites (cone-averaged).

    teact.site_profiles per orbit, with radii from r_lo instead of a fixed
    150 A and the cone sized on the lattice (0.3 x neighbour angle = 6 deg at
    copia's T=9, teact's value).
    """
    vol, apix, c = mapio.load_map(path)
    radii = np.arange(max(r_lo, apix), min(vol.shape) / 2 * apix, apix / 2)
    half = np.degrees(0.3 * model["nn_angle"]) if half_angle is None else half_angle
    rng = np.random.default_rng(0)
    a = np.radians(half) * np.sqrt(rng.uniform(0, 1, n_dirs))
    p = rng.uniform(0, 2 * np.pi, n_dirs)
    cone = np.stack([np.sin(a) * np.cos(p), np.sin(a) * np.sin(p), np.cos(a)])
    out = {}
    for o, name in enumerate(model["orbit_names"]):
        dirs = model["directions"][model["orbits"] == o]
        prof = np.zeros(len(radii))
        for d in dirs:
            v = (geo.frame(d) @ cone).T
            prof += np.array([cut.sample(vol, apix, c, v * r).mean() for r in radii])
        out[name] = prof / len(dirs)
    return radii, out
