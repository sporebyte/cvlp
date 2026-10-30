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
    ("dimers", L.edge_midpoints, 0.1, 0.25, 0.125),              # HBV, TCV: spikes on the 2-folds
)


def lattice_scores(path, r_outer, cands, n_dirs=20000, top=0.5, log=None, morphologies=MORPHOLOGIES):
    """Does the surface of the map have the pattern of lattice (h, k), and nothing else?

    Two morphologies are tried: a blob on every capsomer site (10T + 2), or,
    for capsids clustered in dimers, a blob on every lattice edge (30T); their
    capsomer centres are pores. (Testing for pores on the sites instead does not
    work: a dimer-clustered T=3 has pores on its sites AND its triangle centres,
    which together are the T=9 sites.)

    Per candidate, at that lattice's own scale (theta = its neighbour angle,
    spacing = theta * r_outer):
      - blur the map with a Gaussian of sigma = blur x spacing, which fills
        hexamer pores so each capsomer is one blob
      - average the capsomer tops, r_outer .. r_outer + top x spacing (the whole
        layer puts the peaks on subunits around the pores instead)
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
    out = []
    for hk, m in models.items():
        th = m["nn_angle"]
        sp = th * r_outer
        rs = np.arange(r_outer, r_outer + top * sp + 1e-6, apix / 2)
        row = []
        for name, where, blur, nbhd, tol in morphologies:
            v = gaussian_filter(vol, blur * sp / apix)
            a = np.mean([cut.sample(v, apix, c, dirs * r) for r in rs], 0)
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


def choose_lattice(scores, pass_at=0.8):
    """Finest lattice with recall and precision >= pass_at in either morphology;
    between the two hands of one T, the better one. -> (score dict, passed?)"""
    ok = [s for s in scores if s["score"] >= pass_at]
    if not ok:
        return max(scores, key=lambda s: s["score"]), False
    t = max(s["T"] for s in ok)
    return max((s for s in ok if s["T"] == t), key=lambda s: s["score"]), True


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
