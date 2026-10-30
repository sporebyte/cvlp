"""Find the icosahedral frame of a deposited map and resample it into RELION I1.

The lattice and every cut (as in the copia pipeline) assume the I1 setting with the
5-fold on (0, 1, phi) and the symmetry centre at voxel (n-1)/2. EMDB maps come
in whatever setting the depositor's software used (I1, I2, 5-fold-on-z, ...), and
RELION puts the centre at n/2. So the map is first put into that frame:

  1. coarse search over the standard settings: a 5-, 3- or 2-fold on z, spun
     about z in 0.5 deg steps (covers I1-I4 and the EMAN/Chimera settings)
  2. if none of them is icosahedral, a global search: a 5-fold anywhere on the
     sphere, then a second 5-fold 63.4 deg from it
  3. local refinement of the rotation and the centre
  4. resample onto a grid in the I1 frame, centred at (n-1)/2

The score is the correlation of the map with itself under the group's
generators, on a copy binned to ~64 px.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, zoom
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from . import geometry as geo

PHI = (1 + np.sqrt(5)) / 2


def _generators():
    """5-fold, 3-fold, 2-fold and a second 5-fold of I, in the I1 frame."""
    v, _ = geo.icosahedron()
    ax = geo.symmetry_axes()
    return [R.from_rotvec(v[0] * np.radians(72)).as_matrix(),
            R.from_rotvec(ax["3-fold"][0] * np.radians(120)).as_matrix(),
            R.from_rotvec(ax["2-fold"][0] * np.pi).as_matrix(),
            R.from_rotvec(v[1] * np.radians(72)).as_matrix()]


GENS = _generators()


class _Scorer:
    """Symmetry score of a binned copy of the map, for frames F (I1 -> map)."""

    def __init__(self, vol, apix, n_target=64, n_points=40000, seed=0):
        f = max(1.0, vol.shape[0] / n_target)
        v = gaussian_filter(vol, f / 2) if f > 1 else vol
        self.vol = zoom(v, 1 / f, order=1) if f > 1 else v
        self.apix = apix * (vol.shape[0] - 1) / (self.vol.shape[0] - 1)
        n = self.vol.shape[0]
        self.c0 = np.full(3, (n - 1) / 2)
        g = (np.arange(n) - self.c0[0]) * self.apix
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2).ravel()
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)[r < 0.9 * self.c0[0] * self.apix]
        rng = np.random.default_rng(seed)
        self.pts = pts[rng.choice(len(pts), min(n_points, len(pts)), replace=False)]

    def _at(self, pts, shift):
        return map_coordinates(self.vol, (self.c0 + shift / self.apix + pts / self.apix).T[::-1],
                               order=1, mode="constant")

    def score(self, F, shift=np.zeros(3), ops=GENS):
        """Mean correlation of the map with itself rotated by each op (F g F^T)."""
        a = self._at(self.pts, shift)
        cc = []
        for g in ops:
            b = self._at(self.pts @ (F @ g @ F.T).T, shift)
            cc.append(np.corrcoef(a, b)[0, 1])
        return float(np.mean(cc))


def _align(u, target=np.array([0.0, 0.0, 1.0])):
    """Rotation taking unit vector u onto target."""
    u = u / np.linalg.norm(u)
    ax = np.cross(u, target)
    s, c = np.linalg.norm(ax), u @ target
    if s < 1e-12:
        return np.eye(3) if c > 0 else R.from_rotvec([np.pi, 0, 0]).as_matrix()
    return R.from_rotvec(ax / s * np.arctan2(s, c)).as_matrix()


def _rz(deg):
    return R.from_euler("z", deg, degrees=True).as_matrix()


def standard_search(sc, step=0.5):
    """Best frame with a 5-, 3- or 2-fold on z -> (score, F, label)."""
    v, _ = geo.icosahedron()
    ax = geo.symmetry_axes()
    best = (-1, None, None)
    for name, axis, n in (("5-fold", v[0], 5), ("3-fold", ax["3-fold"][0], 3), ("2-fold", np.array([0, 0, 1.0]), 2)):
        A = _align(axis)
        for psi in np.arange(0, 360 / n, step):
            F = _rz(psi) @ A
            s = sc.score(F, ops=GENS[:3])
            if s > best[0]:
                best = (s, F, f"{name} on z, spun {psi:.1f} deg")
    return best


def global_search(sc, step_deg=2.5):
    """5-fold anywhere, then a neighbouring 5-fold -> (score, F, label)."""
    n = int(4 * np.pi / np.radians(step_deg) ** 2)
    dirs = geo.sphere_points(n)
    dirs = dirs[dirs[:, 2] >= 0]
    s5 = []
    for a in dirs:
        g = R.from_rotvec(a * np.radians(72)).as_matrix()
        b = sc._at(sc.pts @ g.T, np.zeros(3))
        s5.append(np.corrcoef(sc._at(sc.pts, np.zeros(3)), b)[0, 1])
    v, _ = geo.icosahedron()
    j = int(np.argmax(v[1:] @ v[0])) + 1                          # a neighbouring vertex of v0
    best = (-1, None, None)
    for a in dirs[np.argsort(s5)[::-1][:5]]:
        e = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0]); e /= np.linalg.norm(e)
        ang = np.arccos(v[0] @ v[j])
        for t in np.radians(np.arange(0, 72, 1.0)):
            b = R.from_rotvec(a * t).apply(np.cos(ang) * a + np.sin(ang) * e)
            src = np.stack([v[0], v[j], np.cross(v[0], v[j])], 1)
            dst = np.stack([a, b, np.cross(a, b)], 1)
            F = dst @ np.linalg.inv(src)
            U, _, Vt = np.linalg.svd(F); F = U @ Vt
            s = sc.score(F, ops=GENS[:3])
            if s > best[0]:
                best = (s, F, "global search")
    return best


def refine(sc, F, shift=np.zeros(3)):
    """Local optimisation of rotation (rotvec on top of F) and centre shift (A)."""
    def cost(x):
        return -sc.score(R.from_rotvec(x[:3]).as_matrix() @ F, x[3:])
    x0 = np.r_[np.zeros(3), shift]
    res = minimize(cost, x0, method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-5, maxiter=600, initial_simplex=np.vstack(
                       [x0] + [x0 + np.eye(6)[i] * (np.radians(1) if i < 3 else sc.apix / 2) for i in range(6)])))
    return -res.fun, R.from_rotvec(res.x[:3]).as_matrix() @ F, res.x[3:]


def find_frame(vol, apix, min_score=0.85, log=print):
    """-> dict(F, shift_A, score, setting). F maps I1 coordinates into the map's."""
    sc = _Scorer(vol, apix)
    s, F, how = standard_search(sc)
    log(f"  standard settings: best {s:.3f} ({how})")
    if s < min_score:
        s, F, how = global_search(sc)
        log(f"  global search:     best {s:.3f}")
    s, F, shift = refine(sc, F)
    # a second pass on a finer copy
    sc2 = _Scorer(vol, apix, n_target=min(128, vol.shape[0]))
    s, F, shift = refine(sc2, F, shift)
    log(f"  refined:           {s:.3f}, rotation {np.degrees(np.arccos(np.clip((np.trace(F) - 1) / 2, -1, 1))):.2f} deg "
        f"from I1, centre shift {np.round(shift, 2).tolist()} A")
    if s < min_score:
        raise RuntimeError(f"map is not icosahedral (best symmetry score {s:.2f})")
    return dict(F=F, shift_A=shift, score=s, setting=how)


def to_I1(vol, apix, frame, n_out=None):
    """Resample the map into the I1 frame, symmetry centre at (n-1)/2."""
    from scipy.ndimage import spline_filter
    n = vol.shape[0]
    n_out = n if n_out is None else n_out
    c_in = np.full(3, (n - 1) / 2) + frame["shift_A"] / apix
    g = (np.arange(n_out) - (n_out - 1) / 2) * apix
    coef = spline_filter(vol, order=3)                    # once, then each slab samples it
    out = np.empty((n_out, n_out, n_out), np.float32)    # (z, y, x)
    Y, X = np.meshgrid(g, g, indexing="ij")
    for iz, z in enumerate(g):                            # one z-slice at a time
        p = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)])      # I1 frame, xyz A
        src = c_in[:, None] + frame["F"] @ p / apix                   # map voxels, xyz
        out[iz] = map_coordinates(coef, src[::-1], order=3, mode="constant",
                                  prefilter=False).reshape(n_out, n_out)
    return out
