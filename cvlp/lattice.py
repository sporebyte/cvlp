"""Caspar-Klug lattices for any T, both hands, folded onto the I1 icosahedron.

The construction is the one copia's capsid model used (teact's capsid_model),
which only built k <= h, i.e. one hand of every skewed lattice (T = 7, 13, 19,
21, ...), and labelled sites by local symmetry only, which merges inequivalent
hexamers (T = 13 has two kinds of C1 hexamer). Here both (h, k) and (k, h) are
built, in the same site order, and sites are split into icosahedral orbits:

  orbit of 12 -> C5 (pentamers on the 5-folds)
  orbit of 20 -> C3 (hexamers on the 3-folds)
  orbit of 30 -> C2 (hexamers on the 2-folds)
  orbit of 60 -> C1 (hexamers in general position; C1a, C1b, ... if several)

One template is cut per orbit. At T = 9 this is exactly copia's model: the
same 92 directions in the same order, orbits C1 / C3 / C5 (tests check it).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation as R

from . import geometry as geo

C1, C2, C3, C5 = geo.C1, geo.C2, geo.C3, geo.C5
ORBIT_CODE = {12: C5, 20: C3, 30: C2, 60: C1}


def icosahedral_group():
    """The 60 rotations of I in the I1 frame, (60, 3, 3); [0] is the identity."""
    v, f = geo.icosahedron()
    gens = [R.from_rotvec(v[0] * np.radians(72)).as_matrix(),
            R.from_rotvec(geo.symmetry_axes()["3-fold"][0] * np.radians(120)).as_matrix()]
    group = [np.eye(3)]
    i = 0
    while i < len(group):
        for g in gens:
            m = g @ group[i]
            if min(np.abs(m - q).max() for q in group) > 1e-6:
                group.append(m)
        i += 1
    assert len(group) == 60, len(group)
    return np.array(group)


def _faces_outward():
    """teact's faces, re-ordered where needed so every face is counter-clockwise
    seen from outside. A skewed lattice is only continuous across edges if all
    faces are wound the same way."""
    v, f = geo.icosahedron()
    f = f.copy()
    for t in f:
        a, b, c = v[t]
        if np.cross(b - a, c - a) @ (a + b + c) < 0:
            t[[1, 2]] = t[[2, 1]]
    return v, f


def lattice(h, k, tol_deg=2.0):
    """Ideal capsomer directions for the (h, k) lattice. Same construction and
    site order as teact.capsid_model, without the k <= h restriction."""
    a1, a2 = np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2])
    P = h * a1 + k * a2
    Q = np.array([[0.5, -np.sqrt(3) / 2], [np.sqrt(3) / 2, 0.5]]) @ P
    Minv = np.linalg.inv(np.column_stack([P, Q]))
    lim, bary = h + k + 2, []
    for i in range(-lim, lim + 1):
        for j in range(-lim, lim + 1):
            u = Minv @ (i * a1 + j * a2)
            b = np.array([1 - u[0] - u[1], u[0], u[1]])
            if np.all(b >= -1e-9) and np.all(b <= 1 + 1e-9):
                bary.append(np.clip(b, 0, 1))
    bary = np.array(bary)
    verts, faces = geo.icosahedron() if (k == 0 or h == k) else _faces_outward()
    pts = np.vstack([bary @ verts[f] for f in faces])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    keep = []
    for p in pts:
        if not any(p @ q > np.cos(np.radians(tol_deg)) for q in keep):
            keep.append(p)
    return np.array(keep)


def capsid_model(h, k, group=None):
    """dict(directions, classes, orbits, orbit_names, rules, counts, T, h, k).

    `classes` holds the local-symmetry codes C1/C2/C3/C5 per site (what
    cutout.neighbour_contrast reads); `orbits` and `orbit_names` split them further.
    """
    T = h * h + h * k + k * k
    dirs = lattice(h, k)
    if len(dirs) != 10 * T + 2:
        raise RuntimeError(f"(h,k)=({h},{k}): {len(dirs)} sites, expected {10 * T + 2}")
    G = icosahedral_group() if group is None else group
    tree = cKDTree(dirs)
    orbit = -np.ones(len(dirs), int)
    n_orb = 0
    for i in range(len(dirs)):
        if orbit[i] < 0:
            _, j = tree.query(np.einsum("gij,j->gi", G, dirs[i]))
            orbit[j] = n_orb
            n_orb += 1
    size = np.bincount(orbit)
    code = np.array([ORBIT_CODE[s] for s in size])
    # name orbits by local symmetry; several C1 (or C2) orbits get a, b, c...
    # in order of distance from the nearest 5-fold
    fivefold = geo.symmetry_axes()["5-fold"]
    dist5 = [np.degrees(np.arccos(np.clip((fivefold @ dirs[orbit == o].T).max(), -1, 1)))
             for o in range(n_orb)]
    names = [""] * n_orb
    for c in (C5, C3, C2, C1):
        same = sorted(np.flatnonzero(code == c), key=lambda o: dist5[o])
        for n, o in enumerate(same):
            names[o] = geo.CLASS_NAMES[c] + ("abcdefghij"[n] if len(same) > 1 else "")
    # orbits in teact's order: by first site index (T=9 -> C5 site 0, C1 site 1, C3 site 5)
    first = np.array([np.flatnonzero(orbit == o)[0] for o in range(n_orb)])
    order = np.argsort(first)
    remap = np.empty(n_orb, int)
    remap[order] = np.arange(n_orb)
    orbit = remap[orbit]
    names = [names[o] for o in order]
    code = code[order]

    d, nb = tree.query(dirs, k=2)
    r_nb = np.median(d[:, 1]) * 1.35
    rules, counts = {}, {}
    for o, name in enumerate(names):
        m = orbit == o
        counts[name] = int(m.sum())
        comp = {}
        for j in np.flatnonzero(m):
            for q in tree.query_ball_point(dirs[j], r_nb):
                if q != j:
                    comp[names[orbit[q]]] = comp.get(names[orbit[q]], 0) + 1
        rules[name] = {a: round(v / m.sum()) for a, v in sorted(comp.items())}
    return dict(directions=dirs, classes=code[orbit], orbits=orbit, orbit_names=names,
                representative={n: int(np.flatnonzero(orbit == o)[0]) for o, n in enumerate(names)},
                nn_angle=float(np.median(d[:, 1])), rules=rules, counts=counts, T=T, h=h, k=k)


def label(h, k):
    T = h * h + h * k + k * k
    return f"T={T}" if (k == 0 or h == k) else f"T={T} (h,k)=({h},{k})"


def candidates(t_max=31, T=None):
    """(h, k) pairs with T <= t_max (or == T): both hands of every skewed lattice."""
    out = []
    for h in range(1, t_max + 1):
        for k in range(0, t_max + 1):
            t = h * h + h * k + k * k
            if t <= t_max and (T is None or t == T):
                out.append((h, k))
    return sorted(out, key=lambda hk: (hk[0] ** 2 + hk[0] * hk[1] + hk[1] ** 2, -hk[0]))


def neighbour_distances(model, radius):
    """Arc distance (A) from each orbit's representative site to its nearest
    neighbours (5 for C5, 6 otherwise). teact's version, per orbit."""
    d = model["directions"]
    out = {}
    for name, i in model["representative"].items():
        n = 5 if model["classes"][i] == C5 else 6
        ang = np.sort(np.arccos(np.clip(d @ d[i], -1, 1)))[1:n + 1]
        out[name] = ang * radius
    return out


def holes(model):
    """Directions of the gaps between capsomers: centroids of the lattice's
    Delaunay triangles (the local 3-fold points between three capsomers)."""
    d = model["directions"]
    tri = ConvexHull(d).simplices
    c = d[tri].mean(1)
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def contains(fine, coarse, tol=0.15):
    """Is every site of `coarse` also a site of `fine` (a CK superlattice)?
    Tolerance in units of the fine lattice's neighbour angle."""
    d, _ = cKDTree(fine["directions"]).query(coarse["directions"])
    return bool((d < 2 * np.sin(tol * fine["nn_angle"] / 2)).all())


def edge_midpoints(model):
    """Directions of the 30T lattice edges' midpoints: where the dimers sit in a
    capsid that clusters in dimers (HBV, TCV) rather than in capsomers."""
    d = model["directions"]
    pairs = np.array(sorted(cKDTree(d).query_pairs(2 * np.sin(1.35 * model["nn_angle"] / 2))))
    m = d[pairs[:, 0]] + d[pairs[:, 1]]
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def decoy_pair(model):
    """The two neighbouring sites whose midpoint is the decoy, by copia's recipe
    (make_decoy_reference.py): a site of the most numerous orbit and its nearest
    neighbour of the same orbit; if that orbit has no neighbour of its own kind
    (T = 3, 4), its nearest neighbour of any kind.

    Neighbours at the same distance are ordered so the edge's midpoint is off
    the symmetry axes first (a general-position saddle, as copia's was), then
    highest site index first. At T = 9 the tie is between (1, 4) and (1, 10),
    symmetry mates; the index rule picks (1, 10), copia's pair, so the decoy
    comes out in copia's in-plane orientation."""
    d, orbit, names = model["directions"], model["orbits"], model["orbit_names"]
    big = max(names, key=lambda n: (model["counts"][n], n == "C1"))
    i = model["representative"][big]
    o = names.index(big)
    ang = np.degrees(np.arccos(np.clip(d @ d[i], -1, 1)))
    axes = np.vstack(list(geo.symmetry_axes().values()))

    def on_axis(j):
        u = (d[i] + d[j]) / np.linalg.norm(d[i] + d[j])
        return bool((axes @ u).max() > np.cos(np.radians(1.0)))
    near = sorted(np.argsort(ang)[1:8], key=lambda j: (round(ang[j], 1), on_axis(j), -j))
    same = [j for j in near if orbit[j] == o and ang[j] < 1.35 * np.degrees(model["nn_angle"])]
    return i, int(same[0] if same else near[0])
