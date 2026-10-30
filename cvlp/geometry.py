"""Icosahedral geometry in the RELION I1 setting, and the box frame of a direction.

Taken from teact (sociology/vlp_oversample.py and sociology/orientations.py)
unchanged, so cvlp's lattices and cuts are the ones the copia references were
made with.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

C1, C2, C3, C5 = 1, 2, 3, 5
CLASS_NAMES = {C1: "C1", C2: "C2", C3: "C3", C5: "C5"}


def icosahedron():
    """Vertices (12,3) and faces (20,3), RELION I1 setting (5-fold on (0,1,phi))."""
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([[-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                  [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                  [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    return v / np.linalg.norm(v[0]), f


def symmetry_axes():
    v, f = icosahedron()
    unit = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
    edges = sorted({tuple(sorted(e)) for t in f for e in combinations(t, 2)})
    return {"5-fold": unit(v),
            "3-fold": unit(np.array([v[t].mean(0) for t in f])),
            "2-fold": unit(np.array([(v[a] + v[b]) / 2 for a, b in edges]))}


def sphere_points(n):
    """Fibonacci sphere, (n, 3) unit vectors."""
    i = np.arange(n, dtype=float) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(np.clip(1 - z * z, 0, None))
    phi = i * np.pi * (1 + np.sqrt(5))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def frame(d):
    """Orthonormal frame, columns (e1, e2, d): maps box +Z onto d.

    ONE direction at a time. Right-handed: det = +1. This fixes the in-plane
    orientation of every template, so it must stay as it is to reproduce copia.
    """
    d = np.asarray(d, float) / np.linalg.norm(d)
    a = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(d, a); e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(d, e1), d], axis=1)
