"""Synthetic capsids for testing: a ring of 5 or 6 subunit blobs on every site
of an (h, k) lattice, plus an inner layer, in a random orientation and off-centre.

    python tests/synthetic.py 1,3 out.mrc [--seed 0] [--apix 4]

The rings have a pore, like real hexamers, so the lattice test has to see
capsomers rather than subunits. Ring spin is tied to the nearest 5-fold, so the
map is exactly icosahedral.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvlp import lattice as L                     # noqa: E402
from cvlp import geometry as geo, mapio         # noqa: E402


def make(h, k, apix=4.0, seed=0, radius=None):
    m = L.capsid_model(h, k)
    d, th = m["directions"], m["nn_angle"]
    radius = radius or 60 * np.sqrt(m["T"]) + 120          # ~ copia's 232 A at T=9 (=300)
    sp = th * radius
    v5 = geo.icosahedron()[0]
    pts = []
    for i, u in enumerate(d):
        n = 5 if m["classes"][i] == L.C5 else 6
        near = v5[np.argsort(-(v5 @ u))]
        ref = near[1] if near[0] @ u > 0.999 else near[0]
        e1 = ref - (ref @ u) * u; e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        for t in 2 * np.pi * np.arange(n) / n:
            pts.append(u * radius + 0.28 * sp * (np.cos(t) * e1 + np.sin(t) * e2))
    pts = np.array(pts)
    F = R.random(random_state=seed).as_matrix()
    shift = np.random.default_rng(seed).uniform(-0.5, 0.5, 3) * apix * 2
    nbox = int(2 * (radius + 80) / apix) // 2 * 2
    vol = np.zeros((nbox,) * 3, np.float32)
    c = (nbox - 1) / 2
    for p in (pts @ F.T + shift) / apix + c:
        ix = np.round(p).astype(int)
        vol[ix[2], ix[1], ix[0]] += 1.0
    vol = gaussian_filter(vol, 0.12 * sp / apix)
    # inner layer, 40 A below the capsomers
    g = (np.arange(nbox) - c) * apix
    Z, Y, X = np.meshgrid(g - shift[2], g - shift[1], g - shift[0], indexing="ij")
    rr = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    vol += 0.5 * vol.max() * np.exp(-0.5 * ((rr - (radius - 45)) / 6) ** 2)
    vol += np.random.default_rng(seed).normal(0, 0.05 * vol.max(), vol.shape).astype(np.float32)
    return vol, F, shift


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("hk")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--apix", type=float, default=4.0)
    a = ap.parse_args()
    h, k = (int(x) for x in a.hk.split(","))
    vol, F, shift = make(h, k, a.apix, a.seed)
    mapio.write_mrc(a.out, vol, a.apix)
    print(f"{L.label(h, k)}  {vol.shape[0]}^3 x {a.apix} A  rotation {np.degrees(R.from_matrix(F).magnitude()):.1f} deg  shift {shift.round(2)} A")
