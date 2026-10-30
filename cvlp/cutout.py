"""The capsomer cutout: sampling the averaged map, cutting references, the mask,
and the reference checks.

Taken from teact (sociology/vlp_oversample.py) unchanged, except that
icosahedral_selfcheck takes its radial band as arguments.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from .geometry import C5, frame, sphere_points
from .mapio import load_map, write_mrc


# ==========================================================================
# the averaged VLP map
# ==========================================================================

def sample(vol, apix, centre, pts_A, order=1):
    """Density at points (N,3) given in Angstrom relative to the map centre."""
    from scipy.ndimage import map_coordinates
    vox = centre[:, None] + np.asarray(pts_A).T / apix
    return map_coordinates(vol, vox[::-1], order=order, mode="constant")


def radial_profile(path, step=None, n_dirs=20000):
    vol, apix, c = load_map(path)
    step = apix / 2 if step is None else step
    radii = np.arange(step, min(vol.shape) / 2 * apix, step)
    dirs = sphere_points(n_dirs)
    s = np.array([sample(vol, apix, c, dirs * r).mean() for r in radii])
    return radii, s


def icosahedral_selfcheck(path, r_lo, r_hi):
    """Correlation of the map with itself rotated about I1 axes (1.0 = symmetric)."""
    from scipy.ndimage import map_coordinates
    vol, apix, c = load_map(path)
    n = vol.shape[0]
    g = np.arange(n) - c[0]
    Z, Y, X = np.meshgrid(g, g, g, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()])
    rr = np.sqrt(X ** 2 + Y ** 2 + Z ** 2) * apix
    msk = (rr > r_lo) & (rr < r_hi)
    phi = (1 + np.sqrt(5)) / 2
    out = {}
    for name, ax, deg in (("5-fold (0,1,phi) 72", (0, 1, phi), 72),
                          ("3-fold (1,1,1) 120", (1, 1, 1), 120),
                          ("2-fold z 180", (0, 0, 1), 180),
                          ("control: random axis 72", (0.3, 0.5, 0.8), 72)):
        a = np.array(ax, float) / np.linalg.norm(ax)
        Rm = R.from_rotvec(a * np.radians(deg)).as_matrix()
        w = map_coordinates(vol, (Rm @ P + c[:, None])[::-1], order=1).reshape(vol.shape)
        out[name] = float(np.corrcoef(vol[msk], w[msk])[0, 1])
    return out


def cut_outer_shell(path, gap_radius, out_path, soft=3):
    """Map with everything inside gap_radius removed (raised-cosine edge)."""
    vol, apix, c = load_map(path)
    n = np.array(vol.shape)
    zz, yy, xx = np.mgrid[0:n[0], 0:n[1], 0:n[2]].astype(np.float32)
    r = np.sqrt(((xx - c[0]) ** 2 + (yy - c[1]) ** 2 + (zz - c[2]) ** 2)) * apix
    m = 0.5 - 0.5 * np.cos(np.pi * np.clip((r - gap_radius) / (soft * apix), 0, 1))
    write_mrc(out_path, vol * m, apix)
    load_map.cache_clear()
    return str(out_path)


# ==========================================================================
# references and mask
# ==========================================================================

def cut_reference(map_path, direction, radius, box, apix_out, out_path=None,
                  lowpass=None, order=3):
    """Resample a box centred on direction*radius, outward = +Z. -> array (z,y,x)."""
    from scipy.ndimage import map_coordinates, gaussian_filter
    vol, apix, c = load_map(map_path)
    if lowpass:
        vol = gaussian_filter(vol, lowpass / apix / 2.355)
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    g = (np.arange(box) - box / 2) * apix_out           # RELION box centre = n/2
    bx, by, bz = np.meshgrid(g, g, g, indexing="ij")
    local = np.stack([bx.ravel(), by.ravel(), bz.ravel()])
    world = frame(d) @ local + (d * radius)[:, None]
    dens = map_coordinates(vol, (c[:, None] + world / apix)[::-1], order=order,
                           mode="constant").reshape(box, box, box)
    arr = np.ascontiguousarray(dens.T, dtype=np.float32)   # (x,y,z) -> (z,y,x)
    if out_path:
        write_mrc(out_path, arr, apix_out)
    return arr


def bud_offset(arr, apix, r_local, floor=0.5):
    """Centre-of-mass offset (px, xyz) of the density inside r_local of the box centre."""
    nz, ny, nx = arr.shape
    cen = np.array([nx, ny, nz], float) / 2
    gz, gy, gx = [(np.arange(n) - cc) * apix for n, cc in zip((nz, ny, nx), cen[::-1])]
    d = np.sqrt(gz[:, None, None] ** 2 + gy[None, :, None] ** 2 + gx[None, None, :] ** 2)
    inside = d <= r_local
    w = np.where(inside, arr - np.median(arr[inside]), 0.0)
    w = np.clip(w - floor * w.max(), 0, None)
    t = w.sum()
    com = np.array([(w.sum((0, 1)) * np.arange(nx)).sum() / t,
                    (w.sum((0, 2)) * np.arange(ny)).sum() / t,
                    (w.sum((1, 2)) * np.arange(nz)).sum() / t])
    return com - cen


def centre_site_radius(map_path, direction, radius, apix, box, r_local, tol_px=0.25, max_iter=6):
    """Move the cut radius until the capsomer's centre of mass sits at box z=0."""
    r = float(radius)
    for _ in range(max_iter):
        arr = cut_reference(map_path, direction, r, box, apix)
        off = bud_offset(arr, apix, r_local)
        if abs(off[2]) <= tol_px:
            break
        r += off[2] * apix
    return r, off


def normalise_in_mask(arr, mask):
    m = mask > 0.5
    return ((arr - arr[m].mean()) / arr[m].std()).astype(np.float32)


def _ramp(x, width):
    """Raised cosine 0 -> 1 over [-width/2, +width/2] around x = 0."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(x / width + 0.5, 0, 1))


def capsomer_mask(box, apix, capsomer_radius, r_cut, diameter, r_outer=None,
                  soft_sphere=30.0, soft_inner=20.0, soft_outer=30.0):
    """Shell-section mask in a capsomer box (VLP centre at (0,0,-capsomer_radius)).

    = sphere of `diameter` centred in the box
      x radial band r_cut < r < r_outer measured from the VLP CENTRE.
    Every soft edge (widths in A) is CENTRED on its radius, so the inner edge
    removes the RNP layer without dimming the capsomer's own base, and the outer
    edge keeps the capsomer top while suppressing a touching neighbour VLP.
    """
    g = (np.arange(box) - box / 2) * apix
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    m = _ramp(diameter / 2 - np.sqrt(X ** 2 + Y ** 2 + Z ** 2), soft_sphere)
    r = np.sqrt(X ** 2 + Y ** 2 + (Z + capsomer_radius) ** 2)
    m *= _ramp(r - r_cut, soft_inner)
    if r_outer is not None:
        m *= _ramp(r_outer - r, soft_outer)
    return np.ascontiguousarray(m.T, dtype=np.float32)


def ring_peaks(arr, apix, capsomer_radius, nb_dist, lowpass=50.0, n=360):
    """Density around the neighbour ring (following the sphere), low-passed first
    so it sees capsomers rather than the subunit lobes around a hexamer pore."""
    from scipy.ndimage import gaussian_filter, map_coordinates
    from scipy.signal import find_peaks
    v = gaussian_filter(arr, lowpass / apix / 2.355)
    nz, ny, nx = v.shape
    c = np.array([nx, ny, nz], float) / 2
    a = nb_dist / capsomer_radius
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    lat, dz = capsomer_radius * np.sin(a), capsomer_radius * (np.cos(a) - 1)
    x, y = c[0] + lat / apix * np.cos(th), c[1] + lat / apix * np.sin(th)
    z = np.full(n, c[2] + dz / apix)
    prof = map_coordinates(v, np.stack([z, y, x]), order=1, mode="nearest")
    pk, _ = find_peaks(np.r_[prof, prof, prof], prominence=0.25 * np.ptp(prof))
    return th, prof, int(((pk >= n) & (pk < 2 * n)).sum())


def neighbour_contrast(arr, apix, model, site, radius, lowpass=50.0):
    """Is the neighbour ring of a reference where the lattice model puts it?

    Density (low-passed) at the model's neighbour positions, expressed in the box
    frame, versus the same positions spun about the box axis by half the ring
    period. Ratio of (on - off) to the box std: clearly > 0 means the box frame,
    the site and the lattice agree. Works for C1 sites, whose ring is not circular.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates
    v = gaussian_filter(arr, lowpass / apix / 2.355)
    d, cls = model["directions"], model["classes"]
    n_nb = 5 if cls[site] == C5 else 6
    ang = np.arccos(np.clip(d @ d[site], -1, 1))
    nb = np.argsort(ang)[1:n_nb + 1]
    F = frame(d[site])
    pts = (F.T @ (d[nb] * radius - d[site] * radius).T)          # box frame, A
    nz, ny, nx = v.shape
    c = np.array([nx, ny, nz], float) / 2

    def dens(p):
        return map_coordinates(v, (c[:, None] + p / apix)[::-1], order=1).mean()
    half = np.pi / n_nb
    Rz = lambda t: np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    on = dens(pts)
    off = 0.5 * (dens(Rz(half) @ pts) + dens(Rz(-half) @ pts))
    return float((on - off) / v.std()), n_nb
