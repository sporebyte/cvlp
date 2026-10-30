"""The copia capsomer-template cutout, for any T.

Same steps and same teact functions as make_references_and_seeds.py sections
A-C and make_decoy_reference.py, run on an I1 map:

  A  layers -> outer shell only (inner layer removed at the gap)
  B  site profiles -> reference radius, neighbour distances, capsomer spacing
  C  one reference per capsomer orbit: radius centred on the fine grid, cut at
     the subtomogram pixel size, low-passed, normalised in one shared
     shell-section mask
  D  the decoy: the same cut on the saddle between two neighbouring capsomers

Low-pass: copia's is a real-space Gaussian of FWHM 2 x pixel, applied while
cutting. It passes 50% only at ~2.3 x its FWHM (53 A for 11.8 A/px), so the
templates are far blurrier than the pixel allows. lowpass="nyquist" or a
resolution in A uses a Fourier low-pass on the shell map instead (see
cutout.fourier_lowpass), which keeps everything coarser than the cutoff.

Box and mask scale with the capsomer spacing the way copia's were set
(300 A mask at an 82 A spacing; 32 px x 11.8 A box per 300 A mask), so copia's
own numbers come back at T=9. On small capsids (T=1-4) that mask would reach
+-60 deg or more round the particle, a quarter of its surface, so it is capped
at +-50 deg of arc (max_arc). Measured on HAV, TCV, DHBV and PCV3, the cap
changes how distinct the templates are by <= 0.05 in correlation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import lattice as L
from . import shell as S
from . import cutout as cut, mapio

COPIA_MASK_PER_SPACING = 300.0 / 82.02114748843148     # copia params.json: mask_diameter / spacing
COPIA_BOX_PER_MASK = 32 * 11.8 / 300.0                 # copia: 32 px x 11.8 A box for a 300 A mask
MASK_R_MARGIN = 15.0                                   # A past the capsomer layer top (teact config)
MAX_ARC_DEG = 50.0      # default mask stays within +-50 deg of arc: T>=7 untouched, T=1-4 cut to 18% of the surface


def _even(x):
    return int(2 * max(1, round(x / 2)))


def _arc(mask_diameter, radius):
    """Half-angle (deg, seen from the capsid centre) that a mask sphere centred on
    a capsomer at `radius` reaches round the capsid."""
    return float(2 * np.degrees(np.arcsin(min(1.0, mask_diameter / 4 / radius))))


def _cc(a, b, w):
    a = a - (a * w).sum() / w.sum()
    b = b - (b * w).sum() / w.sum()
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum()))


def _orbit_order(names):
    """C1..., C2, C3, C5: copia's file order (refs_C1_C3_C5)."""
    return sorted(names, key=lambda n: (int(n[1]), n))


def _lowpass(lowpass, apix, shell_path, out, log):
    """-> (map to cut the templates from, cut_reference kwargs, params record)."""
    if lowpass == "copia":
        fwhm = 2 * apix
        info = dict(mode="copia", gaussian_fwhm_A=fwhm, half_amplitude_A=round(cut.gaussian_half_amplitude(fwhm), 1))
        log(f"  low-pass: copia's Gaussian, FWHM {fwhm:.1f} A (50% at {info['half_amplitude_A']} A)")
        return shell_path, dict(lowpass=fwhm), info
    res = 2 * apix if lowpass == "nyquist" else float(lowpass)
    if res < 2 * apix - 1e-6:
        raise ValueError(f"--lowpass {res:g} A is finer than Nyquist at {apix} A/px ({2 * apix:g} A)")
    vol, vapix, _ = mapio.load_map(shell_path)
    src = out / f"outer_shell_lowpass_{res:.1f}A.mrc"
    mapio.write_mrc(src, cut.fourier_lowpass(vol, vapix, res), vapix)
    info = dict(mode="fourier", cutoff_A=res, half_amplitude_A=round(res / (1 - cut.FOURIER_EDGE / 2), 1))
    log(f"  low-pass: Fourier, nothing finer than {res:.1f} A (50% at {info['half_amplitude_A']} A)")
    return src, {}, info


def build(map_I1, out, model, apix, box=None, mask_diameter=None, relion_prefix=None,
          gap_radius=None, lowpass="copia", max_arc=MAX_ARC_DEG, unfiltered=True, log=print):
    """Write every template file into `out` -> params dict (also params.json).

    lowpass: "copia" (default, reproduces copia), "nyquist" (2 x apix), or a
    resolution in A for a Fourier low-pass. max_arc: the largest half-angle
    (deg) the default mask may span round the capsid."""
    out = Path(out)
    refdir = out / "references"
    # Unfiltered cuts, for looking at: same centre, orientation and box (in A) as
    # each template, on the fine grid, no low-pass, no mask, no normalisation, so
    # the map's own density scale (and the EMDB contour level) still applies.
    unf_dir = refdir / "unfiltered"
    (unf_dir if unfiltered else refdir).mkdir(parents=True, exist_ok=True)
    unf_file = {}
    relion_prefix = str(out) if relion_prefix is None else relion_prefix.rstrip("/")
    fine_apix = apix / 4                                   # copia: 2.95 A for 11.8 A
    shell_path = out / "outer_shell_only.mrc"
    d = model["directions"]
    names = _orbit_order(model["orbit_names"])

    # ---- A. layers and the outer shell ------------------------------------
    gap = S.find_layers(map_I1, gap_radius=gap_radius)
    inner = f"{gap['inner']:.1f} A" if gap["has_inner"] else "-"
    note = ("  (given)" if gap_radius is not None else "" if gap["has_inner"] else
            "  (no inner layer: density filled to the centre, nothing removed - set --gap-radius)" if gap.get("filled") else
            "  (no inner layer: cut 10 A inside the capsid's inner edge)")
    log(f"  inner layer {inner} | gap {gap['gap']:.1f} A | outer layer {gap['outer']:.1f} A" + note)
    cut.cut_outer_shell(map_I1, gap["gap"], shell_path)
    lp_src, lp_kw, lp_info = _lowpass(lowpass, apix, shell_path, out, log)

    # ---- B. lattice geometry ----------------------------------------------
    site_r, site_prof = S.site_profiles(shell_path, model, r_lo=gap["gap"] - 20)
    R_REF = float(np.mean([site_r[np.argmax(p)] for p in site_prof.values()]))
    nbd = L.neighbour_distances(model, R_REF)
    SPACING = float(min(v.min() for v in nbd.values()))
    r_local = 0.45 * nbd["C5"][0]
    for n in names:
        log(f"  {n:4s} x{model['counts'][n]:3d}  peak r {site_r[np.argmax(site_prof[n])]:.1f} A  "
            f"neighbours at {np.round(nbd[n], 1).tolist()} A  {model['rules'][n]}")
    log(f"  reference radius {R_REF:.1f} A, shortest capsomer spacing {SPACING:.1f} A")

    # On a small capsid the neighbour ring sits far round the particle, and the
    # copia-sized mask becomes a quarter of the capsid. Keep it within +-max_arc
    # of arc around the capsomer (copia's own spans +-38 deg at T=9).
    ring_mask = float(np.round(COPIA_MASK_PER_SPACING * SPACING, 1))
    cap = float(np.round(4 * R_REF * np.sin(np.radians(max_arc) / 2), 1))
    small = ring_mask > cap
    if small:
        log(f"  note: small capsid. A capsomer and its neighbour ring ({ring_mask:.0f} A) span "
            f"+-{_arc(ring_mask, R_REF):.0f} deg of it, so any\n"
            f"        template is a large part of the particle. Oversampling works best on larger VLPs\n"
            f"        (T>=7, copia T=9: +-38 deg); for small rigid capsids, averaging whole particles and\n"
            f"        expanding the symmetry is usually more reliable.")
    if mask_diameter is None:
        mask_diameter = ring_mask
        if small:
            log(f"  mask capped at {cap:.0f} A (+-{max_arc:g} deg)")
            mask_diameter = cap
    arc = _arc(mask_diameter, R_REF)
    box = _even(COPIA_BOX_PER_MASK * mask_diameter / apix) if box is None else int(box)
    fine_box = _even(box * apix / fine_apix)
    log(f"  box {box} px x {apix} A = {box * apix:.0f} A, mask {mask_diameter:.0f} A "
        f"(+-{arc:.0f} deg, {100 * (1 - np.cos(np.radians(arc))) / 2:.0f}% of the capsid surface), "
        f"centring on {fine_box} px x {fine_apix:.3g} A")

    # ---- C. one reference per orbit ---------------------------------------
    refs, radii_ref, checks = {}, {}, {}
    for n in names:
        i = model["representative"][n]
        r, off_fine = cut.centre_site_radius(shell_path, d[i], R_REF, fine_apix, fine_box, r_local=r_local)
        arr = cut.cut_reference(lp_src, d[i], r, box, apix, **lp_kw)
        refs[n], radii_ref[n] = arr, float(r)
        fine = cut.cut_reference(shell_path, d[i], r, fine_box, fine_apix)
        nc, nn = cut.neighbour_contrast(fine, fine_apix, model, i, r)
        _, _, npk = cut.ring_peaks(fine, fine_apix, r, float(np.median(nbd[n])))
        checks[n] = dict(site=i, radius=float(r), off_fine_px=np.round(off_fine, 2).tolist(),
                         off_final_px=np.round(cut.bud_offset(arr, apix, r_local), 2).tolist(),
                         neighbour_contrast=round(nc, 2), ring_peaks=npk, expected=nn)
        if unfiltered:
            unf_file[n] = f"ref_{n}_{fine_box}px_{fine_apix:.2f}A.mrc"
            mapio.write_mrc(unf_dir / unf_file[n], fine, fine_apix)
        log(f"  {n:4s} cut at r={r:.1f} A | centring offset fine {checks[n]['off_fine_px']} px, "
            f"final {checks[n]['off_final_px']} px | neighbour contrast {nc:+.2f} (want >0) | "
            f"ring peaks {npk} (expect {nn})")

    R_CAPS = float(np.mean(list(radii_ref.values())))
    mp = np.mean(list(site_prof.values()), axis=0)
    jpk = int(np.argmax(mp))
    # copia: where the site profile first drops below zero. A map whose solvent
    # sits above zero never gets there, so also stop within 10% of the solvent.
    lvl = max(0.0, gap["solvent"] + 0.1 * (mp[jpk] - gap["solvent"]))
    below = np.flatnonzero(mp[jpk:] < lvl)
    R_LAYER_TOP = float(site_r[jpk + below[0]]) if len(below) else float(site_r[-1])
    R_OUTER = R_LAYER_TOP + MASK_R_MARGIN
    mask = cut.capsomer_mask(box, apix, R_CAPS, gap["gap"], mask_diameter, r_outer=R_OUTER)
    mask_path = out / f"mask_capsomer_{box}px.mrc"
    mapio.write_mrc(mask_path, mask, apix)
    ref_file = {}
    for n, arr in refs.items():
        refs[n] = cut.normalise_in_mask(arr, mask)
        ref_file[n] = f"ref_{n}_{box}px_{apix:.1f}A.mrc"
        mapio.write_mrc(refdir / ref_file[n], refs[n], apix)
    log(f"  mask keeps {mask.mean():.1%} of the box; radial band {gap['gap']:.0f}-{R_OUTER:.0f} A "
        f"(capsomer layer ends at {R_LAYER_TOP:.0f} A), sphere {mask_diameter:.0f} A")

    # ---- D. the decoy -----------------------------------------------------
    i, j = L.decoy_pair(model)
    mid = d[i] + d[j]
    mid /= np.linalg.norm(mid)
    sep = float(np.degrees(np.arccos(np.clip(d[i] @ d[j], -1, 1))))
    pair = f"{model['orbit_names'][model['orbits'][i]]}-{model['orbit_names'][model['orbits'][j]]}"
    r_dec, off_dec = cut.centre_site_radius(shell_path, mid, R_CAPS, fine_apix, fine_box, r_local=r_local)
    dec = cut.normalise_in_mask(cut.cut_reference(lp_src, mid, r_dec, box, apix, **lp_kw), mask)
    ref_file["decoy"] = f"ref_decoy_{box}px_{apix:.1f}A.mrc"
    mapio.write_mrc(refdir / ref_file["decoy"], dec, apix)
    if unfiltered:
        unf_file["decoy"] = f"ref_decoy_{fine_box}px_{fine_apix:.2f}A.mrc"
        mapio.write_mrc(unf_dir / unf_file["decoy"],
                        cut.cut_reference(shell_path, mid, r_dec, fine_box, fine_apix), fine_apix)
    log(f"  decoy: saddle between sites {i} and {j} ({pair}, {sep:.1f} deg apart), cut at r = {r_dec:.1f} A")

    # ---- references must be distinct, the decoy most of all ----------------
    allrefs = {**refs, "decoy": dec}
    corr = {a: {b: round(_cc(allrefs[a], allrefs[b], mask), 3) for b in allrefs} for a in allrefs}
    log("  correlation in the mask:")
    for a in allrefs:
        log("    " + f"{a:6s}" + " ".join(f"{corr[a][b]:6.3f}" for b in allrefs))

    # ---- RELION --ref lists ------------------------------------------------
    tag = "_".join(names)
    stars = {"refs": out / f"refs_{tag}.star", "refs_decoy": out / f"refs_{tag}_decoy.star"}
    mapio.write_reference_star(stars["refs"], [f"{relion_prefix}/references/{ref_file[n]}" for n in names])
    mapio.write_reference_star(stars["refs_decoy"],
                                [f"{relion_prefix}/references/{ref_file[n]}" for n in names + ["decoy"]])

    params = dict(T=model["T"], h=model["h"], k=model["k"], bin_apix=apix, box=box,
                  mask_diameter=mask_diameter, mask_arc_deg=round(arc, 1), small_capsid=bool(small),
                  gap_radius=gap["gap"], mask_r_outer=R_OUTER,
                  inner_layer=gap["inner"], outer_layer=gap["outer"], capsomer_radius=R_CAPS,
                  reference_radius=R_REF, ref_radii=radii_ref, spacing=SPACING,
                  fine_apix=fine_apix, fine_box=fine_box, lowpass=lp_info,
                  orbits={n: dict(count=model["counts"][n], neighbours=model["rules"][n]) for n in names},
                  reference_checks=checks,
                  decoy=dict(sites=[i, j], pair=pair, separation_deg=round(sep, 2), cut_radius=round(float(r_dec), 2),
                             centring_offset_fine_px=np.round(off_dec, 2).tolist()),
                  correlation_in_mask=corr,
                  files=dict(outer_shell=shell_path.name, mask=mask_path.name,
                             lowpassed_shell=None if lp_src == shell_path else lp_src.name,
                             references={n: f"references/{f}" for n, f in ref_file.items()},
                             references_unfiltered={n: f"references/unfiltered/{f}" for n, f in unf_file.items()},
                             refs_star=stars["refs"].name, refs_decoy_star=stars["refs_decoy"].name),
                  relion_prefix=relion_prefix)
    params["_profiles"] = dict(radii=gap["radii"], profile=gap["profile"], site_r=site_r, site_prof=site_prof)
    return params


def write_params(params, path):
    p = {k: v for k, v in params.items() if not k.startswith("_")}
    Path(path).write_text(json.dumps(p, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x)))
