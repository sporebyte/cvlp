"""The copia capsomer-template cutout, for any T.

Same steps and same teact functions as make_references_and_seeds.py sections
A-C and make_decoy_reference.py, run on an I1 map:

  A  layers -> outer shell only (inner layer removed at the gap)
  B  site profiles -> reference radius, neighbour distances, capsomer spacing
  C  one reference per capsomer orbit: radius centred on the fine grid, cut at
     the subtomogram pixel size, low-pass 2 x pixel, normalised in one shared
     shell-section mask
  D  the decoy: the same cut on the saddle between two neighbouring capsomers

Box and mask scale with the capsomer spacing the way copia's were set
(300 A mask at an 82 A spacing; 32 px x 11.8 A box per 300 A mask), so copia's
own numbers come back at T=9.
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


def _even(x):
    return int(2 * max(1, round(x / 2)))


def _cc(a, b, w):
    a = a - (a * w).sum() / w.sum()
    b = b - (b * w).sum() / w.sum()
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum()))


def _orbit_order(names):
    """C1..., C2, C3, C5: copia's file order (refs_C1_C3_C5)."""
    return sorted(names, key=lambda n: (int(n[1]), n))


def build(map_I1, out, model, apix, box=None, mask_diameter=None, relion_prefix=None,
          gap_radius=None, display=True, log=print):
    """Write every template file into `out` -> params dict (also params.json)."""
    out = Path(out)
    refdir = out / "references"
    dispdir = refdir / "display"                           # unfiltered fine-grid cuts, for ChimeraX
    (dispdir if display else refdir).mkdir(parents=True, exist_ok=True)
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

    if mask_diameter is None:
        mask_diameter = float(np.round(COPIA_MASK_PER_SPACING * SPACING, 1))
        if mask_diameter > 2 * R_REF:
            log(f"  ! mask {mask_diameter:.0f} A would reach round the capsid; capped at the diameter {2 * R_REF:.0f} A")
            mask_diameter = float(np.round(2 * R_REF, 1))
    box = _even(COPIA_BOX_PER_MASK * mask_diameter / apix) if box is None else int(box)
    fine_box = _even(box * apix / fine_apix)
    log(f"  box {box} px x {apix} A = {box * apix:.0f} A, mask {mask_diameter:.0f} A, "
        f"centring on {fine_box} px x {fine_apix:.3g} A")

    # ---- C. one reference per orbit ---------------------------------------
    refs, radii_ref, checks = {}, {}, {}
    for n in names:
        i = model["representative"][n]
        r, off_fine = cut.centre_site_radius(shell_path, d[i], R_REF, fine_apix, fine_box, r_local=r_local)
        arr = cut.cut_reference(shell_path, d[i], r, box, apix, lowpass=2 * apix)
        refs[n], radii_ref[n] = arr, float(r)
        fine = cut.cut_reference(shell_path, d[i], r, fine_box, fine_apix)
        nc, nn = cut.neighbour_contrast(fine, fine_apix, model, i, r)
        _, _, npk = cut.ring_peaks(fine, fine_apix, r, float(np.median(nbd[n])))
        checks[n] = dict(site=i, radius=float(r), off_fine_px=np.round(off_fine, 2).tolist(),
                         off_final_px=np.round(cut.bud_offset(arr, apix, r_local), 2).tolist(),
                         neighbour_contrast=round(nc, 2), ring_peaks=npk, expected=nn)
        if display:
            mapio.write_mrc(dispdir / f"{n}_{fine_box}px_{fine_apix:.3g}A_unfiltered.mrc", fine, fine_apix)
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
    dec = cut.normalise_in_mask(cut.cut_reference(shell_path, mid, r_dec, box, apix, lowpass=2 * apix), mask)
    ref_file["decoy"] = f"ref_decoy_{box}px_{apix:.1f}A.mrc"
    mapio.write_mrc(refdir / ref_file["decoy"], dec, apix)
    if display:
        mapio.write_mrc(dispdir / f"decoy_{fine_box}px_{fine_apix:.3g}A_unfiltered.mrc",
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
                  mask_diameter=mask_diameter, gap_radius=gap["gap"], mask_r_outer=R_OUTER,
                  inner_layer=gap["inner"], outer_layer=gap["outer"], capsomer_radius=R_CAPS,
                  reference_radius=R_REF, ref_radii=radii_ref, spacing=SPACING,
                  fine_apix=fine_apix, fine_box=fine_box, lowpass=2 * apix,
                  orbits={n: dict(count=model["counts"][n], neighbours=model["rules"][n]) for n in names},
                  reference_checks=checks,
                  decoy=dict(sites=[i, j], pair=pair, separation_deg=round(sep, 2), cut_radius=round(float(r_dec), 2),
                             centring_offset_fine_px=np.round(off_dec, 2).tolist()),
                  correlation_in_mask=corr,
                  files=dict(outer_shell=shell_path.name, mask=mask_path.name,
                             references={n: f"references/{f}" for n, f in ref_file.items()},
                             refs_star=stars["refs"].name, refs_decoy_star=stars["refs_decoy"].name),
                  relion_prefix=relion_prefix)
    params["_profiles"] = dict(radii=gap["radii"], profile=gap["profile"], site_r=site_r, site_prof=site_prof)
    return params


def write_params(params, path):
    p = {k: v for k, v in params.items() if not k.startswith("_")}
    Path(path).write_text(json.dumps(p, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x)))
