"""cvlp: capsomer templates for an icosahedral VLP, from an EMDB entry or your own map.

    cvlp EMD-19708                           # copia: T=9 -> C1, C3, C5 + decoy
    cvlp EMD-19708 --apix 11.8 --box 32 --relion-prefix surface_vlps_1808/seeding_v2
    cvlp --map my_vlp_average.mrc            # your own average (no EMDB entry)
    cvlp EMD-1234 --map local.mrc            # a local copy of an EMDB map (metadata still read)
    cvlp EMD-1234 --T 7                      # T given; the hand is still read from the map
    cvlp EMD-1234 --hk 1,2                   # force one lattice
    cvlp EMD-1234 --lowpass nyquist          # sharper templates than copia's Gaussian
    cvlp-gui                                 # the same, with a preview of the cutouts

Steps: fetch or read the map -> find its icosahedral frame and centre ->
resample into RELION I1 -> pick the lattice (T, and the hand if it is skewed)
-> the copia template cutout (one reference per capsomer orbit + an edge decoy
+ the shared mask + the RELION --ref lists).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import emdb, pipeline, recipe


def lowpass_arg(s):
    if s.lower() in ("copia", "nyquist"):
        return s.lower()
    try:
        return float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected copia, nyquist or a resolution in A, got {s!r}")


def default_out(emdb_id, map_path):
    return Path(emdb.normalise_id(emdb_id) if emdb_id else Path(map_path).stem)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("emdb_id", nargs="?", help="EMDB entry, e.g. EMD-19708 (optional with --map)")
    ap.add_argument("--map", help="a map file: your own average, or a local copy of the entry's map")
    ap.add_argument("--out", "-o", help="output folder (default: ./<EMDB id>, or the map's file name)")
    ap.add_argument("--apix", type=float, default=11.8, help="A/px of the subtomograms the templates are for (11.8)")
    ap.add_argument("--box", type=int, help="template box, px (default: scaled from copia's 32 px / 300 A mask)")
    ap.add_argument("--mask-diameter", type=float,
                    help="A (default: scaled from copia's 300 A at an 82 A spacing, within --max-arc)")
    ap.add_argument("--max-arc", type=float, default=recipe.MAX_ARC_DEG,
                    help="deg; the default mask spans at most this much arc round the capsid (50)")
    ap.add_argument("--T", type=int, help="triangulation number, if known (hand still read from the map)")
    ap.add_argument("--hk", help="force the lattice, e.g. 2,1")
    ap.add_argument("--t-max", type=int, default=31, help="largest T tried when T is not given (31)")
    ap.add_argument("--lowpass", type=lowpass_arg, default="copia",
                    help="template low-pass: 'copia' (default; Gaussian FWHM 2 x pixel, 50%% at ~4.5 x pixel), "
                         "'nyquist' (Fourier, to 2 x pixel), or a resolution in A (Fourier)")
    ap.add_argument("--gap-radius", type=float, help="A; remove everything inside this radius (default: measured)")
    ap.add_argument("--relion-prefix", help="how RELION sees the output folder (default: the folder as given)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    if not args.emdb_id and not args.map:
        ap.error("give an EMDB id, --map FILE, or both")

    t0 = time.time()
    log = lambda s: print(s, flush=True)
    out = Path(args.out) if args.out else default_out(args.emdb_id, args.map)
    st = pipeline.load(out, emdb_id=args.emdb_id, map_path=args.map, apix=args.apix, log=log)
    try:
        pipeline.lattice(st, T=args.T, hk=args.hk, t_max=args.t_max, gap_radius=args.gap_radius, log=log)
    except ValueError as e:
        ap.error(str(e))
    params = pipeline.templates(st, out, apix=args.apix, box=args.box, mask_diameter=args.mask_diameter,
                                max_arc=args.max_arc, lowpass=args.lowpass, gap_radius=args.gap_radius,
                                relion_prefix=args.relion_prefix, write_figures=not args.no_figures, log=log)

    log(f"=== done in {time.time() - t0:.0f} s -> {out}/")
    for f in params["files"]["references"].values():
        log(f"  {f}")
    log(f"  {params['files']['mask']}\n  {params['files']['refs_star']}\n  {params['files']['refs_decoy_star']}\n  params.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
