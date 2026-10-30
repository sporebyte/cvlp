"""cvlp: capsomer templates for an icosahedral VLP, straight from EMDB.

    python -m cvlp EMD-19708                    # copia: T=9 -> C1, C3, C5 + decoy
    python -m cvlp EMD-19708 --apix 11.8 --box 32 --relion-prefix surface_vlps_1808/seeding_v2
    python -m cvlp EMD-1234 --T 7               # T given; the hand is still read from the map
    python -m cvlp EMD-1234 --hk 1,2            # force one lattice
    python -m cvlp EMD-1234 --map local.mrc     # use a local copy of the map

Steps: download the map -> find its icosahedral frame and centre -> resample
into RELION I1 -> pick the lattice (T, and the hand if it is skewed) ->
the copia template cutout (one reference per capsomer orbit + an edge decoy +
the shared mask + the RELION --ref lists).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import emdb, lattice as L, recipe, shell as S, symmetry
from . import cutout as cut, mapio


def pick_lattice(args, entry, shell_path, r_outer, log):
    if args.hk:
        h, k = (int(x) for x in args.hk.split(","))
        log(f"  lattice forced: {L.label(h, k)}")
        return (h, k), []
    if args.T:
        cands = L.candidates(T=args.T)
        if not cands:
            raise SystemExit(f"T={args.T} is not a Caspar-Klug number (h^2 + hk + k^2)")
        if len(cands) == 1:
            log(f"  T={args.T} given (one lattice)")
            return cands[0], []
        log(f"  T={args.T} given; reading the hand from the map:")
    else:
        cands = L.candidates(args.t_max)
        log(f"  testing {len(cands)} lattices up to T={args.t_max}:")
    scores = S.lattice_scores(shell_path, r_outer, cands, log=log)
    best, passed = S.choose_lattice(scores)
    hint = entry.get("triangulation") if entry else None
    if not passed:
        if hint and not args.T:
            alt = [s for s in scores if s["T"] == hint]
            best = max(alt, key=lambda s: s["score"]) if alt else best
            log(f"  ! no lattice passes cleanly; using the EMDB entry's T={hint}")
        else:
            log(f"  ! no lattice passes cleanly; best is {best['label']} - check B_lattice_test.png, or give --T/--hk")
    log(f"  -> {best['label']}  ({best['morphology']} on the lattice: "
        f"recall {best['recall']:.2f}, precision {best['precision']:.2f})")
    if hint and hint != best["T"]:
        log(f"  ! the EMDB entry says T={hint}")
    return (best["h"], best["k"]), scores


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("emdb_id", help="EMDB entry, e.g. EMD-19708")
    ap.add_argument("--out", "-o", help="output folder (default: ./<EMDB id>)")
    ap.add_argument("--map", help="use this map file instead of downloading the entry's primary map")
    ap.add_argument("--apix", type=float, default=11.8, help="A/px of the subtomograms the templates are for (11.8)")
    ap.add_argument("--box", type=int, help="template box, px (default: scaled from copia's 32 px / 300 A mask)")
    ap.add_argument("--mask-diameter", type=float, help="A (default: scaled from copia's 300 A at an 82 A spacing)")
    ap.add_argument("--T", type=int, help="triangulation number, if known (hand still read from the map)")
    ap.add_argument("--hk", help="force the lattice, e.g. 2,1")
    ap.add_argument("--t-max", type=int, default=31, help="largest T tried when T is not given (31)")
    ap.add_argument("--gap-radius", type=float, help="A; remove everything inside this radius (default: measured)")
    ap.add_argument("--relion-prefix", help="how RELION sees the output folder (default: the folder as given)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    log = lambda s: print(s, flush=True)
    eid = emdb.normalise_id(args.emdb_id)
    out = Path(args.out or eid)
    out.mkdir(parents=True, exist_ok=True)

    log(f"=== {eid}: entry")
    entry = None
    try:
        entry = emdb.fetch_entry(eid, out / "emdb")
        log(f"  {entry['title']}")
        log(f"  {entry['method']}, {entry['resolution']} A, point group {entry['point_group']}, "
            f"{entry['dims'][0]}^3 x {entry['apix']} A" + (f", T={entry['triangulation']} in entry" if entry["triangulation"] else ""))
        if entry["point_group"] and entry["point_group"].upper() != "I":
            log(f"  ! entry symmetry is {entry['point_group']}, not I; continuing, the map decides")
    except Exception as e:                                        # offline with --map is fine
        if not args.map:
            raise
        log(f"  (no metadata: {e})")
    map_path = Path(args.map) if args.map else emdb.download_map(entry, out / "emdb")
    vol, apix_map = emdb.read_map(map_path, max_apix=args.apix / 4)    # no finer than the centring grid
    log(f"  map {map_path}  used at {vol.shape[0]}^3 x {apix_map:.3f} A")

    log("=== icosahedral frame")
    frame = symmetry.find_frame(vol, apix_map, log=log)
    map_I1 = out / "map_I1.mrc"
    mapio.write_mrc(map_I1, symmetry.to_I1(vol, apix_map, frame), apix_map)
    mapio.load_map.cache_clear()
    del vol

    log("=== lattice")
    gap = S.find_layers(map_I1, gap_radius=args.gap_radius)
    shell_tmp = out / "outer_shell_only.mrc"
    cut.cut_outer_shell(map_I1, gap["gap"], shell_tmp)
    (h, k), scores = pick_lattice(args, entry, shell_tmp, gap["outer"], log)
    model = L.capsid_model(h, k)
    log(f"  {model['counts']}  rules {model['rules']}")
    chosen = next((s for s in scores if (s["h"], s["k"]) == (h, k) and s["score"] >= 0.8), None)
    if chosen and chosen["morphology"] == "dimers":
        log("  note: the surface is clustered in dimers on the lattice edges; capsomer centres are pores, so the\n"
            "        templates show a pore ringed by dimers, the decoy is centred on a dimer, and the\n"
            "        neighbour-contrast check (which expects blobs on the sites) reads negative")
    chk = cut.icosahedral_selfcheck(map_I1, r_lo=gap["gap"], r_hi=gap["outer"] + 30)
    log("  I1 self-check: " + ", ".join(f"{k} {v:.3f}" for k, v in chk.items()))

    log(f"=== templates ({args.apix} A/px)")
    params = recipe.build(map_I1, out, model, args.apix, box=args.box, mask_diameter=args.mask_diameter,
                          relion_prefix=args.relion_prefix, gap_radius=args.gap_radius, log=log)
    params["emdb"] = entry
    params["map"] = str(map_path)
    params["symmetry"] = dict(setting=frame["setting"], score=round(frame["score"], 4),
                              rotation_to_I1=frame["F"].round(6).tolist(),
                              centre_shift_A=frame["shift_A"].round(3).tolist(), I1_selfcheck=chk)
    params["lattice_test"] = scores
    recipe.write_params(params, out / "params.json")
    if not args.no_figures:
        try:
            d = __import__("cvlp.figures", fromlist=["write_all"]).write_all(params, scores, out)
            log(f"  figures in {d}")
        except ImportError as e:
            log(f"  (no figures: {e})")

    log(f"=== done in {time.time() - t0:.0f} s -> {out}/")
    for n, f in params["files"]["references"].items():
        log(f"  {f}")
    log(f"  {params['files']['mask']}\n  {params['files']['refs_star']}\n  {params['files']['refs_decoy_star']}\n  params.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
