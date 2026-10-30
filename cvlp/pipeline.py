"""The run, in three steps the command line and the GUI share.

    st = load(out, emdb_id="EMD-19708")          # or map_path="my_average.mrc", or both
    lattice(st)                                  # T (and hand) from the map, or given
    params = templates(st, out)                  # the copia cutout, written to out

load is the slow part (download, frame search, resampling) and runs once per
map; lattice and templates can be re-run with other settings on the same state.
"""
from __future__ import annotations

from pathlib import Path

from . import cutout as cut, emdb, figures, lattice as L, mapio, recipe, shell as S, symmetry


def _quiet(_):
    pass


def load(out, emdb_id=None, map_path=None, apix=11.8, log=print):
    """Fetch (or read) the map, find its icosahedral frame, resample into I1.

    With map_path and no emdb_id, the map is the user's own average: no
    metadata, no download. -> state dict.
    """
    if not emdb_id and not map_path:
        raise ValueError("give an EMDB id, a map file, or both")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    entry = None
    if emdb_id:
        eid = emdb.normalise_id(emdb_id)
        log(f"=== {eid}: entry")
        try:
            entry = emdb.fetch_entry(eid, out / "emdb")
            log(f"  {entry['title']}")
            log(f"  {entry['method']}, {entry['resolution']} A, point group {entry['point_group']}, "
                f"{entry['dims'][0]}^3 x {entry['apix']} A"
                + (f", T={entry['triangulation']} in entry" if entry["triangulation"] else ""))
            if entry["point_group"] and entry["point_group"].upper() != "I":
                log(f"  ! entry symmetry is {entry['point_group']}, not I; continuing, the map decides")
        except Exception as e:                                    # offline with a map file is fine
            if not map_path:
                raise
            log(f"  (no metadata: {e})")
    else:
        log(f"=== own map: {map_path}")
    map_path = Path(map_path) if map_path else emdb.download_map(entry, out / "emdb")
    vol, apix_map = emdb.read_map(map_path, max_apix=apix / 4)    # no finer than the centring grid
    log(f"  map {map_path}  used at {vol.shape[0]}^3 x {apix_map:.3f} A")

    log("=== icosahedral frame")
    frame = symmetry.find_frame(vol, apix_map, log=log)
    map_I1 = out / "map_I1.mrc"
    mapio.write_mrc(map_I1, symmetry.to_I1(vol, apix_map, frame), apix_map)
    mapio.load_map.cache_clear()
    return dict(out=out, entry=entry, map_path=map_path, map_apix=apix_map, frame=frame, map_I1=map_I1)


def lattice(st, T=None, hk=None, t_max=31, gap_radius=None, log=print):
    """Layers, capsid surface and lattice. Adds gap, surface, hk, scores, model
    to the state and returns it."""
    log("=== lattice")
    gap = S.find_layers(st["map_I1"], gap_radius=gap_radius)
    shell_path = st["out"] / "outer_shell_only.mrc"
    cut.cut_outer_shell(st["map_I1"], gap["gap"], shell_path)
    surface = S.surface_model(shell_path, gap["gap"])
    faceted = S.is_faceted(surface)
    log(f"  capsid surface: 5-fold vertex {surface['vertex']:.0f} A, face centre {surface['face']:.0f} A "
        f"({surface['vertex'] - surface['face']:+.0f} A: "
        + ("faceted, the lattice test follows the facets)" if faceted else "read as a sphere)"))
    (h, k), scores = _pick(st["entry"], shell_path, gap["outer"], surface if faceted else None,
                           T, hk, t_max, log)
    model = L.capsid_model(h, k)
    log(f"  {model['counts']}  rules {model['rules']}")
    chosen = [s for s in scores if (s["h"], s["k"]) == (h, k) and s.get("pass")]
    morphology = max(chosen, key=lambda s: s["score"])["morphology"] if chosen else None
    if morphology in ("dimers", "trimers"):
        where = "on the lattice edges" if morphology == "dimers" else "in the gaps between lattice sites"
        log(f"  note: the surface is clustered in {morphology} {where}; the templates are still centred on\n"
            f"        the lattice sites, so they show the junction of the {morphology} around each site, and the\n"
            f"        neighbour-contrast check (which expects blobs on the sites) can read negative")
    chk = cut.icosahedral_selfcheck(st["map_I1"], r_lo=gap["gap"], r_hi=gap["outer"] + 30)
    log("  I1 self-check: " + ", ".join(f"{a} {b:.3f}" for a, b in chk.items()))
    st.update(gap=gap, surface=surface, faceted=faceted, hk=(h, k), scores=scores, model=model,
              morphology=morphology, selfcheck=chk)
    return st


def _pick(entry, shell_path, r_outer, surface, T, hk, t_max, log, scores=None):
    """-> ((h, k), scores). `scores` (from an earlier lattice_scores run) skips the test."""
    if hk:
        h, k = (int(x) for x in str(hk).split(",")) if isinstance(hk, str) else hk
        log(f"  lattice forced: {L.label(h, k)}")
        return (h, k), []
    if T:
        cands = L.candidates(T=T)
        if not cands:
            raise ValueError(f"T={T} is not a Caspar-Klug number (h^2 + hk + k^2)")
        if len(cands) == 1:
            log(f"  T={T} given (one lattice)")
            return cands[0], []
        log(f"  T={T} given; reading the hand from the map:")
    else:
        allc = L.candidates(t_max)
        cands = S.plausible(allc, r_outer)
        t_top = max(a * a + a * b + b * b for a, b in cands)
        log(f"  testing {len(cands)} lattices up to T={t_top} (capsomers >= {S.MIN_SPACING:.0f} A apart at "
            f"r = {r_outer:.0f} A" + (f"; T={t_top + 1}-{t_max} would be closer)" if len(cands) < len(allc) else ")")
            + ":")
    if scores is None:
        scores = S.lattice_scores(shell_path, r_outer, cands, surface=surface, log=log)
    best, status = S.choose_lattice(scores)
    hint = entry.get("triangulation") if entry else None
    alt = [s for s in scores if hint and s["T"] == hint]
    if status == "pass" and alt and not T and best["T"] != hint:
        # The map's answer is weaker than the annotation if it is only a partial match, or a finer
        # lattice that contains the annotated one (sub-capsomer detail: P22's 420 subunits sit on the
        # 420 gaps of T=21). Then the annotation decides.
        ann = max(alt, key=lambda s: s["score"])
        fine = L.capsid_model(best["h"], best["k"])
        finer = any(L.contains(fine, L.capsid_model(a["h"], a["k"])) for a in alt)    # either hand
        if best["pass"] == "partial" or finer:
            log(f"  ! the map's best is {best['label']} ({best['morphology']}, "
                + ("a partial match" if best["pass"] == "partial" else f"a finer lattice containing T={hint}")
                + f"); using the EMDB entry's T={hint}")
            best = ann
    if status != "pass":
        why = ("only T=1 passes, and its 12 vertices stand out on almost any icosahedral map"
               if status == "only_T1" else "no lattice passes")
        if alt and not T:
            best = max(alt, key=lambda s: s["score"])
            log(f"  ! {why}; using the EMDB entry's T={hint}")
        elif status == "only_T1":
            log(f"  ! {why}: either a T=1 capsid, or the lattice does not show on the surface "
                f"(several layers, or capsomers the test cannot see) - check B_lattice_test.png, or give T/(h,k)")
        else:
            log(f"  ! {why}; best is {best['label']} - check B_lattice_test.png, or give T/(h,k)")
    log(f"  -> {best['label']}  ({best['morphology']} on the lattice: recall {best['recall']:.2f}, "
        f"precision {best['precision']:.2f}" + (", partial match" if best.get("pass") == "partial" else "") + ")")
    if hint and hint != best["T"]:
        log(f"  ! the EMDB entry says T={hint}. For a particle with several layers it may describe another "
            f"layer than the outermost one cvlp reads")
    return (best["h"], best["k"]), scores


def templates(st, out=None, apix=11.8, box=None, mask_diameter=None, max_arc=recipe.MAX_ARC_DEG,
              lowpass="copia", gap_radius=None, relion_prefix=None, write_figures=True, log=print):
    """The copia cutout into `out` (default: the state's folder). -> params."""
    out = Path(out or st["out"])
    out.mkdir(parents=True, exist_ok=True)
    log(f"=== templates ({apix} A/px)")
    if apix / 4 < st["map_apix"] - 1e-6:
        log(f"  note: the map was read at {st['map_apix']:.2f} A/px; for a centring grid of {apix / 4:.2f} A "
            f"reload it with this pixel size")
    params = recipe.build(st["map_I1"], out, st["model"], apix, box=box, mask_diameter=mask_diameter,
                          relion_prefix=relion_prefix if relion_prefix else str(out), gap_radius=gap_radius,
                          lowpass=lowpass, max_arc=max_arc, log=log)
    params["emdb"] = st["entry"]
    params["map"] = str(st["map_path"])
    f = st["frame"]
    params["symmetry"] = dict(setting=f["setting"], score=round(f["score"], 4),
                              rotation_to_I1=f["F"].round(6).tolist(),
                              centre_shift_A=f["shift_A"].round(3).tolist(), I1_selfcheck=st["selfcheck"])
    params["lattice_test"] = st["scores"]
    sm = st["surface"]
    params["surface"] = dict(vertex_A=round(sm["vertex"], 1), face_A=round(sm["face"], 1),
                             faceted=bool(st["faceted"]), coef=sm["coef"].round(3).tolist(),
                             orders=list(sm["orders"]))
    recipe.write_params(params, out / "params.json")
    if write_figures:
        log(f"  figures in {figures.write_all(params, st['scores'], out)}")
    return params
