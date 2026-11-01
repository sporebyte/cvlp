# cvlp: capsomer templates for icosahedral VLPs, from an EMDB id or your own map

Gives the copia capsomer-template cutout (`make_references_and_seeds.py`
sections A-C + `make_decoy_reference.py`) for any icosahedral capsid. You pass
an EMDB id or a map file and get one template per capsomer class, an edge
decoy, the shared mask and the RELION `--ref` star files, at your subtomogram
pixel size. It only makes templates: no seeding, and no tomogram needed.

**When to use it.** Capsomer oversampling needs capsomer references. If you
have your own whole-VLP average, as for copia, give cvlp that map
(`--map`). When your data has too few whole VLPs to average first, but you
know the species, give it the species' EMDB id and it cuts the references
from the published map instead.

**Best on larger VLPs (T >= 7).** A useful template holds a capsomer and its
first neighbour ring, because the classes often differ only in their
neighbours. Capsomers sit 80-100 A apart in most capsids, so on a small capsid
that ring reaches far round the particle: +-38 deg at copia's T=9 and +-45 deg
at T=7, but +-60-78 deg at T=3-4, where every template is a quarter of the
particle. cvlp caps the mask at +-50 deg there and prints a note. For small
rigid capsids, averaging whole particles and expanding the symmetry is usually
the more reliable route.

Standalone: needs only numpy, scipy, mrcfile and matplotlib, which pip
installs with it. It does not import teact.

Install from GitHub, or from a local clone (editable, so changes apply
without reinstalling):

```bash
pip install git+https://github.com/sporebyte/cvlp.git
pip install -e ~/cvlp
```

```bash
cvlp EMD-19708                                 # copia T=9 -> C1, C3, C5 + decoy
cvlp EMD-19708 --relion-prefix surface_vlps_1808/seeding_v2
cvlp --map my_vlp_average.mrc                  # your own average, no EMDB entry
cvlp EMD-10800 --apix 11.8 --box 32            # any other capsid / sampling
cvlp-gui                                       # the same with a preview (below)
```

On your own copia average (`vlp.mrc`, the map `seeding_v2` was cut from),
`cvlp --map vlp.mrc` finds T=9 by itself and its templates and mask correlate
0.998-0.999 with `seeding_v2`. The rest is a half-voxel recentring of the map.

### GUI

`cvlp-gui` (or `python -m cvlp.gui`) does the same run in three steps:

1. **Load** an EMDB id or a map file (or both: a local copy of an entry).
   This downloads the map, finds the frame and the lattice, and takes about a
   minute, once per map. The detected T, the classes and the EMDB annotation
   are shown. You can override T or (h,k) and re-check the lattice.
2. **Preview** with the current parameters (pixel size, box, mask, max arc,
   low-pass, gap radius, RELION prefix). Each class is shown as the low-passed
   template with the mask outline, and as the unfiltered cutout from the map,
   with the box, mask and how distinct the templates are (their correlations).
   A re-preview takes 20-60 s.
   **Open figures folder** shows the QC figures in Finder. `A_map_profiles.png`
   has the radial density of the map you are cutting from: its spherical
   average with the inner layer, gap cut, capsid layer and mask band marked,
   and the radial profile on each capsomer class's sites. `B_lattice_test.png`
   has the lattice test and `C_templates.png` the templates.
3. **Save** writes the previewed templates, unfiltered references, mask, star
   files, `params.json` and figures to the output folder.

The preview runs the same code as the command line, into a hidden `.preview`
folder, so the saved files are identical to `cvlp` with the same settings.

| option | default | |
|---|---|---|
| `--apix` | 11.8 | A/px of the subtomograms the templates are for |
| `--box` | scaled | px; copia's 32 px per 300 A mask, scaled with the mask |
| `--mask-diameter` | scaled | A; copia's 300 A at an 82 A capsomer spacing, scaled with the spacing, within `--max-arc` |
| `--max-arc` | 50 | deg; the default mask reaches at most this far round the capsid (see below) |
| `--T` | detected | the triangulation number; the hand of a skewed lattice is still read from the map |
| `--hk 2,1` | | force one lattice |
| `--lowpass` | `copia` | template low-pass: `copia`, `nyquist`, or a resolution in A (see below) |
| `--gap-radius` | measured | remove everything inside this radius (inner layer / RNP) |
| `--relion-prefix` | the output folder | how RELION sees the output folder, used in the star files |
| `--map file.mrc` | | your own map; with an EMDB id, a local copy of the entry's map |
| `-o DIR` | `./EMD-xxxxx` or the map's name | output folder |

## Code borrowed from teact

The geometry and cutout functions were copied from teact
(`sociology/vlp_oversample.py`, `sociology/orientations.py`, `io/tomogram.py`)
without changes, so the templates come out the same as copia's. On EMD-19708
the standalone version writes bit-identical templates and mask to the
teact-backed one. Where they are:

| cvlp module | from teact |
|---|---|
| `geometry.py` | `icosahedron`, `symmetry_axes`, `sphere_points`, `frame` |
| `cutout.py` | `sample`, `radial_profile`, `icosahedral_selfcheck` (band as arguments), `cut_outer_shell`, `cut_reference`, `bud_offset`, `centre_site_radius`, `normalise_in_mask`, `capsomer_mask`, `ring_peaks`, `neighbour_contrast` |
| `mapio.py` | `load_map`, `write_mrc`; the `--ref` star is written as plain text (no starfile/pandas) |
| `lattice.py` | `capsid_model`'s construction, extended to both hands and per-orbit classes |
| `shell.py` | `find_shell_gap`, `site_profiles`, generalised to any capsid size |

If teact's versions change, cvlp keeps the copies it has.

## What it does

1. **Fetch.** Gets the EMDB metadata and the primary map (cached in `emdb/`). Maps
   sampled finer than apix/4 (the centring grid, 2.95 A for 11.8) are block-averaged
   down to it.
2. **Frame.** Finds the icosahedral setting (5-, 3- or 2-fold on z, any spin; if
   none fits, a global search), refines the rotation and the centre, and resamples
   into RELION I1 (5-fold on (0,1,phi), centre at (n-1)/2) -> `map_I1.mrc`.
3. **Lattice.** Unless `--T`/`--hk` is given, it tries every (h,k) up to T=31,
   both hands. A lattice passes if the capsid surface has a blob on every site and
   nowhere else, with recall and precision >= 0.8. Each lattice is tested at its own
   scale for three morphologies: **capsomers** on the sites (copia, papilloma,
   herpes), **dimers** on the lattice edges (HBV, TCV, e15, whose capsomer
   centres are pores), or **trimers** in the gaps between sites (birnavirus).
   A lattice also passes as a partial match if nearly all maxima are on it
   (precision >= 0.95) and they cover at least half of it; some capsomer kinds
   are too flat to make a maximum (Sputnik). Coarser lattices also pass, so the
   finest one that passes wins. T=1 is not counted as evidence: the 12 vertices
   stand out on almost any icosahedral map. If only T=1 passes, cvlp uses the
   EMDB entry's T when there is one, and otherwise says so. See
   `figures/B_lattice_test.png`. The surface is read relative to the capsid's
   own shape, not at one radius: faceted capsids such as mature phage heads are
   icosahedra, and T7 (EMD-1810) sits ~30 A further out on its vertices than
   in its face centres. cvlp fits the capsid's inner surface with the lowest
   icosahedral harmonics and logs the vertex and face-centre radii. If they
   differ by more than 5% of the radius (T7 11%, TCV 10%), the test follows the
   facets; otherwise the capsid is read as a sphere (copia, HPV45, DHBV, HAV and
   PCV3 are all at or below 3%). Below that, the fit is mostly capsomer relief,
   especially on small capsids where the capsomer pattern sits in the same low
   harmonics, and following it hides the capsomers.
4. **Templates.** This is copia's recipe, using the borrowed functions above: shell
   gap -> `outer_shell_only.mrc` -> site profiles -> one reference per capsomer
   orbit (radius centred on the fine grid, cut at `--apix`, low-passed,
   normalised in one shell-section mask) -> decoy on the saddle between two
   neighbouring capsomers of the most numerous class.

Capsomer classes are icosahedral orbits named by local symmetry: C5 (12
pentamers), C3 (20, on 3-folds), C2 (30, on 2-folds), C1 (60, general
position; C1a, C1b, ... when a lattice has several, e.g. T=13, 16, 19).

### Low-pass

| `--lowpass` | filter | at 11.8 A/px: 50% at | nothing finer than |
|---|---|---|---|
| `copia` (default) | real-space Gaussian, FWHM 2 x pixel, as in the copia run | 53.5 A | (Gaussian tail; 3% left at 23.6 A) |
| `nyquist` | Fourier, raised-cosine edge over the last 15% | 25.5 A | 23.6 A |
| e.g. `30` | Fourier, as above | 32.4 A | 30 A |

The default reproduces copia's templates exactly. That filter is much stronger
than its "2 x pixel" suggests, and the templates come out similar to each
other: on hepatitis A (EMD-55772), C3, C5 and the decoy correlate 0.91-0.93
with `copia` and 0.82-0.86 with `nyquist`. Use `nyquist` for templates as
sharp as the pixel allows. A Fourier low-pass is applied to the shell map once
(`outer_shell_lowpass_<res>A.mrc`) before cutting, so resampling onto the
template grid cannot alias. Values finer than Nyquist are refused.

## Output

```
EMD-19708/
  references/ref_C1_32px_11.8A.mrc  ref_C3_...  ref_C5_...  ref_decoy_32px_11.8A.mrc
  references/unfiltered/ref_C1_128px_2.95A.mrc ...   the same cuts unfiltered, for looking at (below)
  mask_capsomer_32px.mrc
  refs_C1_C3_C5.star                      --ref list, capsomers only
  refs_C1_C3_C5_decoy.star                --ref list, capsomers + decoy
  params.json                             every number (copia's params.json keys + frame, lattice test, decoy)
  map_I1.mrc  outer_shell_only.mrc
  outer_shell_lowpass_23.6A.mrc           only with a Fourier --lowpass
  figures/A_map_profiles.png  B_lattice_test.png  C_templates.png
  emdb/                                   metadata + downloaded map
```

The unfiltered references are plain cutouts for looking at in ChimeraX or
any other viewer, not for RELION. They have the same centre, orientation and box (in A) as each
template, on the fine grid (`--apix`/4, 2.95 A for 11.8), with no low-pass,
mask or normalisation. So they keep the map's own density scale, and the EMDB
entry's contour level works on them. They are cut from `outer_shell_only.mrc`,
so the inner layer is removed as in the templates. They are not offered at the
template pixel size: an unfiltered cut onto an 11.8 A grid would alias.
`figures/C_templates.png` shows them under the templates.

## Checked on

| entry | capsid | result |
|---|---|---|
| EMD-19708 | copia, T=9 | T=9; C1/C3/C5 correlate 0.995 / 0.993 / 0.985 with `seeding_v2/references`, decoy 0.983 with `ref_edge_decoy_v2` (same sites 1 & 10); box 32, mask 300 A, spacing 82.1 A |
| EMD-66072 | HPV45, T=7 | T=7 (h,k)=(1,2); the other hand scores 0.17 |
| EMD-10800 | duck HBV, T=4 | T=4, dimers |
| EMD-53397 | turnip crinkle virus, T=3 | T=3, dimers |
| EMD-66102 | PCV3, T=1 | T=1 |
| EMD-55772 | hepatitis A virus, pseudo-T=3 | T=3 |
| EMD-1810 | phage T7 empty head, T=7 (faceted by 30 A) | T=7 (h,k)=(2,1); C5 cut at 305 A on the vertices, C1 at 282 A on the faces. Read at one radius it gave T=1 |
| synthetic | T=7 both hands, 13 (1,3), 12, 16 | correct in random orientations, centre offset recovered to ~0.05 A |
| `vlp.mrc` (own map) | copia's own average | T=9 found without an entry; templates and mask 0.998-0.999 with `seeding_v2` |

### Icosahedral entries with T >= 7

15 entries that EMDB itself annotates as point group I with T >= 7, across
families and sizes, including two subtomogram averages:

| entry | capsid | annotated | cvlp | how |
|---|---|---|---|---|
| EMD-22101 | HK97 prohead | 7 | 7 (2,1) | map: capsomers |
| EMD-1557 | phage e15 | 7 | 7 (2,1) | map: dimers |
| EMD-5182 | SV40 VLP | 7 | 7 (1,2) | map: capsomers |
| EMD-3174 | P22 expanded capsid, 14.5 A | 7 | 7 (1,2) | annotation; the map's best (T=21 trimers) holds T=7: its 420 subunits sit on T=21's gaps |
| EMD-70800 | phage D3 prohead | 9 | 9 | map: capsomers |
| EMD-0299 | phi6 nucleocapsid | 13 | 13 (3,1) | annotation; the map gives only a partial T=3 |
| EMD-5353 | birnavirus | 13 | 13 (1,3) | annotation; only T=1 passes |
| EMD-49465 | VZV capsid in situ (subtomogram average) | 16 | 16 | map: capsomers |
| EMD-1011 | phage PRD1 | 25 | 25 | annotation; only T=1 passes |
| EMD-1662 | Sputnik virophage | 27 | 27 | map: partial match (140 maxima, all on the 272 sites) |
| EMD-1177 | Broadhaven virus (orbivirus) | 13 | 7 (2,1) + warning | map: its outer layer, which is not the annotated T=13 layer |
| EMD-16762 | rotavirus SLP in situ | 13 | 1 + warning | the map is only the VP2 shell (T=1); 13 is the whole virion |
| EMD-1507, EMD-5012 | phage lambda | 7 | refused | deposited as a slab through the particle, not a whole capsid |
| EMD-2054 | mud crab reovirus | 13 | refused | the same |

So the map alone settles T on 8 of the 12 usable entries. Where it is
inconclusive or weak (PRD1's three-tower trimers, birnavirus and phi6 at
12-16 A, P22's subunits), the entry's annotation is used, and the run says so.
For your own map with no annotation, check `figures/B_lattice_test.png`, and
give `--T` if the answer is one of those cases.

`python -m pytest tests` runs the lattice checks; `tests/synthetic.py` makes
the synthetic maps.

## Caveats

- **The outermost layer.** cvlp reads the capsid's outermost protein layer. For
  particles with several (reo-, rota-, orbiviruses), an EMDB entry's T may
  describe another layer; cvlp warns when they disagree.
- **Capsomer spacing.** Only lattices whose capsomers are at least 60 A apart
  at the capsid's radius are tried (real ones: 76-150 A). Finer ones would
  match subunits, not capsomers.

- **Hand labels.** (h,k) is the lattice construction in the I1 frame, not a
  laevo/dextro label. The hand is read off the map as deposited, so a map
  deposited with the wrong hand gives the mirror lattice. That still matches the
  map.
- **Dimer-clustered capsids.** The templates show a pore ringed by dimers, and
  the decoy sits on a dimer. The neighbour-contrast check reads negative
  there, which is expected; the run prints a note.
- **Small capsids (T=1-4).** The mask is sized to hold a capsomer and its first
  neighbour ring (copia: 1.83 spacings in radius, +-38 deg, 10% of the
  surface). On a small capsid the ring sits much further round the particle,
  so the same rule would span +-60 deg or more and a quarter of the capsid. The
  default mask is therefore capped at +-50 deg of arc (18% of the surface);
  T>=7 capsids are not affected. Measured with Nyquist-filtered templates, the
  cap changes how distinct they are by <= 0.05 in correlation:

  | capsid | T | uncapped: arc / surface | capped | class corr. | decoy corr. |
  |---|---|---|---|---|---|
  | HAV EMD-55772 | 3 | 61 deg / 26% | 50 deg / 18% | 0.86 -> 0.85 | 0.84 -> 0.83 |
  | TCV EMD-53397 | 3 | 64 deg / 28% | 50 deg / 18% | 0.23 -> 0.26 | 0.29 -> 0.30 |
  | DHBV EMD-10800 | 4 | 64 deg / 28% | 50 deg / 18% | 0.42 -> 0.47 | 0.33 -> 0.34 |
  | HPV45 EMD-66072 | 7 | 45 deg / 14% | not capped | 0.44 | 0.30 |
  | copia EMD-19708 | 9 | 38 deg / 10% | not capped | 0.81 | 0.62 |

  Shrinking the mask further does not help in general: copia's hexamers (0.97)
  and HPV's pentamers (0.78) can only be told apart by their neighbours. At
  11.8 A/px a T=1-4 capsid is only 15-30 px across; for small rigid capsids
  whole-particle averaging plus symmetry expansion is usually the more
  reliable route.
- **Filled capsids.** If density fills the particle to the centre, nothing is
  cut unless you give `--gap-radius`.

## License

MIT, see [LICENSE](LICENSE).
