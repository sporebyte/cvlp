# cvlp: capsomer templates for icosahedral VLPs, from an EMDB id

Gives the copia capsomer-template cutout (`make_references_and_seeds.py`
sections A-C + `make_decoy_reference.py`) for any icosahedral capsid. You pass
an EMDB id and get one template per capsomer class, an edge decoy, the shared
mask and the RELION `--ref` star files, at your subtomogram pixel size. It only
makes templates: no seeding, and no tomogram needed.

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
cvlp EMD-10800 --apix 11.8 --box 32            # any other capsid / sampling
```

| option | default | |
|---|---|---|
| `--apix` | 11.8 | A/px of the subtomograms the templates are for |
| `--box` | scaled | px; copia's 32 px per 300 A mask, scaled with the mask |
| `--mask-diameter` | scaled | A; copia's 300 A at an 82 A capsomer spacing, scaled with the spacing (capped at the capsid diameter) |
| `--T` | detected | the triangulation number; the hand of a skewed lattice is still read from the map |
| `--hk 2,1` | | force one lattice |
| `--lowpass` | `copia` | template low-pass: `copia`, `nyquist`, or a resolution in A (see below) |
| `--gap-radius` | measured | remove everything inside this radius (inner layer / RNP) |
| `--relion-prefix` | the output folder | how RELION sees the output folder, used in the star files |
| `--map file.mrc` | | use a local map instead of downloading |
| `-o DIR` | `./EMD-xxxxx` | output folder |

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
   scale for two morphologies: **capsomers** on the sites (copia, papilloma), or
   **dimers** on the lattice edges (HBV, TCV, whose capsomer centres are pores).
   Coarser lattices also pass, so the finest one that passes wins. See
   `figures/B_lattice_test.png`.
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
  references/display/*_unfiltered.mrc     fine-grid cuts, no low-pass (for ChimeraX)
  mask_capsomer_32px.mrc
  refs_C1_C3_C5.star                      --ref list, capsomers only
  refs_C1_C3_C5_decoy.star                --ref list, capsomers + decoy
  params.json                             every number (copia's params.json keys + frame, lattice test, decoy)
  map_I1.mrc  outer_shell_only.mrc
  outer_shell_lowpass_23.6A.mrc           only with a Fourier --lowpass
  figures/A_map_profiles.png  B_lattice_test.png  C_templates.png
  emdb/                                   metadata + downloaded map
```

## Checked on

| entry | capsid | result |
|---|---|---|
| EMD-19708 | copia, T=9 | T=9; C1/C3/C5 correlate 0.995 / 0.993 / 0.985 with `seeding_v2/references`, decoy 0.983 with `ref_edge_decoy_v2` (same sites 1 & 10); box 32, mask 300 A, spacing 82.1 A |
| EMD-66072 | HPV45, T=7 | T=7 (h,k)=(1,2); the other hand scores 0.17 |
| EMD-10800 | duck HBV, T=4 | T=4, dimers |
| EMD-53397 | turnip crinkle virus, T=3 | T=3, dimers |
| EMD-66102 | PCV3, T=1 | T=1 |
| synthetic | T=7 both hands, 13 (1,3), 12, 16 | correct in random orientations, centre offset recovered to ~0.05 A |

`python -m pytest tests` runs the lattice checks; `tests/synthetic.py` makes
the synthetic maps.

## Caveats

- **Hand labels.** (h,k) is the lattice construction in the I1 frame, not a
  laevo/dextro label. The hand is read off the map as deposited, so a map
  deposited with the wrong hand gives the mirror lattice. That still matches the
  map.
- **Dimer-clustered capsids.** The templates show a pore ringed by dimers, and
  the decoy sits on a dimer. The neighbour-contrast check reads negative
  there, which is expected; the run prints a note.
- **Small or filled capsids** (T=1, T=3) only have room for a mask capped at the
  capsid diameter. If density fills the particle to the centre, nothing is cut
  unless you give `--gap-radius`.

## License

MIT, see [LICENSE](LICENSE).
