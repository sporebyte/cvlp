"""python -m pytest tests  (fast: no maps, no network)"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvlp import emdb, lattice as L          # noqa: E402
from cvlp import mapio                       # noqa: E402

# copia's T=9 model (teact capsid_model(9), 2026-10-30), frozen so the test
# needs no teact: any change to the construction or the site order shows here
COPIA_T9_FIRST = np.array([
    [-0.5257311121191336, 0.85065080835204, 0.0],
    [-0.40354821233519766, 0.8547288298252084, 0.3264773618288047],
    [-0.2017741061675989, 0.7300255741640024, 0.6529547236576094],
    [-4.722060566742181e-17, 0.5257311121191336, 0.85065080835204],
    [-0.7300255741640024, 0.6529547236576094, 0.2017741061675988],
])
COPIA_T9_SHA1 = ("898d61eb50272d65a3545cdf03e25525244208e6",    # directions, rounded to 1e-12
                 "9aafcc54d06da383b89069132dbe7893f22388d1")    # classes, int64


def test_group_has_60_rotations():
    G = L.icosahedral_group()
    assert len(G) == 60 and np.allclose(np.linalg.det(G), 1)


@pytest.mark.parametrize("hk", L.candidates(31))
def test_site_count_and_orbits(hk):
    m = L.capsid_model(*hk)
    assert len(m["directions"]) == 10 * m["T"] + 2
    assert m["counts"]["C5"] == 12
    assert sum(m["counts"].values()) == 10 * m["T"] + 2


def test_T9_is_copias_model():
    import hashlib
    m = L.capsid_model(3, 0)
    assert np.allclose(m["directions"][:5], COPIA_T9_FIRST)
    assert hashlib.sha1(np.round(m["directions"], 12).tobytes()).hexdigest() == COPIA_T9_SHA1[0]
    assert hashlib.sha1(m["classes"].astype(np.int64).tobytes()).hexdigest() == COPIA_T9_SHA1[1]
    assert m["representative"] == {"C5": 0, "C1": 1, "C3": 5}    # copia params.json sites
    assert L.decoy_pair(m) == (1, 10)                             # copia params_decoy.json


def test_hands_are_mirror_images_not_rotations():
    a, b = L.capsid_model(2, 1)["directions"], L.capsid_model(1, 2)["directions"]
    dist = lambda p, q: np.abs(p[:, None] - q[None]).sum(-1).min(1).max()
    assert dist(a, b) > 1e-3                           # different site sets in the same frame
    assert dist(a, b * np.array([-1, 1, 1])) < 1e-6    # x -> -x keeps I1 and swaps the hand


def test_reference_star_is_relion_readable(tmp_path):
    p = tmp_path / "refs.star"
    mapio.write_reference_star(p, ["a/ref_C1.mrc", "a/ref_C3.mrc"])
    assert p.read_text().splitlines()[1:] == ["data_model_classes", "", "loop_",
                                              "_rlnReferenceImage #1", "a/ref_C1.mrc", "a/ref_C3.mrc"]


def test_fourier_lowpass_passes_coarse_and_stops_fine():
    from cvlp import cutout as cut
    n, apix = 64, 4.0
    x = np.arange(n) * apix
    wave = lambda period: np.broadcast_to(np.cos(2 * np.pi * x / period), (n, n, n)).copy()
    for period, kept in ((64.0, 1.0), (32.0, 1.0), (25.6, 0.0)):    # cutoff 26 A, roll-off from 30.6 A
        out = cut.fourier_lowpass(wave(period), apix, 26.0)
        assert np.isclose(np.abs(out).max(), kept, atol=1e-3), period


def test_gaussian_half_amplitude_is_53A_for_copia():
    from cvlp import cutout as cut
    assert abs(cut.gaussian_half_amplitude(23.6) - 53.5) < 0.1


@pytest.mark.parametrize("l", (6, 10, 12))
def test_ico_harmonics_are_invariant_with_zero_mean(l):
    from cvlp import geometry as geo, shell as S
    u = geo.sphere_points(4000)
    f = S.ico_harmonic(u, l)
    for g in L.icosahedral_group()[1::7]:
        assert np.allclose(S.ico_harmonic(u @ g.T, l), f, atol=1e-9)
    assert abs(f.mean()) < 0.02 * f.std()
    v = geo.symmetry_axes()
    if l == 6:                                   # the faceting mode: vertices up, faces down
        assert S.ico_harmonic(v["5-fold"][:1], 6)[0] > 0 > S.ico_harmonic(v["3-fold"][:1], 6)[0]


def test_ids():
    assert emdb.normalise_id("emd_19708") == emdb.normalise_id("19708") == "EMD-19708"
