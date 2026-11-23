"""Tests of spectrum/ensdf2nuc/ (ENSDF -> .nuc), of spectrum/input/extra/Y-86.nuc and of the Y-86 spectrum made from it.

Fast: the structure of Y-86.nuc (check_nuc.py, which does not import the converter), its β+ branches, the extra-folder overlay of generate_spectra.
Slow: the Y-86 spectrum through PENNUC, and re-running the converter on ensdf/Y86.3.ensdf (needs numba, scipy and the Geant4 PhotonEvaporation data).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import generate_spectra as g

ROOT = Path(__file__).resolve().parent.parent            # spectrum/
E2N = ROOT / "ensdf2nuc"
Y86 = ROOT / "input" / "extra" / "Y-86.nuc"
Y86_ENSDF = E2N / "ensdf" / "Y86.3.ensdf"

sys.path.insert(0, str(E2N))


def bep_records(path):
    """[(level number, probability, endpoint keV, forbiddenness)] of the BEP records."""
    out = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("BEP"):
            f = [s.strip() for s in line[3:].split(";")]
            out.append((int(f[2]), float(f[0]), float(f[3]), int(f[5])))
    return out


def test_y86_nuc_is_structurally_sound(capsys):
    import check_nuc
    assert check_nuc.check(str(Y86), str(Y86_ENSDF)), capsys.readouterr().out


def test_y86_beta_plus_branches():
    b = bep_records(Y86)
    assert len(b) == 33
    assert sum(p for _, p, _, _ in b) == pytest.approx(0.2766, abs=1e-3)           # ENSDF: %β+ = 27.2 (20)
    main = max(b, key=lambda r: r[1])
    assert main[2] == pytest.approx(1221.19, abs=0.01) and main[1] == pytest.approx(0.1037, abs=1e-4)   # level 2996.81 keV, ENSDF Iβ+ 10.39 %
    top = max(b, key=lambda r: r[2])
    assert top[2] == pytest.approx(3141.37, abs=0.01) and top[3] == 1             # level 1076.63 keV, first-forbidden, endpoint Q - 1022 - E_level
    for _, p, e, _ in b:
        assert 0 < e < 5240 - 1022 and p > 0


def test_extra_nuclides_overlay(tmp_path):
    db, extra = tmp_path / "db", tmp_path / "extra"
    db.mkdir(); extra.mkdir()
    (db / "F-18.nuc").write_text("db F18")
    (db / "Y-86.nuc").write_text("db Y86")
    (db / "pdatconf.p14").write_text("atomic data")
    (extra / "Y-86.nuc").write_text("extra Y86")
    (extra / "notes.txt").write_text("not a nuc file")
    g.link_nuclides(tmp_path / "scratch", db, extra)
    s = tmp_path / "scratch"
    assert sorted(p.name for p in s.iterdir()) == ["F-18.nuc", "Y-86.nuc", "pdatconf.p14"]       # the database's atomic data come along, extra's non-.nuc do not
    assert (s / "Y-86.nuc").read_text() == "extra Y86" and (s / "F-18.nuc").read_text() == "db F18"
    g.link_nuclides(tmp_path / "scratch2", db, None)
    assert (tmp_path / "scratch2" / "Y-86.nuc").read_text() == "db Y86"


@pytest.mark.slow
def test_y86_spectrum_through_pennuc(pennuc):
    rows = pennuc.parse_spcnuc(pennuc.run_decays("Y86", 1e6))
    e = np.array([float(x) for x, _ in rows]) / 1e3
    p = np.array([v for _, v in rows]); p /= p.sum()
    assert (e * p).sum() == pytest.approx(613, rel=0.03)                              # ENSDF per-level mean energies: 592-607 keV
    assert e[np.searchsorted(np.cumsum(p), 1 - 1e-4)] < 3200                           # no positrons above the 3141 keV endpoint
    assert p[e > 3200].sum() < 1e-6
    shipped = np.loadtxt(ROOT / "output" / "spectra" / "Y86.txt")
    assert shipped[:, 1].sum() == pytest.approx(1.0, abs=1e-3)


@pytest.mark.slow
def test_converter_reproduces_y86_nuc(tmp_path):
    pytest.importorskip("numba")
    pytest.importorskip("scipy")
    cands = [os.environ.get("PHOTON_EVAPORATION", "")] + [str(p) for p in (ROOT.parent / ".pixi" / "envs").glob("*/lib/python*/site-packages/opengate_core/geant4_data/PhotonEvaporation*")]
    g4 = next((c for c in cands if c and Path(c).is_dir()), None)
    if g4 is None:
        pytest.skip("no Geant4 PhotonEvaporation data (set PHOTON_EVAPORATION)")
    import ensdf2nuc
    lines = ensdf2nuc.convert(str(Y86_ENSDF), g4, "balance", log=lambda *a: None)
    assert "\n".join(lines) + "\n" == Y86.read_text()
