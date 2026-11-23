"""Tests of spectrum/src/generate_spectra.py and of the shipped spectra (spectrum/output/spectra).

The shipped files come from decays_positron.x built against PENNUC 2018 (Mn52, Zr89 verbatim; F18, Ga68, I124, Rb82, Cu64
rounded to 3-4 digits by their authors, with a leading "3.0e6 0" row). Rebuilding with PENNUC 2023 (src/build.sh) gives
the same physics at the level of the statistical noise of 1e7 decays, which is what the slow tests assert.
"""
import numpy as np
import pytest

import generate_spectra as g

SPCNUC = """\
 #  Results from DECAYS. Radionuclide: X-1.nuc
 #  Total CPU simulation time = 1.0E+00 seconds
 #
 # 1st column: particle energy (eV)
   1.000000E+03  1.0E-05  1.0E-06  2.0E-05  1.0E-06  0.0E+00  1.0E-35
   2.000000E+03  1.0E-05  1.0E-06  2.0E-05  1.0E-06  3.0E-01  1.0E-06
   3.000000E+03  1.0E-05  1.0E-06  2.0E-05  1.0E-06  1.0E-01  1.0E-06
"""


def test_nuclide_file():
    assert g.nuclide_file("F18") == "F-18.nuc"
    assert g.nuclide_file("Ga68") == "Ga-68.nuc"
    assert g.nuclide_file("Ag108m") == "Ag-108m.nuc"
    for bad in ("18F", "F", "F-18", "../x"):
        with pytest.raises(ValueError):
            g.nuclide_file(bad)


def test_parse_and_format_hmc():
    rows = g.parse_spcnuc(SPCNUC)
    assert rows == [("1.000000E+03", 0.0), ("2.000000E+03", 0.3), ("3.000000E+03", 0.1)]   # energy keeps its text, comments dropped
    assert g.format_spectrum(rows) == "1.000000E+03\t0\n2.000000E+03\t0.75\n3.000000E+03\t0.25\n"


def test_format_peneasy_has_header_and_end_marker():
    lines = g.format_spectrum(g.parse_spcnuc(SPCNUC), "peneasy").splitlines()
    assert lines[0].startswith("# Energy(eV)")
    assert lines[-1] == "3.000000E+03\t-1"
    assert len(lines) == 3 + 2


def test_no_positrons_is_an_error():
    with pytest.raises(ValueError, match="positron"):
        g.format_spectrum([("1.0E+03", 0.0), ("2.0E+03", 0.0)])


def test_shipped_spectra_are_probability_tables(spectra_dir):
    files = sorted(spectra_dir.glob("*.txt"))
    assert {f.stem for f in files} >= {"Cu64", "F18", "Ga68", "I124", "Mn52", "Rb82", "Zr89"}
    for f in files:
        a = np.loadtxt(f)
        assert a.ndim == 2 and a.shape[1] == 2 and len(a) in (400, 401), f
        assert (a[:, 1] >= 0).all()
        assert abs(a[:, 1].sum() - 1.0) < 1e-3, f


def test_main_does_not_overwrite(tmp_path, capsys):
    (tmp_path / "F18.txt").write_text("keep\n")
    assert g.main(["F18", "--out", str(tmp_path)]) == 1
    assert (tmp_path / "F18.txt").read_text() == "keep\n"


@pytest.mark.slow
@pytest.mark.parametrize("iso", ["Mn52", "Zr89", "F18", "Rb82"])
def test_pennuc_reproduces_shipped_spectrum(pennuc, spectra_dir, iso):
    new = np.loadtxt(__import__("io").StringIO(pennuc.generate(iso)))
    old = np.loadtxt(spectra_dir / f"{iso}.txt")
    if len(old) == len(new) + 1:                       # hybrid-team files start with a "3.0e6 0" row
        old = old[1:]
    assert new.shape == old.shape
    assert np.abs(new[:, 0] / old[:, 0] - 1).max() < 6e-3          # rounded energies in F18/Rb82
    assert np.abs(new[:, 1] - old[:, 1]).sum() < 5e-4              # L1 over the bins (statistical noise of 1e7 decays)
    mean = lambda a: (a[:, 0] * a[:, 1]).sum() / a[:, 1].sum()
    assert mean(new) / mean(old) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.slow
def test_generate_is_deterministic(pennuc):
    assert pennuc.generate("Mn52", 1e6) == pennuc.generate("Mn52", 1e6)
