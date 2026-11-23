"""Unit tests for PENELOPE ``.mat`` file I/O.

Covers:
  * _read_penelope_mat  -- parse density + elements from a real .mat file
  * match_penelope_file -- identify the Schneider material of a .mat file
                          (both the index fast-path and the composition
                          fall-back strategy)
  * generate_mat_file   -- integration: round-trip a material through
                          material.x and verify the result (slow)
"""

import pytest

from schneider_materials import (
    _read_penelope_mat,
    match_penelope_file,
    generate_mat_file,
    get_material,
)


# ---------------------------------------------------------------------------
# _read_penelope_mat
# ---------------------------------------------------------------------------
class TestReadPenelopeMat:
    def test_mat1_is_air(self, mats_dir):
        rho, wt = _read_penelope_mat(mats_dir / "mat1.mat")
        assert rho == pytest.approx(0.02675, rel=1e-6)
        assert set(wt) == {"N", "O", "Ar"}
        assert wt["N"] + wt["O"] + wt["Ar"] == pytest.approx(1.0, abs=1e-3)
        # N is the dominant component (~75.5 wt%)
        assert wt["N"] > 0.7

    def test_empty_file_returns_none(self, tmp_path):
        # _read_penelope_mat returns None when it cannot parse a density;
        # it raises FileNotFoundError if the path does not exist.
        p = tmp_path / "empty.mat"
        p.write_text("")
        assert _read_penelope_mat(p) is None

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _read_penelope_mat(tmp_path / "nope.mat")

    def test_all_30_files_parse(self, mats_dir):
        for i in range(1, 31):
            res = _read_penelope_mat(mats_dir / f"mat{i}.mat")
            assert res is not None, f"mat{i}.mat failed to parse"
            rho, wt = res
            assert rho > 0
            assert sum(wt.values()) == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.parametrize("i,expected_rho",
                             [(1, 0.02675), (5, 0.47995), (9, 0.926455),
                              (15, 1.0885), (30, 1.83695)])
    def test_density_matches_input(self, mats_dir, i, expected_rho):
        rho, _ = _read_penelope_mat(mats_dir / f"mat{i}.mat")
        assert rho == pytest.approx(expected_rho, rel=1e-6)


# ---------------------------------------------------------------------------
# match_penelope_file
# ---------------------------------------------------------------------------
class TestMatchPenelopeFile:
    def test_mat1_identified_as_air(self, mats_dir):
        m = match_penelope_file(str(mats_dir / "mat1.mat"))
        assert m is not None
        assert m.label == "Air"

    @pytest.mark.parametrize("i", [2, 3, 4, 5, 6, 7, 8])
    def test_lung_subdivisions_all_match_lung(self, mats_dir, i):
        # the 7 lung density levels all use the Lung composition
        m = match_penelope_file(str(mats_dir / f"mat{i}.mat"))
        assert m is not None
        assert m.label == "Lung"

    def test_mat9_adipose_1(self, mats_dir):
        m = match_penelope_file(str(mats_dir / "mat9.mat"))
        assert m.label == "Adipose_1"

    def test_mat17_index_strategy_aligns(self, mats_dir):
        # mat17's composition matches Table-6 row 17 -> strategy 1 (index)
        m = match_penelope_file(str(mats_dir / "mat17.mat"))
        assert m.label == "Skeletal_200_300"

    def test_mat15_composition_strategy_wins(self, mats_dir):
        # mat15 (rho=1.0885) -> Soft_mean composition under the density-only
        # workflow; its index (14 = Skeletal_500_600) no longer matches, so
        # the composition-based fall-back (strategy 2) must take over.
        m = match_penelope_file(str(mats_dir / "mat15.mat"))
        assert m.label == "Soft_mean"

    def test_mat30_composition_strategy_wins(self, mats_dir):
        # mat30 -> Skeletal_1300_1400 composition; index 29 would be
        # Skeletal_1500_1600 -> composition strategy must override.
        m = match_penelope_file(str(mats_dir / "mat30.mat"))
        assert m.label == "Skeletal_1300_1400"

    def test_matched_density_preserved_from_file(self, mats_dir):
        m = match_penelope_file(str(mats_dir / "mat1.mat"))
        # match returns the *file's* density, not the bin's canonical one
        assert m.density == pytest.approx(0.02675, rel=1e-6)


# ---------------------------------------------------------------------------
# generate_mat_file  (integration -- requires material.x + pdfiles/)
# ---------------------------------------------------------------------------
class TestGenerateMatFile:
    @pytest.mark.slow
    def test_round_trip_single_material(self, tmp_path, material_exe, penelope_workdir):
        mat = get_material(density=1.0)
        out = tmp_path / "rt.mat"
        generate_mat_file(mat, str(material_exe), str(out),
                          workdir=str(penelope_workdir))
        assert out.is_file()
        rho, wt = _read_penelope_mat(str(out))
        assert rho == pytest.approx(mat.density, rel=1e-7)
        # composition ratios should match what was fed in (normalised)
        s = sum(mat.composition.values())
        exp = {el: w / s for el, w in mat.composition.items()}
        for el in set(wt) | set(exp):
            assert wt.get(el, 0.0) == pytest.approx(exp.get(el, 0.0), abs=1e-4)

    @pytest.mark.slow
    def test_custom_label_propagates(self, tmp_path, material_exe, penelope_workdir):
        mat = get_material(density=0.5)
        out = tmp_path / "lbl.mat"
        generate_mat_file(mat, str(material_exe), str(out),
                          label="TestLabel42", workdir=str(penelope_workdir))
        header = out.read_text(errors="replace")[:400]
        assert "TestLabel42" in header