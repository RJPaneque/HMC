"""Unit tests for weight-fraction -> atoms/molecule conversion and
``build_material_input`` (the material.x stdin contract).

PENELOPE's ``material.x`` expects composition as atoms/molecule, normalised
so that the first element (lowest Z) among those present has exactly 1 atom.
"""

import pytest

from schneider_materials import (
    _weight_to_atoms,
    build_material_input,
    get_material,
    _Z_ORDER,
    _ATOMIC_MASS,
)


# ---------------------------------------------------------------------------
# _weight_to_atoms
# ---------------------------------------------------------------------------
class TestWeightToAtoms:
    def test_water_like_two_element(self):
        comp = {"H": 0.1119, "O": 0.8881}
        atoms = _weight_to_atoms(comp)
        assert atoms["H"] == pytest.approx(1.0)
        o_expected = (0.8881 / _ATOMIC_MASS["O"]) / (0.1119 / _ATOMIC_MASS["H"])
        assert atoms["O"] == pytest.approx(o_expected, rel=1e-9)

    def test_lowest_z_present_normalises_to_one(self):
        # composition with C, O, Ca present -- C is lowest Z, must be 1.0
        comp = {"C": 0.5, "O": 0.3, "Ca": 0.2}
        atoms = _weight_to_atoms(comp)
        assert atoms["C"] == pytest.approx(1.0)

    def test_drops_trace_elements(self):
        # elements with weight fraction <= 1e-4 (0.01%) are omitted
        comp = {"H": 0.5, "O": 0.5 - 1e-5, "Ca": 1e-5}
        atoms = _weight_to_atoms(comp)
        assert "Ca" not in atoms
        assert set(atoms) == {"H", "O"}

    def test_keeps_element_just_above_threshold(self):
        # 1e-3 fraction (0.1 wt%) survives
        comp = {"H": 0.999, "Ca": 1e-3}
        atoms = _weight_to_atoms(comp)
        assert "H" in atoms
        assert "Ca" in atoms

    def test_full_soft_tissue_material(self):
        m = get_material(hu=49)  # Soft_mean: many elements
        atoms = _weight_to_atoms(m.composition)
        # H is always lowest-Z in soft tissue compositions
        assert atoms["H"] == pytest.approx(1.0)
        # every element above the threshold is represented
        for el, w in m.composition.items():
            if w > 1e-4:
                assert el in atoms, f"{el} missing from atoms dict"

    def test_atomic_ratios_independent_of_normalisation(self):
        # doubling all weight fractions (still renormalised by the function)
        # must give identical atom ratios
        comp_a = {"H": 0.1, "C": 0.4, "O": 0.5}
        comp_b = {"H": 0.2, "C": 0.8, "O": 1.0}
        a = _weight_to_atoms(comp_a)
        b = _weight_to_atoms(comp_b)
        for el in a:
            assert a[el] == pytest.approx(b[el], rel=1e-9)


# ---------------------------------------------------------------------------
# build_material_input
# ---------------------------------------------------------------------------
class TestBuildMaterialInput:
    @pytest.fixture
    def sample(self):
        return get_material(hu=0)  # Soft_1 bin

    def test_overall_line_structure(self, sample):
        s = build_material_input(sample, "out.mat")
        lines = s.split("\n")
        assert lines[0] == "1"                       # keyboard input
        assert lines[1] == sample.label              # material name
        n = int(lines[2])
        assert lines[3] == "1"                        # stoichiometric formula flag
        # element block: lines[4:4+n]
        assert lines[4 + n] == "2"                    # keep calculated I-value
        assert lines[4 + n + 1] == f"{sample.density:.8E}"
        assert lines[4 + n + 2] == "2"                # keep Fcb/Wcb
        assert lines[4 + n + 3] == "out.mat"          # output file
        assert len(lines) == 4 + n + 4

    def test_n_elem_matches_composition(self, sample):
        lines = build_material_input(sample, "out.mat").split("\n")
        assert int(lines[2]) == len(sample.composition)

    def test_label_override(self, sample):
        s = build_material_input(sample, "out.mat", label="MyMaterial")
        assert s.split("\n")[1] == "MyMaterial"

    def test_label_truncated_to_60(self, sample):
        long_label = "X" * 80
        s = build_material_input(sample, "out.mat", label=long_label)
        assert s.split("\n")[1] == "X" * 60

    def test_element_lines_in_z_order(self, sample):
        s = build_material_input(sample, "out.mat")
        lines = s.split("\n")
        n = int(lines[2])
        elem_lines = lines[4:4 + n]
        zs = [int(L.split()[0]) for L in elem_lines]
        assert zs == sorted(zs), "Z values not in ascending order"
        expected_zs = [z for z, el in _Z_ORDER if el in sample.composition]
        assert zs == expected_zs

    def test_density_format_is_scientific(self, sample):
        lines = build_material_input(sample, "out.mat").split("\n")
        n = int(lines[2])
        density_line = lines[4 + n + 1]
        # %.8E format: e.g. 1.01800000E+00
        assert "E" in density_line
        assert density_line == f"{sample.density:.8E}"

    def test_complex_skeletal_material(self):
        m = get_material(hu=1000)  # skeletal bin with ~9 elements
        s = build_material_input(m, "/tmp/x.mat")
        lines = s.split("\n")
        n = int(lines[2])
        assert n >= 8  # skeletal bins are element-rich
        assert len(lines) == 4 + n + 4