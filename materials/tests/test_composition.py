"""Unit tests for Schneider tissue composition lookup.

Covers:
  * get_all_materials() -- the 24 Table-6 bins
  * get_material(hu=...) -- bin assignment + composition per HU
  * get_material(density=...) -- the density->HU->bin path
  * Table 6 integrity (row sums, regression for the typos we fixed)
"""

import pytest

from schneider_materials import (
    get_material,
    get_all_materials,
    hu_to_density,
    density_to_hu,
    _schneider_table6,
    _ALL_ELEMENTS,
    _composition_from_entry,
)


# ---------------------------------------------------------------------------
# get_all_materials
# ---------------------------------------------------------------------------
class TestGetAllMaterials:
    def test_returns_24(self):
        assert len(get_all_materials()) == 24

    def test_labels_unique(self):
        labels = [m.label for m in get_all_materials()]
        assert len(set(labels)) == 24

    def test_hu_ranges_non_overlapping(self):
        # bins are integer-HU based; consecutive bins do not strictly
        # touch in the reals (e.g. Adipose_1 ends at -83, Adipose_2 starts
        # at -82), but they must not overlap.
        mats = get_all_materials()
        for a, b in zip(mats, mats[1:]):
            assert a.h_max <= b.h_min

    def test_all_integer_hus_assigned(self):
        # every integer HU in [-1000, 1599] must map to a real bin (no
        # fall-through to the Air fallback for non-air HUs).
        for h in range(-949, 1600):
            assert get_material(hu=h).label != "Air"

    def test_first_is_air(self):
        assert get_all_materials()[0].label == "Air"

    def test_last_is_highest_skeletal(self):
        assert get_all_materials()[-1].label == "Skeletal_1500_1600"

    @pytest.mark.parametrize("idx", range(24))
    def test_each_matches_table6_entry(self, idx):
        entry = _schneider_table6[idx]
        m = get_all_materials()[idx]
        assert m.label == entry["label"]
        assert m.h_min == entry["h_min"]
        assert m.h_max == entry["h_max"]


# ---------------------------------------------------------------------------
# get_material(hu=...)
# ---------------------------------------------------------------------------
class TestGetMaterialByHu:
    @pytest.mark.parametrize("idx", range(24))
    def test_midpoint_picks_correct_bin(self, idx):
        entry = _schneider_table6[idx]
        hu_mid = (entry["h_min"] + entry["h_max"]) / 2.0
        m = get_material(hu=hu_mid)
        assert m.label == entry["label"]
        assert m.h_min == entry["h_min"]
        assert m.h_max == entry["h_max"]

    @pytest.mark.parametrize("idx", range(24))
    def test_density_matches_calibration(self, idx):
        entry = _schneider_table6[idx]
        hu_mid = (entry["h_min"] + entry["h_max"]) / 2.0
        m = get_material(hu=hu_mid)
        assert m.density == pytest.approx(hu_to_density(hu_mid))

    @pytest.mark.parametrize("idx", range(24))
    def test_composition_matches_table6(self, idx):
        entry = _schneider_table6[idx]
        expected = _composition_from_entry(entry, _ALL_ELEMENTS)
        m = get_material(hu=(entry["h_min"] + entry["h_max"]) / 2.0)
        assert m.composition == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("idx", range(24))
    def test_composition_all_positive(self, idx):
        entry = _schneider_table6[idx]
        m = get_material(hu=(entry["h_min"] + entry["h_max"]) / 2.0)
        assert m.composition, "composition empty"
        for el, w in m.composition.items():
            assert w > 0.0, f"{el}={w} in {m.label} should be positive"

    def test_below_range_clamps_to_air(self):
        assert get_material(hu=-2000).label == "Air"

    def test_above_range_clamps_to_last_bin(self):
        assert get_material(hu=9999).label == "Skeletal_1500_1600"

    def test_hu_at_bin_boundary_goes_to_lower_bin_convention(self):
        # half-open intervals [h_min, h_max): HU=-120 -> Adipose_1, not Lung
        m = get_material(hu=-120)
        assert m.label == "Adipose_1"


# ---------------------------------------------------------------------------
# get_material(density=...)
# ---------------------------------------------------------------------------
class TestGetMaterialByDensity:
    # HU midpoints, one per bin, excluding:
    #   - the (14, 23] plateau collapse (HU -> rho=1.03 -> 13.44)
    #   - HU in [101, 174] whose forward density falls in [1.0769, 1.1199]
    #     -- ambiguous across the HU=100 discontinuity (inverse picks soft)
    @pytest.mark.parametrize("hu_mid",
                             [-975, -535, -101, -67, -37, -7, 13, 49, 100, 250,
                              350, 450, 550, 650, 750, 850, 950, 1050, 1150, 1250,
                              1350, 1450, 1550])
    def test_density_path_matches_hu_path(self, hu_mid):
        rho = hu_to_density(hu_mid)
        by_hu = get_material(hu=hu_mid)
        by_rho = get_material(density=rho)
        assert by_rho.label == by_hu.label
        assert by_rho.composition == pytest.approx(by_hu.composition, rel=1e-12)

    def test_no_arguments_raises(self):
        with pytest.raises(ValueError):
            get_material()

    def test_hu_takes_precedence_over_density(self):
        # provide density inconsistent with hu; hu must win
        m = get_material(hu=100, density=0.5)
        assert m.label == "Connective"
        assert m.density == pytest.approx(hu_to_density(100))

    def test_density_only_path_uses_inverse_calibration(self):
        # rho=1.0 -> HU ~ -20 -> Soft_1 bin (inverse calibration)
        m = get_material(density=1.0)
        assert m.label == get_material(hu=density_to_hu(1.0)).label


# ---------------------------------------------------------------------------
# Table 6 data integrity (regressions for the typos fixed in this branch)
# ---------------------------------------------------------------------------
class TestTable6Integrity:
    @pytest.mark.parametrize("idx", range(24))
    def test_row_sums_to_100(self, idx):
        entry = _schneider_table6[idx]
        total = sum(entry[el] for el in _ALL_ELEMENTS)
        assert total == pytest.approx(100.0, abs=0.05), \
            f"{entry['label']} sums to {total}"

    def test_lung_has_k_not_ca(self):
        # regression: Lung row had Ca/K swapped in an earlier version
        lung = next(e for e in _schneider_table6 if e["label"] == "Lung")
        assert lung["K"] == 0.2
        assert lung["Ca"] == 0.0

    def test_skeletal_400_500_has_chlorine(self):
        # regression: Cl was 0.0 in an earlier version (paper says 0.1)
        row = next(e for e in _schneider_table6 if e["label"] == "Skeletal_400_500")
        assert row["Cl"] == 0.1

    def test_air_has_no_hydrogen(self):
        air = next(e for e in _schneider_table6 if e["label"] == "Air")
        assert air["H"] == 0.0
        assert air["C"] == 0.0
        assert air["N"] == 75.5