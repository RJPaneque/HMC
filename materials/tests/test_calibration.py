"""Unit tests for the Schneider HU <-> mass-density calibration curve.

The calibration (Schneider et al. 2000, eq. and Fig. 3 mapping CT number to
mass density) is piecewise-linear in five segments:

    HU <          -1000 :  constant  (0.00121 g/cm3, air)
    -1000 <= HU <    -98 :  linear, air -> lung endpoint 0.93
    -98   <= HU <=    14 :  linear, rho = 1.018 + 0.893e-3 * HU
    14    <  HU <=    23 :  plateau, rho = 1.03
    23    <  HU <=   100 :  linear, rho = 1.003 + 1.169e-3 * HU
    HU >           100 :  linear, rho = 1.017 + 0.592e-3 * HU

``density_to_hu`` is the inverse, except in the (14, 23] plateau where it
collapses onto the midpoint 18.5.
"""

import pytest

from schneider_materials import hu_to_density, density_to_hu


# ---------------------------------------------------------------------------
# hu_to_density
# ---------------------------------------------------------------------------
class TestHuToDensity:
    def test_below_range_returns_air(self):
        assert hu_to_density(-2000) == pytest.approx(0.00121)

    def test_at_minus_1000(self):
        assert hu_to_density(-1000) == pytest.approx(0.00121)

    def test_lung_segment_endpoints(self):
        assert hu_to_density(-1000) == pytest.approx(0.00121)
        # the lung branch is hu < -98 (strict), so just below -98 it
        # approaches 0.93; AT -98 the adipose branch already takes over
        assert hu_to_density(-98.001) == pytest.approx(0.93, abs=1e-4)
        assert hu_to_density(-98) == pytest.approx(1.018 + 0.893e-3 * -98)

    def test_lung_segment_internal(self):
        # midpoint of [-1000, -98] -> midpoint of densities
        hu_mid = (-1000 + -98) / 2.0
        rho_mid = (0.00121 + 0.93) / 2.0
        assert hu_to_density(hu_mid) == pytest.approx(rho_mid, abs=1e-9)

    def test_adipose_segment_endpoints(self):
        assert hu_to_density(-98) == pytest.approx(1.018 + 0.893e-3 * -98)
        assert hu_to_density(14) == pytest.approx(1.018 + 0.893e-3 * 14)

    def test_plateau_segment(self):
        assert hu_to_density(18) == 1.03
        assert hu_to_density(23) == 1.03

    def test_soft_segment_endpoints(self):
        # note: at HU=23 the function returns 1.03 (plateau, last match);
        # the soft segment strictly starts after 23 -> test 100 (its right end)
        assert hu_to_density(100) == pytest.approx(1.003 + 1.169e-3 * 100)
        # interior point (24 just above plateau)
        assert hu_to_density(50) == pytest.approx(1.003 + 1.169e-3 * 50)

    def test_skeletal_segment_endpoints(self):
        assert hu_to_density(101) == pytest.approx(1.017 + 0.592e-3 * 101)
        assert hu_to_density(1524) == pytest.approx(1.017 + 0.592e-3 * 1524)

    def test_above_range_extrapolates(self):
        assert hu_to_density(5000) == pytest.approx(1.017 + 0.592e-3 * 5000)

    def test_monotonic_within_each_segment(self):
        # the calibration is piecewise-monotonic; there is a known downward
        # discontinuity at HU=100 (see test_discontinuity_at_hu100).
        for seg_start, seg_end in [(-999, -99), (-98, 15), (24, 100), (101, 1599)]:
            rhos = [hu_to_density(h) for h in range(seg_start, seg_end)]
            for a, b in zip(rhos, rhos[1:]):
                assert b >= a - 1e-12

    def test_discontinuity_at_hu100(self):
        # documented: soft segment ends at rho(100)=1.1199; the skeletal
        # branch (hu > 100) starts lower at rho(101)=1.0769 -- a downward jump.
        assert hu_to_density(100) == pytest.approx(1.1199)
        assert hu_to_density(101) == pytest.approx(1.017 + 0.592e-3 * 101)
        assert hu_to_density(101) < hu_to_density(100)

    def test_positive_density_everywhere(self):
        for h in range(-1100, 2000, 50):
            assert hu_to_density(h) > 0


# ---------------------------------------------------------------------------
# density_to_hu
# ---------------------------------------------------------------------------
class TestDensityToHu:
    def test_zero_density_returns_minus_1000(self):
        assert density_to_hu(0.0) == -1000.0

    def test_below_air_returns_minus_1000(self):
        assert density_to_hu(0.0001) == -1000.0

    def test_at_air_density(self):
        assert density_to_hu(0.00121) == -1000.0

    def test_lung_endpoint(self):
        assert density_to_hu(0.93) == pytest.approx(-98.0)

    def test_adipose_endpoint_hu14(self):
        rho14 = 1.018 + 0.893e-3 * 14.0
        assert density_to_hu(rho14) == pytest.approx(14.0)

    def test_plateau_density_returns_midpoint(self):
        # rho = 1.03 is the plateau (HU in (14, 23]); the inverse maps it
        # to the representative midpoint 18.5.
        assert density_to_hu(1.03) == 18.5
        assert hu_to_density(density_to_hu(1.03)) == pytest.approx(1.03, rel=1e-9)

    def test_soft_endpoint_hu100(self):
        rho100 = 1.003 + 1.169e-3 * 100.0
        assert density_to_hu(rho100) == pytest.approx(100.0)

    def test_skeletal_high(self):
        rho = 1.017 + 0.592e-3 * 1524.0
        assert density_to_hu(rho) == pytest.approx(1524.0)

    def test_monotonic(self):
        rhos = [0.00121 * x for x in range(1, 2000, 7)]
        hus = [density_to_hu(r) for r in rhos]
        for a, b in zip(hus, hus[1:]):
            assert b >= a - 1e-6


# ---------------------------------------------------------------------------
# round trips
# ---------------------------------------------------------------------------
class TestRoundTrip:
    # representative HU values, one per segment, avoiding the (14, 23] plateau
    @pytest.mark.parametrize("hu", [-950, -500, -98, -50, 0, 14, 50, 100, 500, 1000])
    def test_hu_rho_hu(self, hu):
        rho = hu_to_density(hu)
        assert density_to_hu(rho) == pytest.approx(hu, abs=1e-6)

    # representative densities spanning the calibration range, avoiding the
    # (1.030502, 1.03] plateau window and rho=0.93 (HU=-98 boundary, where
    # the inverse selects the lung endpoint but the forward picks adipose).
    @pytest.mark.parametrize("rho",
                             [0.00121, 0.05, 0.1, 0.5, 1.0, 1.018, 1.03, 1.05, 1.1, 1.3, 1.6, 1.92])
    def test_rho_hu_rho(self, rho):
        hu = density_to_hu(rho)
        assert hu_to_density(hu) == pytest.approx(rho, rel=1e-6)

    @pytest.mark.parametrize("hu", [15, 18, 18.5, 20, 23])
    def test_plateau_hus_collapse_to_midpoint(self, hu):
        # every HU in the (14, 23] plateau maps to rho=1.03, which inverts
        # to the representative midpoint 18.5 (not back to the original HU).
        rho = hu_to_density(hu)
        assert rho == pytest.approx(1.03, rel=1e-9)
        assert density_to_hu(rho) == 18.5