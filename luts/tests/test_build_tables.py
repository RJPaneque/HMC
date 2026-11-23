"""Unit tests for ``build_tables.py`` (compaction) and ``lut_contract.py``.

Covers:
  * synthetic raw text -> monotone theta / u|theta ICDFs
  * DEAD rows carry zeros and valid == 0; live rows never hold zero padding
  * FULL / POOLED / SPARSE classification as a function of N_FULL and K_MAX
  * consolidated round trip (write -> read equality) and select() as combine's inverse
  * combine() rejects energy-grid and quantile-dimension mismatches
  * numeric material ordering (mat2 before mat10) in raw-file discovery
  * end-to-end build_engine on a tiny synthetic tree, census contents
"""
import struct

import numpy as np
import pytest

import json

import build_tables
import lut_contract
from lut_contract import DEAD, FULL, POOLED, SPARSE


# ---------------------------------------------------------------------------
# synthetic raw generator
# ---------------------------------------------------------------------------
STEP = 0.05
RHO = 1.05
CONFIG = (f"# config: engine=test step_cm= {STEP:.8E} emin_eV=1000 nenrg=6 nhist=300 seed=4000"
          f" rho_gcm3= {RHO:.8E}\n")
CONFIG_LEGACY = f"# config: engine=test step_cm= {STEP:.8E} emin_eV=1000 nenrg=6 nhist=300 seed=4000\n"
HEADER = "# E0_keV theta_rad DE_total_eV dE_kin_eV E_final_eV annihil_flag matID z_cm r_cm\n"
HEADER_LEGACY = "# E0_keV theta_rad DE_total_eV dE_kin_eV E_final_eV annihil_flag matID\n"


def write_raw(path, e_keV, n_hist, p_ann, mat_id, seed=0, theta_scale=0.3, config=CONFIG,
              legacy=False):
    """Write a raw simulation file. p_ann is per energy; survivors get
    theta ~ |N(0, theta_scale)| clipped to [0, pi] and u ~ Beta(2, 5).
    Stopped histories end at z ~ U(0, 0.6 STEP), r ~ U(0, 0.3 STEP); survivors
    at z = 0.95 STEP cos(theta), r = 0.5 STEP sin(theta). legacy=True writes
    the 7-column schema of raw files older than the stopping tables."""
    rng = np.random.default_rng(seed)
    lines = ([config] if config else []) + [HEADER_LEGACY if legacy else HEADER]
    for e, pa in zip(e_keV, p_ann):
        e_eV = e * 1000.0
        ann = rng.random(n_hist) < pa
        th = np.clip(np.abs(rng.normal(0, theta_scale, n_hist)), 0, np.pi)
        u = rng.beta(2, 5, n_hist)
        zs = rng.uniform(0, 0.6 * STEP, n_hist)
        rs = rng.uniform(0, 0.3 * STEP, n_hist)
        for a, t, uu, zz, rr in zip(ann, th, u, zs, rs):
            if a:
                line = f"{e:10.4f} {0.0:11.6f} {e_eV + 1.022e6:.6e} {e_eV:.6e} 0.000000e+00 1 {mat_id}"
                pos = f" {zz:11.7f} {rr:11.7f}"
            else:
                de = uu * e_eV
                line = f"{e:10.4f} {t:11.6f} {de:.6e} {de:.6e} {e_eV - de:.6e} 0 {mat_id}"
                pos = f" {0.95 * STEP * np.cos(t):11.7f} {0.5 * STEP * np.sin(t):11.7f}"
            lines.append(line + ("" if legacy else pos) + "\n")
    path.write_text("".join(lines))
    return path


E_GRID_KEV = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]


@pytest.fixture
def synthetic_tree(tmp_path):
    """raw/mat1.dat, mat2.dat, mat10.dat with 300 histories per energy.

    mat1: row 0 DEAD (p=1), row 1 sparse-ish (p=0.95 -> ~15 survivors), rest live.
    mat2 / mat10: all live, differing distributions so order is detectable.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    write_raw(raw / "mat1.dat", E_GRID_KEV, 300, [1.0, 0.95, 0.2, 0.1, 0.05, 0.0], 1, seed=1)
    write_raw(raw / "mat2.dat", E_GRID_KEV, 300, [0.3, 0.2, 0.1, 0.05, 0.0, 0.0], 2, seed=2, theta_scale=0.1)
    write_raw(raw / "mat10.dat", E_GRID_KEV, 300, [0.3, 0.2, 0.1, 0.05, 0.0, 0.0], 10, seed=3, theta_scale=1.0)
    return tmp_path


def _load_grouped(path):
    e0, th, de, ann = lut_contract.load_raw(path)
    return build_tables.group_by_energy(e0, th, de, ann)


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------
class TestGroupByEnergy:
    def test_counts_and_p_ann(self, synthetic_tree):
        e_grid, n_hist, n_surv, p_ann, st, su = _load_grouped(synthetic_tree / "raw" / "mat1.dat")
        assert np.allclose(e_grid, np.array(E_GRID_KEV) * 1000.0)
        assert np.all(n_hist == 300)
        assert n_surv[0] == 0 and p_ann[0] == 1.0
        assert n_surv[5] == 300 and p_ann[5] == 0.0
        assert np.allclose(p_ann, (n_hist - n_surv) / n_hist)
        for i in range(len(e_grid)):
            assert st[i].size == su[i].size == n_surv[i]

    def test_u_is_fraction_of_e0(self, synthetic_tree):
        _, _, _, _, st, su = _load_grouped(synthetic_tree / "raw" / "mat2.dat")
        for u in su:
            assert u.size and np.all(u >= 0) and np.all(u < 1)

    def test_group_stops(self, synthetic_tree):
        path = synthetic_tree / "raw" / "mat1.dat"
        assert lut_contract.raw_has_stop(path)
        raw = lut_contract.load_raw_full(path)
        assert raw["z"] is not None and raw["z"].dtype == np.float32
        e_grid, n_hist, n_surv, *_ = build_tables.group_by_energy(raw["E0_eV"], raw["theta"],
                                                                   raw["dE_kin"], raw["ann"])
        sz, sr = build_tables.group_stops(raw["E0_eV"], e_grid, raw["z"], raw["r"], raw["ann"])
        assert [z.size for z in sz] == (n_hist - n_surv).tolist()
        assert sz[0].size == 300 and sz[5].size == 0
        assert all(np.all((z >= 0) & (z <= 0.6 * STEP)) for z in sz if z.size)
        assert all(np.all((r >= 0) & (r <= 0.3 * STEP)) for r in sr if r.size)

    def test_legacy_raw_has_no_positions(self, tmp_path):
        path = write_raw(tmp_path / "mat1.dat", E_GRID_KEV, 20, [0.5] * 6, 1, legacy=True)
        assert not lut_contract.raw_has_stop(path)
        raw = lut_contract.load_raw_full(path)
        assert raw["z"] is None and raw["E0_eV"].size == 120
        e0, th, de, ann = lut_contract.load_raw(path)          # the 4-column reader still works
        assert np.array_equal(e0, raw["E0_eV"])


# ---------------------------------------------------------------------------
# row construction: monotone ICDFs
# ---------------------------------------------------------------------------
class TestBuildRow:
    @pytest.mark.parametrize("n_theta,n_u", [(8, 5), (32, 16), (128, 64)])
    def test_monotone_icdfs(self, synthetic_tree, n_theta, n_u):
        _, _, _, _, st, su = _load_grouped(synthetic_tree / "raw" / "mat2.dat")
        for t, u in zip(st, su):
            th, uu = lut_contract.build_row(t, u, n_theta, n_u)
            assert th.shape == (n_theta,) and uu.shape == (n_theta - 1, n_u)
            assert th.dtype == np.float32 and uu.dtype == np.float32
            assert np.all(np.diff(th) >= 0)
            assert np.all(np.diff(uu, axis=1) >= 0)
            assert 0.0 <= th[0] <= th[1] and th[-2] <= th[-1] <= np.pi
            assert np.all(uu >= 0) and np.all(uu < 1)

    def test_edge_intervals_keep_their_mean(self):
        # heavy upper tail: the sample maximum is far from the top interval's events
        rng = np.random.default_rng(1)
        theta = np.clip(rng.exponential(0.1, 100000), 0, np.pi)
        u = np.clip(rng.exponential(0.02, theta.size), 0, 0.999)
        n_theta, n_u = 128, 64
        th, uu = lut_contract.build_row(theta, u, n_theta, n_u)
        th = th.astype(float)
        top = theta[theta >= th[-2]]
        assert 0.5 * (th[-2] + th[-1]) == pytest.approx(top.mean(), rel=1e-5)
        assert th[-1] < theta.max()                       # extreme outlier no longer sets the knot
        # LUT mean (mean of interval midpoints) now matches the sample mean
        assert np.mean(0.5 * (th[:-1] + th[1:])) == pytest.approx(theta.mean(), rel=2e-3)
        raw = lut_contract.quantiles(theta, n_theta)
        assert np.mean(0.5 * (raw[:-1] + raw[1:])) / theta.mean() - 1 > 0.01   # old behaviour was biased
        # same for one conditional u ICDF
        b = np.clip(np.searchsorted(raw, theta, side="right") - 1, 0, n_theta - 2)
        sel = u[b == 60]
        uq = uu[60].astype(float)
        assert np.mean(0.5 * (uq[:-1] + uq[1:])) == pytest.approx(sel.mean(), rel=5e-3)

    def test_mean_preserving_ends_clamps(self):
        q = np.array([0.0, 0.1, 0.2, 0.3])
        x = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 3.1, 3.1])
        out = lut_contract.mean_preserving_ends(q, x, 0.0, np.pi)
        assert out[-1] == pytest.approx(np.pi)            # 2*mean - q[-2] > pi -> clamped
        assert out[0] == pytest.approx(0.0, abs=1e-12) and np.array_equal(out[1:-1], q[1:-1])

    def test_conditional_bins_are_conditional(self):
        # u strongly increasing with theta -> low theta bins must have lower u quantiles
        rng = np.random.default_rng(0)
        theta = rng.uniform(0, np.pi, 20000)
        u = np.clip(theta / np.pi * 0.8 + rng.normal(0, 0.01, theta.size), 0, 0.999)
        th, uu = lut_contract.build_row(theta, u, 16, 8)
        medians = uu[:, 4]
        assert np.all(np.diff(medians) > 0)

    def test_degenerate_bin_detection(self):
        theta = np.concatenate([np.zeros(500), np.linspace(0.1, 1.0, 50)])
        u = np.linspace(0.01, 0.99, theta.size)
        assert build_tables.count_degenerate_theta_bins(theta, 32) > 0
        th, uu = lut_contract.build_row(theta, u, 32, 8)          # fallback must still yield live data
        assert np.all(np.diff(th) >= 0) and np.all(np.diff(uu, axis=1) >= 0)
        assert np.all(np.any(uu != 0, axis=1))
        assert build_tables.count_degenerate_theta_bins(np.linspace(0, 1, 5000), 32) == 0


# ---------------------------------------------------------------------------
# classification: DEAD / FULL / POOLED / SPARSE
# ---------------------------------------------------------------------------
def _rows(sizes, seed=0):
    rng = np.random.default_rng(seed)
    st = [rng.uniform(0, 1, n) for n in sizes]
    su = [rng.uniform(0, 0.5, n) for n in sizes]
    return st, su


class TestClassifyAndPool:
    def test_codes_with_k_max(self):
        sizes = [0, 50, 5000, 5000, 100, 0, 0, 0, 20, 5000]
        st, su = _rows(sizes)
        valid, pt, pu = lut_contract.classify_and_pool(st, su, n_full=2000, k_max=2)
        assert valid.tolist() == [DEAD, POOLED, FULL, FULL, POOLED, DEAD, DEAD, DEAD, POOLED, FULL]
        # row 1: own 50 + k=1 neighbours (row 0 empty, row 2 = 5000) -> POOLED
        assert pt[1].size == pu[1].size == 50 + 5000
        # row 4: own 100 + k=1 (row 3 = 5000, row 5 empty) -> POOLED
        assert pt[4].size == 100 + 5000
        # row 8: own 20 + k=1 (row 7 empty, row 9 = 5000) -> POOLED
        assert pt[8].size == 20 + 5000
        assert pt[0] is None and pu[0] is None
        for i in (5, 6, 7):
            assert pt[i] is None

    def test_sparse_when_pooling_insufficient(self):
        sizes = [10, 20, 30, 40, 50]
        st, su = _rows(sizes)
        valid, pt, pu = lut_contract.classify_and_pool(st, su, n_full=2000, k_max=2)
        assert np.all(valid == SPARSE)
        assert pt[2].size == 10 + 20 + 30 + 40 + 50      # k=2 reaches everything
        assert pt[0].size == 10 + 20 + 30               # only right-hand neighbours exist

    def test_k_max_limits_pool(self):
        sizes = [100, 0, 0, 3000]
        st, su = _rows(sizes)
        v1, pt1, _ = lut_contract.classify_and_pool(st, su, n_full=2000, k_max=2)
        assert v1[0] == SPARSE and pt1[0].size == 100   # row 3 is k=3 away
        v2, pt2, _ = lut_contract.classify_and_pool(st, su, n_full=2000, k_max=3)
        assert v2[0] == POOLED and pt2[0].size == 3100

    def test_pooling_stops_at_first_sufficient_k(self):
        sizes = [1500, 1000, 1500, 9000, 9000]
        st, su = _rows(sizes)
        valid, pt, _ = lut_contract.classify_and_pool(st, su, n_full=2000, k_max=2)
        assert valid[1] == POOLED and pt[1].size == 1000 + 1500 + 1500   # k=1 suffices
        assert valid[0] == POOLED and pt[0].size == 1500 + 1000           # k=1: only row 1

    def test_n_full_boundary(self):
        st, su = _rows([2000, 1999])
        valid, _, _ = lut_contract.classify_and_pool(st, su, n_full=2000, k_max=2)
        assert valid.tolist() == [FULL, POOLED]


class TestBuildMaterial:
    def test_dead_rows_zero_and_valid_zero(self, synthetic_tree):
        g = _load_grouped(synthetic_tree / "raw" / "mat1.dat")
        lut, info = build_tables.build_material(*g, step_cm=STEP, n_theta=16, n_u=8, n_full=100, k_max=1)
        assert lut.n_m == 1 and lut.n_e == 6
        assert lut.valid[0, 0] == DEAD
        assert np.all(lut.theta_icdf[0, 0] == 0) and np.all(lut.u_icdf[0, 0] == 0)
        assert lut.p_ann[0, 0] == 1.0 and lut.n_surv[0, 0] == 0
        assert info[0]["n_pooled"] == 0
        # row 1 (~15 survivors) pooled with rows 0 (dead) and 2 -> POOLED
        assert lut.valid[0, 1] == POOLED
        assert info[1]["n_pooled"] == lut.n_surv[0, 1] + lut.n_surv[0, 2]
        assert lut.n_surv[0, 1] < 100                       # n_surv stays this row's own count
        assert np.all(lut.valid[0, 2:] == FULL)
        live = lut.valid[0] != DEAD
        # live rows: no zero padding anywhere
        assert np.all(np.any(lut.theta_icdf[0, live] != 0, axis=-1))
        assert np.all(np.any(lut.u_icdf[0, live] != 0, axis=-1))
        assert np.all(np.diff(lut.theta_icdf[0, live], axis=-1) >= 0)
        assert np.all(np.diff(lut.u_icdf[0, live], axis=-1) >= 0)
        assert lut.check()

    def test_sparse_classification_from_raw(self, synthetic_tree):
        g = _load_grouped(synthetic_tree / "raw" / "mat1.dat")
        lut, info = build_tables.build_material(*g, step_cm=STEP, n_theta=8, n_u=4, n_full=100000, k_max=2)
        assert lut.valid[0, 0] == DEAD
        assert np.all(lut.valid[0, 1:] == SPARSE)
        assert np.all(np.any(lut.theta_icdf[0, 1:] != 0, axis=-1))   # SPARSE rows still carry data
        assert not lut.has_stop and np.all(lut.svalid == DEAD)        # no end positions passed

    def test_stopping_tables(self, synthetic_tree):
        lut, info = build_tables.process_material(synthetic_tree / "raw" / "mat1.dat", 8, 4, 100, 1,
                                                  step_cm=STEP, verbose=False, n_z=8, n_r=4,
                                                  n_full_stop=100, rho_nom=RHO)
        assert lut.has_stop and lut.n_z == 8 and lut.n_r == 4 and lut.rho_nom[0] == np.float32(RHO)
        # p_ann = [1, .95, .2, .1, .05, 0] over 300 histories: stops 300, ~285, ~60, ~30, ~15, 0
        assert lut.svalid[0, 0] == FULL and lut.svalid[0, 1] == FULL
        assert lut.svalid[0, 2] == POOLED and lut.svalid[0, 5] == DEAD
        assert info[0]["n_stop"] == 300 and info[5]["n_stop"] == 0 and info[0]["stop_code"] == FULL
        assert info[2]["n_stop_pooled"] == info[1]["n_stop"] + info[2]["n_stop"] + info[3]["n_stop"]
        slive = lut.svalid[0] != DEAD
        assert np.all(np.diff(lut.z_icdf[0, slive], axis=-1) >= 0)
        assert np.all(np.diff(lut.r_icdf[0, slive], axis=-1) >= 0)
        # interior knots are sample quantiles, so they stay inside the sample range; the
        # edge knots are mean-preserving and may sit outside it (within the table range)
        assert np.all(lut.z_icdf[0, slive][:, 1:-1] >= 0) and np.all(lut.z_icdf[0, slive][:, 1:-1] <= 0.6 * STEP)
        assert np.all(lut.r_icdf[0, slive][:, :, 1:-1] <= 0.3 * STEP)
        assert np.all(np.abs(lut.z_icdf) <= STEP) and np.all((lut.r_icdf >= 0) & (lut.r_icdf <= STEP))
        assert np.all(np.any(lut.r_icdf[0, slive] != 0, axis=-1))
        assert np.all(lut.z_icdf[0, ~slive] == 0) and np.all(lut.r_icdf[0, ~slive] == 0)
        # the stopping sampler on the product: has where a bracketing row is live
        # (mat2 has no stopped history at 20 and 50 keV, so both rows there are DEAD)
        lut2, _ = build_tables.process_material(synthetic_tree / "raw" / "mat2.dat", 8, 4, 100, 1,
                                                step_cm=STEP, verbose=False, n_z=8, n_r=4, n_full_stop=100)
        assert lut2.svalid[0].tolist()[4:] == [DEAD, DEAD]
        both = lut_contract.combine([lut, lut2])
        has, z, r = lut_contract.sample_stop_many(both, [0, 0, 1], [1000.0, 7000.0, 50000.0],
                                                  [0.5, 0.1, 0.9], [0.5, 0.9, 0.1])
        assert has.tolist() == [True, True, False] and np.all(z[:2] >= -0.01 * STEP) and r[2] == 0
        got = lut_contract.sample_stop(both, 0, 7000.0, 0.1, 0.9)
        assert got[0] and got[1] == pytest.approx(z[1]) and got[2] == pytest.approx(r[1])
        assert lut_contract.sample_stop(both, 1, 50000.0, 0.5, 0.5) == (False, 0.0, 0.0)

    def test_stop_rows_classified_with_their_own_threshold(self, synthetic_tree):
        lut, _ = build_tables.process_material(synthetic_tree / "raw" / "mat1.dat", 8, 4, 100, 1,
                                               step_cm=STEP, verbose=False, n_z=4, n_r=3,
                                               n_full_stop=100000)
        assert np.all(lut.svalid[0, :5] == SPARSE) and lut.svalid[0, 5] == DEAD
        lut2, _ = build_tables.process_material(synthetic_tree / "raw" / "mat1.dat", 8, 4, 100, 1,
                                                step_cm=STEP, verbose=False, stop=False)
        assert not lut2.has_stop


# ---------------------------------------------------------------------------
# round trips
# ---------------------------------------------------------------------------
class TestRoundTrip:
    def test_select_inverts_combine(self, synthetic_tree):
        luts = []
        for name in ("mat1.dat", "mat2.dat", "mat10.dat"):
            g = _load_grouped(synthetic_tree / "raw" / name)
            luts.append(build_tables.build_material(*g, step_cm=STEP, n_theta=16, n_u=8, n_full=50, k_max=1)[0])
        lut = lut_contract.combine(luts)
        for k, one in enumerate(luts):
            build_tables.assert_lut_equal(lut_contract.select(lut, k), one)

    def test_consolidated_round_trip(self, synthetic_tree, tmp_path):
        luts = []
        for name in ("mat1.dat", "mat2.dat", "mat10.dat"):
            g = _load_grouped(synthetic_tree / "raw" / name)
            luts.append(build_tables.build_material(*g, step_cm=STEP, n_theta=16, n_u=8, n_full=50, k_max=1)[0])
        lut = lut_contract.combine(luts)
        assert lut.n_m == 3 and not lut.has_stop
        a, s = tmp_path / "ann_LUT.raw", tmp_path / "scatt-eloss_LUT.raw"
        lut_contract.write_ann_lut(a, lut)
        lut_contract.write_scatt_lut(s, lut)
        back = lut_contract.read_luts(a, s)
        build_tables.assert_lut_equal(back, lut)
        assert a.stat().st_size == 4 + 7 * 4 + 6 * 4 + 3 * 4 + 5 * 3 * 6 * 4
        assert s.stat().st_size == 4 + 7 * 4 + 6 * 4 + 3 * 6 * 16 * 4 + 3 * 6 * 15 * 8 * 4 + 3 * 6 * 4

    def test_round_trip_with_stopping_tables(self, synthetic_tree, tmp_path):
        luts = [build_tables.process_material(synthetic_tree / "raw" / n, 16, 8, 50, 1, step_cm=STEP,
                                              verbose=False, n_z=8, n_r=4, n_full_stop=50, rho_nom=k + 1.0)[0]
                for k, n in enumerate(("mat1.dat", "mat2.dat", "mat10.dat"))]
        lut = lut_contract.combine(luts)
        assert lut.has_stop and lut.rho_nom.tolist() == [1.0, 2.0, 3.0]
        a, s = tmp_path / "ann_LUT.raw", tmp_path / "scatt-eloss_LUT.raw"
        lut_contract.write_ann_lut(a, lut); lut_contract.write_scatt_lut(s, lut)
        back = lut_contract.read_luts(a, s)
        build_tables.assert_lut_equal(back, lut)
        assert a.stat().st_size == 32 + 6 * 4 + 3 * 4 + 5 * 3 * 6 * 4 + 3 * 6 * 8 * 4 + 3 * 6 * 7 * 4 * 4
        assert a.read_bytes()[20:32] == struct.pack("<fii", STEP, 8, 4)
        for k, one in enumerate(luts):
            build_tables.assert_lut_equal(lut_contract.select(back, k), one)
        with pytest.raises(ValueError, match="stopping-table dimension"):
            lut_contract.combine([luts[0], build_tables.process_material(
                synthetic_tree / "raw" / "mat2.dat", 16, 8, 50, 1, step_cm=STEP, verbose=False, n_z=4, n_r=4)[0]])

    def test_version_3_round_trip(self, synthetic_tree, tmp_path):
        """A v3 pair (old layout) is still readable; it carries no densities or stopping tables."""
        lut, _ = build_tables.process_material(synthetic_tree / "raw" / "mat1.dat", 8, 4, 50, 1,
                                               step_cm=STEP, verbose=False, rho_nom=RHO)
        assert lut.has_stop
        a, s = tmp_path / "ann_LUT.raw", tmp_path / "scatt-eloss_LUT.raw"
        lut_contract.write_ann_lut(a, lut, version=3); lut_contract.write_scatt_lut(s, lut, version=3)
        assert a.read_bytes()[4:8] == (3).to_bytes(4, "little")
        assert a.stat().st_size == 32 + 6 * 4 + 4 * 6 * 4
        back = lut_contract.read_luts(a, s)
        assert not back.has_stop and np.all(back.rho_nom == 0) and np.all(back.svalid == DEAD)
        for attr in ("e_grid", "p_ann", "n_hist", "n_surv", "valid", "theta_icdf", "u_icdf"):
            assert np.array_equal(getattr(back, attr), getattr(lut, attr))
        lut_contract.write_ann_lut(a, lut)                     # mixed versions are rejected
        with pytest.raises(ValueError, match="version"):
            lut_contract.read_luts(a, s)

    def test_read_luts_rejects_validity_disagreement(self, synthetic_tree, tmp_path):
        g = _load_grouped(synthetic_tree / "raw" / "mat1.dat")
        lut, _ = build_tables.build_material(*g, step_cm=STEP, n_theta=8, n_u=4, n_full=50, k_max=1)
        a, s = tmp_path / "ann_LUT.raw", tmp_path / "scatt-eloss_LUT.raw"
        lut_contract.write_ann_lut(a, lut)
        lut2 = lut_contract.LutV2(lut.e_grid, lut.p_ann, lut.n_hist, lut.n_surv,
                            lut.valid.copy(), lut.theta_icdf, lut.u_icdf, lut.step_cm)
        lut2.valid[0, 2] = POOLED
        lut_contract.write_scatt_lut(s, lut2)
        with pytest.raises(ValueError, match="disagree"):
            lut_contract.read_luts(a, s)


# ---------------------------------------------------------------------------
# combine: mismatch rejection
# ---------------------------------------------------------------------------
class TestCombine:
    def _lut(self, e_grid_eV, n_theta=8, n_u=4, seed=0):
        rng = np.random.default_rng(seed)
        n_e = len(e_grid_eV)
        st = [rng.uniform(0, 1, 500) for _ in range(n_e)]
        su = [rng.uniform(0, 0.5, 500) for _ in range(n_e)]
        g = (np.array(e_grid_eV, float), np.full(n_e, 500, np.int32), np.full(n_e, 500, np.int32),
             np.zeros(n_e), st, su)
        return build_tables.build_material(*g, step_cm=STEP, n_theta=n_theta, n_u=n_u, n_full=100, k_max=1)[0]

    def test_rejects_grid_mismatch(self):
        a = self._lut([1e3, 2e3, 5e3])
        b = self._lut([1e3, 2e3, 5.1e3])
        with pytest.raises(ValueError, match="grid mismatch"):
            lut_contract.combine([a, b])

    def test_rejects_grid_length_mismatch(self):
        a = self._lut([1e3, 2e3, 5e3])
        b = self._lut([1e3, 2e3])
        with pytest.raises(ValueError):
            lut_contract.combine([a, b])

    def test_rejects_quantile_dim_mismatch(self):
        a = self._lut([1e3, 2e3, 5e3], n_theta=8)
        b = self._lut([1e3, 2e3, 5e3], n_theta=16)
        with pytest.raises(ValueError, match="dimension mismatch"):
            lut_contract.combine([a, b])

    def test_combine_concatenates_in_given_order(self):
        a = self._lut([1e3, 2e3], seed=1)
        b = self._lut([1e3, 2e3], seed=2)
        c = lut_contract.combine([a, b])
        assert np.array_equal(c.theta_icdf[0], a.theta_icdf[0])
        assert np.array_equal(c.theta_icdf[1], b.theta_icdf[0])
        assert np.array_equal(c.u_icdf[1], b.u_icdf[0])


# ---------------------------------------------------------------------------
# numeric material ordering
# ---------------------------------------------------------------------------
class TestMaterialOrdering:
    def test_material_id_parsing(self):
        assert build_tables.material_id("mat7.dat") == 7
        assert build_tables.material_id("/x/y/mat30.dat") == 30
        with pytest.raises(ValueError):
            build_tables.material_id("ann_LUT.raw")

    def test_numeric_sort_raw(self, tmp_path):
        names = ["mat10.dat", "mat2.dat", "mat1.dat", "mat30.dat", "mat3.dat"]
        for n in names:
            (tmp_path / n).write_text(HEADER)
        found = build_tables.find_raw_files(tmp_path)
        assert [p.name for p in found] == ["mat1.dat", "mat2.dat", "mat3.dat", "mat10.dat", "mat30.dat"]
        assert [p.name for p in sorted(tmp_path.glob("mat*.dat"))][1] == "mat10.dat"  # lexicographic would be wrong

    def test_duplicate_material_rejected(self, tmp_path):
        (tmp_path / "mat1.dat").write_text(HEADER)
        (tmp_path / "mat01.dat").write_text(HEADER)
        with pytest.raises(ValueError, match="duplicate"):
            build_tables.find_raw_files(tmp_path)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
class TestEndToEnd:
    def test_build_engine_synthetic(self, synthetic_tree, monkeypatch):
        out = synthetic_tree / "out"
        lut, rows = build_tables.build_engine(
            "penelope", synthetic_tree / "raw", out, n_theta=16, n_u=8, n_full=100, k_max=1, workers=2)
        assert not (out / "tables").exists()
        assert (out / "ann_LUT.raw").is_file() and (out / "scatt-eloss_LUT.raw").is_file()
        back = lut_contract.read_luts(out / "ann_LUT.raw", out / "scatt-eloss_LUT.raw")
        build_tables.assert_lut_equal(back, lut)
        assert back.n_m == 3 and back.n_e == 6 and back.n_theta == 16 and back.n_u == 8
        # material slot order is numeric: slot 1 = mat2 (narrow theta), slot 2 = mat10 (wide theta)
        m2 = build_tables.process_material(synthetic_tree / "raw" / "mat2.dat", 16, 8, 100, 1, verbose=False)[0]
        m10 = build_tables.process_material(synthetic_tree / "raw" / "mat10.dat", 16, 8, 100, 1, verbose=False)[0]
        assert np.array_equal(back.theta_icdf[1], m2.theta_icdf[0])
        assert np.array_equal(back.theta_icdf[2], m10.theta_icdf[0])
        assert back.theta_icdf[1, 5, 8] < back.theta_icdf[2, 5, 8]
        # census
        assert len(rows) == 3 * 6
        assert [r["material"] for r in rows[:6]] == [1] * 6
        assert rows[0]["valid_code"] == DEAD and rows[0]["n_surv"] == 0 and rows[0]["p_ann"] == 1.0
        assert rows[1]["valid_code"] == POOLED
        assert set(build_tables.summarise_census(rows)) == {"DEAD", "FULL", "POOLED", "SPARSE"}
        census = (out / "gap_census.csv").read_text().splitlines()
        assert census[0].split(",")[:8] == ["engine", "material", "energy_index", "E_keV",
                                            "n_hist", "n_surv", "p_ann", "valid_code"]
        assert census[0].split(",")[-4:] == ["n_stop", "stop_code", "stop_name", "n_stop_pooled"]
        assert len(census) == 1 + 18
        # stopping tables and densities came through the whole pipeline
        assert back.has_stop and back.n_z == 32 and back.n_r == 16
        assert back.rho_nom.tolist() == [np.float32(RHO)] * 3
        # 300 stops < N_FULL_STOP = 500: pooled with the next row's ~285
        assert rows[0]["n_stop"] == 300 and rows[0]["stop_code"] == POOLED and rows[5]["stop_code"] == DEAD
        assert rows[0]["n_stop_pooled"] == 300 + rows[1]["n_stop"]
        info = json.loads((out / "run.json").read_text())
        assert info["n_z"] == 32 and info["n_r"] == 16 and info["n_full_stop"] == 500
        assert info["rho_nom"] == [RHO] * 3
        # sampling contract runs on the product and never returns u >= 1 or theta outside [0, pi]
        rng = np.random.default_rng(0)
        n = 5000
        ann, th, u = lut_contract.sample_many(back, np.full(n, 1), np.full(n, 7000.0),
                                        rng.random(n), rng.random(n), rng.random(n))
        assert np.all((th >= 0) & (th <= np.pi)) and np.all((u >= 0) & (u < 1))

    def test_get_engine_dirs(self, tmp_path):
        raw, out = build_tables.get_engine_dirs("penelope")
        assert raw.parts[-3:] == ("output", "penelope", "raw") and out.parts[-2:] == ("output", "penelope")
        raw, out = build_tables.get_engine_dirs("geant4", tmp_path / "r", tmp_path / "o")
        assert raw == tmp_path / "r" and out == tmp_path / "o"
        with pytest.raises(ValueError, match="Unknown engine"):
            build_tables.get_engine_dirs("mcnp")


# ---------------------------------------------------------------------------
# step length: raw config line -> LUT header (format version 3)
# ---------------------------------------------------------------------------
class TestStep:
    def test_raw_config_parsed(self, synthetic_tree):
        cfg = lut_contract.read_raw_config(synthetic_tree / "raw" / "mat1.dat")
        assert cfg["step_cm"] == STEP and cfg["engine"] == "test" and cfg["nenrg"] == 6
        e0, *_ = lut_contract.load_raw(synthetic_tree / "raw" / "mat1.dat")
        assert e0.size == 6 * 300                               # config line is a comment

    def test_header_round_trip_and_version(self, synthetic_tree, tmp_path):
        g = _load_grouped(synthetic_tree / "raw" / "mat2.dat")
        lut, _ = build_tables.build_material(*g, step_cm=0.025, n_theta=8, n_u=4, n_full=50, k_max=1)
        a, s = tmp_path / "ann_LUT.raw", tmp_path / "scatt-eloss_LUT.raw"
        lut_contract.write_ann_lut(a, lut); lut_contract.write_scatt_lut(s, lut)
        assert lut_contract.read_luts(a, s).step_cm == np.float32(0.025)
        assert a.read_bytes()[4:8] == (4).to_bytes(4, "little")          # VERSION 4

    def test_step_required_and_consistent(self, synthetic_tree, tmp_path):
        g = _load_grouped(synthetic_tree / "raw" / "mat2.dat")
        lut, _ = build_tables.build_material(*g, n_theta=8, n_u=4, n_full=50, k_max=1)   # no step
        with pytest.raises(ValueError, match="step_cm must be > 0"):
            lut_contract.write_ann_lut(tmp_path / "a.raw", lut)
        a = build_tables.build_material(*g, step_cm=0.05, n_theta=8, n_u=4, n_full=50, k_max=1)[0]
        b = build_tables.build_material(*g, step_cm=0.1, n_theta=8, n_u=4, n_full=50, k_max=1)[0]
        with pytest.raises(ValueError, match="step_cm mismatch"):
            lut_contract.combine([a, b])

    def test_build_takes_step_from_raw(self, synthetic_tree):
        out = synthetic_tree / "o"
        lut, _ = build_tables.build_engine("penelope", synthetic_tree / "raw", out, n_theta=8, n_u=4,
                                           n_full=100, k_max=1, workers=1)
        assert lut.step_cm == STEP
        assert lut_contract.read_luts(out / "ann_LUT.raw", out / "scatt-eloss_LUT.raw").step_cm == np.float32(STEP)
        with pytest.raises(SystemExit, match="contradicts"):
            build_tables.resolve_step(build_tables.find_raw_files(synthetic_tree / "raw"), step_cm=0.1)

    def test_old_raw_falls_back(self, tmp_path):
        raw = tmp_path / "raw"; raw.mkdir()
        write_raw(raw / "mat1.dat", E_GRID_KEV, 50, [0.1] * 6, 1, config=None)
        files = build_tables.find_raw_files(raw)
        assert build_tables.resolve_step(files, step_cm=0.2) == 0.2
        sim = tmp_path / "sim.in"; sim.write_text("0.075   ! STEP_CM\n1000.0\n")
        assert build_tables.resolve_step(files, sim_in=sim) == 0.075


# ---------------------------------------------------------------------------
# nominal densities: raw config line, .mat files, or unknown
# ---------------------------------------------------------------------------
class TestDensities:
    def _mat(self, d, n, rho):
        d.mkdir(parents=True, exist_ok=True)
        (d / f"mat{n}.mat").write_text(" PENELOPE (v. 2018)  Material data file\n Material: x\n"
                                       f" Mass density = {rho:.8E} g/cm**3\n Number of elements = 1\n")

    def test_from_config_line(self, synthetic_tree):
        files = build_tables.find_raw_files(synthetic_tree / "raw")
        assert build_tables.resolve_densities(files, synthetic_tree / "nomats") == [RHO] * 3
        assert build_tables.resolve_densities(files) == [RHO] * 3        # default mats dir: config wins

    def test_from_mat_files_when_config_lacks_it(self, tmp_path):
        raw = tmp_path / "raw"; raw.mkdir()
        write_raw(raw / "mat1.dat", E_GRID_KEV, 20, [0.5] * 6, 1, config=CONFIG_LEGACY, legacy=True)
        write_raw(raw / "mat3.dat", E_GRID_KEV, 20, [0.5] * 6, 3, config=CONFIG_LEGACY, legacy=True)
        self._mat(tmp_path / "mats", 1, 0.3)
        assert build_tables.read_mat_density(tmp_path / "mats" / "mat1.mat") == 0.3
        files = build_tables.find_raw_files(raw)
        assert build_tables.resolve_densities(files, tmp_path / "mats", verbose=False) == [0.3, 0.0]
        out = tmp_path / "out"
        lut, _ = build_tables.build_engine("penelope", raw, out, n_theta=8, n_u=4, n_full=10, k_max=1,
                                           workers=1, mat_dir=tmp_path / "mats")
        assert lut.rho_nom.tolist() == [np.float32(0.3), 0.0] and not lut.has_stop
        back = lut_contract.read_luts(out / "ann_LUT.raw", out / "scatt-eloss_LUT.raw")
        assert back.rho_nom.tolist() == [np.float32(0.3), 0.0] and not back.has_stop

    def test_config_and_mat_file_must_agree(self, synthetic_tree):
        self._mat(synthetic_tree / "mats", 1, RHO)
        self._mat(synthetic_tree / "mats", 2, 2 * RHO)
        files = build_tables.find_raw_files(synthetic_tree / "raw")
        with pytest.raises(SystemExit, match="differs"):
            build_tables.resolve_densities(files, synthetic_tree / "mats")
