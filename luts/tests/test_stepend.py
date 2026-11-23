"""Step-end tables (LUT version 5 / 6, luts/STEPEND_V5_SPEC.md): file format, row
construction, pooling, the direction frames, the sampling contract, the v6
tail knots (geom 0 only: the sphere layout was dropped on 2026-10-30 and is
refused), and the build pipeline."""
import json
import struct
import warnings

import numpy as np
import pytest

import build_tables
import lut_contract as lc
from lut_contract import DEAD, FULL, POOLED, SPARSE

STEP = 0.05


def synth_histories(n, seed=0, step=STEP):
    """Correlated survivors: z, r on a curling path, theta grows with r, dphi
    signed, u correlated with the path's curl."""
    rng = np.random.default_rng(seed)
    curl = rng.beta(2, 3, n)                                   # 0 straight .. 1 curled
    z = step * (1.0 - 1.3 * curl) * rng.uniform(0.85, 1.0, n)
    r = step * 0.6 * np.sqrt(curl) * rng.uniform(0.5, 1.0, n)
    theta = np.clip(2.5 * curl + rng.normal(0, 0.15, n), 0, np.pi)
    dphi = rng.normal(0.4 * curl, 0.6, n)
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
    u = np.clip(0.05 + 0.3 * curl + rng.normal(0, 0.03, n), 0, 0.99)
    return z, r, theta, dphi, u


def synth_conditional(n, frame, seed=0, step=STEP):
    """Histories whose (a1, a2, u) in `frame` are independent given the curl
    that sets (z, r) -- the step-end model's assumption -- mapped back to the
    raw (theta, dphi) through stepend_geometry."""
    rng = np.random.default_rng(seed)
    curl = rng.beta(2, 3, n)
    z = step * (1.0 - 1.3 * curl) * rng.uniform(0.97, 1.0, n)
    r = step * 0.6 * np.sqrt(curl) * rng.uniform(0.9, 1.0, n)
    a1 = np.clip(2.0 * curl + rng.normal(0, 0.15, n), 0, np.pi)
    a2 = np.clip(0.5 + 1.5 * curl + rng.normal(0, 0.5, n), 0, np.pi)
    u = np.clip(0.05 + 0.3 * curl + rng.normal(0, 0.03, n), 0, 0.99)
    s = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    _, nn = lc.stepend_geometry(z, r, a1, a2, s, frame)
    theta = np.arccos(np.clip(nn[:, 2], -1, 1))
    dphi = np.arctan2(nn[:, 1], nn[:, 0])
    return z, r, theta, dphi, u


def lut_from_rows(cols, frame, e_keV=(100.0, 200.0), n_z=16, n_r=8, n_a=8, geom=lc.GEOM_PATH,
                  knots=lc.KNOTS_UNIFORM):
    """Two-energy one-material StepEndLut with the same row at both energies."""
    row = lc.build_stepend_row(*cols, frame=frame, n_z=n_z, n_r=n_r, n_a=n_a, step_cm=STEP, geom=geom,
                               knots=knots)
    n_e = len(e_keV)
    tabs = [np.repeat(a[None, None], n_e, axis=1) for a in row]
    lut = lc.StepEndLut(np.array(e_keV, dtype=np.float32) * 1000, np.full((1, n_e), FULL, np.int32),
                        *tabs, frame=frame, step_cm=STEP, n_full_se=len(cols[0]), geom=geom, knots=knots)
    lut.check()
    return lut


def n_from_raw(theta, dphi):
    st = np.sin(theta)
    return np.stack([st * np.cos(dphi), st * np.sin(dphi), np.cos(theta)], axis=-1)


# ---------------------------------------------------------------------------
class TestFile:
    def test_header_and_roundtrip(self, tmp_path):
        lut = lut_from_rows(synth_histories(3000), frame=1, n_z=5, n_r=4, n_a=3)
        lut.evalid[0, 1] = POOLED
        p = tmp_path / "stepend_LUT.raw"
        lc.write_stepend_lut(p, lut)
        b = p.read_bytes()
        assert b[:4] == b"HMC2"
        assert struct.unpack("<8i", b[4:36]) == (6, 3, 2, 1, 5, 4, 3, 1)
        step, nfull, geom, knots, res = struct.unpack("<fiiii", b[36:56])
        assert step == np.float32(STEP) and nfull == 3000 and (geom, knots, res) == (0, 0, 0)
        n_e, n_m = 2, 1
        assert np.array_equal(np.frombuffer(b, "<f4", 3, 56 + 4 * n_e), [0, 0.5, 1])     # the levels
        assert len(b) == 56 + 4 * (n_e + 3 + n_m * n_e * (1 + 5 + 4 * 4 + 3 * 4 * 3 * 3))
        back = lc.read_stepend_lut(p)
        for a in ("e_grid", "evalid", "z_icdf_s", "r_icdf_s", "a1_icdf", "a2_icdf", "u_icdf_s"):
            assert np.array_equal(getattr(back, a), getattr(lut, a)), a
        assert back.frame == 1 and back.step_cm == np.float32(STEP) and back.n_full_se == 3000
        assert (back.version, back.geom, back.knots) == (6, 0, 0)
        # a v5 file (48-byte header, no levels) is still read
        lut.version = 5
        lc.write_stepend_lut(tmp_path / "v5.raw", lut)
        b5 = (tmp_path / "v5.raw").read_bytes()
        assert struct.unpack("<i", b5[4:8])[0] == 5 and len(b5) == len(b) - 8 - 4 * 3
        v5 = lc.read_stepend_lut(tmp_path / "v5.raw")
        assert (v5.version, v5.geom, v5.knots) == (5, 0, 0) and np.array_equal(v5.a1_icdf, lut.a1_icdf)
        lut.knots = lc.KNOTS_TAIL
        with pytest.raises(ValueError, match="v5"):
            lc.write_stepend_lut(tmp_path / "x5.raw", lut)
        lut.version, lut.knots = 6, lc.KNOTS_UNIFORM
        # combine / select are inverses
        two = lc.combine_stepend([lut, back])
        assert two.n_m == 2 and np.array_equal(lc.select_stepend(two, 1).a1_icdf, lut.a1_icdf)

    def test_rejects(self, tmp_path):
        lut = lut_from_rows(synth_histories(500), frame=0, n_z=4, n_r=3, n_a=3)
        p = tmp_path / "s.raw"
        lc.write_stepend_lut(p, lut)
        b = bytearray(p.read_bytes())
        bad = bytearray(b); bad[4:8] = (4).to_bytes(4, "little")
        (tmp_path / "v4.raw").write_bytes(bad)
        with pytest.raises(ValueError, match="version/kind"):
            lc.read_stepend_lut(tmp_path / "v4.raw")
        (tmp_path / "short.raw").write_bytes(b[:-4])
        with pytest.raises(ValueError, match="truncated"):
            lc.read_stepend_lut(tmp_path / "short.raw")
        with pytest.raises(ValueError, match="version/kind"):          # ann/scatt readers keep v3/v4
            lc.read_luts(p, p)
        lut.step_cm = 0.0
        with pytest.raises(ValueError, match="step_cm"):
            lc.write_stepend_lut(tmp_path / "x.raw", lut)
        a = lut_from_rows(synth_histories(500), frame=0, n_z=4, n_r=3, n_a=3)
        c = lut_from_rows(synth_histories(500), frame=1, n_z=4, n_r=3, n_a=3)
        with pytest.raises(ValueError, match="frame"):
            lc.combine_stepend([a, c])
        # v6 header fields
        for off, val, msg in ((44, 2, "geom"), (44, 1, "no longer supported"), (48, 7, "knots"), (52, 1, "reserved")):
            bad = bytearray(b); bad[off:off + 4] = val.to_bytes(4, "little")
            (tmp_path / "h.raw").write_bytes(bad)
            with pytest.raises(ValueError, match=msg):
                lc.read_stepend_lut(tmp_path / "h.raw")
        bad = bytearray(b); bad[48:52] = (1).to_bytes(4, "little")          # tail with N_a = 3
        (tmp_path / "h.raw").write_bytes(bad)
        with pytest.raises(ValueError, match="N_a"):
            lc.read_stepend_lut(tmp_path / "h.raw")


# ---------------------------------------------------------------------------
class TestRow:
    @pytest.mark.parametrize("frame", [0, 1])
    def test_shapes_monotone_ranges(self, frame):
        z, r, th, dp, u = synth_histories(20000, seed=3)
        zq, rq, a1, a2, uq = lc.build_stepend_row(z, r, th, dp, u, frame, 32, 16, 16, STEP)
        assert zq.shape == (32,) and rq.shape == (31, 16)
        assert a1.shape == a2.shape == uq.shape == (31, 15, 16)
        assert all(x.dtype == np.float32 for x in (zq, rq, a1, a2, uq))
        for x in (zq, rq, a1, a2, uq):
            assert np.all(np.diff(x, axis=-1) >= 0)
        assert np.all(np.abs(zq) <= np.float32(STEP)) and np.all((rq >= 0) & (rq <= np.float32(STEP)))
        assert np.all((a1 >= 0) & (a1 <= np.float32(np.pi))) and np.all((a2 >= 0) & (a2 <= np.float32(np.pi)))
        assert np.all((uq >= 0) & (uq < 1))
        # z knots are build_row's (the stop tables' construction) on the same z
        zr, _ = lc.build_row(z, r, 32, 16, *lc.stop_ranges(STEP))
        assert np.array_equal(zq, zr)

    def test_r_given_z_matches_build_row(self):
        """r | z bin uses build_row's construction exactly."""
        z, r, th, dp, u = synth_histories(5000, seed=4)
        zq, rq, *_ = lc.build_stepend_row(z, r, th, dp, u, 1, 8, 6, 4, STEP)
        zr, rr = lc.build_row(z, r, 8, 6, *lc.stop_ranges(STEP))
        assert np.array_equal(zq, zr) and np.array_equal(rq, rr)

    def test_empty_bins_borrow(self):
        """Many tied z values: empty z bins and (z, r) cells never hold zeros."""
        z, r, th, dp, u = synth_histories(4000, seed=5)
        z[:3000] = 0.02
        zq, rq, a1, a2, uq = lc.build_stepend_row(z, r, th, dp, u, 1, 16, 8, 8, STEP)
        assert np.all(np.any(rq != 0, axis=-1)) and np.all(np.any(uq != 0, axis=-1))

    def test_pooling(self):
        rng = np.random.default_rng(0)
        sizes = [0, 10, 60, 5, 0, 200, 3]
        rows = [rng.random((n, 5)) for n in sizes]
        valid, pooled = lc.classify_and_pool_stepend(rows, n_full=50, k_max=2)
        tags = [np.full(n, i) for i, n in enumerate(sizes)]
        v_ref, t_ref, _ = lc.classify_and_pool(tags, tags, n_full=50, k_max=2)
        assert np.array_equal(valid, v_ref)
        assert valid.tolist() == [DEAD, POOLED, FULL, POOLED, DEAD, FULL, POOLED]
        for i in range(len(sizes)):
            if valid[i] == DEAD:
                assert pooled[i] is None
                continue
            assert pooled[i].shape == (t_ref[i].size, 5)
            want = np.concatenate([rows[q] for q in dict.fromkeys(t_ref[i].tolist())])
            assert np.array_equal(pooled[i], want)
        assert pooled[2] is rows[2]                                     # FULL rows are not copied
        # default threshold is the spec's
        assert lc.N_FULL_SE == 5000
        v, _ = lc.classify_and_pool_stepend([rng.random((4999, 5)), rng.random((5000, 5))], k_max=0)
        assert v.tolist() == [SPARSE, FULL]


# ---------------------------------------------------------------------------
class TestFrames:
    def test_frame1_known_values(self):
        h = np.pi / 2
        a, p = lc.stepend_angles(0.0, 1.0, h, 0.0, 1)          # n along x = chord
        assert a == pytest.approx(0.0, abs=1e-12)
        a, p = lc.stepend_angles(1.0, 1.0, 0.0, 0.0, 1)        # n = d0, chord at 45 deg
        assert a == pytest.approx(np.pi / 4) and p == pytest.approx(np.pi)    # b1 points away from d0
        a, p = lc.stepend_angles(1.0, 0.0, h, h, 1)            # chord = d0, n = +y
        assert a == pytest.approx(h) and p == pytest.approx(h)
        a, p = lc.stepend_angles(1.0, 0.0, h, -h, 1)           # mirror: |psi| stored
        assert p == pytest.approx(h)
        a, p = lc.stepend_angles(0.0, 0.0, 0.7, 1.1, 1)        # rho = 0
        assert a == 0.7 and p == 0.0
        a, p = lc.stepend_angles(0.3, 0.2, 0.7, -1.1, 0)       # frame 0
        assert a == 0.7 and p == pytest.approx(1.1)

    def test_frame1_vs_vector_algebra(self):
        z, r, th, dp, u = synth_histories(1000, seed=7)
        al, ps = lc.stepend_angles(z, r, th, dp, 1)
        n = n_from_raw(th, dp)
        rho = np.hypot(z, r)
        c = np.stack([r, 0 * r, z], -1) / rho[:, None]
        b1 = np.stack([z, 0 * r, -r], -1) / rho[:, None]
        b2 = np.cross(c, b1)
        assert np.allclose(b2, [0, 1, 0])
        assert np.allclose(np.cos(al), np.einsum("ij,ij->i", n, c))
        psi = np.arctan2(np.einsum("ij,ij->i", n, b2), np.einsum("ij,ij->i", n, b1))
        assert np.allclose(ps, np.abs(psi))

    @pytest.mark.parametrize("frame", [0, 1])
    def test_geometry_inverts_angles(self, frame):
        """(a1, a2, sign) -> direction reproduces the raw direction (phi = 0)."""
        z, r, th, dp, u = synth_histories(2000, seed=8)
        a1, a2 = lc.stepend_angles(z, r, th, dp, frame)
        s = np.where(dp >= 0, 1.0, -1.0)
        disp, n = lc.stepend_geometry(z, r, a1, a2, s, frame)
        assert np.allclose(n, n_from_raw(th, dp), atol=1e-9)
        assert np.allclose(disp, np.stack([r, 0 * r, z], -1))
        assert np.allclose(np.linalg.norm(n, axis=-1), 1.0)

    @pytest.mark.parametrize("frame", [0, 1])
    def test_geometry_rotates_with_phi(self, frame):
        z, r, th, dp, u = synth_histories(300, seed=9)
        a1, a2 = lc.stepend_angles(z, r, th, np.abs(dp), frame)
        phi = 1.234
        d1, n1 = lc.stepend_geometry(z, r, a1, a2, 1.0, frame, phi=phi, scale=0.5)
        d0_, n0 = lc.stepend_geometry(z, r, a1, a2, 1.0, frame)
        rot = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
        assert np.allclose(n1, n0 @ rot.T) and np.allclose(d1, 0.5 * d0_ @ rot.T)

    @pytest.mark.parametrize("frame", [0, 1])
    def test_single_history_table_reproduces_it(self, frame):
        """Tables from one repeated history: every sample (sign +) is that history."""
        z0, r0, th0, dp0, u0 = 0.031, 0.017, 1.9, 0.8, 0.37
        n = 200
        cols = [np.full(n, v) for v in (z0, r0, th0, dp0, u0)]
        lut = lut_from_rows(cols, frame)
        rng = np.random.default_rng(1)
        rr = rng.random((6, 500))
        rr[5] *= 0.5                                       # r6 < 0.5: sign +1
        has, z, r, a1, a2, u, s = lc.sample_stepend_many(lut, np.zeros(500, int), np.full(500, 1.3e5), *rr)
        assert has.all() and np.all(s == 1.0)
        assert np.allclose(z, z0, atol=1e-7) and np.allclose(r, r0, atol=1e-7)
        assert np.allclose(u, u0, atol=1e-7)
        disp, nn = lc.stepend_geometry(z, r, a1, a2, s, frame)
        assert np.allclose(nn, n_from_raw(th0, dp0), atol=1e-6)
        assert np.allclose(disp, [r0, 0, z0], atol=1e-7)


# ---------------------------------------------------------------------------
class TestSampling:
    def test_within_cell_independence_is_a_model_limit(self):
        """Documented limit: on histories where a1 and a2 are correlated INSIDE a
        (z, r) cell, the tables (independent r3, r4) do not reproduce corr(a1, a2)."""
        z, r, th, dp, u = synth_histories(100_000, seed=12)
        lut = lut_from_rows((z, r, th, dp, u), 1, n_z=32, n_r=16, n_a=16)
        rr = np.random.default_rng(3).random((6, 300_000))
        _, zs, rs, a1, a2, us, s = lc.sample_stepend_many(lut, np.zeros(300_000, int), np.full(300_000, 1e5), *rr)
        a1r, a2r = lc.stepend_angles(z, r, th, dp, 1)
        assert abs(np.corrcoef(a1, a2)[0, 1] - np.corrcoef(a1r, a2r)[0, 1]) > 0.1
        assert np.corrcoef(zs, a1)[0, 1] == pytest.approx(np.corrcoef(z, a1r)[0, 1], abs=0.03)

    def test_scalar_equals_vector(self):
        lut = lut_from_rows(synth_histories(5000, seed=11), frame=1, e_keV=(50.0, 100.0, 300.0))
        lut.evalid[0, 2] = DEAD
        for a in ("z_icdf_s", "r_icdf_s", "a1_icdf", "a2_icdf", "u_icdf_s"):
            getattr(lut, a)[0, 2] = 0
        lut.check()
        rng = np.random.default_rng(2)
        n = 400
        E = np.exp(rng.uniform(np.log(4e4), np.log(4e5), n))
        rr = rng.random((6, n))
        vec = lc.sample_stepend_many(lut, np.zeros(n, int), E, *rr)
        for q in range(n):
            sc = lc.sample_stepend(lut, 0, E[q], *rr[:, q])
            assert sc == tuple(x[q] for x in vec) or np.allclose(sc, [x[q] for x in vec], rtol=0, atol=1e-15)
        # above the last live row the lower row takes all weight; only DEAD rows -> has False
        lut.evalid[0, 1] = DEAD
        for a in ("z_icdf_s", "r_icdf_s", "a1_icdf", "a2_icdf", "u_icdf_s"):
            getattr(lut, a)[0, 1] = 0
        assert lc.sample_stepend(lut, 0, 2e5, *([0.3] * 6))[0] is False
        assert lc.sample_stepend(lut, 0, 7e4, *([0.3] * 6))[0] is True

    @pytest.mark.parametrize("frame", [0, 1])
    def test_joint_moments(self, frame):
        """Sampling a table built from N correlated histories reproduces their
        means, correlations and the new direction's distribution."""
        N = 200_000
        z, r, th, dp, u = synth_conditional(N, frame, seed=12)
        lut = lut_from_rows((z, r, th, dp, u), frame, n_z=32, n_r=16, n_a=16)
        M = 1_000_000
        rr = np.random.default_rng(13).random((6, M))
        has, zs, rs, a1, a2, us, s = lc.sample_stepend_many(lut, np.zeros(M, int), np.full(M, 1e5), *rr)
        a1r, a2r = lc.stepend_angles(z, r, th, dp, frame)
        ref = np.stack([z, r, a1r, a2r, u])
        smp = np.stack([zs, rs, a1, a2, us])
        assert np.allclose(smp.mean(1), ref.mean(1), rtol=5e-3, atol=2e-4 * np.abs(ref).mean(1))
        assert np.allclose(smp.std(1), ref.std(1), rtol=2e-2)
        # a1, a2, u are sampled independently within a (z, r) cell (spec): with
        # histories that satisfy that, every pairwise correlation is reproduced
        assert np.allclose(np.corrcoef(smp), np.corrcoef(ref), atol=0.03)
        # the new direction and its correlation with the end point
        _, n_s = lc.stepend_geometry(zs, rs, a1, a2, s, frame)
        n_r = n_from_raw(th, dp)
        assert n_s[:, 2].mean() == pytest.approx(n_r[:, 2].mean(), abs=5e-3)      # <cos theta>
        assert n_s[:, 0].mean() == pytest.approx(n_r[:, 0].mean(), abs=5e-3)      # toward the lateral shift
        assert abs(n_s[:, 1].mean()) < 5e-3                                       # mirror symmetric
        assert np.corrcoef(rs, n_s[:, 2])[0, 1] == pytest.approx(np.corrcoef(r, n_r[:, 2])[0, 1], abs=0.03)


# ---------------------------------------------------------------------------
HEADER10 = "# E0_keV theta_rad DE_total_eV dE_kin_eV E_final_eV annihil_flag matID z_cm r_cm dphi_rad\n"


def write_raw10(path, e_keV, n_hist, p_ann, mat_id, seed=0, columns=10, geom=None):
    rng = np.random.default_rng(seed)
    g = f" geom={geom}" if geom else ""
    lines = [f"# config: engine=test step_cm= {STEP:.8E} nhist={n_hist} rho_gcm3= 1.0{g}\n",
             HEADER10 if columns == 10 else HEADER10.replace(" dphi_rad", "")]
    for e, pa in zip(e_keV, p_ann):
        e_eV = e * 1000.0
        ann = rng.random(n_hist) < pa
        z, r, th, dp, u = synth_histories(n_hist, seed=seed + int(e))
        for a, zz, rr, t, d, uu in zip(ann, z, r, th, dp, u):
            if a:
                line = f"{e:10.4f} {0.0:11.6f} {e_eV + 1.022e6:.6e} {e_eV:.6e} 0.0 1 {mat_id} {0.3 * zz:11.7f} {rr:11.7f}"
            else:
                de = uu * e_eV
                line = f"{e:10.4f} {t:11.6f} {de:.6e} {de:.6e} {e_eV - de:.6e} 0 {mat_id} {zz:11.7f} {rr:11.7f}"
            lines.append(line + (f" {d:10.6f}\n" if columns == 10 else "\n"))
    path.write_text("".join(lines))


class TestBuild:
    def test_raw_detection_and_load(self, tmp_path):
        write_raw10(tmp_path / "mat1.dat", [10.0, 20.0], 30, [0.2, 0.0], 1)
        write_raw10(tmp_path / "mat2.dat", [10.0, 20.0], 30, [0.2, 0.0], 2, columns=9)
        assert lc.raw_has_dphi(tmp_path / "mat1.dat") and lc.raw_has_stop(tmp_path / "mat1.dat")
        assert not lc.raw_has_dphi(tmp_path / "mat2.dat") and lc.raw_has_stop(tmp_path / "mat2.dat")
        raw = lc.load_raw_full(tmp_path / "mat1.dat")
        assert raw["dphi"].shape == (60,) and raw["dphi"].dtype == np.float32
        assert lc.load_raw_full(tmp_path / "mat2.dat")["dphi"] is None

    def test_chunked_read_equals_whole(self, tmp_path, monkeypatch):
        write_raw10(tmp_path / "mat1.dat", [10.0, 20.0, 50.0], 40, [0.2, 0.1, 0.0], 1)
        whole = lc.load_raw_full(tmp_path / "mat1.dat")
        monkeypatch.setattr(lc, "RAW_CHUNK_ROWS", 7)
        chunked = lc.load_raw_full(tmp_path / "mat1.dat")
        for k in whole:
            assert np.array_equal(whole[k], chunked[k]), k

    @pytest.mark.parametrize("frame", [0, 1])
    def test_build_engine_writes_stepend(self, tmp_path, frame):
        raw = tmp_path / "raw"; raw.mkdir()
        e = [10.0, 20.0, 50.0, 100.0]
        write_raw10(raw / "mat1.dat", e, 300, [1.0, 0.5, 0.1, 0.0], 1, seed=1)
        write_raw10(raw / "mat3.dat", e, 300, [0.3, 0.2, 0.1, 0.0], 3, seed=2)
        write_raw10(raw / "mat7.dat", e, 300, [0.3, 0.2, 0.1, 0.0], 7, seed=3)
        out = tmp_path / "out"
        lut, rows = build_tables.build_engine("penelope", raw, out, n_theta=8, n_u=4, n_full=100, k_max=1,
                                              workers=2, materials=[1, 7], frame=frame, n_z_se=6,
                                              n_r_se=4, n_a=5, n_full_se=250, knots=lc.KNOTS_UNIFORM)
        assert lut.n_m == 2
        se = lc.read_stepend_lut(out / "stepend_LUT.raw")
        assert se.frame == frame and (se.n_z, se.n_r, se.n_a, se.n_full_se) == (6, 4, 5, 250)
        assert np.array_equal(se.e_grid, lut.e_grid) and se.n_m == 2
        assert se.evalid[0].tolist() == [DEAD, POOLED, FULL, FULL]      # 150 survivors + neighbour
        assert se.evalid[1].tolist() == [POOLED, POOLED, FULL, FULL]     # 210, 240 survivors < 250
        se.check()
        info = json.loads((out / "run.json").read_text())
        assert info["materials"] == [1, 7] and info["stepend"] is True
        assert info["frame"] == frame and info["n_a"] == 5 and info["n_full_se"] == 250
        assert info["stepend_version"] == 6 and info["geom"] == "path" and info["knots"] == "uniform"
        assert (se.version, se.geom, se.knots) == (6, lc.GEOM_PATH, lc.KNOTS_UNIFORM)
        census = (out / "stepend_census.csv").read_text().splitlines()
        assert census[0].startswith("engine,material") and len(census) == 1 + 8
        # same tables as the per-material path, slot order numeric
        one = build_tables.process_material(raw / "mat7.dat", 8, 4, 100, 1, verbose=False, step_cm=STEP,
                                            stepend=dict(frame=frame, n_z=6, n_r=4, n_a=5, n_full_se=250,
                                                         knots=lc.KNOTS_UNIFORM))
        assert np.array_equal(one[2].a1_icdf[0], se.a1_icdf[1])
        with pytest.raises(SystemExit, match="no raw file"):
            build_tables.build_engine("penelope", raw, out, 8, 4, 100, 1, workers=1, materials=[2])

    def test_mixed_raw_builds_no_stepend(self, tmp_path):
        raw = tmp_path / "raw"; raw.mkdir()
        e = [10.0, 20.0]
        write_raw10(raw / "mat1.dat", e, 50, [0.1, 0.0], 1)
        write_raw10(raw / "mat2.dat", e, 50, [0.1, 0.0], 2, columns=9)
        out = tmp_path / "out"
        with pytest.warns(RuntimeWarning, match="no step-end tables"):
            lut, _ = build_tables.build_engine("penelope", raw, out, 8, 4, 10, 1, workers=1)
        assert lut.has_stop and not (out / "stepend_LUT.raw").exists()
        assert json.loads((out / "run.json").read_text())["stepend"] is False
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            build_tables.build_engine("penelope", raw, tmp_path / "o2", 8, 4, 10, 1, workers=1, stepend=False,
                                      materials=[1])
        assert not (tmp_path / "o2" / "stepend_LUT.raw").exists()


# ---------------------------------------------------------------------------
# v6: tail-dense knots (emulator emu6.py, model zrapt); the sphere layout (sphut) is gone
class TestV6:
    def _cells(self, z, s2, n_z, n_r):
        """Members of every (z bin, slot-2 bin) cell, by the builder's own binning."""
        zq = lc.quantiles(z, n_z)
        out = {}
        for j, idx in enumerate(lc._bin_members(lc._qbins(z, zq), n_z - 1)):
            rq = lc.quantiles(s2[idx], n_r)
            for k, kk in enumerate(lc._bin_members(lc._qbins(s2[idx], rq), n_r - 1)):
                out[j, k] = idx[kk]
        return out

    def test_tail_knots_are_the_level_quantiles(self):
        z, r, th, dp, u = synth_histories(20000, seed=21)
        zq, s2q, a1, a2, uq = lc.build_stepend_row(z, r, th, dp, u, 1, 8, 6, 16, STEP, knots=lc.KNOTS_TAIL)
        al, ps = lc.stepend_angles(z, r, th, dp, 1)
        s2 = r
        for (j, k), sel in self._cells(z, s2, 8, 6).items():
            for tab, x in ((a1, al), (a2, ps), (uq, u)):
                want = np.clip(np.quantile(x[sel], lc.TAIL_LEVELS.astype(np.float32).astype(np.float64)),
                               0, np.pi if x is not u else lc.U_MAX).astype(np.float32)
                assert np.array_equal(tab[j, k], want)
                assert tab[j, k, 0] == np.float32(x[sel].min()) and tab[j, k, -1] == np.float32(x[sel].max())
        # z and slot 2 (r) keep v5's construction (mean-preserving ends)
        zr, rr = lc.build_row(z, s2, 8, 6, lc.stop_ranges(STEP)[0], lc.stop_ranges(STEP)[1])
        assert np.array_equal(zq, zr) and np.array_equal(s2q, rr)
        with pytest.raises(ValueError, match="N_a"):
            lc.build_stepend_row(z, r, th, dp, u, 1, 8, 6, 8, STEP, knots=lc.KNOTS_TAIL)

    def test_uniform_path_is_v5(self):
        z, r, th, dp, u = synth_histories(5000, seed=22)
        a = lc.build_stepend_row(z, r, th, dp, u, 1, 8, 6, 16, STEP)
        b = lc.build_stepend_row(z, r, th, dp, u, 1, 8, 6, 16, STEP, geom=lc.GEOM_PATH, knots=lc.KNOTS_UNIFORM)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))

    def test_sampling_interpolates_on_the_levels(self):
        """a1 knots = pi * level: the sample is pi * r3 exactly (linear inverse on the levels);
        uniform knots on the same table would not be."""
        cols = synth_histories(3000, seed=23)
        lut = lut_from_rows(cols, 1, n_z=4, n_r=3, n_a=16, knots=lc.KNOTS_TAIL)
        lev = lc.TAIL_LEVELS.astype(np.float32)
        lut.a1_icdf[...] = np.float32(np.pi) * lev
        lut.check()
        rr = np.random.default_rng(4).random((6, 2000))
        rr[2, :16] = lev                                                 # exactly on the levels too
        rr[2, 15] = np.nextafter(1.0, 0.0)
        _, _, _, a1, *_ = lc.sample_stepend_many(lut, np.zeros(2000, int), np.full(2000, 1.5e5), *rr)
        assert np.allclose(a1, np.float32(np.pi) * rr[2], rtol=0, atol=2e-6)
        lut.knots = lc.KNOTS_UNIFORM
        _, _, _, b1, *_ = lc.sample_stepend_many(lut, np.zeros(2000, int), np.full(2000, 1.5e5), *rr)
        assert not np.allclose(b1, np.float32(np.pi) * rr[2], atol=1e-2)
        lut.knots = lc.KNOTS_TAIL
        for q in range(0, 2000, 97):                                    # scalar == vector
            sc = lc.sample_stepend(lut, 0, 1.5e5, *rr[:, q])
            vec = lc.sample_stepend_many(lut, [0], [1.5e5], *rr[:, q:q + 1])
            assert np.allclose(sc[1:], [x[0] for x in vec[1:]], rtol=0, atol=1e-15)

    def test_sphere_geometry_is_refused(self):
        """geom 1 (the sphere layout, dropped 2026-10-30): the row builder and the reader refuse it."""
        z, r, th, dp, u = synth_histories(2000, seed=24)
        with pytest.raises(ValueError, match="sphere layout"):
            lc.build_stepend_row(z, r, th, dp, u, 1, 8, 6, 16, STEP, geom=1, knots=lc.KNOTS_TAIL)
        assert lc.GEOM_NAMES == {lc.GEOM_PATH: "path"}

    def test_v6_path_roundtrip(self, tmp_path):
        lut = lut_from_rows(synth_histories(4000, seed=25), 1, n_z=5, n_r=4, n_a=16, knots=lc.KNOTS_TAIL)
        p = tmp_path / "s.raw"
        lc.write_stepend_lut(p, lut)
        b = p.read_bytes()
        assert struct.unpack("<3i", b[44:56]) == (0, 1, 0)
        assert np.array_equal(np.frombuffer(b, "<f4", 16, 56 + 8), lc.TAIL_LEVELS.astype(np.float32))
        back = lc.read_stepend_lut(p)
        assert (back.geom, back.knots, back.version) == (0, 1, 6) and np.array_equal(back.r_icdf_s, lut.r_icdf_s)
        bad = bytearray(b); bad[56 + 8 + 4:56 + 8 + 8] = struct.pack("<f", 0.15)
        (tmp_path / "b.raw").write_bytes(bad)
        with pytest.raises(ValueError, match="levels"):
            lc.read_stepend_lut(tmp_path / "b.raw")
        bad = bytearray(b); bad[44:48] = struct.pack("<i", 1)              # a sphere-layout header
        (tmp_path / "g.raw").write_bytes(bad)
        with pytest.raises(ValueError, match="no longer supported"):
            lc.read_stepend_lut(tmp_path / "g.raw")
        with pytest.raises(ValueError, match="knots"):
            lc.combine_stepend([lut, lut_from_rows(synth_histories(4000, seed=25), 1, n_z=5, n_r=4, n_a=16)])

    def test_build_engine_refuses_sphere_raw(self, tmp_path):
        raw = tmp_path / "raw"; raw.mkdir()
        e = [10.0, 20.0, 50.0]
        for m in (1, 2):
            write_raw10(raw / f"mat{m}.dat", e, 300, [0.3, 0.1, 0.0], m, seed=m)
        out = tmp_path / "out"
        build_tables.build_engine("penelope", raw, out, n_theta=8, n_u=4, n_full=100, k_max=1, workers=1,
                                  n_z_se=6, n_r_se=4, n_full_se=250)
        se = lc.read_stepend_lut(out / "stepend_LUT.raw")
        assert (se.version, se.geom, se.knots, se.n_a) == (6, lc.GEOM_PATH, lc.KNOTS_TAIL, 16)
        info = json.loads((out / "run.json").read_text())
        assert info["geom"] == "path" and info["knots"] == "tail" and len(info["knot_levels"]) == 16
        write_raw10(raw / "mat3.dat", e, 50, [0.3, 0.1, 0.0], 3, geom="sphere")   # a sphere raw among path ones
        with pytest.raises(SystemExit, match="sphere"):
            build_tables.build_engine("penelope", raw, tmp_path / "o2", 8, 4, 100, 1, workers=1)
        with pytest.raises(SystemExit, match="sphere"):
            build_tables.build_engine("penelope", raw, tmp_path / "o3", 8, 4, 100, 1, workers=1, materials=[3])
        with pytest.raises(ValueError, match="N_a"):
            build_tables.build_engine("penelope", raw, tmp_path / "o4", 8, 4, 100, 1, workers=1, materials=[1],
                                      n_a=8)
