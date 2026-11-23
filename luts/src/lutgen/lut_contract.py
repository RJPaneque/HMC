"""
lut_contract.py -- normative reference implementation of the LUT binary format
and sampling contract (see the "LUT format" section of the README).

Everything here is written to be ported line-by-line to Fortran/CUDA; the
vectorised sampler `sample_many` follows the scalar `sample` exactly.
"""
from __future__ import annotations

import contextlib
import io
import re
import struct
import subprocess
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"HMC2"
VERSION = 4          # 4: ann_LUT carries N_z, N_r, rho_nom and the stopping-position tables
                     # 3: header carries step_cm (float32); 2 had reserved zeros there
READABLE_VERSIONS = (3, 4)
KIND_ANN = 1
KIND_SCATT = 2
KIND_STEPEND = 3     # stepend_LUT.raw: its own version (VERSION_STEPEND), see STEPEND_V5_SPEC.md
VERSION_STEPEND = 6  # 6: header carries geom (0 = path, the only value) and the a-axis knot scheme + levels
READABLE_SE_VERSIONS = (5, 6)

DEAD, FULL, POOLED, SPARSE = 0, 1, 2, 3
VALID_NAMES = {DEAD: "DEAD", FULL: "FULL", POOLED: "POOLED", SPARSE: "SPARSE"}

DEFAULT_N_THETA = 128
DEFAULT_N_U = 64
DEFAULT_N_FULL = 2000
DEFAULT_K_MAX = 2
DEFAULT_N_Z = 32          # stopping-position knots along the initial direction
DEFAULT_N_R = 16          # ... and perpendicular to it, per z bin
DEFAULT_N_FULL_STOP = 500 # stopped histories for a FULL stopping-position row
DEFAULT_N_Z_SE = 32       # step-end tables (v5): z knots,
DEFAULT_N_R_SE = 16       # ... r knots per z bin,
DEFAULT_N_A = 16          # ... a1 / a2 / u knots per (z, r) cell
N_FULL_SE = 5000          # survivors for a FULL step-end row; 20000 pooled the rows just above the stopping
                          # energy with mostly higher-energy neighbours (+3.6 % end distance, review 2026-10-30)
DEFAULT_FRAME = 1         # 0 'theta' (theta, |dphi|), 1 'chord' (alpha, |psi|)
FRAME_NAMES = {0: "theta", 1: "chord"}
# v6 header field geom: the geometry of the raw data, 0 = path (fixed path length; slot 2 = r | z,
# a1 a2 u | (z, r bin)), the only valid value. 1 was the sphere first-passage layout (slot 2 = u | z),
# evaluated on 2026-10-30 and dropped (same step count, slightly worse r10); such files are refused.
GEOM_PATH = 0
GEOM_SPHERE_DROPPED = 1
GEOM_NAMES = {GEOM_PATH: "path"}
# v6: knot scheme of the a1 / a2 / u axes (N_a knots): uniform levels j/(N_a-1) with mean-preserving
# end knots (v5), or 'tail': the fixed levels TAIL_LEVELS, exact sample min / max at the ends, no
# mean-preserving ends, sampled by linear interpolation on the levels (stored in the file).
KNOTS_UNIFORM, KNOTS_TAIL = 0, 1
KNOT_NAMES = {KNOTS_UNIFORM: "uniform", KNOTS_TAIL: "tail"}
TAIL_LEVELS = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98, .99, .995, .999, 1.0])
DEFAULT_KNOTS = KNOTS_TAIL

F32 = np.float32
I32 = np.int32
U_MAX = float(np.nextafter(np.float32(1.0), np.float32(0.0)))   # u is stored in [0, 1) as float32:
                                                                 # the largest float32 below 1


# ---------------------------------------------------------------------------
# Row construction (§3)
# ---------------------------------------------------------------------------
def quantiles(x: np.ndarray, n: int) -> np.ndarray:
    """n quantiles at levels j/(n-1), linear order-statistic interpolation."""
    q = np.linspace(0.0, 1.0, n)
    return np.quantile(np.asarray(x, dtype=np.float64), q, method="linear")


def mean_preserving_ends(q: np.ndarray, x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Move the first and last knot so each edge interval keeps its sample mean.

    Linear ICDF interpolation spreads an interval's mass uniformly, so its mean
    is the interval midpoint. For the edge intervals [x_min, q_1] and
    [q_{n-2}, x_max] the sample extremes are outliers and the midpoint is far
    from the events' mean (biasing mean theta / mean u by up to tens of percent
    at MeV energies). Setting the end knot to 2*mean - inner_knot makes the
    uniform interval's mean equal the conditional mean of its events. Interior
    knots are untouched; results are clamped to [lo, hi] and stay monotone.
    """
    q = np.array(q, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    top = x[x >= q[-2]]
    if top.size:
        q[-1] = min(max(2.0 * top.mean() - q[-2], q[-2]), hi)
    bot = x[x <= q[1]]
    if bot.size:
        q[0] = max(min(2.0 * bot.mean() - q[1], q[1]), lo)
    return q


def build_row(theta: np.ndarray, u: np.ndarray, n_theta: int = DEFAULT_N_THETA,
              n_u: int = DEFAULT_N_U, theta_range=(0.0, np.pi),
              u_range=(0.0, U_MAX)) -> tuple[np.ndarray, np.ndarray]:
    """Conditional-quantile row: quantiles of `theta`, and of `u` conditioned
    on the theta-quantile bin. Built for the survivors' (theta, u); the
    stopping-position tables reuse it with (z, r) and their own ranges.

    Returns
    -------
    theta_icdf : float32 (n_theta,)      non-decreasing
    u_icdf     : float32 (n_theta-1, n_u) non-decreasing along axis 1
    """
    theta = np.asarray(theta, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    assert theta.shape == u.shape and theta.ndim == 1 and theta.size > 0
    t_lo, t_hi = theta_range
    u_lo, u_hi = u_range
    th_icdf = quantiles(theta, n_theta)
    # assign every survivor to a theta-quantile bin b in [0, n_theta-2]
    b = np.searchsorted(th_icdf, theta, side="right") - 1
    b = np.clip(b, 0, n_theta - 2)
    u_icdf = np.empty((n_theta - 1, n_u), dtype=np.float64)
    for bi in range(n_theta - 1):
        sel = u[b == bi]
        if sel.size == 0:
            # degenerate bin (many tied theta values): borrow the nearest
            # non-empty neighbour so the table never holds zeros for a live row
            lo, hi = bi - 1, bi + 1
            while True:
                cand = []
                if lo >= 0:
                    cand.append(u[b == lo])
                if hi <= n_theta - 2:
                    cand.append(u[b == hi])
                cand = [c for c in cand if c.size]
                if cand:
                    sel = np.concatenate(cand)
                    break
                lo -= 1
                hi += 1
                if lo < 0 and hi > n_theta - 2:
                    sel = u
                    break
        u_icdf[bi] = mean_preserving_ends(quantiles(sel, n_u), sel, u_lo, u_hi)
    # edge knots are adjusted only after binning, so bins follow the raw quantiles
    th_icdf = mean_preserving_ends(th_icdf, theta, t_lo, t_hi)
    th_icdf = np.maximum.accumulate(th_icdf)
    u_icdf = np.maximum.accumulate(u_icdf, axis=1)
    return (np.clip(th_icdf, t_lo, t_hi).astype(F32),
            np.clip(u_icdf, u_lo, u_hi).astype(F32))


def stop_ranges(step_cm: float):
    """Knot ranges of the stopping-position tables: z in [-step, step], r in [0, step]."""
    return (-float(step_cm), float(step_cm)), (0.0, float(step_cm))


# ---------------------------------------------------------------------------
# Row classification and pooling (§5)
# ---------------------------------------------------------------------------
def classify_and_pool(surv_theta: list[np.ndarray], surv_u: list[np.ndarray],
                      n_full: int = DEFAULT_N_FULL, k_max: int = DEFAULT_K_MAX):
    """For one material: lists indexed by energy row of survivor arrays.

    Returns (valid codes int32[N_E], pooled_theta list, pooled_u list) where
    pooled_* are the samples to feed build_row (None for DEAD rows).
    """
    n_e = len(surv_theta)
    valid = np.zeros(n_e, dtype=I32)
    out_t: list = [None] * n_e
    out_u: list = [None] * n_e
    for i in range(n_e):
        n = surv_theta[i].size
        if n == 0:
            valid[i] = DEAD
            continue
        if n >= n_full:
            valid[i] = FULL
            out_t[i], out_u[i] = surv_theta[i], surv_u[i]
            continue
        # pool neighbours
        k = 0
        tt, uu = [surv_theta[i]], [surv_u[i]]
        total = n
        while total < n_full and k < k_max:
            k += 1
            for j in (i - k, i + k):
                if 0 <= j < n_e and surv_theta[j].size:
                    tt.append(surv_theta[j]); uu.append(surv_u[j])
                    total += surv_theta[j].size
        out_t[i] = np.concatenate(tt); out_u[i] = np.concatenate(uu)
        valid[i] = POOLED if total >= n_full else SPARSE
    return valid, out_t, out_u


# ---------------------------------------------------------------------------
# Containers and I/O (§6)
# ---------------------------------------------------------------------------
@dataclass
class LutV2:
    e_grid: np.ndarray                     # float32 (N_E,)
    p_ann: np.ndarray                      # float32 (N_M, N_E)
    n_hist: np.ndarray                     # int32   (N_M, N_E)
    n_surv: np.ndarray                     # int32   (N_M, N_E)
    valid: np.ndarray                      # int32   (N_M, N_E)
    theta_icdf: np.ndarray                 # float32 (N_M, N_E, N_theta)
    u_icdf: np.ndarray                     # float32 (N_M, N_E, N_theta-1, N_u)
    step_cm: float = 0.0                   # path length per step the tables describe [cm]
    # version 4 additions (defaults: no stopping-position tables, unknown densities)
    rho_nom: np.ndarray = None             # float32 (N_M,) nominal density [g/cm3], 0 = unknown
    svalid: np.ndarray = None              # int32   (N_M, N_E) stopping-position row code (§5)
    z_icdf: np.ndarray = None              # float32 (N_M, N_E, N_z)         [cm]
    r_icdf: np.ndarray = None              # float32 (N_M, N_E, N_z-1, N_r)  [cm]

    def __post_init__(self):
        n_m, n_e = self.p_ann.shape
        if self.rho_nom is None:
            self.rho_nom = np.zeros(n_m, dtype=F32)
        if self.svalid is None:
            self.svalid = np.zeros((n_m, n_e), dtype=I32)
        if self.z_icdf is None:
            self.z_icdf = np.zeros((n_m, n_e, 0), dtype=F32)
        if self.r_icdf is None:
            self.r_icdf = np.zeros((n_m, n_e, 0, 0), dtype=F32)

    @property
    def n_m(self): return self.p_ann.shape[0]
    @property
    def n_e(self): return self.e_grid.shape[0]
    @property
    def n_theta(self): return self.theta_icdf.shape[2]
    @property
    def n_u(self): return self.u_icdf.shape[3]
    @property
    def n_z(self): return self.z_icdf.shape[2]
    @property
    def n_r(self): return self.r_icdf.shape[3]
    @property
    def has_stop(self): return self.n_z > 0

    def check(self):
        assert self.theta_icdf.shape == (self.n_m, self.n_e, self.n_theta)
        assert self.u_icdf.shape == (self.n_m, self.n_e, self.n_theta - 1, self.n_u)
        assert np.all(np.diff(self.e_grid) > 0), "energy grid not increasing"
        assert np.all((self.p_ann >= 0) & (self.p_ann <= 1))
        live = self.valid != DEAD
        assert np.all(np.diff(self.theta_icdf[live], axis=-1) >= 0), "theta ICDF not monotone"
        assert np.all(np.diff(self.u_icdf[live], axis=-1) >= 0), "u ICDF not monotone"
        assert not np.isnan(self.theta_icdf).any() and not np.isnan(self.u_icdf).any()
        assert np.all(self.theta_icdf[~live] == 0) and np.all(self.u_icdf[~live] == 0)
        assert self.rho_nom.shape == (self.n_m,) and np.all(self.rho_nom >= 0)
        assert self.svalid.shape == (self.n_m, self.n_e)
        if self.has_stop:
            assert self.n_z >= 3 and self.n_r >= 3
            assert self.z_icdf.shape == (self.n_m, self.n_e, self.n_z)
            assert self.r_icdf.shape == (self.n_m, self.n_e, self.n_z - 1, self.n_r)
            slive = self.svalid != DEAD
            assert np.all(np.diff(self.z_icdf[slive], axis=-1) >= 0), "z ICDF not monotone"
            assert np.all(np.diff(self.r_icdf[slive], axis=-1) >= 0), "r ICDF not monotone"
            assert not np.isnan(self.z_icdf).any() and not np.isnan(self.r_icdf).any()
            assert np.all(self.z_icdf[~slive] == 0) and np.all(self.r_icdf[~slive] == 0)
            s = np.float32(self.step_cm)
            assert np.all(np.abs(self.z_icdf) <= s) and np.all((self.r_icdf >= 0) & (self.r_icdf <= s))
        else:
            assert self.r_icdf.shape[2:] == (0, 0) and np.all(self.svalid == DEAD)
        return True


def _hdr(kind, n_e, n_m, n_theta, n_u, step_cm, n_z=0, n_r=0, version=VERSION):
    """32-byte header. ann:   magic version kind N_E N_M step_cm N_z N_r  (v3: 0 0)
                       scatt: magic version kind N_E N_M N_theta N_u step_cm"""
    if not step_cm > 0:
        raise ValueError(f"step_cm must be > 0 to write a LUT (got {step_cm})")
    head = MAGIC + struct.pack("<iiii", version, kind, n_e, n_m)
    if kind == KIND_ANN:
        return head + struct.pack("<fii", step_cm, n_z, n_r)
    return head + struct.pack("<iif", n_theta, n_u, step_cm)


def write_ann_lut(path, lut: LutV2, version=VERSION):
    """version=3 writes the old layout (no densities, no stopping tables)."""
    if version not in READABLE_VERSIONS:
        raise ValueError(f"cannot write LUT version {version}")
    n_z, n_r = (lut.n_z, lut.n_r) if version >= 4 else (0, 0)
    with open(path, "wb") as f:
        f.write(_hdr(KIND_ANN, lut.n_e, lut.n_m, 0, 0, lut.step_cm, n_z, n_r, version))
        f.write(lut.e_grid.astype(F32).tobytes())
        if version >= 4:
            f.write(lut.rho_nom.astype(F32).tobytes())
        f.write(lut.p_ann.astype(F32).tobytes())
        f.write(lut.n_hist.astype(I32).tobytes())
        f.write(lut.n_surv.astype(I32).tobytes())
        f.write(lut.valid.astype(I32).tobytes())
        if version >= 4:
            f.write(lut.svalid.astype(I32).tobytes())
            f.write(lut.z_icdf.astype(F32).tobytes())
            f.write(lut.r_icdf.astype(F32).tobytes())


def write_scatt_lut(path, lut: LutV2, version=VERSION):
    if version not in READABLE_VERSIONS:
        raise ValueError(f"cannot write LUT version {version}")
    with open(path, "wb") as f:
        f.write(_hdr(KIND_SCATT, lut.n_e, lut.n_m, lut.n_theta, lut.n_u, lut.step_cm, version=version))
        f.write(lut.e_grid.astype(F32).tobytes())
        f.write(lut.theta_icdf.astype(F32).tobytes())
        f.write(lut.u_icdf.astype(F32).tobytes())
        f.write(lut.valid.astype(I32).tobytes())


def _read_hdr(f, kind_expected):
    """-> (version, n_e, n_m, n_theta, n_u, step_cm, n_z, n_r)"""
    magic = f.read(4)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}; not a LUT file")
    version, kind = struct.unpack("<ii", f.read(8))
    if version not in READABLE_VERSIONS or kind != kind_expected:
        raise ValueError(f"version/kind mismatch: {version}/{kind} "
                         f"(expected {READABLE_VERSIONS}/{kind_expected})")
    n_e, n_m = struct.unpack("<ii", f.read(8))
    if kind == KIND_ANN:
        step_cm, n_z, n_r = struct.unpack("<fii", f.read(12))
        if version < 4:
            n_z = n_r = 0
        if (n_z == 0) != (n_r == 0) or (n_z and (n_z < 3 or n_r < 3)):
            raise ValueError(f"bad stopping-table dimensions N_z={n_z} N_r={n_r}")
        return version, n_e, n_m, 0, 0, step_cm, n_z, n_r
    n_th, n_u, step_cm = struct.unpack("<iif", f.read(12))
    return version, n_e, n_m, n_th, n_u, step_cm, 0, 0


def _arr(f, dtype, shape):
    n = int(np.prod(shape))
    a = np.frombuffer(f.read(n * 4), dtype=dtype)
    if a.size != n:
        raise ValueError("truncated LUT file")
    return a.reshape(shape).copy()


def read_luts(ann_path, scatt_path) -> LutV2:
    """Reads a version 3 or 4 pair (a v3 pair comes back without densities or
    stopping tables); the two files must carry the same version."""
    with open(ann_path, "rb") as f:
        ver, n_e, n_m, _, _, step, n_z, n_r = _read_hdr(f, KIND_ANN)
        eg = _arr(f, F32, (n_e,))
        rho = _arr(f, F32, (n_m,)) if ver >= 4 else np.zeros(n_m, F32)
        pa = _arr(f, F32, (n_m, n_e))
        nh = _arr(f, I32, (n_m, n_e))
        ns = _arr(f, I32, (n_m, n_e))
        va = _arr(f, I32, (n_m, n_e))
        if ver >= 4:
            sv = _arr(f, I32, (n_m, n_e))
            zz = _arr(f, F32, (n_m, n_e, n_z))
            rr = _arr(f, F32, (n_m, n_e, max(n_z - 1, 0), n_r))
        else:
            sv, zz, rr = None, None, None
    with open(scatt_path, "rb") as f:
        ver2, n_e2, n_m2, n_th, n_u, step2, _, _ = _read_hdr(f, KIND_SCATT)
        eg2 = _arr(f, F32, (n_e2,))
        th = _arr(f, F32, (n_m2, n_e2, n_th))
        uu = _arr(f, F32, (n_m2, n_e2, n_th - 1, n_u))
        va2 = _arr(f, I32, (n_m2, n_e2))
    if ver != ver2:
        raise ValueError(f"ann_LUT and scatt-eloss_LUT disagree on version ({ver} vs {ver2})")
    if (n_e, n_m) != (n_e2, n_m2) or not np.array_equal(eg, eg2) or not np.array_equal(va, va2):
        raise ValueError("ann_LUT and scatt-eloss_LUT disagree (dims, grid or validity)")
    if step != step2:
        raise ValueError(f"ann_LUT and scatt-eloss_LUT disagree on step_cm ({step} vs {step2})")
    return LutV2(eg, pa, nh, ns, va, th, uu, float(step), rho, sv, zz, rr)


def combine(materials: list[LutV2]) -> LutV2:
    eg = materials[0].e_grid
    for m in materials:
        if not np.array_equal(m.e_grid, eg):
            raise ValueError("energy grid mismatch across materials")
        if (m.n_theta, m.n_u) != (materials[0].n_theta, materials[0].n_u):
            raise ValueError("quantile dimension mismatch across materials")
        if m.step_cm != materials[0].step_cm:
            raise ValueError("step_cm mismatch across materials")
        if (m.n_z, m.n_r) != (materials[0].n_z, materials[0].n_r):
            raise ValueError("stopping-table dimension mismatch across materials")
    cat = lambda attr: np.concatenate([getattr(m, attr) for m in materials], axis=0)
    return LutV2(eg, cat("p_ann"), cat("n_hist"), cat("n_surv"), cat("valid"),
                 cat("theta_icdf"), cat("u_icdf"), materials[0].step_cm,
                 cat("rho_nom"), cat("svalid"), cat("z_icdf"), cat("r_icdf"))


def select(lut: LutV2, m: int) -> LutV2:
    """Material slot m (0-based) as a one-material LutV2; inverse of combine."""
    s = slice(m, m + 1)
    return LutV2(lut.e_grid, lut.p_ann[s], lut.n_hist[s], lut.n_surv[s], lut.valid[s],
                 lut.theta_icdf[s], lut.u_icdf[s], lut.step_cm,
                 lut.rho_nom[s], lut.svalid[s], lut.z_icdf[s], lut.r_icdf[s])


# ---------------------------------------------------------------------------
# Sampling contract (§4) -- scalar, literal
# ---------------------------------------------------------------------------
def bracket(e_grid: np.ndarray, E: float) -> tuple[int, float]:
    n_e = e_grid.shape[0]
    lg = np.log(e_grid.astype(np.float64))
    lE = np.log(E)
    if lE <= lg[0]:
        return 0, 0.0
    if lE >= lg[n_e - 1]:
        return n_e - 2, 1.0
    i = int(np.searchsorted(lg, lE, side="right") - 1)
    i = min(max(i, 0), n_e - 2)
    t = (lE - lg[i]) / (lg[i + 1] - lg[i])
    return i, float(min(max(t, 0.0), 1.0))


def sample(lut: LutV2, m: int, E: float, r0: float, r1: float, r2: float):
    """Returns (annihilated: bool, theta: float, u: float). m is 0-based."""
    i, t = bracket(lut.e_grid, E)
    p = (1.0 - t) * float(lut.p_ann[m, i]) + t * float(lut.p_ann[m, i + 1])
    if r0 < p:
        return True, 0.0, 0.0
    vi, vi1 = lut.valid[m, i] != DEAD, lut.valid[m, i + 1] != DEAD
    if vi and vi1:
        wi, wi1 = 1.0 - t, t
    elif vi:
        wi, wi1 = 1.0, 0.0
    elif vi1:
        wi, wi1 = 0.0, 1.0
    else:
        return True, 0.0, 0.0
    nth, nu = lut.n_theta, lut.n_u
    p1 = r1 * (nth - 1)
    j = min(max(int(np.floor(p1)), 0), nth - 2)
    w = p1 - j
    th = lut.theta_icdf
    th_lo = wi * float(th[m, i, j]) + wi1 * float(th[m, i + 1, j])
    th_hi = wi * float(th[m, i, j + 1]) + wi1 * float(th[m, i + 1, j + 1])
    theta = (1.0 - w) * th_lo + w * th_hi
    b = j
    p2 = r2 * (nu - 1)
    k = min(max(int(np.floor(p2)), 0), nu - 2)
    w2 = p2 - k
    uu = lut.u_icdf
    u_lo = wi * float(uu[m, i, b, k]) + wi1 * float(uu[m, i + 1, b, k])
    u_hi = wi * float(uu[m, i, b, k + 1]) + wi1 * float(uu[m, i + 1, b, k + 1])
    u = (1.0 - w2) * u_lo + w2 * u_hi
    u = min(max(u, 0.0), np.nextafter(1.0, 0.0))
    return False, theta, u


# ---------------------------------------------------------------------------
# Vectorised sampler -- identical arithmetic to `sample`
# ---------------------------------------------------------------------------
def sample_many(lut: LutV2, m: np.ndarray, E: np.ndarray, r0: np.ndarray,
                r1: np.ndarray, r2: np.ndarray):
    m = np.asarray(m, dtype=np.int64); E = np.asarray(E, dtype=np.float64)
    r0 = np.asarray(r0, dtype=np.float64); r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)
    n_e, nth, nu = lut.n_e, lut.n_theta, lut.n_u
    lg = np.log(lut.e_grid.astype(np.float64)); lE = np.log(E)
    i = np.searchsorted(lg, lE, side="right") - 1
    i = np.clip(i, 0, n_e - 2)
    t = (lE - lg[i]) / (lg[i + 1] - lg[i])
    t = np.where(lE <= lg[0], 0.0, np.where(lE >= lg[-1], 1.0, t))
    t = np.clip(t, 0.0, 1.0)
    pa = lut.p_ann.astype(np.float64)
    p = (1 - t) * pa[m, i] + t * pa[m, i + 1]
    ann = r0 < p
    vi = lut.valid[m, i] != DEAD; vi1 = lut.valid[m, i + 1] != DEAD
    wi = np.where(vi & vi1, 1 - t, np.where(vi, 1.0, 0.0))
    wi1 = np.where(vi & vi1, t, np.where(vi, 0.0, np.where(vi1, 1.0, 0.0)))
    ann |= ~(vi | vi1)
    p1 = r1 * (nth - 1); j = np.clip(np.floor(p1).astype(np.int64), 0, nth - 2); w = p1 - j
    th = lut.theta_icdf.astype(np.float64)
    th_lo = wi * th[m, i, j] + wi1 * th[m, i + 1, j]
    th_hi = wi * th[m, i, j + 1] + wi1 * th[m, i + 1, j + 1]
    theta = (1 - w) * th_lo + w * th_hi
    p2 = r2 * (nu - 1); k = np.clip(np.floor(p2).astype(np.int64), 0, nu - 2); w2 = p2 - k
    uu = lut.u_icdf.astype(np.float64)
    u_lo = wi * uu[m, i, j, k] + wi1 * uu[m, i + 1, j, k]
    u_hi = wi * uu[m, i, j, k + 1] + wi1 * uu[m, i + 1, j, k + 1]
    u = np.clip((1 - w2) * u_lo + w2 * u_hi, 0.0, np.nextafter(1.0, 0.0))
    theta = np.where(ann, 0.0, theta); u = np.where(ann, 0.0, u)
    return ann, theta, u


# ---------------------------------------------------------------------------
# Stopping position (§4bis) -- scalar, literal, and vectorised
# ---------------------------------------------------------------------------
def _row_weights(valid, m, i, t):
    vi, vi1 = valid[m, i] != DEAD, valid[m, i + 1] != DEAD
    if vi and vi1:
        return True, 1.0 - t, t
    if vi:
        return True, 1.0, 0.0
    if vi1:
        return True, 0.0, 1.0
    return False, 0.0, 0.0


def sample_stop(lut: LutV2, m: int, E: float, r3: float, r4: float):
    """Where a positron of energy E that stops within the step ends up, relative
    to the step start: (has, z, r) with z along the initial direction and r
    perpendicular to it [cm]. has = False when neither bracketing row holds a
    stopping-position table (the engine then uses the hinge point)."""
    if not lut.has_stop:
        return False, 0.0, 0.0
    i, t = bracket(lut.e_grid, E)
    ok, wi, wi1 = _row_weights(lut.svalid, m, i, t)
    if not ok:
        return False, 0.0, 0.0
    nz, nr = lut.n_z, lut.n_r
    p3 = r3 * (nz - 1)
    j = min(max(int(np.floor(p3)), 0), nz - 2)
    w = p3 - j
    zz = lut.z_icdf
    z_lo = wi * float(zz[m, i, j]) + wi1 * float(zz[m, i + 1, j])
    z_hi = wi * float(zz[m, i, j + 1]) + wi1 * float(zz[m, i + 1, j + 1])
    z = (1.0 - w) * z_lo + w * z_hi
    p4 = r4 * (nr - 1)
    k = min(max(int(np.floor(p4)), 0), nr - 2)
    w2 = p4 - k
    rr = lut.r_icdf
    r_lo = wi * float(rr[m, i, j, k]) + wi1 * float(rr[m, i + 1, j, k])
    r_hi = wi * float(rr[m, i, j, k + 1]) + wi1 * float(rr[m, i + 1, j, k + 1])
    r = (1.0 - w2) * r_lo + w2 * r_hi
    s = float(lut.step_cm)
    return True, min(max(z, -s), s), min(max(r, 0.0), s)


def sample_stop_many(lut: LutV2, m: np.ndarray, E: np.ndarray, r3: np.ndarray, r4: np.ndarray):
    m = np.asarray(m, dtype=np.int64); E = np.asarray(E, dtype=np.float64)
    r3 = np.asarray(r3, dtype=np.float64); r4 = np.asarray(r4, dtype=np.float64)
    if not lut.has_stop:
        return np.zeros(m.shape, bool), np.zeros(m.shape), np.zeros(m.shape)
    n_e, nz, nr = lut.n_e, lut.n_z, lut.n_r
    lg = np.log(lut.e_grid.astype(np.float64)); lE = np.log(E)
    i = np.clip(np.searchsorted(lg, lE, side="right") - 1, 0, n_e - 2)
    t = (lE - lg[i]) / (lg[i + 1] - lg[i])
    t = np.clip(np.where(lE <= lg[0], 0.0, np.where(lE >= lg[-1], 1.0, t)), 0.0, 1.0)
    vi = lut.svalid[m, i] != DEAD; vi1 = lut.svalid[m, i + 1] != DEAD
    wi = np.where(vi & vi1, 1 - t, np.where(vi, 1.0, 0.0))
    wi1 = np.where(vi & vi1, t, np.where(vi, 0.0, np.where(vi1, 1.0, 0.0)))
    has = vi | vi1
    p3 = r3 * (nz - 1); j = np.clip(np.floor(p3).astype(np.int64), 0, nz - 2); w = p3 - j
    zz = lut.z_icdf.astype(np.float64)
    z = (1 - w) * (wi * zz[m, i, j] + wi1 * zz[m, i + 1, j]) + \
        w * (wi * zz[m, i, j + 1] + wi1 * zz[m, i + 1, j + 1])
    p4 = r4 * (nr - 1); k = np.clip(np.floor(p4).astype(np.int64), 0, nr - 2); w2 = p4 - k
    rr = lut.r_icdf.astype(np.float64)
    r = (1 - w2) * (wi * rr[m, i, j, k] + wi1 * rr[m, i + 1, j, k]) + \
        w2 * (wi * rr[m, i, j, k + 1] + wi1 * rr[m, i + 1, j, k + 1])
    s = float(lut.step_cm)
    z = np.where(has, np.clip(z, -s, s), 0.0); r = np.where(has, np.clip(r, 0.0, s), 0.0)
    return has, z, r


# ---------------------------------------------------------------------------
# Raw-text ingestion helper shared by compaction and validation
# ---------------------------------------------------------------------------
RAW_COLS = ("E0_keV", "theta_rad", "DE_total_eV", "dE_kin_eV", "E_final_eV", "annihil_flag", "matID")
RAW_COLS_STOP = ("z_cm", "r_cm")        # appended by drivers since the stopping tables (LUT v4)


@contextlib.contextmanager
def open_raw(path):
    """A raw file as text: matN.dat as is, matN.dat.zst through `zstd -dc` (zstd frames may be concatenated, e.g.
    one per energy chunk of a material; their '#' header lines are comments for the parser)."""
    path = str(path)
    if not path.endswith(".zst"):
        with open(path) as f:
            yield f
        return
    proc = subprocess.Popen(["zstd", "-dcq", "--", path], stdout=subprocess.PIPE)
    f = io.TextIOWrapper(proc.stdout)
    try:
        yield f
    finally:
        f.close(); proc.kill(); proc.wait()


def read_raw_config(path) -> dict:
    """The '# config: key=value ...' line the drivers write first ({} if absent,
    as in raw files written before it existed). Numbers are returned as floats."""
    out = {}
    with open_raw(path) as f:
        for _ in range(3):
            line = f.readline()
            if not line.startswith("#"): break
            if line.startswith("# config:"):
                for k, v in re.findall(r"(\w+)=\s*(\S+)", line):
                    try: out[k] = float(v)
                    except ValueError: out[k] = v
    return out


RAW_COL_DPHI = "dphi_rad"               # 10th column, since the step-end tables (LUT v5)


def _raw_ncols(path) -> int:
    """Number of columns of a raw file, from its column-name line or else its first data line."""
    with open_raw(path) as f:
        for _ in range(3):
            line = f.readline()
            if line.startswith("# E0_keV"):
                return len(line.split()) - 1
            if line and not line.startswith("#"):
                return len(line.split())
    return 0


def raw_has_stop(path) -> bool:
    """True when the raw file carries the end-position columns (z_cm r_cm)."""
    return _raw_ncols(path) >= len(RAW_COLS) + 2


def raw_has_dphi(path) -> bool:
    """True when the raw file carries the 10th column dphi_rad (step-end tables)."""
    return _raw_ncols(path) >= len(RAW_COLS) + 3


RAW_CHUNK_ROWS = 2_000_000


def _read_cols(path, usecols):
    """Uses pandas if available (≈10x faster than numpy.loadtxt on 850 MB files).
    Read in chunks of RAW_CHUNK_ROWS rows to bound the parser's peak memory."""
    try:
        import pandas as pd
    except ImportError:
        with open_raw(path) as f:
            return np.loadtxt(f, comments="#", usecols=usecols, ndmin=2)
    parts = []
    with open_raw(path) as f, pd.read_csv(f, sep=r"\s+", comment="#", header=None, usecols=list(usecols),
                                          dtype=np.float64, engine="c", chunksize=RAW_CHUNK_ROWS) as reader:
        for df in reader:
            parts.append(df[list(usecols)].to_numpy())
    if not parts:
        return np.zeros((0, len(usecols)))
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def load_raw(path, usecols=(0, 1, 3, 5)):
    """Stream a mat*.dat file -> (E0_eV float64, theta f32, dE_kin f32, ann int8)."""
    a = _read_cols(path, usecols)
    e0 = a[:, 0] * 1000.0
    return e0, a[:, 1].astype(F32), a[:, 2].astype(F32), a[:, 3].astype(np.int8)


def load_raw_full(path):
    """One pass over a mat*.dat file -> dict with E0_eV (f64), theta, dE_kin (f32),
    ann (int8) and, when the file has them, z and r (f32, cm) and dphi (f32, rad),
    else None."""
    ncol = _raw_ncols(path)
    stop = ncol >= len(RAW_COLS) + 2
    dphi = ncol >= len(RAW_COLS) + 3
    cols = (0, 1, 3, 5) + ((7, 8) if stop else ()) + ((9,) if dphi else ())
    a = _read_cols(path, cols)
    out = {"E0_eV": a[:, 0] * 1000.0, "theta": a[:, 1].astype(F32), "dE_kin": a[:, 2].astype(F32),
           "ann": a[:, 3].astype(np.int8), "z": None, "r": None, "dphi": None}
    if stop:
        out["z"], out["r"] = a[:, 4].astype(F32), a[:, 5].astype(F32)
    if dphi:
        out["dphi"] = a[:, 6].astype(F32)
    return out


# ---------------------------------------------------------------------------
# Step-end tables (LUT version 5/6, kind 3; luts/STEPEND_V5_SPEC.md)
#
# One surviving step's end state, after the azimuthal symmetry about the
# initial direction d0: end position (z along d0, r perpendicular), new
# direction (a1, a2; meaning set by `frame`) and energy loss u, all as
# conditional quantiles: z; r | z bin j; a1, a2, u | (j, r bin k).
# ---------------------------------------------------------------------------
def stepend_angles(z, r, theta, dphi, frame: int):
    """(a1, a2) of raw histories in the table's direction frame (arrays or scalars).

    frame 0 'theta': a1 = theta (polar angle from d0), a2 = |dphi|.
    frame 1 'chord': a1 = alpha = angle between the chord c = (r, 0, z)/rho and
    the new direction n = (sin th cos dphi, sin th sin dphi, cos th) (step frame,
    lateral displacement along +x); a2 = |psi|, psi = atan2(n_y, n.b1) with
    b1 = (z, 0, -r)/rho. rho = 0: alpha = theta, psi = 0."""
    z = np.asarray(z, dtype=np.float64); r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64); dphi = np.asarray(dphi, dtype=np.float64)
    if frame == 0:
        return theta, np.abs(dphi)
    if frame != 1:
        raise ValueError(f"unknown step-end frame {frame}")
    st = np.sin(theta)
    nx, ny, nz = st * np.cos(dphi), st * np.sin(dphi), np.cos(theta)
    rho = np.hypot(z, r)
    pos = rho > 0
    rs = np.where(pos, rho, 1.0)
    nc = (nx * r + nz * z) / rs
    nb1 = (nx * z - nz * r) / rs
    alpha = np.where(pos, np.arccos(np.clip(nc, -1.0, 1.0)), theta)
    psi = np.where(pos, np.arctan2(ny, nb1), 0.0)
    return alpha, np.abs(psi)


def _bin_members(b: np.ndarray, nb: int) -> list[np.ndarray]:
    """Index arrays of the samples in each bin 0..nb-1 of the assignment b.
    An empty bin borrows its nearest non-empty neighbours (both sides at the
    same distance; all samples if none), exactly as build_row does for u | theta."""
    order = np.argsort(b, kind="stable")
    parts = np.split(order, np.cumsum(np.bincount(b, minlength=nb))[:-1])
    out = []
    for bi in range(nb):
        if parts[bi].size:
            out.append(parts[bi])
            continue
        lo, hi = bi - 1, bi + 1
        while True:
            cand = []
            if lo >= 0 and parts[lo].size:
                cand.append(parts[lo])
            if hi <= nb - 1 and parts[hi].size:
                cand.append(parts[hi])
            if cand:
                out.append(np.concatenate(cand))
                break
            lo -= 1
            hi += 1
            if lo < 0 and hi > nb - 1:
                out.append(order)
                break
    return out


def _qbins(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Quantile-bin assignment of x by its raw knots (searchsorted, clipped), as in build_row."""
    return np.clip(np.searchsorted(knots, x, side="right") - 1, 0, knots.size - 2)


def knot_levels(n_a: int, knots: int) -> np.ndarray:
    """float32 levels (n_a,) of the a1 / a2 / u knots: j/(n_a-1) (uniform) or TAIL_LEVELS (tail, n_a = 16)."""
    if knots == KNOTS_UNIFORM:
        return np.linspace(0.0, 1.0, n_a).astype(F32)
    if knots == KNOTS_TAIL:
        if n_a != TAIL_LEVELS.size:
            raise ValueError(f"tail knots need N_a = {TAIL_LEVELS.size} (got {n_a})")
        return TAIL_LEVELS.astype(F32)
    raise ValueError(f"unknown knot scheme {knots}")


def build_stepend_row(z, r, theta, dphi, u, frame: int = DEFAULT_FRAME, n_z: int = DEFAULT_N_Z_SE,
                      n_r: int = DEFAULT_N_R_SE, n_a: int = DEFAULT_N_A, step_cm: float = 0.0,
                      geom: int = GEOM_PATH, knots: int = KNOTS_UNIFORM):
    """Step-end row from survivors' (z, r, theta, dphi, u) (one energy, pooled if needed).

    Returns float32 arrays
      z_icdf (n_z,), s2_icdf (n_z-1, n_r), a1_icdf, a2_icdf, u_icdf (n_z-1, n_r-1, n_a),
    every knot vector non-decreasing. Slot 2 (s2) is r | z ([0, step]), and a1, a2, u are
    conditioned on (z bin, r bin); geom must be GEOM_PATH. z and s2:
    quantiles + mean_preserving_ends; a1, a2, u: the same (knots uniform) or quantiles at
    TAIL_LEVELS with the exact sample min / max at the ends (knots tail). Clamped to z in
    [-step, step], a1, a2 in [0, pi], u in [0, U_MAX]."""
    if not step_cm > 0:
        raise ValueError(f"step_cm must be > 0 to build step-end rows (got {step_cm})")
    if geom not in GEOM_NAMES:
        raise ValueError(f"unsupported step-end geometry {geom} (only {GEOM_PATH} = path; the sphere layout "
                         f"was dropped on 2026-10-30)")
    z = np.asarray(z, dtype=np.float64); r = np.asarray(r, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    assert z.ndim == 1 and z.size > 0 and z.shape == r.shape == u.shape == np.shape(theta) == np.shape(dphi)
    assert n_z >= 3 and n_r >= 3 and n_a >= 3
    a1, a2 = stepend_angles(z, r, theta, dphi, frame)
    (z_lo, z_hi), (r_lo, r_hi) = stop_ranges(step_cm)
    s2 = r
    lev = knot_levels(n_a, knots).astype(np.float64)
    ranges = ((a1, 0.0, np.pi), (a2, 0.0, np.pi), (u, 0.0, U_MAX))
    zq = quantiles(z, n_z)
    r_icdf = np.empty((n_z - 1, n_r))
    c_icdf = np.empty((3, n_z - 1, n_r - 1, n_a))
    for j, idx in enumerate(_bin_members(_qbins(z, zq), n_z - 1)):
        rj = s2[idx]
        rq = quantiles(rj, n_r)
        for k, kk in enumerate(_bin_members(_qbins(rj, rq), n_r - 1)):
            sel = idx[kk]
            for c, (x, lo, hi) in enumerate(ranges):
                xs = x[sel]
                if knots == KNOTS_TAIL:
                    c_icdf[c, j, k] = np.quantile(xs, lev, method="linear")   # ends: exact min / max
                else:
                    c_icdf[c, j, k] = mean_preserving_ends(quantiles(xs, n_a), xs, lo, hi)
        # edge knots are adjusted only after binning, so bins follow the raw quantiles
        r_icdf[j] = mean_preserving_ends(rq, rj, r_lo, r_hi)
    zq = mean_preserving_ends(zq, z, z_lo, z_hi)
    zq = np.clip(np.maximum.accumulate(zq), z_lo, z_hi)
    r_icdf = np.clip(np.maximum.accumulate(r_icdf, axis=-1), r_lo, r_hi)
    c_icdf = np.maximum.accumulate(c_icdf, axis=-1)
    out = [np.clip(c_icdf[c], lo, hi).astype(F32) for c, (_, lo, hi) in enumerate(ranges)]
    return (zq.astype(F32), r_icdf.astype(F32), *out)


def classify_and_pool_stepend(rows: list[np.ndarray], n_full: int = N_FULL_SE,
                              k_max: int = DEFAULT_K_MAX):
    """rows[i]: survivors of energy row i as a (n_i, C) array (any C columns,
    e.g. z r theta dphi u). Same classification and neighbour pooling as
    classify_and_pool (applied to row tags), with n_full = N_FULL_SE.

    Returns (valid int32[N_E], pooled list of (n, C) arrays, None for DEAD)."""
    tags = [np.full(len(x), i, dtype=np.int64) for i, x in enumerate(rows)]
    valid, pooled_tags, _ = classify_and_pool(tags, tags, n_full=n_full, k_max=k_max)
    pooled = []
    for i, pt in enumerate(pooled_tags):
        if pt is None:
            pooled.append(None)
            continue
        used = list(dict.fromkeys(pt[np.r_[0, np.flatnonzero(np.diff(pt)) + 1]].tolist()))
        pooled.append(rows[i] if used == [i] else np.concatenate([rows[q] for q in used]))
    return valid, pooled


@dataclass
class StepEndLut:
    e_grid: np.ndarray                     # float32 (N_E,)            must equal ann_LUT's
    evalid: np.ndarray                     # int32   (N_M, N_E)        DEAD/FULL/POOLED/SPARSE
    z_icdf_s: np.ndarray                   # float32 (N_M, N_E, N_z)                  [cm]
    r_icdf_s: np.ndarray                   # float32 (N_M, N_E, N_z-1, N_r)           slot 2: r [cm]
    a1_icdf: np.ndarray                    # float32 (N_M, N_E, N_z-1, N_r-1, N_a)    [rad]
    a2_icdf: np.ndarray                    # float32 (N_M, N_E, N_z-1, N_r-1, N_a)    [rad]
    u_icdf_s: np.ndarray                   # float32 (N_M, N_E, N_z-1, N_r-1, N_a)
    frame: int = DEFAULT_FRAME             # 0 theta, 1 chord
    step_cm: float = 0.0
    n_full_se: int = N_FULL_SE
    geom: int = GEOM_PATH                  # v6: 0 path (slot 2 = r | z), the only valid value
    knots: int = KNOTS_UNIFORM             # v6: a1 / a2 / u knot scheme (v5 files: uniform)
    version: int = VERSION_STEPEND         # 5 (read from an old file, or written for tests) or 6

    @property
    def n_m(self): return self.evalid.shape[0]
    @property
    def n_e(self): return self.e_grid.shape[0]
    @property
    def n_z(self): return self.z_icdf_s.shape[2]
    @property
    def n_r(self): return self.r_icdf_s.shape[3]
    @property
    def n_a(self): return self.a1_icdf.shape[4]
    @property
    def levels(self): return knot_levels(self.n_a, self.knots)

    def check(self):
        n_m, n_e, nz, nr, na = self.n_m, self.n_e, self.n_z, self.n_r, self.n_a
        assert self.frame in FRAME_NAMES, f"bad frame {self.frame}"
        assert self.geom in GEOM_NAMES, f"bad geom {self.geom}"
        assert self.knots in KNOT_NAMES, f"bad knot scheme {self.knots}"
        assert self.version in READABLE_SE_VERSIONS
        if self.version == 5:
            assert self.geom == GEOM_PATH and self.knots == KNOTS_UNIFORM, "v5 has no geom / knot scheme"
        assert nz >= 3 and nr >= 3 and na >= 3
        self.levels                                                     # tail: N_a must match
        assert self.evalid.shape == (n_m, n_e)
        assert self.z_icdf_s.shape == (n_m, n_e, nz)
        assert self.r_icdf_s.shape == (n_m, n_e, nz - 1, nr)
        for a in (self.a1_icdf, self.a2_icdf, self.u_icdf_s):
            assert a.shape == (n_m, n_e, nz - 1, nr - 1, na)
        assert np.all(np.diff(self.e_grid) > 0), "energy grid not increasing"
        live = self.evalid != DEAD
        s = np.float32(self.step_cm)
        for name, a, lo, hi in (("z", self.z_icdf_s, -s, s), ("slot-2", self.r_icdf_s, 0, s),
                                ("a1", self.a1_icdf, 0, np.float32(np.pi)),
                                ("a2", self.a2_icdf, 0, np.float32(np.pi)), ("u", self.u_icdf_s, 0, U_MAX)):
            assert not np.isnan(a).any(), f"{name} ICDF has NaN"
            assert np.all(np.diff(a[live], axis=-1) >= 0), f"{name} ICDF not monotone"
            assert np.all(a[~live] == 0), f"{name} ICDF of a DEAD row not zero"
            assert np.all((a >= lo) & (a <= hi)), f"{name} ICDF outside its range"
        return True


def _se_hdr(lut: StepEndLut) -> bytes:
    """v5, 48 bytes: magic version=5 kind=3 N_E N_M N_z N_r N_a frame step_cm(f32) n_full_se reserved=0.
    v6, 56 bytes: the same up to n_full_se, then geom knots reserved=0."""
    if not lut.step_cm > 0:
        raise ValueError(f"step_cm must be > 0 to write a LUT (got {lut.step_cm})")
    head = (MAGIC + struct.pack("<8i", lut.version, KIND_STEPEND, lut.n_e, lut.n_m, lut.n_z,
                                lut.n_r, lut.n_a, int(lut.frame))
            + struct.pack("<fi", lut.step_cm, int(lut.n_full_se)))
    if lut.version == 5:
        if lut.geom != GEOM_PATH or lut.knots != KNOTS_UNIFORM:
            raise ValueError("a v5 file has no geom / knot scheme: write version 6")
        return head + struct.pack("<i", 0)
    return head + struct.pack("<3i", int(lut.geom), int(lut.knots), 0)


def write_stepend_lut(path, lut: StepEndLut):
    """v6: header, e_grid f32[N_E], levels f32[N_a], evalid, z, slot 2, a1, a2, u (v5: no levels)."""
    with open(path, "wb") as f:
        f.write(_se_hdr(lut))
        f.write(lut.e_grid.astype(F32).tobytes())
        if lut.version >= 6:
            f.write(lut.levels.astype(F32).tobytes())
        f.write(lut.evalid.astype(I32).tobytes())
        for a in (lut.z_icdf_s, lut.r_icdf_s, lut.a1_icdf, lut.a2_icdf, lut.u_icdf_s):
            f.write(np.ascontiguousarray(a, dtype=F32).tobytes())


def read_stepend_lut(path) -> StepEndLut:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"bad magic {magic!r}; not a LUT file")
        version, kind, n_e, n_m, nz, nr, na, frame = struct.unpack("<8i", f.read(32))
        if version not in READABLE_SE_VERSIONS or kind != KIND_STEPEND:
            raise ValueError(f"version/kind mismatch: {version}/{kind} "
                             f"(expected {READABLE_SE_VERSIONS}/{KIND_STEPEND})")
        step, n_full_se = struct.unpack("<fi", f.read(8))
        geom, knots = GEOM_PATH, KNOTS_UNIFORM
        if version >= 6:
            geom, knots = struct.unpack("<2i", f.read(8))
        reserved, = struct.unpack("<i", f.read(4))
        if geom == GEOM_SPHERE_DROPPED:
            raise ValueError(f"{path}: geom 1 (sphere first-passage tables) is no longer supported "
                             f"(dropped 2026-10-30); rebuild from fixed-path raw files")
        if nz < 3 or nr < 3 or na < 3 or frame not in FRAME_NAMES or reserved != 0 \
                or geom not in GEOM_NAMES or knots not in KNOT_NAMES:
            raise ValueError(f"bad step-end header N_z={nz} N_r={nr} N_a={na} frame={frame} "
                             f"geom={geom} knots={knots} reserved={reserved}")
        eg = _arr(f, F32, (n_e,))
        if version >= 6:
            lev = _arr(f, F32, (na,))
            if not np.array_equal(lev, knot_levels(na, knots)):
                raise ValueError(f"knot levels in the file do not match the scheme {KNOT_NAMES[knots]}")
        ev = _arr(f, I32, (n_m, n_e))
        zz = _arr(f, F32, (n_m, n_e, nz))
        rr = _arr(f, F32, (n_m, n_e, nz - 1, nr))
        a1, a2, uu = (_arr(f, F32, (n_m, n_e, nz - 1, nr - 1, na)) for _ in range(3))
        if f.read(1):
            raise ValueError("trailing bytes after the step-end tables")
    return StepEndLut(eg, ev, zz, rr, a1, a2, uu, frame, float(step), n_full_se, geom, knots, version)


def _se_meta(m: StepEndLut):
    return (m.n_z, m.n_r, m.n_a, m.frame, m.n_full_se, m.geom, m.knots, m.version)


def combine_stepend(materials: list[StepEndLut]) -> StepEndLut:
    m0 = materials[0]
    for m in materials:
        if not np.array_equal(m.e_grid, m0.e_grid):
            raise ValueError("energy grid mismatch across materials")
        if _se_meta(m) != _se_meta(m0):
            raise ValueError("step-end dimension / frame / n_full_se / geom / knots / version mismatch "
                             "across materials")
        if np.float32(m.step_cm) != np.float32(m0.step_cm):          # stored as float32
            raise ValueError("step_cm mismatch across materials")
    cat = lambda attr: np.concatenate([getattr(m, attr) for m in materials], axis=0)
    return StepEndLut(m0.e_grid, cat("evalid"), cat("z_icdf_s"), cat("r_icdf_s"), cat("a1_icdf"),
                      cat("a2_icdf"), cat("u_icdf_s"), m0.frame, m0.step_cm, m0.n_full_se,
                      m0.geom, m0.knots, m0.version)


def select_stepend(lut: StepEndLut, m: int) -> StepEndLut:
    s = slice(m, m + 1)
    return StepEndLut(lut.e_grid, lut.evalid[s], lut.z_icdf_s[s], lut.r_icdf_s[s], lut.a1_icdf[s],
                      lut.a2_icdf[s], lut.u_icdf_s[s], lut.frame, lut.step_cm, lut.n_full_se,
                      lut.geom, lut.knots, lut.version)


def knot_position(rn, levels: np.ndarray, knots: int, n: int):
    """Knot interval l and weight wl of a uniform rn on the knot axis (scalar or array):
    uniform: p = rn (n-1), l = clamp(floor p, 0, n-2), wl = p - l;
    tail: l = clamp(searchsorted(levels, rn, right) - 1, 0, n-2), wl = (rn - lev_l) / (lev_{l+1} - lev_l)."""
    if knots == KNOTS_UNIFORM:
        p = np.asarray(rn, dtype=np.float64) * (n - 1)
        l = np.clip(np.floor(p).astype(np.int64), 0, n - 2)
        return l, p - l
    lev = np.asarray(levels, dtype=np.float64)
    rn = np.asarray(rn, dtype=np.float64)
    l = np.clip(np.searchsorted(lev, rn, side="right") - 1, 0, n - 2)
    return l, (rn - lev[l]) / (lev[l + 1] - lev[l])


def sample_stepend(lut: StepEndLut, m: int, E: float, r1: float, r2: float, r3: float,
                   r4: float, r5: float, r6: float):
    """One surviving step's end state (spec "Sampling contract", steps 1-5 and the sign of 7).

    Returns (has, z, r, a1, a2, u, sign): z, r [cm] in the table's (nominal
    density) frame, unscaled; a1, a2 [rad]; u in [0, 1); sign = +1 if r6 < 0.5
    else -1. has = False when neither bracketing row is live (the engine falls
    back to the hinge with the scatt-eloss tables; zeros are returned)."""
    i, t = bracket(lut.e_grid, E)
    s = 1.0 if r6 < 0.5 else -1.0
    ok, wi, wi1 = _row_weights(lut.evalid, m, i, t)
    if not ok:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, s
    nz, nr, na = lut.n_z, lut.n_r, lut.n_a
    p1 = r1 * (nz - 1)
    j = min(max(int(np.floor(p1)), 0), nz - 2)
    w = p1 - j
    zz = lut.z_icdf_s
    z_lo = wi * float(zz[m, i, j]) + wi1 * float(zz[m, i + 1, j])
    z_hi = wi * float(zz[m, i, j + 1]) + wi1 * float(zz[m, i + 1, j + 1])
    z = (1.0 - w) * z_lo + w * z_hi
    p2 = r2 * (nr - 1)
    k = min(max(int(np.floor(p2)), 0), nr - 2)
    w2 = p2 - k
    rr = lut.r_icdf_s
    r_lo = wi * float(rr[m, i, j, k]) + wi1 * float(rr[m, i + 1, j, k])
    r_hi = wi * float(rr[m, i, j, k + 1]) + wi1 * float(rr[m, i + 1, j, k + 1])
    r = (1.0 - w2) * r_lo + w2 * r_hi
    lev = lut.levels
    vals = []
    for tab, rn in ((lut.a1_icdf, r3), (lut.a2_icdf, r4), (lut.u_icdf_s, r5)):
        l, wl = knot_position(rn, lev, lut.knots, na)
        l, wl = int(l), float(wl)
        lo = wi * float(tab[m, i, j, k, l]) + wi1 * float(tab[m, i + 1, j, k, l])
        hi = wi * float(tab[m, i, j, k, l + 1]) + wi1 * float(tab[m, i + 1, j, k, l + 1])
        vals.append((1.0 - wl) * lo + wl * hi)
    a1, a2, u = vals
    st = float(lut.step_cm)
    z = min(max(z, -st), st)
    r = min(max(r, 0.0), st)
    a1 = min(max(a1, 0.0), np.pi)
    a2 = min(max(a2, 0.0), np.pi)
    u = min(max(u, 0.0), np.nextafter(1.0, 0.0))
    return True, z, r, a1, a2, u, s


def sample_stepend_many(lut: StepEndLut, m, E, r1, r2, r3, r4, r5, r6):
    """Vectorised sample_stepend (identical arithmetic). Returns arrays
    (has, z, r, a1, a2, u, sign); rows with has = False hold zeros."""
    m = np.asarray(m, dtype=np.int64); E = np.asarray(E, dtype=np.float64)
    r1, r2, r3, r4, r5, r6 = (np.asarray(x, dtype=np.float64) for x in (r1, r2, r3, r4, r5, r6))
    n_e, nz, nr, na = lut.n_e, lut.n_z, lut.n_r, lut.n_a
    lg = np.log(lut.e_grid.astype(np.float64)); lE = np.log(E)
    i = np.clip(np.searchsorted(lg, lE, side="right") - 1, 0, n_e - 2)
    t = (lE - lg[i]) / (lg[i + 1] - lg[i])
    t = np.clip(np.where(lE <= lg[0], 0.0, np.where(lE >= lg[-1], 1.0, t)), 0.0, 1.0)
    vi = lut.evalid[m, i] != DEAD; vi1 = lut.evalid[m, i + 1] != DEAD
    wi = np.where(vi & vi1, 1 - t, np.where(vi, 1.0, 0.0))
    wi1 = np.where(vi & vi1, t, np.where(vi, 0.0, np.where(vi1, 1.0, 0.0)))
    has = vi | vi1
    p1 = r1 * (nz - 1); j = np.clip(np.floor(p1).astype(np.int64), 0, nz - 2); w = p1 - j
    zz = lut.z_icdf_s
    z = (1 - w) * (wi * zz[m, i, j] + wi1 * zz[m, i + 1, j]) + \
        w * (wi * zz[m, i, j + 1] + wi1 * zz[m, i + 1, j + 1])
    p2 = r2 * (nr - 1); k = np.clip(np.floor(p2).astype(np.int64), 0, nr - 2); w2 = p2 - k
    rr = lut.r_icdf_s
    r = (1 - w2) * (wi * rr[m, i, j, k] + wi1 * rr[m, i + 1, j, k]) + \
        w2 * (wi * rr[m, i, j, k + 1] + wi1 * rr[m, i + 1, j, k + 1])
    lev = lut.levels
    vals = []
    for tab, rn in ((lut.a1_icdf, r3), (lut.a2_icdf, r4), (lut.u_icdf_s, r5)):
        l, wl = knot_position(rn, lev, lut.knots, na)
        lo = wi * tab[m, i, j, k, l] + wi1 * tab[m, i + 1, j, k, l]
        hi = wi * tab[m, i, j, k, l + 1] + wi1 * tab[m, i + 1, j, k, l + 1]
        vals.append((1 - wl) * lo + wl * hi)
    a1, a2, u = vals
    st = float(lut.step_cm)
    z = np.clip(z, -st, st)
    z = np.where(has, z, 0.0)
    r = np.where(has, np.clip(r, 0.0, st), 0.0)
    a1 = np.where(has, np.clip(a1, 0.0, np.pi), 0.0)
    a2 = np.where(has, np.clip(a2, 0.0, np.pi), 0.0)
    u = np.where(has, np.clip(u, 0.0, np.nextafter(1.0, 0.0)), 0.0)
    s = np.where(r6 < 0.5, 1.0, -1.0)
    return has, z, r, a1, a2, u, s


def stepend_geometry(z, r, a1, a2, sign, frame: int, phi=0.0, scale=1.0):
    """A step-end sample -> (displacement, new direction) in the d0 frame
    (orthonormal x, y, d0 = +z; the lateral unit vector is e(phi) = (cos phi, sin phi, 0)).

    displacement = scale * (z d0 + r e(phi))   (scale = sv/step, spec step 6)
    frame 0: n = cos a1 d0 + sin a1 (cos(phi + s a2), sin(phi + s a2), 0)
    frame 1: c = (r e + z d0)/rho, b1 = (z e - r d0)/rho, b2 = c x b1 = d0 x e,
             n = cos a1 c + sin a1 (cos(s a2) b1 + sin(s a2) b2);
             rho = 0: c = d0, b1 = e (the frame-0 direction at azimuth phi + s a2).
    Works on scalars or broadcastable arrays; returns arrays of shape (..., 3)."""
    z, r, a1, a2, sign, phi = (np.asarray(x, dtype=np.float64) for x in (z, r, a1, a2, sign, phi))
    cp, sp = np.cos(phi), np.sin(phi)
    zero = np.zeros(np.broadcast(z, r, a1, a2, sign, phi).shape)
    e = np.stack([cp + zero, sp + zero, zero], axis=-1)
    d0 = np.stack([zero, zero, zero + 1.0], axis=-1)
    disp = scale * (z[..., None] * d0 + r[..., None] * e)
    if frame == 0:
        az = phi + sign * a2
        n = np.stack([np.sin(a1) * np.cos(az), np.sin(a1) * np.sin(az), np.cos(a1) + zero], axis=-1)
        return disp, n
    if frame != 1:
        raise ValueError(f"unknown step-end frame {frame}")
    rho = np.hypot(z, r)
    pos = (rho > 0)[..., None]
    rs = np.where(rho > 0, rho, 1.0)[..., None]
    c = np.where(pos, (r[..., None] * e + z[..., None] * d0) / rs, d0)
    b1 = np.where(pos, (z[..., None] * e - r[..., None] * d0) / rs, e)
    b2 = np.cross(d0, e)
    ang = (sign * a2)[..., None]
    n = np.cos(a1)[..., None] * c + np.sin(a1)[..., None] * (np.cos(ang) * b1 + np.sin(ang) * b2)
    return disp, n
