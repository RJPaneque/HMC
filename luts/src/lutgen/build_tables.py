#!/usr/bin/env python3
"""
build_tables.py -- compaction: raw simulation histories -> conditional-quantile
step tables for the HMC engine.

Reads the raw text files written by a simulation driver (PENELOPE or GEANT4)
and builds, per material, the LUT payload described in the "LUT format"
section of the README and implemented normatively in ``lut_contract.py``
(this directory; imported, never re-implemented here):

  * p_ann[i]            annihilation probability per grid energy
  * n_hist[i], n_surv[i] histories simulated / surviving for this row
  * valid[i]            0 DEAD, 1 FULL, 2 POOLED, 3 SPARSE   (spec §5)
  * theta_icdf[i, j]    N_theta quantiles of the polar deflection
  * u_icdf[i, b, k]     N_u quantiles of u = dE_kin/E0 conditional on the
                        theta-quantile bin b                     (spec §3)
  * svalid[i], z_icdf[i, j], r_icdf[i, b, k]
                        the same construction for the STOPPED histories'
                        end position (z along the initial direction, r
                        perpendicular), when the raw files carry it (§3bis)
  * rho_nom             nominal density of the material [g/cm3], from the raw
                        config line or the .mat file (0 if neither is known)

  * stepend_LUT.raw    (step-end LUT version 6, luts/STEPEND_V5_SPEC.md) the
                        surviving step's end state as conditional quantiles
                        z; r | z; a1, a2, u | (z, r) in direction frame 0
                        (theta, |dphi|) or 1 (chord: alpha, |psi|), rows FULL
                        with N_FULL_SE survivors -- built only when EVERY raw
                        file carries the 10th column dphi_rad. The a1 / a2 /
                        u knots are tail-dense (--knots tail, default) or
                        uniform (v5's construction). Raw files from a fixed
                        path length only (geom path in the header); raw with
                        geom=sphere in its config line (the first-passage
                        mode, dropped 2026-10-30) is refused

Every surviving history contributes to the quantiles. Rows with fewer than
N_FULL survivors borrow survivors from the same material's neighbouring
energies (k = 1 .. K_MAX rows each side) and are flagged POOLED, or SPARSE
if even that does not reach N_FULL. Rows without survivors are DEAD and hold
zeros; live rows never contain zero padding. Stopping-position rows are
classified the same way with N_FULL_STOP stopped histories.

Raw text format (engine-agnostic, one row per history, one header line);
z_cm r_cm are written by drivers since the stopping tables and dphi_rad since
the step-end tables; both are optional (older raw files build LUTs without them):

    E0_keV theta_rad DE_total_eV dE_kin_eV E_final_eV annihil_flag matID z_cm r_cm dphi_rad

Outputs (all little-endian, layouts in the README "LUT format" §6 / lut_contract.py):

    <out-dir>/ann_LUT.raw                 p_ann, n_hist, n_surv, valid
    <out-dir>/scatt-eloss_LUT.raw         theta_icdf, u_icdf, valid
    <out-dir>/stepend_LUT.raw             step-end tables (v6), when the raw has dphi_rad
    <out-dir>/gap_census.csv              one row per (material, energy)
    <out-dir>/stepend_census.csv          step-end row codes, when built
    <out-dir>/run.json                    engine, raw_dir, build parameters, commit

Raw files are named mat1.dat .. mat30.dat (unpadded); materials are always
processed and concatenated in NUMERIC order (mat2 before mat10).

Usage:
    pixi run luts-build --engine penelope     (luts/build_luts.py -> this file)
    python build_luts.py --engine geant4 --raw-dir /data/g4/raw --out-dir ./g4 \
                           --n-theta 128 --n-u 64 --n-full 2000 --k-max 2
    python build_luts.py --raw-dir /data/pen/raw --out-dir /data/pen/m12_30 \
                           --materials 12,30 --frame 1      (a subset: slots in numeric order)
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import re
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# make the sibling reference module importable whether we run as a script,
# or via the tests' conftest (which puts src/lutgen/ on sys.path)
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import lut_contract  # noqa: E402
from lut_contract import (  # noqa: E402
    DEAD, FULL, POOLED, SPARSE, VALID_NAMES, F32, I32, LutV2,
    DEFAULT_N_THETA, DEFAULT_N_U, DEFAULT_N_FULL, DEFAULT_K_MAX,
    DEFAULT_N_Z, DEFAULT_N_R, DEFAULT_N_FULL_STOP,
    StepEndLut, DEFAULT_N_Z_SE, DEFAULT_N_R_SE, DEFAULT_N_A, N_FULL_SE, DEFAULT_FRAME, FRAME_NAMES,
    GEOM_PATH, GEOM_NAMES, KNOT_NAMES, DEFAULT_KNOTS, VERSION_STEPEND,
)

ROOT = _HERE.parents[1]      # luts/
ENGINES = ("penelope", "geant4", "egsnrc", "fluka")
PENELOPE_VERSION = "PenSuite-2023"   # sim_penelope/COMPILAR.sh; the PENELOPE driver writes no engine_version

CENSUS_COLUMNS = ("engine", "material", "energy_index", "E_keV", "n_hist",
                  "n_surv", "p_ann", "valid_code", "valid_name", "n_pooled",
                  "degenerate_theta_bins", "n_stop", "stop_code", "stop_name",
                  "n_stop_pooled")
SE_CENSUS_COLUMNS = ("engine", "material", "energy_index", "E_keV", "n_surv", "se_code",
                     "se_name", "n_se_pooled")

_MAT_RE = re.compile(r"mat(\d+)\.(dat|raw)(\.zst)?$")


# ---------------------------------------------------------------------------
# Paths and material ordering
# ---------------------------------------------------------------------------
def get_engine_dirs(engine: str, raw_dir=None, out_dir=None) -> tuple[Path, Path]:
    """Return (RAW_DIR, OUT_DIR).

    Defaults to ``luts/output/<engine>/raw`` and ``luts/output/<engine>``;
    ``--raw-dir`` and ``--out-dir`` override them.
    """
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine: {engine}")
    base = ROOT / "output" / engine
    raw = Path(raw_dir) if raw_dir is not None else base / "raw"
    out = Path(out_dir) if out_dir is not None else base
    return raw, out


def material_id(path) -> int:
    """Numeric material id from ``mat7.dat``."""
    m = _MAT_RE.search(Path(path).name)
    if m is None:
        raise ValueError(f"cannot parse material id from {path}")
    return int(m.group(1))


def sorted_by_material(paths) -> list[Path]:
    """Sort paths by NUMERIC material id (mat2 before mat10)."""
    paths = [Path(p) for p in paths]
    ids = [material_id(p) for p in paths]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate material ids in {[p.name for p in paths]}")
    return [p for _, p in sorted(zip(ids, paths))]


def find_raw_files(raw_dir: Path) -> list[Path]:
    """matN.dat or matN.dat.zst (zstd, read through `zstd -dc`); both for one material is a duplicate."""
    return sorted_by_material([*Path(raw_dir).glob("mat*.dat"), *Path(raw_dir).glob("mat*.dat.zst")])


# ---------------------------------------------------------------------------
# Per-energy grouping
# ---------------------------------------------------------------------------
def group_by_energy(e0: np.ndarray, theta: np.ndarray, de_kin: np.ndarray,
                    ann: np.ndarray):
    """Group histories by exact E0 value.

    Returns
    -------
    e_grid    : float64 (N_E,) increasing grid energies [eV]
    n_hist    : int32   (N_E,)
    n_surv    : int32   (N_E,)
    p_ann     : float64 (N_E,)  = (n_hist - n_surv) / n_hist
    surv_theta: list of float64 arrays (survivors' theta per row)
    surv_u    : list of float64 arrays (survivors' u = dE_kin / E0 per row)
    """
    e0 = np.asarray(e0, dtype=np.float64)
    e_grid, inverse, n_hist = np.unique(e0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    n_e = e_grid.size
    surv = np.asarray(ann) == 0
    n_surv = np.bincount(inverse[surv], minlength=n_e)
    p_ann = (n_hist - n_surv) / n_hist
    # split survivors into per-row arrays with one stable sort
    idx = np.flatnonzero(surv)
    order = np.argsort(inverse[idx], kind="stable")
    idx = idx[order]
    bounds = np.cumsum(n_surv)[:-1]
    th_parts = np.split(np.asarray(theta, dtype=np.float64)[idx], bounds)
    u_all = np.asarray(de_kin, dtype=np.float64)[idx] / e0[idx]
    u_parts = np.split(u_all, bounds)
    return (e_grid, n_hist.astype(I32), n_surv.astype(I32), p_ann,
            list(th_parts), list(u_parts))


def group_stops(e0: np.ndarray, e_grid: np.ndarray, z: np.ndarray, r: np.ndarray,
                ann: np.ndarray):
    """Stopped histories' end positions per energy row (same grid as
    group_by_energy): lists of float64 arrays (z per row, r per row) [cm]."""
    e0 = np.asarray(e0, dtype=np.float64)
    inverse = np.searchsorted(e_grid, e0)
    assert np.array_equal(e_grid[inverse], e0)
    stop = np.asarray(ann) != 0
    n_e = e_grid.size
    idx = np.flatnonzero(stop)
    order = np.argsort(inverse[idx], kind="stable")
    idx = idx[order]
    n_stop = np.bincount(inverse[stop], minlength=n_e)
    bounds = np.cumsum(n_stop)[:-1]
    z_parts = np.split(np.asarray(z, dtype=np.float64)[idx], bounds)
    r_parts = np.split(np.asarray(r, dtype=np.float64)[idx], bounds)
    return list(z_parts), list(r_parts)


def group_stepend(e0, e_grid, ann, theta, de_kin, z, r, dphi) -> list[np.ndarray]:
    """Survivors' step-end state per energy row (same grid as group_by_energy):
    list of float64 (n_surv_i, 5) arrays with columns z r theta dphi u (u = dE_kin/E0)."""
    e0 = np.asarray(e0, dtype=np.float64)
    inverse = np.searchsorted(e_grid, e0)
    assert np.array_equal(e_grid[inverse], e0)
    surv = np.asarray(ann) == 0
    idx = np.flatnonzero(surv)
    idx = idx[np.argsort(inverse[idx], kind="stable")]
    bounds = np.cumsum(np.bincount(inverse[surv], minlength=e_grid.size))[:-1]
    cols = np.empty((idx.size, 5))
    cols[:, 0] = z[idx]; cols[:, 1] = r[idx]; cols[:, 2] = theta[idx]; cols[:, 3] = dphi[idx]
    cols[:, 4] = np.asarray(de_kin, dtype=np.float64)[idx] / e0[idx]
    return list(np.split(cols, bounds))


def build_stepend_material(e_grid, surv_rows, step_cm, frame=DEFAULT_FRAME, n_z=DEFAULT_N_Z_SE,
                           n_r=DEFAULT_N_R_SE, n_a=DEFAULT_N_A, n_full_se=N_FULL_SE,
                           k_max=DEFAULT_K_MAX, geom=GEOM_PATH, knots=DEFAULT_KNOTS):
    """One-material StepEndLut (N_M = 1) from group_stepend's per-row survivors.
    Returns (lut, info) with info[i] = {"se_code", "n_se_pooled"}."""
    n_e = len(e_grid)
    assert len(surv_rows) == n_e
    valid, pooled = lut_contract.classify_and_pool_stepend(surv_rows, n_full=n_full_se, k_max=k_max)
    zz = np.zeros((1, n_e, n_z), dtype=F32)
    rr = np.zeros((1, n_e, n_z - 1, n_r), dtype=F32)
    aa = [np.zeros((1, n_e, n_z - 1, n_r - 1, n_a), dtype=F32) for _ in range(3)]
    info = []
    for i in range(n_e):
        if valid[i] == DEAD:
            info.append({"se_code": DEAD, "n_se_pooled": 0})
            continue
        x = pooled[i]
        row = lut_contract.build_stepend_row(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], frame,
                                             n_z, n_r, n_a, step_cm, geom=geom, knots=knots)
        zz[0, i], rr[0, i], aa[0][0, i], aa[1][0, i], aa[2][0, i] = row
        info.append({"se_code": int(valid[i]), "n_se_pooled": int(x.shape[0])})
    lut = StepEndLut(np.asarray(e_grid, dtype=F32), np.asarray(valid, dtype=I32).reshape(1, n_e),
                     zz, rr, aa[0], aa[1], aa[2], int(frame), float(step_cm), int(n_full_se),
                     int(geom), int(knots), VERSION_STEPEND)
    lut.check()
    return lut, info


def count_degenerate_theta_bins(theta: np.ndarray, n_theta: int) -> int:
    """Number of empty theta-quantile bins for this sample.

    Mirrors the bin assignment inside ``lut_contract.build_row`` (diagnostic only;
    a non-zero count means build_row used its nearest-neighbour fallback for
    that many conditional ICDFs).
    """
    theta = np.asarray(theta, dtype=np.float64)
    th_icdf = lut_contract.quantiles(theta, n_theta)
    b = np.searchsorted(th_icdf, theta, side="right") - 1
    b = np.clip(b, 0, n_theta - 2)
    return int(np.count_nonzero(np.bincount(b, minlength=n_theta - 1) == 0))


# ---------------------------------------------------------------------------
# Material build
# ---------------------------------------------------------------------------
def build_material(e_grid, n_hist, n_surv, p_ann, surv_theta, surv_u,
                   n_theta=DEFAULT_N_THETA, n_u=DEFAULT_N_U,
                   n_full=DEFAULT_N_FULL, k_max=DEFAULT_K_MAX, step_cm=0.0,
                   stop_z=None, stop_r=None, n_z=DEFAULT_N_Z, n_r=DEFAULT_N_R,
                   n_full_stop=DEFAULT_N_FULL_STOP, rho_nom=0.0):
    """Build a one-material LutV2 (N_M = 1) plus per-row diagnostics.

    stop_z / stop_r: per-row arrays of the stopped histories' end positions
    (from group_stops); None builds no stopping-position tables.

    Returns (lut, info) where info is a list of dicts with keys
    ``n_pooled`` (survivors fed to build_row; 0 for DEAD),
    ``degenerate_theta_bins``, ``n_stop``, ``stop_code`` and ``n_stop_pooled``.
    """
    n_e = len(e_grid)
    assert len(surv_theta) == len(surv_u) == n_e
    has_stop = stop_z is not None
    if has_stop and not step_cm > 0:
        warnings.warn("no step_cm: the stopping-position tables need it for their range and "
                      "are not built", RuntimeWarning)
        has_stop = False
    if has_stop:
        assert len(stop_z) == len(stop_r) == n_e
        assert n_z >= 3 and n_r >= 3
    valid, pooled_t, pooled_u = lut_contract.classify_and_pool(surv_theta, surv_u,
                                                         n_full=n_full, k_max=k_max)
    theta_icdf = np.zeros((1, n_e, n_theta), dtype=F32)
    u_icdf = np.zeros((1, n_e, n_theta - 1, n_u), dtype=F32)
    info = []
    for i in range(n_e):
        if valid[i] == DEAD:
            assert n_surv[i] == 0
            info.append({"n_pooled": 0, "degenerate_theta_bins": 0})
            continue
        t, u = pooled_t[i], pooled_u[i]
        th_row, u_row = lut_contract.build_row(t, u, n_theta, n_u)
        theta_icdf[0, i] = th_row
        u_icdf[0, i] = u_row
        info.append({"n_pooled": int(t.size),
                     "degenerate_theta_bins": count_degenerate_theta_bins(t, n_theta)})
        # live rows must never look like padding
        if not (np.any(th_row != 0) and np.all(np.any(u_row != 0, axis=1))):
            warnings.warn(f"live row i={i} (valid={VALID_NAMES[int(valid[i])]}) "
                          f"has an all-zero ICDF slice", RuntimeWarning)
    # stopping positions: same classification and row construction on (z, r)
    svalid = np.zeros(n_e, dtype=I32)
    z_icdf = np.zeros((1, n_e, n_z if has_stop else 0), dtype=F32)
    r_icdf = np.zeros((1, n_e, n_z - 1 if has_stop else 0, n_r if has_stop else 0), dtype=F32)
    for i in range(n_e):
        info[i].update({"n_stop": int(stop_z[i].size) if has_stop else 0,
                        "stop_code": DEAD, "n_stop_pooled": 0})
    if has_stop:
        z_range, r_range = lut_contract.stop_ranges(step_cm)
        svalid, pooled_z, pooled_r = lut_contract.classify_and_pool(
            stop_z, stop_r, n_full=n_full_stop, k_max=k_max)
        for i in range(n_e):
            if svalid[i] == DEAD:
                continue
            z, r = pooled_z[i], pooled_r[i]
            z_icdf[0, i], r_icdf[0, i] = lut_contract.build_row(z, r, n_z, n_r, z_range, r_range)
            info[i].update({"stop_code": int(svalid[i]), "n_stop_pooled": int(z.size)})
    lut = LutV2(
        e_grid=np.asarray(e_grid, dtype=F32),
        p_ann=np.asarray(p_ann, dtype=F32).reshape(1, n_e),
        n_hist=np.asarray(n_hist, dtype=I32).reshape(1, n_e),
        n_surv=np.asarray(n_surv, dtype=I32).reshape(1, n_e),
        valid=np.asarray(valid, dtype=I32).reshape(1, n_e),
        theta_icdf=theta_icdf,
        u_icdf=u_icdf,
        step_cm=float(step_cm),
        rho_nom=np.array([rho_nom], dtype=F32),
        svalid=np.asarray(svalid, dtype=I32).reshape(1, n_e),
        z_icdf=z_icdf,
        r_icdf=r_icdf,
    )
    lut.check()
    return lut, info


def census_rows(engine, mat, lut: LutV2, info) -> list[dict]:
    rows = []
    for i in range(lut.n_e):
        code = int(lut.valid[0, i])
        rows.append({
            "engine": engine, "material": int(mat), "energy_index": i,
            "E_keV": float(lut.e_grid[i]) / 1000.0,
            "n_hist": int(lut.n_hist[0, i]), "n_surv": int(lut.n_surv[0, i]),
            "p_ann": float(lut.p_ann[0, i]), "valid_code": code,
            "valid_name": VALID_NAMES[code], "n_pooled": int(info[i]["n_pooled"]),
            "degenerate_theta_bins": int(info[i]["degenerate_theta_bins"]),
            "n_stop": int(info[i]["n_stop"]), "stop_code": int(info[i]["stop_code"]),
            "stop_name": VALID_NAMES[int(info[i]["stop_code"])],
            "n_stop_pooled": int(info[i]["n_stop_pooled"]),
        })
    return rows


def process_material(raw_path, n_theta=DEFAULT_N_THETA, n_u=DEFAULT_N_U,
                     n_full=DEFAULT_N_FULL, k_max=DEFAULT_K_MAX, engine="unknown",
                     step_cm=0.0, verbose=True, n_z=DEFAULT_N_Z, n_r=DEFAULT_N_R,
                     n_full_stop=DEFAULT_N_FULL_STOP, rho_nom=0.0,
                     stop=True, stepend: dict | None = None):
    """Raw text file -> one-material LutV2 and its census rows. stop=False
    skips the stopping-position tables even when the raw file carries them.

    stepend: None (default) returns (lut, rows). A dict of build_stepend_material
    keywords (frame, n_z, n_r, n_a, n_full_se, geom, knots) returns (lut, rows, se_lut, se_rows);
    se_lut is None when the raw file has no dphi_rad column."""
    raw_path = Path(raw_path)
    mat = material_id(raw_path)
    t0 = time.time()
    raw = lut_contract.load_raw_full(raw_path)
    e0 = raw["E0_eV"]
    n_rows = e0.size
    e_grid, n_hist, n_surv, p_ann, st, su = group_by_energy(e0, raw["theta"], raw["dE_kin"], raw["ann"])
    sz = sr = None
    if stop and raw["z"] is not None:
        sz, sr = group_stops(e0, e_grid, raw["z"], raw["r"], raw["ann"])
    se_lut, se_rows = None, []
    if stepend is not None and raw["dphi"] is not None:
        se_in = group_stepend(e0, e_grid, raw["ann"], raw["theta"], raw["dE_kin"], raw["z"],
                              raw["r"], raw["dphi"])
        del raw, e0
        se_lut, se_info = build_stepend_material(e_grid, se_in, step_cm, k_max=k_max, **stepend)
        del se_in
        se_rows = [{"engine": engine, "material": int(mat), "energy_index": i,
                    "E_keV": float(e_grid[i]) / 1000.0, "n_surv": int(n_surv[i]),
                    "se_code": d["se_code"], "se_name": VALID_NAMES[d["se_code"]],
                    "n_se_pooled": d["n_se_pooled"]} for i, d in enumerate(se_info)]
    else:
        del raw, e0
    lut, info = build_material(e_grid, n_hist, n_surv, p_ann, st, su,
                               n_theta=n_theta, n_u=n_u, n_full=n_full, k_max=k_max, step_cm=step_cm,
                               stop_z=sz, stop_r=sr, n_z=n_z, n_r=n_r, n_full_stop=n_full_stop,
                               rho_nom=rho_nom)
    rows = census_rows(engine, mat, lut, info)
    if verbose:
        counts = {name: 0 for name in VALID_NAMES.values()}
        for r in rows:
            counts[r["valid_name"]] += 1
        stop_txt = "stopping tables" if lut.has_stop else "no end positions in the raw file"
        if se_lut is not None:
            se_counts = {name: 0 for name in VALID_NAMES.values()}
            for r in se_rows:
                se_counts[r["se_name"]] += 1
            stop_txt += f", step-end frame {se_lut.frame} {se_counts}"
        print(f"  mat{mat:02d}: {n_rows} histories, {lut.n_e} energies, "
              f"{counts}, {stop_txt} ({time.time() - t0:.1f} s)", flush=True)
    if stepend is None:
        return lut, rows
    return lut, rows, se_lut, se_rows


def assert_lut_equal(a: LutV2, b: LutV2):
    if np.float32(a.step_cm) != np.float32(b.step_cm):          # stored as float32
        raise AssertionError(f"LUT step_cm differs: {a.step_cm} vs {b.step_cm}")
    for attr in ("e_grid", "p_ann", "n_hist", "n_surv", "valid", "theta_icdf", "u_icdf",
                 "rho_nom", "svalid", "z_icdf", "r_icdf"):
        x, y = getattr(a, attr), getattr(b, attr)
        if x.shape != y.shape or not np.array_equal(x, y):
            raise AssertionError(f"LUT field {attr} differs")


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------
def combine_tables(mats: list[LutV2], ids: list[int], luts_dir: Path, verbose=True) -> LutV2:
    """One-material LUTs in numeric material order -> ann_LUT.raw + scatt-eloss_LUT.raw."""
    luts_dir = Path(luts_dir)
    lut = lut_contract.combine(mats)          # raises on grid / dimension mismatch
    lut.check()
    luts_dir.mkdir(parents=True, exist_ok=True)
    ann_path = luts_dir / "ann_LUT.raw"
    scatt_path = luts_dir / "scatt-eloss_LUT.raw"
    lut_contract.write_ann_lut(ann_path, lut)
    lut_contract.write_scatt_lut(scatt_path, lut)
    back = lut_contract.read_luts(ann_path, scatt_path)
    assert_lut_equal(back, lut)
    if verbose:
        print(f"  combined {len(mats)} materials (ids {ids[0]}..{ids[-1]}, numeric order) "
              f"N_E={lut.n_e} N_theta={lut.n_theta} N_u={lut.n_u} N_z={lut.n_z} N_r={lut.n_r}")
        for p in (ann_path, scatt_path):
            print(f"  -> {p} ({p.stat().st_size / 1024:.1f} KB)")
    return lut


# ---------------------------------------------------------------------------
# Census
# ---------------------------------------------------------------------------
def summarise_census(rows) -> dict:
    counts = {name: 0 for name in VALID_NAMES.values()}
    for r in rows:
        counts[VALID_NAMES[int(r["valid_code"])]] += 1
    return counts


def write_census(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CENSUS_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in CENSUS_COLUMNS})


def print_census(rows, engine):
    counts = summarise_census(rows)
    print(f"\nCensus [{engine}] over {len(rows)} (material, energy) rows: "
          + ", ".join(f"{k}={v}" for k, v in counts.items()))
    flagged = [r for r in rows if r["valid_code"] in (POOLED, SPARSE)]
    for r in flagged:
        print(f"    mat{r['material']:02d} i={r['energy_index']:3d} E={r['E_keV']:9.3f} keV "
              f"n_surv={r['n_surv']:6d} pooled={r['n_pooled']:6d} {r['valid_name']}")
    if any(r["n_stop"] for r in rows):
        sc = {name: 0 for name in VALID_NAMES.values()}
        for r in rows:
            sc[r["stop_name"]] += 1
        print("  stopping-position rows: " + ", ".join(f"{k}={v}" for k, v in sc.items())
              + " (DEAD = no stopped history to describe; the engine uses the hinge point there)")
    else:
        print("  no stopping-position tables (raw files without z_cm r_cm columns)")
    degen = [r for r in rows if r["degenerate_theta_bins"] > 0]
    if degen:
        print(f"  {len(degen)} live rows hit build_row's degenerate-bin fallback:")
        for r in degen:
            print(f"    mat{r['material']:02d} i={r['energy_index']:3d} E={r['E_keV']:9.3f} keV "
                  f"empty theta bins={r['degenerate_theta_bins']}")
    else:
        print("  no row hit build_row's degenerate-bin fallback")


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------
def _repo_relative(path: Path) -> str:
    """Path relative to the repository root when inside it, else absolute."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT.parent))
    except ValueError:
        return str(path)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write_run_info(out_dir, engine, raw_dir, materials, **params) -> Path:
    """run.json: what these LUTs were built from and how (read by luts-check/compare)."""
    info = {"engine": engine, "raw_dir": _repo_relative(raw_dir), "materials": list(materials),
            **params, "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "commit": _git_commit()}
    path = Path(out_dir) / "run.json"
    text = json.dumps(info, indent=2)
    text = re.sub(r"\[\s+([\d,\s]*?)\s+\]",                       # number lists on one line
                  lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", text)
    path.write_text(text + "\n")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _worker(job):
    path, kw = job
    return process_material(path, **kw)


def read_sim_step(sim_in) -> float:
    """STEP_CM, the first positional line of sim.in."""
    first = next(l for l in Path(sim_in).read_text().splitlines() if l.split("!")[0].strip())
    return float(first.split("!")[0].split()[0])


def resolve_step(raw_files, step_cm=None, sim_in=None) -> float:
    """Path length per step of the raw data, from the drivers' '# config:' line.

    Raw files written before that line existed fall back to --step-cm or, failing
    that, to sim.in (with a warning: sim.in may have changed since the run)."""
    seen = {p.name: lut_contract.read_raw_config(p).get("step_cm") for p in raw_files}
    known = {v for v in seen.values() if v is not None}
    if len(known) > 1:
        raise SystemExit(f"raw files disagree on step_cm: {sorted(known)}")
    if known:
        step = known.pop()
        if step_cm is not None and abs(step_cm - step) > 1e-9:
            raise SystemExit(f"--step-cm {step_cm} contradicts the raw files' step_cm={step}")
        if None in seen.values():
            warnings.warn("some raw files have no config line; assuming they share step_cm="
                          f"{step}", RuntimeWarning)
        return step
    if step_cm is not None:
        return float(step_cm)
    sim_in = Path(sim_in) if sim_in else ROOT / "input" / "sim.in"
    step = read_sim_step(sim_in)
    print(f"  WARNING: raw files carry no config line (written before it existed); taking "
          f"step_cm={step} from {sim_in}. Pass --step-cm if the data used another step.", flush=True)
    return step


def read_mat_density(path) -> float:
    """Mass density [g/cm3] from a PENELOPE .mat file header."""
    for line in Path(path).read_text(errors="replace").splitlines()[:10]:
        if "Mass density" in line and "=" in line:
            return float(line.split("=")[1].split()[0])
    raise ValueError(f"no 'Mass density' line in {path}")


def resolve_densities(raw_files, mat_dir=None, verbose=True) -> list[float]:
    """Nominal density per raw file: the drivers' config line (rho_gcm3=), else
    <mat_dir>/matN.mat (default luts/input/mats), else 0 (unknown; the engine
    then refuses --density). An explicit --mat-dir must agree with the config
    line, since the two name the same material."""
    explicit = mat_dir is not None
    mat_dir = Path(mat_dir) if explicit else ROOT / "input" / "mats"
    out, missing = [], []
    for p in raw_files:
        rho = lut_contract.read_raw_config(p).get("rho_gcm3")
        mat_file = mat_dir / f"mat{material_id(p)}.mat"
        if rho is None and mat_file.is_file():
            rho = read_mat_density(mat_file)
        elif rho is not None and explicit and mat_file.is_file():
            rho_file = read_mat_density(mat_file)
            if abs(rho_file - rho) > 1e-6 * max(rho, rho_file):
                raise SystemExit(f"{p.name}: config density {rho} g/cm3 differs from {mat_file} "
                                 f"({rho_file}); the raw data was not simulated with these materials")
        if rho is None:
            rho = 0.0
            missing.append(p.name)
        out.append(float(rho))
    if missing and verbose:
        print(f"  WARNING: no nominal density for {', '.join(missing)} (no rho_gcm3 in the config "
              f"line and no .mat file in {mat_dir}); the engine will refuse --density for them",
              flush=True)
    return out


def write_stepend(se_mats: list[StepEndLut], lut: LutV2, out_dir: Path, verbose=True) -> StepEndLut:
    """Combine one-material step-end tables -> <out_dir>/stepend_LUT.raw (read back and compared)."""
    se = lut_contract.combine_stepend(se_mats)
    se.check()
    if not np.array_equal(se.e_grid, lut.e_grid) or se.n_m != lut.n_m:
        raise ValueError("step-end tables disagree with ann_LUT (energy grid or materials)")
    if np.float32(se.step_cm) != np.float32(lut.step_cm):
        raise ValueError("step-end tables disagree with ann_LUT on step_cm")
    path = Path(out_dir) / "stepend_LUT.raw"
    lut_contract.write_stepend_lut(path, se)
    back = lut_contract.read_stepend_lut(path)
    for attr in ("e_grid", "evalid", "z_icdf_s", "r_icdf_s", "a1_icdf", "a2_icdf", "u_icdf_s", "geom", "knots",
                 "version"):
        if not np.array_equal(getattr(back, attr), getattr(se, attr)):
            raise AssertionError(f"stepend_LUT.raw round trip: {attr} differs")
    if verbose:
        print(f"  step-end: v{se.version} frame {se.frame} ({FRAME_NAMES[se.frame]}) geom {GEOM_NAMES[se.geom]} "
              f"knots {KNOT_NAMES[se.knots]} N_z={se.n_z} N_r={se.n_r} N_a={se.n_a} N_FULL_SE={se.n_full_se}"
              f"\n  -> {path} ({path.stat().st_size / 1024:.1f} KB)")
    return se


def build_engine(engine, raw_dir, out_root, n_theta, n_u, n_full, k_max, workers=8,
                 census_path=None, step_cm=None, n_z=DEFAULT_N_Z, n_r=DEFAULT_N_R,
                 n_full_stop=DEFAULT_N_FULL_STOP, mat_dir=None, materials=None,
                 stepend=True, frame=DEFAULT_FRAME, n_z_se=DEFAULT_N_Z_SE, n_r_se=DEFAULT_N_R_SE,
                 n_a=DEFAULT_N_A, n_full_se=N_FULL_SE, knots=DEFAULT_KNOTS):
    """materials: optional list of material ids to build (a subset of the raw files,
    slots in numeric order). stepend=False skips the step-end tables; they are also
    skipped, with a warning, unless every raw file carries dphi_rad. Every raw file
    must be a fixed-path one (no geom=sphere in its config line); knots: the a1 / a2 /
    u knot scheme (KNOT_NAMES)."""
    raw_dir, out_dir = get_engine_dirs(engine, raw_dir, out_root)
    if Path(out_dir).is_dir() and any(s.name.startswith("step") and (s / "stepend_LUT.raw").exists()
                                      for s in Path(out_dir).iterdir()):
        # tables at the top of a ladder bundle would turn it into one set for hmc (find_bundle)
        raise SystemExit(f"{out_dir} is a ladder bundle (step*/ table sets): give --out-dir {out_dir}/step<k>")
    raw_files = find_raw_files(raw_dir)
    if materials is not None:
        want = {int(m) for m in materials}
        missing = want - {material_id(p) for p in raw_files}
        if missing:
            raise SystemExit(f"no raw file for material(s) {sorted(missing)} in {raw_dir}")
        raw_files = [p for p in raw_files if material_id(p) in want]
    if not raw_files:
        raise SystemExit(f"no raw files found in {raw_dir}")
    if frame not in FRAME_NAMES:
        raise SystemExit(f"--frame must be one of {sorted(FRAME_NAMES)}")
    have_dphi = [lut_contract.raw_has_dphi(p) for p in raw_files]
    do_se = bool(stepend) and all(have_dphi)
    if stepend and not do_se:
        some = [p.name for p, h in zip(raw_files, have_dphi) if not h]
        msg = (f"raw files without the dphi_rad column ({', '.join(some)}): no step-end tables "
               f"(stepend_LUT.raw) are built")
        warnings.warn(msg, RuntimeWarning)
        print(f"  WARNING: {msg}", flush=True)
    # every raw file must come from the same engine build; the PENELOPE driver records no version in its config line
    versions = {next((str(v) for k, v in lut_contract.read_raw_config(p).items() if "version" in k.lower()), None)
                for p in raw_files}
    if len(versions) > 1:
        raise SystemExit(f"raw files from different engine versions: {sorted(map(str, versions))}")
    engine_version = versions.pop()
    if engine_version is None and engine == "penelope":
        engine_version = PENELOPE_VERSION
    step_cm = resolve_step(raw_files, step_cm)
    geoms = {str(lut_contract.read_raw_config(p).get("geom", "path")) for p in raw_files}
    if geoms != {"path"}:
        raise SystemExit(f"raw files with geometry {sorted(geoms - {'path'})}: only fixed-path raw files are "
                         f"supported (the sphere first-passage mode was dropped on 2026-10-30)")
    geom = GEOM_PATH
    rhos = resolve_densities(raw_files, mat_dir)
    print(f"Building LUTs for {engine}: step={step_cm} cm N_theta={n_theta} N_u={n_u} N_FULL={n_full} "
          f"K_MAX={k_max} N_z={n_z} N_r={n_r} N_FULL_STOP={n_full_stop}, {len(raw_files)} materials, "
          f"{workers} workers")
    print(f"  raw: {raw_dir}\n  out: {out_dir}")
    t0 = time.time()
    se_kw = dict(frame=frame, n_z=n_z_se, n_r=n_r_se, n_a=n_a, n_full_se=n_full_se, geom=geom,
                 knots=knots) if do_se else None
    if do_se:
        lut_contract.knot_levels(n_a, knots)                          # tail: N_a must fit, before any work
        print(f"  step-end tables: frame {frame} ({FRAME_NAMES[frame]}) geom {GEOM_NAMES[geom]} "
              f"knots {KNOT_NAMES[knots]} N_z={n_z_se} N_r={n_r_se} N_a={n_a} N_FULL_SE={n_full_se}")
    jobs = [(p, dict(n_theta=n_theta, n_u=n_u, n_full=n_full, k_max=k_max, engine=engine,
                     step_cm=step_cm, verbose=True, n_z=n_z, n_r=n_r, n_full_stop=n_full_stop,
                     rho_nom=rho, stepend=se_kw))
            for p, rho in zip(raw_files, rhos)]
    mats, rows, se_mats, se_rows = [], [], [], []

    def _take(res):
        mats.append(res[0]); rows.extend(res[1])
        if se_kw is not None:
            se_mats.append(res[2]); se_rows.extend(res[3])

    if workers > 1 and len(jobs) > 1:
        # Python >= 3.14 defaults to 'forkserver', which needs a UNIX socket;
        # the workers are pure numpy and the parent is single-threaded here,
        # so plain fork is both safe and portable to sandboxed hosts.
        ctx = multiprocessing.get_context("fork") if "fork" in multiprocessing.get_all_start_methods() else None
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for res in ex.map(_worker, jobs):     # map keeps numeric material order
                _take(res)
    else:
        for j in jobs:
            _take(_worker(j))
    rows.sort(key=lambda r: (r["material"], r["energy_index"]))
    print(f"\nPer-material LUTs built in {time.time() - t0:.1f} s. Combining ...")
    lut = combine_tables(mats, [material_id(p) for p in raw_files], out_dir)
    if census_path is None:
        census_path = out_dir / "gap_census.csv"
    write_census(rows, census_path)
    print(f"  -> {census_path}")
    print_census(rows, engine)
    se_params = {"stepend": False}
    if do_se:
        se = write_stepend(se_mats, lut, out_dir)
        se_census = out_dir / "stepend_census.csv"
        with open(se_census, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SE_CENSUS_COLUMNS)
            w.writeheader()
            for r in sorted(se_rows, key=lambda r: (r["material"], r["energy_index"])):
                w.writerow({k: r[k] for k in SE_CENSUS_COLUMNS})
        sc = {name: 0 for name in VALID_NAMES.values()}
        for r in se_rows:
            sc[r["se_name"]] += 1
        print("  step-end rows: " + ", ".join(f"{k}={v}" for k, v in sc.items())
              + f"\n  -> {se_census}")
        se_params = {"stepend": True, "stepend_version": se.version, "frame": se.frame, "n_z_se": se.n_z,
                     "n_r_se": se.n_r, "n_a": se.n_a, "n_full_se": se.n_full_se, "geom": GEOM_NAMES[se.geom],
                     "knots": KNOT_NAMES[se.knots], "knot_levels": [round(float(x), 6) for x in se.levels]}
    run_path = write_run_info(out_dir, engine, raw_dir, [material_id(p) for p in raw_files],
                              step_cm=step_cm, n_theta=n_theta, n_u=n_u, n_full=n_full, k_max=k_max,
                              n_z=lut.n_z, n_r=lut.n_r, n_full_stop=n_full_stop,
                              rho_nom=[round(float(r), 6) for r in lut.rho_nom],
                              engine_version=engine_version, **se_params)
    print(f"  -> {run_path}")
    print(f"\nTotal wall time {time.time() - t0:.1f} s")
    return lut, rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", choices=ENGINES, default="penelope")
    p.add_argument("--n-theta", type=int, default=DEFAULT_N_THETA, help="theta quantile knots")
    p.add_argument("--n-u", type=int, default=DEFAULT_N_U, help="u quantile knots per theta bin")
    p.add_argument("--n-full", type=int, default=DEFAULT_N_FULL, help="survivors for a FULL row")
    p.add_argument("--k-max", type=int, default=DEFAULT_K_MAX, help="max neighbour rows pooled each side")
    p.add_argument("--raw-dir", type=str, default=None, help="directory with mat*.dat or mat*.dat.zst")
    p.add_argument("--out-dir", type=str, default=None,
                   help="output root (the two LUTs and gap_census.csv go under it)")
    p.add_argument("--workers", type=int, default=8, help="parallel material workers")
    p.add_argument("--step-cm", type=float, default=None,
                   help="step length of the raw data [cm]; only for raw files without a config "
                        "line (default then: luts/input/sim.in)")
    p.add_argument("--n-z", type=int, default=DEFAULT_N_Z,
                   help="stopping-position knots along the initial direction")
    p.add_argument("--n-r", type=int, default=DEFAULT_N_R,
                   help="stopping-position knots perpendicular to it, per z bin")
    p.add_argument("--n-full-stop", type=int, default=DEFAULT_N_FULL_STOP,
                   help="stopped histories for a FULL stopping-position row")
    p.add_argument("--mat-dir", type=str, default=None,
                   help="matN.mat files giving the nominal densities when the raw config line "
                        "has no rho_gcm3 (default luts/input/mats)")
    p.add_argument("--materials", type=str, default=None,
                   help="comma-separated material ids to build (default: every raw file)")
    p.add_argument("--no-stepend", action="store_true",
                   help="do not build stepend_LUT.raw even if the raw files carry dphi_rad")
    p.add_argument("--frame", type=int, choices=sorted(FRAME_NAMES), default=DEFAULT_FRAME,
                   help="step-end direction frame: 0 theta (theta, |dphi|), 1 chord (alpha, |psi|)")
    p.add_argument("--n-z-se", type=int, default=DEFAULT_N_Z_SE, help="step-end z knots")
    p.add_argument("--n-r-se", type=int, default=DEFAULT_N_R_SE, help="step-end r knots per z bin")
    p.add_argument("--n-a", type=int, default=DEFAULT_N_A,
                   help="step-end a1 / a2 / u knots per (z, r) cell")
    p.add_argument("--n-full-se", type=int, default=N_FULL_SE,
                   help="survivors for a FULL step-end row")
    p.add_argument("--knots", choices=list(KNOT_NAMES.values()), default=KNOT_NAMES[DEFAULT_KNOTS],
                   help="step-end a1 / a2 / u knots: tail (fixed tail-dense levels, N_a = 16) or uniform (v5)")
    args = p.parse_args(argv)
    if min(args.n_theta, args.n_u, args.n_z, args.n_r, args.n_z_se, args.n_r_se, args.n_a) < 3:
        p.error("--n-theta, --n-u, --n-z, --n-r, --n-z-se, --n-r-se and --n-a must be >= 3")
    mats = [int(x) for x in args.materials.split(",") if x.strip()] if args.materials else None

    build_engine(args.engine, args.raw_dir, args.out_dir, args.n_theta,
                 args.n_u, args.n_full, args.k_max, workers=args.workers, step_cm=args.step_cm,
                 n_z=args.n_z, n_r=args.n_r, n_full_stop=args.n_full_stop, mat_dir=args.mat_dir,
                 materials=mats, stepend=not args.no_stepend, frame=args.frame, n_z_se=args.n_z_se,
                 n_r_se=args.n_r_se, n_a=args.n_a, n_full_se=args.n_full_se,
                 knots={v: k for k, v in KNOT_NAMES.items()}[args.knots])


if __name__ == "__main__":
    main()
