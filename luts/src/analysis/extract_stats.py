"""
extract_stats.py -- per-row statistics of one or more runs (see runs.py):
the raw simulation output and the LUTs checked against it (read-only): their own
raw data, or an independent check sample (runs.py). Feeds check_figures.py and compare_figures.py.

For every (run, material, energy row) it caches, under --cache:

raw/<raw dir>/mNN.npz (--scope raw|all; reads the raw text only, never a LUT;
shared by all runs that read the same raw file):
  * E, n_hist, n_surv, p_ann
  * raw moments and quantiles of theta and u, Spearman rho(theta, u),
    a 4000-event subsample of (theta, u), and 2e4 events for the materials in KEEP

lut/<table set>@<raw dir>/mNN.npz (--scope lut|all; also reads the run's LUTs):
  * validity code, analytic LUT means (mean of interval midpoints)
  * within-interval centroid offsets (mean - mid) / width for theta and u
  * KS distance of 2e4 draws through lut_contract.sample_many vs the raw survivors (with an
    independent sample, also its own sampling noise: ks_noise)
  * leave-one-out log-E interpolation error (row i predicted from i-1, i+1)
  * 2e4 LUT draws for the materials in KEEP

Each raw file is read once per call, however many runs share it. Cached
statistics whose source file has not changed are reused (--force recomputes).

Usage:
    python luts/src/analysis/extract_stats.py RUN [RUN ...] [--scope raw|lut|all]
                                              [--materials 1 13 30] [--workers 10] [--cache DIR] [--force]
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lutgen"))
import lut_contract  # noqa: E402
from build_tables import group_by_energy  # noqa: E402
from runs import CACHE, resolve  # noqa: E402

ND = 20000          # LUT draws per row
KEEP = (1, 13, 30)  # materials whose 2e4-event raw and LUT samples are kept
NSUB = 4000         # raw subsample per row kept for every material


def ks(a, b):
    a = np.sort(a); b = np.sort(b); x = np.concatenate([a, b])
    return np.max(np.abs(np.searchsorted(a, x, "right") / a.size - np.searchsorted(b, x, "right") / b.size))


def rank(x):
    r = np.empty(x.size); r[np.argsort(x, kind="stable")] = np.arange(x.size); return r


def spearman(x, y):
    if x.size < 3: return np.nan
    return np.corrcoef(rank(x), rank(y))[0, 1]


def sub(L, idx):
    return lut_contract.LutV2(L.e_grid[idx], L.p_ann[:, idx], L.n_hist[:, idx], L.n_surv[:, idx],
                        L.valid[:, idx], L.theta_icdf[:, idx], L.u_icdf[:, idx])


def bin_offsets(x, edges):
    """(mean - mid)/width per interval, NaN where empty or zero width."""
    nb = edges.size - 1
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nb - 1)
    cnt = np.bincount(b, minlength=nb); s = np.bincount(b, weights=x, minlength=nb)
    w = np.diff(edges); mid = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (s / cnt - mid) / w
    d[(cnt == 0) | (w <= 0)] = np.nan
    return d, b


def raw_part(eg, ns, st, su, m):
    """Statistics of the raw survivors alone (no LUT involved)."""
    NE = eg.size
    rng_sub, rng_keep = np.random.default_rng(1000 + m), np.random.default_rng(2000 + m)
    R = {k: np.full(NE, np.nan) for k in ["mth", "mu", "sth", "su", "rho"]}
    qth = np.full((NE, 3), np.nan); qu = np.full((NE, 3), np.nan); extra = {}
    for i in range(NE):
        t, u = st[i], su[i]
        if t.size == 0: continue
        R["mth"][i], R["mu"][i], R["sth"][i], R["su"][i] = t.mean(), u.mean(), t.std(), u.std()
        R["rho"][i] = spearman(t, u)
        qth[i] = np.quantile(t, [0.1, 0.5, 0.9]); qu[i] = np.quantile(u, [0.1, 0.5, 0.9])
        sel = rng_sub.permutation(t.size)[:NSUB]
        extra[f"sub_{i}"] = np.stack([t[sel], u[sel]]).astype(np.float32)
        if m in KEEP:
            sel = rng_keep.permutation(t.size)[:ND]
            extra[f"keep_{i}"] = np.stack([t[sel], u[sel]]).astype(np.float32)
    return dict(qth=qth, qu=qu, **R, **extra)


def lut_part(run, m, eg, st, su):
    """LUT row vs the raw survivors it was built from."""
    L = lut_contract.select(lut_contract.read_luts(run.ann, run.scatt), run.slot(m))
    assert np.allclose(L.e_grid, eg, rtol=1e-6)
    NE, NT, NU = L.n_e, L.n_theta, L.n_u
    Lev, Lod = sub(L, np.arange(0, NE, 2)), sub(L, np.arange(1, NE, 2))
    rng = np.random.default_rng(3000 + m)
    R = {k: np.full(NE, np.nan) for k in
         ["mth_lut", "mu_lut", "ks_th", "ks_u", "rho_lut",
          "loo_mth", "loo_mu", "loo_ksth", "loo_ksu", "ks_noise"]}
    dth = np.full((NE, NT - 1), np.nan); du = np.full((NE, NU - 1), np.nan); extra = {}
    for i in range(NE):
        t, u = st[i], su[i]
        if t.size == 0 or L.valid[0, i] == 0: continue
        thi, ui = L.theta_icdf[0, i].astype(float), L.u_icdf[0, i].astype(float)
        # analytic LUT means: linear interpolation => uniform within each knot interval
        R["mth_lut"][i] = np.mean(0.5 * (thi[:-1] + thi[1:]))
        R["mu_lut"][i] = np.mean(0.5 * (ui[:, :-1] + ui[:, 1:]))
        if L.valid[0, i] == 1:   # own survivors define the bins
            dth[i], b = bin_offsets(t, thi)
            acc = np.zeros(NU - 1); n = np.zeros(NU - 1)
            for bb in range(NT - 1):
                d, _ = bin_offsets(u[b == bb], ui[bb]); ok = ~np.isnan(d)
                acc[ok] += d[ok]; n[ok] += 1
            with np.errstate(invalid="ignore"): du[i] = acc / n
        r1, r2 = rng.random(ND), rng.random(ND)
        _, ts, us = lut_contract.sample_many(L, np.zeros(ND, int), np.full(ND, float(eg[i])),
                                             np.full(ND, 2.0), r1, r2)
        R["ks_th"][i], R["ks_u"][i], R["rho_lut"][i] = ks(ts, t), ks(us, u), spearman(ts, us)
        R["ks_noise"][i] = 1.36 * np.sqrt(1 / ND + 1 / t.size)   # 95% KS critical value
        if 0 < i < NE - 1 and L.valid[0, i - 1] and L.valid[0, i + 1]:
            Ls = Lev if i % 2 else Lod
            _, tl, ul = lut_contract.sample_many(Ls, np.zeros(ND, int), np.full(ND, float(eg[i])),
                                                 np.full(ND, 2.0), r1, r2)
            R["loo_mth"][i], R["loo_mu"][i] = tl.mean() / t.mean() - 1, ul.mean() / u.mean() - 1
            R["loo_ksth"][i], R["loo_ksu"][i] = ks(tl, t), ks(ul, u)
        if m in KEEP:
            extra[f"lut_{i}"] = np.stack([ts, us]).astype(np.float32)
    return dict(valid=L.valid[0], dth=dth, du=du, **R, **extra)


def work(args):
    """One raw file: its raw statistics and/or the LUT statistics of every run built from it."""
    raw_run, lut_runs, m, cache, do_raw = args
    path = raw_run.raw_file(m)
    e0, th, de, ann = lut_contract.load_raw(path)
    eg, nh, ns, pa, st, su = group_by_energy(e0, th, de, ann)
    del e0, th, de, ann
    if do_raw:
        out = raw_run.raw_cache(m, cache); out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, E=eg, n_hist=nh, n_surv=ns, p_ann=pa, src_mtime=path.stat().st_mtime,
                            **raw_part(eg, ns, st, su, m))
    for run in lut_runs:
        out = run.lut_cache(m, cache); out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, src_mtime=run.scatt.stat().st_mtime, **lut_part(run, m, eg, st, su))
    print(f"m{m:02d} {path.parent}: " + ", ".join((["raw"] if do_raw else []) + [r.key for r in lut_runs]),
          flush=True)


def _fresh(cache_file, src):
    """Cached statistics exist and were extracted from the current source file."""
    if not cache_file.is_file(): return False
    with np.load(cache_file) as z:
        return "src_mtime" in z.files and abs(float(z["src_mtime"]) - src.stat().st_mtime) < 1e-3


def plan(runs, scope, materials=None, cache=CACHE, force=False):
    """Group the work by raw file so each one is read once; skip fresh cache entries."""
    by_file = {}                                 # raw file -> (a run that reads it, material, every run that reads it)
    for r in runs:
        if scope != "raw" and not r.has_luts():
            raise SystemExit(f"run {r.key}: no LUTs in {r.dir} (use --scope raw, or run luts-build)")
        for m in r.materials:
            if materials is None or m in materials:
                by_file.setdefault(r.raw_file(m), (r, m, []))[2].append(r)
    jobs = []
    for f, (raw_run, m, group) in by_file.items():
        do_raw = scope != "lut" and (force or not _fresh(raw_run.raw_cache(m, cache), f))
        lut_runs = [r for r in group if scope != "raw" and (force or not _fresh(r.lut_cache(m, cache), r.scatt))]
        if do_raw or lut_runs:
            jobs.append((raw_run, lut_runs, m, cache, do_raw))
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="engine name, directory under luts/output, or path")
    ap.add_argument("--scope", choices=("raw", "lut", "all"), default="all",
                    help="raw: raw survivors only, never reads the LUTs; lut: LUT-vs-raw only; all: both")
    ap.add_argument("--materials", nargs="+", type=int, default=None, help="material ids (default: all of the run)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--force", action="store_true", help="recompute even when the cache is up to date")
    a = ap.parse_args(argv)
    runs = [resolve(r) for r in a.runs]
    jobs = plan(runs, a.scope, a.materials, a.cache, a.force)
    if not jobs:
        print("statistics up to date (--force to recompute)"); return
    with Pool(min(a.workers, len(jobs))) as p:
        p.map(work, jobs, chunksize=1)


if __name__ == "__main__":
    main()
