"""metrics.py -- scoring of positron annihilation maps (numpy only), as the HMC validation harness scores them.

Maps are float arrays of shape (Nx,Ny,Nz) (the .raw layout of rawio / make_hmc_inputs: x fastest). Radii are measured from the
source centre to *voxel centres*, so every code is scored identically. Quantiles are taken on the discrete radial CDF (no
interpolation): r_q = radius of the first voxel whose cumulative fraction >= q. Fractions are normalised to the annihilated total
unless stated otherwise; `escaped_fraction` uses nhist. Copied from validation/src/analysis/metrics.py of the development
repository (without its clinical-image part), so the numbers agree with results/qoi.csv.
"""
from __future__ import annotations

import numpy as np

QS = {"r50": 0.5, "r90": 0.9, "r999": 0.999}


# ------------------------------------------------------------------ geometry
def source_centre_cm(source_map: np.ndarray, d_cm) -> np.ndarray:
    """Activity-weighted centroid of the source map [cm]."""
    idx = np.indices(source_map.shape).reshape(3, -1)
    w = source_map.reshape(-1).astype(np.float64)
    c = (idx * w).sum(1) / w.sum()
    return (c + 0.5) * np.asarray(d_cm)


def voxel_radii(shape, d_cm, centre_cm) -> np.ndarray:
    ax = [(np.arange(n) + 0.5) * d_cm[i] - centre_cm[i] for i, n in enumerate(shape)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    return np.sqrt(X * X + Y * Y + Z * Z)


# ------------------------------------------------------------------ radial
def radial_cdf(counts, radii):
    """Return (unique radii, cumulative fraction) with voxels at equal radius merged."""
    c = np.asarray(counts, dtype=np.float64).ravel()
    r = np.round(np.asarray(radii).ravel(), 9)
    ur, inv = np.unique(r, return_inverse=True)
    h = np.bincount(inv, weights=c, minlength=ur.size)
    tot = h.sum()
    return ur, np.cumsum(h) / tot if tot > 0 else np.zeros_like(h)


def radial_qoi(counts, radii) -> dict:
    ur, F = radial_cdf(counts, radii)
    out = {k: float(ur[min(np.searchsorted(F, q - 1e-12), ur.size - 1)]) for k, q in QS.items()}
    c = np.asarray(counts, dtype=np.float64).ravel()
    out["rmean"] = float((c * np.asarray(radii).ravel()).sum() / c.sum())
    return out


def bootstrap_radial(counts, radii, n_boot=200, seed=12345) -> dict:
    """Poisson bootstrap of voxel counts. Returns {qoi: {value, sigma, lo, hi}}."""
    rng = np.random.default_rng(seed)
    base = radial_qoi(counts, radii)
    c = np.asarray(counts, dtype=np.float64).ravel()
    nz = np.flatnonzero(c)
    r = np.asarray(radii).ravel()
    # bin the occupied voxels by unique radius once; each resample is then O(n_occupied)
    ur, inv = np.unique(np.round(r[nz], 9), return_inverse=True)
    c_nz, r_nz = c[nz], r[nz]
    samples = {k: [] for k in base}
    for _ in range(n_boot):
        cb = rng.poisson(c_nz).astype(np.float64)
        tot = cb.sum()
        if tot == 0:
            continue
        F = np.cumsum(np.bincount(inv, weights=cb, minlength=ur.size)) / tot
        q = {k: float(ur[min(np.searchsorted(F, p - 1e-12), ur.size - 1)]) for k, p in QS.items()}
        q["rmean"] = float((cb * r_nz).sum() / tot)
        for k in base:
            samples[k].append(q[k])
    res = {}
    for k, v in base.items():
        s = np.asarray(samples[k])
        res[k] = {"value": v, "sigma": float(s.std(ddof=1)),
                  "lo": float(np.percentile(s, 2.5)), "hi": float(np.percentile(s, 97.5))}
    return res


def escaped_fraction(m, nhist) -> float:
    return float((nhist - np.sum(m, dtype=np.float64)) / nhist)


def ks_w1(counts_a, counts_b, radii) -> dict:
    """KS statistic and 1-Wasserstein distance [cm] between radial CDFs."""
    ur, Fa = radial_cdf(counts_a, radii)
    _, Fb = radial_cdf(counts_b, radii)
    dr = np.diff(ur)
    w1 = float(np.sum(np.abs(Fa - Fb)[:-1] * dr))
    return {"ks": float(np.max(np.abs(Fa - Fb))), "w1_cm": w1}


def ks_critical(na, nb, alpha=0.05) -> float:
    """Two-sample KS critical value (continuous approximation)."""
    c = np.sqrt(-0.5 * np.log(alpha / 2))
    return float(c * np.sqrt((na + nb) / (na * nb)))


def voxel_ratio(a, b, min_counts=10, min_total=10) -> dict:
    """Per-voxel ratio a/b (both normalised to unit sum) with Poisson errors.

    Returns summary over voxels where both have >= min_counts raw counts, and the two-sample
    homogeneity chi2/ndf: for each voxel with a + b >= min_total the expected counts under a common
    normalised intensity are E_a = (a+b) ta/(ta+tb), E_b = (a+b) tb/(ta+tb) and
    chi2 = sum (a-E_a)^2/E_a + (b-E_b)^2/E_b, ndf = n_voxels - 1.  This form is unbiased for unequal
    totals ta != tb (the earlier scaled-Poisson form was biased upward by sparse voxels when ta != tb).
    """
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    ta, tb = a.sum(), b.sum()
    sel = (a >= min_counts) & (b >= min_counts)
    ratio = (a[sel] / ta) / (b[sel] / tb)
    err = ratio * np.sqrt(1 / a[sel] + 1 / b[sel])
    z = (ratio - 1) / err
    tot = a + b
    use = tot >= min_total
    fa, fb = ta / (ta + tb), tb / (ta + tb)
    ea, eb = tot[use] * fa, tot[use] * fb
    ch = (a[use] - ea) ** 2 / ea + (b[use] - eb) ** 2 / eb
    return {"n_voxels": int(sel.sum()), "ratio_mean": float(ratio.mean()) if sel.any() else float("nan"),
            "ratio_median": float(np.median(ratio)) if sel.any() else float("nan"),
            "ratio_std": float(ratio.std()) if sel.any() else float("nan"),
            "z_mean": float(z.mean()) if sel.any() else float("nan"), "z_std": float(z.std()) if sel.any() else float("nan"),
            "chi2_ndf": float(ch.sum() / max(ch.size - 1, 1)), "n_chi2": int(ch.size)}


# ------------------------------------------------------------------ profiles
def profile_1d(m, axis=0) -> np.ndarray:
    other = tuple(i for i in range(3) if i != axis)
    return np.sum(m, axis=other, dtype=np.float64)


def _crossings(x, y, level):
    """Leftmost and rightmost positions where y crosses `level` (linear interp)."""
    above = y >= level
    if not above.any():
        return None
    i0 = int(np.argmax(above)); i1 = len(y) - 1 - int(np.argmax(above[::-1]))
    def cross(i, j):
        if i < 0 or j >= len(y) or j < 0:
            return x[min(max(i, 0), len(y) - 1)]
        return x[i] + (level - y[i]) * (x[j] - x[i]) / (y[j] - y[i])
    left = x[0] if i0 == 0 else cross(i0 - 1, i0)
    right = x[-1] if i1 == len(y) - 1 else cross(i1 + 1, i1)
    return left, right


def full_width(x, y, frac) -> float:
    """Width at `frac` of the maximum (0.5 FWHM, 0.1 FWTM, 0.01 FW1%)."""
    c = _crossings(x, y, frac * y.max())
    return float("nan") if c is None else float(c[1] - c[0])


def profile_widths(m, d_cm, axis=0, n_boot=200, seed=7, centre_cm=None) -> dict:
    """FWHM/FWTM/FW1% (cm) of the projected profile along `axis` with bootstrap sigma."""
    x = (np.arange(m.shape[axis]) + 0.5) * d_cm[axis]
    fr = {"fwhm": 0.5, "fwtm": 0.1, "fw1pct": 0.01}
    p = profile_1d(m, axis)
    res = {k: full_width(x, p, f) for k, f in fr.items()}
    rng = np.random.default_rng(seed)
    s = {k: [] for k in fr}
    for _ in range(n_boot):
        pb = rng.poisson(p)
        if pb.max() == 0:
            continue
        for k, f in fr.items():
            s[k].append(full_width(x, pb.astype(float), f))
    return {k: {"value": res[k], "sigma": float(np.nanstd(s[k], ddof=1))} for k in fr}


# ------------------------------------------------------------------ image metrics
def _box_mean(a, k=3):
    """Mean over a k^3 window (valid region), via cumulative sums."""
    c = a
    for ax in range(3):
        cs = np.cumsum(c, axis=ax, dtype=np.float64)
        pad = [(0, 0)] * 3; pad[ax] = (1, 0)
        cs = np.pad(cs, pad)
        n = c.shape[ax]
        c = (np.take(cs, np.arange(k, n + 1), axis=ax) - np.take(cs, np.arange(0, n - k + 1), axis=ax))
    return c / k ** 3


def ssim(a, b, local=False, k=3, data_range=None):
    """SSIM (Wang 2004 constants K1=.01, K2=.03). global scalar or k^3-window map."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    L = data_range if data_range is not None else max(a.max(), b.max()) - min(a.min(), b.min())
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    if not local:
        ma, mb = a.mean(), b.mean()
        va, vb = a.var(), b.var(); cov = ((a - ma) * (b - mb)).mean()
        return float(((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2)))
    ma, mb = _box_mean(a, k), _box_mean(b, k)
    va = _box_mean(a * a, k) - ma ** 2
    vb = _box_mean(b * b, k) - mb ** 2
    cov = _box_mean(a * b, k) - ma * mb
    return ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))


def gamma_pass_rate(test, ref, dd_percent=3.0, dta_vox=1.0, threshold=0.1) -> float:
    """3D global gamma pass rate. dd relative to max(ref); voxels with ref >= threshold*max."""
    test = np.asarray(test, dtype=np.float64); ref = np.asarray(ref, dtype=np.float64)
    dmax = ref.max()
    dd = dd_percent / 100.0 * dmax
    w = int(np.ceil(2 * dta_vox))
    pad = np.pad(test, w, mode="edge")
    n = ref.shape
    g2 = np.full(n, np.inf)
    for i in range(-w, w + 1):
        for j in range(-w, w + 1):
            for k in range(-w, w + 1):
                dist2 = (i * i + j * j + k * k) / dta_vox ** 2
                if dist2 > 4:
                    continue
                sh = pad[w + i:w + i + n[0], w + j:w + j + n[1], w + k:w + k + n[2]]
                g2 = np.minimum(g2, dist2 + ((sh - ref) / dd) ** 2)
    sel = ref >= threshold * dmax
    return float(np.mean(g2[sel] <= 1.0))


def image_metrics(test, ref, rel_threshold=0.01, gamma=True) -> dict:
    """Image-comparison metrics of `test` against `ref` (both normalised to unit sum)."""
    t = np.asarray(test, dtype=np.float64); r = np.asarray(ref, dtype=np.float64)
    t = t / t.sum(); r = r / r.sum()
    diff = t - r
    rng_ = r.max() - r.min()
    mse = float(np.mean(diff ** 2))
    sel = r >= rel_threshold * r.max()
    rel = diff[sel] / r[sel]
    ss_res = np.sum(diff ** 2); ss_tot = np.sum((r - r.mean()) ** 2)
    # linear R^2 of test vs ref (squared Pearson correlation)
    rr = float(np.corrcoef(t.ravel(), r.ravel())[0, 1] ** 2)
    out = {
        "mae": float(np.mean(np.abs(diff))), "rmse": float(np.sqrt(mse)),
        "nrmse": float(np.sqrt(mse) / rng_),
        "psnr_db": float(10 * np.log10(rng_ ** 2 / mse)) if mse > 0 else float("inf"),
        "ssim_global": ssim(t, r), "ssim_local_mean": float(ssim(t, r, local=True).mean()),
        "r2_linear": rr, "r2_determination": float(1 - ss_res / ss_tot),
        "rel_err": {"mean": float(rel.mean()), "std": float(rel.std()),
                    "median": float(np.median(rel)),
                    "p95_abs": float(np.percentile(np.abs(rel), 95)), "n": int(sel.sum())},
    }
    if gamma:
        out["gamma_3pct_1vox"] = gamma_pass_rate(t, r, 3.0, 1.0)
    return out


# ------------------------------------------------------------------ interface metrics
def interface_metrics(m, vm, d_cm, case_geometry: dict, density: dict | None = None,
                      axis=0) -> dict:
    """Metrics for slab/sphere/lamella cases.

    fraction per material; for slab/sphere: left/right (x-index < interface) ratio,
    position (mm from the interface, positive into the dense side) and height
    (fraction of annihilations per mm) of the dense-side peak of the x profile, and
    depletion depth on the light side = distance from the interface (mm) to the
    point where the light-side profile first reaches 90 % of its light-side maximum.
    """
    tot = float(np.sum(m, dtype=np.float64))
    out = {"fraction_per_material": {int(k): float(m[vm == k].sum() / tot) for k in np.unique(vm)}}
    p = profile_1d(m, axis) / tot / (d_cm[axis] * 10)          # fraction per mm
    x_mm = (np.arange(m.shape[axis]) + 0.5) * d_cm[axis] * 10
    if "interface_index" in case_geometry:
        ii = case_geometry["interface_index"]
        xi = ii * d_cm[axis] * 10
        left = float(p[:ii].sum() * d_cm[axis] * 10); right = float(p[ii:].sum() * d_cm[axis] * 10)
        out["left_fraction"] = left; out["right_fraction"] = right
        out["LR_ratio"] = left / right if right > 0 else float("inf")
        dl = density[case_geometry["mat_left"]] if density else 1.0
        dr = density[case_geometry["mat_right"]] if density else 1.0
        dense_left = dl > dr
        sl = slice(0, ii) if dense_left else slice(ii, None)
        ll = slice(ii, None) if dense_left else slice(0, ii)
        pd, xd = p[sl], x_mm[sl]
        k = int(np.argmax(pd))
        out["dense_side"] = "left" if dense_left else "right"
        out["dense_peak_pos_mm"] = float(abs(xd[k] - xi))
        out["dense_peak_height_per_mm"] = float(pd[k])
        pl, xl = p[ll], x_mm[ll]
        # walk away from the interface on the light side
        order = np.argsort(np.abs(xl - xi))
        pl_o = pl[order]; xl_o = np.abs(xl[order] - xi)
        hit = np.flatnonzero(pl_o >= 0.9 * pl.max())
        out["light_depletion_depth_mm"] = float(xl_o[hit[0]]) if hit.size else float("nan")
    if "lamella_index_range" in case_geometry:
        a, b = case_geometry["lamella_index_range"]
        out["lamella_fraction"] = float(m[a:b].sum() / tot)
        out["beyond_fraction"] = float(m[b:].sum() / tot)      # transmitted: past the lamella, on the side away from the source
    return out


# ------------------------------------------------------------------ driver
def summarize_map(m, d_cm, centre_cm, nhist=None, n_boot=200) -> dict:
    r = voxel_radii(m.shape, d_cm, centre_cm)
    out = {"annihilated": float(m.sum()),
           "radial_cm": bootstrap_radial(m, r, n_boot)}
    if nhist:
        out["escaped_fraction"] = escaped_fraction(m, nhist)
    out["profile_x_cm"] = profile_widths(m, d_cm, 0, n_boot)
    return out


def compare_maps(m, ref, d_cm, centre_cm) -> dict:
    r = voxel_radii(m.shape, d_cm, centre_cm)
    out = ks_w1(m, ref, r)
    out["ks_crit_5pct"] = ks_critical(m.sum(), ref.sum())
    out["voxel_ratio"] = voxel_ratio(m, ref)
    out["image"] = image_metrics(m, ref)
    return out
