"""
figcommon.py -- style, cached-statistics loading and the panels shared by
check_figures.py (one run) and compare_figures.py (two runs).
"""
from __future__ import annotations

import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from runs import LUTS, Run  # noqa: E402

INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
ENGINE_COLOR = {"penelope": "#2a78d6", "geant4": "#eb6834", "egsnrc": "#2a9d6f", "fluka": "#8a5cc2"}
ENGINE_CMAP = {"penelope": "Blues", "geant4": "Oranges", "egsnrc": "Greens", "fluka": "Purples"}
PAIR_COLOR = ("#2a78d6", "#eb6834")          # run A, run B in comparisons
DIV = LinearSegmentedColormap.from_list("div", ["#2b5c9e", "#8fb0d9", "#f0efec", "#e7a083", "#b23a1f"])
DIV.set_bad("#d9d8d4")
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#9cc3e6", "#2a78d6", "#0d3b73", "#d03b3b"])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})
warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)   # DEAD rows are NaN by design
warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
TICKS = [1, 10, 100, 1000, 4600]
NMIN_RHO = 1000   # Spearman rho from n survivors has s.e. ~ 1/sqrt(n); 100 would be pure noise here
TISSUE = {1: "air", **{m: f"lung {m-1}" for m in range(2, 9)}, 9: "adipose", 10: "adipose", 11: "adipose",
          12: "soft tissue", 13: "soft tissue", 14: "soft tissue", 15: "connective",
          **{m: "skeletal" for m in range(16, 30)}, 30: "cortical bone"}
LEVELS = [(0.9, ":", "E₉₀"), (0.5, "-", "E₅₀"), (0.1, "--", "E₁₀")]


def tissue(m): return TISSUE.get(m, f"material {m}")
def run_color(run: Run): return ENGINE_COLOR.get(run.engine, "#5b5b5b")
def run_cmap(run: Run): return ENGINE_CMAP.get(run.engine, "Greys")
def mat_color(mats, m): return plt.cm.viridis(mats.index(m) / max(len(mats) - 1, 1))


_RHO = {}                                    # filled from run.json rho_nom by load(); else the .mat files
def density(m):
    if m not in _RHO:
        txt = (LUTS / "input" / "mats" / f"mat{m}.mat").read_text()
        _RHO[m] = float(re.search(r"Mass density =\s*([\d.E+-]+)", txt).group(1))
    return _RHO[m]


def focus_material(mats):
    """Material for the single-material figures: 13 (soft tissue) when present.
    Must be one of extract_stats.KEEP, whose 2e4-event samples L7/S4 need."""
    for m in (13, 1, 30):
        if m in mats: return m
    return mats[len(mats) // 2]


# ---------------------------------------------------------------- loading
def load(run: Run, parts, cache) -> dict:
    """{material: merged statistics} from the cache; warns when a source changed since."""
    D, stale = {}, []
    _RHO.update({m: float(r) for m, r in run.rho.items() if r})
    for m in run.materials:
        d = {}
        for part in parts:
            f = run.raw_cache(m, cache) if part == "raw" else run.lut_cache(m, cache)
            if not f.is_file():
                sys.exit(f"missing {part} statistics for {run.key}, material {m} ({f}); "
                         f"run luts-check without --figures-only first")
            z = dict(np.load(f))
            src = run.raw_file(m) if part == "raw" else run.scatt
            if abs(src.stat().st_mtime - float(z.pop("src_mtime", 0))) > 1e-3:
                stale.append(f"{part} m{m}")
            d.update(z)
        D[m] = d
    if stale:
        print(f"warning: {run.key}: sources changed since the statistics were extracted "
              f"({', '.join(stale[:4])}{' ...' if len(stale) > 4 else ''}); rerun without --figures-only",
              file=sys.stderr)
    return D


def energies(D): return next(iter(D.values()))["E"] / 1e3
def stack(D, mats, k): return np.array([D[m][k] for m in mats])


# ---------------------------------------------------------------- axes helpers
def clean(ax, grid="y"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if grid: ax.grid(axis=grid, color=GRID, lw=0.6); ax.set_axisbelow(True)


def ylog_e(ax):
    ax.set_xscale("log"); ax.set_xlabel("E₀ (keV)"); ax.set_xticks(TICKS, [f"{t:g}" for t in TICKS])


def e_ticks(ax, E):
    ax.set_xticks([np.interp(np.log(t), np.log(E), np.arange(E.size)) for t in TICKS], [f"{t:g}" for t in TICKS])


def ematrix(ax, M, E, mats, cmap, norm, title, cb=None, fig=None):
    """Material x energy map (rows in run order)."""
    im = ax.pcolormesh(np.arange(E.size + 1) - 0.5, np.arange(len(mats) + 1) + 0.5, np.ma.masked_invalid(M),
                       cmap=cmap, norm=norm, edgecolors=BG, linewidth=0.2)
    e_ticks(ax, E)
    ax.set_xlim(*np.interp(np.log([8, 4600]), np.log(E), np.arange(E.size)))
    step = max(1, len(mats) // 6)
    ax.set_yticks(range(1, len(mats) + 1, step), [mats[i] for i in range(0, len(mats), step)])
    ax.invert_yaxis(); ax.set_xlabel("E₀ (keV)"); ax.set_ylabel("material"); ax.set_title(title, color=INK)
    if cb: fig.colorbar(im, ax=ax, label=cb, fraction=0.04, pad=0.02)
    return im


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=130, bbox_inches="tight"); plt.close(fig); print(out.name + "/" + name)


# ---------------------------------------------------------------- panels shared by check and compare
def e_at(D, m, E, lev):
    """Energy where p_ann drops through lev (log-linear), NaN if it never does."""
    p = D[m]["p_ann"]; hit = np.flatnonzero((p[:-1] >= lev) & (p[1:] < lev))
    if not hit.size: return np.nan
    i = hit[0]; t = (p[i] - lev) / (p[i] - p[i + 1])
    return float(np.exp(np.log(E[i]) + t * np.log(E[i + 1] / E[i])))


def levels_vs_density(ax, D, E, mats, color, filled=True, marker="o"):
    rho = np.array([density(m) for m in mats])
    for lev, ls, lab in LEVELS:
        y = [e_at(D, m, E, lev) for m in mats]
        ax.plot(rho, y, ls=ls, lw=0.8, color=color)
        ax.scatter(rho, y, s=10 if filled else 30, zorder=3, lw=1.1, edgecolor=color,
                   facecolor=color if filled else "none", marker=marker)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mass density (g/cm³)"); ax.set_ylabel("energy at stopping level (keV)")


def bias_maps(axs, D, E, mats, tag, fig, cb=True):
    """LUT mean / raw mean − 1 for θ and u (two axes)."""
    F = stack(D, mats, "valid") == 1
    for ax, (k, sym) in zip(axs, [("mth", "θ"), ("mu", "u")]):
        B = stack(D, mats, k + "_lut") / stack(D, mats, k) - 1
        ematrix(ax, B * 100, E, mats, DIV, TwoSlopeNorm(0, -10, 10),
                f"{tag}: mean {sym}, LUT / raw − 1\nFULL rows: median {np.nanmedian(B[F])*100:+.2f} %, "
                f"max |Δ| {np.nanmax(np.abs(B[F]))*100:.2f} %",
                cb="%" if (cb and sym == "u") else None, fig=fig)


def ks_maps(axs, D, E, mats, tag, fig, cb=True):
    """KS distance of 2e4 LUT draws vs the raw survivors, over its 95 % critical value."""
    for ax, sym in zip(axs, ["th", "u"]):
        K = stack(D, mats, f"ks_{sym}") / stack(D, mats, "ks_noise")
        ematrix(ax, K, E, mats, SEQ, matplotlib.colors.Normalize(0, 2.5),
                f"{tag}: KS({'θ' if sym == 'th' else 'u'}) / KS₉₅  ({np.nanmean(K[~np.isnan(K)] > 1)*100:.1f} % of live rows > 1)",
                cb="KS / KS₉₅" if (cb and sym == "u") else None, fig=fig)


def centroid_profile(ax, D, mats, key, color, label, band=True, ls="-"):
    """Median and 5–95 % band over all FULL rows of the within-interval centroid offset."""
    A = np.concatenate([D[m][key] for m in mats]); A = A[~np.isnan(A).all(1)]
    nb = A.shape[1]; x = (np.arange(nb) + 0.5) / nb
    q = np.nanpercentile(A, [5, 50, 95], axis=0)
    if band: ax.fill_between(x, q[0], q[2], color=color, alpha=0.18, lw=0)
    ax.plot(x, q[1], color=color, lw=2, ls=ls, label=label)


def loo_error(D, mats, k):
    """Leave-one-out mean error in %, with the row's own LUT bias divided out."""
    return ((1 + stack(D, mats, "loo_" + k)) / (stack(D, mats, k + "_lut") / stack(D, mats, k)) - 1) * 100
