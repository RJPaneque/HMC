"""
compare_figures.py -- figures comparing TWO runs (see runs.py) from the cached
statistics of extract_stats.py: two engines (penelope vs geant4) or two LUT
methods built from the same raw data. Written to <out>/<A>_vs_<B>/.

  raw   (--scope raw|all; raw statistics only, never a LUT; skipped when both
         runs use the same raw directory)
    S1  p_ann overlay, stopping-level energies vs density, B/A shift
    S2  B / A − 1 maps of mean u, mean theta, std theta
    S4  theta and u marginals overlaid (focus material)
    R3  paired violins of theta and u per energy (focus material)
  lut   (--scope lut|all; each run's LUTs against its own raw survivors)
    L2  within-interval centroid offset profiles overlaid
    L3  LUT mean bias maps, A above B
    L4  KS fidelity maps, A above B
    L5  leave-one-out interpolation error, medians overlaid

Both runs must share the energy grid and the material set.

Usage:
    python luts/src/analysis/compare_figures.py RUN_A RUN_B [--labels LA LB]
                                                [--scope raw|lut|all] [--cache DIR] [--out DIR]
"""
import argparse
from pathlib import Path

import numpy as np
from matplotlib.colors import TwoSlopeNorm

import figcommon as fc
from figcommon import DIV, INK, INK2, PAIR_COLOR, plt
from runs import CACHE, OUTPUT, pair_labels, resolve


# ================================================================ raw
def raw_figures(runs, Ds, E, mats, out):
    (A, B), (DA, DB), (CA, CB) = runs, Ds, PAIR_COLOR
    fm = fc.focus_material(mats)

    # S1 p_ann overlay, stopping levels, shift
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    ax = axs[0]; fc.clean(ax, "both")
    for m in mats:
        for D, c in ((DA, CA), (DB, CB)): ax.plot(E, D[m]["p_ann"], color=c, lw=0.8, alpha=0.75)
    for r, c in ((A, CA), (B, CB)): ax.plot([], [], color=c, label=r.label)
    fc.ylog_e(ax); ax.set_xlim(5, 4600); ax.set_ylabel("p_ann within one step")
    ax.set_title(f"a  annihilation probability, {len(mats)} materials (right = densest)", color=INK)
    ax.legend(frameon=False)
    ax = axs[1]; fc.clean(ax, "both")
    fc.levels_vs_density(ax, DA, E, mats, CA)
    fc.levels_vs_density(ax, DB, E, mats, CB, filled=False, marker="s")
    ax.set_title("b  E at p_ann = 0.9 / 0.5 / 0.1 vs density", color=INK)
    ax.legend(handles=[plt.Line2D([], [], color=CA, marker="o", ms=4, label=A.label),
                       plt.Line2D([], [], color=CB, marker="s", mfc="none", ms=5, label=B.label)],
              frameon=False, loc="upper left")
    ax = axs[2]; fc.clean(ax)
    for lev, ls, name in fc.LEVELS:
        ax.plot(mats, [100 * (fc.e_at(DB, m, E, lev) / fc.e_at(DA, m, E, lev) - 1) for m in mats],
                color=INK2, ls=ls, marker="o", ms=3, lw=1, label=name)
    ax.axhline(0, color=INK2, lw=0.6); ax.set_xlabel("material"); ax.set_ylabel(f"{B.label} / {A.label} − 1 (%)")
    ax.set_title("c  stopping-level shift", color=INK); ax.legend(frameon=False)
    fc.save(fig, out, "S1_pann_levels.png")

    # S2 ratio maps
    fig, axs = plt.subplots(1, 3, figsize=(18, 5)); plt.subplots_adjust(wspace=0.18)
    ok = (fc.stack(DA, mats, "n_surv") >= 1000) & (fc.stack(DB, mats, "n_surv") >= 1000)
    for ax, (k, lab) in zip(axs, [("mu", "mean u (ΔE/E₀)"), ("mth", "mean θ"), ("sth", "std θ")]):
        R = np.where(ok, fc.stack(DB, mats, k) / fc.stack(DA, mats, k) - 1, np.nan) * 100
        fc.ematrix(ax, R, E, mats, DIV, TwoSlopeNorm(0, -8, 8),
                   f"{B.label} / {A.label} − 1: {lab}\n(mean |Δ| = {np.nanmean(np.abs(R)):.2f} %)", cb="%", fig=fig)
        if ax is not axs[0]: ax.set_ylabel("")
    fc.save(fig, out, "S2_ratio_maps.png")

    # S4 marginals, focus material
    pick = [int(np.argmin(np.abs(E - x))) for x in (230, 350, 700, 1500, 4000)]
    pick = [i for i in pick if f"keep_{i}" in DA[fm] and f"keep_{i}" in DB[fm]]
    fig, axs = plt.subplots(2, len(pick), figsize=(3.6 * len(pick), 6.4), squeeze=False)
    plt.subplots_adjust(hspace=0.4, wspace=0.25)
    for c, i in enumerate(pick):
        for r, (col, lab) in enumerate([(0, "θ (rad)"), (1, "u = ΔE/E₀")]):
            ax = axs[r, c]; fc.clean(ax, None)
            hi = max(np.percentile(D[fm][f"keep_{i}"][col], 99.9) for D in (DA, DB))
            for D, run, cc in ((DA, A, CA), (DB, B, CB)):
                ax.hist(D[fm][f"keep_{i}"][col], 60, range=(0, hi), histtype="step", density=True,
                        color=cc, lw=1.6, label=run.label)
            ax.set_xlabel(lab); ax.set_yscale("log"); ax.tick_params(axis="y", labelsize=7)
            if r == 0: ax.set_title(f"{E[i]:.0f} keV  (n_surv {DA[fm]['n_surv'][i]/1e3:.0f}k / "
                                    f"{DB[fm]['n_surv'][i]/1e3:.0f}k)", fontsize=9, color=INK)
    axs[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Material {fm}: survivor marginals (log density, x up to the 99.9th percentile)",
                 fontsize=12, fontweight="bold", color=INK, x=0.5, y=0.99)
    fc.save(fig, out, f"S4_marginals_mat{fm:02d}.png")

    # R3 paired violins, focus material
    def subs(D, i, c):
        k = f"sub_{i}"; return D[fm][k][c] if k in D[fm] else np.empty(0)
    live = np.flatnonzero((DA[fm]["n_surv"] >= 2000) & (DB[fm]["n_surv"] >= 2000))
    rows = np.unique(np.round(np.linspace(live[1], live[-1], 8)).astype(int))
    fig, axs = plt.subplots(1, 2, figsize=(17, 4.8)); plt.subplots_adjust(wspace=0.15)
    for ax, (c, lab) in zip(axs, [(0, "θ (rad)"), (1, "u = ΔE/E₀")]):
        fc.clean(ax)
        for k, i in enumerate(rows):
            for D, dx, cc in ((DA, -0.2, CA), (DB, 0.2, CB)):
                x = subs(D, i, c); x = x[x <= np.percentile(x, 99.5)]
                vp = ax.violinplot([x], positions=[k + dx], widths=0.38, showextrema=False, points=200)
                for b in vp["bodies"]: b.set_facecolor(cc); b.set_edgecolor("none"); b.set_alpha(0.75)
                ax.plot([k + dx - 0.15, k + dx + 0.15], [np.median(x)] * 2, color=INK, lw=0.9)
        ax.set_xticks(range(len(rows)), [f"{E[i]:.0f}" for i in rows]); ax.set_xlabel("E₀ (keV)"); ax.set_ylabel(lab)
        if c == 1: ax.set_yscale("log")
        ax.set_title(f"material {fm} ({fc.tissue(fm)}): {lab.split(' ')[0]} per energy, "
                     f"{A.label} left / {B.label} right, bar = median", color=INK)
    fc.save(fig, out, f"R3_violins_mat{fm:02d}.png")


# ================================================================ lut
def lut_figures(runs, Ds, E, mats, out):
    (A, B), (DA, DB), (CA, CB) = runs, Ds, PAIR_COLOR

    # L2 centroid-offset profiles
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8)); plt.subplots_adjust(wspace=0.2)
    for ax, (key, sym) in zip(axs, [("dth", "θ"), ("du", "u")]):
        fc.clean(ax)
        for D, run, c in ((DA, A, CA), (DB, B, CB)):
            fc.centroid_profile(ax, D, mats, key, c, f"{run.label} median, 5–95 %")
        ax.axhline(0, color=INK2, lw=0.8); ax.set_ylim(-0.25, 0.25)
        ax.set_xlabel(f"{sym} interval as quantile level"); ax.set_ylabel("(mean − mid) / width")
        ax.set_title(f"within-{sym}-interval centroid offset, all FULL rows", color=INK)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fc.save(fig, out, "L2_bin_centroid_offset.png")

    # L3 mean bias, L4 KS fidelity: A above B
    for name, fn in (("L3_lut_mean_bias.png", fc.bias_maps), ("L4_ks_grid.png", fc.ks_maps)):
        fig, axs = plt.subplots(2, 2, figsize=(15, 9.5)); plt.subplots_adjust(hspace=0.4, wspace=0.12)
        for r, (D, run) in enumerate(((DA, A), (DB, B))):
            fn(axs[r], D, E, mats, run.label, fig); axs[r, 1].set_ylabel("")
        fc.save(fig, out, name)

    # L5 leave-one-out interpolation error
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.6)); plt.subplots_adjust(wspace=0.18)
    for ax, (k, sym) in zip(axs, [("mth", "θ"), ("mu", "u")]):
        fc.clean(ax)
        for D, run, c in ((DA, A, CA), (DB, B, CB)):
            Q = np.nanpercentile(fc.loo_error(D, mats, k), [25, 50, 75], axis=0)
            ax.fill_between(E, Q[0], Q[2], color=c, alpha=0.18, lw=0)
            ax.plot(E, Q[1], color=c, lw=2, label=f"{run.label}, median (band: 25–75 %)")
        ax.axhline(0, color=INK2, lw=0.8); ax.set_ylim(-6, 6); fc.ylog_e(ax)
        ax.set_ylabel(f"mean {sym}: interpolated / LUT row − 1 (%)")
        ax.set_title(f"row i rebuilt from rows i±1 only, over {len(mats)} materials", color=INK)
        ax.legend(frameon=False, loc="upper right", fontsize=8)
    fc.save(fig, out, "L5_energy_interp.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs=2, metavar="RUN", help="engine name, directory under luts/output, or path")
    ap.add_argument("--labels", nargs=2, default=None, metavar="L", help="names used in titles and legends")
    ap.add_argument("--scope", choices=("raw", "lut", "all"), default="all")
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, default=OUTPUT / "figs", help="figures go to <out>/<A engine>_vs_<B engine>/<set>/ (<out>/<A>_vs_<B>/ otherwise)")
    a = ap.parse_args(argv)
    A, B = (resolve(r) for r in a.runs)
    A.label, B.label = a.labels or pair_labels(A, B)
    if A.materials != B.materials:
        raise SystemExit(f"runs differ in materials: {A.materials} vs {B.materials}")
    parts = ("raw",) if a.scope == "raw" else ("raw", "lut")
    DA, DB = fc.load(A, parts, a.cache), fc.load(B, parts, a.cache)
    E = fc.energies(DA)
    if not np.allclose(E, fc.energies(DB), rtol=1e-6):
        raise SystemExit("runs differ in their energy grid")
    same_set = A.name == B.name and A.fig_dir.count("/") == 1 and B.fig_dir.count("/") == 1
    out = a.out / (f"{A.engine}_vs_{B.engine}/{A.name}" if same_set else f"{A.key}_vs_{B.key}")
    if a.scope in ("raw", "all"):
        if A.raw_key == B.raw_key:
            print(f"same raw data ({A.raw_dir}): raw comparison skipped")
        else:
            raw_figures((A, B), (DA, DB), E, A.materials, out)
    if a.scope in ("lut", "all"):
        lut_figures((A, B), (DA, DB), E, A.materials, out)


if __name__ == "__main__":
    main()
