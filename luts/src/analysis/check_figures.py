"""
check_figures.py -- figures for ONE run (see runs.py) from the cached
statistics of extract_stats.py. Written to <out>/<run>/.

  raw   (--scope raw|all; raw statistics only, never a LUT)
    S1  p_ann vs E for every material; stopping-level energies vs density
    S3  median theta, median u and Spearman rho(theta, u) vs E
    R1  ridgelines of theta and u by material at ~510 keV
    R2  per-energy density columns with 10/50/90 % quantiles (focus material)
  lut   (--scope lut|all; the LUTs, and the LUTs against their raw survivors)
    L0  LUT contents: theta quantile functions and the u|theta knot map
    L1  row validity (gap census)
    L2  within-interval centroid offset
    L3  LUT mean bias
    L4  KS fidelity of LUT draws
    L5  log-E interpolation error (leave-one-out)
    L6  theta-u rank correlation, raw vs LUT
    L7  joint (theta, u), raw vs LUT draws (focus material)

Usage:
    python luts/src/analysis/check_figures.py RUN [--label L] [--scope raw|lut|all]
                                              [--lut-material M] [--cache DIR] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, TwoSlopeNorm
from matplotlib.ticker import NullFormatter
from matplotlib.patches import Patch

import figcommon as fc
from figcommon import BG, DIV, INK, INK2, plt
from runs import CACHE, OUTPUT, resolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lutgen"))
import lut_contract  # noqa: E402


# ================================================================ raw
def raw_figures(run, D, E, mats, out):
    col, lab = fc.run_color(run), run.label
    fm = fc.focus_material(mats)

    # S1 annihilation probability and stopping levels
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axs[0]; fc.clean(ax, "both")
    for m in mats: ax.plot(E, D[m]["p_ann"], color=fc.mat_color(mats, m), lw=0.9)
    fc.ylog_e(ax); ax.set_xlim(5, 4600); ax.set_ylabel("p_ann within one step")
    ax.set_title(f"a  {lab}: annihilation probability per material (light = densest)", color=INK)
    ax = axs[1]; fc.clean(ax, "both")
    fc.levels_vs_density(ax, D, E, mats, col)
    for lev, _, name in fc.LEVELS:
        ax.text(fc.density(mats[-1]) * 1.12, fc.e_at(D, mats[-1], E, lev), name, fontsize=8, va="center", color=INK2)
    ax.set_title("b  E at p_ann = 0.9 / 0.5 / 0.1 vs density", color=INK)
    fc.save(fig, out, "S1_pann_levels.png")

    # S3 median theta, median u, rho vs E
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.4)); plt.subplots_adjust(wspace=0.22)
    for ax, (k, ylab, nmin) in zip(axs, [("qth", "median θ (rad)", 100), ("qu", "median u = ΔE/E₀", 100),
                                         ("rho", f"Spearman ρ(θ, u), n_surv ≥ {fc.NMIN_RHO}", fc.NMIN_RHO)]):
        fc.clean(ax)
        for m in mats:
            y = D[m][k][:, 1] if k != "rho" else D[m][k]
            ax.plot(E, np.where(D[m]["n_surv"] >= nmin, y, np.nan), color=fc.mat_color(mats, m), lw=0.9)
        fc.ylog_e(ax); ax.set_xlim(10, 4600); ax.set_ylabel(ylab)
        if k == "qu": ax.set_yscale("log")
        if k == "rho": ax.axhline(0, color=INK2, lw=0.5); ax.set_ylim(-0.05, 0.2)
        ax.set_title(f"{lab}: {ylab.split(', n_surv')[0]}", color=INK)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(mats[0], mats[-1]))
    fig.colorbar(sm, ax=axs, fraction=0.015, pad=0.01, label="material (density increases)")
    fc.save(fig, out, "S3_moments_vs_E.png")

    def subs(m, i, c):
        k = f"sub_{i}"; return D[m][k][c] if k in D[m] else np.empty(0)

    # R1 ridgelines by material at ~510 keV
    ir = int(np.argmin(np.abs(E - 510)))
    fig, axs = plt.subplots(1, 2, figsize=(11, 7.5), sharey=True); plt.subplots_adjust(wspace=0.08)
    for ax, (c, xl) in zip(axs, [(0, "θ (rad)"), (1, "u = ΔE/E₀")]):
        live = [m for m in mats if subs(m, ir, c).size >= 1000]
        xmax = max(np.percentile(subs(m, ir, c), 99.5) for m in live)
        bins = np.linspace(0, xmax, 70); ctr = 0.5 * (bins[1:] + bins[:-1])
        for k in range(len(mats) - 1, -1, -1):        # back to front
            m = mats[k]; x = subs(m, ir, c)
            if x.size < 1000: continue
            h, _ = np.histogram(x, bins=bins, density=True); y = k + 1 + 1.6 * h / h.max()
            ax.fill_between(ctr, k + 1, y, color=fc.mat_color(mats, m), alpha=0.85, lw=0, zorder=len(mats) - k)
            ax.plot(ctr, y, color="k", lw=0.3, zorder=len(mats) - k)
        ax.set_xlim(0, xmax); ax.set_ylim(0.5, len(mats) + 2.5); ax.set_xlabel(xl)
        ax.set_title(f"{lab}: {xl.split(' ')[0]} at {E[ir]:.0f} keV", color=INK)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    tick = [k for k, m in enumerate(mats) if m in (1, 2, 9, 12, 15, 16, 30)] or list(range(0, len(mats), 5))
    axs[0].set_yticks([k + 1 for k in tick], [f"{mats[k]}  {fc.tissue(mats[k])}" for k in tick])
    axs[0].set_ylabel("material")
    fc.save(fig, out, "R1_ridges_510keV.png")

    # R2 per-energy density columns, focus material
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.6)); plt.subplots_adjust(wspace=0.18)
    le = np.log(E)
    ed = np.exp(np.concatenate([[le[0] - (le[1] - le[0]) / 2], 0.5 * (le[1:] + le[:-1]), [le[-1] + (le[-1] - le[-2]) / 2]]))
    live = [i for i in range(E.size) if D[fm]["n_surv"][i] >= 1000]
    for ax, (c, qk, yl) in zip(axs, [(0, "qth", "θ (rad)"), (1, "qu", "u = ΔE/E₀")]):
        allx = np.concatenate([subs(fm, i, c) for i in live])
        yb = (np.linspace(0, np.percentile(allx, 99.9), 61) if c == 0
              else np.logspace(np.log10(np.percentile(allx, 0.1)), np.log10(np.percentile(allx, 99.9)), 61))
        H = np.full((60, E.size), np.nan)
        for i in live:
            h, _ = np.histogram(subs(fm, i, c), bins=yb, density=True)
            hh = h / h.max(); hh[hh == 0] = np.nan; H[:, i] = hh
        cm = plt.get_cmap(fc.run_cmap(run)).copy(); cm.set_bad(BG)
        im = ax.pcolormesh(ed, yb, np.ma.masked_invalid(H), cmap=cm, vmin=0, vmax=1, rasterized=True)
        q = np.where((D[fm]["n_surv"] >= 1000)[:, None], D[fm][qk], np.nan)
        for j, ls in enumerate([":", "-", "--"]): ax.plot(E, q[:, j], color=INK, lw=1.0, ls=ls)
        ax.set_xscale("log"); ax.set_xlim(ed[live[0]] * 0.9, ed[-1])
        if c == 1: ax.set_yscale("log")
        tk = [t for t in (200, 500, 1000, 2000, 4600) if t >= ed[live[0]]]
        ax.set_xticks(tk, [str(t) for t in tk]); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("E₀ (keV)"); ax.set_ylabel(yl)
        ax.set_title(f"{lab}, material {fm} ({fc.tissue(fm)}): {yl.split(' ')[0]} vs E₀", color=INK)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="density / column max").set_ticks([0, 0.5, 1])
    axs[0].legend(handles=[plt.Line2D([], [], color=INK, ls=ls, label=l) for ls, l in [(":", "10 %"), ("-", "50 %"), ("--", "90 %")]],
                  frameon=False, title="quantiles (full sample)", fontsize=8, title_fontsize=8, loc="upper right")
    fc.save(fig, out, f"R2_columns_mat{fm:02d}.png")


# ================================================================ lut
def lut_figures(run, D, E, mats, out, lut_material):
    col, lab = fc.run_color(run), run.label
    fm = fc.focus_material(mats)

    # L0 LUT contents
    lm = lut_material if lut_material in mats else fm
    L = lut_contract.select(lut_contract.read_luts(run.ann, run.scatt), run.slot(lm))
    ek = L.e_grid / 1e3; live = np.flatnonzero(L.valid[0] != 0)
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    plt.subplots_adjust(wspace=0.2)
    ax = axs[0]; fc.clean(ax, "both")
    for k, i in enumerate(live[np.linspace(0, live.size - 1, 6).astype(int)]):
        ax.plot(np.linspace(0, 1, L.n_theta), L.theta_icdf[0, i], lw=1.4, color=plt.cm.viridis(k / 5),
                label=f"{ek[i]:.0f} keV")
    ax.set_xlabel("probability level q"); ax.set_ylabel("θ knot (rad)")
    ax.set_title(f"a  {lab}, mat {lm}: θ quantile function per energy", color=INK)
    ax.legend(frameon=False, fontsize=8, title="E₀", title_fontsize=8)
    full = np.flatnonzero(L.valid[0] == 1)
    i = int(full[np.argmin(np.abs(ek[full] - 600))])
    U = L.u_icdf[0, i].T
    im = axs[1].imshow(U, origin="lower", aspect="auto", extent=[0, 1, 0, 1],
                       cmap=fc.run_cmap(run),
                       norm=LogNorm(U[U > 0].min(), U.max()))
    fig.colorbar(im, ax=axs[1], label="u = ΔE/E₀ knot (log scale)", fraction=0.05, pad=0.02)
    axs[1].set_xlabel("θ bin b (as θ-quantile level)"); axs[1].set_ylabel("u probability level")
    axs[1].set_title(f"b  {lab}, mat {lm}: u | θ knots at {ek[i]:.0f} keV (n_surv = {int(L.n_surv[0, i])})", color=INK)
    fc.save(fig, out, f"L0_lut_contents_mat{lm:02d}.png")

    # L1 row trust map
    C = pd.read_csv(run.census)
    C = C[C.material.isin(mats)]
    nb = L.n_theta - 1
    C["c"] = np.select([C.valid_code == 0, C.valid_code == 3, C.valid_code == 2, C.n_surv / nb >= 100], [0, 4, 3, 1], 2)
    COL = ["#e6e5e1", "#0ca30c", "#fab219", "#ec835a", "#d03b3b"]
    LAB = ["DEAD (p_ann = 1)", "FULL, ≥100 survivors / θ bin", "FULL, thin (<100 / θ bin)",
           "POOLED (neighbour energies)", "SPARSE"]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    M = C.pivot(index="material", columns="energy_index", values="c").loc[mats].values
    fc.ematrix(ax, M, E, mats, ListedColormap(COL), BoundaryNorm(np.arange(-0.5, 5), 5), "")
    ax.set_xlim(-0.5, E.size - 0.5)
    n = C.c.value_counts()
    ax.set_title(f"{lab}: {n.get(1, 0)} FULL · {n.get(2, 0)} thin · {n.get(3, 0)} POOLED · "
                 f"{n.get(4, 0)} SPARSE · {n.get(0, 0)} DEAD", color=INK)
    fig.legend(handles=[Patch(facecolor=c, edgecolor=INK2, lw=0.4, label=l) for c, l in zip(COL, LAB)],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fc.save(fig, out, "L1_row_trust.png")

    # L2 within-interval centroid offset
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), gridspec_kw={"width_ratios": [1.1, 1]})
    plt.subplots_adjust(hspace=0.35, wspace=0.4)
    for r, (key, sym, ylab) in enumerate([("dth", "θ", "θ bin b (θ-quantile level)"),
                                          ("du", "u", "u interval k (u-quantile level, averaged over θ bins)")]):
        ax = axs[r, 0]; M = D[fm][key]
        im = ax.pcolormesh(np.arange(E.size + 1) - 0.5, np.linspace(0, 1, M.shape[1] + 1), np.ma.masked_invalid(M.T),
                           cmap=DIV, norm=TwoSlopeNorm(0, -0.15, 0.15), rasterized=True)
        fc.e_ticks(ax, E); lv = np.flatnonzero(~np.isnan(M).all(1)); ax.set_xlim(lv[0] - 0.5, E.size - 0.5)
        ax.set_xlabel("E₀ (keV)"); ax.set_ylabel(ylab)
        ax.set_title(f"{'ac'[r]}  {lab}, mat {fm}, per {sym} interval", color=INK)
        fig.colorbar(im, ax=ax, label="offset / interval width", fraction=0.05, pad=0.02)
        ax = axs[r, 1]; fc.clean(ax)
        fc.centroid_profile(ax, D, mats, key, col, f"{lab} median, 5–95 %")
        ax.axhline(0, color=INK2, lw=0.8); ax.set_ylim(-0.25, 0.25)
        ax.set_xlabel(f"{sym} interval as quantile level"); ax.set_ylabel("(mean − mid) / width")
        ax.set_title(f"{'bd'[r]}  all FULL rows, {len(mats)} materials", color=INK)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fc.save(fig, out, "L2_bin_centroid_offset.png")

    # L3 mean bias, L4 KS fidelity
    for name, fn in (("L3_lut_mean_bias.png", fc.bias_maps), ("L4_ks_grid.png", fc.ks_maps)):
        fig, axs = plt.subplots(1, 2, figsize=(15, 4.8)); plt.subplots_adjust(wspace=0.12)
        fn(axs, D, E, mats, lab, fig); axs[1].set_ylabel("")
        fc.save(fig, out, name)

    # L5 leave-one-out energy interpolation
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.6)); plt.subplots_adjust(wspace=0.18)
    for ax, (k, sym) in zip(axs, [("mth", "θ"), ("mu", "u")]):
        fc.clean(ax); A = fc.loo_error(D, mats, k)
        for j, m in enumerate(mats): ax.plot(E, A[j], color=plt.cm.Greys(0.25 + 0.7 * (j + 1) / len(mats)), lw=0.7)
        ax.plot(E, np.nanmedian(A, 0), color=col, lw=2.2, label="median over materials")
        ax.axhline(0, color=INK2, lw=0.8); ax.set_ylim(-6, 6); fc.ylog_e(ax)
        ax.set_ylabel(f"mean {sym}: interpolated / LUT row − 1 (%)")
        ax.set_title(f"{lab}: row i rebuilt from rows i±1 only (2× coarser grid)", color=INK)
        ax.legend(frameon=False, loc="upper right", fontsize=8)
    fc.save(fig, out, "L5_energy_interp.png")

    # L6 theta-u correlation, raw vs LUT
    fig, ax = plt.subplots(figsize=(8, 4.6)); fc.clean(ax)
    for m, ls in zip([m for m in (1, 13, 30) if m in mats] or [fm], [":", "-", "--"]):
        ax.plot(E, D[m]["rho"], color=INK2, ls=ls, lw=1.4, label=f"raw, mat {m}")
        ax.plot(E, D[m]["rho_lut"], color=col, ls=ls, lw=1.8, label=f"LUT, mat {m}")
    fc.ylog_e(ax); ax.set_ylabel("Spearman ρ(θ, u)"); ax.set_title(f"{lab}: θ–u rank correlation", color=INK)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fc.save(fig, out, "L6_correlation.png")

    # L7 joint (theta, u), raw vs LUT draws
    pick = [int(np.argmin(np.abs(E - x))) for x in (250, 600, 3000)]
    pick = [i for i in pick if f"keep_{i}" in D[fm] and f"lut_{i}" in D[fm]]
    fig, axs = plt.subplots(2, len(pick), figsize=(4.2 * len(pick), 7), squeeze=False)
    plt.subplots_adjust(wspace=0.15, hspace=0.35)
    for c, i in enumerate(pick):
        raw, lut = D[fm][f"keep_{i}"], D[fm][f"lut_{i}"]
        rng_ = [[0, np.nanpercentile(raw[0], 99.5)], [np.nanpercentile(raw[1], 0.2), np.nanpercentile(raw[1], 99.5)]]
        h1, xe, ye = np.histogram2d(raw[0], raw[1], 50, rng_); h2, _, _ = np.histogram2d(lut[0], lut[1], 50, rng_)
        vmax = max(h1.max(), h2.max())
        for r, (h, tag) in enumerate([(h1, "raw"), (h2, "LUT")]):
            ax = axs[r, c]
            ax.pcolormesh(xe, ye, h.T, cmap=fc.run_cmap(run), vmin=0, vmax=vmax)
            ax.set_title(f"{E[i]:.0f} keV, {tag}", fontsize=9, color=INK)
            ax.set_xlabel("θ (rad)", fontsize=8)
            if c == 0: ax.set_ylabel("u = ΔE/E₀", fontsize=8)
    fig.suptitle(f"{lab}, material {fm}: joint (θ, u) of survivors, raw vs 2·10⁴ draws through the sampling contract",
                 fontsize=12, fontweight="bold", color=INK, x=0.5, y=1.0)
    fc.save(fig, out, "L7_joint_hist.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="engine name, directory under luts/output, or path")
    ap.add_argument("--label", default=None, help="name used in titles (default: from run.json)")
    ap.add_argument("--scope", choices=("raw", "lut", "all"), default="all")
    ap.add_argument("--lut-material", type=int, default=13, help="material shown in L0 (LUT contents)")
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, default=OUTPUT / "figs", help="figures go to <out>/<engine>/<set>/ (<out>/<run key>/ elsewhere)")
    a = ap.parse_args(argv)
    run = resolve(a.run, a.label)
    parts = ("raw",) if a.scope == "raw" else ("raw", "lut")
    D = fc.load(run, parts, a.cache)
    E, mats, out = fc.energies(D), run.materials, a.out / run.fig_dir
    if a.scope in ("raw", "all"): raw_figures(run, D, E, mats, out)
    if a.scope in ("lut", "all"): lut_figures(run, D, E, mats, out, a.lut_material)


if __name__ == "__main__":
    main()
