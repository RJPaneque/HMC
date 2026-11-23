#!/usr/bin/env python3
"""plot_results.py -- the validation figures from the tables of results/, an example of reading them.

    python validation/tools/plot_results.py [--results validation/results] [--out validation/figures] [--fonts DIR]

  fig1_profiles.png         the annihilation profile along x of four cases (Tiers 1 to 4), HMC against the ten reference codes, with the ratio
                            to PENELOPE underneath                                                   (results/profiles.csv)
  fig2_hmc_vs_engine.png    mean radius, per tier: HMC against the code that made its tables (violins, white dot = median, bar = quartiles),
                            and the other engines against PENELOPE for scale (grey)                  (results/qoi.csv)
  fig3_hmc_in_band.png      is HMC inside the band of the ten reference codes? share of the radial quantities (r50, mean, r90, r99.9)
                            inside the band, per HMC variant and tier                                (results/qoi.csv, envelope.csv)
  fig4_interfaces.png       Tiers 2-4: the fraction of the annihilations that crossed the interface (or got past the lamella), HMC variants
                            against the band of the references, per case                             (results/interfaces.csv, envelope.csv)
  fig5_code_distances.png   the 11 codes (no GPU ports) against each other: median Kolmogorov-Smirnov distance of the radial distributions (in units of its
                            5 % critical value) and median difference of the mean radius, over the 305 cases   (results/pairs.csv)
  fig6_speed.png            wall time per history of one run, Tiers 2-4                              (results/qoi.csv)

The style is that of the NSS/MIC 2026 poster of HMC: Carlito or Calibri if installed (DejaVu Sans otherwise; the PNGs shipped were made with Carlito) on white,
1.8 pt axes without top and right spines, one colour per engine family (PENELOPE blue, EGSnrc green, Geant4 orange, FLUKA purple),
HMC itself black, grey for the spread between engines. The numbers behind every figure are the CSV files.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
REFERENCES = ["pvoxel", "peneasy", "gate", "g4app", "egsnrc", "gpen", "gegs", "gg4", "penred", "fluka"]
# table family: (key, HMC variant, the code that made its tables, label, colour); the colours of the poster
FAMILIES = [("pen", "hmc_pen_lad", "pvoxel", "PENELOPE", "#2a78d6"), ("egs", "hmc_egs_lad", "egsnrc", "EGSnrc", "#1baf7a"),
            ("g4", "hmc_g4_lad", "g4app", "Geant4", "#eb6834"), ("fluka", "hmc_fl_lad", "fluka", "FLUKA", "#8a5cc2")]
GREY, INK, GRID = "#9aa3ad", "#0b0b0b", "#d9d9d9"
RADIAL = ("r50", "rmean", "r90", "r999")
TIER_NAMES = {"1": "Tier 1\npoint sources", "2": "Tier 2\nslabs", "3": "Tier 3\nramps", "4": "Tier 4\nlamellae"}


def use_style(base: float = 12.0, fonts: Path | None = None):
    """The poster's font and axes (poster_style.use_fonts and prettify of the development repository), at a size for a page.
    `fonts`: a folder of .ttf files to register (the shipped PNGs were drawn with Carlito from the development repository)."""
    for ttf in sorted(fonts.glob("*.ttf")) if fonts else []:
        font_manager.fontManager.addfont(str(ttf))
    have = {fe.name for fe in font_manager.fontManager.ttflist}
    fam = [n for n in ("Calibri", "Carlito") if n in have] + ["DejaVu Sans"]
    plt.rcParams.update({"font.family": fam, "savefig.dpi": 200, "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "font.size": base, "axes.labelsize": base + 1, "xtick.labelsize": base - 1,
                         "ytick.labelsize": base - 1, "legend.fontsize": base, "text.color": INK, "axes.labelcolor": INK,
                         "mathtext.fontset": "custom", "mathtext.cal": fam[0], "mathtext.rm": fam[0], "mathtext.it": f"{fam[0]}:italic", "mathtext.bf": f"{fam[0]}:bold"})


def prettify(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.spines["left"].set_linewidth(1.8)
    ax.tick_params(width=1.8)


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load(results: Path):
    cases = {r["id"]: r for r in read(results / "cases.csv")}
    qoi = defaultdict(dict)                       # case -> code -> row
    for r in read(results / "qoi.csv"):
        qoi[r["case"]][r["code"]] = r
    env = {(r["case"], r["qoi"]): r for r in read(results / "envelope.csv")}
    dens = {int(r["id"]): float(r["density_g_cm3"]) for r in read(results.parent / "materials.csv")}
    return cases, qoi, env, dens


def load_profiles(results: Path):
    """results/profiles.csv -> {case: {code: {"x": mm, "n": counts, "f": fraction per mm}}}"""
    tmp = defaultdict(lambda: defaultdict(lambda: {"x": [], "n": [], "f": []}))
    for r in read(results / "profiles.csv"):
        d = tmp[r["case"]][r["code"]]
        d["x"].append(float(r["x_mm"]))
        d["n"].append(float(r["counts"]))
        d["f"].append(float(r["fraction_per_mm"]))
    return {c: {k: {kk: np.array(vv) for kk, vv in d.items()} for k, d in codes.items()} for c, codes in tmp.items()}


FAMILY_OF = {"pvoxel": "pen", "peneasy": "pen", "gpen": "pen", "penred": "pen", "hmc_pen_lad": "pen",
             "g4app": "g4", "gate": "g4", "gg4": "g4", "hmc_g4_lad": "g4",
             "egsnrc": "egs", "gegs": "egs", "hmc_egs_lad": "egs", "fluka": "fluka", "hmc_fl_lad": "fluka"}
COLOUR = {f[0]: f[4] for f in FAMILIES}
ORDER = ["pvoxel", "peneasy", "penred", "hmc_pen_lad", "g4app", "gate", "hmc_g4_lad", "egsnrc", "hmc_egs_lad", "fluka", "hmc_fl_lad"]   # fig5: the GPU ports are left out
LABEL = {"pvoxel": "PENELOPE", "peneasy": "PenEasy", "gpen": "gpen", "penred": "PenRed", "hmc_pen_lad": "HMC (PENELOPE tables)",
         "g4app": "Geant4", "gate": "GATE", "gg4": "gg4", "hmc_g4_lad": "HMC (Geant4 tables)", "egsnrc": "EGSnrc", "gegs": "gegs",
         "hmc_egs_lad": "HMC (EGSnrc tables)", "fluka": "FLUKA", "hmc_fl_lad": "HMC (FLUKA tables)"}


def fig1(cases, profiles, dens, out):
    """The profile along x (summed over y and z) of one case per tier: the ten reference codes (grey; PENELOPE, our voxel program `pvoxel`, in blue)
    and HMC with the PENELOPE tables (black); underneath the ratio to PENELOPE, shaded at one Poisson sigma of the two profiles. Bins with fewer
    than 400 counts in PENELOPE are left out of the ratio."""
    ids = list(profiles)
    fig, axes = plt.subplots(2, len(ids), figsize=(14, 7.0), sharex="col", gridspec_kw={"height_ratios": [2, 1.6]})
    for j, cid in enumerate(ids):
        top, bot = axes[0, j], axes[1, j]
        prettify(top)
        prettify(bot)
        bot.grid(axis="x", visible=False)
        prof = profiles[cid]
        x = prof["pvoxel"]["x"]
        dx = x[1] - x[0]
        geo = json.loads(cases[cid]["geometry"])
        if "interface_index" in geo and "ramp_index_range" not in geo:
            for ax in (top, bot):
                ax.axvline(x[geo["interface_index"]] - dx / 2, color=INK, lw=1, ls=(0, (4, 3)))
        for key in ("ramp_index_range", "lamella_index_range"):
            if key in geo:
                a, b = geo[key]
                for ax in (top, bot):
                    ax.axvspan(x[a] - dx / 2, x[b - 1] + dx / 2, color="#e6e9ee", lw=0, zorder=0)
        for k in REFERENCES:
            if k != "pvoxel":
                top.plot(x, np.maximum(prof[k]["f"], 1e-9), color=GREY, lw=1.1, alpha=0.8, zorder=2)
        top.plot(x, np.maximum(prof["pvoxel"]["f"], 1e-9), color=COLOUR["pen"], lw=2.0, zorder=3)
        top.plot(x, np.maximum(prof["hmc_pen_lad"]["f"], 1e-9), color=INK, lw=1.5, zorder=4)
        top.set_yscale("log")
        top.set_ylim(top.get_ylim()[1] * 1e-4, top.get_ylim()[1] * 1.5)
        ref = prof["pvoxel"]
        ok = ref["n"] >= 400
        hm = prof["hmc_pen_lad"]
        ratio = np.where(ok, hm["f"] / np.where(ok, ref["f"], 1), np.nan)
        sig = np.where(ok, ratio * np.sqrt(1 / np.where(ok, ref["n"], 1) + 1 / np.maximum(hm["n"], 1)), np.nan)
        bot.axhline(1, color=INK, lw=1)
        for k in REFERENCES:
            if k != "pvoxel":
                r = np.where(ok, prof[k]["f"] / np.where(ok, ref["f"], 1), np.nan)
                bot.plot(x, r, color=GREY, lw=1.0, alpha=0.8, zorder=2)
        bot.fill_between(x, ratio - sig, ratio + sig, color=INK, alpha=0.18, lw=0, zorder=3)
        bot.plot(x, ratio, color=INK, lw=1.4, zorder=4)
        bot.set_ylim(0.8, 1.2)
        bot.set_yticks([0.8, 1.0, 1.2])
        bot.set_xlabel("x from the source (mm)")
        r = cases[cid]
        d = lambda m: f"{dens[int(m)]:.2f}"
        t = r["tier"]
        if t == "1":
            what = f"homogeneous {d(r['materials'])} g/cm³"
        elif t in ("2", "3"):
            what = ("slab" if t == "2" else "2 mm ramp") + f" {d(geo['mat_left'])}|{d(geo['mat_right'])} g/cm³"
        else:
            what = f"1.5 mm lamella {d(geo['mat_lamella'])} in {d(geo['mat_host'])} g/cm³"
        top.set_title(f"Tier {r['tier']}, {r['isotope']}\n{what}", loc="left", fontsize=11.5, linespacing=1.4)
    axes[0, 0].set_ylabel("fraction of annihilations per mm")
    axes[1, 0].set_ylabel("ratio to PENELOPE")
    hs = [plt.Line2D([], [], color=GREY, lw=1.6, label="the other 9 reference codes"),
          plt.Line2D([], [], color=COLOUR["pen"], lw=2.2, label="PENELOPE"),
          plt.Line2D([], [], color=INK, lw=1.8, label="HMC, PENELOPE tables")]
    fig.legend(handles=hs, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=2.2)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.835, bottom=0.085, wspace=0.18, hspace=0.08)
    fig.savefig(out)
    plt.close(fig)


def offsets_vs_engine(cases, qoi):
    """% offset of the mean radius: HMC against the code of its tables, and each other engine against PENELOPE. A cloud with a mean radius
    below half a voxel has no meaningful relative offset and is left out."""
    hmc = {f[0]: [] for f in FAMILIES}
    eng = []
    for c, r in cases.items():
        d = qoi[c]
        half_vox = 0.5 * float(r["voxel_mm"].split("x")[0])
        for fam, variant, parent, *_ in FAMILIES:
            p, v = float(d[parent]["rmean_mm"]), float(d[variant]["rmean_mm"])
            if p >= half_vox:
                hmc[fam].append((r["tier"], 100 * (v / p - 1)))
        pen = float(d["pvoxel"]["rmean_mm"])
        if pen >= half_vox:
            for _, _, parent, *_ in FAMILIES[1:]:
                eng.append((r["tier"], 100 * (float(d[parent]["rmean_mm"]) / pen - 1)))
    return hmc, eng


def fig2(cases, qoi, out):
    hmc, eng = offsets_vs_engine(cases, qoi)
    tiers = ["1", "2", "3", "4"]
    lim = 2.0
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    prettify(ax)
    ax.grid(axis="x", visible=False)
    series = [("eng", GREY)] + [(f[0], f[4]) for f in FAMILIES]
    ncase = {}
    for i, t in enumerate(tiers):
        if i % 2:
            ax.axvspan(i - 0.5, i + 0.5, color="#f2f4f7", lw=0, zorder=0)
        for j, (key, col) in enumerate(series):
            vals = [v for tt, v in (eng if key == "eng" else hmc[key]) if tt == t]
            if len(vals) < 2:
                continue
            x = i - 0.32 + j * 0.16
            vp = ax.violinplot(np.clip(vals, -lim, lim), positions=[x], widths=0.15, showextrema=False, bw_method=0.4)
            for b in vp["bodies"]:
                b.set_facecolor(col); b.set_edgecolor(col); b.set_alpha(0.55 if key == "eng" else 0.8); b.set_lw(0.8)
            q25, q50, q75 = np.percentile(vals, [25, 50, 75])
            ax.plot([x, x], [max(q25, -lim), min(q75, lim)], color="#222", lw=2.2, zorder=3, solid_capstyle="butt")
            ax.scatter([x], [np.clip(q50, -lim, lim)], s=34, color="white", edgecolor="#222", lw=0.9, zorder=4)
        ncase[t] = sum(1 for tt, _ in hmc["pen"] if tt == t)
    ax.axhline(0, color=INK, lw=1, zorder=1)
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([f"{TIER_NAMES[t]}\n{ncase[t]} cases" for t in tiers])
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_xlim(-0.5, len(tiers) - 0.5)
    ax.set_ylim(-lim, lim)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels(["−2", "−1", "0", "+1", "+2"])
    ax.set_ylabel("offset of the mean radius (%)")
    hs = [Patch(color=GREY, alpha=0.55, label="other engines vs PENELOPE"), plt.Line2D([], [], color="none", label="HMC vs the code of its tables:")]
    hs += [Patch(color=f[4], alpha=0.8, label=f[3]) for f in FAMILIES]
    leg = fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=6, frameon=False, columnspacing=1.4, handlelength=1.2, handletextpad=0.5)
    leg.get_texts()[1].set_fontweight("bold")
    leg.legend_handles[1].set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.22)
    fig.savefig(out)
    plt.close(fig)


def fig3(cases, qoi, env, out):
    """Share of the radial quantities (r50, mean, r90, r99.9) inside the band of the references, per HMC variant and tier: a table whose
    cells are tinted in the colour of the variant, darker the higher the share."""
    tiers = ["1", "2", "3", "4"]
    inb = {(f[1], t): [0, 0] for f in FAMILIES for t in tiers}
    for (c, k), e in env.items():
        t = cases[c]["tier"]
        if k not in RADIAL or t not in tiers:
            continue
        for _, variant, *_ in FAMILIES:
            r = qoi[c].get(variant)
            if r:
                v = float(r[f"{k}_mm"])
                lo, hi, tol = float(e["lo_mm"]), float(e["hi_mm"]), float(e["tol_mm"])
                inb[(variant, t)][0] += lo - tol <= v <= hi + tol
                inb[(variant, t)][1] += 1
    fig, ax = plt.subplots(figsize=(12.5, 4.9))
    ax.set_xlim(-1.35, len(tiers) - 0.5)
    ax.set_ylim(len(FAMILIES) - 0.5, -1.75)
    ax.text(-1.3, -1.55, "Is HMC inside the band of the ten reference codes?", ha="left", va="center", fontsize=16, fontweight="bold")
    ax.text(-1.3, -1.18, "Share of the radial quantities (r50, mean radius, r90, r99.9) of every case that lie inside the band of the references.\n"
            "A low share for the EGSnrc or Geant4 tables is the behaviour of those codes near interfaces, which HMC reproduces (fig2, fig4).",
            ha="left", va="center", fontsize=11.5, color="#333", linespacing=1.5)
    ax.axis("off")
    for j, t in enumerate(tiers):
        ax.text(j, -0.62, TIER_NAMES[t].replace("\n", "  "), ha="center", va="center", fontsize=13)
    for i, (_, variant, _, label, col) in enumerate(FAMILIES):
        ax.add_patch(plt.Rectangle((-1.3, i - 0.42), 0.07, 0.84, color=col, lw=0))
        ax.text(-1.14, i, label, ha="left", va="center", fontsize=15, color=INK)
        for j, t in enumerate(tiers):
            n, m = inb[(variant, t)]
            share = n / m
            depth = float(np.clip((share - 0.7) / 0.3, 0, 1))
            ax.add_patch(plt.Rectangle((j - 0.46, i - 0.42), 0.92, 0.84, facecolor=col, alpha=0.12 + 0.78 * depth, lw=0))
            ax.text(j, i - 0.07, f"{100 * share:.0f} %" if share in (0, 1) or abs(100 * share - round(100 * share)) < 0.05 else f"{100 * share:.1f} %",
                    ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
            ax.text(j, i + 0.2, f"{n} of {m}", ha="center", va="center", fontsize=10.5, color="#222")
    ax.text(-1.3, -0.62, "HMC, tables of", ha="left", va="center", fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    fig.savefig(out)
    plt.close(fig)
    return {k: f"{v[0]}/{v[1]}" for k, v in inb.items()}


def fig4(cases, interfaces, env, out):
    """Tiers 2-4: the fraction of the annihilations that crossed the interface (slab, ramp: onto the side without the source) or got past the
    lamella. Per case, as the offset from the mean of the references (%): grey bar = their band [lo - tol, hi + tol] (envelope.csv), dots = the
    four HMC variants. The y axis is linear within +-2 % and logarithmic beyond."""
    qname = {"2": None, "3": None, "4": "iface_beyond_fraction"}
    titles = {"2": "Tier 2: slabs", "3": "Tier 3: density ramps", "4": "Tier 4: thin lamellae"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6), sharey=True, gridspec_kw={"width_ratios": [12, 18, 12]})
    for ax, t in zip(axes, ("2", "3", "4")):
        prettify(ax)
        ax.axhline(0, color=INK, lw=1, zorder=1)
        items = []
        for cid in (c for c, r in cases.items() if r["tier"] == t):
            geo = json.loads(cases[cid]["geometry"])
            q = qname[t] or ("iface_right_fraction" if geo["source_side"] == "L" else "iface_left_fraction")
            items.append((float(env[(cid, q)]["mean_mm"]), cid, env[(cid, q)]))
        items.sort()
        for i, (mean, cid, e) in enumerate(items):
            lo, hi, tol = float(e["lo_mm"]), float(e["hi_mm"]), float(e["tol_mm"])
            ax.bar(i, 100 * ((hi + tol) / mean - 1) - 100 * ((lo - tol) / mean - 1), bottom=100 * ((lo - tol) / mean - 1), width=0.8, color=GREY, alpha=0.5, lw=0, zorder=2)
            for j, (_, variant, _, _, col) in enumerate(FAMILIES):
                ax.scatter(i + 0.2 * (j - 1.5), 100 * (float(interfaces[cid][variant]) / mean - 1), s=24, color=col, zorder=3, lw=0)
        ax.set_yscale("symlog", linthresh=2, linscale=1.0)
        ax.set_yticks([-30, -10, -3, 0, 3, 10, 30])
        ax.set_yticklabels(["−30", "−10", "−3", "0", "+3", "+10", "+30"])
        ax.set_ylim(-60, 12)
        ax.set_xlim(-0.7, len(items) - 0.3)
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([f"{m:.2g}" for m, _, _ in items], rotation=90, fontsize=9.5)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
        ax.set_title(titles[t], loc="left", fontsize=13)
        ax.set_xlabel("cases, in order of the fraction crossing\n(mean of the references)" if t != "4" else "cases, in order of the fraction past the lamella\n(mean of the references)")
    axes[0].set_ylabel("offset from the references (%)")
    hs = [Patch(color=GREY, alpha=0.6, label="band of the 10 reference codes")] + [plt.Line2D([], [], color=f[4], marker="o", lw=0, ms=7, label=f"HMC, {f[3]} tables") for f in FAMILIES]
    fig.legend(handles=hs, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=1.6, handletextpad=0.5)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.87, bottom=0.2, wspace=0.06)
    fig.savefig(out)
    plt.close(fig)


def fig5(out, results: Path):
    """The 11 codes without the GPU ports against each other: median over the 305 cases of the KS distance of the radial distributions in units of its 5 % critical
    value (1 = as far apart as two runs of the same code would be at the statistics of the case, roughly) and of the |difference| of the mean
    radius in %."""
    ks, dr = defaultdict(list), defaultdict(list)
    for r in read(results / "pairs.csv"):
        a, b = r["pair"].split("|")
        try:
            k, d = float(r["ks"]) / float(r["ks_crit_5pct"]), abs(float(r["d_rmean_pct"]))
        except ValueError:
            continue
        ks[frozenset((a, b))].append(k)
        dr[frozenset((a, b))].append(d)
    n = len(ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.9))
    for ax, data, title, fmt in ((axes[0], ks, "KS distance of the radial distribution\n(median, in units of its 5 % critical value)", "{:.1f}"),
                                 (axes[1], dr, "difference of the mean radius\n(median |%|)", "{:.2f}")):
        m = np.full((n, n), np.nan)
        for i, a in enumerate(ORDER):
            for j, b in enumerate(ORDER):
                if i != j:
                    m[i, j] = np.median(data[frozenset((a, b))])
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=np.nanmax(m) * 1.05)
        for i in range(n):
            for j in range(n):
                if i != j:
                    ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center", fontsize=8.5, color="white" if m[i, j] > 0.62 * np.nanmax(m) else INK)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([LABEL[c].replace("HMC (", "HMC\n(") for c in ORDER], rotation=90, fontsize=9.5)
        ax.set_yticklabels([LABEL[c] for c in ORDER], fontsize=9.5)
        for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            c = next(k for k, v in LABEL.items() if v.replace("HMC (", "HMC\n(") == lab.get_text() or v == lab.get_text())
            lab.set_color(COLOUR[FAMILY_OF[c]])
            if c.startswith("hmc"):
                lab.set_fontweight("bold")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, loc="left", fontsize=12.5)
        for edge in (3.5, 6.5, 8.5):
            ax.axhline(edge, color="white", lw=3)
            ax.axvline(edge, color="white", lw=3)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.9, bottom=0.2, wspace=0.43)
    fig.savefig(out)
    plt.close(fig)


GPU_CODES = {"gpen", "gegs", "gg4", "hmc_pen_lad"}      # the ports exist only to compare the speed of HMC on a GPU


def fig6(cases, qoi, out):
    """Wall time per history of one run (Tiers 2-4, 3e6 histories, median over the cases), on the validation machines (CPU codes one run
    per core, GPU codes one GPU). The wall time includes the setup of the run. Not a controlled benchmark."""
    t = defaultdict(list)
    for c, r in cases.items():
        if r["tier"] in ("2", "3", "4"):
            for code, row in qoi[c].items():
                if row["wall_s"] and row["nhist"]:
                    t[code].append(1e6 * float(row["wall_s"]) / float(row["nhist"]))
    med = {c: float(np.median(v)) for c, v in t.items()}
    codes = sorted([c for c in REFERENCES] + ["hmc_pen_lad"], key=lambda c: -med[c])
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    prettify(ax)
    ax.grid(axis="y", visible=False)
    for i, c in enumerate(codes):
        col = INK if c.startswith("hmc") else COLOUR[FAMILY_OF[c]]
        ax.hlines(i, 1, med[c], color=col, lw=1.2, alpha=0.5, zorder=2)
        ax.scatter(med[c], i, s=130, color=col, marker="*" if c in GPU_CODES else "o", zorder=3, lw=0)
        ax.text(med[c] * 1.35, i, (f"{med[c]:,.0f}" if med[c] >= 100 else f"{med[c]:.1f}") + " µs", va="center", fontsize=11.5)
    ax.set_yticks(range(len(codes)))
    PORT = {"gpen": "gpen (GPU port of PENELOPE)", "gegs": "gegs (GPU port of EGSnrc)", "gg4": "gg4 (GPU port of Geant4)"}
    ax.set_yticklabels(["HMC (any table set)" if c.startswith("hmc") else PORT.get(c, LABEL[c]) for c in codes], fontsize=12)
    ax.set_xscale("log")
    ax.set_xlim(1, 1e5)
    ax.set_ylim(-0.6, len(codes) - 0.4)
    ax.minorticks_off()
    ax.set_xlabel("wall time per history (µs), one run of 3·10⁶ histories")
    ax.invert_yaxis()
    hs = [plt.Line2D([], [], color=GREY, marker="o", lw=0, ms=8, label="CPU code"), plt.Line2D([], [], color=GREY, marker="*", lw=0, ms=12, label="GPU code")]
    ax.legend(handles=hs, loc="lower right", frameon=False)
    ax.set_title("Median over the 42 cases of Tiers 2-4, on the validation machines (setup included; not a controlled benchmark)",
                 loc="left", fontsize=11.5, color="#333")
    fig.text(0.01, 0.015, "gpen, gegs and gg4 are our GPU ports of PENELOPE, EGSnrc and Geant4, respectively. They were written only to check the speed of HMC "
             "against the full physics on a GPU, and are not distributed.", ha="left", va="bottom", fontsize=10.5, color="#333")
    fig.subplots_adjust(left=0.23, right=0.98, top=0.92, bottom=0.17)
    fig.savefig(out)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE.parent / "results")
    ap.add_argument("--out", type=Path, default=HERE.parent / "figures")
    ap.add_argument("--fonts", type=Path, help="folder of .ttf files to register (optional)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    use_style(fonts=a.fonts)
    cases, qoi, env, dens = load(a.results)
    fig1(cases, load_profiles(a.results), dens, a.out / "fig1_profiles.png")
    fig2(cases, qoi, a.out / "fig2_hmc_vs_engine.png")
    counts = fig3(cases, qoi, env, a.out / "fig3_hmc_in_band.png")
    ifc = defaultdict(dict)
    for r in read(a.results / "interfaces.csv"):
        ifc[r["case"]][r["code"]] = r["crossed_fraction"]
    fig4(cases, ifc, env, a.out / "fig4_interfaces.png")
    fig5(a.out / "fig5_code_distances.png", a.results)
    fig6(cases, qoi, a.out / "fig6_speed.png")
    print("wrote 6 figures to", a.out)
    return counts


if __name__ == "__main__":
    main()
