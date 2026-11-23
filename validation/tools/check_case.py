#!/usr/bin/env python3
"""check_case.py -- score an hmc.x run of a validation case against the reference codes' band.

    python validation/tools/check_case.py CASE HMC_MAP.raw [--reference REF.npz] [--nhist N] [--n-boot 200]

CASE is a case id (a row of results/cases.csv) or a case file, validation/cases/t<tier>/<id>.npz (see make_hmc_inputs.py).

Computes, from the annihilation map of your run, the quantities the validation compares (the radial quantiles r50, r90, r999 and
the mean radius from the source centre, and for slab / ramp / lamella cases the interface quantities: fractions on each side,
per material, beyond the lamella), with a bootstrap sigma for the statistics of your run, and puts each of them next to

  * the band of the reference codes of results/envelope.csv: [lo - tol, hi + tol], lo/hi = smallest/largest value over the codes
    PENELOPE (pvoxel, PenEasy, gpen), GATE, Geant4, EGSnrc, gegs, gg4, PenRed and FLUKA, tol = the larger of twice their
    statistical sigma, the seed-to-seed sigma and, for radial quantiles, half a voxel; your run adds twice its own sigma;
  * what the published HMC runs gave (hmc_pen_lad in results/qoi.csv).

A quantity "in" the band is consistent with the references at the statistics of the case; the exit status is 1 if any quantity is
out. With --reference (a file of reference_maps/) it also compares the two maps voxel by voxel: Kolmogorov-Smirnov distance of the
radial CDFs against its 5 % critical value, 1-Wasserstein distance, voxel chi2/ndf, SSIM. Lengths are in mm.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_hmc_inputs as io  # noqa: E402
import metrics  # noqa: E402

RESULTS = HERE.parent / "results"
RADIAL = {"r50": "r50", "rmean": "rmean", "r90": "r90", "r999": "r999"}


def rows(path, case):
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r["case"] == case]


def run_quantities(case_file, map_file, nhist, n_boot):
    """-> {qoi: (value, sigma)} for this run, in mm (radial) or unitless (interface)."""
    c = io.load_case(case_file)
    m, d = io.read_raw(map_file)
    case = c["case"]
    if m.shape != c["voxel_map"].shape:
        raise SystemExit(f"{map_file}: grid {m.shape} is not the case's {c['voxel_map'].shape}")
    centre = metrics.source_centre_cm(c["source"], d)
    r = metrics.voxel_radii(m.shape, d, centre)
    boot = metrics.bootstrap_radial(m, r, n_boot)
    q = {k: (10 * boot[k]["value"], 10 * boot[k]["sigma"]) for k in ("r50", "rmean", "r90", "r999")}
    n_ann = float(m.sum(dtype=np.float64))
    if case["kind"] in ("slab", "ramp", "sphere", "lamella"):
        dens = io.nominal_densities()
        im = metrics.interface_metrics(m, c["voxel_map"], d, case["geometry"], dens)
        for k, v in im["fraction_per_material"].items():
            q[f"iface_frac_mat{k}"] = (v, np.sqrt(v * (1 - v) / n_ann))
        for k in ("left_fraction", "right_fraction", "lamella_fraction", "beyond_fraction"):
            if k in im:
                v = im[k]
                q[f"iface_{k}"] = (v, np.sqrt(max(v * (1 - v), 0) / n_ann))
        if "LR_ratio" in im:
            L, R = im["left_fraction"], im["right_fraction"]
            q["iface_LR_ratio"] = (im["LR_ratio"], im["LR_ratio"] * np.sqrt((1 - L) / (n_ann * L) + (1 - R) / (n_ann * R)))
    return c, m, d, q, n_ann


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", help="a case id (results/cases.csv) or a validation/cases/t*/*.npz file")
    ap.add_argument("map", type=Path, help="annihilation.raw written by hmc.x")
    ap.add_argument("--reference", type=Path, help="a reference_maps/*.npz of the same case")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--results", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    c, m, d, q, n_ann = run_quantities(a.case, a.map, None, a.n_boot)
    case = c["case"]
    js = Path(str(a.map) + ".json")
    nh = json.loads(js.read_text())["n_hist"] if js.exists() else case["nhist"]
    env = {r["qoi"]: r for r in rows(a.results / "envelope.csv", case["id"])}
    pub = {r["code"]: r for r in rows(a.results / "qoi.csv", case["id"])}.get("hmc_pen_lad", {})
    print(f"case {case['id']}: {nh:.3g} histories, {n_ann:.6g} annihilated ({100 * (1 - n_ann / nh):.3f} % escaped); "
          f"the band of {len(env)} quantities from {a.results.name}/envelope.csv\n")
    print(f"{'quantity':22s} {'this run':>20s} {'band of the references':>30s} {'published HMC':>14s}  verdict")
    bad = 0
    for k, (v, s) in q.items():
        e = env.get(k)
        if e is None:
            continue
        lo, hi, tol = float(e["lo_mm"]), float(e["hi_mm"]), float(e["tol_mm"])
        ok = lo - tol - 2 * s <= v <= hi + tol + 2 * s
        bad += not ok
        ref = pub.get(f"{k}_mm") if k in RADIAL else None
        refs = f"{float(ref):14.4f}" if ref not in (None, "") else f"{'':14s}"
        print(f"{k:22s} {v:11.4f} ± {s:6.4f}   [{lo - tol:11.4f}, {hi + tol:11.4f}] {refs}  {'in' if ok else 'OUT'}")
    if a.reference:
        with np.load(a.reference, allow_pickle=False) as z:
            ref_map = z["map"].astype(np.float64)
            code = str(z["code"]) if "code" in z.files else "reference"
        centre = metrics.source_centre_cm(c["source"], d)
        cmp_ = metrics.compare_maps(m.astype(np.float64), ref_map, d, centre)
        vr = cmp_["voxel_ratio"]
        print(f"\nagainst the {code} map: KS {cmp_['ks']:.4f} (5 % critical {cmp_['ks_crit_5pct']:.4f}), W1 {10 * cmp_['w1_cm']:.4f} mm, "
              f"voxel chi2/ndf {vr['chi2_ndf']:.3f}, SSIM {cmp_['image']['ssim_global']:.5f}")
    print(f"\n{len(q) - bad - sum(1 for k in q if k not in env)} in the band, {bad} out")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
