#!/usr/bin/env python3
"""make_hmc_inputs.py -- turn a validation case into the inputs of hmc.x.

    python validation/tools/make_hmc_inputs.py t1_mono_E1000keV_mat12_dx0.50 --out work/t1
    hmc/hmc.x --config work/t1/hmc.in                       # from the repository root
    python validation/tools/check_case.py t1_mono_E1000keV_mat12_dx0.50 work/t1/annihilation.raw

The case is a case id (a row of validation/results/cases.csv) or the path of a case file. The 261 Tier 1 cases are homogeneous cubes
with a point source in the central voxel; they are built here from their row of cases.csv and, for an isotope, its spectrum in
hmc/input/spectra (a mono-energetic one writes its two-line spectrum). The 44 cases of Tiers 0, 2, 3 and 4 are files,
validation/cases/t<tier>/<id>.npz: the specification (`case_json`), the material of every voxel (`voxel_map`, global ids 1..30, 0 = void),
the relative activity (`source`), the voxel size (`d_cm`) and the emission spectrum (`spectrum_txt`). This script writes, in
--out: density.raw (the mass density of each voxel: the nominal density of its material, validation/materials.csv),
activity.raw, spectrum.txt and hmc.in. The .raw files are the format hmc.x reads: int32 Nx Ny Nz, float32 dx dy dz [cm], then
float32 values, x fastest. Paths in hmc.in are relative to the directory you run hmc.x from (the repository root).

Defaults reproduce what the validation ran: PENELOPE ladder tables (`tables = luts/output/penelope`, the Zenodo set), the case's
histories and emission sampling, backend auto. `--seed` is the master seed (the validation used three per case).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MATERIALS = HERE.parent / "materials.csv"
CASES_CSV = HERE.parent / "results" / "cases.csv"
CASES_DIR = HERE.parent / "cases"


def write_raw(path, data, d_cm):
    """float32 .raw with the header hmc.x expects; data has shape (Nx, Ny, Nz)."""
    data = np.asarray(data, dtype="<f4")
    hdr = np.array(data.shape, dtype="<i4").tobytes() + np.asarray(d_cm, dtype="<f4").tobytes()
    Path(path).write_bytes(hdr + np.ascontiguousarray(data.T).tobytes())


def read_raw(path):
    """-> (float32 array (Nx, Ny, Nz), voxel size [cm])."""
    b = Path(path).read_bytes()
    n = np.frombuffer(b[:12], "<i4")
    d = np.frombuffer(b[12:24], "<f4")
    a = np.frombuffer(b[24:], "<f4").reshape(n[2], n[1], n[0])
    return np.ascontiguousarray(a.T), tuple(float(x) for x in d)


def case_id(case):
    """The id of a case given as an id or as the path of its .npz file (ids contain dots: only a final .npz is cut)."""
    name = Path(str(case)).name
    return name[:-4] if name.endswith(".npz") else name


def spectra_dir():
    """hmc/input/spectra of the repository this package sits in (the isotope spectra of the Tier 1 cases)."""
    for up in HERE.parents:
        if (up / "hmc" / "input" / "spectra").is_dir():
            return up / "hmc" / "input" / "spectra"
    raise SystemExit("hmc/input/spectra not found above " + str(HERE))


def build_tier1(cid):
    """A Tier 1 case (homogeneous cube, point source in the central voxel) from its row of results/cases.csv.
    Same content as the case files of the other tiers: {case, voxel_map, source, d_cm, spectrum_txt, true_density}."""
    with open(CASES_CSV, newline="") as f:
        row = next((r for r in csv.DictReader(f) if r["id"] == cid), None)
    if row is None or row["tier"] != "1":
        raise SystemExit(f"{cid}: not a case file ({CASES_DIR}/{cid[:2]}/{cid}.npz) and not a Tier 1 case of {CASES_CSV}")
    n = [int(x) for x in row["grid"].split("x")]
    d_mm = [float(x) for x in row["voxel_mm"].split("x")]
    centre = tuple(i // 2 for i in n)
    vmap = np.full(n, int(row["materials"]), np.int32)
    src = np.zeros(n, np.float32)
    src[centre] = 1
    if row["kind"] == "mono":
        e = float(row["energy_keV"]) * 1e3
        spec = f"# Energy[eV]  Intensity\n{e:.1f}  1.0\n{e + 1:.1f}  0.0\n"
    else:
        spec = (spectra_dir() / f"{row['isotope']}.txt").read_text()
    case = {"id": cid, "tier": 1, "kind": row["kind"], "isotope": row["isotope"] or None,
            "energy_keV": float(row["energy_keV"]) if row["energy_keV"] else None, "materials": [int(row["materials"])],
            "grid": {"n": n, "d_mm": d_mm, "d_cm": [x / 10 for x in d_mm]}, "nhist": int(float(row["nhist"])),
            "seeds": [int(s) for s in row["seeds"].split()],
            "source": {"type": "voxel", "sampling": row["sampling"], "voxel": list(centre)},
            "geometry": json.loads(row["geometry"]), "ecut_eV": 1000.0}
    return {"case": case, "voxel_map": vmap, "source": src, "d_cm": tuple(x / 10 for x in d_mm), "spectrum_txt": spec,
            "true_density": None}


def load_case(case):
    """`case`: a case id or the path of a case file. The files of Tiers 0, 2, 3, 4 are validation/cases/t<tier>/<id>.npz."""
    cid = case_id(case)
    path = Path(case) if str(case).endswith(".npz") and Path(case).is_file() else CASES_DIR / cid[:2] / f"{cid}.npz"
    if not path.is_file():
        return build_tier1(cid)
    with np.load(path, allow_pickle=False) as z:
        case = json.loads(str(z["case_json"].item()))
        out = {"case": case, "voxel_map": z["voxel_map"].astype(np.int32), "source": z["source"].astype(np.float32),
               "d_cm": tuple(float(x) for x in z["d_cm"]), "spectrum_txt": str(z["spectrum_txt"].item()),
               "true_density": z["true_density"].astype(np.float32) if "true_density" in z.files else None}
    return out


def nominal_densities(path=MATERIALS):
    with open(path) as f:
        return {int(r["id"]): float(r["density_g_cm3"]) for r in csv.DictReader(f)}


def make_inputs(case_file, out, tables="luts/output/penelope", nhist=None, seed=1, backend="auto", extra=None):
    c = load_case(case_file)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if c["true_density"] is not None:
        dens = c["true_density"]
    else:
        rho = nominal_densities()
        lut = np.zeros(int(c["voxel_map"].max()) + 1, np.float32)
        for m in np.unique(c["voxel_map"]):
            lut[int(m)] = rho.get(int(m), 0.0)                 # material 0 = void: density 0
        dens = lut[c["voxel_map"]]
    write_raw(out / "density.raw", dens, c["d_cm"])
    write_raw(out / "activity.raw", c["source"], c["d_cm"])
    (out / "spectrum.txt").write_text(c["spectrum_txt"])
    case = c["case"]
    cfg = {"density": out / "density.raw", "activity": out / "activity.raw", "spectrum": out / "spectrum.txt",
           "output": out / "annihilation.raw", "tables": tables, "backend": backend,
           "nhist": int(nhist or case["nhist"]), "seed": seed, "sampling": case["source"].get("sampling", "center")}
    cfg.update(extra or {})
    (out / "hmc.in").write_text("".join(f"{k} = {v}\n" for k, v in cfg.items()))
    (out / "case.json").write_text(json.dumps(case, indent=1))
    return out / "hmc.in"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", help="a case id (results/cases.csv) or a validation/cases/t*/*.npz file")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tables", default="luts/output/penelope", help="table directory for hmc.in (penelope, geant4, egsnrc or fluka under luts/output)")
    ap.add_argument("--nhist", type=float, help="histories (default: the case's)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--backend", default="auto", choices=["auto", "cpu", "gpu"])
    a = ap.parse_args(argv)
    cfg = make_inputs(a.case, a.out, a.tables, a.nhist, a.seed, a.backend)
    print(f"wrote {cfg.parent}/{{density.raw, activity.raw, spectrum.txt, hmc.in, case.json}}\nrun: hmc/hmc.x --config {cfg}")


if __name__ == "__main__":
    main()
