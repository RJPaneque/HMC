#!/usr/bin/env python3
"""
build_30mat_tables.py

Converts the raw per-history positron transport data produced by
HMC-2/tablegen (positron_raw_matXX.txt, 30 materials, 103-point natural-log
energy grid from 1 keV to 4.6 MeV) into the file formats consumed by the
hybrid_final Fortran/CUDA engine (SOURCES_v9), replacing the old 24-material
tables.

Output files (written into tablas-positrones/), matching the *content*
conventions of the old 24MAT files but for 30 materials on the new grid:

  input-materiales.dat            densities (from materials/output/30mats/matN.mat)
  TABLAS-MATERIALES.dat           p_posit(103,30): 100*(1 - P_annihilated) vs grid energy
  tables-edepo-posit-30MAT.dat    en_prom(30,4600): mean deposited energy (eV) per integer keV,
                                   log-E-interpolated from the 103-point grid means
  table_energy_posit-30MAT.raw    ener_posit(100000,103,30): raw E_final_eV samples
  table_angle_posit-30MAT.raw     angle_posit(100000,103,30): 1 - cos(theta_rad) samples

Field mapping derived from SOURCES_v9/parte_CPU-positrons.f:
  - ener_posit(sample, ie, mat)  = E_final_eV            (raw column 5)
  - angle_posit(sample, ie, mat) = 1 - cos(theta_rad)     (raw column 2)
  - p_posit(ie, mat)             = 100 * (1 - fraction annihilated at grid energy ie)
  - en_prom(mat, kk)             = mean TOTDE_eV (raw column 3) at incident energy kk keV,
                                    log-E interpolated from the 103 grid-energy means since the
                                    raw data was only simulated at the 103 grid points (not at
                                    every integer keV, unlike the original 24MAT source runs).

New energy grid (must match SOURCES_v9 update): natural-log spaced,
E(ie) = 1.0 * exp((ie-1)*FAC) keV, FAC = ln(4600)/102, ie = 1..103.
"""

import numpy as np
from pathlib import Path

N_ENERGIES = 103
N_SAMPLES = 100000
N_MATERIALS = 30
EMIN_KEV = 1.0
EMAX_KEV = 4600.0
FAC = np.log(EMAX_KEV / EMIN_KEV) / (N_ENERGIES - 1)
MAX_EDEPO_KEV = 4600

RAW_DIR = Path("/home/rpy/Projects/HMC-2/tablegen/output/raw")
MAT_DIR = Path("/home/rpy/Projects/HMC-2/materials/output/30mats")
OUT_DIR = Path("/home/rpy/Projects/HMC/source/hybrid_final/tablas-positrones")

ENERGY_GRID_KEV = EMIN_KEV * np.exp(np.arange(N_ENERGIES) * FAC)


def read_density(mat_id: int) -> float:
    text = (MAT_DIR / f"mat{mat_id}.mat").read_text()
    for line in text.splitlines():
        if "Mass density" in line:
            return float(line.split("=")[1].split("g/cm")[0].strip())
    raise ValueError(f"density not found for mat{mat_id}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    densities = [read_density(i) for i in range(1, N_MATERIALS + 1)]

    p_posit = np.zeros((N_ENERGIES, N_MATERIALS), dtype=np.float64)
    edepo_grid_mean = np.zeros((N_ENERGIES, N_MATERIALS), dtype=np.float64)

    energy_path = OUT_DIR / "table_energy_posit-30MAT.raw"
    angle_path = OUT_DIR / "table_angle_posit-30MAT.raw"

    with open(energy_path, "wb") as fener, open(angle_path, "wb") as fangle:
        for mat_id in range(1, N_MATERIALS + 1):
            raw_path = RAW_DIR / f"positron_raw_mat{mat_id}.txt"
            print(f"Reading material {mat_id}: {raw_path.name} ...", flush=True)

            data = np.loadtxt(
                raw_path,
                dtype=np.float64,
                skiprows=1,
                usecols=(0, 1, 2, 4, 5),
                # columns: 0=E0_keV 1=theta_rad 2=DE_total_eV 4=E_final_eV 5=annihil_flag
            )
            assert data.shape[0] == N_ENERGIES * N_SAMPLES, \
                f"mat{mat_id}: expected {N_ENERGIES * N_SAMPLES} rows, got {data.shape[0]}"

            e0 = data[:, 0].reshape(N_ENERGIES, N_SAMPLES)
            theta = data[:, 1].reshape(N_ENERGIES, N_SAMPLES)
            totde = data[:, 2].reshape(N_ENERGIES, N_SAMPLES)
            efinal = data[:, 3].reshape(N_ENERGIES, N_SAMPLES)
            annihil = data[:, 4].reshape(N_ENERGIES, N_SAMPLES)

            assert np.allclose(e0[:, 0], ENERGY_GRID_KEV, rtol=1e-4), \
                f"mat{mat_id}: energy grid mismatch vs expected natural-log grid"

            # ener_posit: raw final energy (eV), angle_posit: 1 - cos(theta)
            ener_block = efinal.astype(np.float32)
            angle_block = (1.0 - np.cos(theta)).astype(np.float32)

            # Fortran column-major (100000, 103, 30): for fixed material, samples
            # vary fastest, then energy index -- i.e. write each energy's 100000
            # samples contiguously, in ascending energy order. That is exactly the
            # (N_ENERGIES, N_SAMPLES) row-major layout already in memory here.
            fener.write(np.ascontiguousarray(ener_block).tobytes())
            fangle.write(np.ascontiguousarray(angle_block).tobytes())

            frac_annihil = annihil.mean(axis=1)
            p_posit[:, mat_id - 1] = 100.0 * (1.0 - frac_annihil)
            edepo_grid_mean[:, mat_id - 1] = totde.mean(axis=1)

    print(f"Wrote {energy_path} ({energy_path.stat().st_size/1e9:.2f} GB)")
    print(f"Wrote {angle_path} ({angle_path.stat().st_size/1e9:.2f} GB)")

    # input-materiales.dat
    inp_path = OUT_DIR / "input-materiales.dat"
    with open(inp_path, "w") as f:
        for i in range(1, N_MATERIALS + 1):
            f.write(f"# MATERIAL {i}: MAT{i}. Density (g/cm**3)  [Schneider_mat{i}]\n")
            f.write(f"{densities[i-1]:.8E}\n")
    print(f"Wrote {inp_path}")

    # TABLAS-MATERIALES.dat
    mat_path = OUT_DIR / "TABLAS-MATERIALES.dat"
    with open(mat_path, "w") as f:
        for mat_id in range(1, N_MATERIALS + 1):
            f.write(f"#MATERIAL  {mat_id}\n")
            f.write("#Particulas de cada tipo en porcentaje   que salen de las histoiras iniciales\n")
            f.write("#Energia    Positrones\n")
            for ie in range(N_ENERGIES):
                e_ev = ENERGY_GRID_KEV[ie] * 1000.0
                f.write(f"  {e_ev:.8E}  {p_posit[ie, mat_id-1]:.7E}\n")
    print(f"Wrote {mat_path}")

    # tables-edepo-posit-30MAT.dat : log-E interpolation of the 103 grid means
    # onto every integer keV from 1 to MAX_EDEPO_KEV.
    edepo_path = OUT_DIR / "tables-edepo-posit-30MAT.dat"
    kk_vals = np.arange(1, MAX_EDEPO_KEV + 1, dtype=np.float64)
    log_kk = np.log(kk_vals)
    log_grid = np.log(ENERGY_GRID_KEV)
    with open(edepo_path, "w") as f:
        for mat_id in range(1, N_MATERIALS + 1):
            interp_vals = np.interp(log_kk, log_grid, edepo_grid_mean[:, mat_id-1])
            f.write(f"# MATERIAL: MAT{mat_id}\n")
            f.write(f"{MAX_EDEPO_KEV}\n")
            for kk, val in zip(kk_vals, interp_vals):
                f.write(f" {kk*1000.0:.9E}  {val:.9E}\n")
    print(f"Wrote {edepo_path}")

    print("Done.")


if __name__ == "__main__":
    main()
