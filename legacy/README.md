# HMC-1 — Hybrid Monte Carlo positron range simulator (`legacy/`)

**HMC-1** is the GPU-accelerated, condensed-history Monte Carlo code of the first paper: it transports positrons through heterogeneous,
voxelised tissue geometries (CT or µ-map derived) and writes 3D annihilation maps for PET positron-range modelling and correction. It is
written in CUDA Fortran and distributed here **precompiled**, in two versions:

| folder | executable | tissue library | built |
|---|---|---|---|
| `v9-30mat/` | `HMC_UCM.run` (v9.0) | 30 Schneider materials, 1 keV to 4.6 MeV | 2026-07, the public release of this repository |
| `v8.1-24mat/` | `hybrid_positron_UCM.x` (v8.1) | 24 Schneider materials, 10 keV to 3.55 MeV | the executable of the paper |

The paper:

> Paneque-Yunta, R.J.; Encina-Baranda, N.; Herraiz, J.L.; Abushab, K.M.; Udías, J.M.; Ibáñez, P.
> **"Condensed-History GPU Positron Range Simulator in Heterogeneous Media for PET Resolution Modelling."** *Appl. Sci.* **2026**, *16*, 6706.
> [https://doi.org/10.3390/app16136706](https://doi.org/10.3390/app16136706)

v9 differs from the paper's code only in the 30-material library, its energy grid and a configurable step variation (10 % by default,
the 0.45 to 0.55 mm step of the paper); the transport algorithm, the table conventions and the validation methodology are those of the
paper. The newer engine of this repository, `hmc/` at the root, replaces HMC-1: it reads tables made with PENELOPE, EGSnrc, Geant4 or
FLUKA and has its own validation (`validation/`).

```
legacy/
  v9-30mat/     HMC_UCM.run, config.in, spectrum.txt (its 30-material tables are on Zenodo, below)
  v8.1-24mat/   hybrid_positron_UCM.x, config.in, tablas-positrones/ (the two small table files; the large ones are on Zenodo),
                schneider24mats.txt
  spectra/      β⁺ emission spectra (Cu64, F18, Ga68, I124, Mn52, Rb82, Zr89)
  scripts/      voxel_processor.py (30 materials), voxel_processor_24mat.py, build_30mat_tables.py, multirun.py
  LICENSE       MIT
```

## How it works

Each positron history is simulated with a **condensed-history** approach (Berger's formulation): rather than tracking every individual
elastic/inelastic collision, many interactions over a fixed step are represented collectively using pre-computed look-up tables (LUTs)
of interaction probabilities and energy/angle outcomes.

1. **Source sampling**: a source voxel is drawn from the cumulative activity distribution (`DECAY.raw`) via binary search; the position
   is sampled uniformly inside the voxel, the direction isotropically, and the initial kinetic energy from the β⁺ emission spectrum
   (`spectrum.txt`) via inverse-transform sampling.
2. **Step-length sampling**: a random step around the 0.5 mm of the LUTs (`s = 0.045 + 0.01·ξ` cm by default, a ±10 % variation) is drawn
   each iteration to suppress grid-alignment aliasing artefacts (`config.in` sets the percentage in v9).
3. **Material/energy indexing**: the material ID of the current voxel and the particle energy are mapped to LUT indices.
4. **Annihilation decision**: an annihilation probability is interpolated from the LUT and compared with a uniform random draw; the positron
   is either terminated (annihilation scored in the current voxel) or continues.
5. **Position/energy/direction update**: the particle advances along its direction; a new energy and scattering angle are drawn from the
   posterior LUTs for the current material and energy bin, corrected for the local density (`DENSCORR.raw`).

The GPU kernel runs one history per CUDA thread, with the annihilation map accumulated through atomic adds. The same code runs on the CPU.

## Requirements

- Linux x86-64 and, for the GPU path, an NVIDIA GPU (the CPU path needs none). `HMC_UCM.run` carries code for every architecture from
  Maxwell (sm_50) to Blackwell (sm_121), plus PTX; `hybrid_positron_UCM.x` from Turing (sm_75) to Blackwell.
- The run-time libraries of the **NVIDIA HPC SDK 26.1** (`libcudafor`, `libnvf`, `libnvomp`, ...): both executables link them dynamically and
  they are not redistributed here. Install the (free) SDK, or put its `compilers/lib` and `REDIST` library folders on `LD_LIBRARY_PATH`.
- About 2.5 to 3 GB of free GPU memory at least (the fixed LUT footprint, see below).
- Python 3 with `numpy` and `scipy` for the scripts.

## Inputs

All input files are read from the working directory:

| File | Type | Description |
|---|---|---|
| `MATERIAL.raw` | `int32` | voxelised material-ID map |
| `DENSCORR.raw` | `float32` | voxelised density-correction factor λ = ρ_real / ρ_reference |
| `DECAY.raw` | `float32` | voxelised number of decays per voxel (source activity) |
| `spectrum.txt` | ASCII, 2 columns | β⁺ emission spectrum: energy (eV), probability |
| `config.in` | ASCII | RNG seeds, GPU threads / histories per thread, CPU batch size, transport energy cutoff, anti-aliasing step variation (v9) |
| `tablas-positrones/` | binary | the look-up tables (below) |

`MATERIAL.raw` and `DENSCORR.raw` are made from a patient CT or µ-map with

```bash
python3 scripts/voxel_processor.py <ct_or_umap.raw> --type ct --shape NX NY NZ [--blur-sigma 1.0]          # 30 materials (v9)
python3 scripts/voxel_processor_24mat.py <ct_or_umap.raw> --type ct --shape NX NY NZ [--blur-sigma 1.0]    # 24 materials (v8.1)
```

which segments the volume into the tissue library by nearest physical density (through a piecewise HU→density calibration) and writes
`MATERIAL.raw`, `DENSITY.raw`, `DENSCORR.raw` and a blurred `DENSCORR.blur.raw`.

## Running

Download the legacy tables from Zenodo ([doi:10.5281/zenodo.24379591](https://doi.org/10.5281/zenodo.24379591)), unpack them where the executable looks for them, copy or link your
input files next to it, and run:

```bash
cd legacy/v9-30mat
unzip legacy-tables-penelope-30mat.zip                 # creates tablas-positrones/
# put (or link) your MATERIAL.raw, DENSCORR.raw and DECAY.raw in this folder; spectrum.txt is the one to simulate (copy one of ../spectra)
./HMC_UCM.run <device> <Nx> <Ny> <Nz> <dx> <dy> <dz>
```

- `device`: `1` for the GPU, `0` for the CPU.
- `Nx Ny Nz`: number of voxels along each axis. `dx dy dz`: voxel size in cm.

The output is a 3D annihilation map, `ANNIHILATION.{gpu,cpu}.raw` (`float32`), usable as the input of PET reconstruction or resolution
modelling (for instance as a spatially variant PSF kernel). `scripts/multirun.py` batch-runs a list of isotopes with the spectra of
`spectra/`. For the paper's executable use `legacy/v8.1-24mat/` and its own tables (below).

## Tables

Interaction LUTs are made offline from event-by-event PENELOPE simulations per material and energy (10⁵ histories, 0.5 mm fixed step):
the annihilation probability, the posterior energy and angular distributions after a step and the mean deposited energy per step.

- **v9, 30 materials** (`legacy-tables-penelope-30mat.zip` of the Zenodo record, 1.4 GB): 30 Schneider reference tissues (0.027 to 1.837 g/cm³) on a 103-point natural-log
  energy grid from 1 keV to 4.6 MeV. `scripts/build_30mat_tables.py` builds them from the raw per-history PENELOPE output (see its header;
  paths through the `RAW_DIR`, `MAT_DIR` and `OUT_DIR` environment variables).
- **v8.1, 24 materials** (the paper; `legacy-tables-penelope-24mat.zip` of the same record, 1.6 GB): the two large files,
  `table_energy_posit-24MAT.raw` and `table_angle_posit-24MAT.raw`, and the deposited-energy table; unzip it into
  `legacy/v8.1-24mat/tablas-positrones/` (`unzip -d tablas-positrones`), next to the two small files already there.

## Memory footprint

The dominant GPU memory cost is the fixed LUT footprint (independent of image size): the posterior energy and angular tables are each a
`100000 × 103 × 30` `float32` array (about 1.24 GB) in v9, 2.47 GB together, loaded into GPU memory at start-up. The per-voxel maps
(`MATERIAL.raw`, `DENSCORR.raw`, `ANNIHILATION.raw`) scale linearly with the image (4 bytes per voxel each). A GPU with at least 4 GB of
memory is recommended.

## Validation and performance (published, 24-material configuration)

The paper validates HMC (both backends) against PenEasy 2024 on five scenarios (point-source profiles and convergence, heterogeneous sphere
interfaces, a clinical [¹⁸F]FDG-derived ⁸²Rb myocardium benchmark and runtime scaling) for ¹⁸F, ⁶⁸Ga, ⁸²Rb and ¹²⁴I:

- point-source FWHM and FWTM residuals against PenEasy below 0.3 mm at 10⁷ histories;
- myocardium benchmark: voxel-wise **R² = 0.995**, **SSIM = 0.995**, **PSNR ≈ 36.4 dB** against PenEasy 2024, with a simulation time of
  about 8 h 45 min (PenEasy, single thread), about 14 min (HMC CPU) or **about 1 s (HMC GPU)**;
- terminal throughput 3.32·10⁸ histories/s (HMC GPU, NVIDIA RTX 5080), 4.77·10⁵ histories/s (HMC CPU, Intel i9-14900KF) and 1.41·10⁴
  histories/s (PenEasy 2024): a combined **about 23 600×** speed-up of the GPU over single-thread PenEasy (34× from the condensed-history
  approximation, 697× from GPU parallelism);
- the code-to-code disagreement concentrates near sharp material interfaces, attributed to the 0.5 mm LUT step being coarse for the abrupt
  changes of annihilation probability there.

See the paper for the methodology, the figures and the discussion.

## Licence

MIT, see `LICENSE` (Copyright (c) 2026 Robert J. Paneque Yunta), as published with the first release of this repository.
