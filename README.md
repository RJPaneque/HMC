# HMC — condensed-history Monte Carlo for positron range

HMC transports positrons through a voxelised phantom and writes a 3D map of where they annihilate. You give it a density map, an
activity map and a β⁺ emission spectrum; it returns annihilation counts per voxel, from which positron-range kernels, blurred images
or resolution models follow. It runs on NVIDIA GPUs (one history per thread, up to billions of histories per second for the lightest
isotopes) and on any x86-64 CPU, from the same executable and with the same physics line for line.

The physics is a *condensed-history* model: instead of following every collision, each step samples the outcome of a fixed mass
thickness from look-up tables computed once with an event-by-event code (PENELOPE by default; Geant4, EGSnrc and FLUKA tables are
provided too). This repository ships the **precompiled executable**, the tables and the scripts that made the materials, the tables and
the spectra, not the source of the engine.

## Quick start

1. **Clone this repository.** It holds the executables (`hmc/hmc.x` starts the right one, see *Executables* below), an example case and the isotope spectra.
2. **Download the tables** from Zenodo ([doi:10.5281/zenodo.24379591](https://doi.org/10.5281/zenodo.24379591)), and unzip them at the root of the clone. One zip per engine; the
   PENELOPE one (`HMC-tables-penelope-ladder.zip`, 0.75 GB) is all you need. It creates `luts/output/penelope/step1-1 … step1-8`.
   ```
   unzip HMC-tables-penelope-ladder.zip
   ```
3. **Run the example** (a 1 MeV planar source in a block of cortical bone, 10⁶ histories):
   ```
   mkdir -p hmc/output
   hmc/hmc.x --config hmc/input/hmc.in
   ```
   The map is `hmc/output/test_map.raw`, with a summary in `test_map.raw.json`.

**Requirements.** Linux x86-64 (Windows: through WSL2, below). For the GPU backend an NVIDIA GPU from Maxwell (cc 5.0) on and a current
driver; no CUDA toolkit is needed (the runtime is inside the executable). Without a GPU, or with `backend = cpu`, it runs on the CPU.
The Python tools and tests need `requirements.txt` (numpy, scipy, matplotlib, pytest) and the `zstd` command.

## Executables

`hmc/hmc.x` is a small shell script that starts the one of `hmc/bin/` that suits the machine; `hmc/hmc.x --which` says which and why, and
`HMC_BINARY=hmc-cpu-x86_64 hmc/hmc.x ...` forces another. They are the same program, built three ways:

| file | runs on | needs |
|---|---|---|
| `hmc-cuda13-x86_64` | NVIDIA GPUs from Turing (cc 7.5) on: RTX 20 series, A100, RTX 30/40/50, H100, B-series; and the CPU | driver 580 or later, glibc 2.27, `libstdc++`, an AVX2 CPU |
| `hmc-cuda12-x86_64` | NVIDIA GPUs from Maxwell (cc 5.0) on, including Pascal and Volta (GTX 750 to V100); and the CPU | a recent driver, glibc 2.27, an AVX2 CPU |
| `hmc-cpu-x86_64` | any x86-64 Linux, no GPU | nothing (statically linked, no AVX2 requirement) |

The selector picks the CUDA 13 build when the driver is 580 or later and the GPU is Turing or newer, the CUDA 12 build for any other NVIDIA
GPU, and the CPU build when there is none. Which one a GPU uses makes little difference (the CUDA 13 build is about 1.5 % faster on an
RTX 5080); the point is to cover every GPU and driver. Not covered: Kepler and older GPUs, non-NVIDIA GPUs, ARM machines (an aarch64 CPU build
is planned), and macOS.

**Windows.** Use WSL2 (Windows 10 21H2 or later, or Windows 11): the Linux executables run in it unchanged, on the CPU and, with NVIDIA's
Windows driver installed (not a Linux driver inside WSL), on the GPU. Keep the repository in the Linux file system (`~`), not under
`/mnt/c`, which is slow.

**Tested.** On an RTX 5080 (Blackwell, driver 595), natively and in clean Ubuntu 22.04 and AlmaLinux 8 containers (`tools/fresh_start/` of
the development repository does it), and on a Quadro GV100 (Volta, driver 570.169, CUDA 12.8) under Debian 13, where the CUDA 12 build ran
on that older driver and a validation case came out inside the band of the references. In WSL2 (Ubuntu 24.04 on Windows, Windows driver 560.94, CUDA 12.6) the CUDA 12 executable ran on both backends on a GTX 970 (Maxwell, cc 5.2, 4 GB),
the oldest GPU generation it targets. The code for the other GPU generations is in the
executables but has not been run on them. `sh check_gpu.sh` (after the tables are unzipped) tells in half a minute whether it runs on your
machine, on both backends, and `sh check_gpu.sh all` runs each executable of `hmc/bin/`; if something fails, its output is what to send in a bug report.

## Inputs

Everything a run needs is in one configuration file of `key = value` lines (`#` starts a comment, paths are relative to the working
directory). `hmc/input/hmc.in` documents every key, and `hmc/hmc.x --config-example` prints it. The command line is only
`--config FILE [--resume] [--quiet]`.

| key | meaning |
|---|---|
| `density` | float32 `.raw` map of mass density [g/cm³], the only geometry input. Header: `int32 Nx Ny Nz`, `float32 dx dy dz` [cm], then the data, x fastest (`index = ix + Nx·(iy + Ny·iz)`). Each voxel takes the table material of nearest nominal density (the 30 Schneider tissue bins, lung to cortical bone, 0.027–1.84 g/cm³) and a step length scaled by ρ/ρ_nom. A density ≤ `rho_void` is void: positrons entering it escape |
| `activity` | float32 `.raw` map on the same grid, relative activity per voxel. By default every voxel gets exactly its share of the histories (no sampling noise from the source) |
| `spectrum` | two columns, `E_eV weight`; a monoenergetic source is one line `E 1.0` followed by `E+1 0.0` |
| `tables` | the table directory: `luts/output/penelope` (default), `geant4`, `egsnrc` or `fluka` |
| `output` | the float32 `.raw` map of annihilation counts |
| `nhist`, `seed` | histories (int64) and master seed. The random numbers are counter-based (Philox), so a result is reproducible and identical on the CPU and the GPU for the same number of workers |
| `backend` | `auto` (default: the GPU if there is one), `cpu` or `gpu` |

Further keys choose the step model (`ladder`, the default and the engine, or `endpoint` / `hinge` for comparisons), the emission point
inside the voxel, checkpoints (`checkpoint`, `max_wall_s`, `--resume`, for cluster time limits) and the GPU launch geometry. An
unknown key or a bad value is an error that names the file and line.

**Spectra.** `hmc/input/spectra/` has the β⁺ spectra of C-11, N-13, O-15, F-18, Na-22, Sc-44, Mn-52, Cu-64, Ga-66, Ga-68, Br-76, Rb-82, Y-86, Zr-89 and I-124 (the validation cases
use seven of them, Cu-64, F-18, Ga-68, I-124, Mn-52, Rb-82 and Zr-89; the others were generated later with the same pipeline; Y-86 from a `.nuc` file we made from ENSDF, `spectrum/ensdf2nuc/`). Copy one to use it as
`spectrum`. Their origin is in `THIRD_PARTY_NOTICES.md`.

## Outputs

`output` is a float32 `.raw` map of annihilation counts per voxel with the voxel size in its header, plus `<output>.json` (backend,
histories, annihilated and escaped counts, seed, step model, table version, grid, wall time, steps per history) and
`<output minus .raw>.material.raw`, the material index the engine assigned to each voxel, for inspection.

## Tables

The tables (`ann_LUT.raw`, `scatt-eloss_LUT.raw`, `stepend_LUT.raw`, `run.json`) are in four sets of the same layout, one per
engine, each a *ladder* of four step lengths (`step1-1` = 0.05 cm, `step1-2`, `step1-4`, `step1-8` = 0.00625 cm): a step takes the
longest set that fits the distance to the nearest material face. PENELOPE is the default; the others are alternatives for
comparing the transport code, not different physics options of equal standing. Each set carries its own checksums (`SHA256SUMS`).
Licence of the tables: CC BY-NC 4.0, see `LICENSE`.

## Performance

Measured with PENELOPE tables, on an RTX 5080 (1 000 000 000 histories unless stated):

| case | histories/s |
|---|---|
| F-18 point source, 49³ voxels of 0.5 mm | 3300 M |
| Ga-68 point source | 1220 M |
| Rb-82 point source, 101³ | 695 M |
| QUADRA_HC phantom, 86³, Rb-82 (4.5 × 10⁸ histories) | 513 M |
| Ga-68 point source, CPU, one thread (10⁷ histories) | 1.8 M |

The cost of a run is its number of steps, not of histories: a history takes about its range divided by the step length (two steps
for F-18, fourteen for Rb-82). The summary reports `steps_per_hist` and `steps_per_s`.

## What it does not do

Positron transport only: it tallies annihilation positions. There is no photon transport, no dose, no detector response. The
geometry is voxel density only (materials come from the density map), and the positron cutoff is the lowest table energy (1 keV).

## `materials/`, `luts/`, `spectrum/`: where the inputs of the engine come from

`materials/` generates the 30 tissue materials (PENELOPE `.mat` files, from the Schneider et al. CT calibration), `luts/` runs the
event-by-event simulations and compacts them into the tables (`luts/LUT_FORMAT.md` is the format and the sampling contract `hmc.x`
follows), `spectrum/` computes the β⁺ spectra. The PENELOPE and Geant4 pipelines are complete (they need PenSuite 2023 with `ifx`,
resp. Docker), with their tests; the EGSnrc and FLUKA drivers depend on modules and a licence we cannot distribute, so only their
settings are documented. Each folder has a README.

## `validation/`: what HMC was checked against

305 synthetic cases (point sources of seven isotopes in six tissues, slabs, density ramps, thin lamellae) run on HMC and on ten
reference codes (PENELOPE, PenEasy, GATE, Geant4, EGSnrc, PenRed, FLUKA and our GPU ports of three of them), with the case files, the
30 PENELOPE material files, the results of every code, three figures and the Python tools to run any case on `hmc.x` and score it
against the reference band. `validation/README.md` has the details and a walk-through. With the PENELOPE tables, all 1220 radial
quantities of the 305 cases are inside the band of the references.

## `legacy/`: HMC-1

The code of the first paper, precompiled, in two versions: `legacy/v9-30mat/HMC_UCM.run` (the release this repository started with: 30
tissue materials, 1 keV to 4.6 MeV) and `legacy/v8.1-24mat/hybrid_positron_UCM.x` (the executable of
the paper, 24 materials). Their tables are the legacy tables of the Zenodo record. With
their configuration, the spectra and the scripts that segment a CT or µ-map (`legacy/scripts/`). MIT licence (part 3 of `LICENSE`).
`legacy/README.md` has the details. The history of this repository continues the one of HMC-1: the tag `hmc1-v9-30mat` is the state
before the new layout.

## Licence

The executable is free for research and other non-commercial use and may not be redistributed or modified
(part 2 of `LICENSE`); the example data, spectra and documentation are CC BY 4.0; `legacy/` is MIT; the tables are CC BY-NC 4.0.
`LICENSE` lists which is which and `THIRD_PARTY_NOTICES.md` the components inside the executable.
The executable contains runtime libraries of NVIDIA; the licence of the executable asks you not to extract or redistribute them.
The licence also applies when an AI assistant or another automated tool works for you: it may help you run HMC and analyse its results, not take the executables apart (clause 12 of part 2; `AGENTS.md`).

## Citation

If you use HMC, please cite:

```bibtex
@article{paneque2026hmc,
  title   = {Condensed-History GPU Positron Range Simulator in Heterogeneous Media for PET Resolution Modelling},
  author  = {Paneque-Yunta, Robert J. and Encina-Baranda, Nerea and Herraiz, Joaqu{\'\i}n L. and Abushab, Khaled M. and Ud{\'\i}as, Jos{\'e} Manuel and Ib{\'a}{\~n}ez, Paula},
  journal = {Applied Sciences},
  volume  = {16},
  number  = {13},
  pages   = {6706},
  year    = {2026},
  doi     = {10.3390/app16136706}
}
```

and the tables you use: PENELOPE (Salvat), Geant4, EGSnrc or FLUKA, as listed in `THIRD_PARTY_NOTICES.md`.
The validation of this version was presented as a poster at the IEEE NSS/MIC 2026 conference; the paper is in preparation and this README will cite it when it is available.

Contact: Robert J. Paneque-Yunta (rpaneque@ucm.es), Nuclear Physics Group, Universidad Complutense de Madrid.
