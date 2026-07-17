# HMC — Hybrid Monte Carlo Positron Range Simulator

**HMC** is a GPU-accelerated, condensed-history Monte Carlo code that explicitly transports
positrons through heterogeneous, voxelised tissue geometries (CT/µ-map derived) to produce
3D annihilation maps for PET positron-range (PR) modelling and correction. It is implemented
in CUDA Fortran (`.cuf`) and compiled with the NVIDIA HPC SDK (`nvfortran`), sharing a single
codebase between GPU and CPU execution paths.

This repository implements the method described in:

> Paneque-Yunta, R.J.; Encina-Baranda, N.; Herraiz, J.L.; Abushab, K.M.; Udías, J.M.; Ibáñez, P.
> **"Condensed-History GPU Positron Range Simulator in Heterogeneous Media for PET Resolution
> Modelling."** *Appl. Sci.* **2026**, *16*, 6706.
> [https://doi.org/10.3390/app16136706](https://doi.org/10.3390/app16136706)

> **Note:** the published paper validates the simulator against a 24-material tissue library
> (Schneider et al.) covering 10 keV–3.55 MeV. This repository has since been extended to a
> **30-material** library spanning **1 keV–4.6 MeV** (see [Tissue and energy tables](#tissue-and-energy-tables)
> below); the transport algorithm, table conventions, and validation methodology are otherwise
> unchanged from the paper.

## How it works

Each positron history is simulated with a **condensed-history** approach (Berger's formulation):
rather than tracking every individual elastic/inelastic collision, many interactions over a
fixed step are represented collectively using pre-computed look-up tables (LUTs) of interaction
probabilities and energy/angle outcomes. This avoids explicit cross-section sampling at every
step while retaining the aggregate transport physics needed to reproduce positron range and
annihilation-site distributions in heterogeneous media.

1. **Source sampling** — a source voxel is drawn from the cumulative activity distribution
   (`DECAY.raw`) via binary search; position is sampled uniformly inside the voxel, direction
   isotropically, and initial kinetic energy from the β⁺ emission spectrum (`spectrum.txt`) via
   inverse-transform sampling.
2. **Step-length sampling** — a random step `s = 0.045 + 0.01·ξ` cm (i.e. 0.45–0.55 mm, a ±10%
   variation around the fixed 0.5 mm step used to generate the LUTs) is drawn each iteration to
   suppress grid-alignment aliasing artefacts.
3. **Material/energy indexing** — the current voxel's material ID and the particle's energy are
   mapped to LUT indices (logarithmic energy grid for the interaction/angle/energy tables, linear
   1-keV grid for mean deposited energy).
4. **Annihilation decision** — an annihilation probability is linearly interpolated from the LUT
   and compared to a uniform random draw; the positron is either terminated (annihilation scored
   in the current voxel) or continues transport.
5. **Position/energy/direction update** — the particle is advanced along its direction; a new
   outgoing energy and scattering angle are drawn from the posterior LUTs for the current
   material/energy bin, corrected for the local voxel density (`DENSCORR.raw`) via the
   density-scaling relationship between density and stopping power.

The GPU kernel uses a one-thread-per-history strategy (each CUDA thread simulates one complete
history, from emission to annihilation or energy cutoff), with the annihilation map accumulated
via atomic adds, the annihilation-probability table cached in shared memory, geometry constants
in constant memory, and histories batched to stay within device thread limits. The same source
compiles for a CPU-only execution path, useful for validation or environments without a GPU.

## Repository layout

```
SOURCES_v9/                  CUDA Fortran source
  hibrido-positrons-v9.cuf     main program: I/O, table loading, batching, GPU/CPU dispatch
  parte_CPU-positrons.f        CPU transport kernel (condensed-history algorithm)
  parte_GPU-positrons_d.f      GPU transport kernel (device code, CUDA Fortran)
  compila_sharable.sh          build script (nvfortran)
tablas-positrones/            Pre-computed look-up tables (see below)
scripts/
  voxel_processor.py            CT/µ-map -> MATERIAL.raw, DENSITY.raw, DENSCORR.raw
  build_30mat_tables.py         raw PENELOPE per-history data -> LUTs (offline table generation)
  aliasing.py / aliasing.sh     step-length aliasing sensitivity sweep
  benchmark.sh                  CPU benchmark/comparison harness across source versions
  compare_annihilation.py       compare an ANNIHILATION.raw output against a reference
  multirun.py                   batch-run multiple isotopes/spectra
spectra/                      example β⁺ emission spectra (Cu64, F18, Ga68, I124, Rb82)
config.in                     runtime configuration (seeds, batching, transport cutoff, ...)
spectrum.txt                  active spectrum file (copied from spectra/ before a run)
hybrid_positron_UCM.x         pre-built executable
```

## Requirements

- NVIDIA HPC SDK (`nvfortran`) — developed against 26.1.
- An NVIDIA CUDA-capable GPU for the GPU path (optional; the CPU path requires no GPU). The
  provided build targets compute capabilities sm_50 through sm_121 (Maxwell through Blackwell)
  plus PTX for forward JIT compatibility with newer architectures; anything older than Maxwell
  is unsupported by the underlying CUDA toolchain.
- ~2.5–3 GB of free GPU memory at minimum (mostly the fixed LUT footprint, independent of image
  size; see [Memory footprint](#memory-footprint)).
- Python 3 with `numpy`/`scipy` for the preprocessing and analysis scripts.

## Building

```bash
cd SOURCES_v9
sh compila_sharable.sh
```

This produces `hybrid_positron_UCM.x` in the repository root.

## Inputs

All input files are read from the working directory:

| File | Type | Description |
|---|---|---|
| `MATERIAL.raw` | `int32` | Voxelised material-ID map |
| `DENSCORR.raw` | `float32` | Voxelised density-correction factor λ = ρ_real / ρ_reference |
| `DECAY.raw` | `float32` | Voxelised number of decays per voxel (source activity) |
| `spectrum.txt` | ASCII, 2 columns | β⁺ emission spectrum: energy (eV), probability |
| `config.in` | ASCII | RNG seeds, GPU threads/histories-per-thread, CPU batch size, transport energy cutoff, anti-aliasing step variation |

`MATERIAL.raw` and `DENSCORR.raw` are generated from a patient CT or µ-map with:

```bash
python3 scripts/voxel_processor.py <ct_or_umap.raw> --type ct --shape NX NY NZ [--blur-sigma 1.0]
```

which segments the volume into the tissue library by nearest physical density (via a piecewise
HU→density calibration) and writes `MATERIAL.raw`, `DENSITY.raw`, `DENSCORR.raw`, and a blurred
`DENSCORR.blur.raw`.

## Running

```bash
./hybrid_positron_UCM.x <device> <Nx> <Ny> <Nz> <dx> <dy> <dz>
```

- `device`: `1` for GPU execution, `0` for CPU.
- `Nx Ny Nz`: number of voxels along each axis.
- `dx dy dz`: voxel spacing in cm.

Output is a 3D annihilation map, `ANNIHILATION.{gpu,cpu}.raw` (`float32`), giving the posterior
spatial distribution of annihilation events — usable directly as input to PET image
reconstruction or resolution modelling (e.g. as a spatially variant PSF kernel).

## Tissue and energy tables

Interaction LUTs are generated offline by running detailed event-by-event PENELOPE simulations
per material and energy bin (10⁵ histories, 0.5 mm fixed step), recording:

- **Annihilation probability** per (material, energy) — used to decide whether a positron
  terminates at each transport step.
- **Posterior energy distribution** — sampled outgoing kinetic energy after a step.
- **Posterior angular distribution** — sampled scattering angle (stored as 1 − cos θ) after a step.
- **Mean deposited energy** per step, on a 1-keV linear grid, used for the density-correction scaling.

This repository's tables (`tablas-positrones/*-30MAT.*`) cover **30 Schneider reference-tissue
materials** (0.027–1.837 g/cm³) on a **103-point natural-log energy grid from 1 keV to 4.6 MeV**,
built from raw per-history PENELOPE output via `scripts/build_30mat_tables.py`. The original
24-material/10 keV–3.55 MeV tables from the published paper are still present
(`*-24MAT.*`) but unused by the current build.

## Memory footprint

The dominant GPU memory cost is the fixed LUT footprint (independent of image size): the posterior
energy and angular tables are each a `100000 × 103 × 30` `float32` array (~1.24 GB), ~2.47 GB
combined, loaded into GPU memory at initialisation. Per-voxel maps (`MATERIAL.raw`, `DENSCORR.raw`,
`ANNIHILATION.raw`) scale linearly with image size (4 bytes/voxel each). A CUDA-capable GPU with
at least 4 GB of VRAM is recommended.

## Validation and performance (published results, 24-material configuration)

The published paper validates HMC (both CPU and GPU backends) against PenEasy 2024 across five
scenarios — point-source profiles/convergence, heterogeneous sphere interfaces, a clinical
[¹⁸F]FDG-derived ⁸²Rb myocardium benchmark, and runtime scaling — for ¹⁸F, ⁶⁸Ga, ⁸²Rb, and ¹²⁴I:

- Point-source FWHM/FWTM residuals vs. PenEasy below 0.3 mm at 10⁷ histories.
- Myocardium benchmark: voxelwise **R² = 0.995**, **SSIM = 0.995**, **PSNR ≈ 36.4 dB** against
  PenEasy 2024, while reducing simulation time from ~8 h 45 min (PenEasy, single-thread) to
  ~14 min (HMC CPU) or **~1 s (HMC GPU)**.
- Terminal throughput: 3.32·10⁸ histories/s (HMC GPU, NVIDIA RTX 5080) vs. 4.77·10⁵ histories/s
  (HMC CPU, Intel i9-14900KF) vs. 1.41·10⁴ histories/s (PenEasy 2024) — a combined **~23,600×**
  speedup of HMC GPU over single-thread PenEasy 2024 (34× from the condensed-history
  approximation, 697× from GPU parallelism).
- Code-to-code disagreement concentrates near sharp material interfaces, attributed to the 0.5 mm
  LUT generation step being coarse relative to abrupt annihilation-probability changes there.

See the paper for full methodology, figures, and discussion.

## Auxiliary scripts

- `scripts/aliasing.sh` / `aliasing.py` — sweep the anti-aliasing step-variation percentage in
  `config.in` and compare resulting profiles to a reference.
- `scripts/benchmark.sh` — build and run multiple `SOURCES_*` versions back-to-back, reporting
  wall-clock time and simulation speed.
- `scripts/compare_annihilation.py` — compare a candidate `ANNIHILATION.raw` against a reference
  (sum ratio, RMSE).
- `scripts/multirun.py` — batch-run a list of isotopes using the spectra in `spectra/`.

## Citation

If you use this code, please cite:

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

## License

MIT License — see [LICENSE](LICENSE).
