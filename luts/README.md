# luts/ — building the look-up tables

`luts/` runs event-by-event Monte Carlo positron transport and compacts it into the tables that `hmc.x` samples from: for each of the
30 materials and 116 energies (1 keV to 4.6 MeV, log-spaced), 10⁶ positrons are followed over a fixed true path length (the *step*),
and the final direction, the energy lost, whether they annihilated and where they ended are recorded. The raw histories are then
compacted into conditional quantile tables (`ann_LUT.raw`, `scatt-eloss_LUT.raw`, `stepend_LUT.raw`). The result is what the Zenodo
record holds. The binary format and the sampling contract `hmc.x` follows are in [LUT_FORMAT.md](LUT_FORMAT.md).

| engine | here | what it needs |
|---|---|---|
| PENELOPE | `src/sim_penelope/` (`run_penelope.f`, `COMPILAR.sh`, `run_step_variant.sh`) | the Intel `ifx` compiler and the PenSuite 2023 sources (NEA Data Bank) |
| Geant4 | `src/sim_geant4/` (`run_geant4.cc`, `CMakeLists.txt`, `Dockerfile`, `run_geant4.sh`, `run_mats.sh`) | Docker: the image builds Geant4 11.3.2 from source on first use |
| EGSnrc, FLUKA | not distributed | their drivers depend on modules and images of the validation harness (and FLUKA's on user routines that its licence does not let us share); the settings used are listed below |

```
luts/run_sim.sh --engine penelope --levels 4 --nhist 1000000 --seed 7000 --jobs 30 --no-thermal     # raw data + tables of the step ladder
luts/run_sim.sh --engine geant4  --levels 4 --nhist 1000000 --seed 7000 --no-thermal
python luts/build_luts.py --engine penelope --raw-dir DIR --out-dir DIR     # only the compaction, from raw data
python luts/check_luts.py penelope/step1-1                                  # figures and statistics of one table set
python luts/compare_luts.py penelope/step1-1 geant4/step1-1                 # two sets against each other
pytest luts/tests -m "not slow"
```

`input/sim.in` has the six lines (its histories line is 10⁵; the production runs pass `--nhist 1000000`) that drive every engine (step length, minimum and maximum energy, number of energies, histories per
energy, material folder; `!` starts a comment). The material folder holds `mat1.mat ... mat30.mat`: `luts/input/mats` is a link you make
to `materials/output/30mats`. `run_sim.sh` (`src/run_bundle.py`) simulates the **step ladder** of an engine, `--levels 4` halvings of the
step of `sim.in` (0.05, 0.025, 0.0125 and 0.00625 cm), and compacts each step into `output/<engine>/step1-<n>/`; a step folder that holds
tables is never overwritten without `--force`. Add `--no-thermal` unless you have the harness' thermal watch (container engines run under
it by default). 10⁶ histories need about 0.4 TB of raw text per PENELOPE step: the shipped tables were made on a cloud machine.

**Settings of the shipped tables** (made 2026-11-01 to 11-03; 10⁶ histories per material and energy, seed base 7000, `--frame 1`, tail
knots, the defaults of `build_luts.py`):

- PENELOPE: PenSuite 2023, the 30 `.mat` files, absorption energies 50 eV, C1 = C2 = 0.01, WCC = WCR = 100 eV, a single process per material
  with the seed `base + material number`.
- Geant4 11.3.2: `G4EmLivermorePhysics`, production cut 1 µm with a floor of 1 keV, a step cap of 1 µm (`CUT_UM=1`, `CUT_FLOOR_EV=1000`,
  `MAX_STEP_UM=1`).
- EGSnrc v2025a (egs++): media built from the `.mat` compositions without PEGS4 files, PRESTA-II, exact boundary crossing, skin depth 3,
  spin on, NRC bremsstrahlung, ECUT = AE = 0.512 MeV total (1 keV kinetic), ESTEPE 0.25, XIMAX 0.5; photons never produced, every secondary
  killed at creation (its energy counts in the loss); the true path length of the step is enforced with `Global Smax` and a deferred discard.
- FLUKA 4-5.2: `DEFAULTS EM-CASCA`, e+/e- transport cut-off and production thresholds 1 keV kinetic, photon production 1 keV, `EMFFIX` 0.05,
  `IONFLUCT` accuracy 4, spin-relativistic multiple-scattering corrections, `MULSOPT GLOBEMF 1 1 0 1 1 1`, electrons and photons discarded;
  compositions and densities of the `.mat` files with their mean excitation energies; a user routine kills the positron when the step is done.

The materials are in [`../materials/`](../materials/README.md), the spectra of the isotopes in [`../spectrum/`](../spectrum/README.md).
