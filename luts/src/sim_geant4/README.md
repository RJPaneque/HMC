# GEANT4 simulation driver — physics and path-length settings

Run with `src/sim_geant4/run_geant4.sh` (from any directory) (`FORCE_BUILD=1` after editing the
driver). Output: `output/geant4/raw/matN.dat`, same schema as the PENELOPE driver.

Since 2026-10-30 the rows have a 10th column `dphi_rad` = atan2(x v − y u, x u + y v) (end position (x, y), final direction
(u, v); 0 when either vanishes), and the config line records `engine_version=` (from G4Version.hh; 11.3.2). The first nine
columns are bit-identical to the 9-column driver (materials 12 and 30, 1e5 histories); the previous image is tagged
`luts-geant4:9col-20261029`. `run_mats.sh OUT [--mats A-B] [--jobs N] [--nhist N] [--step CM]` runs one container per
material into OUT/raw/matN.dat (N in parallel, resumable, optional thermal guard with T_START).

## Consistency fix (Oct 2026)

The previous driver (G4EmPenelopePhysics, no step limiter) disagreed with the
PENELOPE driver by ~43% in the survivors' mean dE_kin and ~20% in mean theta
(up to 2x and 45% at MeV energies). Changes:

1. **Exact path length.** `G4StepLimiterPhysics` is registered and a
   `G4UserLimits` is attached to the world volume; `EventAction` resets its
   max step to STEP_CM and `SteppingAction` lowers it to the *remaining*
   path after every step, so the last step ends at STEP_CM (within 1 nm).
   Previously the last step overshot by up to one physics step, inflating
   dE and theta. This was the dominant error.
2. **Physics list.** `PHYSICS LIV` (default, G4EmLivermorePhysics),
   `PHYSICS OPT4` (G4EmStandardPhysics_option4) or `PHYSICS OPT3`.
3. **Tracking cut** `LOWEST_E_EV 50` (PENELOPE EABS = 50 eV; standard default 1 keV).
4. **Production threshold** `CUT_UM 0.001` (1 nm) with table floor `CUT_FLOOR_EV 100`,
   so every material's threshold sits at the 100 eV floor
   (PENELOPE WCC = WCR = 100 eV; Geant4 default floor 990 eV).
5. `STEP_LIMIT 0` and `MAX_STEP_FRAC f` exist only for A/B and convergence tests.
6. **Mean excitation energy** taken from the `.mat` file (PENELOPE's I), not
   computed by Geant4. In practice Geant4's own value was already the same:
   outputs changed only in the last printed digit.
7. **Seeding.** Material N is seeded with `SEED + N` (default `SEED 4000`), the
   same rule as the PENELOPE driver (ISEED1 = ISEED2 = SEED + N).

The driver prints `CUTS mat=N I=... e-=...` once per material: the energy
thresholds the range cut turned into. A 1 um cut (the earlier default) gave
100 eV only for materials 1-6, rising to 628 eV for material 30. Moving to
1 nm changed the agreement with PENELOPE only slightly (E50 shift 0.60% ->
0.52%, mean theta 2.7% -> 3.0%, dE_kin unchanged); it is kept for
consistency of the setup.

Optional config keywords (after the six positional lines of `sim.in`;
path overridable with `TABLEGEN_IN`): `NSURV_MIN`, `NHIST_MAX`, `MAT_LIST`,
`E_INDEX_LIST`, `SEED`, `OUT_TAG`, plus the ones above. A `MAT_LIST` or
`E_INDEX_LIST` run writes `output/geant4/raw_extra/matN_<OUT_TAG>.dat`.

## Validation against PENELOPE (30 materials, 116 energies x 1e5 histories)

Errors are |G4/PEN - 1| over energies with >= 1000 PENELOPE survivors,
averaged over materials (worst material in parentheses). E50 is the energy at
which half of the positrons annihilate within STEP_CM (a range proxy).

| driver            | mean dE_kin   | mean theta    | E50 shift  |
|-------------------|---------------|---------------|------------|
| old (Penelope EM) | 42.6% (71%)   | 20.1% (31%)   | <= 0.5%    |
| fix, option4      | 2.4% (4.3%)   | 2.5% (4.1%)   | up to 5.0% |
| fix, Livermore    | 1.2% (1.6%)   | 2.7% (3.9%)   | <= 0.9%    |
| + 1 nm cut (prod) | 1.2% (1.8%)   | 3.0% (3.9%)   | <= 0.9%    |

On material 13, limiting every step to STEP_CM/4 changed nothing (step-size
converged), and option3 gave a 12% theta error. The remaining ~3% theta
difference comes from the msc models, not from stepping.
