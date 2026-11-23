# ensdf2nuc — making a PENNUC `.nuc` file from an ENSDF data set

The PENNUC database (NUCLEIDE, DDEP evaluations) has no file for some positron emitters (Y-86, Cu-62, Tb-152, Ti-45, Co-55, As-72, Br-75 ...). This
folder converts an **ENSDF** EC/β⁺ decay data set into a `.nuc` file with the layout of the DDEP ones, so that
`spectrum/generate_spectra.py` (PENNUC, `decays_positron.x`) can compute the β⁺ spectrum of that nuclide. It is the baseline for making more of them.
First use: **`spectrum/input/extra/Y-86.nuc`**, made from `ensdf/Y86.3.ensdf` (NNDC, 86Y ε+β⁺ decay 14.74 h, ref. 2020Gu18, NDS 203, 283 (2025)),
which gives `spectrum/output/spectra/Y86.txt`.

```
ensdf2nuc.py   ENSDF data set -> .nuc          python ensdf2nuc.py ensdf/Y86.3.ensdf -o ../input/extra/Y-86.nuc --g4dir <PhotonEvaporation dir> [--feeding balance|ensdf]
ensdf.py       small ENSDF reader (levels, gammas, E/B records, conversion coefficients)
icc.py         BrIcc sub-shell systematics read from the Geant4 PhotonEvaporation level files (z<Z>.a<A>)
dhfs.py        Dirac-Hartree-Fock-Slater atom (numba) for the relative capture probabilities of sub-shells
check_nuc.py   independent structural check of a .nuc file (does not import the converter), optionally against the %IG of the ENSDF file
ensdf/         the ENSDF data sets used: Y86.3.ensdf (86Y, 14.74 h) and Mn52.3.ensdf (52Mn, 5.591 d, the DDEP file the atomic parameters were calibrated on)
report_Y86.txt the converter's own report for Y-86 (verbatim)
```

**Needs** Python with `numpy`, `scipy`, `numba` (all in the pixi environment: `pixi run python spectrum/ensdf2nuc/ensdf2nuc.py ...`) and the
Geant4 `PhotonEvaporation` data (not distributed; e.g. `.pixi/envs/gate/lib/python3.11/site-packages/opengate_core/geant4_data/PhotonEvaporation6.1`).
The converter and its report were written with Claude (Fable 5.2) at the author's request; the checks below were made independently afterwards (2026-11-20).

## What comes from ENSDF and what is modelled

Straight from ENSDF: level energies and half-lives, gamma energies and absolute intensities, K/L/M/N conversion coefficients, the Q value, the β⁺ and EC
shares. Modelled (ENSDF does not give them): the L1:L2:L3 split of the conversion coefficients (BrIcc systematics for the daughter) and the capture
sub-shell fractions (CK, CL1, CL2, CL3, CM, CN: DHFS amplitudes with exchange/overlap factors **calibrated on the DDEP Mn-52 file**). The level feeding is by default
the *gamma-intensity balance* (`--feeding balance`); `--feeding ensdf` takes the TI/IE printed on the E records instead.

**What matters for HMC.** The β⁺ spectrum depends only on the `BEP` records: the intensity, the endpoint (Q − 1022 keV − level energy) and the forbiddenness of each
branch. The capture fractions, conversion shares and gammas (X-ray, Auger and gamma emission) do not enter it. So the calibration on Mn-52 does not bias the spectra.

## Checks of the Y-86 conversion (independent of the converter, 2026-11-20)

| check | result |
|---|---|
| reproducibility | running `ensdf2nuc.py` on `ensdf/Y86.3.ensdf` gives a file byte-identical to `Y-86.nuc` |
| the report against the file | Q = 5240 keV, 53 levels, 288 branch records, 203 gammas, 444 conversion-electron records, β⁺ 0.2767 / EC 0.7233: all as in the file |
| `check_nuc.py` | no structural problem (energy balance of every branch, level/transition counts, gamma and electron lines connect their levels); γ intensities equal the ENSDF %Iγ for 202 of 203 lines (the other has none) |
| β⁺ intensity | Σ Iβ⁺ = 27.66 % of decays; ENSDF table 27.32 % (limits excluded); ENSDF quotes %β⁺ = 27.2 (20) from a 2022 measurement (the older evaluation had 32.5 (20)) |
| level by level | the ten levels with Iβ⁺ ≥ 0.3 % agree with the ENSDF table to 0.1–1.4 %, except 2877.90 keV (−6.8 %: gamma balance 0.41 % against the E record 0.44 %) |
| main branches | 1221 keV from level 2996.81, 10.37 % (ENSDF 10.39 %; measured endpoint 1248 (15) and 1180 (60) keV, 1965Va02 and 1962Ya01); 1988 keV from level 2229.50, 3.65 % (ENSDF 3.7 %; measured 2019 (20) and 1960 (20) keV) |
| highest endpoint | 3141 keV, 0.52 %, level 1076.63 keV, first-forbidden (ENSDF measured 3153 (40) keV, Iβ = 0.6 %) |
| mean β⁺ energy | PENNUC spectrum 613 keV; ENSDF per-level mean energies weighted with the intensities give 592 keV without the 1076.63 level and 607 keV with it |

**Sensitivity.** With `--feeding ensdf` the file has 281 branch records, β⁺ = 27.32 % and the spectrum mean is 596 keV (−2.8 %). The whole difference is the 0.52 % branch to the
1076.63 level, which ENSDF gives only as a limit (< 0.5 %) and as 0.6 (20) % from the gamma balance. The shipped file uses the balance (it includes the branch). So the uncertainty of the
Y-86 spectrum from the evaluation is a few percent in the mean energy and rests on that one branch.

**Not written** (ENSDF gives limits only, the converter writes no branch): level 1853.90 keV (β⁺ < 0.4 %, endpoint 2364 keV; the old spectra measured 0.6–1.3 % at 2335 keV),
levels 2641.15 and 4224.93 keV (negligible). The spectrum therefore has no feeding near 2.3 MeV beyond what the other branches give and ends at 3141 keV; at most a few percent
of the positrons are affected.

## Check against Mn-52 (the file the atomic parameters were calibrated on)

Converting `ensdf/Mn52.3.ensdf` and comparing with the DDEP file (evaluation of 22/11/2020, the one in the PENNUC database):

| | DDEP | from ENSDF |
|---|---|---|
| β⁺ to the 575 keV-endpoint level (main branch) | 30.85 % | 29.6 % |
| weak first-forbidden branches (1320, 922, 275 keV endpoints; 0.3 %, 0.01 %, 1e-6) | present | not written (the ENSDF set has no E record for those levels) |
| mean energy of the PENNUC spectrum | 247.1 keV | 243.4 keV (−1.5 %) |
| L1 distance between the two spectra (400 bins) | | 1.5 % |

The difference is mostly between the two evaluations (DDEP and ENSDF 2015 do not give the same feeding), not a converter error, and the spectra agree to ~1.5 %. This is the precision to expect
for a spectrum made this way: a few percent on the mean energy; the DDEP files remain the reference where they exist.

## Making another file

1. Get the ENSDF data set of the nuclide's decay (NNDC, `ensdf/` here), 2. run `ensdf2nuc.py` (above), 3. run `check_nuc.py FILE.nuc ensdf/DATASET.ensdf`, 4. put the file in `spectrum/input/extra/<El>-<A>.nuc`,
5. `python spectrum/generate_spectra.py <El><A>`, 6. compare the main branches, the β⁺ total and the mean energy with the ENSDF table as in the checks above, and add them to this README.
The PENNUC build used for the spectra has `NBM = 400` branches per daughter (the stock 200 is too small for the 288 records of Y-86).
