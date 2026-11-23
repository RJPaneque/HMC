# spectrum/ — β⁺ emission spectra

```
spectrum/
  generate_spectra.py -> src/generate_spectra.py     python spectrum/generate_spectra.py F18 Ga68 [--samples 1e7] [--out DIR] [--format hmc|peneasy] [--force]
  input/nuclides      the PENNUC nuclide database: NOT distributed, link it here (see below)
  input/extra/        Y-86.nuc, made from ENSDF (the database has none); used before the database's files
  ensdf2nuc/          the ENSDF -> .nuc converter, its checks and the report of the Y-86 conversion (README.md there)
  output/spectra/     C11 N13 O15 F18 Na22 Sc44 Mn52 Cu64 Ga66 Ga68 Br76 Rb82 Y86 Zr89 I124 .txt  (the files of hmc/input/spectra; legacy/spectra has the seven of HMC-1)
  src/                generate_spectra.py, decays_positron.f (our main program around PENNUC), build.sh
  tests/              pytest spectrum/tests   (-m "not slow" skips the PENNUC runs)
```

`src/build.sh` builds `decays_positron.x` from `decays_positron.f` and, from a PenSuite install (`PENSUITE=<dir>`), PENNUC
(`pennucm.f`, the subroutine package of PENELOPE's distribution) with its data directory set to `nuclides/`, plus the PENELOPE routines it
calls (RITA, RAND). `generate_spectra.py` runs it in a scratch directory (1e7 decays, about 3 s), takes the positron column of the
400-bin energy table (`spcnuc.dat`), normalises it and writes `<isotope>.txt`: energy (eV) and probability, no header (`--format peneasy`
adds the header and the closing `-1` row of PenEasy's source-spectrum convention). Existing files are not overwritten without `--force`.

To regenerate a spectrum you need PenSuite 2023 (or PenEasy 2024) from the NEA Data Bank, which has the PENNUC source and the nuclide
database (`nuclides/*.nuc`, with `pdatconf.p14` and `pdrelax.p11`); link its `nuclides/` folder to `spectrum/input/nuclides`. The
database is not distributed here, the spectra computed from it are.

The shipped files are older than this pipeline: Mn52 and Zr89 are verbatim output of the original `decays_positron.x` (PENNUC 2018);
Cu64, F18, Ga68, I124 and Rb82 were rounded to 3–4 digits and start with a `3.0e6 0` row. Rebuilt with PENNUC 2023 the pipeline gives
the same spectra at the level of the statistical noise of 1e7 decays (L1 over the bins ≤ 2e-4, mean energy within 1.4e-6 for Mn52 and
Zr89, 4e-5 for the rounded F18); `tests/` asserts that.
