# Third-party notices

`hmc/bin/hmc-cuda12-x86_64` was built with the NVIDIA HPC SDK 26.1 (nvfortran, CUDA 12.9 toolchain) and `hmc/bin/hmc-cuda13-x86_64` with the same
SDK and its CUDA 13.1 toolchain; both statically contain the NVIDIA and OpenMP components below (`hmc-cuda13` also uses the system's `libstdc++`).
`hmc/bin/hmc-cpu-x86_64` was built with GNU Fortran and musl, contains none of the NVIDIA components and has its own section at the end.
`hmc/hmc.x` is a shell script that starts one of them. `legacy/v8.1-24mat/hybrid_positron_UCM.x` and `legacy/v9-30mat/HMC_UCM.run` (HMC-1, MIT) are
linked dynamically (they load the NVIDIA HPC SDK run-time libraries installed on your system) and ship none of these components.

## NVIDIA runtime (CUDA and HPC SDK run-time libraries)

Copyright NVIDIA Corporation. Linked into `hmc-cuda12-x86_64` and `hmc-cuda13-x86_64` under the *NVIDIA Software License Agreement for HPC Software Development Kit*
(section 2, distribution of software compiled or linked into an application), which is why part 2 of `LICENSE` forbids reverse
engineering, notice removal and separate use or redistribution of these components, and disclaims all warranties on NVIDIA's behalf.
They may be used only as part of these executables, on the "Target Systems" that agreement defines (systems running an operating system the SDK supports).
The CUDA runtime library (`libcudart_static`) inside the two GPU executables is distributed under the *CUDA Toolkit End User License Agreement*
(https://docs.nvidia.com/cuda/eula/index.html), which lists it among the redistributable components; the conditions above apply to it in the same way.

## OpenMP runtime

The OpenMP runtime of the HPC SDK (`nvomp`) derives from the LLVM OpenMP runtime, Copyright (c) 1997-2016 Intel Corporation, dual
licensed under the University of Illinois/NCSA Open Source License and the MIT license: part 4.5 of `LICENSE`, and for LLVM
itself part 4.4 (both as shipped by NVIDIA with the SDK: the LLVM project itself has used the Apache License 2.0 with LLVM exceptions since 2019,
the texts in parts 4.4 and 4.5 are those of the SDK's licence files).

## hwloc 2.0.3

The OpenMP runtime includes the Portable Hardware Locality library, BSD 3-Clause: part 4.6 of `LICENSE`.

## Philox random number generator

`hmc.x` draws its random numbers with Philox4x32-10 (Salmon, Moraes, Dror and Shaw, "Parallel random numbers: as easy as 1, 2, 3",
Proceedings of SC '11, 2011), implemented independently for HMC. The known-answer vectors it is tested against come from Random123
(D. E. Shaw Research, BSD 3-Clause); no Random123 code is distributed.

## Physics data

- The condensed-history tables (Zenodo record) are numerical results of simulations with PENELOPE2023 (OECD Nuclear Energy Agency Data Bank
  package NEA-1525, https://www.oecd-nea.org/tools/abstract/detail/nea-1525/; the "PenSuite 2023" named in the READMEs is this package):
  F. Salvat, *PENELOPE2023, a code system for Monte Carlo simulation of electron and photon transport*, NEA package NEA-1525 (version 2023),
  and for the previous version F. Salvat, *PENELOPE-2018: A Code System for Monte Carlo Simulation of Electron and Photon Transport*, OECD Nuclear
  Energy Agency, NEA/MBDAV/R(2019)1, https://doi.org/10.1787/32da5043-en. PENELOPE itself is not part of this repository; please cite it when you
  use the tables.
- The alternative tables were computed with Geant4 11.3.2 (Agostinelli et al., *Nucl. Instrum. Methods A* 506 (2003) 250-303; Allison
  et al., *IEEE Trans. Nucl. Sci.* 53 (2006) 270-278 and *Nucl. Instrum. Methods A* 835 (2016) 186-225), EGSnrc v2025a (Kawrakow,
  *Med. Phys.* 27 (2000) 485-498; the EGSnrc software record https://doi.org/10.4224/40001303, which names the software and not release v2025a; NRC Canada) and FLUKA 4-5.2 (Ahdida et al., *Front. Phys.* 9
  (2022) 788253; Battistoni et al., *Ann. Nucl. Energy* 82 (2015) 10-18; Hugo et al., *EPJ Nucl. Sci. Technol.* 10 (2024) 20, and the references at
  https://fluka.cern/documentation/references, as section 6 of the FLUKA licence requires). The
  tables are numerical results of runs with these programs; no code or data library of any of them, and no FLUKA user routine, is
  part of this repository or of the Zenodo record. Please cite the program whose tables you use.
- The tissue compositions behind the materials of the tables follow W. Schneider, T. Bortfeld and W. Schlegel, *Phys. Med. Biol.* 45
  (2000) 459.
- The beta+ emission spectra (`hmc/input/spectra`, `legacy/spectra`) were computed with PENNUC (a part of PENELOPE) from the
  nuclide decay data of its NUCLEIDE database, which are evaluated nuclear data (Laboratoire National Henri Becquerel, ENSDF).
  The Y-86 spectrum is computed from `spectrum/input/extra/Y-86.nuc`, which we made from the ENSDF evaluation of the 86Y ε+β+ decay (14.74 h; Nuclear
  Data Sheets 203, 283 (2025); data set 2020Gu18; National Nuclear Data Center) with `spectrum/ensdf2nuc/`; internal-conversion shares in that file come from
  BrIcc systematics as stored in the Geant4 PhotonEvaporation data, which are not distributed. The checks are in `spectrum/ensdf2nuc/README.md`.

## Validation package (`validation/`)

- The 30 `.mat` files of `materials/` are PENELOPE material files, generated with PENELOPE's `material` program;
  the notice of PENELOPE is in part 4.1 of `LICENSE`.
- The results tables hold the output of: PENELOPE (via our `pvoxel` program) and PenEasy (J. Sempau, A. Badal and L. Brualla, *Med. Phys.*
  38 (2011) 5887; version 2024), PenRed (V. Giménez-Alventosa, V. Giménez Gómez and S. Oliver, *Comput. Phys. Commun.* 267 (2021)
  108065, doi:10.1016/j.cpc.2021.108065), GATE/OpenGATE 10.0.2 (D. Sarrut et al., *Phys. Med. Biol.* 66 (2021) 10TR03) on Geant4 11.3.2, a voxel Geant4 11.3.2
  application, EGSnrc v2025a, FLUKA 4-5.2, and our GPU re-implementations of the PENELOPE (`gpen`), EGSnrc (`gegs`) and Geant4 (`gg4`)
  positron transport. Only numbers are distributed, no code, data library or user routine of any of them. Please cite the code whose
  results you use; for FLUKA, the references at https://fluka.cern/documentation/references.

## Reproducibility material (`materials/`, `luts/`, `spectrum/`)

- `luts/src/sim_penelope/run_penelope.f` is a driver that includes the PENELOPE routines at compile time from your own PenSuite install;
  no PENELOPE source is distributed. `spectrum/src/decays_positron.f` is a main program written for HMC from the documented interface of PENNUC: it
  contains no PENNUC or PENELOPE code, it calls PENNUC and PENELOPE's random-number generator, which `build.sh` compiles from your own install. PENNUC
  itself and the nuclide database are not distributed, and neither is `material`.
- `luts/src/sim_geant4/run_geant4.cc` links against Geant4 (Geant4 Software License), which the Docker image downloads and builds.
- The drivers for EGSnrc and FLUKA are not distributed (see `luts/README.md`); the tables made with them are.
- The figures of `validation/figures/` were drawn with Carlito (Copyright 2013 The Carlito Project Authors, SIL Open Font License 1.1), the
  metric-compatible open twin of Calibri; the font is not distributed.

## CPU-only executable (`hmc-cpu-x86_64`)

Built with GNU Fortran 15 (Alpine Linux) and linked statically with:

- **musl libc** 1.2, MIT licence, Copyright 2005-2020 Rich Felker et al.: part 4.2 of `LICENSE`.
- **libgfortran, libgomp (the GNU OpenMP runtime) and libgcc**, Copyright Free Software Foundation, GPL version 3 or later **with the GCC
  Runtime Library Exception 3.1** (part 4.3 of `LICENSE`), which allows the executable to be distributed under other
  terms because it was produced by the GCC compilation process alone. The source of GCC is at https://gcc.gnu.org.
- **libbacktrace** (linked in through libgfortran, which uses it for error tracebacks), BSD 3-Clause, Copyright 2012-2025 Free Software Foundation, Inc.,
  written by Ian Lance Taylor, Google: part 4.7 of `LICENSE`.
- libgfortran asks for **libquadmath** (LGPL) only to print 128-bit reals, which HMC never uses; the executable contains a one-function stub
  in its place and none of libquadmath.

No NVIDIA component is in this executable; part 2 of `LICENSE` applies to it all the same.
