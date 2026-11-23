#!/bin/sh
# Build decays_positron.x from decays_positron.f (PENNUC main: decay paths of a radionuclide -> spcnuc.dat, positrons.dat).
# PENNUC itself (pennucm.f, the PENNUC 2023 subroutine package) and the PENELOPE routines it calls (RITA, RAND) come from a
# PenSuite install, not from this repo:   PENSUITE=<dir holding penelope/>  (default ~/MC-codes/PenSuite-2023)   FC=ifx|gfortran
# The PENNUC data directory is set to nuclides/ (RPATH), the working-directory layout generate_spectra.py uses.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PENSUITE=${PENSUITE:-$HOME/MC-codes/PenSuite-2023}
FC=${FC:-$(command -v ifx || command -v gfortran)}
SRC=$PENSUITE/penelope
BUILD=$HERE/build
mkdir -p "$BUILD"
cp "$HERE/decays_positron.f" "$BUILD/"
# NBM (max. branches per daughter) 200 -> 400: the Y-86 file made from ENSDF (spectrum/input/extra/) has 288 branch records, the DDEP ones fewer
sed -e "s|CHARACTER\*2 :: RPATH='./'|CHARACTER*9 :: RPATH='nuclides/'|" \
    -e "s|INTEGER\*4, PARAMETER :: NBM=200 |INTEGER*4, PARAMETER :: NBM=400 |" "$SRC/mains/penmain/pennucm.f" > "$BUILD/pennuc.f"
grep -q "RPATH='nuclides/'" "$BUILD/pennuc.f" || { echo "build.sh: RPATH line not found in pennucm.f (PENNUC version changed?)" >&2; exit 1; }
grep -q "PARAMETER :: NBM=400 " "$BUILD/pennuc.f" || { echo "build.sh: NBM line not found in pennucm.f (PENNUC version changed?)" >&2; exit 1; }
cp "$SRC/fsource/rita.f" "$SRC/fsource/penelope.f" "$BUILD/"
cd "$BUILD"
case $(basename "$FC") in
  ifx|ifort) $FC -O2 -fpp decays_positron.f rita.f penelope.f -o "$HERE/decays_positron.x" ;;
  *)         $FC -O2 -std=legacy -fallow-argument-mismatch decays_positron.f rita.f penelope.f -o "$HERE/decays_positron.x" ;;
esac
rm -f ./*.mod
echo "built $HERE/decays_positron.x ($(basename "$FC"))"
