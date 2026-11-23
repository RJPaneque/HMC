#!/bin/bash
# Build the PENELOPE simulation driver.
#
# PENELOPE sources are included from the PenSuite-2023 distribution:
#   ~/MC-codes/PenSuite-2023/penelope/fsource/
#
# The executable is written next to its source (gitignored), so this script
# can be invoked from anywhere. Run it from luts/: the driver reads
# input/sim.in and writes output/penelope/raw/ and
# output/penelope/run_penelope.err relative to the working directory.
#
# Usage:
#   ./COMPILAR.sh                 # builds ./run_penelope.x
#   outX=/some/path ./COMPILAR.sh # builds to custom location

set -euo pipefail
cd "$(dirname "$0")"

outX=${outX:-run_penelope.x}

ifx run_penelope.f -o "$outX" -O2 -fpp -static-intel \
    -I"$HOME/MC-codes/PenSuite-2023/penelope/fsource/" \
    -diag-disable 8290,8291

rm -f *.mod
