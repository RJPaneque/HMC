#!/bin/sh
# Run one validation case with hmc.x and score it against the reference codes. From the repository root:
#   (CASE_ID: a row of validation/results/cases.csv)
#   validation/examples/run_example.sh t2_slab_Rb82_m12-m20_R5.25_dx0.50 [penelope|geant4|egsnrc|fluka] [work dir]
# Needs the Zenodo tables unzipped (luts/output/<engine>), python3 with numpy.
set -e
CASE=${1:?"usage: run_example.sh CASE_ID [table engine] [work dir]"}
ENGINE=${2:-penelope}
WORK=${3:-work/$CASE}
V=validation
python3 $V/tools/make_hmc_inputs.py $CASE --out "$WORK" --tables luts/output/$ENGINE
hmc/hmc.x --config "$WORK/hmc.in" --quiet
REF=$V/examples/reference_maps/$CASE.pvoxel.npz
if [ -f "$REF" ]; then python3 $V/tools/check_case.py $CASE "$WORK/annihilation.raw" --reference "$REF"
else python3 $V/tools/check_case.py $CASE "$WORK/annihilation.raw"; fi
