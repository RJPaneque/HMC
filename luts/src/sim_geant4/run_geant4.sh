#!/bin/bash
# Build (if needed) and run the GEANT4 simulation driver in Docker.
#
# Reads luts/input/ (shared with PENELOPE: sim.in, mats/) and
# writes to luts/output/geant4/raw/. Mirrors running
# src/sim_penelope/run_penelope.x from luts/ for PENELOPE.
#
# Usage:
#   src/sim_geant4/run_geant4.sh                # build image if missing, then run
#   FORCE_BUILD=1 src/sim_geant4/run_geant4.sh  # rebuild image even if it exists

set -euo pipefail
cd "$(dirname "$0")/../.."  # luts/

IMAGE=luts-geant4

if [[ "${FORCE_BUILD:-0}" == "1" ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Building $IMAGE image (this compiles GEANT4 from source; first run only)..."
    docker build -t "$IMAGE" src/sim_geant4/
fi

if [[ ! -d output/geant4/raw ]]; then
    mkdir -p output/geant4/raw
fi 

# input/mats is a symlink out to ../materials/output/30mats; a plain bind
# mount of input/ leaves that symlink dangling inside the container (it
# resolves relative to the container root, not the host), so bind the
# real target directly over the mount point as well.
MATS_DIR="$(readlink -f input/mats)"

# Run as the invoking host user so output files aren't left root-owned
# (the image's ENTRYPOINT otherwise runs as root by default).
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$(pwd)/input:/work/input:ro" \
    -v "$MATS_DIR:/work/input/mats:ro" \
    -v "$(pwd)/output/geant4:/work/output/geant4" \
    "$IMAGE"
