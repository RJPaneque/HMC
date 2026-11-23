#!/bin/bash
# Simulate one engine's step-size ladder and build its tables: luts/output/<engine>/step1-<n>/ (raw/, tables, sim.json).
#
#   luts/run_sim.sh --engine penelope|egsnrc|geant4|fluka [--levels 4 | --steps ...] [--nhist N] [--seed S]
#                   [--mats A-B,C] [--jobs N] [--out ROOT] [--raw-only | --build-only] [--force] [--dry-run] ...
#
# Thin wrapper of luts/src/run_bundle.py (see its docstring, or --help). Step folders that already hold tables are
# skipped unless --force.
set -euo pipefail
exec python3 "$(dirname "$(readlink -f "$0")")/src/run_bundle.py" "$@"
