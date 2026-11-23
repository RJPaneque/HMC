#!/bin/bash
# Run the Geant4 step driver (image luts-geant4) for a range of materials, one container per material, N in parallel,
# into an output directory of its own; resumable (2026-10-30).
#
#   luts/src/sim_geant4/run_mats.sh OUT [--mats A-B|A,B,...] [--jobs N] [--nhist N] [--step CM] [--image IMG]
#                                       [--e-index LIST]
#
# Per material m: OUT/work/mat<m>/sim.in = luts/input/sim.in with NHIST (and STEP_CM) replaced plus "MAT_LIST m",
# "OUT_TAG run"; the container writes OUT/work/mat<m>/raw_extra/mat<m>_run.dat, which is moved to OUT/raw/mat<m>.dat only
# when the container exits 0 (the driver's output for material m depends on (sim.in, SEED, m) only, not on MAT_LIST, so the
# file equals the one of a full run). A material whose OUT/raw/mat<m>.dat exists is skipped, so a rerun resumes.
# Logs: OUT/logs/mat<m>.log. Containers carry the label hmc2.lutsrun=<OUT basename>; on Ctrl-C / TERM this script stops
# its own containers by ID.
#
# --e-index LIST appends "E_INDEX_LIST LIST" (0-based energy indices, tests). Without it sim.in is exactly as before.
# Environment SEED=n, CUT_UM=x, CUT_FLOOR_EV=e append the driver keywords of the same names (seed base; range cut and
# lower edge of the production-threshold table); unset, the driver defaults (4000, 1 nm, 100 eV) apply.
# MAX_STEP_UM=x caps every true step at x um (driver keyword MAX_STEP_FRAC = x um / STEP_CM, at most 1): without it the
# boundary-free world lets msc steps grow, and with CUT_UM=1 / CUT_FLOOR_EV=1000 low-energy stop distances come out ~9 %
# short (2026-11-02, 94 keV in mat12: 79.7 um uncapped, 88.4 um with a 1 um cap).
#
# Optional thermal guard (local shared machine only; off unless T_START is set): a new material starts only while the
# `sensors` "Package id 0" temperature is below T_START (C). The cloud VM needs none.
#
# Build tables afterwards with: python luts/build_luts.py --engine geant4 --raw-dir OUT/raw --out-dir OUT
set -euo pipefail
LUTS="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
out="$(readlink -m "${1:?usage: run_mats.sh OUT [--mats A-B] [--jobs N] [--nhist N] [--step CM]}")"; shift
mats=""; jobs=4; nhist=""; step=""; image="${IMAGE:-luts-geant4}"; eidx=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mats) mats="$2"; shift 2 ;;
        --jobs) jobs="$2"; shift 2 ;;
        --nhist) nhist="$2"; shift 2 ;;
        --step) step="$2"; shift 2 ;;
        --image) image="$2"; shift 2 ;;
        --e-index) eidx="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
mats_dir="$(readlink -f "$LUTS/input/mats")"
if [[ -z $mats ]]; then n=0; while [[ -f $mats_dir/mat$((n + 1)).mat ]]; do n=$((n + 1)); done; mats="1-$n"; fi
list=()
for part in ${mats//,/ }; do
    if [[ $part == *-* ]]; then for ((m = ${part%-*}; m <= ${part#*-}; m++)); do list+=("$m"); done; else list+=("$part"); fi
done
docker image inspect "$image" >/dev/null || { echo "image $image missing: FORCE_BUILD=1 luts/src/sim_geant4/run_geant4.sh or docker build" >&2; exit 1; }
mkdir -p "$out/raw" "$out/logs" "$out/work"
label="hmc2.lutsrun=$(basename "$out")"
echo "$(date +%T) geant4 -> $out: materials ${list[*]}, jobs $jobs, nhist ${nhist:-sim.in}, step ${step:-sim.in}${eidx:+, e-index $eidx}, image $image"

cleanup() { local ids; ids=$(docker ps -q --filter "label=$label"); [[ -n $ids ]] && docker stop $ids >/dev/null; exit 130; }
trap cleanup INT TERM
temp() { sensors 2>/dev/null | awk '/Package id 0/{match($0,/[0-9]+\.[0-9]/); print int(substr($0,RSTART,RLENGTH)); exit}'; }

run_one() {
    local m=$1 w="$out/work/mat$1"
    [[ -e $w ]] && mv "$w" "$w.prev.$(date +%Y%m%d%H%M%S)"     # a failed earlier attempt: kept, never deleted
    mkdir -p "$w"
    awk -v n="$nhist" -v s="$step" '
        /^[[:space:]]*(!|$)/ { print; next }
        { k++ } k == 1 && s != "" { printf "%s   ! STEP_CM (run_mats.sh)\n", s; next }
        k == 2 || k == 3 || k == 4 { print; next }
        k == 5 && n != "" { printf "%s   ! NHIST (run_mats.sh)\n", n; next }
        k == 6 { print "input/mats   ! MAT_DIR (mounted)"; next }
        { print }' "$LUTS/input/sim.in" > "$w/sim.in"
    printf 'MAT_LIST %s\nOUT_TAG run\n' "$m" >> "$w/sim.in"
    [[ -n $eidx ]] && echo "E_INDEX_LIST $eidx   ! (run_mats.sh)" >> "$w/sim.in"
    for k in SEED CUT_UM CUT_FLOOR_EV; do [[ -z ${!k:-} ]] || echo "$k ${!k}   ! (run_mats.sh)" >> "$w/sim.in"; done
    if [[ -n ${MAX_STEP_UM:-} ]]; then     # first non-comment line of sim.in is STEP_CM
        cap=$(awk -v um="$MAX_STEP_UM" '/^[[:space:]]*(!|$)/ { next } { f = um * 1e-4 / $1; if (f > 1) f = 1;
            printf "MAX_STEP_FRAC %.6g   ! %s um cap (run_mats.sh)", f, um; exit }' "$w/sim.in")
        echo "$cap" >> "$w/sim.in"
    fi
    # hmc2.validation / hmc2.watch: validation/src/drivers/thermal_watch.py stops these containers on a trip
    if docker run --rm --init --label "$label" --label hmc2.validation=1 ${HMC2_WATCH_ID:+--label hmc2.watch=$HMC2_WATCH_ID} --user "$(id -u):$(id -g)" \
            -v "$mats_dir:/work/input/mats:ro" -v "$w:/work/output/geant4" -e TABLEGEN_IN=/work/output/geant4/sim.in \
            "$image" > "$out/logs/mat$m.log" 2>&1 && [[ -s $w/raw_extra/mat${m}_run.dat ]]; then
        mv "$w/raw_extra/mat${m}_run.dat" "$out/raw/mat$m.dat"     # (OUT/work/mat<m>/sim.in stays as the record of the run)
        echo "$(date +%T) mat $m done"
    else
        echo "$(date +%T) mat $m FAILED (see $out/logs/mat$m.log)" >&2; return 1
    fi
}

t0=$(date +%s); rc=0; pids=()
for m in "${list[@]}"; do
    if [[ -f $out/raw/mat$m.dat ]]; then echo "mat $m already done"; continue; fi
    while [[ $(jobs -rp | wc -l) -ge $jobs ]]; do sleep 2; done
    if [[ -n ${T_START:-} ]]; then while t=$(temp) && [[ -n $t && $t -ge $T_START ]]; do sleep 20; done; fi
    run_one "$m" & pids+=($!)
    echo "$(date +%T) started mat $m (pid $!)"
done
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "$(date +%T) all done in $(( $(date +%s) - t0 )) s (rc $rc)"
exit $rc
