#!/bin/bash
# Run the PENELOPE step driver with another step length, into its own directory (TODO item 14, 2026-10-29).
#
# The driver reads input/sim.in and writes output/penelope/{raw,logs}/ relative to its working directory, so this
# script gives it a private working directory OUT/work/ (input/sim.in = luts/input/sim.in with STEP_CM replaced,
# input/mats -> luts/input/mats, output/penelope -> OUT) and runs one process per material. Raw files land in OUT/raw/,
# logs in OUT/logs/. luts/output/penelope/ and luts/input/sim.in are not touched.
#
# Thermal guard (shared, hot machine): a new material starts only when `sensors` "Package id 0" is below T_START
# (default 78 C) and fewer than MAXJ processes run; MAXJ is re-read from OUT/work/max_jobs before every start, so it
# can be changed while the run goes on. While the package is at or above T_PAUSE (default 85 C) the script pauses its own
# processes (SIGSTOP by PID) and resumes them below T_START. T_START / T_PAUSE are re-read from OUT/work/t_start and
# OUT/work/t_pause when those files exist. Materials whose log already says "Done material" are skipped (restart resumes).
# The guard is optional: without `sensors` (e.g. the cloud VM) or with THERMAL=0 it never pauses or delays anything.
#
# NHIST=n in the environment replaces the NHIST line of sim.in (histories per energy point); unset, the generated sim.in
# is luts/input/sim.in with STEP_CM replaced. SEED=n appends "SEED n" (seed base of run_penelope.f, default 4000): a set
# that must be independent of an earlier one at the same step needs another base. Several instances may share one OUT (one material each): sim.in and the
# symlinks are replaced only when they differ, atomically.
#
# Usage:  [NHIST=n] [SEED=n] luts/src/sim_penelope/run_step_variant.sh STEP_CM OUT [--jobs N] [--mats A-B,C,...]
# Build:  python luts/build_luts.py --engine penelope --raw-dir OUT/raw --out-dir OUT
set -euo pipefail
LUTS="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
step="${1:?STEP_CM}"; out="$(readlink -m "${2:?OUT dir}")"; shift 2
jobs=3; mats="1-30"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs) jobs="$2"; shift 2 ;;
        --mats) mats="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
T_START="${T_START:-78}"; T_PAUSE="${T_PAUSE:-85}"
exe="$LUTS/src/sim_penelope/run_penelope.x"
list=()                                     # --mats A-B,C,...: the materials, in order
for part in ${mats//,/ }; do
    if [[ $part == *-* ]]; then for ((k = ${part%-*}; k <= ${part#*-}; k++)); do list+=("$k"); done; else list+=("$part"); fi
done
[[ -x $exe && ! $LUTS/src/sim_penelope/run_penelope.f -nt $exe ]] || { echo "build $exe first (COMPILAR.sh)" >&2; exit 1; }

mkdir -p "$out/raw" "$out/logs" "$out/work/input" "$out/work/output"
[[ "$(readlink "$out/work/input/mats")" == "$LUTS/input/mats" ]] || ln -sfn "$LUTS/input/mats" "$out/work/input/mats"
[[ "$(readlink "$out/work/output/penelope")" == "$out" ]] || ln -sfn "$out" "$out/work/output/penelope"
simtmp="$out/work/input/sim.in.tmp.$$"
{
    awk -v s="$step" -v n="${NHIST:-}" 'NR==1{printf "%s                  ! STEP_CM  -- distance traveled per step [cm] (run_step_variant.sh)\n", s; next}
        n != "" && /! NHIST/ {printf "%s                  ! NHIST    -- histories per energy point (run_step_variant.sh)\n", n; next} {print}' \
        "$LUTS/input/sim.in"
    [[ -z ${SEED:-} ]] || echo "SEED $SEED"
} > "$simtmp"
if cmp -s "$simtmp" "$out/work/input/sim.in"; then rm -f "$simtmp"; else mv -f "$simtmp" "$out/work/input/sim.in"; fi
[[ -f $out/work/max_jobs ]] || echo "$jobs" > "$out/work/max_jobs"
echo "step $step cm${NHIST:+, nhist $NHIST} -> $out (materials ${list[*]}, max_jobs $(cat "$out/work/max_jobs"), T_START $T_START C)"

# package temperature in C; 0 when `sensors` is missing or reports no "Package id 0" (cloud VM: the guard is then inert), or
# when THERMAL=0
temp() {
    local t=""
    [[ ${THERMAL:-1} != 0 ]] && t=$(sensors 2>/dev/null | awk '/Package id 0/{match($0,/[0-9]+\.[0-9]/); print int(substr($0,RSTART,RLENGTH)); exit}')
    echo "${t:-0}"
}
declare -A pid_of=()
paused=0
exe_real="$(readlink -f "$exe")"
# our live driver processes (checked by /proc/PID/exe, so a recycled PID is never signalled)
alive() { local p; for p in "${pid_of[@]}"; do [[ "$(readlink /proc/$p/exe 2>/dev/null)" == "$exe_real" ]] && echo "$p"; done; return 0; }
# pause our own processes (SIGSTOP by PID) while the package is at or above T_PAUSE, resume below T_START
throttle() {
    local t; t=$(temp)
    [[ -f t_start ]] && T_START=$(cat t_start); [[ -f t_pause ]] && T_PAUSE=$(cat t_pause)
    if [[ $paused == 0 && $t -ge $T_PAUSE ]]; then
        for p in $(alive); do kill -STOP "$p" 2>/dev/null || true; done; paused=1; echo "$(date +%T) T=$t C: paused"
    elif [[ $paused == 1 && $t -lt $T_START ]]; then
        for p in $(alive); do kill -CONT "$p" 2>/dev/null || true; done; paused=0; echo "$(date +%T) T=$t C: resumed"
    fi
}
cd "$out/work"
i=0; nm=${#list[@]}
while true; do
    throttle
    n=$(alive | wc -l)
    m=${list[$i]:-}
    if [[ $i -lt $nm ]] && grep -q "Done material" "$out/logs/run_penelope_${m}.log" 2>/dev/null; then
        echo "mat $m already done"; i=$((i + 1)); continue
    fi
    if [[ $i -lt $nm ]]; then
        maxj=$(cat max_jobs 2>/dev/null || echo "$jobs")
        if [[ $paused == 0 && $n -lt $maxj && $(temp) -lt $T_START ]]; then
            "$exe" "$m" "$m" > "$out/logs/run_penelope_${m}.log" 2>&1 &
            pid_of[$m]=$!
            echo "$(date +%T) started mat $m (pid $!, T=$(temp) C, running $((n + 1)))"
            i=$((i + 1)); sleep 10; continue
        fi
    elif [[ $n -eq 0 ]]; then
        break
    fi
    sleep 20
done
rc=0
for k in "${!pid_of[@]}"; do wait "${pid_of[$k]}" || { echo "mat $k failed" >&2; rc=1; }; done
for k in "${list[@]}"; do grep -q "Done material" "$out/logs/run_penelope_${k}.log" 2>/dev/null || { echo "mat $k: no 'Done material' in its log" >&2; rc=1; }; done
echo "$(date +%T) all done (rc $rc)"
exit $rc
