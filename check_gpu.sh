#!/bin/sh
# check_gpu.sh -- does the precompiled HMC run on this machine, CPU and GPU? A 30-second check; run it from the repository root, with
# the PENELOPE tables unzipped (see README.md).
#   sh check_gpu.sh          the executable that hmc/hmc.x selects
#   sh check_gpu.sh all      every executable of hmc/bin/, one after the other (the CPU-only one is not asked for the GPU)
# It prints the system and the GPU, runs the example of hmc/input/hmc.in on the CPU and on the GPU with 2e5 histories, and checks that
# every history is accounted for. If it fails, the output is what to send with a bug report.
set -u
cd "$(dirname "$0")" || exit 2
echo "system : $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}"), glibc $(ldd --version 2>/dev/null | head -1 | grep -o '[0-9]*\.[0-9]*$'), kernel $(uname -r), $(uname -m)"
echo "GPU    : $(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)"
[ -x hmc/hmc.x ] || { echo "hmc/hmc.x not found or not executable"; exit 2; }
[ -s luts/output/penelope/step1-1/stepend_LUT.raw ] || { echo "the PENELOPE tables are not unzipped at luts/output/penelope (README.md, step 2)"; exit 2; }
mkdir -p work

# the rules of hmc/hmc.x: the CUDA 13 build is for Turing (cc 7.5) and newer with driver 580 or later, the CUDA 12 build for cc 5.0 and newer
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | awk -F. '{print $1*10+$2}'); CC=${CC:-0}
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1); DRV=${DRV:-0}
fits_gpu() {
  case $1 in
    hmc-cuda13-*) [ "$CC" -ge 75 ] && [ "$DRV" -ge 580 ];;
    hmc-cuda12-*) [ "$CC" -ge 50 ];;
    *) return 0;;
  esac
}

run_one() {      # $1 label, $2 executable name ("" = the selector's choice), $3 backends
  label=$1; exe=$2; backends=$3
  echo "--- $label: $(HMC_BINARY=$exe hmc/hmc.x --which)"
  for b in $backends; do
    if [ "$b" = gpu ] && [ -n "$exe" ] && ! fits_gpu "$exe"; then echo "skipped: gpu backend ($exe does not target this GPU or driver, see README.md)"; continue; fi
    sed -e "s|^backend .*|backend = $b|" -e "s|^nhist .*|nhist = 200000|" -e "s|^output .*|output = work/check_${label}_$b.raw|" hmc/input/hmc.in > work/check_${label}_$b.in
    if HMC_BINARY=$exe hmc/hmc.x --config work/check_${label}_$b.in --quiet >work/check_${label}_$b.log 2>&1; then echo "ran    : $b backend"
    else echo "FAILED : $b backend (work/check_${label}_$b.log)"; tail -5 work/check_${label}_$b.log; return 1; fi
    python3 - "work/check_${label}_$b.raw.json" <<'PY' || return 1
import json, sys
j = json.load(open(sys.argv[1]))
ok = j["n_hist"] == j["n_annihilated"] + j["n_escaped"]
print(f"         backend={j['backend']} n_hist={j['n_hist']} annihilated={j['n_annihilated']} escaped={j['n_escaped']} wall={j['wall_s']} s {'ok' if ok else 'NOT CONSERVED'}")
sys.exit(0 if ok else 1)
PY
  done
}

rc=0
if [ "${1:-}" = all ]; then
  for f in hmc/bin/hmc-*; do
    n=$(basename "$f")
    case $n in hmc-cpu-*) bk="cpu";; *) bk="cpu gpu";; esac
    run_one "$n" "$n" "$bk" || rc=1
  done
else
  run_one default "" "cpu gpu" || rc=1
fi
[ $rc = 0 ] && echo "check_gpu: done, everything ran" || echo "check_gpu: something FAILED (see above)"
exit $rc
