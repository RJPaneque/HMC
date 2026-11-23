#!/bin/sh
# hmc/hmc.x -- starts the HMC executable that suits this machine. Everything after the name goes to it unchanged:
#     hmc/hmc.x --config hmc/input/hmc.in
# The executables are in hmc/bin/, named hmc-<backend>-<architecture> (the architecture is `uname -m`):
#     hmc-cuda13-x86_64   NVIDIA GPUs from Turing (cc 7.5) on, driver 580 or later; also runs on the CPU (CUDA 13.1 toolkit)
#     hmc-cuda12-x86_64   NVIDIA GPUs from Maxwell (cc 5.0) on, any recent driver; also runs on the CPU (CUDA 12.9 toolkit)
#     hmc-cpu-x86_64      the CPU only, no NVIDIA libraries, no AVX2 requirement (statically linked)
# The choice, in this order: the newest CUDA build the GPU and driver allow (and whose libraries are all present), then the CUDA 12
# build for any other NVIDIA GPU, then the CPU build. The configuration's `backend` key still decides what runs inside it.
#     HMC_BINARY=NAME hmc/hmc.x ...    use hmc/bin/NAME (or the file at that path) instead
#     hmc/hmc.x --which                print the executable that would start, and why
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=$HERE/bin
ARCH=$(uname -m)

if [ -n "${HMC_BINARY:-}" ]; then
  case $HMC_BINARY in */*) forced=$HMC_BINARY;; *) forced=$BIN/$HMC_BINARY;; esac
  if [ "${1:-}" = "--which" ]; then echo "$forced  (forced by HMC_BINARY)"; exit 0; fi
  exec "$forced" "$@"
fi

usable() {      # executable, and no shared library missing (when ldd can tell)
  [ -x "$1" ] || return 1
  if command -v ldd >/dev/null 2>&1 && ldd "$1" 2>&1 | grep -q "not found"; then return 1; fi
  return 0
}

why=""; pick=""
gpu=""
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu=$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null | head -n 1)
fi
if [ -n "$gpu" ]; then
  drv=$(printf '%s' "$gpu" | awk -F', ' '{print $2}'); cc=$(printf '%s' "$gpu" | awk -F', ' '{print $3}')
  drvmajor=${drv%%.*}; ccn=$(printf '%s' "$cc" | awk -F. '{print $1*10+$2}')
  case $drvmajor in ''|*[!0-9]*) drvmajor=0;; esac
  case $ccn in ''|*[!0-9]*) ccn=0;; esac
  if [ "$drvmajor" -ge 580 ] && [ "$ccn" -ge 75 ] && usable "$BIN/hmc-cuda13-$ARCH"; then
    pick=$BIN/hmc-cuda13-$ARCH; why="GPU $gpu: CUDA 13 build"
  elif [ "$ccn" -ge 50 ] && usable "$BIN/hmc-cuda12-$ARCH"; then
    pick=$BIN/hmc-cuda12-$ARCH; why="GPU $gpu: CUDA 12 build"
  fi
fi
if [ -z "$pick" ]; then
  if usable "$BIN/hmc-cpu-$ARCH"; then pick=$BIN/hmc-cpu-$ARCH; why="no usable NVIDIA GPU: CPU build"
  elif usable "$BIN/hmc-cuda12-$ARCH"; then pick=$BIN/hmc-cuda12-$ARCH; why="no usable NVIDIA GPU, no CPU build: CPU path of the CUDA 12 build"
  fi
fi
if [ -z "$pick" ]; then
  echo "hmc.x: no executable of hmc/bin runs on this machine ($ARCH). Available: $(ls "$BIN" 2>/dev/null | tr '\n' ' ')" >&2
  exit 127
fi
if [ "${1:-}" = "--which" ]; then echo "$pick  ($why)"; exit 0; fi
[ -n "${HMC_VERBOSE:-}" ] && echo "hmc.x: $pick ($why)" >&2
exec "$pick" "$@"
