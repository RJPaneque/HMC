#!/usr/bin/env python3
"""run_bundle.py -- one engine's step-size ladder: raw simulation output, then the tables, into a bundle HMC reads.

    luts/run_sim.sh --engine penelope|egsnrc|geant4|fluka [--levels 4 | --steps 0.05,0.025,...]
                    [--nhist N] [--seed S] [--mats A-B,C] [--jobs N] [--workers N] [--out ROOT]
                    [--raw-only | --build-only] [--force] [--keep-dat] [--no-thermal] [--dry-run]

Layout: <ROOT>/<engine>/step1-<n>/ for the step L / n, L = STEP_CM of luts/input/sim.in (the longest step) and
n = 1, 2, 4, ... (--levels halvings; HMC orders the sets by the step_cm of their run.json, the name is only a label):

    step1-<n>/raw/mat<m>.dat.zst   raw histories (zstd once the step's simulation ends; build_luts reads .zst)
    step1-<n>/logs/, work/         the engine runner's logs and work files (kept as the record of the run)
    step1-<n>/sim.json             this script's record: engine, step, nhist, seed, settings, commands, times
    step1-<n>/*_LUT.raw, run.json, gap_census.csv, stepend_census.csv, build.log   the tables (luts/build_luts.py)

ROOT defaults to luts/output, the production bundles. A step folder that already holds tables is never written to (its raw
output would no longer be the one its tables were built from) unless --force; --build-only skips it. An independent sample
for the table checks goes elsewhere, e.g.  --out luts/output/.cache/check --raw-only --seed 9000.

Resume: a material counts as done when raw/mat<m>.dat or raw/mat<m>.dat.zst exists; only the missing ones are run.

Engine runners (each runs one step, one process or container per material, and writes <step dir>/{raw,logs,work}):
    penelope  luts/src/sim_penelope/run_step_variant.sh   (own thermal guard: starts below T_START, pauses at T_PAUSE)
    geant4    luts/src/sim_geant4/run_mats.sh             (image luts-geant4)
    egsnrc    luts/src/sim_egsnrc/run_egsnrc.py           (image hmc2-egsnrc, app built by build_app.sh)
    fluka     luts/src/sim_fluka/run_fluka_luts.py        (image luts-fluka, built by build_image.sh)
The container engines run under validation/src/drivers/thermal_watch.py (CPU 85 C, GPU 83 C) unless --no-thermal.

Production settings (the tables of 2026-11-01..03): 1e6 histories per energy (--nhist 1000000), seed base 7000, frame 1,
tail knots (build_luts defaults), and for Geant4 CUT_UM=1, CUT_FLOOR_EV=1000, MAX_STEP_UM=1 (the 1 um step cap; set here
unless already in the environment). 1e6 histories need ~0.4 TB of raw text per PENELOPE step: the 2026-10-31 sets were made
on a cloud VM (validation/cloud/campaign_20261031/cloud_home/{pen1e6.sh,luts1e6.py,g4cap.sh}).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

LUTS = Path(__file__).resolve().parents[1]
REPO = LUTS.parent
ENGINES = ("penelope", "egsnrc", "geant4", "fluka")
TABLES = ("ann_LUT.raw", "scatt-eloss_LUT.raw", "stepend_LUT.raw")
G4_PRODUCTION = {"CUT_UM": "1", "CUT_FLOOR_EV": "1000", "MAX_STEP_UM": "1"}
THERMAL_WATCH = REPO / "validation" / "src" / "drivers" / "thermal_watch.py"


def log(msg: str) -> None:
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def sim_in_values(path: Path) -> list[str]:
    """The positional values of sim.in (text after '!' ignored): STEP_CM EMIN EMAX NENRG NHIST MAT_DIR."""
    vals = [ln.split("!")[0].strip() for ln in path.read_text().splitlines()]
    vals = [v for v in vals if v]
    if len(vals) < 6:
        raise SystemExit(f"{path}: expected the six positional lines of sim.in")
    return vals


def ladder(base: float, levels: int | None = None, steps: list[float] | None = None) -> list[tuple[str, float]]:
    """[(name, step_cm)], longest first: step1-<n> for base / n, n = 1, 2, 4, ... (levels) or n = base / s (steps)."""
    if steps is None:
        if not levels or levels < 1:
            raise ValueError("levels must be >= 1")
        return [(f"step1-{2 ** k}", base / 2 ** k) for k in range(levels)]
    out = []
    for s in steps:
        n = base / s
        if s <= 0 or abs(n - round(n)) > 1e-6 * n:
            raise ValueError(f"step {s} cm is not the base step {base} cm divided by an integer: name it by hand")
        out.append((f"step1-{round(n)}", s))
    if [s for _, s in out] != sorted((s for _, s in out), reverse=True) or len({n for n, _ in out}) != len(out):
        raise ValueError("steps must be distinct and decreasing")
    return out


def parse_mats(spec: str | None, mat_dir: Path) -> list[int]:
    if not spec:
        n = 0
        while (mat_dir / f"mat{n + 1}.mat").exists():
            n += 1
        if n == 0:
            raise SystemExit(f"no mat1.mat in {mat_dir}")
        return list(range(1, n + 1))
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return sorted(set(out))


def has_tables(d: Path) -> bool:
    return any((d / f).exists() for f in TABLES)


def missing_mats(d: Path, mats: list[int]) -> list[int]:
    raw = d / "raw"
    return [m for m in mats if not (raw / f"mat{m}.dat").exists() and not (raw / f"mat{m}.dat.zst").exists()]


def ranges(mats: list[int]) -> list[tuple[int, int]]:
    """Contiguous runs of a sorted list: [1,2,3,7] -> [(1,3), (7,7)]."""
    out: list[tuple[int, int]] = []
    for m in mats:
        if out and m == out[-1][1] + 1:
            out[-1] = (out[-1][0], m)
        else:
            out.append((m, m))
    return out


def runner_commands(engine: str, d: Path, step: float, mats: list[int], a) -> tuple[list[list[str]], dict]:
    """(commands, extra environment) that simulate `mats` at `step` into d/{raw,logs,work}."""
    py, env = sys.executable, {}
    lst = ",".join(map(str, mats))
    if engine == "penelope":
        if a.nhist:
            env["NHIST"] = str(a.nhist)
        if a.seed is not None:
            env["SEED"] = str(a.seed)
        if a.no_thermal:
            env["THERMAL"] = "0"
        sh = str(LUTS / "src" / "sim_penelope" / "run_step_variant.sh")
        spec = ",".join(f"{lo}-{hi}" if hi > lo else str(lo) for lo, hi in ranges(mats))
        return [["bash", sh, repr(step), str(d), "--jobs", str(a.jobs), "--mats", spec]], env
    if engine == "geant4":
        env.update({k: os.environ.get(k, v) for k, v in G4_PRODUCTION.items()})
        if a.seed is not None:
            env["SEED"] = str(a.seed)
        cmd = ["bash", str(LUTS / "src" / "sim_geant4" / "run_mats.sh"), str(d), "--mats", lst, "--jobs", str(a.jobs),
               "--step", repr(step)] + (["--nhist", str(a.nhist)] if a.nhist else [])
        return [cmd], env
    drv = {"egsnrc": LUTS / "src" / "sim_egsnrc" / "run_egsnrc.py", "fluka": LUTS / "src" / "sim_fluka" / "run_fluka_luts.py"}[engine]
    cmd = [py, str(drv), "--out-dir", str(d), "--mats", lst, "--jobs", str(a.jobs), "--step-cm", repr(step)]
    cmd += (["--nhist", str(a.nhist)] if a.nhist else []) + (["--seed", str(a.seed)] if a.seed is not None else [])
    return [cmd], env


def prepare_engine(engine: str, dry: bool) -> None:
    """Build what a runner needs and cannot build itself."""
    pen = LUTS / "src" / "sim_penelope"
    if engine == "penelope" and (not (pen / "run_penelope.x").exists()
                                 or (pen / "run_penelope.f").stat().st_mtime > (pen / "run_penelope.x").stat().st_mtime):
        cmd = ["bash", str(pen / "COMPILAR.sh")]
    elif engine == "egsnrc" and not (LUTS / "src" / "sim_egsnrc" / "bin" / "lutstep").exists():
        cmd = ["bash", str(LUTS / "src" / "sim_egsnrc" / "build_app.sh")]
    elif engine == "geant4" and subprocess.run(["docker", "image", "inspect", "luts-geant4"], capture_output=True).returncode:
        cmd = ["docker", "build", "-t", "luts-geant4", str(LUTS / "src" / "sim_geant4")]
    else:
        return
    log(f"prepare {engine}: {shlex.join(cmd)}")
    if not dry and subprocess.run(cmd).returncode:
        raise SystemExit(f"{engine}: {shlex.join(cmd)} failed")


def run(cmd: list[str], env: dict, dry: bool, watch: bool, label: str) -> int:
    if watch:
        cmd = [sys.executable, str(THERMAL_WATCH), "--label", label, "--"] + cmd
    log(("(dry run) " if dry else "") + " ".join(f"{k}={v}" for k, v in env.items()) + (" " if env else "") + shlex.join(cmd))
    if dry:
        return 0
    return subprocess.run(cmd, env={**os.environ, **env}, cwd=REPO).returncode


def compress(d: Path, mats: list[int], dry: bool) -> None:
    for m in mats:
        f = d / "raw" / f"mat{m}.dat"
        if f.exists() and not dry:
            subprocess.run(["zstd", "-q", "-3", "-T4", "--rm", "-f", str(f)], check=True)


def write_record(d: Path, rec: dict) -> None:
    p = d / "sim.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.setdefault("runs", []).append(rec)
    old.update({k: rec[k] for k in ("engine", "step_cm", "name")})
    p.write_text(json.dumps(old, indent=2) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, choices=ENGINES)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--levels", type=int, default=4, help="halvings of sim.in's STEP_CM: step1-1, step1-2, step1-4, ... (default 4)")
    g.add_argument("--steps", help="explicit steps in cm, comma-separated, decreasing (each STEP_CM / integer)")
    ap.add_argument("--nhist", type=int, help="histories per energy (default: NHIST of sim.in)")
    ap.add_argument("--seed", type=int, help="seed base (default: the runners' 4000; production 7000)")
    ap.add_argument("--mats", help="materials A-B,C (default: every matN.mat of luts/input/mats)")
    ap.add_argument("--jobs", type=int, default=8, help="materials simulated in parallel (default 8, the harness rule)")
    ap.add_argument("--workers", type=int, default=8, help="build_luts workers")
    ap.add_argument("--out", default=str(LUTS / "output"), help="bundle root: <out>/<engine>/step1-<n>/ (default luts/output)")
    ph = ap.add_mutually_exclusive_group()
    ph.add_argument("--raw-only", action="store_true", help="simulate, do not build tables")
    ph.add_argument("--build-only", action="store_true", help="build tables from the raw output already there")
    ap.add_argument("--force", action="store_true", help="also write into step folders that hold tables (rebuilds them)")
    ap.add_argument("--keep-dat", action="store_true", help="leave the raw output uncompressed")
    ap.add_argument("--no-thermal", action="store_true", help="no thermal guard (e.g. a cloud VM)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = ap.parse_args(argv)

    if a.jobs > 12:
        raise SystemExit("--jobs > 12 overheats this machine (harness rule): ask first, or run elsewhere with --no-thermal")
    vals = sim_in_values(LUTS / "input" / "sim.in")
    base, mat_dir = float(vals[0]), LUTS / vals[5].split()[0]
    sets = ladder(base, a.levels, [float(s) for s in a.steps.split(",")] if a.steps else None)
    mats = parse_mats(a.mats, mat_dir)
    root = Path(a.out).resolve() / a.engine
    log(f"{a.engine}: {len(sets)} steps {[s for _, s in sets]} cm, {len(mats)} materials, nhist {a.nhist or vals[4]}, "
        f"seed {a.seed if a.seed is not None else 'runner default'} -> {root}")

    todo = []
    for name, step in sets:
        d = root / name
        if has_tables(d) and not a.force:
            log(f"{name}: holds tables, skipped (--force rebuilds it)")
            continue
        todo.append((name, step, d))
    if not a.build_only and any(missing_mats(d, mats) for _, _, d in todo):
        prepare_engine(a.engine, a.dry_run)

    rc = 0
    for name, step, d in todo:
        if not a.build_only:
            miss = missing_mats(d, mats)
            if miss:
                if not a.dry_run:
                    d.mkdir(parents=True, exist_ok=True)
                cmds, env = runner_commands(a.engine, d, step, miss, a)
                t0 = time.time()
                codes = [run(c, env, a.dry_run, watch=a.engine != "penelope" and not a.no_thermal,
                             label=f"luts {a.engine} {name}") for c in cmds]
                if not a.dry_run:
                    write_record(d, {"engine": a.engine, "name": name, "step_cm": step, "materials": miss,
                                     "nhist": a.nhist or int(float(vals[4])), "seed": a.seed, "env": env,
                                     "commands": [shlex.join(c) for c in cmds], "codes": codes,
                                     "start": time.strftime("%FT%T", time.localtime(t0)), "wall_s": round(time.time() - t0)})
                left = [] if a.dry_run else missing_mats(d, mats)
                done = [m for m in miss if m not in left]
                if not a.keep_dat:
                    compress(d, done, a.dry_run)
                if left or any(codes):
                    log(f"{name}: raw incomplete (missing {left}, exit codes {codes}); not built")
                    rc = 1
                    continue
                log(f"{name}: raw done ({len(done)} materials in {time.time() - t0:.0f} s)")
            else:
                log(f"{name}: raw complete")
        if a.raw_only:
            continue
        if missing_mats(d, mats) and not a.dry_run:
            log(f"{name}: raw incomplete, not built"); rc = 1; continue
        cmd = [sys.executable, str(LUTS / "build_luts.py"), "--engine", a.engine, "--raw-dir", str(d / "raw"),
               "--out-dir", str(d), "--workers", str(a.workers)]
        if a.mats:
            cmd += ["--materials", ",".join(map(str, mats))]
        log(("(dry run) " if a.dry_run else "") + shlex.join(cmd) + f" > {d / 'build.log'}")
        if a.dry_run:
            continue
        with open(d / "build.log", "w") as f:
            code = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode
        log(f"{name}: tables " + ("ok" if code == 0 else f"FAILED (see {d}/build.log)"))
        rc |= code != 0
    return rc


if __name__ == "__main__":
    sys.exit(main())
