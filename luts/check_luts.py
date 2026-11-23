#!/usr/bin/env python3
"""
check_luts.py -- check one or more runs: per-row statistics (extract_stats.py),
then figures (check_figures.py) in <figs-dir>/<engine>/<set>/.

A run is TABLES[@RAW[+RAW ...]] (runs.py): an engine (penelope -> the bundle
luts/output/penelope, i.e. its longest set step1-1), a table set, a directory
under luts/output or a path, checked against RAW, by default the independent
check samples luts/output/.cache/check1e5 then check1e4 (run_bundle.py --raw-only).

--scope raw checks the raw simulation output alone and never reads a LUT, so
it can run before luts-build; --scope lut checks the LUTs against the raw
output; the default runs both. --stats-only / --figures-only select one step.
Statistics are cached and reused while their source files are unchanged
(--force recomputes them); figures always redraw from the cache.

Usage:
    python luts/check_luts.py RUN [RUN ...] [--label L]          e.g.  penelope egsnrc geant4 fluka [--scope raw|lut|all]
                              [--stats-only | --figures-only] [--materials 1 13 30]
                              [--workers N] [--force] [--lut-material M]
                              [--stats-dir DIR] [--figs-dir DIR]
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # luts/src/analysis/
sys.path.insert(0, str(HERE))
from runs import CACHE, OUTPUT  # noqa: E402


def run_step(script, args):
    cmd = [sys.executable, str(HERE / script), *map(str, args)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def common_args(ap):
    step = ap.add_mutually_exclusive_group()
    step.add_argument("--stats-only", action="store_true", help="only extract per-row statistics")
    step.add_argument("--figures-only", action="store_true", help="only draw figures from cached statistics")
    ap.add_argument("--scope", choices=("raw", "lut", "all"), default="all",
                    help="raw: raw simulation output only, never reads a LUT; "
                         "lut: LUTs against the raw output; all: both (default)")
    ap.add_argument("--materials", nargs="+", default=None, help="stats: material ids (default: all)")
    ap.add_argument("--workers", default=None, help="stats: parallel workers")
    ap.add_argument("--force", action="store_true", help="stats: recompute even when the cache is up to date")
    ap.add_argument("--stats-dir", default=str(CACHE), help="statistics cache")
    ap.add_argument("--figs-dir", default=str(OUTPUT / "figs"), help="figures root")


def stats(a, runs):
    args = [*runs, "--scope", a.scope, "--cache", a.stats_dir]
    if a.materials: args += ["--materials", *a.materials]
    if a.workers: args += ["--workers", a.workers]
    if a.force: args += ["--force"]
    run_step("extract_stats.py", args)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", metavar="RUN", help="engine name, directory under luts/output, or path")
    ap.add_argument("--label", default=None, help="name used in titles (one run only)")
    ap.add_argument("--lut-material", default=None, help="figures: material shown in L0 (default 13)")
    common_args(ap)
    a = ap.parse_args(argv)
    if a.label and len(a.runs) > 1:
        ap.error("--label takes a single RUN")
    if not a.figures_only:
        stats(a, a.runs)
    if not a.stats_only:
        for r in a.runs:
            args = [r, "--scope", a.scope, "--cache", a.stats_dir, "--out", a.figs_dir]
            if a.label: args += ["--label", a.label]
            if a.lut_material: args += ["--lut-material", a.lut_material]
            run_step("check_figures.py", args)


if __name__ == "__main__":
    main()
