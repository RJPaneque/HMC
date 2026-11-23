#!/usr/bin/env python3
"""
compare_luts.py -- compare two runs (TABLES[@RAW[+RAW ...]], runs.py): per-row
statistics (extract_stats.py), then figures (compare_figures.py) in
<figs-dir>/<A engine>_vs_<B engine>/<set>/ (e.g. penelope fluka).

The runs can be two engines (penelope geant4) or two LUT methods built from
the same raw data (e.g. penelope and a luts-build --out-dir variant); labels
default to the engine names plus the build parameters that differ.

--scope raw compares the raw simulation output only (skipped when both runs
share it); --scope lut compares how well each run's LUTs reproduce their raw
output; the default runs both.

Usage:
    python luts/compare_luts.py RUN_A RUN_B [--labels LA LB] [--scope raw|lut|all]
                                [--stats-only | --figures-only] [--materials 1 13 30]
                                [--workers N] [--force] [--stats-dir DIR] [--figs-dir DIR]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_luts import common_args, run_step, stats  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs=2, metavar="RUN", help="engine name, directory under luts/output, or path")
    ap.add_argument("--labels", nargs=2, default=None, metavar="L", help="names used in titles and legends")
    common_args(ap)
    a = ap.parse_args(argv)
    if not a.figures_only:
        stats(a, a.runs)
    if not a.stats_only:
        args = [*a.runs, "--scope", a.scope, "--cache", a.stats_dir, "--out", a.figs_dir]
        if a.labels: args += ["--labels", *a.labels]
        run_step("compare_figures.py", args)


if __name__ == "__main__":
    main()
