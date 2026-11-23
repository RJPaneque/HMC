#!/usr/bin/env python3
"""generate_spectra.py -- beta+ emission spectra of radionuclides from the PENNUC database.

    python spectrum/generate_spectra.py F18 Ga68 Rb82 [--samples 1e7] [--out DIR] [--format hmc|peneasy] [--force]

For each isotope (F18, Ga68, Ag108m, ...: the PENNUC file is <El>-<A>.nuc, e.g. F-18.nuc in input/nuclides) it runs
decays_positron.x (decays_positron.f: PENNUC decay paths, build with src/build.sh) in a scratch directory, takes the
positron column of the energy-spectrum table it writes (spcnuc.dat, 400 bins up to the end-point energy), normalises it
to a probability per bin and writes <out>/<isotope>.txt:

  hmc      two columns, energy (eV) and probability, no header, no end marker -- what hmc/ and the harness read
           (spectrum/output/spectra/);
  peneasy  the same table with a header line and a closing "<last E> -1" row, the PenEasy source-spectrum convention
           (what penEasy's pnnc.spc and HMC-1's getSpec.sh produced).

The numbers are deterministic (PENNUC starts from fixed seeds): the same isotope and --samples always give the same file.
Existing files are never overwritten without --force.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                  # spectrum/
DECAYS_X = HERE / "decays_positron.x"
NUCLIDES = ROOT / "input" / "nuclides"              # symlink to the PENNUC database (the merged ~/MC-codes/nuclides)
EXTRA = ROOT / "input" / "extra"                    # .nuc files made here from ENSDF (ensdf2nuc/), for nuclides the database lacks (Y-86)
OUT = ROOT / "output" / "spectra"


def nuclide_file(isotope: str) -> str:
    """F18 -> F-18.nuc, Ag108m -> Ag-108m.nuc"""
    if not re.fullmatch(r"[A-Za-z]{1,2}\d+m?", isotope):
        raise ValueError(f"isotope name {isotope!r}: expected e.g. F18, Ga68, Ag108m")
    return re.sub(r"^([A-Za-z]+)", r"\1-", isotope) + ".nuc"


def parse_spcnuc(text: str):
    """spcnuc.dat -> [(energy string as written, positron spectrum float)], comments dropped."""
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        f = line.split()
        rows.append((f[0], float(f[5])))
    return rows


def format_spectrum(rows, fmt: str = "hmc") -> str:
    """Normalise the positron column to a probability per bin and format it (awk-compatible %.6g, as getSpec.sh)."""
    total = sum(p for _, p in rows)
    if total <= 0:
        raise ValueError("no positron emission in the spectrum table (not a beta+ emitter?)")
    body = [f"{e}\t{p / total:.6g}" for e, p in rows]
    if fmt == "hmc":
        return "\n".join(body) + "\n"
    if fmt == "peneasy":
        return "\n".join(["# Energy(eV) probability"] + body + [f"{rows[-1][0]}\t-1"]) + "\n"
    raise ValueError(f"unknown format {fmt!r}")


def link_nuclides(dest: Path, nuclides: Path, extra: Path | None) -> None:
    """dest/ gets a symlink to every file of the PENNUC database (the .nuc files and PENNUC's atomic-data files) and then to every
    .nuc file of `extra`, which wins when both have the name (the files made here are versioned with the repository)."""
    dest.mkdir()
    for src, only_nuc in ((Path(nuclides), False), (Path(extra) if extra else None, True)):
        if src is not None and src.is_dir():
            for f in src.iterdir():
                if f.is_file() and (f.suffix == ".nuc" or not only_nuc):
                    target = dest / f.name
                    if target.is_symlink():
                        target.unlink()
                    target.symlink_to(f.resolve())


def run_decays(isotope: str, samples: float, nuclides: Path = NUCLIDES, exe: Path = DECAYS_X, extra: Path | None = EXTRA) -> str:
    """Run decays_positron.x in a scratch directory; returns the text of spcnuc.dat."""
    nuc = nuclide_file(isotope)
    if not exe.exists():
        raise FileNotFoundError(f"{exe} not built: run spectrum/src/build.sh")
    if not (Path(nuclides) / nuc).exists() and not (extra and (Path(extra) / nuc).exists()):
        raise FileNotFoundError(f"{Path(nuclides) / nuc} not found (input/nuclides -> the PENNUC database; input/extra has the ones made from ENSDF)")
    with tempfile.TemporaryDirectory(prefix="decays_") as d:
        d = Path(d)
        link_nuclides(d / "nuclides", nuclides, extra)
        # standard input: the nuclide file, then the number of decays
        r = subprocess.run([str(exe)], cwd=d, input=f"{nuc}\n{int(samples)}\n", text=True, capture_output=True)
        if r.returncode != 0 or not (d / "spcnuc.dat").exists():
            raise RuntimeError(f"decays_positron.x failed for {nuc} (rc {r.returncode}):\n{r.stdout[-800:]}{r.stderr[-400:]}")
        return (d / "spcnuc.dat").read_text()


def generate(isotope: str, samples: float = 1e7, fmt: str = "hmc", **kw) -> str:
    return format_spectrum(parse_spcnuc(run_decays(isotope, samples, **kw)), fmt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("isotopes", nargs="+")
    ap.add_argument("--samples", type=float, default=1e7, help="decays simulated by PENNUC (default 1e7)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--format", choices=["hmc", "peneasy"], default="hmc")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    status = 0
    for iso in a.isotopes:
        dst = a.out / f"{iso}.txt"
        if dst.exists() and not a.force:
            print(f"{dst}: exists, skipped (--force to overwrite)")
            status = 1
            continue
        dst.write_text(generate(iso, a.samples, a.format))
        print(f"wrote {dst}")
    return status


if __name__ == "__main__":
    sys.exit(main())
