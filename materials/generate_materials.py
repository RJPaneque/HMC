#!/usr/bin/env python3
"""
generate_materials.py -- Generate PENELOPE .mat files from a density list.

Reads mass densities [g/cm3] from an ASCII input file (default:
materials/input/materials.in), assigns each one a Schneider et al. (2000)
tissue composition (density -> HU -> Table 6 bin) and runs PENELOPE's
material.x to produce one .mat file per density.

Output directory defaults to materials/output/<N>mats/ where N is the
number of densities read (e.g. output/30mats/ for 30 densities).

Usage:
    pixi run materials-gen
    python materials/src/generate_materials.py [--input F] [--output D]
                                               [--exe X] [--workdir D]

Input file format: one density per line; '#' starts a comment;
blank lines are ignored. Example:

    1.0885      # connective tissue
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schneider_materials import (  # noqa: E402
    density_to_hu,
    generate_mat_file,
    get_material,
)

ROOT = Path(__file__).resolve().parent.parent  # materials/


def parse_densities(path: Path) -> list[float]:
    """Read densities [g/cm3] from an ASCII file (one per line)."""
    densities: list[float] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            rho = float(line.split()[0])
        except ValueError:
            raise ValueError(f"{path}:{lineno}: not a density: {raw!r}") from None
        if rho <= 0.0:
            raise ValueError(f"{path}:{lineno}: density must be > 0 (got {rho})")
        densities.append(rho)
    if not densities:
        raise ValueError(f"{path}: no densities found")
    return densities


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", type=Path, default=ROOT / "input" / "materials.in",
                    help="density list file (default: %(default)s)")
    ap.add_argument("--output", type=Path, default=None,
                    help="output directory (default: materials/output/<N>mats/)")
    ap.add_argument("--exe", type=Path, default=ROOT / "material.x",
                    help="path to material.x (default: %(default)s)")
    ap.add_argument("--workdir", type=Path, default=ROOT / "input",
                    help="CWD for material.x; must contain pdfiles/ "
                         "(default: %(default)s)")
    args = ap.parse_args()

    densities = parse_densities(args.input)
    n = len(densities)
    out_dir = args.output or (ROOT / "output" / f"{n}mats")

    if not args.exe.is_file():
        sys.exit(f"error: material.x not found: {args.exe}")
    if not (args.workdir / "pdfiles").is_dir():
        sys.exit(f"error: pdfiles/ not found in material.x workdir: {args.workdir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {n} material(s) from {args.input}")
    print(f"  exe:     {args.exe}")
    print(f"  workdir: {args.workdir}")
    print(f"  output:  {out_dir}\n")

    for i, rho in enumerate(densities, 1):
        mat = get_material(density=rho)
        target = out_dir / f"mat{i}.mat"
        generate_mat_file(mat, str(args.exe), str(target),
                          label=f"Schneider_mat{i}", workdir=str(args.workdir))
        print(f"  mat{i:<2d}  rho={rho:<12.7g}  HU={density_to_hu(rho):8.1f}  {mat.label}")

    print(f"\nDone: {n} .mat files written to {out_dir}")


if __name__ == "__main__":
    main()
