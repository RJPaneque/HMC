"""
runs.py -- a "run" is one table set plus the raw data its tables are checked against. luts-check takes one or more runs,
luts-compare takes two.

    RUN = TABLES[@RAW[+RAW ...]]

TABLES is an engine (``penelope`` -> the bundle luts/output/penelope), a bundle, a table set (e.g. penelope/step1-2), a
directory under luts/output or any path; a bundle stands for its longest step (step1-1). RAW are raw directories
(mat<m>.dat or mat<m>.dat.zst), each a path or a directory under luts/output; every material takes its raw file from the
first RAW that has it, so a dense sample of a few materials can precede a sparse one of all.

Without @RAW: the raw directory of the set's run.json when it exists (a set built locally from its own raw data), else the
independent check samples written by run_bundle.py --raw-only: luts/output/.cache/check1e5/<engine>/<step>/raw, then
.cache/check1e4/<engine>/<step>/raw (the production sets' own raw data, 1e6 per energy on a cloud VM, was not kept).
Materials of the tables without raw data in any RAW are left out, with a note.

Statistics are cached under luts/output/.cache/stats: raw statistics per raw directory and material (shared by every run
that reads that file), table statistics per (table set, raw directory, material).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

LUTS = Path(__file__).resolve().parents[2]       # luts/
REPO = LUTS.parent
OUTPUT = LUTS / "output"
CACHE = OUTPUT / ".cache" / "stats"
CHECK_SAMPLES = (OUTPUT / ".cache" / "check1e5", OUTPUT / ".cache" / "check1e4")   # in order of preference
ENGINES = ("penelope", "geant4", "egsnrc", "fluka")
PARAMS = ("step_cm", "n_theta", "n_u", "n_full", "k_max", "n_z", "n_r", "n_full_stop")
TABLES = ("ann_LUT.raw", "scatt-eloss_LUT.raw")
_MAT = re.compile(r"^mat(\d+)\.dat(\.zst)?$")       # raw files, plain or zstd (run_bundle compresses them)


def _key(path: Path) -> str:
    """Cache/figure key: path relative to luts/output with '/' -> '__';
    outside luts/output, the directory name plus a short hash of its path."""
    path = path.resolve()
    try:
        return "__".join(path.relative_to(OUTPUT.resolve()).parts)
    except ValueError:
        return f"{path.name}-{hashlib.sha1(str(path).encode()).hexdigest()[:6]}"


def _raw_file(d: Path, m: int) -> Path | None:
    for name in (f"mat{m}.dat", f"mat{m}.dat.zst"):
        if (d / name).exists():
            return d / name
    return None


@dataclass
class Run:
    dir: Path
    engine: str
    raw_dirs: list[Path]
    materials: list[int]
    params: dict = field(default_factory=dict)
    label: str = ""
    rho: dict = field(default_factory=dict)          # material -> nominal density (run.json rho_nom)

    @property
    def name(self) -> str: return self.dir.name
    @property
    def key(self) -> str: return _key(self.dir)
    @property
    def raw_dir(self) -> Path: return self.raw_dirs[0]
    @property
    def raw_key(self) -> str: return "+".join(_key(d) for d in self.raw_dirs)
    @property
    def fig_dir(self) -> str:
        """Figure folder: <engine>/<set> for a set of a bundle under luts/output, else the key."""
        return "/".join(self.key.split("__")) if self.key.count("__") == 1 else self.key
    @property
    def ann(self) -> Path: return self.dir / "ann_LUT.raw"
    @property
    def scatt(self) -> Path: return self.dir / "scatt-eloss_LUT.raw"
    @property
    def census(self) -> Path: return self.dir / "gap_census.csv"
    def has_luts(self) -> bool: return self.ann.is_file() and self.scatt.is_file()
    def raw_file(self, m: int) -> Path:
        """mat<m>.dat[.zst] of the first raw directory that has it (lut_contract.open_raw reads both)."""
        for d in self.raw_dirs:
            f = _raw_file(d, m)
            if f is not None:
                return f
        return self.raw_dirs[0] / f"mat{m}.dat"
    def slot(self, m: int) -> int: return self.all_materials.index(m)    # material slot in the LUT
    def raw_cache(self, m: int, root: Path = CACHE) -> Path:
        return root / "raw" / _key(self.raw_file(m).parent) / f"m{m:02d}.npz"
    def lut_cache(self, m: int, root: Path = CACHE) -> Path:
        return root / "lut" / f"{self.key}@{_key(self.raw_file(m).parent)}" / f"m{m:02d}.npz"

    all_materials: list = field(default_factory=list)     # every material of the tables (slots), checked or not


def _dir(spec: str) -> Path | None:
    for cand in (Path(spec), OUTPUT / spec):
        if cand.is_dir():
            return cand.resolve()
    return None


def _set_dir(d: Path) -> Path:
    """A bundle (no tables of its own, step*/ sets) -> its longest set, by run.json step_cm."""
    if any((d / f).exists() for f in TABLES):
        return d
    sets = [s for s in d.iterdir() if s.is_dir() and s.name.startswith("step") and (s / TABLES[0]).exists()]
    if not sets:
        return d
    step = lambda s: json.loads((s / "run.json").read_text()).get("step_cm", 0) if (s / "run.json").is_file() else 0
    return max(sets, key=step)


def resolve(spec: str | Path, label: str | None = None) -> Run:
    """RUN = TABLES[@RAW[+RAW ...]] -> Run (see the module docstring)."""
    tables, _, raws = str(spec).partition("@")
    d = _dir(tables)
    if d is None:
        raise SystemExit(f"run '{tables}' not found (looked for {tables} and {OUTPUT / tables})")
    d = _set_dir(d)
    info = json.loads((d / "run.json").read_text()) if (d / "run.json").is_file() else {}
    engine = info.get("engine") or next((e for e in ENGINES if e in d.parts), d.name)
    if raws:
        raw_dirs = []
        for r in raws.split("+"):
            rd = _dir(r)
            if rd is None:
                raise SystemExit(f"raw directory '{r}' not found (looked for {r} and {OUTPUT / r})")
            raw_dirs.append(rd)
    else:
        own = Path(info["raw_dir"]) if "raw_dir" in info else d / "raw"
        own = own if own.is_absolute() else REPO / own
        rel = d.relative_to(OUTPUT.resolve()).parts[-2:] if d.is_relative_to(OUTPUT.resolve()) else (engine, d.name)
        raw_dirs = [own.resolve()] if own.is_dir() else [c / Path(*rel) / "raw" for c in CHECK_SAMPLES
                                                         if (c / Path(*rel) / "raw").is_dir()]
        if not raw_dirs:
            raise SystemExit(f"run '{spec}': no raw data (neither {own} nor a check sample under "
                             f"{', '.join(str(c / Path(*rel) / 'raw') for c in CHECK_SAMPLES)}); give TABLES@RAW")
    found = sorted({int(_MAT.match(p.name).group(1)) for rd in raw_dirs for p in rd.glob("mat*.dat*") if _MAT.match(p.name)})
    all_mats = list(info.get("materials") or found)
    mats = [m for m in all_mats if m in found]
    if not mats:
        raise SystemExit(f"run '{spec}': no raw files of its materials in {[str(r) for r in raw_dirs]}")
    if len(mats) < len(all_mats):
        print(f"note: {d.name}: no raw data for materials {[m for m in all_mats if m not in found]}: left out",
              file=sys.stderr)
    rho = dict(zip(all_mats, info.get("rho_nom", [])))
    run = Run(d, engine, raw_dirs, mats, {k: info[k] for k in PARAMS if k in info}, rho=rho, all_materials=all_mats)
    run.label = label or default_label(run)
    return run


def default_label(run: Run) -> str:
    base = run.engine.upper() if run.engine in ENGINES else run.name
    return base if run.name in (run.engine, "step1-1") else f"{base} ({run.name})"


def pair_labels(a: Run, b: Run) -> tuple[str, str]:
    """Engine names, plus the build parameters that differ when the engines match."""
    la, lb = default_label(a), default_label(b)
    if a.engine == b.engine:
        diff = [k for k in PARAMS if a.params.get(k) != b.params.get(k)]
        if diff:
            fmt = lambda r: ", ".join(f"{k.upper()}={r.params.get(k, '?')}" for k in diff)
            la, lb = f"{a.engine.upper()} {fmt(a)}", f"{b.engine.upper()} {fmt(b)}"
    if la == lb:
        la, lb = a.name, b.name
    return la, lb
