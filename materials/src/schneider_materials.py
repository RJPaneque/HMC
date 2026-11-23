"""
Schneider et al. (2000) Phys. Med. Biol. 45 459-478
CT number to tissue parameter conversion.

Two modes:
  continuous=False   Table 6 step-function (24 bins, default).
  continuous=True    Equations (20, 22, 24) from the paper
                     interpolating composition between anchor tissues.
"""

from dataclasses import dataclass
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Table 6: HU bins and elemental compositions (weight percent / "pp")
# ---------------------------------------------------------------------------
_schneider_table6: list[dict] = [
    {"label": "Air",           "h_min": -1000, "h_max": -950,
     "H":0.0,  "C":0.0,  "N":75.5, "O":23.2,
     "Na":0.0, "Mg":0.0, "P":0.0,  "S":0.0, "Cl":0.0, "Ar":1.3, "K":0.0, "Ca":0.0},
    {"label": "Lung",          "h_min": -950, "h_max": -120,
     "H":10.3, "C":10.5, "N":3.1,  "O":74.9,
     "Na":0.2, "Mg":0.0, "P":0.2,  "S":0.3, "Cl":0.3, "Ar":0.0, "K":0.2, "Ca":0.0},
    {"label": "Adipose_1",     "h_min": -120, "h_max": -82,
     "H":11.6, "C":68.1, "N":0.2,  "O":19.8,
     "Na":0.1, "Mg":0.0, "P":0.0,  "S":0.1, "Cl":0.1, "Ar":0.0, "K":0.0, "Ca":0.0},
    {"label": "Adipose_2",     "h_min": -82,  "h_max": -52,
     "H":11.3, "C":56.7, "N":0.9,  "O":30.8,
     "Na":0.1, "Mg":0.0, "P":0.0,  "S":0.1, "Cl":0.1, "Ar":0.0, "K":0.0, "Ca":0.0},
    {"label": "Adipose_3",     "h_min": -52,  "h_max": -22,
     "H":11.0, "C":45.8, "N":1.5,  "O":41.1,
     "Na":0.1, "Mg":0.0, "P":0.1,  "S":0.2, "Cl":0.2, "Ar":0.0, "K":0.0, "Ca":0.0},
    {"label": "Soft_1",        "h_min": -22,  "h_max":  8,
     "H":10.8, "C":35.6, "N":2.2,  "O":50.9,
     "Na":0.0, "Mg":0.0, "P":0.1,  "S":0.2, "Cl":0.2, "Ar":0.0, "K":0.0, "Ca":0.0},
    {"label": "Soft_2",        "h_min":   8,  "h_max":  19,
     "H":10.6, "C":28.4, "N":2.6,  "O":57.8,
     "Na":0.0, "Mg":0.0, "P":0.1,  "S":0.2, "Cl":0.2, "Ar":0.0, "K":0.1, "Ca":0.0},
    {"label": "Soft_mean",     "h_min":  19,  "h_max":  80,
     "H":10.3, "C":13.4, "N":3.0,  "O":72.3,
     "Na":0.2, "Mg":0.0, "P":0.2,  "S":0.2, "Cl":0.2, "Ar":0.0, "K":0.2, "Ca":0.0},
    {"label": "Connective",    "h_min":  80,  "h_max": 120,
     "H":9.4,  "C":20.7, "N":6.2,  "O":62.2,
     "Na":0.6, "Mg":0.0, "P":0.0,  "S":0.6, "Cl":0.3, "Ar":0.0, "K":0.0, "Ca":0.0},
    {"label": "Skeletal_120_200",  "h_min": 120, "h_max": 200,
     "H":9.5,"C":45.5,"N":2.5,"O":35.5, "Na":0.1,"Mg":0.0,"P":2.1,"S":0.1,"Cl":0.1,"Ar":0.0,"K":0.1,"Ca":4.5},
    {"label": "Skeletal_200_300",  "h_min": 200, "h_max": 300,
     "H":8.9,"C":42.3,"N":2.7,"O":36.3, "Na":0.1,"Mg":0.0,"P":3.0,"S":0.1,"Cl":0.1,"Ar":0.0,"K":0.1,"Ca":6.4},
    {"label": "Skeletal_300_400",  "h_min": 300, "h_max": 400,
     "H":8.2,"C":39.1,"N":2.9,"O":37.2, "Na":0.1,"Mg":0.0,"P":3.9,"S":0.1,"Cl":0.1,"Ar":0.0,"K":0.1,"Ca":8.3},
    {"label": "Skeletal_400_500",  "h_min": 400, "h_max": 500,
     "H":7.6,"C":36.1,"N":3.0,"O":38.0, "Na":0.1,"Mg":0.1,"P":4.7,"S":0.2,"Cl":0.1,"Ar":0.0,"K":0.0,"Ca":10.1},
    {"label": "Skeletal_500_600",  "h_min": 500, "h_max": 600,
     "H":7.1,"C":33.5,"N":3.2,"O":38.7, "Na":0.1,"Mg":0.1,"P":5.4,"S":0.2,"Cl":0.0,"Ar":0.0,"K":0.0,"Ca":11.7},
    {"label": "Skeletal_600_700",  "h_min": 600, "h_max": 700,
     "H":6.6,"C":31.0,"N":3.3,"O":39.4, "Na":0.1,"Mg":0.1,"P":6.1,"S":0.2,"Cl":0.0,"Ar":0.0,"K":0.0,"Ca":13.2},
    {"label": "Skeletal_700_800",  "h_min": 700, "h_max": 800,
     "H":6.1,"C":28.7,"N":3.5,"O":40.0, "Na":0.1,"Mg":0.1,"P":6.7,"S":0.2,"Cl":0.0,"Ar":0.0,"K":0.0,"Ca":14.6},
    {"label": "Skeletal_800_900",  "h_min": 800, "h_max": 900,
     "H":5.6,"C":26.5,"N":3.6,"O":40.5, "Na":0.1,"Mg":0.2,"P":7.3,"S":0.3,"Cl":0.0,"Ar":0.0,"K":0.0,"Ca":15.9},
    {"label": "Skeletal_900_1000", "h_min": 900, "h_max": 1000,
     "H":5.2,"C":24.6,"N":3.7,"O":41.1, "Na":0.1,"Mg":0.2,"P":7.8,"S":0.3,"Cl":0.0,"Ar":0.0,"K":0.0,"Ca":17.0},
    {"label": "Skeletal_1000_1100","h_min":1000, "h_max": 1100,
     "H":4.9,"C":22.7,"N":3.8,"O":41.6, "Na":0.1,"Mg":0.2,"P":8.3,"S":0.3,"Cl":0.0,"Ar":0.0,"K":0.0,"Ca":18.1},
    {"label": "Skeletal_1100_1200","h_min":1100, "h_max": 1200,
     "H":4.5,"C":21.0,"N":3.9,"O":42.0, "Na":0.1,"Mg":0.2,"P":8.8,"S":0.3,"Cl":0.0,"Ar":0.0,"K":0.0,"Ca":19.2},
    {"label": "Skeletal_1200_1300","h_min":1200, "h_max": 1300,
     "H":4.2,"C":19.4,"N":4.0,"O":42.5, "Na":0.1,"Mg":0.2,"P":9.2,"S":0.3,"Cl":0.0,"Ar":0.0,"K":0.0,"Ca":20.1},
    {"label": "Skeletal_1300_1400","h_min":1300, "h_max": 1400,
     "H":3.9,"C":17.9,"N":4.1,"O":42.9, "Na":0.1,"Mg":0.2,"P":9.6,"S":0.3,"Cl":0.0,"Ar":0.0,"K":0.0,"Ca":21.0},
    {"label": "Skeletal_1400_1500","h_min":1400, "h_max": 1500,
     "H":3.6,"C":16.5,"N":4.2,"O":43.2, "Na":0.1,"Mg":0.2,"P":10.0,"S":0.3,"Cl":0.0,"Ar":0.0,"K":0.0,"Ca":21.9},
    {"label": "Skeletal_1500_1600","h_min":1500, "h_max": 1600,
     "H":3.4,"C":15.5,"N":4.2,"O":43.5, "Na":0.1,"Mg":0.2,"P":10.3,"S":0.3,"Cl":0.0,"Ar":0.0,"K":0.0,"Ca":22.5},
]


# ---------------------------------------------------------------------------
# Anchor tissues for continuous interpolation (equations 20, 22, 24)
# "other" = Na + Mg + S + Cl + K + Fe (trace elements combined by the paper)
# ---------------------------------------------------------------------------

# Eq. (22):  [-98, 14]   adipose tissue 3 <-> adrenal gland
_ANCHOR_ADIPOSE = {"H": 11.6, "C": 68.1, "N": 0.2, "O": 19.8, "P": 0.0, "Ca": 0.0, "other": 0.3}
_ANCHOR_ADRENAL = {"H": 10.6, "C": 28.4, "N": 2.6, "O": 57.8, "P": 0.1, "Ca": 0.0, "other": 0.5}

# Eq. (24):  [23, 100]   small intestine (wall) <-> connective tissue
_ANCHOR_SMALL_INT = {"H": 10.6, "C": 11.5, "N": 2.2, "O": 75.1, "P": 0.1, "Ca": 0.0, "other": 0.5}
_ANCHOR_CONNECTIVE = {"H": 9.4, "C": 20.7, "N": 6.2, "O": 62.2, "P": 0.0, "Ca": 0.0, "other": 1.5}

# Eq. (20):  [-22, 1524]  red/yellow marrow (1:1) <-> skeleton cortical bone
_ANCHOR_MARROW = {"H": 11.0, "C": 52.9, "N": 2.1, "O": 33.5, "P": 0.1, "Ca": 0.0, "other": 0.4}
_ANCHOR_CORTICAL = {"H": 3.4, "C": 15.5, "N": 4.2, "O": 43.5, "P": 10.3, "Ca": 22.5, "other": 0.6}

_ALL_ELEMENTS = ["H", "C", "N", "O", "Na", "Mg", "P", "S", "Cl", "Ar", "K", "Ca"]


def hu_to_density(hu: float) -> float:
    """Schneider calibration: CT number -> mass density (g/cm^3)."""
    if hu < -1000.0:
        return 0.00121
    elif hu < -98.0:
        return 0.00121 + (hu + 1000.0) * (0.93 - 0.00121) / 902.0
    elif hu <= 14.0:
        return 1.018 + 0.893e-3 * hu
    elif hu <= 23.0:
        return 1.03
    elif hu <= 100.0:
        return 1.003 + 1.169e-3 * hu
    else:
        return 1.017 + 0.592e-3 * hu


def density_to_hu(rho: float) -> float:
    """Inverse Schneider calibration: mass density (g/cm^3) -> CT number."""
    if rho <= 0.00121:
        return -1000.0
    elif rho <= 0.93:
        return (rho - 0.00121) * 902.0 / (0.93 - 0.00121) - 1000.0
    elif abs(rho - 1.03) < 1e-9:
        # plateau (HU in (14, 23] -> rho = 1.03) -> representative midpoint 18.5.
        # rho = 1.03 is also reachable from the adipose branch (HU = 13.44),
        # but we resolve the ambiguity in favour of the plateau midpoint.
        return 18.5
    elif rho <= 1.018 + 0.893e-3 * 14.0:
        return (rho - 1.018) / 0.893e-3
    elif rho <= 1.003 + 1.169e-3 * 100.0:
        return (rho - 1.003) / 1.169e-3
    else:
        return (rho - 1.017) / 0.592e-3


# ---------------------------------------------------------------------------
# Continuous interpolation of composition using anchor tissue pairs
# ---------------------------------------------------------------------------

def _interp_w1(
    hu: float,
    rho1: float, h1: float,
    rho2: float, h2: float,
) -> float:
    """Fraction (by weight) of the first component in the mixture.

       W1 = rho1*(h2 - hu) / (rho1*h2 - rho2*h1 + (rho2 - rho1)*hu)
       Eq (16) from Schneider et al. 2000.
    """
    denom = rho1 * h2 - rho2 * h1 + (rho2 - rho1) * hu
    if abs(denom) < 1e-20:
        return 0.5
    return rho1 * (h2 - hu) / denom


def _interpolate(
    hu: float,
    w1: dict[str, float],   # first component  (pp)
    w2: dict[str, float],   # second component (pp)
    rho1: float, h1: float,
    rho2: float, h2: float,
) -> dict[str, float]:
    """Eqs (12)-(13): continuous composition between two anchor tissues.

       w_i(H) = W1*(w1_i - w2_i) + w2_i      with W1 from eq (16)

    Returns weight fractions (0-1), non-zero entries only.
    """
    f1 = _interp_w1(hu, rho1, h1, rho2, h2)
    result: dict[str, float] = {}
    for key in w1:
        val = f1 * (w1[key] - w2[key]) + w2[key]
        if val > 1e-9:
            result[key] = val / 100.0
    return result


def _comp_continuous(hu: float) -> dict[str, float]:
    """Continuous composition via equations (20), (22), (24)."""
    h = max(-1000.0, min(hu, 1524.0))

    # --- air / lung: stepped (paper gives no interpolation function) ---
    if h < -98.0:
        if h <= -950.0:
            return {"N": 0.755, "O": 0.232, "Ar": 0.013}
        return {"H": 0.103, "C": 0.105, "N": 0.031, "O": 0.749,
                "Na": 0.002, "P": 0.002, "S": 0.003, "Cl": 0.003, "K": 0.002}

    # --- Eq (22): adipose tissue 3 <-> adrenal gland  [-98, 14] ---
    if h <= 14.0:
        return _interpolate(h, _ANCHOR_ADIPOSE, _ANCHOR_ADRENAL,
                            0.93, -98.0, 1.03, 14.0)

    # --- Gap [14, 23]: linear blend endpoint -> start ---
    if h <= 23.0:
        ep = _interpolate(14.0, _ANCHOR_ADIPOSE, _ANCHOR_ADRENAL,
                          0.93, -98.0, 1.03, 14.0)
        st = _interpolate(23.0, _ANCHOR_SMALL_INT, _ANCHOR_CONNECTIVE,
                          1.03, 23.0, 1.12, 100.0)
        t = (h - 14.0) / 9.0
        result: dict[str, float] = {}
        for k in set(ep) | set(st):
            val = (1.0 - t) * ep.get(k, 0.0) + t * st.get(k, 0.0)
            if val > 1e-9:
                result[k] = val
        return result

    # --- Eq (24): small intestine <-> connective  [23, 100] ---
    if h <= 100.0:
        return _interpolate(h, _ANCHOR_SMALL_INT, _ANCHOR_CONNECTIVE,
                            1.03, 23.0, 1.12, 100.0)

    # --- Eq (20): marrow <-> cortical bone  [-22, 1524] ---
    return _interpolate(h, _ANCHOR_MARROW, _ANCHOR_CORTICAL,
                        1.00, -22.0, 1.92, 1524.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class Material:
    label: str
    h_min: int
    h_max: int
    density: float
    composition: dict[str, float]


def _composition_from_entry(entry: dict, elist: list[str]) -> dict[str, float]:
    """Convert Table 6 entry to weight-fraction dict filtering zero values."""
    return {el: entry[el] / 100.0 for el in elist if entry[el] > 0.0}


def get_material(
    hu: Optional[float] = None,
    density: Optional[float] = None,
    continuous: bool = False,
) -> Material:
    """Return Schneider material composition + density.

    Parameters
    ----------
    hu : float, optional
        Hounsfield units. Takes precedence over *density*.
    density : float, optional
        Mass density in g/cm^3. Used if *hu* is not given.
    continuous : bool
        If True, use equations (20,22,24) to interpolate composition
        continuously between anchor tissues. Otherwise use Table 6 bins.

    Returns
    -------
    Material
    """
    if hu is None and density is None:
        raise ValueError("Provide either 'hu' or 'density'.")
    if hu is None:
        assert density is not None
        hu = density_to_hu(density)
    else:
        density = hu_to_density(hu)

    if continuous:
        comp = _comp_continuous(hu)
        hu_clamped = max(-1000, min(int(hu), 1600))
        return Material(
            label=f"CT={hu:.1f}",
            h_min=hu_clamped,
            h_max=hu_clamped,
            density=density,
            composition=comp,
        )

    # ---- step-function (Table 6 bins) ----
    clamped = max(-1000, min(hu, 1599))
    entry = None
    for e in _schneider_table6:
        if e["h_min"] <= clamped < e["h_max"]:
            entry = e
            break
    if entry is None:
        entry = _schneider_table6[-1] if clamped >= _schneider_table6[-1]["h_max"] else _schneider_table6[0]

    comp = _composition_from_entry(entry, _ALL_ELEMENTS)
    return Material(
        label=entry["label"],
        h_min=entry["h_min"],
        h_max=entry["h_max"],
        density=density,
        composition=comp,
    )


def get_all_materials(continuous: bool = False) -> list[Material]:
    """Return all 24 Table 6 materials (bin midpoints)."""
    result = []
    for entry in _schneider_table6:
        hu_mid = (entry["h_min"] + entry["h_max"]) / 2.0
        rho = hu_to_density(hu_mid)
        comp = _composition_from_entry(entry, _ALL_ELEMENTS)
        result.append(Material(
            label=entry["label"],
            h_min=entry["h_min"],
            h_max=entry["h_max"],
            density=rho,
            composition=comp,
        ))
    return result


def _read_penelope_mat(filepath: str) -> Optional[tuple[float, dict[str, float]]]:
    """Parse PENELOPE .mat file header: return (density, {el: wt_frac})."""
    import re

    atomic_mass = {
        "H": 1.00794, "C": 12.0107, "N": 14.0067, "O": 15.9994,
        "Na": 22.98977, "Mg": 24.3050, "P": 30.97376, "S": 32.065,
        "Cl": 35.453, "Ar": 39.948, "K": 39.0983, "Ca": 40.078,
    }
    z_to_el = {1: "H", 6: "C", 7: "N", 8: "O", 11: "Na", 12: "Mg",
               15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca"}

    with open(filepath) as fh:
        content = fh.read(8192)

    rho_match = re.search(r"Mass density\s*=\s*([\d.E+\-]+)", content)
    if not rho_match:
        return None
    rho = float(rho_match.group(1))

    atoms: dict[str, float] = {}
    for m in re.finditer(r"atomic number\s*=\s*(\d+)\s*,\s*atoms/molecule\s*=\s*([\d.E+\-]+)", content):
        z = int(m.group(1))
        if z in z_to_el:
            atoms[z_to_el[z]] = float(m.group(2))

    total_mass = sum(am * atomic_mass[el] for el, am in atoms.items())
    wt_frac = {el: am * atomic_mass[el] / total_mass for el, am in atoms.items() if am > 0}
    return rho, wt_frac


def match_penelope_file(filepath: str) -> Optional[Material]:
    """Read a PENELOPE .mat file and identify which Schneider material it is.

    Matching strategy (in order):
    1. If the name matches ``matN.mat`` with N in 1..24, the N-th
       Table 6 entry is used (original file set).
    2. Otherwise the material composition (atoms/molecule) is compared
       against every Table 6 entry and the closest match is returned.
    """
    import os, re

    rho, wt_frac = _read_penelope_mat(filepath)
    if rho is None:
        return None

    # Strategy 1: index-based for original 24-material set
    basename = os.path.basename(filepath)
    m = re.match(r"mat(\d+)", basename)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(_schneider_table6):
            # Quick check: does the indexed entry's composition actually
            # match the file?  If not, fall through.
            expected = _composition_from_entry(_schneider_table6[idx], _ALL_ELEMENTS)
            match_score = sum(
                (expected.get(el, 0.0) - wt_frac.get(el, 0.0)) ** 2
                for el in set(expected) | set(wt_frac)
            )
            if match_score < 1e-4:  # very close match (<0.1% rms diff)
                entry = _schneider_table6[idx]
                return Material(
                    label=entry["label"],
                    h_min=entry["h_min"],
                    h_max=entry["h_max"],
                    density=rho,
                    composition=expected,
                )

    # Strategy 2: composition-based matching
    best_entry = None
    best_dist = float("inf")
    for entry in _schneider_table6:
        comp = _composition_from_entry(entry, _ALL_ELEMENTS)
        dist = sum(
            (comp.get(el, 0.0) - wt_frac.get(el, 0.0)) ** 2
            for el in set(comp) | set(wt_frac)
        )
        if dist < best_dist:
            best_dist = dist
            best_entry = entry
    if best_entry is None:
        return None
    return Material(
        label=best_entry["label"],
        h_min=best_entry["h_min"],
        h_max=best_entry["h_max"],
        density=rho,
        composition=_composition_from_entry(best_entry, _ALL_ELEMENTS),
    )


# ---------------------------------------------------------------------------
# Generating PENELOPE .mat files via material.x
# ---------------------------------------------------------------------------

_ATOMIC_MASS = {
    "H": 1.00794, "C": 12.0107, "N": 14.0067, "O": 15.9994,
    "Na": 22.98977, "Mg": 24.3050, "P": 30.97376, "S": 32.065,
    "Cl": 35.453, "Ar": 39.948, "K": 39.0983, "Ca": 40.078,
}
_Z_ORDER = [(1, "H"), (6, "C"), (7, "N"), (8, "O"), (11, "Na"),
            (12, "Mg"), (15, "P"), (16, "S"), (17, "Cl"), (18, "Ar"),
            (19, "K"), (20, "Ca")]


def _weight_to_atoms(
    composition: dict[str, float],
) -> dict[str, float]:
    """Convert weight fractions to atoms-per-molecule.

    Normalized so the first element in Z-order among those present
    has exactly 1 atom/molecule. Elements below 0.01% weight
    fraction are omitted.
    """
    mol_per_el: dict[str, float] = {}
    ref_mol = 0.0
    ref_el = "H"
    for _z, el in _Z_ORDER:
        w = composition.get(el, 0.0)
        if w > 1e-4:
            m = w / _ATOMIC_MASS[el]
            mol_per_el[el] = m
            if ref_mol == 0.0:
                ref_mol = m
                ref_el = el
    return {el: m / ref_mol for el, m in mol_per_el.items()}


def build_material_input(
    material: Material,
    output_path: str,
    *,
    label: Optional[str] = None,
) -> str:
    """Return the stdin content expected by material.x.

    Parameters
    ----------
    material : Material
        The Schneider material to encode.
    output_path : str
        Path for the output .mat file (relative to material.x CWD).
    label : str, optional
        Material name (up to 60 chars, defaults to material.label).

    Returns
    -------
    str
        Content to pipe to material.x stdin.
    """
    atoms = _weight_to_atoms(material.composition)
    n_elem = len(atoms)
    lines = [
        "1",                                    # keyboard input
        (label or material.label)[:60],         # material name
        str(n_elem),                            # number of elements
        "1",                                    # stoichiometric formula
    ]
    for _z, el in _Z_ORDER:
        if el in atoms:
            z = _z
            val = atoms[el]
            lines.append(f"{z} {val:.8E}")
    lines += [
        "2",                                    # keep calculated I-value
        f"{material.density:.8E}",              # mass density
        "2",                                    # keep Fcb/Wcb
        output_path,                            # output file
    ]
    return "\n".join(lines)


def generate_mat_file(
    material: Material,
    material_exe: str,
    output_path: str,
    *,
    label: Optional[str] = None,
    workdir: Optional[str] = None,
    timeout: float = 120.0,
) -> Material:
    """Run material.x to produce a full PENELOPE .mat file.

    Parameters
    ----------
    material : Material
        Composition + density.
    material_exe : str
        Path to the material.x executable.
    output_path : str
        Where to write the .mat file (absolute path recommended).
    label : str, optional
        Material name in the file header.
    workdir : str, optional
        Working directory for material.x (must contain pdfiles/).
        Defaults to the executable's directory.
    timeout : float
        Seconds before the subprocess is killed.

    Returns
    -------
    Material
        The same material (unchanged), for chaining.
    """
    import os, subprocess, pathlib

    material_exe = os.path.abspath(material_exe)
    if workdir is None:
        workdir = str(pathlib.Path(material_exe).parent)

    # Use a filename relative to workdir if output_path is absolute
    output_basename = os.path.basename(output_path)
    stdin = build_material_input(material, output_basename, label=label)

    result = subprocess.run(
        [material_exe],
        input=stdin,
        capture_output=True,
        text=True,
        cwd=workdir,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr_tail = result.stderr[-500:] if result.stderr else ""
        raise RuntimeError(
            f"material.x failed (rc={result.returncode})\n"
            f"stderr tail: {stderr_tail}"
        )

    # The .mat file is written to workdir/output_basename
    generated = os.path.join(workdir, output_basename)
    if os.path.abspath(generated) != os.path.abspath(output_path):
        import shutil
        shutil.move(generated, output_path)
    return material
