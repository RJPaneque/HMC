"""Shared pytest fixtures for the luts test suite.

Puts ``luts/src/lutgen`` and ``luts/src/analysis`` on ``sys.path`` so test
modules can import ``build_tables``, ``lut_contract`` and ``runs`` directly, and exposes the locations of
the PENELOPE executable, material files, and the GEANT4 Docker image.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# --- make luts/src/lutgen importable ----------------------------------------
SRC = Path(__file__).resolve().parent.parent / "src" / "lutgen"
ANALYSIS = SRC.parent / "analysis"
for p in (ANALYSIS, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# --- canonical locations ----------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent          # luts/
TABLEGEN_EXE = _ROOT / "src" / "sim_penelope" / "run_penelope.x"
MATS_DIR = _ROOT / "input" / "mats"
INPUT_DIR = _ROOT / "input"
GEANT4_IMAGE = "luts-geant4"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: requires run_penelope.x + mat files, "
        "or the luts-geant4 Docker image",
    )


# --- fixtures ----------------------------------------------------------------
@pytest.fixture
def luts_root():
    return _ROOT


@pytest.fixture
def penelope_exe():
    if not TABLEGEN_EXE.is_file():
        pytest.skip("luts/src/sim_penelope/run_penelope.x not found (run COMPILAR.sh)")
    return TABLEGEN_EXE


@pytest.fixture
def luts_mats():
    if not MATS_DIR.is_dir() or not (MATS_DIR / "mat1.mat").exists():
        pytest.skip("luts/input/mats/ missing")
    return MATS_DIR


@pytest.fixture
def geant4_image():
    """Skip unless Docker is available and the luts-geant4 image is built."""
    if shutil.which("docker") is None:
        pytest.skip("docker not found on PATH")
    result = subprocess.run(
        ["docker", "image", "inspect", GEANT4_IMAGE],
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip(f"docker image '{GEANT4_IMAGE}' not built (run luts/src/sim_geant4/run_geant4.sh)")
    return GEANT4_IMAGE
