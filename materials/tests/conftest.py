"""Shared pytest fixtures for the materials test suite.

Puts ``materials/src`` on ``sys.path`` so test modules can import
``schneider_materials`` directly, and exposes the locations of the
generated ``.mat`` files and the PENELOPE executables.
"""

import sys
from pathlib import Path

import pytest

# --- make materials/src importable as 'schneider_materials' ---------------
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --- canonical locations --------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent          # materials/
MATS_DIR = _ROOT / "output" / "30mats"
MATERIAL_EXE = _ROOT / "material.x"
PENELOPE_WORKDIR = _ROOT / "input"                       # contains pdfiles/


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: requires material.x + pdfiles/")


# --- fixtures --------------------------------------------------------------
@pytest.fixture
def materials_root():
    return _ROOT


@pytest.fixture
def mats_dir():
    if not MATS_DIR.is_dir():
        pytest.skip("materials/output/30mats/ missing -- run `pixi run materials-gen`")
    return MATS_DIR


@pytest.fixture
def material_exe():
    if not MATERIAL_EXE.is_file():
        pytest.skip("materials/material.x not found")
    return MATERIAL_EXE


@pytest.fixture
def penelope_workdir():
    if not (PENELOPE_WORKDIR / "pdfiles").is_dir():
        pytest.skip("pdfiles/ not found in materials/input/")
    return PENELOPE_WORKDIR