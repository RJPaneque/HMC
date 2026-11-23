"""Shared pytest fixtures for the spectrum test suite.

Puts ``spectrum/src`` on ``sys.path`` (``import generate_spectra``) and skips the tests that run PENNUC when
``decays_positron.x`` (``spectrum/src/build.sh``) or the nuclide database (``spectrum/input/nuclides``) is missing.
"""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent          # spectrum/
SRC = _ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs decays_positron.x (needs spectrum/src/decays_positron.x + input/nuclides)")


@pytest.fixture
def spectra_dir():
    return _ROOT / "output" / "spectra"


@pytest.fixture
def pennuc():
    import generate_spectra as g
    if not g.DECAYS_X.is_file():
        pytest.skip("spectrum/src/decays_positron.x not built -- run spectrum/src/build.sh")
    if not (g.NUCLIDES / "F-18.nuc").is_file():
        pytest.skip("spectrum/input/nuclides not linked to the PENNUC database (PenEasy_2024/nuclides)")
    return g
