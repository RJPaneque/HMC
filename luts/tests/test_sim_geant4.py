"""Integration tests for the GEANT4 driver (Docker).

Runs the containerized GEANT4 driver with a tiny configuration and
verifies the raw-text output matches the same schema PENELOPE's driver
produces, in output/geant4/raw/.

Slow: requires Docker and the luts-geant4 image (build with
``src/sim_geant4/run_geant4.sh`` or ``FORCE_BUILD=1 src/sim_geant4/run_geant4.sh``).
"""

import subprocess

import pytest

import lut_contract


class TestGeant4Driver:
    @pytest.mark.slow
    def test_end_to_end_mini_config(
        self, tmp_path, geant4_image, luts_mats
    ):
        """2 energies x 10 histories x 1 material -> clean run, valid raw output."""
        (tmp_path / "input").mkdir()
        (tmp_path / "output" / "geant4" / "raw").mkdir(parents=True)

        config = tmp_path / "input" / "sim.in"
        config.write_text(
            "0.05\n"
            "1000.0\n"
            "4600000.0\n"
            "2\n"
            "10\n"
            "input/mats\n"
        )

        mats_dir = tmp_path / "input" / "mats"
        mats_dir.mkdir()
        (mats_dir / "mat1.mat").write_text(
            (luts_mats / "mat1.mat").read_text()
        )

        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{tmp_path / 'input'}:/work/input:ro",
                "-v", f"{tmp_path / 'output' / 'geant4'}:/work/output/geant4",
                geant4_image,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"GEANT4 driver failed (rc={result.returncode}):\n"
            f"STDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        )

        raw_files = sorted(
            (tmp_path / "output" / "geant4" / "raw").glob("mat*.dat")
        )
        assert len(raw_files) == 1, f"Expected 1 raw file, got {len(raw_files)}"
        raw_lines = raw_files[0].read_text().strip().split("\n")
        # line 1: run configuration (read by build_tables.py), line 2: column names
        cfg = lut_contract.read_raw_config(raw_files[0])
        assert cfg["engine"] == "geant4" and cfg["step_cm"] == 0.05 and cfg["nhist"] == 10
        assert cfg["rho_gcm3"] == pytest.approx(
            float((luts_mats / "mat1.mat").read_text().split("Mass density =")[1].split()[0]), rel=1e-6)
        assert lut_contract.raw_has_stop(raw_files[0])
        assert raw_lines[0].startswith("# config:") and raw_lines[1].startswith("# E0_keV")
        assert len(raw_lines) - 2 == 20, (  # 2 energies x 10 histories
            f"Expected 20 data rows, got {len(raw_lines) - 2}"
        )

        # Same schema as PENELOPE: E0_keV theta_rad DE_total_eV dE_kin_eV
        # E_final_eV annihil_flag matID z_cm r_cm
        for line in raw_lines[2:]:
            fields = line.split()
            # 9 columns, or 10 with dphi_rad (step-end tables, LUT v5)
            assert len(fields) in (9, 10), f"Expected 9 or 10 fields, got {len(fields)}: {line}"
            annihil_flag = int(fields[5])
            mat_id = int(fields[6])
            z, r = float(fields[7]), float(fields[8])
            assert annihil_flag in (0, 1)
            assert mat_id == 1
            assert -0.051 <= z <= 0.051 and 0 <= r <= 0.051
        high = [l.split() for l in raw_lines[2:] if l.split()[0].startswith("4600") and l.split()[5] == "0"]
        assert high and all(float(f[7]) > 0.04 for f in high)
