"""Integration tests for the PENELOPE driver ``run_penelope.f``.

Runs the PENELOPE driver with a tiny configuration and verifies
the raw-text output is produced correctly in output/penelope/raw/. Compact
binary tables are built separately by build_tables.py --engine penelope
(shared with the GEANT4 engine -- see test_sim_geant4.py).

Slow: requires ``src/sim_penelope/run_penelope.x`` and the material files
(the .mat files are self-contained; PENELOPE's pdfiles/ are not needed).
"""

import subprocess

import pytest

import lut_contract


class TestPenelopeDriver:
    @pytest.mark.slow
    def test_end_to_end_mini_config(
        self, tmp_path, penelope_exe, luts_mats
    ):
        """2 energies × 10 histories × 1 material → clean run, valid raw output."""
        (tmp_path / "input").mkdir()
        (tmp_path / "output" / "penelope" / "raw").mkdir(parents=True)
        (tmp_path / "output" / "penelope" / "logs").mkdir()          # the driver writes per-material warning files there (2026-10-28)

        config = tmp_path / "input" / "sim.in"
        config.write_text(
            "0.05\n"
            "1000.0\n"
            "4600000.0\n"
            "2\n"
            "10\n"
            "input/mats\n"
        )

        mats_link = tmp_path / "input" / "mats"
        mats_link.mkdir()
        (mats_link / "mat1.mat").symlink_to(luts_mats / "mat1.mat")

        result = subprocess.run(
            [str(penelope_exe)],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"run_penelope.x failed (rc={result.returncode}):\n"
            f"STDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        )

        raw_files = sorted(
            (tmp_path / "output" / "penelope" / "raw").glob("mat*.dat")
        )
        assert len(raw_files) == 1, f"Expected 1 raw file, got {len(raw_files)}"
        raw_lines = raw_files[0].read_text().strip().split("\n")
        # line 1: run configuration (read by build_tables.py), line 2: column names
        cfg = lut_contract.read_raw_config(raw_files[0])
        assert cfg["engine"] == "penelope" and cfg["step_cm"] == 0.05 and cfg["nhist"] == 10
        assert cfg["rho_gcm3"] == pytest.approx(
            float((luts_mats / "mat1.mat").read_text().split("Mass density =")[1].split()[0]), rel=1e-6)
        assert lut_contract.raw_has_stop(raw_files[0])
        assert raw_lines[0].startswith("# config:") and raw_lines[1].startswith("# E0_keV")
        assert len(raw_lines) - 2 == 20, (  # 2 energies × 10 histories
            f"Expected 20 data rows, got {len(raw_lines) - 2}"
        )

        # Same schema as GEANT4: E0_keV theta_rad DE_total_eV dE_kin_eV
        # E_final_eV annihil_flag matID z_cm r_cm dphi_rad (10th column since 2026-10-30)
        for line in raw_lines[2:]:
            fields = line.split()
            assert len(fields) == 10, f"Expected 10 fields, got {len(fields)}: {line}"
            assert -3.1416 <= float(fields[9]) <= 3.1416
            annihil_flag = int(fields[5])
            mat_id = int(fields[6])
            z, r = float(fields[7]), float(fields[8])
            assert annihil_flag in (0, 1)
            assert mat_id == 1
            assert -0.05 <= z <= 0.05 and 0 <= r <= 0.05
            if annihil_flag == 0:                          # survivors travelled the whole step
                assert z * z + r * r > 0.5 * 0.05 ** 2 or z < 0.05
        # 4.6 MeV positrons leave the step almost straight: z close to STEP_CM
        high = [l.split() for l in raw_lines[2:] if l.split()[0].startswith("4600") and l.split()[5] == "0"]
        assert high and all(float(f[7]) > 0.04 for f in high)
        assert not list((tmp_path / "output" / "penelope").glob("*.err"))   # empty logs are removed

    @pytest.mark.slow
    def test_material_range_arguments(self, tmp_path, penelope_exe, luts_mats):
        """run_penelope.x A B simulates materials A..B only; out of range is an error."""
        (tmp_path / "input" / "mats").mkdir(parents=True)
        (tmp_path / "output" / "penelope" / "raw").mkdir(parents=True)
        (tmp_path / "output" / "penelope" / "logs").mkdir()          # the driver writes per-material warning files there (2026-10-28)
        (tmp_path / "input" / "sim.in").write_text("0.05\n1000.0\n4600000.0\n2\n5\ninput/mats\n")
        for n in (1, 2, 3):
            (tmp_path / "input" / "mats" / f"mat{n}.mat").symlink_to(luts_mats / f"mat{n}.mat")
        r = subprocess.run([str(penelope_exe), "2", "3"], cwd=str(tmp_path), capture_output=True,
                           text=True, timeout=60)
        assert r.returncode == 0, r.stdout + r.stderr
        assert sorted(p.name for p in (tmp_path / "output" / "penelope" / "raw").glob("mat*.dat")) \
            == ["mat2.dat", "mat3.dat"]
        r = subprocess.run([str(penelope_exe), "4"], cwd=str(tmp_path), capture_output=True, text=True)
        assert r.returncode != 0 and "outside" in r.stdout + r.stderr
