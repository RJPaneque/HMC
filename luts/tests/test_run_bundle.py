"""luts/src/run_bundle.py (luts/run_sim.sh): ladder names, material lists, resume, the guard on step folders with tables,
and the runner commands (dry run; no simulation)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_bundle as rb  # noqa: E402


def test_ladder_halvings():
    assert rb.ladder(0.05, 4) == [("step1-1", 0.05), ("step1-2", 0.025), ("step1-4", 0.0125), ("step1-8", 0.00625)]
    assert rb.ladder(0.05, 1) == [("step1-1", 0.05)]


def test_ladder_explicit_steps():
    assert rb.ladder(0.05, steps=[0.05, 0.0125]) == [("step1-1", 0.05), ("step1-4", 0.0125)]
    assert rb.ladder(0.05, steps=[0.05 / 3]) == [("step1-3", 0.05 / 3)]
    with pytest.raises(ValueError, match="integer"):
        rb.ladder(0.05, steps=[0.03])
    with pytest.raises(ValueError, match="decreasing"):
        rb.ladder(0.05, steps=[0.025, 0.05])
    with pytest.raises(ValueError):
        rb.ladder(0.05, 0)


def test_mats_and_ranges(tmp_path):
    assert rb.parse_mats("12,30,1-3", tmp_path) == [1, 2, 3, 12, 30]
    for m in (1, 2, 3):
        (tmp_path / f"mat{m}.mat").touch()
    assert rb.parse_mats(None, tmp_path) == [1, 2, 3]
    assert rb.ranges([1, 2, 3, 7, 9, 10]) == [(1, 3), (7, 7), (9, 10)]


def test_missing_counts_dat_and_zst(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "mat1.dat").touch()
    (tmp_path / "raw" / "mat2.dat.zst").touch()
    assert rb.missing_mats(tmp_path, [1, 2, 3]) == [3]


def _sim_in(tmp_path, monkeypatch):
    """A luts/ tree with sim.in (STEP_CM 0.05) and three materials."""
    luts = tmp_path / "luts"
    (luts / "input" / "mats").mkdir(parents=True)
    (luts / "input" / "sim.in").write_text("0.05 ! STEP_CM\n1000 ! EMIN\n4.6e6 ! EMAX\n116 ! NENRG\n100000 ! NHIST\n"
                                           "input/mats ! MAT_DIR\n")
    for m in (1, 2, 3):
        (luts / "input" / "mats" / f"mat{m}.mat").touch()
    monkeypatch.setattr(rb, "LUTS", luts)
    return luts


def test_dry_run_skips_sets_with_tables_and_resumes(tmp_path, monkeypatch, capsys):
    _sim_in(tmp_path, monkeypatch)
    out = tmp_path / "out"
    s1 = out / "egsnrc" / "step1-1"
    s1.mkdir(parents=True)
    (s1 / "stepend_LUT.raw").touch()                     # a built set: never written to
    s2 = out / "egsnrc" / "step1-2" / "raw"
    s2.mkdir(parents=True)
    (s2 / "mat1.dat.zst").touch()                        # material 1 of step1-2 done
    monkeypatch.setattr(rb, "prepare_engine", lambda e, dry: None)
    assert rb.main(["--engine", "egsnrc", "--levels", "2", "--out", str(out), "--dry-run", "--seed", "9000"]) == 0
    txt = capsys.readouterr().out
    assert "step1-1: holds tables, skipped" in txt
    run_lines = [ln for ln in txt.splitlines() if "run_egsnrc.py" in ln]
    assert len(run_lines) == 1 and "--mats 2,3" in run_lines[0] and "--step-cm 0.025" in run_lines[0]
    assert "--seed 9000" in run_lines[0] and "thermal_watch.py" in run_lines[0]
    assert not (out / "egsnrc" / "step1-2" / "sim.json").exists()          # a dry run writes nothing


def test_force_takes_sets_with_tables(tmp_path, monkeypatch, capsys):
    _sim_in(tmp_path, monkeypatch)
    out = tmp_path / "out"
    (out / "fluka" / "step1-1").mkdir(parents=True)
    (out / "fluka" / "step1-1" / "ann_LUT.raw").touch()
    monkeypatch.setattr(rb, "prepare_engine", lambda e, dry: None)
    rb.main(["--engine", "fluka", "--levels", "1", "--out", str(out), "--dry-run", "--force", "--no-thermal"])
    txt = capsys.readouterr().out
    assert "run_fluka_luts.py" in txt and "thermal_watch" not in txt and "build_luts.py" in txt


def test_runner_settings(tmp_path):
    class A:
        nhist, seed, jobs, no_thermal = 1000, 7000, 4, False
    cmds, env = rb.runner_commands("geant4", tmp_path, 0.0125, [1, 2, 5], A)
    assert env["MAX_STEP_UM"] == "1" and env["CUT_UM"] == "1" and env["CUT_FLOOR_EV"] == "1000" and env["SEED"] == "7000"
    assert cmds[0][cmds[0].index("--step") + 1] == "0.0125" and cmds[0][cmds[0].index("--mats") + 1] == "1,2,5"
    cmds, env = rb.runner_commands("penelope", tmp_path, 0.05, [1, 2, 3, 7], A)
    assert len(cmds) == 1 and cmds[0][cmds[0].index("--mats") + 1] == "1-3,7"
    assert env == {"NHIST": "1000", "SEED": "7000"}


def test_jobs_cap(tmp_path, monkeypatch):
    _sim_in(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="overheats"):
        rb.main(["--engine", "penelope", "--jobs", "16", "--dry-run", "--out", str(tmp_path)])
