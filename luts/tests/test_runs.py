"""Tests for runs (run.json, run resolution, labels) and the statistics plan
shared by luts-check and luts-compare."""
import json
import shutil

import numpy as np
import pytest

import build_tables
import extract_stats
import runs
from test_build_tables import synthetic_tree  # noqa: F401  (fixture)


def _build(tree, out, n_full=100):
    build_tables.build_engine("penelope", tree / "raw", out, n_theta=8, n_u=4, n_full=n_full, k_max=1, workers=1)
    return out


class TestRunInfo:
    def test_build_writes_run_json(self, synthetic_tree):
        out = _build(synthetic_tree, synthetic_tree / "a")
        info = json.loads((out / "run.json").read_text())
        assert info["engine"] == "penelope" and info["materials"] == [1, 2, 10]
        assert (info["n_theta"], info["n_u"], info["n_full"], info["k_max"]) == (8, 4, 100, 1)
        assert '"materials": [1, 2, 10]' in (out / "run.json").read_text()      # one line

    def test_resolve_from_run_json(self, synthetic_tree):
        out = _build(synthetic_tree, synthetic_tree / "a")
        r = runs.resolve(out)
        assert r.raw_dir == (synthetic_tree / "raw").resolve() and r.engine == "penelope"
        assert r.materials == [1, 2, 10] and r.slot(10) == 2 and r.has_luts()
        assert r.params == {"step_cm": 0.05, "n_theta": 8, "n_u": 4, "n_full": 100, "k_max": 1,
                            "n_z": 32, "n_r": 16, "n_full_stop": 500}

    def test_resolve_without_run_json(self, synthetic_tree):
        r = runs.resolve(synthetic_tree)                      # <dir>/raw, engine unknown
        assert r.raw_dir == (synthetic_tree / "raw").resolve() and r.materials == [1, 2, 10]
        assert not r.has_luts()

    def test_unknown_run(self, tmp_path):
        with pytest.raises(SystemExit, match="not found"):
            runs.resolve(tmp_path / "nope")

    def test_pair_labels(self, synthetic_tree):
        a = runs.resolve(_build(synthetic_tree, synthetic_tree / "a", n_full=100))
        b = runs.resolve(_build(synthetic_tree, synthetic_tree / "b", n_full=50))
        assert runs.pair_labels(a, b) == ("PENELOPE N_FULL=100", "PENELOPE N_FULL=50")
        assert a.raw_key == b.raw_key and a.key != b.key


class TestStatsPlan:
    def test_shared_raw_read_once_then_cached(self, synthetic_tree, tmp_path):
        a = runs.resolve(_build(synthetic_tree, synthetic_tree / "a", n_full=100))
        b = runs.resolve(_build(synthetic_tree, synthetic_tree / "b", n_full=50))
        cache = tmp_path / "cache"
        jobs = extract_stats.plan([a, b], "all", cache=cache)
        assert len(jobs) == 3                                  # one per material, not per run
        assert all(do_raw and len(lut_runs) == 2 for _, lut_runs, _, _, do_raw in jobs)
        for j in jobs:
            extract_stats.work(j)
        assert extract_stats.plan([a, b], "all", cache=cache) == []          # fresh: nothing to do
        assert len(extract_stats.plan([a, b], "all", cache=cache, force=True)) == 3
        z = np.load(a.raw_cache(2, cache))
        assert "valid" not in z.files and "mth" in z.files    # raw part holds no LUT quantity
        assert "valid" in np.load(b.lut_cache(2, cache)).files

    def test_raw_scope_needs_no_luts(self, synthetic_tree, tmp_path):
        r = runs.resolve(synthetic_tree)                       # raw only, no LUTs
        jobs = extract_stats.plan([r], "raw", cache=tmp_path)
        assert len(jobs) == 3 and all(not lut_runs for _, lut_runs, _, _, _ in jobs)
        with pytest.raises(SystemExit, match="no LUTs"):
            extract_stats.plan([r], "lut", cache=tmp_path)


@pytest.mark.skipif(shutil.which("zstd") is None, reason="compressed raw files are read through the zstd command")
class TestCheckSamples:
    """RUN = TABLES@RAW[+RAW]: tables checked against independent raw data; bundles stand for their longest set."""

    def _two_samples(self, tree):
        import shutil
        import subprocess
        dense, sparse = tree / "dense", tree / "sparse"
        dense.mkdir(); sparse.mkdir()
        shutil.copy(tree / "raw" / "mat2.dat", dense / "mat2.dat")             # a dense sample of one material
        for m in (1, 2, 10):                                                     # a sparse sample of all, zstd
            subprocess.run(["zstd", "-q", "-o", str(sparse / f"mat{m}.dat.zst"), str(tree / "raw" / f"mat{m}.dat")],
                           check=True)
        return dense, sparse

    def test_first_raw_dir_with_the_material_wins(self, synthetic_tree):
        out = _build(synthetic_tree, synthetic_tree / "a")
        dense, sparse = self._two_samples(synthetic_tree)
        r = runs.resolve(f"{out}@{dense}+{sparse}")
        assert r.materials == [1, 2, 10] and r.raw_dirs == [dense.resolve(), sparse.resolve()]
        assert r.raw_file(2) == dense.resolve() / "mat2.dat"
        assert r.raw_file(10) == sparse.resolve() / "mat10.dat.zst"
        assert r.raw_cache(2).parent.name != r.raw_cache(10).parent.name     # cached per raw directory
        assert r.lut_cache(2) != runs.resolve(out).lut_cache(2)              # table stats depend on the sample

    def test_materials_without_raw_left_out(self, synthetic_tree):
        out = _build(synthetic_tree, synthetic_tree / "a")
        dense, _ = self._two_samples(synthetic_tree)
        r = runs.resolve(f"{out}@{dense}")
        assert r.materials == [2] and r.all_materials == [1, 2, 10] and r.slot(2) == 1

    def test_stats_from_zstd_and_mixed_samples(self, synthetic_tree, tmp_path):
        out = _build(synthetic_tree, synthetic_tree / "a")
        dense, sparse = self._two_samples(synthetic_tree)
        r = runs.resolve(f"{out}@{dense}+{sparse}")
        jobs = extract_stats.plan([r], "all", cache=tmp_path / "c")
        assert sorted(j[2] for j in jobs) == [1, 2, 10]
        for j in jobs:
            extract_stats.work(j)
        assert extract_stats.plan([r], "all", cache=tmp_path / "c") == []

    def test_bundle_is_its_longest_set(self, synthetic_tree):
        bundle = synthetic_tree / "bundle"
        for name, step in (("step1-2", 0.025), ("step1-1", 0.05)):
            d = _build(synthetic_tree, bundle / name)
            info = json.loads((d / "run.json").read_text()); info["step_cm"] = step
            (d / "run.json").write_text(json.dumps(info))
        r = runs.resolve(f"{bundle}@{synthetic_tree / 'raw'}")
        assert r.dir == (bundle / "step1-1").resolve() and r.params["step_cm"] == 0.05

    def test_missing_raw_explains(self, synthetic_tree, tmp_path):
        out = _build(synthetic_tree, synthetic_tree / "a")
        info = json.loads((out / "run.json").read_text()); info["raw_dir"] = "/nonexistent/raw"
        (out / "run.json").write_text(json.dumps(info))
        with pytest.raises(SystemExit, match="TABLES@RAW"):
            runs.resolve(out)
