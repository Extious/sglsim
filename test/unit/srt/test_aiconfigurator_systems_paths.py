from sglsim.srt.time_predictor.aiconfigurator import (
    _LFS_POINTER_PREFIX,
    _systems_paths_have_lfs_pointers,
    resolve_systems_paths,
)


def test_detects_lfs_pointer_paths(tmp_path):
    systems = tmp_path / "systems"
    gemm = systems / "data/a100_sxm/sglang/0.5.10"
    gemm.mkdir(parents=True)
    (gemm / "gemm_perf.txt").write_text(
        f"{_LFS_POINTER_PREFIX}\noid sha256:abc\nsize 1\n",
        encoding="utf-8",
    )
    assert _systems_paths_have_lfs_pointers([str(systems)])


def test_resolve_systems_paths_uses_env(monkeypatch, tmp_path):
    systems = tmp_path / "systems"
    systems.mkdir()
    monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", str(systems))
    assert resolve_systems_paths({}) == [str(systems)]
