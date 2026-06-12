from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
ONLINE_ROOT = REPO_ROOT / "benchmark" / "online"


def _active_online_files() -> list[Path]:
    if not ONLINE_ROOT.exists():
        return []
    ignored_parts = {"__pycache__", "results", "logs", "docs"}
    return [
        path
        for path in ONLINE_ROOT.rglob("*")
        if path.is_file()
        and not any(part in ignored_parts for part in path.relative_to(ONLINE_ROOT).parts)
    ]


def test_online_benchmark_replaces_multi_agent_directory():
    assert ONLINE_ROOT.is_dir()
    assert not (REPO_ROOT / "benchmark" / "multi_agent").exists()


def test_online_path_config_loads_env_yml():
    src_root = ONLINE_ROOT / "src"
    sys.path.insert(0, str(src_root))
    from paths import PathConfig

    cfg = PathConfig.from_env_yml()

    assert cfg.repo_root == REPO_ROOT
    assert cfg.online_root == ONLINE_ROOT
    assert cfg.venv == REPO_ROOT / ".venv"


def test_online_remote_backup_a100_fault_delay_defaults_to_immediate():
    src_root = ONLINE_ROOT / "src"
    sys.path.insert(0, str(src_root))
    from utils import build_experiment_config

    cfg = build_experiment_config(
        [
            "--config",
            "remote_backup-fixed-a100",
            "--output-dir",
            "/tmp/sglsim-test-output",
        ]
    )

    assert cfg.fault_injection.delay == 0


def test_online_remote_backup_a100_config_enables_simulator_predictor():
    src_root = ONLINE_ROOT / "src"
    sys.path.insert(0, str(src_root))
    from utils import build_experiment_config

    cfg = build_experiment_config(
        [
            "--config",
            "remote_backup-fixed-a100",
            "--output-dir",
            "/tmp/sglsim-test-output",
        ]
    )

    assert cfg.server.sim_config_path
    assert (REPO_ROOT / cfg.server.sim_config_path).is_file()
    assert cfg.server.simulator_mode == "blocking"


def test_online_server_config_forwards_simulator_flags():
    src_root = ONLINE_ROOT / "src"
    sys.path.insert(0, str(src_root))
    from server.config import ServerConfig

    cfg = ServerConfig(
        sim_config_path="benchmark/online/src/config/simulator/a100-qwen3-8b.json",
        simulator_mode="blocking",
        simulator_num_warmup=2,
    )

    flags = cfg.build_deploy_flags(
        model_path="Qwen/Qwen3-8B",
        server_port=28000,
        kv_backup="remote_backup",
        venv_dir=str(REPO_ROOT / ".venv"),
    )

    assert "--sim-config-path" in flags
    assert "benchmark/online/src/config/simulator/a100-qwen3-8b.json" in flags
    assert "--simulator-mode" in flags
    assert "blocking" in flags
    assert "--simulator-num-warmup" in flags
    assert "2" in flags


def test_online_deploy_server_script_has_src_import_path():
    proc = subprocess.run(
        [
            sys.executable,
            str(ONLINE_ROOT / "src" / "server" / "deploy_server.py"),
            "--help",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "SGLSim server deployer" in proc.stdout


def test_online_deploy_server_accepts_absolute_config_directory():
    src_root = ONLINE_ROOT / "src"
    sys.path.insert(0, str(src_root))
    from server.deploy_server import load_config

    cfg = load_config([
        str(REPO_ROOT / "benchmark/fault_tolerance/multi_agent/config/aegiskv_coding")
    ])

    assert cfg["kv_backup"] == "gpu_backup"
    assert cfg["peer_backup_lag_tokens"] == 8


def test_online_fixed_prompt_builder_does_not_require_transformers(monkeypatch):
    import builtins

    src_root = ONLINE_ROOT / "src"
    sys.path.insert(0, str(src_root))
    from client.fixed_worker import PromptBuilder

    real_import = builtins.__import__

    def reject_transformers(name, *args, **kwargs):
        if name == "transformers":
            raise ModuleNotFoundError("No module named 'transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_transformers)

    prompt = PromptBuilder("Qwen/Qwen3-8B", 17, 1).build(0)

    assert len(prompt.split()) == 17


def test_online_fixed_prompt_builder_constructs_large_fallback_prompt_in_one_pass(monkeypatch):
    src_root = ONLINE_ROOT / "src"
    sys.path.insert(0, str(src_root))
    from client.fixed_worker import PromptBuilder

    class CountingTokenizer:
        def __init__(self):
            self.encode_calls = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            self.encode_calls += 1
            if self.encode_calls > 1:
                raise RuntimeError("prompt builder should not repeatedly tokenize")
            return list(range(len(text.split())))

        def decode(self, token_ids, **kwargs):
            del kwargs
            return " ".join(f"tok{int(token_id):05d}" for token_id in token_ids)

    tokenizer = CountingTokenizer()
    monkeypatch.setattr(PromptBuilder, "_load_tokenizer", staticmethod(lambda model_path: tokenizer))

    prompt = PromptBuilder("Qwen/Qwen3-8B", 32000, 1).build(0)

    assert tokenizer.encode_calls == 1
    assert len(prompt.split()) == 32000


def test_online_plot_generation_skips_when_matplotlib_is_missing(monkeypatch):
    import importlib.util

    src_root = ONLINE_ROOT / "src"
    sys.path.insert(0, str(src_root))
    from main import _plot_dependencies_available

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "matplotlib":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    assert not _plot_dependencies_available("backup_cache_hits.py")


def test_online_plot_generation_writes_fallback_png_without_matplotlib(monkeypatch, tmp_path):
    import importlib.util

    src_root = ONLINE_ROOT / "src"
    sys.path.insert(0, str(src_root))
    from main import ExperimentPipeline
    from client.config import FixedClientConfig
    from failure.config import FaultInjectionConfig
    from server.config import ServerConfig

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "matplotlib":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    out = tmp_path / "out"
    pipeline = ExperimentPipeline(
        server_cfg=ServerConfig(),
        fi_cfg=FaultInjectionConfig(),
        fixed_cfg=FixedClientConfig(),
        kv_backup="remote_backup",
        log_dir=tmp_path / "logs",
        output_dir=out,
        timestamp="run",
        venv=REPO_ROOT / ".venv",
        deploy_config_name="remote_backup-fixed-a100",
    )
    cache_csv = pipeline.output_dir / "cache_hits.csv"
    cache_csv.write_text(
        "job_id,task_label,agent_role,rid,l1_match,l2_match,remote_match,remote_prefetch,"
        "reused_device,reused_host,reused_storage,is_failover_retried,"
        "pre_failover_output_tokens,pre_failover_backed_up_tokens\n"
        "1,fixed_request,fixed_client,r1,0,0,8,8,0,0,8,True,2,8\n",
        encoding="utf-8",
    )

    pipeline._generate_plots(pipeline.output_dir)

    figure = pipeline.output_dir / "figures" / "backup_cache_hit_profile.png"
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_online_main_assigns_unique_run_id(monkeypatch):
    src_root = ONLINE_ROOT / "src"
    sys.path.insert(0, str(src_root))
    import main as online_main

    monkeypatch.delenv("SGLSIM_RUN_ID", raising=False)

    run_id = online_main._ensure_run_id("20260518_141200")

    assert run_id == "20260518_141200"
    assert online_main.os.environ["SGLSIM_RUN_ID"] == "20260518_141200"


def test_online_benchmark_has_fixed_and_multi_agent_clients_without_slurm_or_crewai_runtime():
    active_names = {path.name for path in _active_online_files()}

    assert "fixed_worker.py" in active_names
    assert "multi_agent_worker.py" in active_names
    assert "crewai_worker.py" not in active_names
    assert "crewai_core.py" not in active_names
    assert "crewai_config.json" not in active_names
    assert not list(ONLINE_ROOT.rglob("*.slurm"))


def test_online_config_accepts_aegiskv_multi_agent_gpu_backup():
    src_root = ONLINE_ROOT / "src"
    sys.path.insert(0, str(src_root))
    from utils import build_experiment_config

    cfg = build_experiment_config(
        [
            "--config",
            str(REPO_ROOT / "benchmark/fault_tolerance/multi_agent/config/aegiskv_coding"),
            "--output-dir",
            "/tmp/sglsim-test-output",
        ]
    )

    assert cfg.client_mode == "aegiskv_workflow"
    assert cfg.kv_backup == "gpu_backup"
    assert cfg.fault_injection.inject_match_task_label == "S21"


def test_online_benchmark_has_no_upstream_sglang_runtime_references():
    forbidden = [
        "sglang.launch_server",
        "server_runtime=sglang",
        "SGLANG_REPO_ROOT",
        "SGLANG_VENV",
        "sglang.srt.mem_cache.storage.remote_backup.launch_remote_backup_server",
        "__MULTI_AGENT_CONTROL__",
    ]
    hits: list[str] = []
    for path in _active_online_files():
        if path.suffix not in {".py", ".json", ".yml", ".yaml", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in forbidden:
            if needle in text:
                hits.append(f"{path.relative_to(REPO_ROOT)}: {needle}")

    assert hits == []
