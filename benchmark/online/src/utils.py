from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple, Optional

from client.config import FixedClientConfig
from failure.config import FaultInjectionConfig
from paths import PathConfig
from server.config import ServerConfig


class ExperimentConfig(NamedTuple):
    server: ServerConfig
    fault_injection: FaultInjectionConfig
    fixed: FixedClientConfig
    kv_backup: str
    log_dir: Path
    output_dir: Path
    venv: Path
    deploy_config_name: str


def _resolve_venv(paths: PathConfig) -> Path:
    override = os.environ.get("SGLSIM_VENV", "").strip()
    if override:
        return Path(override).resolve()
    return paths.venv


def build_experiment_config(
    argv: Optional[list[str]] = None,
    paths: Optional[PathConfig] = None,
) -> ExperimentConfig:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Fixed online SGLSim benchmark")
    parser.add_argument("--config", required=True, help="Config directory name or absolute path")
    parser.add_argument("--dp-size", type=int, default=0)
    parser.add_argument("--pp-size", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=0)
    parser.add_argument("--nnodes", type=int, default=0)
    parser.add_argument("--hicache-size", type=int, default=0)
    parser.add_argument("--quantization", default="")
    parser.add_argument("--load-balance-method", default="")
    parser.add_argument(
        "--kv-backup",
        default="none",
        choices=["none", "host", "device", "remote_backup"],
    )
    parser.add_argument("--fixed-input-len", type=int, default=0)
    parser.add_argument("--fixed-output-len", type=int, default=0)
    parser.add_argument("--fixed-num-requests", type=int, default=0)
    parser.add_argument("--fixed-app-workers", type=int, default=0)
    parser.add_argument("--fixed-seed", type=int, default=-1)
    parser.add_argument("--inject-after-job", default="")
    parser.add_argument("--inject-after-task", default="")
    parser.add_argument("--timeline-after-start-s", default="")
    parser.add_argument("--inject-delay", type=int, default=-1)
    parser.add_argument("--fault-dp-rank", default="")
    parser.add_argument("--fault-pp-rank", default="")
    parser.add_argument("--fault-tp-rank", default="")
    parser.add_argument("--recovery-delay", default="")
    parser.add_argument("--recover-after-job", default="")
    parser.add_argument("--recover-after-task", default="")
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args(argv)

    if paths is None:
        paths = PathConfig.from_env_yml(log_dir=args.log_dir, output_dir=args.output_dir)

    cfg_dir = Path(args.config)
    if not cfg_dir.is_absolute():
        cfg_dir = paths.config_dir / cfg_dir.name

    def load_json(name: str) -> dict:
        path = cfg_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    top = load_json("server.json")
    srv_json = dict(top.get("server", {}))
    srv_json["hicache_storage"] = top.get("hicache_storage", {})
    srv_json["simulator"] = top.get("simulator", {})
    fi_json = load_json("failure.json")
    client_json = load_json("client.json")

    srv = ServerConfig.from_dict(srv_json)
    if args.dp_size > 0:
        srv.dp_size = args.dp_size
    if args.pp_size > 0:
        srv.pp_size = args.pp_size
    if args.tp_size > 0:
        srv.tp_size = args.tp_size
    if args.nnodes > 0:
        srv.nnodes = args.nnodes
    if args.hicache_size > 0:
        srv.hicache_size_gb = args.hicache_size
    if args.quantization:
        srv.quantization = args.quantization
    if args.load_balance_method:
        srv.load_balance_method = args.load_balance_method.strip()

    fi = FaultInjectionConfig.from_dict(fi_json)
    if args.inject_after_job:
        fi.after_job = args.inject_after_job
    if args.inject_after_task:
        fi.after_task = args.inject_after_task
    if args.timeline_after_start_s:
        fi.timeline_after_start_s = args.timeline_after_start_s
    if args.inject_delay >= 0:
        fi.delay = args.inject_delay
    if args.fault_dp_rank:
        fi.dp_rank = args.fault_dp_rank
    if args.fault_pp_rank:
        fi.pp_rank = args.fault_pp_rank
    if args.fault_tp_rank:
        fi.tp_rank = args.fault_tp_rank
    fi.method = "api"
    if args.recovery_delay:
        fi.recovery_delay = args.recovery_delay
    if args.recover_after_job:
        fi.recover_after_job = args.recover_after_job
    if args.recover_after_task:
        fi.recover_after_task = args.recover_after_task

    fixed = FixedClientConfig.from_dict(client_json)
    if args.fixed_input_len > 0:
        fixed.input_len = args.fixed_input_len
    if args.fixed_output_len > 0:
        fixed.output_len = args.fixed_output_len
    if args.fixed_num_requests > 0:
        fixed.num_requests = args.fixed_num_requests
    if args.fixed_app_workers > 0:
        fixed.app_workers = args.fixed_app_workers
    if args.fixed_seed >= 0:
        fixed.seed = args.fixed_seed

    kv_backup = top.get("kv_backup", args.kv_backup)
    if isinstance(kv_backup, str) and kv_backup not in {"none", "host", "device", "remote_backup"}:
        kv_backup = args.kv_backup

    return ExperimentConfig(
        server=srv,
        fault_injection=fi,
        fixed=fixed,
        kv_backup=str(kv_backup),
        log_dir=paths.log_dir,
        output_dir=paths.output_dir,
        venv=_resolve_venv(paths),
        deploy_config_name=str(args.config),
    )
