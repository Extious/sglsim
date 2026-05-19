from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _configure_native_env(server_args) -> None:
    config_path = getattr(server_args, "sim_config_path", None)
    if config_path:
        os.environ["SGLSIM_SIMULATOR_CONFIG_PATH"] = str(config_path)

    output_dir = Path(
        str(getattr(server_args, "simulator_output_dir", "") or "/tmp/sglsim/output")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "SGLANG_FAILOVER_EVENTS_FILE",
        str(output_dir / "failover_events.jsonl"),
    )


def main(argv: list[str] | None = None) -> None:
    from sglsim.srt.server_args import ServerArgs
    from sglsim.srt.simulator.native.bootstrap import (
        install_native_simulation,
        run_native_scheduler_process,
    )
    from sglsim.srt.utils import kill_process_tree

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    raw_args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    # ServerArgs.__post_init__ may access ModelConfig. Install this patch before
    # constructing ServerArgs so simulator-only model ids never hit HF/network.
    from sglsim.srt.simulator.native import model_config_patch

    model_config_patch.install(
        type(
            "EarlyNativeConfig",
            (),
            {
                "enable_native_runtime": True,
                "sim_config_path": getattr(raw_args, "sim_config_path", None) or "",
            },
        )()
    )
    server_args = ServerArgs.from_cli_args(raw_args)
    if os.getenv("SGLSIM_NATIVE_TRACE", "0") == "1":
        print("[SGLSIM_NATIVE_TRACE] ServerArgs ready", file=sys.stderr, flush=True)

    _configure_native_env(server_args)
    install_native_simulation(server_args)
    if os.getenv("SGLSIM_NATIVE_TRACE", "0") == "1":
        print("[SGLSIM_NATIVE_TRACE] native patches installed", file=sys.stderr, flush=True)
    from sglsim.srt.entrypoints.http_server import launch_server

    try:
        launch_server(
            server_args,
            run_scheduler_process_func=run_native_scheduler_process,
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
