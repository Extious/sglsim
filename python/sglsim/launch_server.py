from __future__ import annotations

from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulator.config import SimulatorSettings, set_active_settings


def main(argv: list[str] | None = None) -> None:
    args = ServerArgs.from_cli_args(argv)
    settings = SimulatorSettings.from_server_args(args)
    set_active_settings(settings)
    print(
        f"SGLSim simulator server ready on {args.host}:{args.port}\n"
        f"mode={settings.mode}\n"
        f"config={settings.config_path}\n"
        f"output_dir={settings.output_dir}"
    )


if __name__ == "__main__":
    main()
