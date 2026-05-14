from __future__ import annotations

from sglsim.srt.server_args import ServerArgs


def main(argv: list[str] | None = None) -> None:
    args = ServerArgs.from_cli_args(argv)
    print(
        f"SGLSim simulator server ready on {args.host}:{args.port}\n"
        f"mode={args.simulator_mode}\n"
        f"config={args.sim_config_path}\n"
        f"output_dir={args.simulator_output_dir}"
    )


if __name__ == "__main__":
    main()
