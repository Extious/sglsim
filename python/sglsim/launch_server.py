from __future__ import annotations

import argparse
import sys

from sglsim.srt.entrypoints.http_server import launch_server
from sglsim.srt.server_args import ServerArgs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch SGLSim metadata server")
    ServerArgs.add_cli_args(parser)
    raw = argv if argv is not None else sys.argv[1:]
    launch_server(ServerArgs.from_cli_args(parser.parse_args(raw)))


if __name__ == "__main__":
    main()
