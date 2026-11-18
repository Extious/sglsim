from __future__ import annotations

import argparse
import time

from .common import paint, request_json, wait_for_server


def _select_rank(base_url: str, requested_rank: int | None) -> tuple[int, str]:
    if requested_rank is not None:
        return requested_rank, "command line"
    while True:
        status = request_json(base_url, "/aegis/status", timeout=5.0)
        hints = status.get("recovery_hints", [])
        if hints:
            hint = hints[0]
            return int(hint["source_dp_rank"]), str(hint.get("backup_key", ""))
        print("  尚无 ACK/Recovery Hint，等待 workflow 的第一个请求 ...", flush=True)
        time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject one independent DP failure")
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--mode", choices=("soft", "hard"), default="soft")
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--hold-seconds", type=float, default=10.0)
    args = parser.parse_args()

    wait_for_server(args.base_url)
    print(paint("Failure injector 已连接；等待可恢复的请求", "bold"), flush=True)
    rank, source = _select_rank(args.base_url, args.rank)
    print(f"  选择 DP rank {rank}（依据: {source}）", flush=True)
    time.sleep(max(args.delay_seconds, 0.0))

    fail_path = "/simulate/fail_dp" if args.mode == "soft" else "/simulate/kill_dp"
    result = request_json(
        args.base_url,
        fail_path,
        payload={"dp_rank": rank},
    )
    accepted = result.get("accepted") if args.mode == "soft" else result.get("accepted", True)
    if not accepted:
        raise RuntimeError(f"failure injection rejected: {result}")
    print(paint(f"{args.mode.upper()} FAILURE ON  DP rank {rank}", "red"), flush=True)

    deadline = time.monotonic() + max(args.hold_seconds, 0.0)
    last_sequence = 0
    while time.monotonic() < deadline:
        status = request_json(
            args.base_url,
            f"/aegis/status?after_sequence={last_sequence}",
            timeout=5.0,
        )
        last_sequence = int(status.get("latest_sequence", last_sequence))
        for event in status.get("events", []):
            if event.get("kind") != "recovery_reroute":
                continue
            print(
                paint(
                    "  observed retry: "
                    f"rank {event.get('failed_dp_rank')} -> {event.get('retry_dp_rank')}, "
                    f"action={event.get('action')}, hint={event.get('used_hint')}",
                    "yellow",
                ),
                flush=True,
            )
        remaining = max(int(deadline - time.monotonic()), 0)
        print(f"  故障保持中 ... {remaining:2d}s", end="\r", flush=True)
        time.sleep(0.5)

    recover_path = (
        "/simulate/recover_dp" if args.mode == "soft" else "/simulate/restart_dp"
    )
    if args.mode == "soft":
        print("  等待 rank 到达安全边界并清空 GPU/L1 KV ...", flush=True)
    recovered = request_json(
        args.base_url,
        recover_path,
        payload={"dp_rank": rank},
        timeout=1900.0 if args.mode == "soft" else 120.0,
    )
    print(" " * 60, end="\r")
    print(
        paint(f"RECOVERED   DP rank {rank}", "green"),
        "active failures="
        f"{recovered.get('soft_failed_ranks', []) + recovered.get('hard_failed_ranks', [])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
