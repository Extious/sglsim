from __future__ import annotations

from unittest.mock import patch

from sglsim.srt.entrypoints.engine import launch_subprocesses
from sglsim.srt.server_args import ServerArgs


def test_launch_subprocesses_spawn_order():
    order: list[str] = []

    def fake_scheduler(*args, **kwargs):
        order.append("scheduler")
        return ([], [], None, [])

    def fake_detokenizer(*args, **kwargs):
        order.append("detokenizer")
        ready = args[3]
        ready.set()

    class FakeTokenizerManager:
        def __init__(self, *args, **kwargs) -> None:
            order.append("tokenizer_manager")

    with (
        patch(
            "sglsim.srt.entrypoints.engine.launch_scheduler_processes",
            side_effect=fake_scheduler,
        ),
        patch(
            "sglsim.srt.entrypoints.engine.mp.Process",
            side_effect=lambda target, args, daemon: type(
                "P",
                (),
                {
                    "start": lambda self: target(*args),
                    "is_alive": lambda self: False,
                    "join": lambda self, timeout=None: None,
                },
            )(),
        ),
        patch(
            "sglsim.srt.entrypoints.engine.run_detokenizer_process",
            side_effect=fake_detokenizer,
        ),
        patch(
            "sglsim.srt.entrypoints.engine.TokenizerManager",
            FakeTokenizerManager,
        ),
        patch("sglsim.srt.entrypoints.engine.DataParallelController"),
    ):
        launch_subprocesses(ServerArgs(model_path="mock-model"))

    assert order == ["scheduler", "detokenizer", "tokenizer_manager"]
