from sglsim.srt.entrypoints.engine import Engine
from sglsim.srt.managers.io_struct import GenerateReqInput
from sglsim.srt.server_args import ServerArgs

from test.unit.srt.engine_helpers import drive_engine


def test_engine_completes_under_wall_clock_simulation():
    engine = Engine(ServerArgs(model_path="mock-model"))
    result = drive_engine(
        engine,
        [
            GenerateReqInput(
                rid="r0",
                input_len=8,
                max_new_tokens=2,
                created_time=0.0,
            )
        ],
    )
    engine.shutdown()
    assert result.request_stats["r0"].status == "completed"
    assert result.request_stats["r0"].generated_tokens == 2
