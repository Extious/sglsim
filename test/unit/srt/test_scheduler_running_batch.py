from sglsim.srt.entrypoints.engine import Engine
from sglsim.srt.managers.io_struct import GenerateReqInput
from sglsim.srt.server_args import ServerArgs

from test.unit.srt.engine_helpers import drive_engine


def test_running_batch_merge_and_complete():
    engine = Engine(ServerArgs(dp_size=1))
    result = drive_engine(
        engine,
        [
            GenerateReqInput(
                rid="r0",
                input_len=8,
                max_new_tokens=2,
                simulation={"created_time": 0.0},
            )
        ],
    )
    assert result.request_stats["r0"].status == "completed"
    engine.shutdown()
