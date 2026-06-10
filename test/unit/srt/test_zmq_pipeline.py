from sglsim.srt.entrypoints.engine import Engine
from sglsim.srt.managers.io_struct import GenerateReqInput
from sglsim.srt.managers.scheduler import REQUEST_STATS
from sglsim.srt.server_args import ServerArgs

from test.unit.srt.engine_helpers import drive_engine


def test_engine_streaming_single_request_pipeline():
    engine = Engine(ServerArgs())
    drive_engine(
        engine,
        [
            GenerateReqInput(
                rid="r0",
                input_len=4,
                max_new_tokens=2,
                created_time=0.0,
            )
        ],
    )
    assert "r0" in REQUEST_STATS
    assert REQUEST_STATS["r0"].status == "completed"
    assert len(REQUEST_STATS["r0"].gen_token_latencies) >= 1
