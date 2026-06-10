from __future__ import annotations

from sglsim.srt.entrypoints.engine import Engine, EngineResult
from sglsim.srt.managers.io_struct import GenerateReqInput


def drive_engine(engine: Engine, reqs: list[GenerateReqInput]) -> EngineResult:
    engine.start_background_scheduler()
    for req in reqs:
        engine.submit(req)
    return engine.run_until_idle()
