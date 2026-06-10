from sglsim.srt.simulation.manager.scheduler_config import SchedulerSimConfig
from sglsim.srt.time_predictor.aiconfigurator import attention_dp_size_for_scheduler


def test_dense_replica_dp_keeps_attention_dp_size_one():
    sched = SchedulerSimConfig(tp_size=1, dp_size=2, pp_size=1)
    assert attention_dp_size_for_scheduler(sched) == 1


def test_attention_dp_size_one_when_tp_greater_than_dp():
    sched = SchedulerSimConfig(tp_size=4, dp_size=2, pp_size=1)
    assert attention_dp_size_for_scheduler(sched) == 1
