from sglsim.srt.managers.data_parallel_controller import DataParallelController
from sglsim.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.utils.network import _QUEUES


def test_round_robin_distributes_to_dp_workers():
    ConfigManager._raw = {
        "scheduler": {"dp_size": 2, "load_balance_method": "round_robin"},
    }
    ConfigManager._scheduler = None
    server_args = ServerArgs(dp_size=2)
    port_args = PortArgs.init_new(server_args)
    _QUEUES.clear()
    controller = DataParallelController(server_args, port_args)

    for i in range(4):
        req = TokenizedGenerateReqInput.from_generate(
            GenerateReqInput(rid=f"r{i}", input_len=2, max_new_tokens=1)
        )
        controller.dispatch(req)

    q0 = _QUEUES[port_args.scheduler_input_ipc_name_for_dp(0)]
    q1 = _QUEUES[port_args.scheduler_input_ipc_name_for_dp(1)]
    assert len(q0) == 2
    assert len(q1) == 2


def test_routing_key_affinity_keeps_same_key_on_one_dp_worker():
    ConfigManager._raw = {
        "scheduler": {
            "dp_size": 4,
            "load_balance_method": "round_robin",
            "enable_routing_key_affinity": True,
        },
    }
    ConfigManager._scheduler = None
    server_args = ServerArgs(dp_size=4)
    port_args = PortArgs.init_new(server_args)
    _QUEUES.clear()
    controller = DataParallelController(server_args, port_args)

    for i in range(3):
        req = TokenizedGenerateReqInput.from_generate(
            GenerateReqInput(
                rid=f"group-a-{i}",
                input_len=2,
                max_new_tokens=1,
                routing_key="group-a",
            )
        )
        controller.dispatch(req)

    target = 0
    for dp_rank in range(4):
        q = _QUEUES[port_args.scheduler_input_ipc_name_for_dp(dp_rank)]
        assert len(q) == (3 if dp_rank == target else 0)
    assert all(
        msg.sampling_params.custom_params["simulation"]["assigned_dp_rank"] == target
        for msg in _QUEUES[port_args.scheduler_input_ipc_name_for_dp(target)]
    )


def test_routing_key_affinity_spreads_many_groups_across_workers():
    ConfigManager._raw = {
        "scheduler": {
            "dp_size": 4,
            "load_balance_method": "round_robin",
            "enable_routing_key_affinity": True,
        },
    }
    ConfigManager._scheduler = None
    server_args = ServerArgs(dp_size=4)
    port_args = PortArgs.init_new(server_args)
    _QUEUES.clear()
    controller = DataParallelController(server_args, port_args)

    for i in range(16):
        req = TokenizedGenerateReqInput.from_generate(
            GenerateReqInput(
                rid=f"group-{i}",
                input_len=2,
                max_new_tokens=1,
                routing_key=f"group-{i}",
            )
        )
        controller.dispatch(req)

    for dp_rank in range(4):
        q = _QUEUES[port_args.scheduler_input_ipc_name_for_dp(dp_rank)]
        assert len(q) == 4
        assert all(
            msg.sampling_params.custom_params["simulation"]["assigned_dp_rank"] == dp_rank
            for msg in q
        )
