from __future__ import annotations

from types import SimpleNamespace


class _Worker:
    def __init__(self) -> None:
        self.sent = []

    def send_pyobj(self, obj) -> None:
        self.sent.append(obj)


def _controller(*, failed=(), total_tokens=(100, 0), total_requests=(0, 0)):
    from sglsim.srt.managers.data_parallel_controller import (
        DPBudget,
        DataParallelController,
    )

    controller = DataParallelController.__new__(DataParallelController)
    controller.failed_dp_ranks = set(failed)
    controller.workers = [_Worker(), _Worker()]
    controller.status = [True, True]
    controller.round_robin_counter = 0
    controller.dp_budget = DPBudget(2)
    controller.dp_budget.total_tokens = list(total_tokens)
    controller.dp_budget.total_requests = list(total_requests)
    return controller


def test_total_tokens_dispatch_avoids_failover_source_rank():
    controller = _controller(failed=(), total_tokens=(100, 0))
    req = SimpleNamespace(routed_dp_rank=None, failed_dp_rank=1)

    controller.total_tokens_scheduler(req)

    assert controller.workers[0].sent == [req]
    assert controller.workers[1].sent == []


def test_direct_route_to_failed_rank_is_rebalanced():
    controller = _controller(failed={1}, total_tokens=(100, 0))
    req = SimpleNamespace(routed_dp_rank=1, failed_dp_rank=None)

    controller.total_tokens_scheduler(req)

    assert req.routed_dp_rank is None
    assert controller.workers[0].sent == [req]
    assert controller.workers[1].sent == []


def test_dispatch_does_not_fall_back_to_excluded_rank_when_all_excluded():
    from sglsim.srt.managers.data_parallel_controller import (
        DPBudget,
        LoadBalanceMethod,
    )

    budget = DPBudget(2)

    assert budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS, exclude={0, 1}) is None
