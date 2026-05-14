import types
import unittest

from sglang.srt.simulator.scheduler import SchedulerSimulatorRuntime


def make_req(rid, created_time, total_request=2, input_len=4, output_len=3):
    sampling_params = types.SimpleNamespace(
        max_new_tokens=output_len,
        custom_params={
            "simulation": {
                "created_time": created_time,
                "total_request": total_request,
            }
        },
    )
    return types.SimpleNamespace(
        rid=rid,
        input_ids=list(range(input_len)),
        sampling_params=sampling_params,
    )


class TestSchedulerSimulatorRuntime(unittest.TestCase):
    def test_offline_buffers_until_all_requests_arrive(self):
        runtime = SchedulerSimulatorRuntime(
            mode="offline",
            output_dir="/tmp/sim",
            num_warmup=0,
            overlap_schedule=False,
            predictor=None,
        )
        r1 = make_req("r1", 0.0)
        r2 = make_req("r2", 2.0)

        self.assertEqual(runtime.filter_recv_requests([r1]), [])
        arrived = runtime.filter_recv_requests([r2])

        self.assertEqual([req.rid for req in arrived], ["r1"])
        runtime.state.set_global_clock(2.0)
        arrived = runtime.filter_recv_requests([])
        self.assertEqual([req.rid for req in arrived], ["r2"])

    def test_blocking_returns_requests_immediately_and_records_stats(self):
        runtime = SchedulerSimulatorRuntime(
            mode="blocking",
            output_dir="/tmp/sim",
            num_warmup=0,
            overlap_schedule=False,
            predictor=None,
        )
        req = make_req("r1", 0.0, total_request=1)

        arrived = runtime.filter_recv_requests([req])

        self.assertEqual(arrived, [req])
        self.assertEqual(runtime.request_stats["r1"].input_length, 4)
        self.assertEqual(runtime.request_stats["r1"].output_length, 3)


if __name__ == "__main__":
    unittest.main()
