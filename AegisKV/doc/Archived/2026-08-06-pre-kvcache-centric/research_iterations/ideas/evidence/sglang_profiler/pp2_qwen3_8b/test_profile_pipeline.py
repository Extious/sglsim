from __future__ import annotations

import asyncio
import csv
import datetime as dt
import gzip
import inspect
import json
import secrets
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import analyze_run
import build_browser_trace
import profile_load
import sample_resources


class ProfilePipelineTest(unittest.TestCase):
    def test_plain_text_profiler_response_is_not_forced_to_json(self) -> None:
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self) -> bytes:
                return b"Start profiling.\n"

        with mock.patch.object(
            profile_load.urllib.request, "urlopen", return_value=FakeResponse()
        ):
            response = profile_load.post_json(
                "http://127.0.0.1:1/start_profile", {}, 1.0
            )
        self.assertEqual(
            response, {"http_status": 200, "body": "Start profiling."}
        )

    def test_kineto_and_allocator_profiles_use_disjoint_activities(self) -> None:
        payloads: list[dict] = []

        def fake_post_json(
            _url: str, payload: dict, _timeout: float
        ) -> dict:
            payloads.append(payload)
            return {"http_status": 200}

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            profile_load, "post_json", side_effect=fake_post_json
        ):
            asyncio.run(
                profile_load.arm_profile(
                    base_url="http://127.0.0.1:1",
                    output_dir=Path(directory) / "kineto",
                    profile_id="kineto",
                    prefix="kineto",
                    stage="prefill",
                    num_steps=4,
                )
            )
            asyncio.run(
                profile_load.arm_profile(
                    base_url="http://127.0.0.1:1",
                    output_dir=Path(directory) / "memory",
                    profile_id="memory",
                    prefix="memory",
                    stage="prefill",
                    num_steps=4,
                    activities=["MEM"],
                    profile_stages=["prefill", "decode"],
                )
            )
            asyncio.run(
                profile_load.arm_profile(
                    base_url="http://127.0.0.1:1",
                    output_dir=Path(directory) / "combined",
                    profile_id="combined",
                    prefix="combined",
                    stage="combined",
                    num_steps=100,
                    start_step=1,
                    activities=["CPU", "GPU"],
                    profile_stages=[],
                    profile_by_stage=False,
                    with_stack=False,
                )
            )

        self.assertEqual(payloads[0]["activities"], ["CPU", "GPU"])
        self.assertNotIn("MEM", payloads[0]["activities"])
        self.assertEqual(payloads[1]["activities"], ["MEM"])
        self.assertEqual(
            payloads[1]["profile_stages"], ["prefill", "decode"]
        )
        self.assertFalse(payloads[2]["profile_by_stage"])
        self.assertEqual(payloads[2]["start_step"], 1)
        self.assertFalse(payloads[2]["with_stack"])
        self.assertEqual(payloads[2]["num_steps"], 100)

    def test_formal_capture_does_not_use_manual_stop(self) -> None:
        source = inspect.getsource(profile_load.main_async)
        self.assertFalse(hasattr(profile_load, "stop_profile"))
        self.assertNotIn("stop_profile(args.url)", source)
        self.assertIn("start_step=1", source)
        self.assertIn("expected=2", source)

    def test_slurm_script_fails_fast_if_server_exits_before_health(self) -> None:
        script = Path(__file__).with_name(
            "run_qwen3_8b_pp2_profile.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn('kill -0 "${SERVER_PID}"', script)
        self.assertIn('wait "${SERVER_PID}"', script)
        self.assertIn("server_exit_before_ready.log", script)
        self.assertIn("server_health_timeout.log", script)

    def test_trace_readiness_rejects_truncated_gzip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "valid.trace.json.gz"
            with gzip.open(valid, "wb") as handle:
                handle.write(b'{"traceEvents":[]}')
            truncated = Path(directory) / "truncated.trace.json.gz"
            truncated.write_bytes(valid.read_bytes()[:-8])
            self.assertTrue(profile_load.gzip_stream_is_complete(valid))
            self.assertFalse(profile_load.gzip_stream_is_complete(truncated))

    def test_calibration_rejects_four_request_underload(self) -> None:
        calibration = [
            {
                "concurrency": 4,
                "successes": 4,
                "errors": 0,
                "token_throughput_tps": 60.0,
                "queue_time_s": {"p50": 0.004, "p95": 0.008},
                "server_e2e_latency_s": {"p50": 1.0},
            },
            {
                "concurrency": 8,
                "successes": 8,
                "errors": 0,
                "token_throughput_tps": 100.0,
                "queue_time_s": {"p50": 0.003, "p95": 0.020},
                "server_e2e_latency_s": {"p50": 1.0},
            },
        ]
        self.assertEqual(
            profile_load.choose_concurrency(calibration),
            (8, "first_near_saturation_slight_queue_point"),
        )

    def test_calibration_uses_largest_load_when_queue_is_excessive(self) -> None:
        calibration = [
            {
                "concurrency": concurrency,
                "successes": concurrency,
                "errors": 0,
                "token_throughput_tps": float(concurrency),
                "queue_time_s": {"p50": 0.5, "p95": 0.8},
                "server_e2e_latency_s": {"p50": 1.0},
            }
            for concurrency in (4, 8, 12, 16)
        ]
        self.assertEqual(
            profile_load.choose_concurrency(calibration),
            (16, "largest_calibrated_load_fallback"),
        )

    def test_prometheus_queue_histogram_delta_is_rank_aware(self) -> None:
        def snapshot(count: int, total: float, bucket_counts: list[int]) -> str:
            lines = []
            for rank in (0, 1):
                labels = f'engine_type="unified",pp_rank="{rank}",tp_rank="0"'
                lines.append(
                    f"sglang:queue_time_seconds_sum{{{labels}}} "
                    f"{total + rank * 0.001}"
                )
                lines.append(
                    f"sglang:queue_time_seconds_count{{{labels}}} {count}"
                )
                for upper, observed in zip(
                    ("0.001", "0.005", "0.05", "+Inf"), bucket_counts
                ):
                    lines.append(
                        "sglang:queue_time_seconds_bucket"
                        f'{{{labels},le="{upper}"}} {observed}'
                    )
            return "\n".join(lines)

        before = profile_load.parse_queue_histogram(
            snapshot(10, 0.10, [2, 6, 10, 10])
        )
        after = profile_load.parse_queue_histogram(
            snapshot(14, 0.14, [2, 8, 14, 14])
        )
        delta = profile_load.queue_histogram_delta(before, after)
        self.assertEqual(delta["source"], "prometheus_histogram_delta")
        self.assertEqual(delta["count"], 4)
        self.assertAlmostEqual(delta["mean"], 0.01)
        self.assertEqual(delta["p50"], 0.005)
        self.assertEqual(delta["p95"], 0.05)

    def test_offered_rate_selector_requires_near_peak_and_bounded_queue(
        self,
    ) -> None:
        points = []
        for rate, throughput, mean, p95 in (
            (4.0, 8000.0, 0.001, 0.005),
            (5.0, 9500.0, 0.010, 0.050),
            (5.5, 10000.0, 0.200, 1.000),
        ):
            points.append(
                {
                    "offered_rate_rps": rate,
                    "requests": 20,
                    "successes": 20,
                    "errors": 0,
                    "total_retractions": 0,
                    "token_throughput_tps": throughput,
                    "queue_time_s": {
                        "source": "prometheus_histogram_delta",
                        "mean": mean,
                        "p95": p95,
                    },
                    "server_e2e_latency_s": {"p50": 1.0},
                }
            )
        self.assertEqual(
            profile_load.choose_offered_rate(points),
            (5.0, "first_near_saturation_bounded_queue_rate"),
        )

    def test_offered_rate_selector_leaves_queue_drift_margin(self) -> None:
        def point(rate: float, throughput: float, mean: float) -> dict:
            return {
                "offered_rate_rps": rate,
                "requests": 40,
                "successes": 40,
                "errors": 0,
                "total_retractions": 0,
                "token_throughput_tps": throughput,
                "queue_time_s": {
                    "source": "prometheus_histogram_delta",
                    "mean": mean,
                    "p95": 0.050,
                },
                "server_e2e_latency_s": {"p50": 1.0},
            }

        self.assertEqual(
            profile_load.choose_offered_rate(
                [
                    point(4.8, 9500.0, 0.004),
                    point(5.2, 10000.0, 0.008),
                ]
            ),
            (5.2, "first_near_saturation_bounded_queue_rate"),
        )

    def test_analysis_load_gate_uses_formal_steady_phase(self) -> None:
        queue = {
            "source": "prometheus_histogram_delta",
            "count": 40,
            "mean": 0.010,
            "p50": 0.005,
            "p95": 0.050,
        }
        point = {
            "offered_rate_rps": 5.0,
            "requests": 40,
            "successes": 40,
            "errors": 0,
            "total_retractions": 0,
            "token_throughput_tps": 9500.0,
            "queue_time_s": queue,
            "server_e2e_latency_s": {"p50": 1.0},
        }
        workload = {
            "selected_rate_rps": 5.0,
            "rate_calibration": [
                point,
                {
                    **point,
                    "offered_rate_rps": 5.5,
                    "token_throughput_tps": 10000.0,
                },
            ],
            "steady_load": {**point},
        }
        self.assertEqual(
            analyze_run.load_gate(workload),
            (
                True,
                "throughput is near saturation and scheduler queue is visible but bounded",
            ),
        )
        workload["steady_load"] = {
            **point,
            "queue_time_s": {**queue, "p95": 1.0},
        }
        self.assertFalse(analyze_run.load_gate(workload)[0])

    def test_dmon_pcie_values_remain_mb_per_second(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pcie.csv"
            path.write_text(
                "#Date, Time, gpu, rxpci, txpci\n"
                "20260729, 12:30:48, 0, 1024, 2048\n"
                "20260729, 12:30:48, 1, 256, 512\n",
                encoding="utf-8",
            )
            rows = analyze_run.read_pcie(path)
        self.assertEqual(rows[0]["rxpci_mb_s"], 1024.0)
        self.assertEqual(rows[0]["txpci_mb_s"], 2048.0)
        self.assertEqual(rows[1]["rxpci_mb_s"], 256.0)

    def test_compute_app_sampler_preserves_pid_uuid_name_and_memory(self) -> None:
        completed = mock.Mock()
        completed.stdout = (
            "123, GPU-a, /usr/bin/python, 4096\n"
            "456, GPU-a, /usr/local/lib/ollama/llama-server, 8192\n"
        )
        with mock.patch.object(
            sample_resources.subprocess, "run", return_value=completed
        ):
            apps = sample_resources.query_compute_apps()
        self.assertEqual([item["pid"] for item in apps["GPU-a"]], ["123", "456"])
        self.assertEqual(apps["GPU-a"][1]["used_memory_mib"], "8192")
        self.assertIn("ollama", apps["GPU-a"][1]["process_name"])

    def test_stage_row_converts_microseconds_to_milliseconds(self) -> None:
        stage = {
            "trace": "rank.trace.json.gz",
            "wanted_stage": "decode",
            "device": "RTX 4090",
            "selected_step_count": 4,
            "aggregate": {
                "step_duration_us": {"median_us": 2000.0, "p95_us": 3000.0},
                "kernel_union_busy_us": {"median_us": 1500.0},
                "kernel_union_occupancy_pct": {
                    "median_us": 75.0,
                    "p95_us": 80.0,
                },
                "observed_hardware_idle_gaps_us": {
                    "median_us": 2.0,
                    "p95_us": 8.0,
                },
                "nccl_hardware_event_count_in_trace": 9,
                "memcpy": {},
            },
        }
        row = analyze_run.stage_row(stage)
        self.assertEqual(row["step_ms_p50"], 2.0)
        self.assertEqual(row["kernel_busy_ms_p50"], 1.5)
        self.assertEqual(row["kernel_occupancy_p50"], 75.0)

    def test_resource_findings_cover_pp_balance_and_both_phases(self) -> None:
        stage_rows = []
        for stage in ("prefill", "decode"):
            for rank in (0, 1):
                stage_rows.append(
                    {
                        "stage": stage,
                        "pp_rank": rank,
                        "step_ms_p50": 10.0 + rank,
                        "kernel_occupancy_p50": 80.0 - rank,
                        "nccl_events": 4,
                        "memcpy": {
                            "Memcpy DtoD": {
                                "bytes": 1024,
                                "duration_us": 5.0,
                            }
                        },
                    }
                )
        phase_resources = {}
        for phase in ("profile_prefill", "profile_decode"):
            phase_resources[phase] = {
                "gpus": {
                    str(rank): {
                        "gpu_util_pct": {"p50": 90.0 - rank},
                        "memory_controller_util_pct": {"p50": 70.0 - rank},
                        "memory_used_mib": {"p50": 12000.0},
                        "memory_total_mib": {"p50": 24564.0},
                        "power_w": {"p50": 300.0},
                        "pcie_rx_mb_s": {"p95": 100.0},
                        "pcie_tx_mb_s": {"p95": 200.0},
                        "server_tree_cpu_pct": {"p50": 250.0},
                    }
                    for rank in (0, 1)
                }
            }
        findings = analyze_run.analysis_findings(
            {
                "load_gate": {"status": "pass", "reason": "fixture"},
                "stage_rows": stage_rows,
                "phase_resources": phase_resources,
            }
        )
        joined = "\n".join(findings)
        self.assertIn("prefill: PP0/PP1", joined)
        self.assertIn("decode: PP0/PP1", joined)
        self.assertIn("Profiled prefill phase", joined)
        self.assertIn("Profiled decode phase", joined)
        self.assertIn("HBM headroom", joined)

    def test_deployment_contract_requires_exact_pp2_4090_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment_path = Path(directory) / "environment.txt"
            environment_path.write_text(
                "2026-07-29T13:00:00+08:00\n"
                "gpu16\n"
                "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-a)\n"
                "GPU 1: NVIDIA GeForce RTX 4090 (UUID: GPU-b)\n",
                encoding="utf-8",
            )
            environment = {
                "sglang_commit": analyze_run.EXPECTED_SGLANG_COMMIT,
                "sglang_profile_v2": "0",
            }
            server_info = {
                "tp_size": 1,
                "pp_size": 2,
                "context_length": 16384,
                "chunked_prefill_size": 1024,
                "max_running_requests": 32,
                "disable_cuda_graph": True,
                "disable_radix_cache": True,
                "enable_metrics": True,
                "enable_request_time_stats_logging": True,
                "attention_backend": "triton",
                "sampling_backend": "pytorch",
                "mem_fraction_static": 0.80,
                "model_path": (
                    "/cache/models--Qwen--Qwen3-8B/snapshots/fixture"
                ),
            }
            models = {
                "data": [
                    {
                        "id": (
                            "/cache/models--Qwen--Qwen3-8B/"
                            "snapshots/fixture"
                        )
                    }
                ]
            }
            contract = analyze_run.deployment_contract(
                environment_path, environment, server_info, models
            )
        self.assertEqual(contract["status"], "pass")
        self.assertEqual(contract["gpu_names"], ["NVIDIA GeForce RTX 4090"] * 2)
        self.assertTrue(all(item["passed"] for item in contract["checks"]))

    def test_final_artifact_contract_rejects_missing_deliverables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)

            def write(relative: str, content: str = "fixture") -> None:
                path = run_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

            raw_traces = [
                f"profile/traces/combined/fixture-PP-{rank}.trace.json.gz"
                for rank in (0, 1)
            ]
            browser_trace = (
                "browser_trace/"
                "pp2-prefill-decode-resources.trace.json.gz"
            )
            trace_payload = {
                "traceEvents": [],
                "padding": secrets.token_hex(70_000),
            }
            for relative in [*raw_traces, browser_trace]:
                path = run_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(path, "wt", encoding="utf-8") as handle:
                    json.dump(trace_payload, handle)

            triage_files = [
                f"triage_combined_pp{rank}.md" for rank in (0, 1)
            ]
            for relative in triage_files:
                write(relative)
            for rank in (0, 1):
                write(f"stage_timeline_pp{rank}.json", "{}")
            memory_files = [
                f"profile/memory-{index}-memory-timeline.html"
                for index in range(4)
            ]
            for relative in memory_files:
                write(relative, "<!doctype html>memory")
            required = [
                "environment.txt",
                "gpu_processes_before_launch.txt",
                "live_server_info.json",
                "live_models.json",
                "metrics_after.prom",
                "resource_samples.csv",
                "pcie_throughput_dmon.csv",
                "profile/workload_summary.json",
                "profile/phase_markers.jsonl",
                "browser_trace/browser_trace_manifest.json",
                "analysis_summary.json",
                "REPORT.md",
            ]
            for relative in required:
                write(
                    relative,
                    "status=empty"
                    if relative == "gpu_processes_before_launch.txt"
                    else "fixture",
                )
            write(
                "index.html",
                "<!doctype html> One-file PP2 browser trace "
                "Resource utilization timeline Operation hot spots "
                "Deployment contract",
            )
            summary = {
                "evidence_grade": "complete",
                "selected_raw_trace_files": raw_traces,
                "browser_trace_files": [browser_trace],
                "stage_rows": [{}, {}, {}, {}],
                "triage_files": triage_files,
                "memory_html_files": memory_files,
                "workload": {
                    "chunked_prefill_size": 1024,
                    "prefill_profile": {
                        "capture_mode": (
                            "single_arm_forward_count_automatic"
                        ),
                        "workload": {"input_len": 4096},
                    },
                    "decode_profile": {
                        "workload": {
                            "input_len": 128,
                            "output_len": 128,
                        },
                    },
                },
            }
            contract = analyze_run.artifact_contract(run_dir, summary)
            self.assertEqual(contract["status"], "pass")

            write(
                "gpu_processes_before_launch.txt",
                "123, GPU-a, /usr/local/lib/ollama/llama-server, 8192 MiB",
            )
            contaminated = analyze_run.artifact_contract(run_dir, summary)
            self.assertEqual(contaminated["status"], "fail")
            self.assertIn(
                "gpu_preflight_empty",
                {
                    item["name"]
                    for item in contaminated["checks"]
                    if not item["passed"]
                },
            )
            write("gpu_processes_before_launch.txt", "status=empty")
            (run_dir / triage_files[0]).unlink()
            failed = analyze_run.artifact_contract(run_dir, summary)
        self.assertEqual(failed["status"], "fail")
        failed_checks = {
            item["name"] for item in failed["checks"] if not item["passed"]
        }
        self.assertIn("listed_artifacts_nonempty", failed_checks)

    def test_combined_browser_trace_contains_four_ranks_stages_and_counters(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            profile_root = run_dir / "profile"
            wall_base = dt.datetime(2026, 7, 29, 12, 0, 0).timestamp()
            trace_base_ns = int((wall_base - 1.0) * 1_000_000_000)
            trace_dir = profile_root / "traces" / "combined"
            trace_dir.mkdir(parents=True, exist_ok=True)
            for rank in (0, 1):
                trace_events = []
                for stage, timestamp_us, flow_id in (
                    ("prefill", 2_000_000.0, 7),
                    ("decode", 4_000_000.0, 8),
                ):
                    trace_events.extend(
                        [
                            {
                                "ph": "X",
                                "name": f"{stage}-rank-{rank}",
                                # Kineto reuses small PIDs across rank files;
                                # the merger must namespace this value.
                                "pid": 0,
                                "tid": 100 + rank,
                                "ts": timestamp_us,
                                "dur": 1000.0,
                            },
                            {
                                "ph": "s",
                                "cat": "fixture-flow",
                                "name": f"{stage}-flow-start-rank-{rank}",
                                "pid": 0,
                                "tid": 1,
                                "id": flow_id,
                                "ts": timestamp_us,
                            },
                            {
                                "ph": "f",
                                "cat": "fixture-flow",
                                "name": f"{stage}-flow-end-rank-{rank}",
                                "pid": 0,
                                "tid": 1,
                                "id": flow_id,
                                "ts": timestamp_us + 500.0,
                            },
                        ]
                    )
                path = trace_dir / f"fixture-PP-{rank}.trace.json.gz"
                payload = {
                    "schemaVersion": 1,
                    "baseTimeNanoseconds": trace_base_ns,
                    "deviceProperties": [
                        {"id": rank, "name": f"fixture-gpu-{rank}"}
                    ],
                    "traceEvents": trace_events,
                }
                with gzip.open(path, "wt", encoding="utf-8") as handle:
                    json.dump(payload, handle)

            (profile_root / "phase_markers.jsonl").write_text(
                "\n".join(
                    json.dumps(item)
                    for item in (
                        {
                            "event": "phase_start",
                            "phase": "profile_prefill",
                            "timestamp": wall_base + 1.0,
                        },
                        {
                            "event": "phase_end",
                            "phase": "profile_prefill",
                            "timestamp": wall_base + 2.0,
                        },
                        {
                            "event": "phase_start",
                            "phase": "profile_decode",
                            "timestamp": wall_base + 3.0,
                        },
                        {
                            "event": "phase_end",
                            "phase": "profile_decode",
                            "timestamp": wall_base + 4.0,
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            resource_columns = [
                "sample_wall_ts",
                "index",
                "utilization.gpu",
                "utilization.memory",
                "memory.used",
                "memory.total",
                "power.draw",
                "temperature.gpu",
                "clocks.current.sm",
                "clocks.current.memory",
                "server_tree_cpu_pct",
                "server_tree_rss_kib",
                "server_tree_processes",
                "gpu_compute_process_count",
                "foreign_gpu_process_count",
            ]
            with (run_dir / "resource_samples.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=resource_columns)
                writer.writeheader()
                for wall_ts in (wall_base + 1.2, wall_base + 3.2):
                    for gpu in (0, 1):
                        writer.writerow(
                            {
                                "sample_wall_ts": wall_ts,
                                "index": gpu,
                                "utilization.gpu": 90,
                                "utilization.memory": 80,
                                "memory.used": 12000,
                                "memory.total": 24576,
                                "power.draw": 300,
                                "temperature.gpu": 70,
                                "clocks.current.sm": 2500,
                                "clocks.current.memory": 10001,
                                "server_tree_cpu_pct": 250,
                                "server_tree_rss_kib": 1048576,
                                "server_tree_processes": 5,
                                "gpu_compute_process_count": 1,
                                "foreign_gpu_process_count": 0,
                            }
                        )
            (run_dir / "pcie_throughput_dmon.csv").write_text(
                "#Date, Time, gpu, rxpci, txpci\n"
                "20260729, 12:00:01, 0, 10, 20\n"
                "20260729, 12:00:01, 1, 11, 21\n"
                "20260729, 12:00:03, 0, 30, 40\n"
                "20260729, 12:00:03, 1, 31, 41\n",
                encoding="utf-8",
            )

            output = (
                run_dir
                / "browser_trace"
                / "pp2-prefill-decode-resources.trace.json.gz"
            )
            manifest = build_browser_trace.build_combined_trace(run_dir, output)
            with gzip.open(output, "rt", encoding="utf-8") as handle:
                merged = json.load(handle)

        self.assertEqual(manifest["source_trace_count"], 2)
        self.assertGreater(manifest["resource_event_count"], 0)
        ranges = {
            int(item["pp_rank"]): (
                int(item["pid_namespace_min"]),
                int(item["pid_namespace_max"]),
            )
            for item in manifest["source_traces"]
        }
        ordered_ranges = sorted(ranges.values())
        self.assertTrue(
            all(
                previous[1] < following[0]
                for previous, following in zip(
                    ordered_ranges, ordered_ranges[1:]
                )
            )
        )
        names = {event.get("name") for event in merged["traceEvents"]}
        self.assertIn("profile_prefill", names)
        self.assertIn("profile_decode", names)
        self.assertIn("GPU telemetry", names)
        self.assertIn("PCIe throughput", names)
        self.assertIn("prefill-rank-0", names)
        self.assertIn("decode-rank-1", names)
        event_pids = {
            event["name"]: event["pid"]
            for event in merged["traceEvents"]
            if str(event.get("name", "")).startswith(("prefill-rank-", "decode-rank-"))
        }
        self.assertEqual(
            event_pids["prefill-rank-0"], event_pids["decode-rank-0"]
        )
        self.assertEqual(
            event_pids["prefill-rank-1"], event_pids["decode-rank-1"]
        )
        self.assertNotEqual(event_pids["prefill-rank-0"], event_pids["prefill-rank-1"])
        flow_ids = {
            event["name"]: event["id"]
            for event in merged["traceEvents"]
            if "fixture-flow" == event.get("cat")
        }
        self.assertEqual(
            flow_ids["prefill-flow-start-rank-0"],
            flow_ids["prefill-flow-end-rank-0"],
        )
        self.assertEqual(len(set(flow_ids.values())), 4)


if __name__ == "__main__":
    unittest.main()
