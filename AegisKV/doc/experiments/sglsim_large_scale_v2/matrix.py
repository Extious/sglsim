#!/usr/bin/env python3
"""Build and validate the preregistered large-scale campaign matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "protocol.json"


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    campaign_axis: str
    model_id: str
    prompt_scale: float
    active_workflow_concurrency: int
    failure_rate_per_1000_worker_hours: float
    total_gpu_count: int
    tp_size: int
    pp_size: int
    system_dp_size: int
    network_effective_gb_s: float
    hbm_quota_gib_per_target: float
    dram_quota_gib_per_target: float
    algorithm1_prediction_error_multiplier: float = 1.0
    policies: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "campaign_axis": self.campaign_axis,
            "model_id": self.model_id,
            "prompt_scale": self.prompt_scale,
            "active_workflow_concurrency": self.active_workflow_concurrency,
            "failure_rate_per_1000_worker_hours": (
                self.failure_rate_per_1000_worker_hours
            ),
            "total_gpu_count": self.total_gpu_count,
            "tp_size": self.tp_size,
            "pp_size": self.pp_size,
            "system_dp_size": self.system_dp_size,
            "network_effective_gb_s": self.network_effective_gb_s,
            "hbm_quota_gib_per_target": self.hbm_quota_gib_per_target,
            "dram_quota_gib_per_target": self.dram_quota_gib_per_target,
            "algorithm1_prediction_error_multiplier": (
                self.algorithm1_prediction_error_multiplier
            ),
            "policies": list(self.policies),
        }


def load_protocol(path: Path = PROTOCOL) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _id(
    *,
    model: str,
    concurrency: int,
    failure: float,
    gpus: int,
    context: float,
    network: float,
    hbm_quota: float,
    dram_quota: float,
    error: float = 1.0,
) -> str:
    failure_label = str(failure).replace(".", "p")
    context_label = str(context).replace(".", "p")
    error_label = str(error).replace(".", "p")
    network_label = str(network).replace(".", "p")
    hbm_label = str(hbm_quota).replace(".", "p")
    dram_label = str(dram_quota).replace(".", "p")
    return (
        f"{model}-c{concurrency}-f{failure_label}-g{gpus}-"
        f"ctx{context_label}-net{network_label}-hbm{hbm_label}-"
        f"dram{dram_label}-e{error_label}"
    )


def build_scenarios(protocol: dict[str, Any]) -> list[Scenario]:
    primary = tuple(str(row) for row in protocol["primary_policies"])
    central_extra = tuple(
        str(row) for row in protocol["central_compositional_supplement"]
    )
    central_concurrency = int(
        protocol["workload"]["central_active_workflow_concurrency"]
    )
    central_failure = float(protocol["failure"]["central_rate"])
    central_context = float(protocol["workload"]["central_prompt_scale"])
    central_gpus = int(protocol["cluster"]["central_total_gpu_count"])
    central_network = float(
        protocol["cluster"]["bandwidth"]["central_network_effective_gb_s"]
    )
    central_hbm_quota = float(protocol["cluster"]["peer_hbm_quota_bytes_per_gpu"]) / (1024**3)
    central_dram_quota = float(protocol["cluster"]["store_dram_bytes_per_worker"]) / (1024**3)
    scenarios: dict[str, Scenario] = {}

    def add(
        axis: str,
        *,
        model_id: str = "qwen3_8b",
        concurrency: int = central_concurrency,
        failure: float = central_failure,
        gpus: int = central_gpus,
        context: float = central_context,
        network: float = central_network,
        hbm_quota: float = central_hbm_quota,
        dram_quota: float = central_dram_quota,
        error: float = 1.0,
        policies: tuple[str, ...] = primary,
    ) -> None:
        model = protocol["models"][model_id]
        pp_size = int(model["pp_size"])
        tp_size = int(model["tp_size"])
        if model_id == "qwen3_8b":
            system_dp = gpus // (pp_size * tp_size)
        else:
            # The 32B point keeps 64 GPUs and sixteen PP=4 replicas.
            system_dp = int(model["system_dp_size"])
        scenario_id = _id(
            model=model_id,
            concurrency=concurrency,
            failure=failure,
            gpus=gpus,
            context=context,
            network=network,
            hbm_quota=hbm_quota,
            dram_quota=dram_quota,
            error=error,
        )
        candidate = Scenario(
            scenario_id=scenario_id,
            campaign_axis=axis,
            model_id=model_id,
            prompt_scale=float(context),
            active_workflow_concurrency=int(concurrency),
            failure_rate_per_1000_worker_hours=float(failure),
            total_gpu_count=int(gpus),
            tp_size=tp_size,
            pp_size=pp_size,
            system_dp_size=system_dp,
            network_effective_gb_s=float(network),
            hbm_quota_gib_per_target=float(hbm_quota),
            dram_quota_gib_per_target=float(dram_quota),
            algorithm1_prediction_error_multiplier=float(error),
            policies=tuple(policies),
        )
        prior = scenarios.get(scenario_id)
        if prior is None:
            scenarios[scenario_id] = candidate
        else:
            merged_axes = "+".join(
                sorted(set(prior.campaign_axis.split("+")) | {axis})
            )
            merged_policies = tuple(dict.fromkeys((*prior.policies, *policies)))
            scenarios[scenario_id] = Scenario(
                **{
                    **prior.as_dict(),
                    "campaign_axis": merged_axes,
                    "policies": merged_policies,
                }
            )

    for concurrency in protocol["workload"][
        "active_workflow_concurrency_sweep"
    ]:
        for failure in protocol["failure"]["rate_sweep"]:
            policies = primary
            if (
                int(concurrency) == central_concurrency
                and float(failure) in {0.0, central_failure}
            ):
                policies = tuple(dict.fromkeys((*primary, *central_extra)))
            add(
                "concurrency_failure",
                concurrency=int(concurrency),
                failure=float(failure),
                policies=policies,
            )

    for gpus in protocol["cluster"]["total_gpu_sweep"]:
        add("cluster_scale", gpus=int(gpus))

    add("model_scale", model_id="qwen3_32b")

    for context in protocol["workload"]["prompt_scale_sweep"]:
        add("context_scale", context=float(context))

    for network in protocol["cluster"]["bandwidth"][
        "network_effective_sweep_gb_s"
    ]:
        add("network_bandwidth", network=float(network))

    for hbm_quota in protocol["cluster"]["peer_hbm_quota_sweep_gib_per_gpu"]:
        add("hbm_capacity", hbm_quota=float(hbm_quota))

    for dram_quota in protocol["cluster"]["store_dram_sweep_gib_per_worker"]:
        add("dram_capacity", dram_quota=float(dram_quota))

    for error in protocol["algorithm1_sensitivity"][
        "prediction_error_multiplier_sweep"
    ]:
        if float(error) == 1.0:
            continue
        add(
            "algorithm1_error",
            error=float(error),
            policies=(str(protocol["algorithm1_sensitivity"]["applies_to_policy"]),),
        )

    return sorted(scenarios.values(), key=lambda row: row.scenario_id)


def iter_runs(
    protocol: dict[str, Any], scenarios: Iterable[Scenario]
) -> Iterable[dict[str, Any]]:
    for scenario in scenarios:
        for policy in scenario.policies:
            for seed in protocol["statistics"]["paired_seeds"]:
                yield {
                    "run_id": f"{scenario.scenario_id}__{policy}__s{seed}",
                    "scenario_id": scenario.scenario_id,
                    "policy": policy,
                    "seed": int(seed),
                }


def matrix_manifest(protocol: dict[str, Any]) -> dict[str, Any]:
    scenarios = build_scenarios(protocol)
    runs = list(iter_runs(protocol, scenarios))
    payload = {
        "schema_version": 1,
        "campaign_id": protocol["campaign_id"],
        "evidence_class": protocol["evidence_class"],
        "scenario_count": len(scenarios),
        "policy_scenario_count": sum(len(row.policies) for row in scenarios),
        "run_count": len(runs),
        "scenarios": [row.as_dict() for row in scenarios],
        "runs": runs,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    payload["canonical_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def validate(payload: dict[str, Any], protocol: dict[str, Any]) -> None:
    scenarios = payload["scenarios"]
    central_hbm_quota = float(
        protocol["cluster"]["peer_hbm_quota_bytes_per_gpu"]
    ) / (1024**3)
    central_dram_quota = float(
        protocol["cluster"]["store_dram_bytes_per_worker"]
    ) / (1024**3)
    if len(scenarios) != 33:
        raise AssertionError(f"expected 33 unique scenarios, got {len(scenarios)}")
    if int(payload["policy_scenario_count"]) != 311:
        raise AssertionError(
            "expected 311 policy-scenarios, got "
            f"{payload['policy_scenario_count']}"
        )
    if int(payload["run_count"]) != 3110:
        raise AssertionError(f"expected 3110 runs, got {payload['run_count']}")
    if len({row["run_id"] for row in payload["runs"]}) != len(payload["runs"]):
        raise AssertionError("run IDs are not unique")
    central = [
        row
        for row in scenarios
        if row["model_id"] == "qwen3_8b"
        and row["prompt_scale"] == protocol["workload"]["central_prompt_scale"]
        and row["active_workflow_concurrency"]
        == protocol["workload"]["central_active_workflow_concurrency"]
        and row["failure_rate_per_1000_worker_hours"]
        == protocol["failure"]["central_rate"]
        and row["total_gpu_count"] == protocol["cluster"]["central_total_gpu_count"]
        and row["network_effective_gb_s"]
        == protocol["cluster"]["bandwidth"]["central_network_effective_gb_s"]
        and row["hbm_quota_gib_per_target"] == central_hbm_quota
        and row["dram_quota_gib_per_target"] == central_dram_quota
        and row["algorithm1_prediction_error_multiplier"] == 1.0
    ]
    if len(central) != 1:
        raise AssertionError("central scenario is not unique")
    expected_algorithms = {
        bits for bits in ("000", "001", "010", "011", "100", "101", "110", "111")
    }
    observed_algorithms = {
        str(policy).split("aegiskv_alg", 1)[1][:3]
        for policy in central[0]["policies"]
        if str(policy).startswith("aegiskv_alg")
    }
    if not expected_algorithms.issubset(observed_algorithms):
        raise AssertionError("central scenario lacks all 2^3 algorithm variants")
    normal_state = [
        row
        for row in scenarios
        if row["model_id"] == "qwen3_8b"
        and row["prompt_scale"] == protocol["workload"]["central_prompt_scale"]
        and row["active_workflow_concurrency"]
        == protocol["workload"]["central_active_workflow_concurrency"]
        and row["failure_rate_per_1000_worker_hours"] == 0.0
        and row["total_gpu_count"] == protocol["cluster"]["central_total_gpu_count"]
        and row["network_effective_gb_s"]
        == protocol["cluster"]["bandwidth"]["central_network_effective_gb_s"]
        and row["hbm_quota_gib_per_target"] == central_hbm_quota
        and row["dram_quota_gib_per_target"] == central_dram_quota
        and row["algorithm1_prediction_error_multiplier"] == 1.0
    ]
    if len(normal_state) != 1:
        raise AssertionError("rate-zero central normal-state scenario is not unique")
    normal_algorithms = {
        str(policy).split("aegiskv_alg", 1)[1][:3]
        for policy in normal_state[0]["policies"]
        if str(policy).startswith("aegiskv_alg")
    }
    if not expected_algorithms.issubset(normal_algorithms):
        raise AssertionError("rate-zero normal state lacks all 2^3 algorithms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=ROOT / "matrix.json")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    payload = matrix_manifest(protocol)
    validate(payload, protocol)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "scenario_count": payload["scenario_count"],
                "policy_scenario_count": payload["policy_scenario_count"],
                "run_count": payload["run_count"],
                "canonical_sha256": payload["canonical_sha256"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
