from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

from AegisKV.experiments.src.global_des_config import (
    CONFIG_ROOT,
    EXPECTED_CELLS,
    GROUP_IDS,
    P0_SCENARIOS,
    ConfigValidationError,
    build_offline_physical_topology,
    generate_placement,
    load_suite,
)


class GlobalDesConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="aegiskv-config-v3-")
        self.root = Path(self.temporary.name) / "configs_v3"
        shutil.copytree(CONFIG_ROOT, self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def mutate(self, relative_path: str, operation: Callable[[dict[str, Any]], None]) -> None:
        path = self.root / relative_path
        value = json.loads(path.read_text(encoding="utf-8"))
        operation(value)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def assert_rejected(self, fragment: str = "") -> None:
        with self.assertRaises(ConfigValidationError) as caught:
            load_suite(self.root)
        if fragment:
            self.assertIn(fragment, str(caught.exception))

    def test_valid_suite_is_exactly_three_layers_and_2146_planned_cells(self) -> None:
        loaded = load_suite(self.root)

        self.assertEqual(tuple(group.group for group in loaded.groups), GROUP_IDS)
        self.assertEqual(
            {group.group: group.cell_count for group in loaded.groups},
            EXPECTED_CELLS,
        )
        self.assertEqual(loaded.total_cells, 2146)
        self.assertEqual(tuple(loaded.suite["layers"]), ("request", "failure", "deployment"))
        manifest = loaded.manifest_plan()
        self.assertEqual(manifest["status"], "planned_unexecuted")
        self.assertEqual(manifest["executed_cells"], 0)
        self.assertEqual(manifest["planned_cells"], 2146)
        self.assertEqual(len(manifest["resolved_config_sha256"]), 64)

    def test_cell_expansion_is_mechanical_and_has_no_hidden_axis(self) -> None:
        loaded = load_suite(self.root)
        p3 = loaded.group_map["P3"]
        cells = list(p3.iter_cells())

        self.assertEqual(len(cells), 640)
        self.assertEqual(
            tuple(cells[0]["axes"]),
            ("seed", "context_tokens", "node_count", "model", "policy"),
        )
        self.assertEqual(cells[0]["cell_ordinal"], 0)
        self.assertEqual(cells[-1]["cell_ordinal"], 639)

    def test_p0_gate_has_twelve_preregistered_scenarios(self) -> None:
        loaded = load_suite(self.root)
        p0 = loaded.group_map["P0"]

        self.assertEqual(tuple(p0.request.axes["contract_scenario"]), P0_SCENARIOS)
        self.assertEqual(p0.cell_count, 12)
        self.assertEqual(tuple(loaded.suite["p0_gate"]["blocks_groups"]), GROUP_IDS[1:])

    def test_placement_expands_every_dp_pp_tp_rank_to_one_physical_gpu(self) -> None:
        loaded = load_suite(self.root)
        deployment = loaded.group_map["P3"].deployment.fixed

        placements_8b = generate_placement(deployment, model_name="qwen3-8b", node_count=4)
        placements_32b = generate_placement(deployment, model_name="qwen3-32b", node_count=4)
        self.assertEqual(len(placements_8b), 32)
        self.assertEqual(len(placements_32b), 32)
        self.assertEqual(len({(item.node_id, item.local_gpu_id) for item in placements_8b}), 32)
        self.assertEqual(len({(item.dp_rank, item.pp_rank, item.tp_rank) for item in placements_32b}), 32)
        self.assertEqual({item.dp_rank for item in placements_32b}, set(range(8)))
        self.assertEqual({item.tp_rank for item in placements_32b}, set(range(4)))

    def test_runtime_topology_builder_preserves_tp_shards_and_node_pools(self) -> None:
        loaded = load_suite(self.root)
        deployment = loaded.group_map["P3"].deployment.fixed

        topology = build_offline_physical_topology(
            deployment,
            model_name="qwen3-32b",
            node_count=4,
        )

        self.assertEqual(len(topology.gpus), 32)
        self.assertEqual(len(topology.nodes), 4)
        self.assertEqual(topology.rank_groups, tuple((dp, 0) for dp in range(8)))
        self.assertTrue(all(len(topology.group(dp, 0)) == 4 for dp in range(8)))
        self.assertEqual(
            {spec.gpu.node_id for spec in topology.gpus},
            {"node0", "node1", "node2", "node3"},
        )
        self.assertTrue(
            all(spec.weights_bytes == 65_600_000_000 // 4 for spec in topology.gpus)
        )

    def test_unknown_suite_top_level_field_fails_closed(self) -> None:
        self.mutate("suite.json", lambda value: value.__setitem__("results", "complete"))
        self.assert_rejected("unknown top-level fields")

    def test_unknown_layer_top_level_field_fails_closed(self) -> None:
        self.mutate("P1/request.json", lambda value: value.__setitem__("fourth_layer", {}))
        self.assert_rejected("unknown top-level fields")

    def test_hidden_fourth_json_layer_fails_closed(self) -> None:
        (self.root / "P1" / "network.json").write_text("{}\n", encoding="utf-8")
        self.assert_rejected("exactly request/failure/deployment")

    def test_noncanonical_extends_fails_closed(self) -> None:
        self.mutate("P1/request.json", lambda value: value.__setitem__("extends", "../P2/request.json"))
        self.assert_rejected("extends must be")

    def test_wrong_axis_owner_fails_closed(self) -> None:
        def move_axis(value: dict[str, Any]) -> None:
            value["axes"]["load_multiplier"] = [1.0]

        self.mutate("P1/deployment.json", move_axis)
        self.assert_rejected("frozen paper matrix")

    def test_equal_product_but_changed_axes_fails_closed(self) -> None:
        def change_axes(value: dict[str, Any]) -> None:
            value["axes"]["policy"] = ["recompute", "aegiskv"]
            value["axes"]["model"] = ["qwen3-8b", "qwen3-32b", "synthetic-a", "synthetic-b"]

        self.mutate("P1/deployment.json", change_axes)
        self.assert_rejected("frozen paper matrix")

    def test_model_geometry_mismatch_fails_closed(self) -> None:
        def corrupt(value: dict[str, Any]) -> None:
            value["fixed"]["models"]["qwen3-32b"]["kv_bytes_per_token_total"] = 1

        self.mutate("common/deployment.json", corrupt)
        self.assert_rejected("KV geometry mismatch")

    def test_placement_capacity_mismatch_fails_closed(self) -> None:
        def corrupt(value: dict[str, Any]) -> None:
            value["fixed"]["models"]["qwen3-32b"]["dp_size_by_node_count"]["4"] = 7

        self.mutate("common/deployment.json", corrupt)
        self.assert_rejected("DP size mismatch")

    def test_per_gpu_hbm_overcommit_fails_closed(self) -> None:
        def corrupt(value: dict[str, Any]) -> None:
            value["fixed"]["gpu_hbm"]["primary_kv_budget_bytes_per_gpu"] = 40 * 1024**3

        self.mutate("common/deployment.json", corrupt)
        self.assert_rejected("per-GPU HBM ledger exceeds capacity")

    def test_per_node_dram_overcommit_fails_closed(self) -> None:
        def corrupt(value: dict[str, Any]) -> None:
            value["fixed"]["cluster"]["host_dram_capacity_bytes_per_node"] = 256 * 1024**3

        self.mutate("common/deployment.json", corrupt)
        self.assert_rejected("per-node DRAM ledger exceeds host capacity")

    def test_effective_link_bandwidth_above_path_capacity_fails_closed(self) -> None:
        def corrupt(value: dict[str, Any]) -> None:
            value["fixed"]["copy_resources"]["links"]["cross_node_fabric"][
                "central_effective_bandwidth_bytes_per_s"
            ] = 30_000_000_000

        self.mutate("common/deployment.json", corrupt)
        self.assert_rejected("effective link bandwidth exceeds path capacity")

    def test_hbm_and_dram_admission_must_be_independent(self) -> None:
        def corrupt(value: dict[str, Any]) -> None:
            value["fixed"]["aegiskv"]["tier_admission"]["node_dram"] = "after_hbm"

        self.mutate("common/deployment.json", corrupt)
        self.assert_rejected("DRAM admission is not independent")

    def test_hbm_to_dram_copy_edge_fails_closed(self) -> None:
        def corrupt(value: dict[str, Any]) -> None:
            value["fixed"]["aegiskv"]["allowed_copy_edges"].append("peer_hbm_to_node_dram")

        self.mutate("common/deployment.json", corrupt)
        self.assert_rejected("HBM-to-DRAM edge is forbidden")

    def test_total_cell_claim_cannot_be_changed(self) -> None:
        self.mutate("suite.json", lambda value: value.__setitem__("expected_total_cells", 613))
        self.assert_rejected("suite total must be 2146")

    def test_p0_gate_cannot_omit_a_scenario(self) -> None:
        def corrupt(value: dict[str, Any]) -> None:
            value["p0_gate"]["required_scenarios"].pop()

        self.mutate("suite.json", corrupt)
        self.assert_rejected("P0 scenario gate mismatch")

    def test_suite_cannot_claim_execution(self) -> None:
        self.mutate("suite.json", lambda value: value.__setitem__("status", "complete"))
        self.assert_rejected("planned_unexecuted")

    def test_duplicate_json_key_fails_closed(self) -> None:
        path = self.root / "P1" / "request.json"
        source = path.read_text(encoding="utf-8")
        source = source.replace(
            '"layer": "request",',
            '"layer": "request",\n  "layer": "failure",',
            1,
        )
        path.write_text(source, encoding="utf-8")
        self.assert_rejected("duplicate JSON key")


if __name__ == "__main__":
    unittest.main()
