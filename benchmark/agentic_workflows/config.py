from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent  # benchmark/agentic_workflows
REPO_ROOT = PACKAGE_DIR.parents[1]
AEGIS_LARGE_DIR = (
    REPO_ROOT / "benchmark" / "AegisKV" / "draft" / "experiments" / "sglsim_large"
)
DEFAULT_BASE_CONFIG_DIR = AEGIS_LARGE_DIR / "config_failover_replay"
DEFAULT_RESULTS_ROOT = PACKAGE_DIR / "results"
DEFAULT_ALPHA_POLICY_PATH = (
    AEGIS_LARGE_DIR / "policies" / "aegiskv_alpha_mdp_policy.json"
)


@dataclass(frozen=True)
class AgenticWorkflowScenario:
    name: str
    description: str
    groups: int
    prompts_per_group: int
    turns: int
    max_concurrency: int
    system_prompt_len: int
    question_len: int
    output_len: int
    dp_size: int
    critical_agent_ratio: float
    failover_replay: bool
    failover_injection: str
    failover_retry_rank_offset: int
    seed: int
    start_port: int
    runs: int = 1
    workflow_shape: tuple[int, ...] = ()
    stage_long_decode_agent_index: int = 0
    long_decode_output_multiplier: float = 4.0
    range_ratio: float = 0.75
    ordered: bool = False
    fast_prepare: bool = False
    send_routing_key: bool = True
    replay_critical_only: bool = True
    disable_overlap_schedule: bool = False
    failover_retry_policy: str = ""
    aegis_policy: str = ""
    aegis_ablation: str = ""
    aegis_gpu_budget_fraction: float | None = None
    aegis_critical_pressure: float | None = None
    alpha_policy_path: str | None = None
    failure_rate_per_window: float | None = None
    failure_after_output_tokens: int | None = None
    backup_kv: bool | None = None

    @property
    def workflow_count(self) -> int:
        agents_per_workflow = (
            sum(self.workflow_shape)
            if self.workflow_shape
            else self.prompts_per_group
        )
        return self.groups * agents_per_workflow

    @property
    def estimated_first_turn_prompt_tokens(self) -> int:
        return int(self.system_prompt_len) + int(self.question_len)

    @property
    def estimated_final_turn_context_tokens(self) -> int:
        return (
            int(self.system_prompt_len)
            + int(self.turns) * int(self.question_len)
            + max(int(self.turns) - 1, 0) * int(self.output_len)
        )

    @property
    def estimated_long_decode_output_tokens(self) -> int:
        return max(
            int(self.output_len) + 1,
            int(round(float(self.output_len) * float(self.long_decode_output_multiplier))),
        )

    def with_overrides(
        self,
        *,
        runs: int | None = None,
        seed: int | None = None,
        start_port: int | None = None,
    ) -> "AgenticWorkflowScenario":
        updates: dict[str, Any] = {}
        if runs is not None:
            updates["runs"] = int(runs)
        if seed is not None:
            updates["seed"] = int(seed)
        if start_port is not None:
            updates["start_port"] = int(start_port)
        return replace(self, **updates) if updates else self


SCENARIOS: dict[str, AgenticWorkflowScenario] = {
    "smoke": AgenticWorkflowScenario(
        name="smoke",
        description="Cheap two-turn generated-shared-prefix workflow without failover replay.",
        groups=1,
        prompts_per_group=2,
        turns=2,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=16,
        dp_size=4,
        critical_agent_ratio=0.5,
        failover_replay=False,
        failover_injection="none",
        failover_retry_rank_offset=1,
        seed=101,
        start_port=30300,
    ),
    "physical_copy_restore_smoke": AgenticWorkflowScenario(
        name="physical_copy_restore_smoke",
        description="Cheap DP4 hard-restart replay for physical-copy-calibrated restore timing.",
        groups=1,
        prompts_per_group=2,
        turns=2,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=16,
        dp_size=4,
        critical_agent_ratio=1.0,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8080,
        start_port=30820,
    ),
    "shared_prefix_dp2": AgenticWorkflowScenario(
        name="shared_prefix_dp2",
        description="DP2 shared-prefix agentic workload for normal-path cache and frontier measurements.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=64,
        dp_size=2,
        critical_agent_ratio=0.25,
        failover_replay=False,
        failover_injection="none",
        failover_retry_rank_offset=1,
        seed=7000,
        start_port=30320,
        runs=3,
    ),
    "staged_13131_motivation": AgenticWorkflowScenario(
        name="staged_13131_motivation",
        description=(
            "1-3-1-3-1 staged workflow; singleton stages are Critical, and one "
            "agent in each 3-way fan-out has a long decode tail and is Critical."
        ),
        groups=8,
        prompts_per_group=9,
        turns=3,
        max_concurrency=24,
        system_prompt_len=512,
        question_len=192,
        output_len=48,
        dp_size=4,
        critical_agent_ratio=0.0,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8700,
        start_port=31420,
        runs=3,
        workflow_shape=(1, 3, 1, 3, 1),
        stage_long_decode_agent_index=0,
        long_decode_output_multiplier=5.0,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        alpha_policy_path=str(DEFAULT_ALPHA_POLICY_PATH),
        failure_rate_per_window=0.01,
        backup_kv=True,
    ),
    "staged_13131_alpha_online": AgenticWorkflowScenario(
        name="staged_13131_alpha_online",
        description=(
            "Small 1-3-1-3-1 staged workflow for online MDP alpha-decision "
            "evidence without failure replay."
        ),
        groups=3,
        prompts_per_group=9,
        turns=3,
        max_concurrency=9,
        system_prompt_len=512,
        question_len=192,
        output_len=48,
        dp_size=4,
        critical_agent_ratio=0.0,
        failover_replay=False,
        failover_injection="none",
        failover_retry_rank_offset=1,
        seed=8750,
        start_port=32420,
        runs=1,
        workflow_shape=(1, 3, 1, 3, 1),
        stage_long_decode_agent_index=0,
        long_decode_output_multiplier=5.0,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        alpha_policy_path=str(DEFAULT_ALPHA_POLICY_PATH),
        failure_rate_per_window=0.01,
        backup_kv=True,
    ),
    "staged_13131_policy_no_backup": AgenticWorkflowScenario(
        name="staged_13131_policy_no_backup",
        description=(
            "1-3-1-3-1 staged workflow policy sweep point with Critical-agent "
            "backup disabled."
        ),
        groups=8,
        prompts_per_group=9,
        turns=3,
        max_concurrency=24,
        system_prompt_len=512,
        question_len=192,
        output_len=48,
        dp_size=4,
        critical_agent_ratio=0.0,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8900,
        start_port=33420,
        runs=3,
        workflow_shape=(1, 3, 1, 3, 1),
        stage_long_decode_agent_index=0,
        long_decode_output_multiplier=5.0,
        failover_retry_policy="peer",
        aegis_policy="no_backup",
        backup_kv=False,
    ),
    "staged_13131_policy_host_only": AgenticWorkflowScenario(
        name="staged_13131_policy_host_only",
        description=(
            "1-3-1-3-1 staged workflow policy sweep point with HostBackup-only "
            "Critical-agent recovery frontier."
        ),
        groups=8,
        prompts_per_group=9,
        turns=3,
        max_concurrency=24,
        system_prompt_len=512,
        question_len=192,
        output_len=48,
        dp_size=4,
        critical_agent_ratio=0.0,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8900,
        start_port=33520,
        runs=3,
        workflow_shape=(1, 3, 1, 3, 1),
        stage_long_decode_agent_index=0,
        long_decode_output_multiplier=5.0,
        failover_retry_policy="peer",
        aegis_policy="host_only",
        backup_kv=True,
    ),
    "staged_13131_policy_gpu_replica": AgenticWorkflowScenario(
        name="staged_13131_policy_gpu_replica",
        description=(
            "1-3-1-3-1 staged workflow policy sweep point with GPUReplica-only "
            "Critical-agent recovery frontier."
        ),
        groups=8,
        prompts_per_group=9,
        turns=3,
        max_concurrency=24,
        system_prompt_len=512,
        question_len=192,
        output_len=48,
        dp_size=4,
        critical_agent_ratio=0.0,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8900,
        start_port=33620,
        runs=3,
        workflow_shape=(1, 3, 1, 3, 1),
        stage_long_decode_agent_index=0,
        long_decode_output_multiplier=5.0,
        failover_retry_policy="peer",
        aegis_policy="gpu_replica",
        backup_kv=True,
    ),
    "staged_13131_policy_aegiskv": AgenticWorkflowScenario(
        name="staged_13131_policy_aegiskv",
        description=(
            "1-3-1-3-1 staged workflow policy sweep point with AegisKV dynamic "
            "Host/GPUReplica recovery frontier."
        ),
        groups=8,
        prompts_per_group=9,
        turns=3,
        max_concurrency=24,
        system_prompt_len=512,
        question_len=192,
        output_len=48,
        dp_size=4,
        critical_agent_ratio=0.0,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8900,
        start_port=33720,
        runs=3,
        workflow_shape=(1, 3, 1, 3, 1),
        stage_long_decode_agent_index=0,
        long_decode_output_multiplier=5.0,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        alpha_policy_path=str(DEFAULT_ALPHA_POLICY_PATH),
        failure_rate_per_window=0.01,
        backup_kv=True,
    ),
    "normal_overhead_baseline": AgenticWorkflowScenario(
        name="normal_overhead_baseline",
        description="DP2 no-failure baseline with critical/Aegis protection disabled.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=64,
        dp_size=2,
        critical_agent_ratio=0.0,
        failover_replay=False,
        failover_injection="none",
        failover_retry_rank_offset=1,
        seed=7900,
        start_port=30720,
        runs=3,
    ),
    "normal_overhead_aegis": AgenticWorkflowScenario(
        name="normal_overhead_aegis",
        description="DP2 no-failure workload with AegisKV selective critical-agent protection enabled.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=64,
        dp_size=2,
        critical_agent_ratio=0.25,
        failover_replay=False,
        failover_injection="none",
        failover_retry_rank_offset=1,
        seed=7900,
        start_port=30760,
        runs=3,
    ),
    "shared_prefix_sweep_low": AgenticWorkflowScenario(
        name="shared_prefix_sweep_low",
        description="Low-overlap shared-prefix sweep point with two turns and long per-turn questions.",
        groups=4,
        prompts_per_group=4,
        turns=2,
        max_concurrency=1,
        system_prompt_len=32,
        question_len=1024,
        output_len=16,
        dp_size=2,
        critical_agent_ratio=0.25,
        failover_replay=False,
        failover_injection="none",
        failover_retry_rank_offset=1,
        seed=7600,
        start_port=30600,
        runs=3,
    ),
    "shared_prefix_sweep_mid": AgenticWorkflowScenario(
        name="shared_prefix_sweep_mid",
        description="Mid-overlap shared-prefix sweep point matching the main three-turn workflow shape.",
        groups=4,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=32,
        dp_size=2,
        critical_agent_ratio=0.25,
        failover_replay=False,
        failover_injection="none",
        failover_retry_rank_offset=1,
        seed=7700,
        start_port=30640,
        runs=3,
    ),
    "shared_prefix_sweep_high": AgenticWorkflowScenario(
        name="shared_prefix_sweep_high",
        description="High-overlap shared-prefix sweep point with five turns and a long shared system prefix.",
        groups=4,
        prompts_per_group=4,
        turns=5,
        max_concurrency=1,
        system_prompt_len=2048,
        question_len=64,
        output_len=16,
        dp_size=2,
        critical_agent_ratio=0.25,
        failover_replay=False,
        failover_injection="none",
        failover_retry_rank_offset=1,
        seed=7800,
        start_port=30680,
        runs=3,
    ),
    "hard_failover_host": AgenticWorkflowScenario(
        name="hard_failover_host",
        description="DP2 hard-restart replay with HostBackup lookup on the retry path.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=64,
        dp_size=2,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=7100,
        start_port=30360,
        runs=3,
    ),
    "policy_no_backup": AgenticWorkflowScenario(
        name="policy_no_backup",
        description="DP4 hard-restart policy sweep point with critical requests recomputing after failure.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8100,
        start_port=30860,
        runs=3,
        aegis_policy="no_backup",
        backup_kv=False,
    ),
    "policy_host_only": AgenticWorkflowScenario(
        name="policy_host_only",
        description="DP4 hard-restart policy sweep point with HostBackup-only recovery frontier.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8100,
        start_port=30900,
        runs=3,
        aegis_policy="host_only",
        backup_kv=True,
    ),
    "policy_gpu_replica": AgenticWorkflowScenario(
        name="policy_gpu_replica",
        description="DP4 hard-restart policy sweep point with GPUReplica-only recovery frontier.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8100,
        start_port=30940,
        runs=3,
        aegis_policy="gpu_replica",
        backup_kv=True,
    ),
    "policy_aegiskv": AgenticWorkflowScenario(
        name="policy_aegiskv",
        description="DP4 hard-restart policy sweep point with AegisKV dynamic Host/GPUReplica frontier.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8100,
        start_port=30980,
        runs=3,
        aegis_policy="aegiskv",
        backup_kv=True,
    ),
    "hard_failover_peer": AgenticWorkflowScenario(
        name="hard_failover_peer",
        description="DP4 hard-restart replay where retry targets the adjacent peer rank.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=7200,
        start_port=30420,
        runs=3,
    ),
    "hard_failover_peer_hot_tail": AgenticWorkflowScenario(
        name="hard_failover_peer_hot_tail",
        description="DP4 hard-restart replay with short prefixes so peer GPUReplica hot-tail restore wins.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=7400,
        start_port=30540,
        runs=3,
    ),
    "hard_failover_nonpeer": AgenticWorkflowScenario(
        name="hard_failover_nonpeer",
        description="DP4 hard-restart replay where retry targets a non-peer rank.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=2,
        seed=7300,
        start_port=30480,
        runs=3,
    ),
    "peer_route_nonpeer_hot_tail": AgenticWorkflowScenario(
        name="peer_route_nonpeer_hot_tail",
        description="DP4 online peer-selection sweep point forcing a healthy non-peer retry route.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=2,
        seed=8500,
        start_port=31020,
        runs=3,
        failover_retry_policy="non_peer",
        aegis_policy="aegiskv",
        backup_kv=True,
    ),
    "peer_route_random_hot_tail": AgenticWorkflowScenario(
        name="peer_route_random_hot_tail",
        description="DP4 online peer-selection sweep point routing retries to a deterministic random healthy rank.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8500,
        start_port=31060,
        runs=3,
        failover_retry_policy="random_healthy",
        aegis_policy="aegiskv",
        backup_kv=True,
    ),
    "peer_route_aegiskv_hot_tail": AgenticWorkflowScenario(
        name="peer_route_aegiskv_hot_tail",
        description="DP4 online peer-selection sweep point routing retries to the recorded peer rank.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8500,
        start_port=31100,
        runs=3,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        backup_kv=True,
    ),
    "ablation_full_aegiskv": AgenticWorkflowScenario(
        name="ablation_full_aegiskv",
        description="DP4 online Fig. 12 ablation baseline with all AegisKV recovery signals enabled.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8600,
        start_port=31140,
        runs=3,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        backup_kv=True,
    ),
    "ablation_no_lifecycle": AgenticWorkflowScenario(
        name="ablation_no_lifecycle",
        description="DP4 online Fig. 12 ablation disabling lifecycle lookup across retries.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8600,
        start_port=31180,
        runs=3,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        aegis_ablation="no_lifecycle",
        backup_kv=True,
    ),
    "ablation_no_segment_granularity": AgenticWorkflowScenario(
        name="ablation_no_segment_granularity",
        description="DP4 online Fig. 12 ablation requiring all-or-nothing restored frontiers.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8600,
        start_port=31220,
        runs=3,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        aegis_ablation="no_segment_granularity",
        backup_kv=True,
    ),
    "ablation_no_restore_recompute": AgenticWorkflowScenario(
        name="ablation_no_restore_recompute",
        description="DP4 online Fig. 12 ablation using restore-first instead of restore/recompute lower envelope.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8600,
        start_port=31260,
        runs=3,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        aegis_ablation="no_restore_recompute",
        backup_kv=True,
    ),
    "ablation_no_pressure": AgenticWorkflowScenario(
        name="ablation_no_pressure",
        description="DP4 online Fig. 12 ablation removing pressure-aware peer placement.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8600,
        start_port=31300,
        runs=3,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        aegis_ablation="no_pressure",
        backup_kv=True,
    ),
    "ablation_no_hysteresis": AgenticWorkflowScenario(
        name="ablation_no_hysteresis",
        description="DP4 online Fig. 12 ablation removing stable peer-frontier hysteresis.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8600,
        start_port=31340,
        runs=3,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        aegis_ablation="no_hysteresis",
        backup_kv=True,
    ),
    "ablation_no_critical_restore": AgenticWorkflowScenario(
        name="ablation_no_critical_restore",
        description="DP4 online Fig. 12 ablation disabling restore for critical agents.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8600,
        start_port=31380,
        runs=3,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        aegis_ablation="no_critical_restore",
        backup_kv=True,
    ),
}


def _register_staged_policy_lite_scenarios() -> None:
    common: dict[str, Any] = {
        "groups": 3,
        "prompts_per_group": 9,
        "turns": 3,
        "max_concurrency": 9,
        "system_prompt_len": 512,
        "question_len": 192,
        "output_len": 48,
        "dp_size": 4,
        "critical_agent_ratio": 0.0,
        "failover_replay": True,
        "failover_injection": "hard-restart",
        "failover_retry_rank_offset": 1,
        "seed": 8950,
        "runs": 3,
        "workflow_shape": (1, 3, 1, 3, 1),
        "stage_long_decode_agent_index": 0,
        "long_decode_output_multiplier": 5.0,
        "failover_retry_policy": "peer",
    }
    policy_specs = (
        ("no_backup", "backup disabled", "no_backup", False, 34420),
        ("host_only", "HostBackup-only recovery frontier", "host_only", True, 34520),
        (
            "gpu_replica",
            "GPUReplica-only recovery frontier",
            "gpu_replica",
            True,
            34620,
        ),
        ("aegiskv", "AegisKV dynamic Host/GPUReplica frontier", "aegiskv", True, 34720),
    )
    for suffix, label, policy, backup_kv, start_port in policy_specs:
        name = f"staged_13131_policy_lite_{suffix}"
        SCENARIOS[name] = AgenticWorkflowScenario(
            name=name,
            description=(
                "Lite 1-3-1-3-1 staged workflow policy sweep point with "
                f"Critical-agent {label}."
            ),
            start_port=start_port,
            aegis_policy=policy,
            backup_kv=backup_kv,
            alpha_policy_path=(
                str(DEFAULT_ALPHA_POLICY_PATH) if policy == "aegiskv" else None
            ),
            failure_rate_per_window=0.01 if policy == "aegiskv" else None,
            **common,
        )


def _register_staged_policy_stress_scenarios() -> None:
    common: dict[str, Any] = {
        "groups": 3,
        "prompts_per_group": 9,
        "turns": 3,
        "max_concurrency": 9,
        "system_prompt_len": 512,
        "question_len": 192,
        "output_len": 128,
        "dp_size": 4,
        "critical_agent_ratio": 0.0,
        "failover_replay": True,
        "failover_injection": "soft",
        "failover_retry_rank_offset": 1,
        "seed": 9050,
        "runs": 3,
        "workflow_shape": (1, 3, 1, 3, 1),
        "stage_long_decode_agent_index": 0,
        "long_decode_output_multiplier": 8.0,
        "failover_retry_policy": "peer",
        "failure_after_output_tokens": 128,
    }
    policy_specs = (
        ("no_backup", "backup disabled", "no_backup", False, 35420),
        ("host_only", "HostBackup-only recovery frontier", "host_only", True, 35520),
        (
            "gpu_replica",
            "GPUReplica-only recovery frontier",
            "gpu_replica",
            True,
            35620,
        ),
        ("aegiskv", "AegisKV dynamic Host/GPUReplica frontier", "aegiskv", True, 35720),
    )
    for suffix, label, policy, backup_kv, start_port in policy_specs:
        name = f"staged_13131_policy_stress_{suffix}"
        SCENARIOS[name] = AgenticWorkflowScenario(
            name=name,
            description=(
                "Stress 1-3-1-3-1 staged workflow policy sweep point with "
                f"soft failure injection and Critical-agent {label}."
            ),
            start_port=start_port,
            aegis_policy=policy,
            backup_kv=backup_kv,
            alpha_policy_path=(
                str(DEFAULT_ALPHA_POLICY_PATH) if policy == "aegiskv" else None
            ),
            failure_rate_per_window=0.01 if policy == "aegiskv" else None,
            **common,
        )


def _register_staged_100k_context_scenarios() -> None:
    common: dict[str, Any] = {
        "groups": 1,
        "prompts_per_group": 9,
        "turns": 3,
        "max_concurrency": 3,
        "system_prompt_len": 98304,
        "question_len": 1024,
        "output_len": 128,
        "dp_size": 4,
        "critical_agent_ratio": 0.0,
        "failover_replay": True,
        "failover_injection": "soft",
        "failover_retry_rank_offset": 1,
        "seed": 9150,
        "runs": 1,
        "workflow_shape": (1, 3, 1, 3, 1),
        "stage_long_decode_agent_index": 0,
        "long_decode_output_multiplier": 8.0,
        "failover_retry_policy": "peer",
        "failure_after_output_tokens": 128,
    }
    policy_specs = (
        ("no_backup", "backup disabled", "no_backup", False, 36420),
        ("host_only", "HostBackup-only recovery frontier", "host_only", True, 36520),
        (
            "gpu_replica",
            "GPUReplica-only recovery frontier",
            "gpu_replica",
            True,
            36620,
        ),
        ("aegiskv", "AegisKV dynamic Host/GPUReplica frontier", "aegiskv", True, 36720),
    )
    for suffix, label, policy, backup_kv, start_port in policy_specs:
        name = f"staged_13131_100k_context_{suffix}"
        SCENARIOS[name] = AgenticWorkflowScenario(
            name=name,
            description=(
                "100K-context 1-3-1-3-1 staged workflow policy point with "
                f"Critical-agent {label}; singleton stages and one long-decode "
                "fan-out agent per width-3 stage are Critical."
            ),
            start_port=start_port,
            aegis_policy=policy,
            backup_kv=backup_kv,
            alpha_policy_path=(
                str(DEFAULT_ALPHA_POLICY_PATH) if policy == "aegiskv" else None
            ),
            failure_rate_per_window=0.01 if policy == "aegiskv" else None,
            **common,
        )


def _budget_suffix(value: float) -> str:
    return f"b{int(round(value * 100)):03d}"


def _register_budget_sweep_scenarios() -> None:
    budgets = (0.0, 0.1, 0.2, 0.5, 1.0)
    common = {
        "groups": 8,
        "prompts_per_group": 4,
        "turns": 3,
        "max_concurrency": 1,
        "output_len": 64,
        "dp_size": 4,
        "critical_agent_ratio": 0.25,
        "failover_replay": True,
        "failover_injection": "hard-restart",
        "failover_retry_rank_offset": 1,
        "seed": 8700,
        "runs": 3,
        "failover_retry_policy": "peer",
        "aegis_policy": "aegiskv",
        "backup_kv": True,
    }
    workload_specs = (
        ("budget_hot_tail", "hot-tail", 64, 32, 31420, "peer", 1),
        ("budget_long_prefix", "long-prefix", 512, 256, 31620, "peer", 1),
        ("budget_nonpeer", "non-peer fan-out", 64, 32, 31820, "non_peer", 2),
    )
    for prefix, label, system_len, question_len, base_port, retry_policy, retry_offset in workload_specs:
        for index, budget in enumerate(budgets):
            name = f"{prefix}_{_budget_suffix(budget)}"
            SCENARIOS[name] = AgenticWorkflowScenario(
                name=name,
                description=(
                    "DP4 online Fig. 3 GPUReplica budget sweep for "
                    f"{label} workload at budget={budget:.2f}."
                ),
                system_prompt_len=system_len,
                question_len=question_len,
                start_port=base_port + index * 10,
                failover_retry_policy=retry_policy,
                failover_retry_rank_offset=retry_offset,
                aegis_gpu_budget_fraction=budget,
                **{
                    key: value
                    for key, value in common.items()
                    if key not in {"failover_retry_policy", "failover_retry_rank_offset"}
                },
            )

    timing_specs = (
        ("budget_timing_early", "early decode", 1, 32020),
        ("budget_timing_late", "late decode", 64, 32220),
    )
    for prefix, label, failure_after_output, base_port in timing_specs:
        for index, budget in enumerate(budgets):
            name = f"{prefix}_{_budget_suffix(budget)}"
            SCENARIOS[name] = AgenticWorkflowScenario(
                name=name,
                description=(
                    "DP4 online Fig. 13 failure-timing budget sweep for "
                    f"{label} at budget={budget:.2f}."
                ),
                system_prompt_len=64,
                question_len=32,
                start_port=base_port + index * 10,
                aegis_gpu_budget_fraction=budget,
                failure_after_output_tokens=failure_after_output,
                **common,
            )


_register_staged_policy_lite_scenarios()
_register_staged_policy_stress_scenarios()
_register_staged_100k_context_scenarios()
_register_budget_sweep_scenarios()


def scenario_names() -> list[str]:
    return sorted(SCENARIOS)


def get_scenario(name: str) -> AgenticWorkflowScenario:
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        known = ", ".join(scenario_names())
        raise ValueError(f"Unknown scenario {name!r}; known scenarios: {known}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def materialize_synthesis_config(
    scenario: AgenticWorkflowScenario,
    *,
    base_config_dir: Path = DEFAULT_BASE_CONFIG_DIR,
    output_dir: Path,
    trial: int = 0,
) -> Path:
    client = _load_json(Path(base_config_dir) / "client.json")
    server = _load_json(Path(base_config_dir) / "server.json")

    client.update(
        {
            "dataset_name": "generated-shared-prefix",
            "seed": int(scenario.seed) + int(trial),
            "request_rate": "inf",
            "max_concurrency": int(scenario.max_concurrency),
            "gsp_num_groups": int(scenario.groups),
            "gsp_prompts_per_group": int(scenario.prompts_per_group),
            "gsp_system_prompt_len": int(scenario.system_prompt_len),
            "gsp_question_len": int(scenario.question_len),
            "gsp_output_len": int(scenario.output_len),
            "gsp_range_ratio": float(scenario.range_ratio),
            "gsp_fast_prepare": bool(scenario.fast_prepare),
            "gsp_send_routing_key": bool(scenario.send_routing_key),
            "gsp_num_turns": int(scenario.turns),
            "gsp_ordered": bool(scenario.ordered),
            "gsp_critical_agent_ratio": float(scenario.critical_agent_ratio),
            "gsp_failover_replay": bool(scenario.failover_replay),
            "gsp_failover_replay_critical_only": bool(scenario.replay_critical_only),
            "gsp_failover_injection": str(scenario.failover_injection),
            "gsp_failover_retry_rank_offset": int(
                scenario.failover_retry_rank_offset
            ),
            "gsp_failover_retry_policy": str(scenario.failover_retry_policy),
            "gsp_aegis_policy": str(scenario.aegis_policy),
            "gsp_aegis_ablation": str(scenario.aegis_ablation),
        }
    )
    if scenario.workflow_shape:
        client["gsp_workflow_shape"] = list(scenario.workflow_shape)
        client["gsp_stage_long_decode_agent_index"] = int(
            scenario.stage_long_decode_agent_index
        )
        client["gsp_long_decode_output_multiplier"] = float(
            scenario.long_decode_output_multiplier
        )
    if scenario.aegis_gpu_budget_fraction is not None:
        client["gsp_aegis_gpu_budget_fraction"] = float(
            scenario.aegis_gpu_budget_fraction
        )
    if scenario.aegis_critical_pressure is not None:
        client["gsp_aegis_critical_pressure"] = float(
            scenario.aegis_critical_pressure
        )
    if scenario.alpha_policy_path:
        client["gsp_alpha_policy_path"] = str(scenario.alpha_policy_path)
    if scenario.failure_rate_per_window is not None:
        client["gsp_failure_rate_per_window"] = float(
            scenario.failure_rate_per_window
        )
    if scenario.failure_after_output_tokens is not None:
        client["gsp_failure_after_output_tokens"] = int(
            scenario.failure_after_output_tokens
        )
    if scenario.backup_kv is not None:
        client["gsp_backup_kv"] = bool(scenario.backup_kv)

    scheduler = dict(server.get("scheduler") or {})
    scheduler["dp_size"] = int(scenario.dp_size)
    scheduler["enable_routing_key_affinity"] = bool(scenario.send_routing_key)
    scheduler["disable_overlap_schedule"] = bool(scenario.disable_overlap_schedule)
    server["scheduler"] = scheduler
    server["port"] = int(scenario.start_port) + int(trial)
    server["experiment_name"] = f"agentic-{scenario.name}-t{int(trial):03d}"

    config_dir = Path(output_dir) / "configs" / scenario.name / f"trial_{int(trial):03d}"
    _write_json(config_dir / "client.json", client)
    _write_json(config_dir / "server.json", server)
    return config_dir


def scenario_plan_row(
    scenario: AgenticWorkflowScenario,
    *,
    config_dir: Path,
    trial: int,
) -> dict[str, Any]:
    return {
        "scenario": scenario.name,
        "trial": int(trial),
        "config_dir": str(config_dir),
        "seed": int(scenario.seed) + int(trial),
        "port": int(scenario.start_port) + int(trial),
        "dp_size": int(scenario.dp_size),
        "workflow_count": int(scenario.workflow_count),
        "turns": int(scenario.turns),
        "system_prompt_len": int(scenario.system_prompt_len),
        "question_len": int(scenario.question_len),
        "output_len": int(scenario.output_len),
        "estimated_first_turn_prompt_tokens": int(
            scenario.estimated_first_turn_prompt_tokens
        ),
        "estimated_final_turn_context_tokens": int(
            scenario.estimated_final_turn_context_tokens
        ),
        "estimated_long_decode_output_tokens": int(
            scenario.estimated_long_decode_output_tokens
        ),
        "max_concurrency": int(scenario.max_concurrency),
        "failover_replay": bool(scenario.failover_replay),
        "failover_injection": scenario.failover_injection,
        "failover_retry_rank_offset": int(scenario.failover_retry_rank_offset),
        "failover_retry_policy": str(scenario.failover_retry_policy),
        "disable_overlap_schedule": bool(scenario.disable_overlap_schedule),
        "aegis_policy": str(scenario.aegis_policy),
        "aegis_ablation": str(scenario.aegis_ablation),
        "aegis_gpu_budget_fraction": scenario.aegis_gpu_budget_fraction,
        "aegis_critical_pressure": scenario.aegis_critical_pressure,
        "alpha_policy_path": scenario.alpha_policy_path,
        "failure_rate_per_window": scenario.failure_rate_per_window,
        "failure_after_output_tokens": scenario.failure_after_output_tokens,
        "backup_kv": scenario.backup_kv,
    }
