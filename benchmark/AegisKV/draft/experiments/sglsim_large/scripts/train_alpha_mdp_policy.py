#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "policies" / "aegiskv_alpha_mdp_policy.json"

PRESSURE_BINS = {
    "low": 0.25,
    "medium": 0.60,
    "high": 0.90,
}
FAILURE_RATES = (0.001, 0.003, 0.01, 0.03, 0.10)
ALPHA_GRID = tuple(round(x / 100.0, 2) for x in range(5, 91, 5))


def _risk_from_failure_rate(rate: float) -> float:
    return min(max((math.log10(max(rate, 1e-6)) + 3.0) / 2.0, 0.0), 1.0)


def _transition_rows(labels: tuple[str, ...], current: str) -> list[tuple[str, float]]:
    idx = labels.index(current)
    rows: list[tuple[str, float]] = [(current, 0.70)]
    if idx > 0:
        rows.append((labels[idx - 1], 0.15))
    else:
        rows[0] = (current, rows[0][1] + 0.15)
    if idx + 1 < len(labels):
        rows.append((labels[idx + 1], 0.15))
    else:
        rows[0] = (current, rows[0][1] + 0.15)
    return rows


def immediate_cost(*, pressure: float, failure_rate: float, alpha: float) -> float:
    risk = _risk_from_failure_rate(failure_rate)
    miss_penalty = pressure * (1.0 + 3.0 * risk) * (1.0 - alpha) ** 1.7
    backup_overhead = alpha * (0.20 + 0.60 * pressure)
    hbm_contention = max(alpha - (0.45 + 0.35 * pressure), 0.0) ** 2 * 2.5
    return miss_penalty + backup_overhead + hbm_contention


def train_policy(*, iterations: int, gamma: float) -> dict:
    pressure_labels = tuple(PRESSURE_BINS)
    failure_labels = tuple(str(rate) for rate in FAILURE_RATES)
    states = [(p, f) for p in pressure_labels for f in failure_labels]
    values = {state: 0.0 for state in states}
    selected: dict[tuple[str, str], tuple[float, float]] = {}

    for _ in range(iterations):
        new_values: dict[tuple[str, str], float] = {}
        for pressure_label, failure_label in states:
            pressure = PRESSURE_BINS[pressure_label]
            failure_rate = float(failure_label)
            best_alpha = ALPHA_GRID[0]
            best_cost = float("inf")
            for alpha in ALPHA_GRID:
                future = 0.0
                for next_pressure, p_prob in _transition_rows(
                    pressure_labels, pressure_label
                ):
                    for next_failure, f_prob in _transition_rows(
                        failure_labels, failure_label
                    ):
                        future += p_prob * f_prob * values[(next_pressure, next_failure)]
                cost = immediate_cost(
                    pressure=pressure,
                    failure_rate=failure_rate,
                    alpha=alpha,
                ) + gamma * future
                if cost < best_cost:
                    best_cost = cost
                    best_alpha = alpha
            new_values[(pressure_label, failure_label)] = best_cost
            selected[(pressure_label, failure_label)] = (best_alpha, best_cost)
        values = new_values

    rows = []
    for pressure_label, failure_label in states:
        alpha, value = selected[(pressure_label, failure_label)]
        rows.append(
            {
                "critical_pressure": pressure_label,
                "critical_pressure_value": PRESSURE_BINS[pressure_label],
                "failure_rate_per_window": float(failure_label),
                "alpha": alpha,
                "value": round(value, 6),
            }
        )
    return {
        "version": 1,
        "source": "aegiskv_alpha_mdp_value_iteration_v1",
        "evidence_tier": "model_based_policy_table_not_measured",
        "state": {
            "critical_pressure": PRESSURE_BINS,
            "failure_rate_per_window": list(FAILURE_RATES),
        },
        "action": {"alpha_grid": list(ALPHA_GRID)},
        "training": {"iterations": iterations, "gamma": gamma},
        "states": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a model-based AegisKV alpha policy table with value iteration."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.92)
    args = parser.parse_args()

    payload = train_policy(iterations=max(args.iterations, 1), gamma=float(args.gamma))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

