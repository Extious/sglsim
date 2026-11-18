from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any


ENV_ALPHA_POLICY = "SGLSIM_AEGIS_ALPHA_POLICY"
ENV_FAILURE_RATE_PER_WINDOW = "SGLSIM_AEGIS_FAILURE_RATE_PER_WINDOW"
ENV_ONLINE_ALPHA_ADAPTER = "SGLSIM_AEGIS_ONLINE_ALPHA_ADAPTER"
DEFAULT_FAILURE_RATE_PER_WINDOW = 0.01


@dataclass(frozen=True)
class AlphaPolicyDecision:
    alpha: float
    critical_pressure: float
    failure_rate_per_window: float
    source: str


@dataclass(frozen=True)
class OnlineAlphaUpdate:
    alpha: float
    delta: float
    source: str


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def _as_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def pressure_from_request(
    *,
    critical_agent: bool,
    decode_dominant: bool,
    stage_width: int,
) -> float:
    stage_width = max(int(stage_width), 1)
    if not critical_agent:
        return 0.18 if stage_width > 1 else 0.25
    if decode_dominant:
        return 0.90
    if stage_width <= 1:
        return 0.75
    return _clamp(0.55 + 0.04 * min(stage_width, 4), 0.55, 0.80)


def resolve_failure_rate_per_window(value: Any = None) -> float:
    if value is None:
        value = os.getenv(ENV_FAILURE_RATE_PER_WINDOW, "")
    rate = _as_float(value, DEFAULT_FAILURE_RATE_PER_WINDOW)
    return _clamp(rate, 1e-6, 1.0)


def _risk_from_failure_rate(rate: float) -> float:
    # Map 1e-3..1e-1 to 0..1, then clamp for out-of-range sweeps.
    return _clamp((math.log10(max(rate, 1e-6)) + 3.0) / 2.0, 0.0, 1.0)


def _fallback_alpha(critical_pressure: float, failure_rate_per_window: float) -> float:
    risk = _risk_from_failure_rate(failure_rate_per_window)
    alpha = 0.10 + 0.55 * critical_pressure + 0.25 * risk
    if critical_pressure < 0.35:
        alpha = min(alpha, 0.18 + 0.12 * risk)
    return round(_clamp(alpha, 0.05, 0.90), 3)


def online_alpha_adapter_enabled(value: Any = None) -> bool:
    if value is not None:
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    return str(os.getenv(ENV_ONLINE_ALPHA_ADAPTER, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass
class OnlineAlphaAdapter:
    _baselines: dict[str, float] = field(default_factory=dict)
    _last_ratio: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def adapter_key(
        *,
        critical_agent: bool,
        decode_dominant: bool,
        stage_width: int,
    ) -> str:
        if not critical_agent:
            return "non_critical"
        if decode_dominant:
            return "long_decode_critical"
        if stage_width <= 1:
            return "singleton_critical"
        return "other_critical"

    def observe(
        self,
        *,
        critical_agent: bool,
        decode_dominant: bool,
        stage_width: int,
        latency_s: float,
    ) -> None:
        if latency_s <= 0.0:
            return
        key = self.adapter_key(
            critical_agent=critical_agent,
            decode_dominant=decode_dominant,
            stage_width=stage_width,
        )
        with self._lock:
            baseline = self._baselines.get(key)
            if baseline is None or latency_s < baseline:
                baseline = latency_s
                self._baselines[key] = baseline
            self._last_ratio[key] = latency_s / max(baseline, 1e-9)

    def wct_ratio(
        self,
        *,
        critical_agent: bool,
        decode_dominant: bool,
        stage_width: int,
    ) -> float:
        key = self.adapter_key(
            critical_agent=critical_agent,
            decode_dominant=decode_dominant,
            stage_width=stage_width,
        )
        with self._lock:
            return float(self._last_ratio.get(key, 1.0))

    def tune_decision(
        self,
        decision: AlphaPolicyDecision,
        *,
        critical_agent: bool,
        decode_dominant: bool,
        stage_width: int,
    ) -> AlphaPolicyDecision:
        if not critical_agent:
            return decision
        update = fine_tune_alpha(
            base_alpha=decision.alpha,
            observed_wct_ratio=self.wct_ratio(
                critical_agent=critical_agent,
                decode_dominant=decode_dominant,
                stage_width=stage_width,
            ),
            failure_rate_per_window=decision.failure_rate_per_window,
            hbm_pressure=decision.critical_pressure,
        )
        return AlphaPolicyDecision(
            alpha=update.alpha,
            critical_pressure=decision.critical_pressure,
            failure_rate_per_window=decision.failure_rate_per_window,
            source=update.source,
        )


def fine_tune_alpha(
    *,
    base_alpha: Any,
    observed_wct_ratio: Any = 1.0,
    failure_rate_per_window: Any = None,
    hbm_pressure: Any = 0.5,
    learning_rate: Any = 0.08,
    min_alpha: Any = 0.05,
    max_alpha: Any = 0.60,
) -> OnlineAlphaUpdate:
    """Apply a bounded online alpha update around the offline MDP action."""

    base = _clamp(_as_float(base_alpha, 0.30), 0.0, 1.0)
    wct_ratio = _clamp(_as_float(observed_wct_ratio, 1.0), 0.25, 4.0)
    rate = resolve_failure_rate_per_window(failure_rate_per_window)
    hbm = _clamp(_as_float(hbm_pressure, 0.5), 0.0, 1.0)
    lr = _clamp(_as_float(learning_rate, 0.08), 0.0, 1.0)
    lower = _clamp(_as_float(min_alpha, 0.05), 0.0, 1.0)
    upper = _clamp(_as_float(max_alpha, 0.60), lower, 1.0)

    risk = _risk_from_failure_rate(rate)
    latency_pressure = wct_ratio - 1.0
    pressure_signal = 0.65 * latency_pressure + 0.35 * (risk - hbm)
    delta = lr * pressure_signal
    updated = round(_clamp(base + delta, lower, upper), 3)
    return OnlineAlphaUpdate(
        alpha=updated,
        delta=round(updated - base, 3),
        source="online_alpha_adapter:v1",
    )


@lru_cache(maxsize=8)
def _load_policy(path: str) -> dict[str, Any]:
    policy_path = Path(path)
    if not policy_path.is_file():
        return {}
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _pressure_label(critical_pressure: float) -> str:
    if critical_pressure < 0.40:
        return "low"
    if critical_pressure < 0.75:
        return "medium"
    return "high"


def _state_distance(
    row: dict[str, Any],
    *,
    pressure_label: str,
    failure_rate_per_window: float,
) -> tuple[int, float]:
    same_pressure = 0 if str(row.get("critical_pressure")) == pressure_label else 1
    row_rate = resolve_failure_rate_per_window(row.get("failure_rate_per_window"))
    return (
        same_pressure,
        abs(math.log10(row_rate) - math.log10(failure_rate_per_window)),
    )


def _alpha_from_policy(
    policy: dict[str, Any],
    *,
    critical_pressure: float,
    failure_rate_per_window: float,
) -> float | None:
    states = policy.get("states")
    if not isinstance(states, list) or not states:
        return None
    pressure_label = _pressure_label(critical_pressure)
    candidates = [row for row in states if isinstance(row, dict)]
    if not candidates:
        return None
    best = min(
        candidates,
        key=lambda row: _state_distance(
            row,
            pressure_label=pressure_label,
            failure_rate_per_window=failure_rate_per_window,
        ),
    )
    if "alpha" not in best:
        return None
    return round(_clamp(_as_float(best["alpha"], -1.0), 0.0, 1.0), 3)


def choose_alpha(
    *,
    critical_agent: bool,
    decode_dominant: bool,
    stage_width: int,
    critical_pressure: Any = None,
    failure_rate_per_window: Any = None,
    policy_path: str | None = None,
) -> AlphaPolicyDecision:
    if critical_pressure is None:
        resolved_pressure = pressure_from_request(
            critical_agent=critical_agent,
            decode_dominant=decode_dominant,
            stage_width=stage_width,
        )
    else:
        parsed_pressure = _as_float(critical_pressure, -1.0)
        if parsed_pressure < 0.0:
            resolved_pressure = pressure_from_request(
                critical_agent=critical_agent,
                decode_dominant=decode_dominant,
                stage_width=stage_width,
            )
        else:
            resolved_pressure = _clamp(parsed_pressure, 0.0, 1.0)
    failure_rate = resolve_failure_rate_per_window(failure_rate_per_window)
    selected_policy = policy_path if policy_path is not None else os.getenv(ENV_ALPHA_POLICY, "")
    if selected_policy:
        policy = _load_policy(selected_policy)
        alpha = _alpha_from_policy(
            policy,
            critical_pressure=resolved_pressure,
            failure_rate_per_window=failure_rate,
        )
        if alpha is not None:
            return AlphaPolicyDecision(
                alpha=alpha,
                critical_pressure=round(resolved_pressure, 3),
                failure_rate_per_window=failure_rate,
                source=f"policy_table:{selected_policy}",
            )
    return AlphaPolicyDecision(
        alpha=_fallback_alpha(resolved_pressure, failure_rate),
        critical_pressure=round(resolved_pressure, 3),
        failure_rate_per_window=failure_rate,
        source="fallback_formula",
    )
