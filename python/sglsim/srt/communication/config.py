from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_KV_BYTES_PER_TOKEN = 256
DEFAULT_H2D_ALPHA_S = 6.67e-6
DEFAULT_D2H_ALPHA_S = 4.0e-6
DEFAULT_DISK_WRITE_ALPHA_S = 30.0e-6
DEFAULT_DISK_READ_ALPHA_S = 30.0e-6
DEFAULT_EFFICIENCY = 0.85


@dataclass(frozen=True)
class AlphaBetaPathConfig:
    alpha_s: float
    efficiency: float = DEFAULT_EFFICIENCY


@dataclass(frozen=True)
class CommunicationConfig:
    mode: str = "alpha_beta"
    kv_bytes_per_token: int | None = None
    alpha_beta: dict[str, AlphaBetaPathConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> CommunicationConfig:
        raw = raw if isinstance(raw, dict) else {}
        mode = str(raw.get("mode", "alpha_beta"))
        kv_override = raw.get("kv_bytes_per_token")
        kv_bytes_per_token = int(kv_override) if kv_override is not None else None

        alpha_raw = raw.get("alpha_beta", {})
        alpha_beta: dict[str, AlphaBetaPathConfig] = {}
        if isinstance(alpha_raw, dict):
            for path_key, path_cfg in alpha_raw.items():
                if not isinstance(path_cfg, dict):
                    continue
                alpha_us = path_cfg.get("alpha_us")
                alpha_s = (
                    float(alpha_us) / 1e6
                    if alpha_us is not None
                    else _default_alpha_s(str(path_key))
                )
                efficiency = float(path_cfg.get("efficiency", DEFAULT_EFFICIENCY))
                alpha_beta[str(path_key)] = AlphaBetaPathConfig(
                    alpha_s=alpha_s,
                    efficiency=efficiency,
                )

        return cls(
            mode=mode,
            kv_bytes_per_token=kv_bytes_per_token,
            alpha_beta=alpha_beta,
        )

    def alpha_for(self, path_key: str) -> AlphaBetaPathConfig:
        if path_key in self.alpha_beta:
            return self.alpha_beta[path_key]
        return AlphaBetaPathConfig(alpha_s=_default_alpha_s(path_key))


def _default_alpha_s(path_key: str) -> float:
    if path_key == "h2d":
        return DEFAULT_H2D_ALPHA_S
    if path_key == "d2h":
        return DEFAULT_D2H_ALPHA_S
    if path_key == "disk_write":
        return DEFAULT_DISK_WRITE_ALPHA_S
    if path_key == "disk_read":
        return DEFAULT_DISK_READ_ALPHA_S
    return 0.0
