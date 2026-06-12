"""Path configuration for the fixed online SGLSim benchmark."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _find_repo_root(start: Optional[Path] = None) -> Path:
    cur = (start or Path(__file__)).resolve().parent
    for _ in range(10):
        if (cur / ".git").exists():
            return cur
        if cur == cur.parent:
            break
        cur = cur.parent
    raise RuntimeError(
        "Cannot locate sglsim repo root. "
        "Set SGLSIM_REPO_ROOT or provide sglsim_root in env.yml."
    )


def _resolve(base: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base / path).resolve()


def _load_env_yml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}

    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]

        data = yaml.safe_load(text) or {}
        return data if isinstance(data, dict) else {}
    except ModuleNotFoundError:
        parsed: dict[str, str] = {}
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip().strip("'\"")
        return parsed


@dataclass
class PathConfig:
    repo_root: Path
    online_root: Path
    src_root: Path
    venv: Path
    log_dir: Path
    output_dir: Path

    @classmethod
    def from_env_yml(
        cls,
        env_yml_path: Optional[Path] = None,
        *,
        log_dir: str = "",
        output_dir: str = "",
    ) -> "PathConfig":
        if env_yml_path is None:
            env_yml_path = Path(__file__).resolve().parent / "config" / "env.yml"

        env = _load_env_yml(env_yml_path)

        root_raw = env.get("sglsim_root") or os.environ.get("SGLSIM_REPO_ROOT")
        repo_root = Path(root_raw).resolve() if root_raw else _find_repo_root()
        online_root = repo_root / "benchmark" / "online"
        src_root = online_root / "src"

        venv_rel = env.get("venv", ".venv")
        venv = _resolve(repo_root, venv_rel)

        logs = log_dir or env.get("exp_log_dir", "logs")
        output = output_dir or env.get("exp_output_dir", "results")

        return cls(
            repo_root=repo_root,
            online_root=online_root,
            src_root=src_root,
            venv=venv,
            log_dir=_resolve(online_root, logs),
            output_dir=_resolve(online_root, output),
        )

    @property
    def config_dir(self) -> Path:
        return self.src_root / "config"
