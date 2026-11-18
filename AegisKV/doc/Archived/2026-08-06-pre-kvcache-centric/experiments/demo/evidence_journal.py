"""Per-run evidence-journal binding for offline experiment entrypoints."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


ENV_EVENT_JOURNAL_PATH = "SGLSIM_AEGIS_EVENT_JOURNAL_PATH"


def event_journal_path_for_artifact(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.events.jsonl")


@contextmanager
def bind_fresh_event_journal(path: Path) -> Iterator[Path]:
    """Bind one unique append-only journal and restore the caller environment."""

    resolved = path.expanduser().resolve()
    if resolved.exists() and resolved.stat().st_size > 0:
        raise FileExistsError(
            "refusing to mix experiment sessions in an existing event journal: "
            f"{resolved}"
        )
    previous = os.environ.get(ENV_EVENT_JOURNAL_PATH)
    os.environ[ENV_EVENT_JOURNAL_PATH] = str(resolved)
    try:
        yield resolved
    finally:
        if previous is None:
            os.environ.pop(ENV_EVENT_JOURNAL_PATH, None)
        else:
            os.environ[ENV_EVENT_JOURNAL_PATH] = previous

