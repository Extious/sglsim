from __future__ import annotations

import ast
from pathlib import Path


FORBIDDEN_PREFIXES = ("sglang", "torch.cuda", "triton")

PROTECTED_ROOTS = (Path("benchmark/synthesis_requests"),)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def test_benchmark_do_not_import_upstream_sglang():
    offenders: list[tuple[str, str]] = []
    for root in PROTECTED_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            for name in _imports(path):
                if name.startswith(FORBIDDEN_PREFIXES):
                    offenders.append((str(path), name))
    assert offenders == []


def test_srt_does_not_import_upstream_sglang():
    root = Path("python/sglsim/srt")
    offenders: list[tuple[str, str]] = []
    if not root.exists():
        return
    for path in root.rglob("*.py"):
        for name in _imports(path):
            if name.startswith("sglang"):
                offenders.append((str(path), name))
    assert offenders == []
