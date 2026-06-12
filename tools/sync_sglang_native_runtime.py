from __future__ import annotations

import shutil
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = Path("/home/comp/zhanzhao/sglang/python/sglang")
LOCAL_ROOT = REPO / "python" / "sglsim"

EXCLUDE_SUFFIXES = {".pyc", ".so"}
EXCLUDE_DIR_NAMES = {
    "__pycache__",
    "fault_tolerance",
}
PROTECTED_LOCAL_DIRS = {
    Path("fault_tolerance"),
    Path("srt/simulator"),
}
PROTECTED_LOCAL_FILES = {
    Path("__init__.py"),
    Path("launch_server.py"),
    Path("bench_serving.py"),
}


def rewrite_imports(text: str) -> str:
    text = re.sub(r"(?m)^(\s*)import sglang as (\w+)\s*$", r"\1import sglsim as \2", text)
    text = re.sub(r"(?m)^(\s*)import sglang\s*$", r"\1import sglsim as sglang", text)
    text = text.replace("from sglang.", "from sglsim.")
    text = text.replace("import sglang.", "import sglsim.")
    text = text.replace("sglang.srt.", "sglsim.srt.")
    text = text.replace("sglang.lang.", "sglsim.lang.")
    return text


def should_copy(path: Path) -> bool:
    if path.name in EXCLUDE_DIR_NAMES:
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    return path.suffix == ".py"


def should_copy_resource(path: Path) -> bool:
    if path.name in EXCLUDE_DIR_NAMES:
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    return path.is_file()


def is_protected_rel(rel: Path) -> bool:
    if rel in PROTECTED_LOCAL_FILES:
        return True
    return any(rel == protected or protected in rel.parents for protected in PROTECTED_LOCAL_DIRS)


def copy_file(rel: str) -> None:
    rel_path = Path(rel)
    if is_protected_rel(rel_path):
        return
    src = UPSTREAM_ROOT / rel
    dst = LOCAL_ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(rewrite_imports(src.read_text()), encoding="utf-8")


def copy_resource_file(rel: Path) -> None:
    if is_protected_rel(rel):
        return
    src = UPSTREAM_ROOT / rel
    dst = LOCAL_ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_resource_dir(rel_dir: str) -> None:
    src_dir = UPSTREAM_ROOT / rel_dir
    if not src_dir.exists():
        return
    for src in src_dir.rglob("*"):
        if src.is_dir() or not should_copy_resource(src):
            continue
        rel = src.relative_to(UPSTREAM_ROOT)
        if any(part in EXCLUDE_DIR_NAMES for part in rel.parts):
            continue
        copy_resource_file(rel)


def clear_unprotected_generated_files() -> None:
    for path in LOCAL_ROOT.rglob("*.py"):
        rel = path.relative_to(LOCAL_ROOT)
        if is_protected_rel(rel):
            continue
        path.unlink()

    for path in sorted(LOCAL_ROOT.rglob("*"), reverse=True):
        if not path.is_dir():
            continue
        rel = path.relative_to(LOCAL_ROOT)
        if is_protected_rel(rel):
            continue
        try:
            path.rmdir()
        except OSError:
            pass


def main() -> int:
    clear_unprotected_generated_files()
    for src in UPSTREAM_ROOT.rglob("*"):
        if src.is_dir() or not should_copy(src):
            continue
        if any(part in EXCLUDE_DIR_NAMES for part in src.relative_to(UPSTREAM_ROOT).parts):
            continue
        copy_file(str(src.relative_to(UPSTREAM_ROOT)))
    copy_resource_dir("jit_kernel/include")
    copy_resource_dir("jit_kernel/csrc")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
