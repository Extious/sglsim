from __future__ import annotations

import argparse
import csv
import json
import ssl
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable


LEGACY_HF_DATASET = "nlile/misc-merged-claude-code-traces-v1"
DEFAULT_HF_DATASET = LEGACY_HF_DATASET
DEFAULT_TRACE_SOURCE = "legacy"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "traces"
    / "claude_code_trace_workload.jsonl"
)


def _text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("content"):
                    parts.append(_text_from_content(item.get("content")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if content.get("text"):
            return str(content["text"])
        if content.get("content"):
            return _text_from_content(content.get("content"))
    return str(content)


def _parse_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str) and value.strip():
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [{"role": "user", "content": value}]
    if not isinstance(value, list):
        return []
    out: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user")
        content = _text_from_content(item.get("content"))
        if content:
            out.append({"role": role, "content": content})
    return out


def _assistant_output_len(row: dict[str, Any], messages: list[dict[str, str]]) -> int:
    texts: list[str] = []
    if row.get("assistant_response"):
        texts.append(str(row["assistant_response"]))
    texts.extend(msg["content"] for msg in messages if msg["role"] == "assistant")
    if not texts:
        return 128
    approx_tokens = max(len(text.split()) for text in texts)
    return max(16, min(1024, approx_tokens))


def row_to_workload(row: dict[str, Any], index: int) -> dict[str, object] | None:
    messages = _parse_messages(row.get("messages_json"))
    if not messages:
        user_prompt = str(row.get("user_prompt") or "").strip()
        assistant = str(row.get("assistant_response") or "").strip()
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    user_turns = [msg["content"] for msg in messages if msg["role"] == "user"]
    if not user_turns:
        return None

    system_prompt = str(row.get("system_prompt") or "").strip()
    if system_prompt:
        user_turns[0] = f"{system_prompt}\n\n{user_turns[0]}"
    rid = str(row.get("request_id") or row.get("id") or f"claude-trace-{index}")
    repo = str(row.get("source_repo") or row.get("source_table") or "claude-code")
    has_tools = bool(str(row.get("tools_json") or "").strip() not in {"", "null", "None"})
    return {
        "rid": rid,
        "prompt": user_turns if len(user_turns) > 1 else user_turns[0],
        "turn_prompts": user_turns,
        "output_len": _assistant_output_len(row, messages),
        "group_id": index,
        "agent_id": f"claude-code-{index}",
        "workflow_id": f"claude-code-trace-{index}",
        "stage_id": 0,
        "stage_width": 1,
        "stage_local_agent_index": 0,
        "critical_agent": True,
        "decode_dominant": has_tools,
        "workflow_shape": "real-claude-code-trace",
        "source_repo": repo,
        "source_table": str(row.get("source_table") or ""),
        "source_dataset": LEGACY_HF_DATASET,
        "model": str(row.get("model") or ""),
        "timestamp": str(row.get("timestamp") or ""),
        "source": LEGACY_HF_DATASET,
    }


def _open_json_url(url: str) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, timeout=60, context=context) as response:
            return json.loads(response.read().decode("utf-8"))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _iter_csv(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def _iter_hf(dataset_name: str, *, split: str, limit: int | None) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        yield from _iter_hf_rows_api(dataset_name, split=split, limit=limit)
        return
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    for idx, row in enumerate(dataset):
        if limit is not None and idx >= limit:
            break
        yield dict(row)


def _iter_hf_rows_api(
    dataset_name: str,
    *,
    split: str,
    limit: int | None,
) -> Iterable[dict[str, Any]]:
    remaining = 100 if limit is None else max(int(limit), 0)
    offset = 0
    page_size = min(max(remaining, 1), 100)
    while remaining > 0:
        query = urllib.parse.urlencode(
            {
                "dataset": dataset_name,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": min(page_size, remaining),
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(url, timeout=60, context=context) as response:
                payload = json.loads(response.read().decode("utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else []
        if not rows:
            break
        for item in rows:
            row = item.get("row") if isinstance(item, dict) else None
            if isinstance(row, dict):
                yield row
                remaining -= 1
                if remaining <= 0:
                    return
        offset += len(rows)


def _parse_offsets(value: str) -> list[int]:
    offsets: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        offsets.append(int(item))
    return offsets


def _iter_hf_offsets_api(
    dataset_name: str,
    *,
    split: str,
    offsets: list[int],
    page_size: int,
    limit: int | None,
) -> Iterable[dict[str, Any]]:
    emitted = 0
    context = ssl._create_unverified_context()
    for offset in offsets:
        if limit is not None and emitted >= limit:
            return
        remaining = page_size if limit is None else min(page_size, limit - emitted)
        if remaining <= 0:
            return
        query = urllib.parse.urlencode(
            {
                "dataset": dataset_name,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": remaining,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            with urllib.request.urlopen(url, timeout=60, context=context) as response:
                payload = json.loads(response.read().decode("utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else []
        for item in rows:
            row = item.get("row") if isinstance(item, dict) else None
            if isinstance(row, dict):
                yield row
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def _prompt_word_count(row: dict[str, object]) -> int:
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        return sum(len(str(item).split()) for item in prompt)
    return len(str(prompt or "").split())


def build_workload(
    rows: Iterable[dict[str, Any]],
    *,
    limit: int | None = None,
    max_input_words: int | None = None,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if limit is not None and len(out) >= limit:
            break
        converted = row_to_workload(row, index)
        if converted is not None:
            if (
                max_input_words is not None
                and _prompt_word_count(converted) > max_input_words
            ):
                continue
            out.append(converted)
    return out


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Claude Code traces into a SGLSim trace workload JSONL."
    )
    parser.add_argument(
        "--source",
        choices=("legacy",),
        default=DEFAULT_TRACE_SOURCE,
        help="Trace adapter to use.",
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--hf-dataset", default="")
    parser.add_argument(
        "--hf-offsets",
        default="",
        help=(
            "Comma-separated Hugging Face row offsets. When set, rows are fetched "
            "from each offset so a small workload can cover multiple trace source tables."
        ),
    )
    parser.add_argument("--hf-page-size", type=int, default=16)
    parser.add_argument(
        "--max-input-words",
        type=int,
        default=None,
        help=(
            "Skip rows whose user-turn text exceeds this word count. This keeps "
            "a trace slice serveable under the current simulator KV/token limits."
        ),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    hf_dataset = args.hf_dataset or LEGACY_HF_DATASET
    offsets = _parse_offsets(args.hf_offsets)
    legacy_limit = 32 if args.limit is None else args.limit
    raw_limit = None if args.max_input_words is not None else legacy_limit
    if offsets:
        rows = _iter_hf_offsets_api(
            hf_dataset,
            split=args.split,
            offsets=offsets,
            page_size=args.hf_page_size,
            limit=raw_limit,
        )
    elif args.hf_dataset:
        rows = _iter_hf(args.hf_dataset, split=args.split, limit=raw_limit)
    elif args.input:
        if args.input.suffix.lower() == ".csv":
            rows = _iter_csv(args.input)
        else:
            rows = _iter_jsonl(args.input)
    else:
        rows = _iter_hf(LEGACY_HF_DATASET, split=args.split, limit=legacy_limit)
    workload = build_workload(
        rows,
        limit=legacy_limit,
        max_input_words=args.max_input_words,
    )
    write_jsonl(args.output, workload)
    print(json.dumps({"output": str(args.output), "rows": len(workload)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
