from __future__ import annotations

import logging
import hashlib
import threading
import time
from functools import lru_cache
from typing import TYPE_CHECKING

from sglsim.srt.utils.prompt_builder import _MOCK_MODEL_PATHS

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_LOCK = threading.Lock()
_TOKENIZERS: dict[str, PreTrainedTokenizerBase | None] = {}


def is_mock_model(model_path: str) -> bool:
    return str(model_path) in _MOCK_MODEL_PATHS


def _mock_encode(text: str, seed: int = 0) -> list[int]:
    if not text:
        return [0]
    ids: list[int] = []
    for i in range(0, len(text), 4):
        chunk = text[i : i + 4]
        value = seed
        for ch in chunk:
            value = (value * 131 + ord(ch)) & 0xFFFFFFFF
        ids.append(value % 32000)
    return ids


def ensure_loaded(
    model_path: str,
    *,
    logger: logging.Logger | None = None,
) -> PreTrainedTokenizerBase | None:
    path = str(model_path)
    if is_mock_model(path):
        return None
    with _LOCK:
        if path in _TOKENIZERS:
            return _TOKENIZERS[path]
        log = logger or logging.getLogger(__name__)
        t0 = time.perf_counter()
        log.info("Loading tokenizer for model_path=%s", path)
        try:
            from transformers import AutoTokenizer
        except ImportError:
            from transformers.models.auto.tokenization_auto import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            use_fast=True,
        )
        _TOKENIZERS[path] = tokenizer
        log.info(
            "Tokenizer ready for model_path=%s in %.3fs",
            path,
            time.perf_counter() - t0,
        )
        return tokenizer


def format_chat_messages(
    model_path: str,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Render OpenAI-style messages to a single prompt string (chat template or mock)."""
    if is_mock_model(model_path):
        parts: list[str] = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", ""))
            parts.append(f"<|{role}|>\n{content}")
        text = "\n".join(parts)
        if add_generation_prompt:
            text = f"{text}\n<|assistant|>\n"
        return text
    tokenizer = ensure_loaded(model_path)
    assert tokenizer is not None
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def encode_text(model_path: str, text: str) -> list[int]:
    if is_mock_model(model_path):
        return _mock_encode(text)
    tokenizer = ensure_loaded(model_path)
    assert tokenizer is not None
    return tokenizer.encode(text, add_special_tokens=False)


@lru_cache(maxsize=32)
def synthetic_assistant_unit(model_path: str) -> tuple[str, int]:
    """Return a cheap one-token text unit for metadata-only generation.

    The fast detokenizer does not run model-token detokenization, but multi-turn
    chat clients still feed assistant text back into the next prompt.  The text
    returned here must tokenize to exactly one token and repeat without BPE
    merging so the scheduler can append the same token id to radix cache.
    """
    for unit in (" a", " x", " A", " hello"):
        ids = encode_text(model_path, unit)
        repeated = encode_text(model_path, unit * 4)
        if len(ids) == 1 and repeated == ids * 4:
            return unit, int(ids[0])
    ids = encode_text(model_path, " a")
    token_id = int(ids[0]) if ids else 0
    return " a", token_id


@lru_cache(maxsize=32)
def _synthetic_assistant_unit_candidates(model_path: str) -> tuple[tuple[str, int], ...]:
    if not is_mock_model(model_path):
        tokenizer = ensure_loaded(model_path)
        assert tokenizer is not None
        vocab = tokenizer.get_vocab()
        by_id = sorted((int(token_id), token) for token, token_id in vocab.items())
        scanned: list[tuple[str, int]] = []
        for token_id, _token in by_id:
            if token_id in set(getattr(tokenizer, "all_special_ids", []) or []):
                continue
            text = tokenizer.decode([token_id], skip_special_tokens=True)
            if (
                not text
                or len(text) > 16
                or text.strip() == ""
                or any(ch in text for ch in "\r\n\t")
                or any(ord(ch) < 32 for ch in text)
            ):
                continue
            ids = encode_text(model_path, text)
            if ids != [token_id]:
                continue
            if encode_text(model_path, text * 4) != [token_id] * 4:
                continue
            scanned.append((text, token_id))
            if len(scanned) >= 4096:
                break
        if scanned:
            return tuple(scanned)

    units = [
        *(f" {ch}" for ch in "abcdefghijklmnopqrstuvwxyz"),
        *(f" {ch}" for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        *(f" {n}" for n in range(10)),
        " alpha",
        " beta",
        " gamma",
        " delta",
        " red",
        " blue",
        " green",
        " north",
        " south",
        " east",
        " west",
    ]
    candidates: list[tuple[str, int]] = []
    seen: set[int] = set()
    for unit in units:
        ids = encode_text(model_path, unit)
        repeated = encode_text(model_path, unit * 4)
        if len(ids) != 1 or repeated != ids * 4:
            continue
        token_id = int(ids[0])
        if token_id in seen:
            continue
        candidates.append((unit, token_id))
        seen.add(token_id)
    if candidates:
        return tuple(candidates)
    return (synthetic_assistant_unit(model_path),)


def synthetic_assistant_unit_for_key(model_path: str, key: str | None) -> tuple[str, int]:
    """Return a stable one-token text unit scoped by request key.

    Using one global synthetic output makes unrelated closed-loop conversations
    share assistant turns in radix cache, which real SGLang generations do not
    generally do.  The selected unit still tokenizes as one repeatable token so
    generated-token counts remain stable.
    """
    candidates = _synthetic_assistant_unit_candidates(model_path)
    if not key:
        return candidates[0]
    digest = hashlib.blake2b(str(key).encode("utf-8"), digest_size=8).digest()
    idx = int.from_bytes(digest, "big") % len(candidates)
    return candidates[idx]


def decode_ids(model_path: str, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    if is_mock_model(model_path):
        return ""
    tokenizer = ensure_loaded(model_path)
    assert tokenizer is not None
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def measure_encode(model_path: str, text: str) -> tuple[list[int], float]:
    t0 = time.perf_counter()
    ids = encode_text(model_path, text)
    return ids, time.perf_counter() - t0


def measure_decode(model_path: str, token_ids: list[int]) -> tuple[str, float]:
    t0 = time.perf_counter()
    text = decode_ids(model_path, token_ids)
    return text, time.perf_counter() - t0


def estimate_decode_dur(token_ids: list[int]) -> float:
    """Cheap analytic detokenization-time estimate (no real tokenizer).

    Returns 0.0 by default; kept as a hook for a per-token analytic constant
    so the simulation can model detok latency without running transformers.
    """
    return 0.0
