from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sglsim.srt.utils.prompt_builder import _MOCK_MODEL_PATHS

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class SynthesisWorkloadConfig:
    model_path: str
    prompt_path: Path
    seed: int
    request_marker_template: str = "request {rid}/"


class SynthesisPromptBuilder:
    """Build token ids from a shared prompt file plus per-request marker prefix."""

    def __init__(self, config: SynthesisWorkloadConfig) -> None:
        self.config = config
        self._mock = config.model_path in _MOCK_MODEL_PATHS
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._lock = threading.Lock()
        self._body_text: str | None = None
        self._body_token_ids: list[int] | None = None

    def _tokenizer_or_raise(self) -> PreTrainedTokenizerBase:
        if self._mock:
            raise RuntimeError("mock model path does not use a HuggingFace tokenizer")
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                from transformers.models.auto.tokenization_auto import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                use_fast=True,
            )
        return self._tokenizer

    def _encode(self, text: str) -> list[int]:
        if self._mock:
            return self._mock_encode(text)
        tokenizer = self._tokenizer_or_raise()
        with self._lock:
            return tokenizer.encode(text, add_special_tokens=False)

    def _mock_encode(self, text: str) -> list[int]:
        if not text:
            return [0]
        ids: list[int] = []
        for i in range(0, len(text), 4):
            chunk = text[i : i + 4]
            value = self.config.seed
            for ch in chunk:
                value = (value * 131 + ord(ch)) & 0xFFFFFFFF
            ids.append(value % 32000)
        return ids

    def _load_body_text(self) -> str:
        if self._body_text is None:
            path = self.config.prompt_path
            self._body_text = path.read_text(encoding="utf-8")
        return self._body_text

    def body_token_ids(self) -> list[int]:
        if self._body_token_ids is None:
            self._body_token_ids = self._encode(self._load_body_text())
        return list(self._body_token_ids)

    def body_token_count(self) -> int:
        return len(self.body_token_ids())

    def _marker_text(self, rid: str) -> str:
        return self.config.request_marker_template.format(rid=rid)

    def build(self, rid: str, request_index: int) -> str:
        del request_index
        return self._marker_text(rid) + self._load_body_text()

    def build_token_ids(self, rid: str, request_index: int) -> list[int]:
        del request_index
        marker_ids = self._encode(self._marker_text(rid))
        return marker_ids + self.body_token_ids()
