from __future__ import annotations

import random
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_MOCK_MODEL_PATHS = frozenset({"mock-model", "mock", ""})


class PromptBuilder:
    """Build deterministic prompts with exact tokenizer token length."""

    def __init__(self, model_path: str, target_input_len: int, seed: int) -> None:
        self.model_path = str(model_path)
        self.target_input_len = max(int(target_input_len), 1)
        self.seed = int(seed)
        self._mock = self.model_path in _MOCK_MODEL_PATHS
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._lock = threading.Lock()

    def _tokenizer_or_raise(self) -> PreTrainedTokenizerBase:
        if self._mock:
            raise RuntimeError("mock model path does not use a HuggingFace tokenizer")
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                try:
                    from transformers.models.auto.tokenization_auto import (
                        AutoTokenizer,
                    )
                except ImportError as exc:
                    raise ImportError(
                        f"HuggingFace tokenizer required for model_path={self.model_path!r}. "
                        "Install with: pip install transformers torch "
                        "or uv sync --extra benchmark"
                    ) from exc

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=True,
            )
        return self._tokenizer

    def _token_ids_from_text(self, text: str) -> list[int]:
        tokenizer = self._tokenizer_or_raise()
        with self._lock:
            return tokenizer.encode(text, add_special_tokens=False)

    def build(self, request_index: int) -> str:
        if self._mock:
            return " ".join(f"tok{(self.seed + request_index + i) % 100000:05d}" for i in range(self.target_input_len))

        mixed_seed = (self.seed * 1000003 + int(request_index)) & 0xFFFFFFFF
        rng = random.Random(mixed_seed)
        words = [f"req{int(request_index):05d}"]
        text = " ".join(words)
        token_ids = self._token_ids_from_text(text)
        while len(token_ids) < self.target_input_len:
            words.append(f"tok{rng.randrange(0, 100000):05d}")
            text = " ".join(words)
            token_ids = self._token_ids_from_text(text)

        if len(token_ids) > self.target_input_len:
            token_ids = token_ids[: self.target_input_len]
            tokenizer = self._tokenizer_or_raise()
            with self._lock:
                text = tokenizer.decode(
                    token_ids,
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=False,
                )
            token_ids = self._token_ids_from_text(text)

        while len(token_ids) < self.target_input_len:
            text = text + " x"
            token_ids = self._token_ids_from_text(text)
        if len(token_ids) > self.target_input_len:
            token_ids = token_ids[: self.target_input_len]
            tokenizer = self._tokenizer_or_raise()
            with self._lock:
                text = tokenizer.decode(
                    token_ids,
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=False,
                )
            token_ids = self._token_ids_from_text(text)

        if len(token_ids) != self.target_input_len:
            raise RuntimeError(
                f"Failed to build exact-length prompt: got {len(token_ids)}, "
                f"want {self.target_input_len}"
            )
        return text

    def build_token_ids(self, request_index: int) -> list[int]:
        if self._mock:
            base = (self.seed * 1000003 + int(request_index)) & 0xFFFFFFFF
            return [(base + i) % 32000 for i in range(self.target_input_len)]
        return self._token_ids_from_text(self.build(request_index))
