import time
from concurrent.futures import ThreadPoolExecutor

from sglsim.srt.utils import tokenizer_runtime


class _SleepyTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        time.sleep(0.05)
        return [len(text), int(add_special_tokens)]


def test_encode_text_allows_parallel_fast_tokenizer_calls(monkeypatch):
    model_path = "real-tokenizer"
    monkeypatch.setitem(tokenizer_runtime._TOKENIZERS, model_path, _SleepyTokenizer())

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(lambda _: tokenizer_runtime.encode_text(model_path, "abc"), range(5)))
    elapsed = time.perf_counter() - start

    assert results == [[3, 0]] * 5
    assert elapsed < 0.15
