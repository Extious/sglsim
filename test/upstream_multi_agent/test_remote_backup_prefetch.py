from __future__ import annotations

import sys
import types
import unittest
from importlib.util import find_spec
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace


PYTHON_ROOT = Path(__file__).resolve().parents[2] / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

if "tqdm" not in sys.modules and find_spec("tqdm") is None:
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.__spec__ = ModuleSpec("tqdm", loader=None)
    tqdm_stub.__path__ = []
    tqdm_stub.tqdm = lambda iterable=None, *args, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_stub
    tqdm_auto_stub = types.ModuleType("tqdm.auto")
    tqdm_auto_stub.__spec__ = ModuleSpec("tqdm.auto", loader=None)
    tqdm_auto_stub.tqdm = tqdm_stub.tqdm
    sys.modules["tqdm.auto"] = tqdm_auto_stub


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _install_cache_controller_dependency_stubs() -> None:
    return


class _StorageBackend:
    def match_prefix_from(self, full_tokens, prefix_pages, lookup_dp_rank=None):
        self.full_tokens = list(full_tokens)
        self.prefix_pages = prefix_pages
        self.lookup_dp_rank = lookup_dp_rank
        return 3, 5


class RemoteBackupPrefetchTest(unittest.TestCase):
    def test_token_prefetch_expected_tokens_use_contiguous_data_pages(self):
        _install_cache_controller_dependency_stubs()
        from sglsim.srt.managers.cache_controller import HiCacheController

        controller = HiCacheController.__new__(HiCacheController)
        controller.page_size = 1
        controller.storage_backend = _StorageBackend()
        controller.get_hash_str = lambda tokens, last_hash: f"{last_hash}:{tokens[0]}"

        operation = SimpleNamespace(
            request_id="rid",
            token_ids=[10, 11, 12, 13, 14, 15],
            prefix_token_ids=[1],
            last_hash="h",
            lookup_dp_rank=1,
        )

        hash_value, storage_query_count = controller._storage_hit_query_tokens(
            operation
        )

        self.assertEqual(len(hash_value), 3)
        self.assertEqual(storage_query_count, 3)
        self.assertEqual(operation.full_token_ids, [1, 10, 11, 12, 13, 14, 15])
        self.assertEqual(operation.page_start, 1)


if __name__ == "__main__":
    unittest.main()
