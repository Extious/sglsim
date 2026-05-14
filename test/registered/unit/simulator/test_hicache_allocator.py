import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.simulator.allocator import alloc_decode_cpu, alloc_extend_cpu
from sglang.srt.simulator.hicache import SimulatorHiCacheStorage


class TestSimulatorHiCacheStorage(unittest.TestCase):
    def test_storage_set_exists_clear(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = str(Path(tmp_dir) / "keys.txt")
            storage = SimulatorHiCacheStorage(storage_file_path=path)

            self.assertFalse(storage.exists("a"))
            self.assertTrue(storage.set("a"))
            self.assertTrue(storage.exists("a"))
            self.assertEqual(storage.batch_exists(["a", "b"], extra_info=None), 1)
            self.assertTrue(storage.clear())
            self.assertFalse(storage.exists("a"))


class TestSimulatorAllocator(unittest.TestCase):
    def test_alloc_decode_cpu_reuses_current_page(self):
        out = torch.empty((1,), dtype=torch.int64)
        alloc_decode_cpu(
            torch.tensor([2]),
            torch.tensor([0]),
            torch.tensor([], dtype=torch.int64),
            out,
            1,
            4,
        )
        self.assertEqual(out.item(), 1)

    def test_alloc_extend_cpu_allocates_new_page(self):
        out = torch.empty((4,), dtype=torch.int64)
        alloc_extend_cpu(
            torch.tensor([0]),
            torch.tensor([4]),
            torch.tensor([-1]),
            torch.tensor([3]),
            out,
            1,
            4,
        )
        self.assertEqual(out.tolist(), [12, 13, 14, 15])


if __name__ == "__main__":
    unittest.main()
