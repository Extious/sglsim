from sglsim.srt.mem_cache.hicache import HiCacheController
from sglsim.srt.mem_cache.memory_pool import DeviceKVPool
from sglsim.srt.mem_cache.memory_pool_host import HostKVPool


def test_write_through_enqueues_l2_backup():
    device = DeviceKVPool(capacity_pages=4, page_size=1)
    host = HostKVPool(capacity_pages=4, page_size=1)
    controller = HiCacheController(device, host, write_policy="write_through", page_size=1)

    host_ids, dur = controller.write([0, 1], node_id=1, owner="r0")

    assert host_ids == [0, 1]
    assert dur > 0
    assert len(controller.write_queue) == 1
    assert controller.pending_write_pages == 2
    assert controller.pending_backup_dur > 0
    assert controller.pop_write_pages_since_check() == 2
    assert host.available_size == 2


def test_load_enqueues_h2d_operation():
    device = DeviceKVPool(capacity_pages=4, page_size=1)
    host = HostKVPool(capacity_pages=4, page_size=1)
    controller = HiCacheController(device, host, write_policy="write_through", page_size=1)
    host.alloc(2, owner="r0")

    device_ids = controller.load([0, 1], node_id=2, owner="r1")

    assert device_ids == [0, 1]
    moved, _ = controller.drain_load_queue()
    assert moved == 2


def test_write_skipped_when_hierarchical_disabled():
    device = DeviceKVPool(capacity_pages=4, page_size=1)
    controller = HiCacheController(device, None, write_policy="write_through", page_size=1)

    result = controller.write([0], node_id=1, owner="r0")
    assert result == (None, 0.0)
