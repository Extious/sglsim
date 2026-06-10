from sglsim.srt.mem_cache.hicache import HiCacheController
from sglsim.srt.mem_cache.memory_pool import DeviceKVPool
from sglsim.srt.mem_cache.memory_pool_host import HostKVPool


def test_overlap_consumes_pending_backup():
    device = DeviceKVPool(capacity_pages=4, page_size=1)
    host = HostKVPool(capacity_pages=4, page_size=1)
    controller = HiCacheController(device, host, write_policy="write_through", page_size=1)
    controller.write([0, 1], node_id=1, owner="r0")
    pending = controller.pending_backup_dur
    tail = controller.consume_backup_overlap(forward_dur=pending * 0.5)
    assert controller.pending_backup_dur == pending * 0.5
    assert tail == pending * 0.5


def test_serial_consumes_all_pending_backup():
    device = DeviceKVPool(capacity_pages=4, page_size=1)
    host = HostKVPool(capacity_pages=4, page_size=1)
    controller = HiCacheController(device, host, write_policy="write_through", page_size=1)
    controller.write([0], node_id=1, owner="r0")
    dur = controller.consume_backup_serial()
    assert dur > 0
    assert controller.pending_backup_dur == 0.0
