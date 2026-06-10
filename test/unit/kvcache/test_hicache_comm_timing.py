from sglsim.srt.communication.segments import split_contiguous_page_segments
from sglsim.srt.communication.types import CommPath, CommunicationRequest
from sglsim.srt.communication.config import CommunicationConfig
from sglsim.srt.communication.kv_bytes import KVByteInfo
from sglsim.srt.communication.predictor import CommunicationTimePredictor
from sglsim.srt.mem_cache.hicache import HiCacheController
from sglsim.srt.mem_cache.memory_pool import DeviceKVPool
from sglsim.srt.mem_cache.memory_pool_host import HostKVPool
from sglsim.srt.simulation.manager.platform_config import PlatformConfig


def test_contiguous_pages_cheaper_than_fragmented_h2d():
    kv_info = KVByteInfo(
        bytes_per_token_all_layers=256,
        bytes_per_token_per_layer=8,
        num_layers=32,
    )
    predictor = CommunicationTimePredictor(
        comm_config=CommunicationConfig(mode="alpha_beta"),
        platform=PlatformConfig(memory_read_bandwidth_gb=64.0),
        kv_info=kv_info,
    )
    contiguous = predictor.predict(
        CommunicationRequest(
            path=CommPath.H2D,
            host_page_ids=(0, 1, 2),
            device_page_ids=(0, 1, 2),
            page_size=16,
        )
    )
    fragmented = predictor.predict(
        CommunicationRequest(
            path=CommPath.H2D,
            host_page_ids=(0, 2, 4),
            device_page_ids=(0, 2, 4),
            page_size=16,
        )
    )
    assert fragmented.duration_s > contiguous.duration_s


def test_segment_splitter_merges_runs():
    assert split_contiguous_page_segments([0, 1, 2], [10, 11, 12]) == [3]
    assert split_contiguous_page_segments([0, 2], [10, 12]) == [1, 1]


def test_hicache_write_duration_scales_with_page_count():
    device = DeviceKVPool(capacity_pages=8, page_size=16)
    host = HostKVPool(capacity_pages=8, page_size=16)
    controller = HiCacheController(device, host, write_policy="write_through", page_size=16)

    controller.write([0, 1], node_id=1, owner="r0")
    two_pages = controller.pending_backup_dur
    controller.write([2], node_id=2, owner="r1")
    three_pages = controller.pending_backup_dur

    assert two_pages > 0
    assert three_pages > two_pages
