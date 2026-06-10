from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CacheOperation:
    host_page_ids: list[int]
    device_page_ids: list[int]
    node_ids: list[int] = field(default_factory=list)

    @staticmethod
    def merge_ops(ops: list[CacheOperation]) -> CacheOperation:
        if len(ops) == 1:
            return ops[0]
        host_page_ids: list[int] = []
        device_page_ids: list[int] = []
        node_ids: list[int] = []
        for op in ops:
            host_page_ids.extend(op.host_page_ids)
            device_page_ids.extend(op.device_page_ids)
            node_ids.extend(op.node_ids)
        return CacheOperation(host_page_ids, device_page_ids, node_ids)
