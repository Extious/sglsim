from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CacheOperation:
    host_page_ids: list[int]
    device_page_ids: list[int]
    node_ids: list[int] = field(default_factory=list)
    # Offline GlobalDES only.  Online HiCache leaves this unset and continues
    # to use the pinned SGLang-style queue/ACK control flow.
    intent_id: str | None = None
    owner: str | None = None

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
        intent_ids = {op.intent_id for op in ops if op.intent_id is not None}
        owners = {op.owner for op in ops if op.owner is not None}
        return CacheOperation(
            host_page_ids,
            device_page_ids,
            node_ids,
            intent_id=(next(iter(intent_ids)) if len(intent_ids) == 1 else None),
            owner=(next(iter(owners)) if len(owners) == 1 else None),
        )
