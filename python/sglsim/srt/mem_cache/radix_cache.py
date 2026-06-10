from __future__ import annotations

import heapq
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from math import ceil

from sglsim.srt.mem_cache.page import KVPage


@dataclass
class PrefixMatch:
    matched_length: int
    page_ids: list[int]
    device_length: int = 0
    host_length: int = 0
    host_page_ids: list[int] = field(default_factory=list)
    last_node: TreeNode | None = None


@dataclass
class InsertResult:
    touched: list[TreeNode]
    prefix_len: int
    last_node: TreeNode
    replaced_page_ids: list[int] = field(default_factory=list)


class TreeNode:
    counter = 0

    def __init__(self, node_id: int | None = None) -> None:
        self.children: dict[tuple[int, ...], TreeNode] = {}
        self.parent: TreeNode | None = None
        self.key: list[int] = []
        self.value: list[int] | None = None
        self.host_value: list[int] | None = None
        self.lock_ref = 0
        self.host_ref_counter = 0
        self.hit_count = 0
        self.storage_acked_len = 0
        self.last_access_time = time.monotonic()
        self.id = TreeNode.counter if node_id is None else node_id
        TreeNode.counter += 1

    @property
    def evicted(self) -> bool:
        return self.value is None

    @property
    def backuped(self) -> bool:
        return self.host_value is not None

    def protect_host(self) -> None:
        self.host_ref_counter += 1

    def release_host(self) -> None:
        if self.host_ref_counter <= 0:
            raise RuntimeError("Host reference counter is already zero.")
        self.host_ref_counter -= 1


def _child_key(token_ids: list[int], page_size: int, idx: int = 0) -> tuple[int, ...]:
    # ``idx`` lets callers index into ``token_ids`` without slicing the whole
    # remaining tail on every loop iteration. Slicing ``key[idx:]`` per step made
    # match/insert/collect O(n^2) in the key length (catastrophic with
    # page_size=1, where every token is its own page).
    if page_size == 1:
        return (token_ids[idx],)
    return tuple(token_ids[idx : idx + page_size])


def _page_align(length: int, page_size: int) -> int:
    if page_size <= 1:
        return length
    return (length // page_size) * page_size


class RadixCache:
    """Radix tree for prefix KV metadata, aligned with sglang.srt.mem_cache.radix_cache."""

    def __init__(self, page_size: int = 1) -> None:
        self.page_size = page_size
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_leaves: set[TreeNode] = set()
        self.evictable_host_leaves: set[TreeNode] = set()
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self._nodes: dict[int, TreeNode] = {self.root_node.id: self.root_node}

    def reset(self) -> None:
        TreeNode.counter = 0
        self.__init__(self.page_size)

    def get_node(self, node_id: int | None) -> TreeNode | None:
        if node_id is None:
            return None
        return self._nodes.get(node_id)

    def match_prefix(self, token_ids: list[int]) -> PrefixMatch:
        aligned_len = _page_align(len(token_ids), self.page_size)
        key = token_ids[:aligned_len]
        if not key:
            return PrefixMatch(matched_length=0, page_ids=[], device_length=0, last_node=self.root_node)

        device_page_ids: list[int] = []
        contiguous_device_len = 0
        device_prefix_open = True
        node = self.root_node
        idx = 0
        last_device_node = self.root_node

        while idx < len(key):
            child_key = _child_key(key, self.page_size, idx)
            child = node.children.get(child_key)
            if child is None:
                break
            prefix_len = min(len(child.key), len(key) - idx)
            if child.key[:prefix_len] != key[idx : idx + prefix_len]:
                break
            node = child
            node.last_access_time = time.monotonic()
            if device_prefix_open and child.value is not None:
                device_page_ids.extend(child.value[:prefix_len])
                contiguous_device_len += prefix_len
                last_device_node = child
            else:
                device_prefix_open = False
            idx += prefix_len
            if prefix_len < len(child.key):
                break

        host_page_ids: list[int] = []
        path_nodes: list[TreeNode] = []
        walk = node
        while walk is not self.root_node:
            path_nodes.append(walk)
            walk = walk.parent or self.root_node
        for path_node in reversed(path_nodes):
            if path_node.evicted and path_node.host_value:
                host_page_ids.extend(path_node.host_value)

        all_page_ids = device_page_ids + host_page_ids
        return PrefixMatch(
            matched_length=len(all_page_ids),
            page_ids=all_page_ids,
            device_length=contiguous_device_len,
            host_length=len(host_page_ids),
            host_page_ids=host_page_ids,
            last_node=node,
        )

    def insert(self, token_ids: list[int], page_ids: list[int]) -> InsertResult:
        if len(page_ids) < len(token_ids):
            raise ValueError("page_ids must cover all token_ids")

        aligned_len = _page_align(len(token_ids), self.page_size)
        key = token_ids[:aligned_len]
        value = page_ids[:aligned_len]
        if not key:
            return InsertResult(touched=[], prefix_len=0, last_node=self.root_node)

        prefix_len = 0
        touched: list[TreeNode] = []
        replaced_page_ids: list[int] = []
        node = self.root_node
        idx = 0
        device_prefix_open = True

        while idx < len(key):
            child_key = _child_key(key, self.page_size, idx)
            child = node.children.get(child_key)
            if child is None:
                break
            segment_len = min(len(child.key), len(key) - idx)
            if child.key[:segment_len] != key[idx : idx + segment_len]:
                break
            segment_value = list(value[idx : idx + segment_len])
            if segment_len < len(child.key):
                child = self._split_node(child, segment_len)
            had_device_value = child.value is not None
            if device_prefix_open and had_device_value:
                prefix_len += segment_len
            else:
                device_prefix_open = False
                if had_device_value:
                    new_ids = set(segment_value)
                    replaced_page_ids.extend(
                        page_id
                        for page_id in dict.fromkeys(child.value or [])
                        if page_id not in new_ids
                    )
                else:
                    if child.lock_ref == 0:
                        self.evictable_size_ += len(child.key)
                    else:
                        self.protected_size_ += len(child.key)
                child.value = segment_value
                touched.append(child)
                self._update_leaf_status(child)
                self._update_host_leaf_status(child)
                if child.parent is not None:
                    self._update_leaf_status(child.parent)
            node = child
            node.last_access_time = time.monotonic()
            idx += segment_len

        while idx < len(key):
            step = self.page_size if self.page_size > 1 else 1
            step = min(step, len(key) - idx)
            segment_key = key[idx : idx + step]
            segment_value = value[idx : idx + step]
            child_key = _child_key(segment_key, self.page_size)
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = list(segment_key)
            new_node.value = list(segment_value)
            self._register_node(new_node)
            node.children[child_key] = new_node
            if new_node.lock_ref == 0:
                self.evictable_size_ += len(new_node.key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            touched.append(new_node)
            node = new_node
            idx += step

        return InsertResult(
            touched=touched,
            prefix_len=prefix_len,
            last_node=node,
            replaced_page_ids=replaced_page_ids,
        )

    def prefix_length_to_node(self, node: TreeNode) -> int:
        total = 0
        walk = node
        while walk is not self.root_node:
            total += len(walk.key)
            walk = walk.parent or self.root_node
        return total

    def insert_suffix(
        self,
        parent_node: TreeNode,
        token_ids: list[int],
        page_ids: list[int],
    ) -> InsertResult:
        """Insert new KV segments under ``parent_node`` (prefix already in tree)."""
        if len(page_ids) < len(token_ids):
            raise ValueError("page_ids must cover all token_ids")
        if not token_ids:
            return InsertResult(
                touched=[],
                prefix_len=self.prefix_length_to_node(parent_node),
                last_node=parent_node,
            )

        key = token_ids
        value = page_ids[: len(token_ids)]
        touched: list[TreeNode] = []
        node = parent_node
        idx = 0

        while idx < len(key):
            step = self.page_size if self.page_size > 1 else 1
            step = min(step, len(key) - idx)
            segment_key = key[idx : idx + step]
            segment_value = value[idx : idx + step]
            child_key = _child_key(segment_key, self.page_size)
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = list(segment_key)
            new_node.value = list(segment_value)
            self._register_node(new_node)
            node.children[child_key] = new_node
            if new_node.lock_ref == 0 and new_node.value is not None:
                self.evictable_size_ += len(new_node.key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            touched.append(new_node)
            node = new_node
            idx += step

        return InsertResult(
            touched=touched,
            prefix_len=self.prefix_length_to_node(parent_node),
            last_node=node,
        )

    def inc_lock_ref(self, node: TreeNode) -> int:
        delta = 0
        while node is not self.root_node:
            if node.lock_ref == 0 and node.value is not None:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            node = node.parent or self.root_node
        return delta

    def dec_lock_ref(self, node: TreeNode) -> int:
        delta = 0
        while node is not self.root_node:
            if node.lock_ref <= 0:
                raise RuntimeError("Lock reference counter is already zero.")
            if node.lock_ref == 1 and node.value is not None:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            node = node.parent or self.root_node
        return delta

    def set_host_value(self, node: TreeNode, host_page_ids: list[int]) -> None:
        node.host_value = list(host_page_ids)

    def collect_nodes_on_path(self, token_ids: list[int]) -> list[TreeNode]:
        aligned_len = _page_align(len(token_ids), self.page_size)
        key = token_ids[:aligned_len]
        nodes: list[TreeNode] = []
        node = self.root_node
        idx = 0
        while idx < len(key):
            child_key = _child_key(key, self.page_size, idx)
            child = node.children.get(child_key)
            if child is None:
                break
            prefix_len = min(len(child.key), len(key) - idx)
            if child.key[:prefix_len] != key[idx : idx + prefix_len]:
                break
            nodes.append(child)
            idx += prefix_len
            node = child
            if prefix_len < len(child.key):
                break
        return nodes

    def evictable_size(self) -> int:
        return self.evictable_size_

    def measured_evictable_tokens(self) -> int:
        """Sum device-resident unlocked node token lengths."""
        return sum(
            len(node.key)
            for node in self._nodes.values()
            if node.lock_ref == 0 and node.value is not None
        )

    def measured_protected_tokens(self) -> int:
        """Sum device-resident locked node token lengths."""
        return sum(
            len(node.key)
            for node in self._nodes.values()
            if node.lock_ref > 0 and node.value is not None
        )

    def reconcile_accounting(self) -> None:
        """Rebuild aggregate counters from authoritative node state.

        Incremental lock accounting mirrors SGLang, but SGLSim can hit stale
        release paths during simulated retract/finalize retries. The node-local
        ``lock_ref`` / ``value`` fields and parent/child links are authoritative,
        so resync counters and leaf sets before pressure eviction.
        """
        self.evictable_size_ = self.measured_evictable_tokens()
        self.protected_size_ = self.measured_protected_tokens()
        self.evictable_leaves.clear()
        self.evictable_host_leaves.clear()
        for node in list(self._nodes.values()):
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)

    def assert_evictable_invariant(self, *, context: str = "", strict: bool = False) -> None:
        if self.evictable_size_ < 0 or self.protected_size_ < 0 or strict:
            self.reconcile_accounting()
        if self.evictable_size_ < 0:
            raise RuntimeError(
                f"evictable_size_ negative ({self.evictable_size_})"
                f"{': ' + context if context else ''}"
            )
        if self.protected_size_ < 0:
            raise RuntimeError(
                f"protected_size_ negative ({self.protected_size_})"
                f"{': ' + context if context else ''}"
            )
        if not strict:
            return
        measured = self.measured_evictable_tokens()
        if self.evictable_size_ != measured:
            raise RuntimeError(
                "evictable_size_ mismatch: "
                f"counter={self.evictable_size_} measured={measured}"
                f"{': ' + context if context else ''}"
            )

    def evictable_pages(self) -> int:
        return ceil(self.evictable_size_ / self.page_size)

    def evictable_host_pages(self) -> int:
        total = 0
        for node in self.evictable_host_leaves:
            if node.host_ref_counter > 0 or not node.evicted or node.host_value is None:
                continue
            total += len(dict.fromkeys(node.host_value))
        return total

    def evict(
        self,
        num_tokens: int,
        device_pool_pages: dict[int, KVPage],
        *,
        write_through: bool = True,
    ) -> list[KVPage]:
        if num_tokens <= 0:
            return []
        heap = [
            (node.last_access_time, node.id, node)
            for node in self.evictable_leaves
            if node.lock_ref == 0 and node.value is not None
        ]
        heapq.heapify(heap)
        num_evicted = 0
        freed: list[KVPage] = []
        while num_evicted < num_tokens and heap:
            _, _, node = heapq.heappop(heap)
            if node.lock_ref > 0 or node.value is None:
                continue
            if write_through and node.backuped:
                pages, count = self._evict_backuped(node, device_pool_pages)
            else:
                pages, count = self._evict_regular(node, device_pool_pages)
            freed.extend(pages)
            num_evicted += count
            self._maybe_push_parent(heap, node.parent)

        return freed

    def evict_write_back(
        self,
        num_tokens: int,
        device_pool_pages: dict[int, KVPage],
        *,
        backup_fn: Callable[[TreeNode], None],
    ) -> list[KVPage]:
        if num_tokens <= 0:
            return []
        heap = [
            (node.last_access_time, node.id, node)
            for node in self.evictable_leaves
            if node.lock_ref == 0 and node.value is not None
        ]
        heapq.heapify(heap)
        num_evicted = 0
        freed: list[KVPage] = []
        while num_evicted < num_tokens and heap:
            _, _, node = heapq.heappop(heap)
            if node.lock_ref > 0 or node.value is None:
                continue
            if not node.backuped:
                backup_fn(node)
            pages, count = self._evict_backuped(node, device_pool_pages)
            freed.extend(pages)
            num_evicted += count
            self._maybe_push_parent(heap, node.parent)
        return freed

    def _maybe_push_parent(self, heap: list, parent: TreeNode | None) -> None:
        if parent is None or parent is self.root_node:
            return
        for child in parent.children.values():
            if not child.evicted:
                return
        if parent.lock_ref > 0 or parent.value is None:
            return
        heapq.heappush(heap, (parent.last_access_time, parent.id, parent))

    def _evict_backuped(
        self,
        node: TreeNode,
        device_pool_pages: dict[int, KVPage],
    ) -> tuple[list[KVPage], int]:
        if node.value is None:
            return [], 0
        unique_ids = list(dict.fromkeys(node.value))
        freed = []
        for page_id in unique_ids:
            page = device_pool_pages.get(page_id)
            if page is not None:
                freed.append(page)
        count = len(node.key)
        self.evictable_size_ -= count
        node.value = None
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        self._update_host_leaf_status(node)
        if node.parent is not None:
            self._update_leaf_status(node.parent)
        return freed, count

    def _evict_regular(
        self,
        node: TreeNode,
        device_pool_pages: dict[int, KVPage],
    ) -> tuple[list[KVPage], int]:
        if node.value is None:
            return [], 0
        unique_ids = list(dict.fromkeys(node.value))
        freed = []
        for page_id in unique_ids:
            page = device_pool_pages.get(page_id)
            if page is not None:
                freed.append(page)
        count = len(node.key)
        self._delete_leaf(node)
        return freed, count

    def evict_host(self, num_pages: int, host_pool_pages: dict[int, KVPage]) -> list[KVPage]:
        if num_pages <= 0:
            return []
        heap = [
            (node.last_access_time, node.id, node)
            for node in self.evictable_host_leaves
            if node.evicted and node.host_ref_counter == 0 and node.host_value
        ]
        heapq.heapify(heap)
        num_evicted = 0
        freed: list[KVPage] = []
        while num_evicted < num_pages and heap:
            _, _, node = heapq.heappop(heap)
            if not node.evicted or node.host_ref_counter > 0 or not node.host_value:
                continue
            unique_ids = list(dict.fromkeys(node.host_value))
            for page_id in unique_ids:
                if num_evicted >= num_pages:
                    break
                page = host_pool_pages.get(page_id)
                if page is not None:
                    freed.append(page)
                    num_evicted += 1
            parent = node.parent
            if node.parent is not None:
                key = _child_key(node.key, self.page_size)
                node.parent.children.pop(key, None)
            node.host_value = None
            if node in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(node)
            self._update_host_leaf_status(parent)
            self._maybe_push_host_parent(heap, parent)
        return freed

    def _maybe_push_host_parent(self, heap: list, parent: TreeNode | None) -> None:
        if parent is None or parent is self.root_node:
            return
        for child in parent.children.values():
            if child.evicted and child.host_value is not None:
                return
        if parent.host_ref_counter > 0 or not parent.evicted or not parent.host_value:
            return
        heapq.heappush(heap, (parent.last_access_time, parent.id, parent))

    def _delete_leaf(self, node: TreeNode) -> None:
        self.evictable_size_ -= len(node.key)
        if node.parent is not None:
            key = _child_key(node.key, self.page_size)
            node.parent.children.pop(key, None)
            self._update_leaf_status(node.parent)
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        if node in self.evictable_host_leaves:
            self.evictable_host_leaves.remove(node)
        self._nodes.pop(node.id, None)

    def _update_leaf_status(self, node: TreeNode) -> None:
        if node is self.root_node:
            return
        if node.evicted or node.lock_ref > 0:
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)
            return
        for child in node.children.values():
            if not child.evicted:
                if node in self.evictable_leaves:
                    self.evictable_leaves.remove(node)
                return
        if node not in self.evictable_leaves:
            self.evictable_leaves.add(node)

    def _update_host_leaf_status(self, node: TreeNode | None) -> None:
        if node is None or node is self.root_node:
            return
        if not node.evicted or node.lock_ref > 0:
            if node in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(node)
            return
        for child in node.children.values():
            if child.evicted:
                if node in self.evictable_host_leaves:
                    self.evictable_host_leaves.remove(node)
                return
        if node.host_value is None or node.host_ref_counter > 0:
            if node in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(node)
            return
        if node not in self.evictable_host_leaves:
            self.evictable_host_leaves.add(node)

    def _split_node(self, node: TreeNode, split_len: int) -> TreeNode:
        new_node = TreeNode()
        new_node.parent = node.parent
        new_node.key = list(node.key[:split_len])
        new_node.value = list(node.value[:split_len]) if node.value else None
        new_node.host_value = list(node.host_value[:split_len]) if node.host_value else None
        new_node.lock_ref = node.lock_ref
        self._register_node(new_node)

        node.key = list(node.key[split_len:])
        if node.value is not None:
            node.value = list(node.value[split_len:])
        if node.host_value is not None:
            node.host_value = list(node.host_value[split_len:])

        if node.parent is not None:
            old_key = _child_key(new_node.key, self.page_size)
            node.parent.children[old_key] = new_node
        new_node.children[_child_key(node.key, self.page_size)] = node
        node.parent = new_node

        if new_node.lock_ref == 0 and new_node.value is not None:
            self.evictable_size_ += len(new_node.key)
        self._update_leaf_status(new_node)
        self._update_leaf_status(node)
        if node.parent is not None and node.parent is not new_node:
            self._update_leaf_status(node.parent)
        return new_node

    def _register_node(self, node: TreeNode) -> None:
        self._nodes[node.id] = node
