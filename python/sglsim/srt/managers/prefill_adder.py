from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from sglsim.srt.managers.schedule_batch import Req, ScheduleBatch
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool

CLIP_MAX_NEW_TOKENS = 4096


class AddReqResult(Enum):
    CONTINUE = auto()
    NO_TOKEN = auto()
    OTHER = auto()


def _decode_reserve_tokens(req: Req, new_token_ratio: float) -> int:
    remaining = min(req.remaining_decode_tokens, CLIP_MAX_NEW_TOKENS)
    return int(remaining * new_token_ratio)


def _new_request_decode_reserve(req: Req) -> int:
    """SGLang add_one_req: decode reserve without new_token_ratio."""
    return min(req.remaining_decode_tokens, CLIP_MAX_NEW_TOKENS)


def _full_extend_tokens(req: Req) -> int:
    """SGLang add_one_req budgets the full remaining extend before chunking."""
    remaining = max(req.prefill_length - req.kv_committed_length, 0)
    if req.prefill_chunks_remaining:
        planned = sum(req.prefill_chunks_remaining)
        return min(planned, remaining)
    return min(req.extend_length, remaining)


@dataclass
class PrefillAdder:
    page_size: int
    cache: KVCachePool
    running_batch: ScheduleBatch
    max_prefill_tokens: int
    max_running_requests: int
    rem_chunk_tokens: int | None = None
    new_token_ratio: float = 1.0
    prefill_max_requests: int | None = None
    chunked_prefill_size: int | None = None
    enable_overlap: bool = True

    rem_input_tokens: int = 0
    rem_total_token_offset: int = 0
    cur_rem_token_offset: int = 0
    log_input_tokens: int = 0
    log_hit_tokens: int = 0
    can_run_list: list[Req] = field(default_factory=list)
    new_chunked_req: Req | None = None

    def __post_init__(self) -> None:
        self.rem_total_token_offset = sum(
            _decode_reserve_tokens(req, self.new_token_ratio)
            for req in self.running_batch.reqs
        )
        self.cur_rem_token_offset = 0
        self.rem_input_tokens = self.max_prefill_tokens

    @property
    def rem_total_tokens(self) -> int:
        return self.cache.available_and_evictable_tokens() - self.rem_total_token_offset

    @property
    def cur_rem_tokens(self) -> int:
        return self.cache.available_and_evictable_tokens() - self.cur_rem_token_offset

    def _ceil_paged_tokens(self, tokens: int) -> int:
        if tokens <= 0:
            return 0
        return -(-tokens // self.page_size) * self.page_size

    def budget_state(self) -> AddReqResult:
        if self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0:
            return AddReqResult.NO_TOKEN
        if self.rem_input_tokens <= 0:
            return AddReqResult.OTHER
        if self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0:
            return AddReqResult.OTHER
        return AddReqResult.CONTINUE

    def _update_prefill_budget(
        self,
        extend_input_len: int,
        max_new_tokens: int,
        *,
        prefix_len: int = 0,
    ) -> None:
        paged = self._ceil_paged_tokens(extend_input_len)
        page_overhead = self.page_size
        self.rem_total_token_offset += paged + max_new_tokens + page_overhead
        self.cur_rem_token_offset += paged + page_overhead
        self.rem_input_tokens -= paged
        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= paged
        self.log_input_tokens += extend_input_len
        self.log_hit_tokens += prefix_len

    def _chunk_rem_tokens(self) -> int:
        if self.rem_chunk_tokens is not None:
            return min(self.rem_chunk_tokens, int(self.rem_total_tokens))
        return min(self.rem_input_tokens, int(self.rem_total_tokens))

    def _apply_chunk_truncation(self, req: Req, trunc_len: int) -> bool:
        if not req.prefill_chunks_remaining:
            return False
        first = req.prefill_chunks_remaining[0]
        if first <= trunc_len:
            return False
        remainder = first - trunc_len
        req.prefill_chunks_remaining[0] = trunc_len
        if remainder > 0:
            req.prefill_chunks_remaining.insert(1, remainder)
        return True

    def _sync_extend_after_load_back(self, req: Req) -> None:
        if not self.cache.hierarchical_enabled or req.prefill_length <= 0:
            return
        matched_device = self.cache.device_resident_length(req)
        # Mirror SGLang req.init_next_round_input + add_one_req: after any
        # load_back attempt, admission must budget against the prefix actually
        # resident on device for this round.  A stale cached_length from an
        # evicted/host-only prefix would otherwise make alloc-only prefill think
        # pages are already covered and repeat the same chunk indefinitely.
        max_skip = max(req.prefill_length - 1, 0)
        device_skip = min(matched_device, max_skip)
        committed_skip = min(max(req.kv_committed_length, 0), max_skip)
        skip = device_skip
        has_request_local_progress = (
            req.cache_protected_len > 0 or bool(req.prefill_alloc_page_ids)
        )
        if (
            req.prefill_chunks_remaining
            and committed_skip > 0
            and has_request_local_progress
        ):
            # SGLang carries chunked-prefill progress in req.prefix_indices. That
            # progress is request-local KV and must not be rolled back merely
            # because a later radix/device prefix match is smaller.
            skip = max(skip, committed_skip)
        elif req.prefill_alloc_page_ids:
            # Keep in-flight chunk progress when deferred pages are still owned by req.
            skip = max(skip, committed_skip)
        skip = min(skip, max(req.prefill_length - 1, 0))
        req.cached_length = max(device_skip, skip if req.prefill_chunks_remaining else 0)
        req.kv_committed_length = max(skip, 0)
        planned_uncached = max(req.prefill_length - skip, 0)
        if sum(req.prefill_chunks_remaining) != planned_uncached:
            from sglsim.srt.managers.chunked_prefill import split_prefill_length

            req.prefill_chunks_remaining = list(
                split_prefill_length(
                    planned_uncached,
                    self.chunked_prefill_size,
                )
            )

    def _admit_kv_finalize(self, req: Req) -> None:
        """Finalize load_back visibility before admission budgeting."""
        if not self.cache.hierarchical_enabled:
            req.admit_load_back_tokens = 0
            req.host_hit_length = 0
            req.last_host_node_id = None
            return
        mem_quota = self.rem_total_tokens
        self.cache.init_load_back(req, mem_quota=mem_quota)
        loaded = 0
        if req.host_hit_length > 0:
            loaded = self.cache.load_back(req, mem_quota=mem_quota)
        if loaded <= 0:
            req.host_hit_length = 0
            req.last_host_node_id = None
        else:
            req.host_hit_length = loaded
        req.admit_load_back_tokens = loaded
        self._sync_extend_after_load_back(req)

    def add_chunked_req(self, req: Req) -> Req | None:
        if not req.prefill_chunks_remaining:
            return None

        if self.cache.hierarchical_enabled:
            self._admit_kv_finalize(req)
        if req.extend_length <= 0:
            self.cache.prepare_prefill(req)
            self.can_run_list.append(req)
            return None

        _rem_tokens = self._chunk_rem_tokens() // self.page_size * self.page_size
        if _rem_tokens <= 0:
            req.admit_load_back_tokens = 0
            return req

        first = req.extend_length
        truncated = first > _rem_tokens
        chunk_len = min(first, _rem_tokens)
        if chunk_len <= 0:
            req.admit_load_back_tokens = 0
            return req

        self._apply_chunk_truncation(req, chunk_len)
        if req.cache_protected_len <= 0 and req.last_node_id is None:
            self.cache.prepare_prefill(req)
        self.can_run_list.append(req)
        max_new = (
            0
            if truncated
            else min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
        )
        # SGLang add_chunked_req reports prefix_len=0 for continuation chunks.
        # The already-processed prefix still participates in forward prediction
        # via past_kv_length, but it is not logged as a fresh cache hit here.
        self._update_prefill_budget(chunk_len, max_new, prefix_len=0)
        return req if truncated else None

    def add_one_req(self, req: Req, *, has_chunked_req: bool = False) -> AddReqResult:
        if (
            self.prefill_max_requests is not None
            and len(self.can_run_list) >= self.prefill_max_requests
        ):
            return AddReqResult.OTHER

        extend_tokens = _full_extend_tokens(req)
        if extend_tokens <= 0:
            return AddReqResult.OTHER

        real_input_tokens = self._ceil_paged_tokens(
            max(extend_tokens - req.host_hit_length, 0)
        )
        total_tokens = extend_tokens + _new_request_decode_reserve(req)
        if total_tokens >= self.rem_total_tokens:
            if self.running_batch.is_empty():
                self.cache.evict_from_tree_cache(real_input_tokens, owner=req.rid)
                if total_tokens >= self.rem_total_tokens:
                    req.admit_load_back_tokens = 0
                    return AddReqResult.NO_TOKEN
            else:
                req.admit_load_back_tokens = 0
                return AddReqResult.NO_TOKEN

        if real_input_tokens >= self.rem_input_tokens and len(self.can_run_list) != 0:
            req.admit_load_back_tokens = 0
            return AddReqResult.OTHER

        # Align with sglang add_one_req: finalize load_back / lock first, then
        # budget against the actual device-resident prefix for this round.
        if self.cache.hierarchical_enabled:
            self._admit_kv_finalize(req)
        # In no-overlap mode the serial event loop accumulates more waiting
        # requests per step; a batch may admit several requests whose
        # load_backs collectively consume pages that earlier per-request
        # budget checks did not see.  Track load_back consumption so every
        # subsequent check inside the same batch sees real remaining capacity.
        if not self.enable_overlap and req.admit_load_back_tokens > 0:
            self.rem_total_token_offset += req.admit_load_back_tokens
        extend_tokens = _full_extend_tokens(req)
        if extend_tokens <= 0:
            self.cache.prepare_prefill(req)
            self.can_run_list.append(req)
            return self.budget_state()

        prefix_len = max(req.cached_length, 0)
        decode_reserve = _new_request_decode_reserve(req)
        input_tokens = self._ceil_paged_tokens(extend_tokens)
        total_tokens = extend_tokens + decode_reserve

        if total_tokens >= self.rem_total_tokens:
            if self.running_batch.is_empty():
                self.cache.evict_from_tree_cache(input_tokens, owner=req.rid)
                if total_tokens >= self.rem_total_tokens:
                    req.admit_load_back_tokens = 0
                    return AddReqResult.NO_TOKEN
            else:
                req.admit_load_back_tokens = 0
                return AddReqResult.NO_TOKEN

        if input_tokens >= self.rem_input_tokens and len(self.can_run_list) != 0:
            req.admit_load_back_tokens = 0
            return AddReqResult.OTHER

        has_more_prefill_chunks = (
            bool(req.prefill_chunks_remaining)
            and req.extend_length < extend_tokens
        )
        if (
            self.rem_chunk_tokens is None
            or (input_tokens <= self.rem_chunk_tokens and not has_more_prefill_chunks)
        ):
            self.cache.prepare_prefill(req)
            extend_tokens = _full_extend_tokens(req)
            prefix_len = max(req.cached_length, 0)
            decode_reserve = _new_request_decode_reserve(req)
            input_tokens = self._ceil_paged_tokens(extend_tokens)
            total_tokens = extend_tokens + decode_reserve
            if total_tokens >= self.rem_total_tokens:
                self.cache.release_prefill_lock(req)
                req.admit_load_back_tokens = 0
                return AddReqResult.NO_TOKEN
            if input_tokens >= self.rem_input_tokens and len(self.can_run_list) != 0:
                self.cache.release_prefill_lock(req)
                req.admit_load_back_tokens = 0
                return AddReqResult.OTHER
            self.can_run_list.append(req)
            self._update_prefill_budget(
                extend_tokens,
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS),
                prefix_len=prefix_len,
            )
        else:
            if has_chunked_req:
                req.admit_load_back_tokens = 0
                return AddReqResult.OTHER
            trunc_len = self.rem_chunk_tokens
            if self.rem_chunk_tokens is not None:
                trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size
            if trunc_len <= 0:
                req.admit_load_back_tokens = 0
                return AddReqResult.OTHER
            self.cache.prepare_prefill(req)
            extend_tokens = _full_extend_tokens(req)
            prefix_len = max(req.cached_length, 0)
            input_tokens = self._ceil_paged_tokens(extend_tokens)
            if input_tokens >= self.rem_total_tokens:
                self.cache.release_prefill_lock(req)
                req.admit_load_back_tokens = 0
                return AddReqResult.NO_TOKEN
            if input_tokens >= self.rem_input_tokens and len(self.can_run_list) != 0:
                self.cache.release_prefill_lock(req)
                req.admit_load_back_tokens = 0
                return AddReqResult.OTHER
            trunc_len = min(trunc_len, int(self.rem_total_tokens))
            trunc_len = trunc_len // self.page_size * self.page_size
            if trunc_len <= 0:
                self.cache.release_prefill_lock(req)
                req.admit_load_back_tokens = 0
                return AddReqResult.OTHER
            self._apply_chunk_truncation(req, trunc_len)
            self.can_run_list.append(req)
            self.new_chunked_req = req
            self._update_prefill_budget(trunc_len, 0, prefix_len=prefix_len)

        return self.budget_state()

    def try_add(self, req: Req, *, has_chunked_req: bool = False) -> AddReqResult:
        return self.add_one_req(req, has_chunked_req=has_chunked_req)
