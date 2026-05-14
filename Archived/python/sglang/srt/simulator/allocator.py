from __future__ import annotations

import torch


class IndexableWrapper:
    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, _):
        return self._fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def ceil_div(a, b):
    return (a + b - 1) // b


def alloc_extend_cpu(
    pre_lens_ptr: torch.Tensor,
    seq_lens_ptr: torch.Tensor,
    last_loc_ptr: torch.Tensor,
    free_page_ptr: torch.Tensor,
    out_indices: torch.Tensor,
    bs_upper: int,
    page_size: int,
):
    del bs_upper
    pre_lens = pre_lens_ptr.cpu().tolist()
    seq_lens = seq_lens_ptr.cpu().tolist()
    last_loc = last_loc_ptr.cpu().tolist()
    free_pages = free_page_ptr.cpu().tolist()
    batch_size = len(pre_lens)
    output_offset = 0
    free_page_offset = 0
    for pid in range(batch_size):
        pre_len = pre_lens[pid]
        seq_len = seq_lens[pid]
        extend_len = seq_len - pre_len
        num_new_pages = ceil_div(seq_len, page_size) - ceil_div(pre_len, page_size)
        if extend_len <= 0:
            free_page_offset += num_new_pages
            continue
        current_page_end = ceil_div(pre_len, page_size) * page_size
        part1_end = min(seq_len, current_page_end)
        part1_size = max(part1_end - pre_len, 0)
        for i in range(part1_size):
            out_indices[output_offset + i] = last_loc[pid] + 1 + i
        output_offset += part1_size
        if output_offset >= extend_len:
            free_page_offset += num_new_pages
            continue
        full_pages_start = current_page_end
        full_pages_end = (seq_len // page_size) * page_size
        full_pages_size = max(full_pages_end - full_pages_start, 0)
        for i in range(full_pages_size):
            page_id = free_pages[free_page_offset + (i // page_size)]
            out_indices[output_offset + i] = page_id * page_size + (i % page_size)
        output_offset += full_pages_size
        tail = seq_len - full_pages_end
        if tail > 0:
            page_id = free_pages[free_page_offset + num_new_pages - 1]
            for i in range(tail):
                out_indices[output_offset + i] = page_id * page_size + i
            output_offset += tail
        free_page_offset += num_new_pages


def alloc_decode_cpu(
    seq_lens_ptr: torch.Tensor,
    last_loc_ptr: torch.Tensor,
    free_page_ptr: torch.Tensor,
    out_indices: torch.Tensor,
    bs_upper: int,
    page_size: int,
):
    del bs_upper
    seq_lens = seq_lens_ptr.cpu().tolist()
    last_loc = last_loc_ptr.cpu().tolist()
    free_pages = free_page_ptr.cpu().tolist()
    free_page_offset = 0
    for pid, seq_len in enumerate(seq_lens):
        pre_len = seq_len - 1
        num_new_pages = ceil_div(seq_len, page_size) - ceil_div(pre_len, page_size)
        if num_new_pages == 0:
            out_indices[pid] = last_loc[pid] + 1
        else:
            out_indices[pid] = free_pages[free_page_offset] * page_size
        free_page_offset += num_new_pages
