from __future__ import annotations

import torch


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def alloc_extend_cpu(
    pre_lens_ptr: torch.Tensor,
    seq_lens_ptr: torch.Tensor,
    last_loc_ptr: torch.Tensor,
    free_page_ptr: torch.Tensor,
    out_indices: torch.Tensor,
    bs_upper: int,
    page_size: int,
) -> None:
    pre_lens = pre_lens_ptr.detach().cpu().tolist()
    seq_lens = seq_lens_ptr.detach().cpu().tolist()
    last_loc = last_loc_ptr.detach().cpu().tolist()
    free_pages = free_page_ptr.detach().cpu().tolist()

    batch_size = len(pre_lens)
    extend_lens = [seq_lens[i] - pre_lens[i] for i in range(batch_size)]
    new_pages = [
        _ceil_div(seq_lens[i], page_size) - _ceil_div(pre_lens[i], page_size)
        for i in range(batch_size)
    ]

    output_offset = 0
    free_page_offset = 0
    for pid in range(batch_size):
        pre_len = int(pre_lens[pid])
        seq_len = int(seq_lens[pid])
        extend_len = int(extend_lens[pid])
        if extend_len <= 0:
            free_page_offset += int(new_pages[pid])
            continue

        current_page_end = _ceil_div(pre_len, page_size) * page_size
        part1_end = min(seq_len, current_page_end)
        num_part1 = part1_end - pre_len
        for i in range(num_part1):
            out_indices[output_offset + i] = int(last_loc[pid]) + 1 + i
        output_offset += num_part1
        if pre_len + num_part1 == seq_len:
            free_page_offset += int(new_pages[pid])
            continue

        full_pages_start = current_page_end
        full_pages_end = (seq_len // page_size) * page_size
        num_part2 = full_pages_end - full_pages_start
        for i in range(num_part2):
            page_id = int(free_pages[free_page_offset + (i // page_size)])
            out_indices[output_offset + i] = page_id * page_size + (i % page_size)
        output_offset += num_part2
        if pre_len + num_part1 + num_part2 == seq_len:
            free_page_offset += int(new_pages[pid])
            continue

        num_part3 = seq_len - full_pages_end
        if num_part3 > 0:
            last_page = int(free_pages[free_page_offset + int(new_pages[pid]) - 1])
            for i in range(num_part3):
                out_indices[output_offset + i] = last_page * page_size + i
            output_offset += num_part3

        free_page_offset += int(new_pages[pid])


def alloc_decode_cpu(
    seq_lens_ptr: torch.Tensor,
    last_loc_ptr: torch.Tensor,
    free_page_ptr: torch.Tensor,
    out_indices: torch.Tensor,
    bs_upper: int,
    page_size: int,
) -> None:
    seq_lens = seq_lens_ptr.detach().cpu().tolist()
    last_loc = last_loc_ptr.detach().cpu().tolist()
    free_pages = free_page_ptr.detach().cpu().tolist()

    new_pages = []
    for seq_len in seq_lens:
        pre_len = int(seq_len) - 1
        new_pages.append(_ceil_div(int(seq_len), page_size) - _ceil_div(pre_len, page_size))

    free_page_offset = 0
    for pid, seq_len in enumerate(seq_lens):
        pre_len = int(seq_len) - 1
        needs_page = _ceil_div(int(seq_len), page_size) - _ceil_div(pre_len, page_size)
        if needs_page == 0:
            out_indices[pid] = int(last_loc[pid]) + 1
        else:
            page_id = int(free_pages[free_page_offset])
            out_indices[pid] = page_id * page_size
        free_page_offset += int(new_pages[pid])
