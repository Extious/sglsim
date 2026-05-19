from __future__ import annotations

import torch

from sgl_kernel.kvcacheio import (
    transfer_kv_per_layer,
    transfer_kv_per_layer_direct_pf_lf,
)


def test_transfer_kv_per_layer_cpu_copy():
    src_k = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    src_v = src_k + 100
    dst_k = torch.zeros_like(src_k)
    dst_v = torch.zeros_like(src_v)

    transfer_kv_per_layer(
        src_k=src_k,
        dst_k=dst_k,
        src_v=src_v,
        dst_v=dst_v,
        src_indices=torch.tensor([1, 3, 5]),
        dst_indices=torch.tensor([0, 2, 4]),
        item_size=src_k.shape[-1] * src_k.element_size(),
    )

    torch.testing.assert_close(dst_k[[0, 2, 4]], src_k[[1, 3, 5]])
    torch.testing.assert_close(dst_v[[0, 2, 4]], src_v[[1, 3, 5]])


def test_transfer_kv_page_first_direct_to_layer_first_cpu_copy():
    page_size = 2
    src = torch.arange(2 * 3 * page_size * 4, dtype=torch.float32).reshape(
        2, 3, page_size, 4
    )
    dst = torch.zeros(3, 4, 4, dtype=torch.float32)

    transfer_kv_per_layer_direct_pf_lf(
        src_ptrs=[src],
        dst_ptrs=[dst[1]],
        src_indices=torch.tensor([0, 1, 2, 3]),
        dst_indices=torch.tensor([0, 1, 2, 3]),
        layer_id=1,
        page_size=page_size,
    )

    expected = torch.stack([src[0, 1, 0], src[0, 1, 1], src[1, 1, 0], src[1, 1, 1]])
    torch.testing.assert_close(dst[1], expected)
