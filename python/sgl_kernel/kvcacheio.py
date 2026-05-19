from __future__ import annotations

from typing import Iterable

import torch


def _cpu_indices(indices: torch.Tensor) -> torch.Tensor:
    return indices.detach().to(device="cpu", dtype=torch.long)


def _copy_indexed(src: torch.Tensor, dst: torch.Tensor, src_indices, dst_indices) -> None:
    dst[_cpu_indices(dst_indices).to(dst.device)] = src[
        _cpu_indices(src_indices).to(src.device)
    ].to(dst.device)


def _copy_page_first_to_layer_first(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    page_size: int,
) -> None:
    src_idx = _cpu_indices(src_indices)
    dst_idx = _cpu_indices(dst_indices).to(dst.device)
    values = []
    for idx in src_idx.tolist():
        values.append(src[idx // page_size][layer_id][idx % page_size])
    if values:
        dst[dst_idx] = torch.stack(values).to(dst.device)


def _copy_layer_first_to_page_first(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    page_size: int,
) -> None:
    src_idx = _cpu_indices(src_indices).to(src.device)
    dst_idx = _cpu_indices(dst_indices)
    for value, idx in zip(src[src_idx].to(dst.device), dst_idx.tolist(), strict=True):
        dst[idx // page_size][layer_id][idx % page_size] = value


def _copy_page_head_to_layer_first(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    page_size: int,
) -> None:
    src_idx = _cpu_indices(src_indices)
    dst_idx = _cpu_indices(dst_indices).to(dst.device)
    values = []
    for idx in src_idx.tolist():
        values.append(src[idx // page_size, :, idx % page_size, layer_id])
    if values:
        dst[dst_idx] = torch.stack(values).to(dst.device)


def _copy_layer_first_to_page_head(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    page_size: int,
) -> None:
    src_idx = _cpu_indices(src_indices).to(src.device)
    dst_idx = _cpu_indices(dst_indices)
    for value, idx in zip(src[src_idx].to(dst.device), dst_idx.tolist(), strict=True):
        dst[idx // page_size, :, idx % page_size, layer_id] = value


def _unsupported_pointer_transfer(name: str) -> None:
    raise NotImplementedError(
        f"{name} received raw data_ptr tensors. The native simulator must route "
        "this through metadata movement in the simulator patch layer."
    )


def transfer_kv_per_layer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
) -> None:
    _copy_indexed(src_k, dst_k, src_indices, dst_indices)
    _copy_indexed(src_v, dst_v, src_indices, dst_indices)


def transfer_kv_per_layer_pf_lf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
) -> None:
    page_size = src_k.shape[2] if src_k.dim() >= 4 else 1
    _copy_page_first_to_layer_first(src_k, dst_k, src_indices, dst_indices, layer_id, page_size)
    _copy_page_first_to_layer_first(src_v, dst_v, src_indices, dst_indices, layer_id, page_size)


def transfer_kv_per_layer_ph_lf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    page_size: int,
    head_num: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
) -> None:
    _copy_page_head_to_layer_first(src_k, dst_k, src_indices, dst_indices, layer_id, page_size)
    _copy_page_head_to_layer_first(src_v, dst_v, src_indices, dst_indices, layer_id, page_size)


def transfer_kv_all_layer(
    src_k_layers: torch.Tensor,
    dst_k_layers: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
) -> None:
    _unsupported_pointer_transfer("transfer_kv_all_layer")


def transfer_kv_all_layer_lf_pf(
    src_k_layers: torch.Tensor,
    dst_k: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
) -> None:
    _unsupported_pointer_transfer("transfer_kv_all_layer_lf_pf")


def transfer_kv_all_layer_lf_ph(
    src_k_layers: torch.Tensor,
    dst_k: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    page_size: int,
    head_num: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
) -> None:
    _unsupported_pointer_transfer("transfer_kv_all_layer_lf_ph")


def transfer_kv_direct(
    src_layers: Iterable[torch.Tensor],
    dst_layers: Iterable[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
) -> None:
    for src, dst in zip(src_layers, dst_layers, strict=True):
        _copy_indexed(src, dst, src_indices, dst_indices)


def transfer_kv_per_layer_direct_pf_lf(
    src_ptrs: list[torch.Tensor],
    dst_ptrs: list[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    page_size: int,
) -> None:
    for src, dst in zip(src_ptrs, dst_ptrs, strict=True):
        _copy_page_first_to_layer_first(src, dst, src_indices, dst_indices, layer_id, page_size)


def transfer_kv_all_layer_direct_lf_pf(
    src_ptrs: list[torch.Tensor],
    dst_ptrs: list[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
) -> None:
    if len(dst_ptrs) == 1:
        for layer_id, src in enumerate(src_ptrs):
            _copy_layer_first_to_page_first(
                src, dst_ptrs[0], src_indices, dst_indices, layer_id, page_size
            )
        return

    half = len(src_ptrs) // 2
    for layer_id, src in enumerate(src_ptrs[:half]):
        _copy_layer_first_to_page_first(
            src, dst_ptrs[0], src_indices, dst_indices, layer_id, page_size
        )
    for layer_id, src in enumerate(src_ptrs[half:]):
        _copy_layer_first_to_page_first(
            src, dst_ptrs[1], src_indices, dst_indices, layer_id, page_size
        )


def transfer_kv_per_layer_mla(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
) -> None:
    _copy_indexed(src, dst, src_indices, dst_indices)


def transfer_kv_per_layer_mla_pf_lf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
) -> None:
    page_size = src.shape[2] if src.dim() >= 4 else 1
    _copy_page_first_to_layer_first(src, dst, src_indices, dst_indices, layer_id, page_size)


def transfer_kv_all_layer_mla(
    src_layers: torch.Tensor,
    dst_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
) -> None:
    _unsupported_pointer_transfer("transfer_kv_all_layer_mla")


def transfer_kv_all_layer_mla_lf_pf(
    src_layers: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
) -> None:
    _unsupported_pointer_transfer("transfer_kv_all_layer_mla_lf_pf")
