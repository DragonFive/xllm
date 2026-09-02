# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.modules.setdefault("torch_npu", types.ModuleType("torch_npu"))

from xllm.python.attention.npu_paged_attention import (  # noqa: E402
    NpuPagedAttentionBackend,
    _build_stable_sfa_page_layout,
)


def test_owner_local_index_write_ignores_non_owned_slots() -> None:
    cache = torch.full((2, 2, 1, 1), -1.0)
    slots = torch.tensor([0, -1, 1, -1, 2], dtype=torch.int64)
    values = torch.arange(5, dtype=torch.float32).view(-1, 1)

    def scatter(var: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor) -> None:
        var.index_copy_(0, indices.flatten(), updates)

    with patch(
        "xllm.python.attention.npu_paged_attention.kernels.scatter_nd_update",
        side_effect=scatter,
        create=True,
    ):
        NpuPagedAttentionBackend._update_mla_index_cache(
            cache,
            None,
            slots,
            values,
            None,
        )

    torch.testing.assert_close(cache.view(-1), torch.tensor([0.0, 2.0, 4.0, -1.0]))


@pytest.mark.parametrize(
    ("kv_split_size", "owner_by_cp_rank", "expanded_block_table"),
    [
        (2, [0, 0, 1, 1], [[6, 7, 14, 15, -1, -1]]),
        (4, [0, 1, 2, 3], [[12, 13, 14, 15, 28, 29, 30, 31, -1, -1, -1, -1]]),
    ],
)
def test_kv_owner_materialization_selects_one_replica_per_owner(
    kv_split_size: int,
    owner_by_cp_rank: list[int],
    expanded_block_table: list[list[int]],
) -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend.device = torch.device("cpu")
    backend._block_table_i32 = torch.tensor([[3, 7, -1]], dtype=torch.int32)
    backend._kv_owner_representatives = None
    backend._materialized_block_table = None
    metadata = SimpleNamespace(
        has_kv_shard=True,
        kv_split_size=kv_split_size,
        kv_split_rank=0,
        expanded_indexer_block_table=torch.tensor(expanded_block_table, dtype=torch.int32),
    )
    cp_context = SimpleNamespace(cp_size=4)

    def all_gather(tensor: torch.Tensor, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        assert dim == 0
        assert world_size == 4
        assert group_name == "cp"
        if tensor.shape == (1,):
            return torch.tensor(owner_by_cp_rank, dtype=torch.int64)
        rank_blocks = []
        for rank in range(4):
            rank_blocks.append(tensor + rank * 100)
        return torch.cat(rank_blocks, dim=0)

    cache = torch.arange(16, dtype=torch.float32).view(8, 2, 1)
    with (
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.cp_world_size",
            return_value=4,
            create=True,
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.all_gather",
            side_effect=all_gather,
            create=True,
        ),
    ):
        backend._prepare_kv_shard_materialization(metadata)
        materialized, block_table = backend._materialize_cp_cache(cache, metadata, cp_context)

    expected_block_table = [
        *range(2 * kv_split_size),
        *([-1] * kv_split_size),
    ]
    assert block_table.tolist() == [expected_block_table]
    for owner_rank in range(kv_split_size):
        representative_rank = owner_by_cp_rank.index(owner_rank)
        torch.testing.assert_close(
            materialized[owner_rank],
            cache[3] + representative_rank * 100,
        )
        torch.testing.assert_close(
            materialized[kv_split_size + owner_rank],
            cache[7] + representative_rank * 100,
        )


def test_kv1_shard_materialization_keeps_persistent_cache_view() -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend.device = torch.device("cpu")
    backend._block_table_i32 = torch.tensor([[3, 1, -1]], dtype=torch.int32)
    metadata = SimpleNamespace(
        has_kv_shard=True,
        kv_split_size=1,
        kv_split_rank=0,
    )
    cp_context = SimpleNamespace(cp_size=4)
    cache = torch.arange(8, dtype=torch.float32).view(4, 2, 1)

    with patch(
        "xllm.python.attention.npu_paged_attention.distributed.all_gather",
        create=True,
    ) as all_gather:
        backend._prepare_kv_shard_materialization(metadata)
        materialized, block_table = backend._materialize_cp_cache(cache, metadata, cp_context)

    all_gather.assert_not_called()
    assert materialized.data_ptr() == cache.data_ptr()
    assert block_table.data_ptr() == backend._block_table_i32.data_ptr()


def test_stable_sfa_layout_handles_multiple_sequences_and_invalid_tail() -> None:
    materialized_block_table = torch.tensor(
        [
            [4, 7, -1],
            [9, -1, -1],
            [3, 2, 8],
        ],
        dtype=torch.int32,
    )

    source_pages, target_pages, block_table, page_count = _build_stable_sfa_page_layout(materialized_block_table)

    assert source_pages.tolist() == [4, 7, 9, 3, 2, 8]
    assert target_pages.tolist() == [1, 0, 3, 7, 6, 8]
    assert block_table.tolist() == [
        [1, 0, -1],
        [3, -1, -1],
        [7, 6, 8],
    ]
    assert page_count == 9


def test_sparse_non_cp_prefill_keeps_persistent_cache_view() -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._metadata = SimpleNamespace(
        has_kv_shard=False,
        is_prefill=True,
        is_chunked_prefill=False,
    )
    backend._block_table_i32 = torch.tensor([[3, 1, -1]], dtype=torch.int32)
    nope_cache = torch.arange(8, dtype=torch.float32).view(4, 2, 1)
    rope_cache = nope_cache + 100
    backend._kv_caches = [SimpleNamespace(key=nope_cache, value=rope_cache)]
    backend._mla_actual_seq_q = torch.tensor([1], dtype=torch.int32)
    backend._mla_actual_seq_kv = torch.tensor([3], dtype=torch.int32)

    with (
        patch(
            "xllm.python.attention.npu_paged_attention.get_forward_context",
            return_value=SimpleNamespace(cp_context=None),
        ),
        patch.object(
            backend,
            "_mla_sparse",
            side_effect=lambda query, *_args: torch.ones_like(query),
        ) as sparse,
    ):
        backend.execute_mla(
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 1),
            None,
            None,
            SimpleNamespace(layer_id=0),
            torch.zeros(1, 1, dtype=torch.int32),
            cache_is_preprocessed=True,
        )

    sparse_args = sparse.call_args.args
    assert sparse_args[2].data_ptr() == nope_cache.data_ptr()
    assert sparse_args[3].data_ptr() == rope_cache.data_ptr()
    assert sparse_args[5].data_ptr() == backend._block_table_i32.data_ptr()


def test_sparse_decode_keeps_persistent_cache_view() -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._metadata = SimpleNamespace(
        has_kv_shard=False,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    backend._block_table_i32 = torch.tensor([[3, 1, -1]], dtype=torch.int32)
    nope_cache = torch.zeros(4, 2, 1)
    rope_cache = torch.zeros(4, 2, 1)
    backend._kv_caches = [SimpleNamespace(key=nope_cache, value=rope_cache)]
    backend._mla_actual_seq_q = torch.tensor([1], dtype=torch.int32)
    backend._mla_actual_seq_kv = torch.tensor([3], dtype=torch.int32)

    with (
        patch(
            "xllm.python.attention.npu_paged_attention.get_forward_context",
            return_value=SimpleNamespace(cp_context=None),
        ),
        patch.object(
            backend,
            "_mla_sparse",
            side_effect=lambda query, *_args: torch.ones_like(query),
        ) as sparse,
    ):
        backend.execute_mla(
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 1),
            None,
            None,
            SimpleNamespace(layer_id=0),
            torch.zeros(1, 1, dtype=torch.int32),
            cache_is_preprocessed=True,
        )

    sparse_args = sparse.call_args.args
    assert sparse_args[2].data_ptr() == nope_cache.data_ptr()
    assert sparse_args[3].data_ptr() == rope_cache.data_ptr()
    assert sparse_args[5].data_ptr() == backend._block_table_i32.data_ptr()


def test_kv2_sparse_attention_uses_runtime_expanded_layout_after_real_materialization() -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend.device = torch.device("cpu")
    backend._block_table_i32 = torch.tensor(
        [
            [2, 0, -1],
            [3, 1, -1],
        ],
        dtype=torch.int32,
    )
    metadata = SimpleNamespace(
        has_kv_shard=True,
        kv_split_size=2,
        kv_split_rank=0,
        kv_split_block_size=2,
        local_slot_mapping=torch.arange(6, dtype=torch.int64),
        slot_mapping=torch.arange(6, dtype=torch.int64),
        expanded_indexer_block_table=torch.tensor(
            [
                [4, 5, 0, 1, -1, -1],
                [6, 7, 2, 3, -1, -1],
            ],
            dtype=torch.int32,
        ),
    )
    backend._metadata = metadata
    backend._mla_actual_seq_q = torch.tensor([1, 2], dtype=torch.int32)
    backend._mla_actual_seq_kv = torch.tensor([4, 8], dtype=torch.int32)
    index_cache = torch.arange(8, dtype=torch.float32).view(4, 2, 1)
    nope_cache = index_cache + 10
    rope_cache = index_cache + 20
    backend._kv_caches = [
        SimpleNamespace(
            key=nope_cache,
            value=rope_cache,
            index=index_cache,
            index_scale=None,
        )
    ]
    cp_context = SimpleNamespace(
        cp_size=4,
        query_index=torch.tensor([0, 1], dtype=torch.int64),
        segment_seq_indices=torch.tensor([0, 1], dtype=torch.int64),
        q_cu_seqlens=[1, 2],
        segment_kv_seq_lens=[4, 8],
    )

    def all_gather(tensor: torch.Tensor, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        assert dim == 0
        assert world_size == 4
        assert group_name == "cp"
        if tensor.shape == (1,):
            return torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        return torch.cat([tensor, tensor, tensor, tensor], dim=0)

    with (
        patch(
            "xllm.python.attention.npu_paged_attention.get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context),
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.all_gather",
            side_effect=all_gather,
            create=True,
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.cp_world_size",
            return_value=4,
            create=True,
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.cp_gather_kv",
            side_effect=lambda tensor, _context: tensor,
        ),
        patch.object(torch.ops.xllm_ops, "reshape_paged_cache", create=True),
        patch.object(
            backend,
            "_mla_sparse",
            side_effect=lambda query, *_args: torch.ones_like(query),
        ) as sparse,
    ):
        backend._prepare_kv_shard_materialization(metadata)
        index_context = backend.mla_index_context(SimpleNamespace(layer_id=0))
        materialized_index_cache, indexer_block_table = index_context.materialize_index_cache()
        backend.execute_mla(
            torch.zeros(2, 1, 1),
            torch.zeros(2, 1, 1),
            torch.zeros(2, 1, 1),
            torch.zeros(2, 1, 1),
            SimpleNamespace(layer_id=0),
            torch.zeros(2, 1, dtype=torch.int32),
        )

    sparse_block_table = sparse.call_args.args[5]
    sparse_nope_cache = sparse.call_args.args[2]
    sparse_rope_cache = sparse.call_args.args[3]
    assert materialized_index_cache.shape[0] == 12
    assert indexer_block_table.data_ptr() == backend._materialized_block_table.data_ptr()
    assert indexer_block_table.tolist() == [
        [0, 1, 2, 3, -1, -1],
        [6, 7, 8, 9, -1, -1],
    ]
    assert sparse_block_table.data_ptr() != indexer_block_table.data_ptr()
    assert sparse_block_table.tolist() == [
        [1, 0, 2, 3, -1, -1],
        [7, 6, 8, 9, -1, -1],
    ]
    assert sparse_nope_cache.shape[0] == 12
    source_pages = backend._sfa_source_page_ids
    target_pages = backend._sfa_target_page_ids
    assert source_pages is not None
    assert target_pages is not None
    expected_nope = torch.zeros_like(sparse_nope_cache)
    expected_nope.index_copy_(
        0,
        target_pages,
        (materialized_index_cache + 10).index_select(0, source_pages),
    )
    torch.testing.assert_close(sparse_nope_cache, expected_nope)
    torch.testing.assert_close(sparse_rope_cache, expected_nope + (expected_nope != 0) * 10)


def test_materialization_rejects_incomplete_owner_distribution() -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend.device = torch.device("cpu")
    backend._block_table_i32 = torch.tensor([[0]], dtype=torch.int32)
    metadata = SimpleNamespace(
        has_kv_shard=True,
        kv_split_size=2,
        kv_split_rank=0,
    )

    with (
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.cp_world_size",
            return_value=4,
            create=True,
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.all_gather",
            return_value=torch.tensor([0, 0, 0, 1], dtype=torch.int64),
            create=True,
        ),
        pytest.raises(RuntimeError, match="KV owner distribution"),
    ):
        backend._prepare_kv_shard_materialization(metadata)


@pytest.mark.parametrize("has_kv_shard", [False, True])
def test_mla_cp_uses_one_paged_sequence_per_zigzag_segment(has_kv_shard: bool) -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._metadata = SimpleNamespace(
        has_kv_shard=has_kv_shard,
        kv_split_size=1,
        local_slot_mapping=torch.arange(6, dtype=torch.int64),
        slot_mapping=torch.arange(6, dtype=torch.int64),
        kv_split_block_size=8,
    )
    backend._block_table_i32 = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    nope_cache = torch.zeros(4, 8, 2)
    rope_cache = torch.zeros(4, 8, 1)
    backend._kv_caches = [SimpleNamespace(key=nope_cache, value=rope_cache)]
    backend._mla_actual_seq_q = torch.tensor([3, 6], dtype=torch.int32)
    backend._mla_actual_seq_kv = torch.tensor([12, 7], dtype=torch.int32)

    cp_context = SimpleNamespace(
        query_index=torch.tensor([0, 1, 3, 4, 5], dtype=torch.int64),
        segment_seq_indices=torch.tensor([0, 0, 1], dtype=torch.int64),
        q_cu_seqlens=[2, 3, 5],
        segment_kv_seq_lens=[10, 12, 7],
    )
    q_latent = torch.zeros(6, 1, 2)
    q_pe = torch.zeros(6, 1, 1)
    k_latent = torch.zeros(6, 1, 2)
    k_pe = torch.zeros(6, 1, 1)
    topk = torch.arange(6, dtype=torch.int32).view(6, 1)

    with (
        patch(
            "xllm.python.attention.npu_paged_attention.get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context),
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.cp_gather_kv",
            side_effect=lambda tensor, _context: tensor,
        ),
        patch.object(torch.ops.xllm_ops, "reshape_paged_cache", create=True),
        patch.object(
            backend,
            "_mla_sparse",
            side_effect=lambda query, *_args: torch.ones_like(query),
        ) as sparse,
    ):
        output = backend.execute_mla(
            q_latent,
            q_pe,
            k_latent,
            k_pe,
            SimpleNamespace(layer_id=0),
            topk,
        )

    sparse_args = sparse.call_args.args
    torch.testing.assert_close(
        sparse_args[5],
        torch.tensor([[0, 1], [0, 1], [2, 3]], dtype=torch.int32),
    )
    assert sparse_args[2].data_ptr() == nope_cache.data_ptr()
    assert sparse_args[3].data_ptr() == rope_cache.data_ptr()
    torch.testing.assert_close(
        sparse_args[6],
        torch.tensor([2, 3, 5], dtype=torch.int32),
    )
    torch.testing.assert_close(
        sparse_args[7],
        torch.tensor([10, 12, 7], dtype=torch.int32),
    )
    torch.testing.assert_close(output[cp_context.query_index], torch.ones(5, 1, 2))
    torch.testing.assert_close(output[2], torch.zeros(1, 2))
