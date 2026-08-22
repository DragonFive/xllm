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


def test_proper_divisor_materialization_selects_one_replica_per_owner() -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend.device = torch.device("cpu")
    backend._block_table_i32 = torch.tensor([[3, 7, -1]], dtype=torch.int32)
    backend._kv_owner_representatives = None
    backend._materialized_block_table = None
    metadata = SimpleNamespace(
        has_kv_shard=True,
        kv_split_size=2,
        kv_split_rank=0,
    )
    cp_context = SimpleNamespace(cp_size=4)

    def all_gather(tensor: torch.Tensor, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        assert dim == 0
        assert world_size == 4
        assert group_name == "cp"
        if tensor.shape == (1,):
            return torch.tensor([0, 0, 1, 1], dtype=torch.int64)
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

    assert block_table.tolist() == [[0, 1, 2, 3, -1, -1]]
    torch.testing.assert_close(materialized[0], cache[3])
    torch.testing.assert_close(materialized[1], cache[3] + 200)
    torch.testing.assert_close(materialized[2], cache[7])
    torch.testing.assert_close(materialized[3], cache[7] + 200)


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


def test_mla_cp_uses_one_paged_sequence_per_zigzag_segment() -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._metadata = SimpleNamespace(
        has_kv_shard=True,
        local_slot_mapping=torch.arange(6, dtype=torch.int64),
    )
    backend._block_table_i32 = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    nope_cache = torch.zeros(4, 2, 2)
    rope_cache = torch.zeros(4, 2, 1)
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
            "_materialize_cp_cache",
            side_effect=[
                (nope_cache, backend._block_table_i32),
                (rope_cache, backend._block_table_i32),
            ],
        ),
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
