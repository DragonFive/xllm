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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from xllm.python.models import glm5_2


def test_glm52_config_preserves_moe_parallel_runtime_fields() -> None:
    cfg = glm5_2.Glm52Config.from_dict(
        {
            "ep_size": 16,
            "ep_rank": 7,
            "tp_size": 16,
            "dp_size": 1,
            "dp_rank": 0,
            "moe_tp_size": 1,
            "moe_tp_rank": 0,
            "world_size": 16,
            "n_routed_experts": 256,
        }
    )

    assert cfg.ep_size == 16
    assert cfg.ep_rank == 7
    assert cfg.dp_size == 1
    assert cfg.dp_rank == 0
    assert cfg.moe_tp_size == 1
    assert cfg.moe_tp_rank == 0
    assert cfg.world_size == 16
    cfg.validate()


class _Embedding(nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._events.append("embedding")
        return input_ids.to(torch.float32).unsqueeze(-1)


class _DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, events: list[str]) -> None:
        super().__init__()
        self._layer_id = layer_id
        self._events = events
        self.positions: torch.Tensor | None = None
        self.prev_topk: torch.Tensor | None = None
        self.output_topk: torch.Tensor | None = None

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        prev_topk: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del cos_sin_cache
        self.positions = positions
        self.prev_topk = prev_topk
        self.output_topk = hidden[:, :1].clone()
        if residual is None:
            residual = hidden
        self._events.append(f"cache_write_{self._layer_id}")
        return hidden + self._layer_id + 1, residual, self.output_topk


class _Norm(nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events
        self.input_rows: torch.Tensor | None = None

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._events.append("norm")
        self.input_rows = hidden
        return hidden + residual, residual


class _Rotary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("cos_sin_cache", torch.empty(0), persistent=False)


def _make_model(events: list[str]) -> tuple[glm5_2.Glm52Model, list[_DecoderLayer]]:
    model = glm5_2.Glm52Model.__new__(glm5_2.Glm52Model)
    nn.Module.__init__(model)
    layers = [_DecoderLayer(layer_id, events) for layer_id in range(2)]
    model.embed_tokens = _Embedding(events)
    model.layers = nn.ModuleList(layers)
    model.norm = _Norm(events)
    model.rotary = _Rotary()
    return model, layers


def test_cp_model_loop_shards_local_rows_and_merges_after_norm() -> None:
    events: list[str] = []
    model, layers = _make_model(events)
    cp_context = object()
    merged_output = torch.tensor([[23.0], [43.0], [63.0], [83.0]])

    def shard_rows(hidden: torch.Tensor, context: object) -> torch.Tensor:
        assert context is cp_context
        events.append("shard_rows")
        return hidden.index_select(0, torch.tensor([3, 0]))

    def shard_positions(positions: torch.Tensor, context: object) -> torch.Tensor:
        assert context is cp_context
        assert positions.dtype == torch.int64
        events.append("shard_positions")
        return positions.index_select(0, torch.tensor([3, 0]))

    def merge_rows(hidden: torch.Tensor, context: object) -> torch.Tensor:
        assert context is cp_context
        torch.testing.assert_close(hidden, torch.tensor([[83.0], [23.0]]))
        events.append("merge_rows")
        return merged_output

    def record_event(layer_id: int) -> bool:
        events.append(f"event_{layer_id}")
        return True

    with (
        patch.object(glm5_2, "get_forward_context", return_value=SimpleNamespace(cp_context=cp_context)),
        patch.object(glm5_2, "cp_shard_rows", side_effect=shard_rows),
        patch.object(glm5_2, "cp_shard_positions", side_effect=shard_positions),
        patch.object(glm5_2, "cp_merge_rows", side_effect=merge_rows),
        patch.object(glm5_2, "record_layer_event", side_effect=record_event),
    ):
        output = model(
            torch.tensor([10, 20, 30, 40]),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

    assert events == [
        "embedding",
        "shard_rows",
        "shard_positions",
        "cache_write_0",
        "event_0",
        "cache_write_1",
        "event_1",
        "norm",
        "merge_rows",
    ]
    torch.testing.assert_close(layers[0].positions, torch.tensor([3, 0]))
    assert layers[1].prev_topk is layers[0].output_topk
    torch.testing.assert_close(layers[1].prev_topk, torch.tensor([[40.0], [10.0]]))
    torch.testing.assert_close(output, merged_output)


def test_cp_one_preserves_full_rows_without_shard_or_merge() -> None:
    events: list[str] = []
    model, layers = _make_model(events)
    shard_rows = MagicMock()
    shard_positions = MagicMock()
    merge_rows = MagicMock()

    def record_event(layer_id: int) -> bool:
        events.append(f"event_{layer_id}")
        return True

    with (
        patch.object(glm5_2, "get_forward_context", return_value=SimpleNamespace(cp_context=None)),
        patch.object(glm5_2, "cp_shard_rows", shard_rows),
        patch.object(glm5_2, "cp_shard_positions", shard_positions),
        patch.object(glm5_2, "cp_merge_rows", merge_rows),
        patch.object(glm5_2, "record_layer_event", side_effect=record_event),
    ):
        output = model(
            torch.tensor([10, 20, 30, 40]),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

    shard_rows.assert_not_called()
    shard_positions.assert_not_called()
    merge_rows.assert_not_called()
    assert events == [
        "embedding",
        "cache_write_0",
        "event_0",
        "cache_write_1",
        "event_1",
        "norm",
    ]
    torch.testing.assert_close(layers[0].positions, torch.tensor([0, 1, 2, 3]))
    torch.testing.assert_close(output, torch.tensor([[23.0], [43.0], [63.0], [83.0]]))


def test_cp_ep_moe_materializes_global_rows_before_expert_reduction() -> None:
    moe = glm5_2.Glm52MoE.__new__(glm5_2.Glm52MoE)
    nn.Module.__init__(moe)
    moe.ep_size = 4
    cp_context = object()
    local_hidden = torch.tensor([[30.0], [10.0]])
    global_hidden = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    global_output = global_hidden + 100.0

    with (
        patch.object(
            glm5_2,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context),
        ),
        patch.object(
            glm5_2,
            "cp_gather_kv",
            return_value=global_hidden,
        ) as gather,
        patch.object(
            glm5_2.DeepseekV3MoE,
            "forward",
            return_value=global_output,
        ) as ep_forward,
        patch.object(
            glm5_2,
            "cp_shard_rows",
            return_value=torch.tensor([[130.0], [110.0]]),
        ) as shard,
    ):
        output = moe(local_hidden)

    gather.assert_called_once_with(local_hidden, cp_context)
    ep_forward.assert_called_once_with(global_hidden)
    shard.assert_called_once_with(global_output, cp_context)
    torch.testing.assert_close(output, torch.tensor([[130.0], [110.0]]))


class _FakeStateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


def _make_w8a8_mlp_holder() -> nn.Module:
    model = nn.Module()
    model.mlp = nn.Module()
    model.mlp.gate_up_proj = nn.Module()
    model.mlp.gate_up_proj.weight = nn.Parameter(torch.empty(8, 4, dtype=torch.int8), requires_grad=False)
    model.mlp.gate_up_proj.register_buffer("weight_scale", torch.empty(8, 1))
    model.mlp.gate_up_proj.register_buffer("weight_offset", torch.empty(8, 1))
    model.mlp.down_proj = nn.Module()
    model.mlp.down_proj.weight = nn.Parameter(torch.empty(4, 4, dtype=torch.int8), requires_grad=False)
    model.mlp.down_proj.register_buffer("weight_scale", torch.empty(4, 1))
    model.mlp.down_proj.register_buffer("weight_offset", torch.empty(4, 1))
    return model


def test_w8a8_mlp_loader_can_override_attention_tp_for_moe() -> None:
    model = _make_w8a8_mlp_holder()
    gate = torch.arange(32, dtype=torch.int8).view(8, 4)
    up = (100 + torch.arange(32, dtype=torch.int8)).view(8, 4)
    gate_scale = torch.arange(8, dtype=torch.float32).view(8, 1)
    up_scale = (100 + torch.arange(8, dtype=torch.float32)).view(8, 1)
    gate_offset = (200 + torch.arange(8, dtype=torch.float32)).view(8, 1)
    up_offset = (300 + torch.arange(8, dtype=torch.float32)).view(8, 1)
    down = torch.arange(32, dtype=torch.int8).view(4, 8)
    state_dict = _FakeStateDict(
        {
            "mlp.gate_proj.weight": gate,
            "mlp.gate_proj.weight_scale": gate_scale,
            "mlp.gate_proj.weight_offset": gate_offset,
            "mlp.up_proj.weight": up,
            "mlp.up_proj.weight_scale": up_scale,
            "mlp.up_proj.weight_offset": up_offset,
            "mlp.down_proj.weight": down,
            "mlp.down_proj.weight_scale": torch.ones(4, 1),
            "mlp.down_proj.weight_offset": torch.zeros(4, 1),
        }
    )

    loader = glm5_2.W8A8WeightLoader(model, [state_dict], tp_size=16, tp_rank=0)
    loader.load_w8a8_b("mlp.", shard_world=2, shard_rank=1)

    torch.testing.assert_close(
        model.mlp.gate_up_proj.weight,
        torch.cat([gate[4:], up[4:]], dim=0),
    )
    torch.testing.assert_close(
        model.mlp.gate_up_proj.weight_scale,
        torch.cat([gate_scale[4:], up_scale[4:]], dim=0),
    )
    torch.testing.assert_close(model.mlp.down_proj.weight, down[:, 4:])


def test_glm52_moe_tracks_ep_local_expert_range() -> None:
    cfg = glm5_2.Glm52Config(
        hidden_size=4,
        n_routed_experts=8,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        moe_intermediate_size=4,
        ep_size=2,
        ep_rank=1,
        world_size=2,
        moe_tp_size=1,
    )
    moe = glm5_2.Glm52MoE(cfg, 0, torch.float32, torch.device("cpu"))

    assert moe.num_local_experts == 4
    assert moe.local_expert_start == 4
    assert moe.local_expert_end == 8
