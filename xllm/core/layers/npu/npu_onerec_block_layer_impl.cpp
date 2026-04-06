/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "npu_onerec_block_layer_impl.h"

#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include <algorithm>
#include <cstring>
#include <set>

#include "common/global_flags.h"
namespace xllm {
namespace layer {
namespace {

// Decoder normal mode: self-attn(29) + cross-attn(28) + layer-norm(4) + mlp(18)
// = 79
static constexpr uint64_t kOneRecWeightCountPerLayer = 79;

// Decoder MoE mode weights count (exclude runtime tensors like expert_array).
static constexpr uint64_t kOneRecMoeWeightCountPerLayer = 97;

enum class OneRecBlockLayerTensorId : int32_t {
  // Self-attention layer norm
  IN_LAYER_NORM_WEIGHT = 0,
  IN_LAYER_NORM_BIAS,
  IN_INPUT_NORM_NEW_WEIGHT,
  IN_INPUT_NORM_NEW_BIAS,
  // Self-attention Q, K, V projections
  IN_Q_WEIGHT,
  IN_Q_BIAS,
  IN_Q_DEQSCALE,
  IN_Q_OFFSET,
  IN_Q_SCALE,
  IN_Q_COMPRESS_IDX,

  IN_K_WEIGHT,
  IN_K_BIAS,
  IN_K_DEQSCALE,
  IN_K_OFFSET,
  IN_K_SCALE,
  IN_K_COMPRESS_IDX,

  IN_V_WEIGHT,
  IN_V_BIAS,
  IN_V_DEQSCALE,
  IN_V_OFFSET,
  IN_V_SCALE,
  IN_V_COMPRESS_IDX,

  // Self-attention output projection
  IN_SELF_ATTN_OUT_WEIGHT,
  IN_SELF_ATTN_OUT_BIAS,
  IN_SELF_ATTN_OUT_DEQSCALE,
  IN_SELF_ATTN_OUT_OFFSET,
  IN_SELF_ATTN_OUT_SCALE,
  IN_SELF_ATTN_OUT_COMPRESS_IDX,

  // ONEREC relative attention bias (encoder only)
  IN_RELATIVE_ATTENTION_BIAS_WEIGHT,

  // Cross-attention layer norm (decoder only)
  IN_CROSS_LAYER_NORM_WEIGHT,
  IN_CROSS_LAYER_NORM_BIAS,
  IN_CROSS_LAYER_NORM_NEW_WEIGHT,
  IN_CROSS_LAYER_NORM_NEW_BIAS,

  // Cross-attention Q, K, V projections (decoder only)
  IN_CROSS_Q_WEIGHT,
  IN_CROSS_Q_BIAS,
  IN_CROSS_Q_DEQSCALE,
  IN_CROSS_Q_OFFSET,
  IN_CROSS_Q_SCALE,
  IN_CROSS_Q_COMPRESS_IDX,

  IN_CROSS_K_WEIGHT,
  IN_CROSS_K_BIAS,
  IN_CROSS_K_DEQSCALE,
  IN_CROSS_K_OFFSET,
  IN_CROSS_K_SCALE,
  IN_CROSS_K_COMPRESS_IDX,

  IN_CROSS_V_WEIGHT,
  IN_CROSS_V_BIAS,
  IN_CROSS_V_DEQSCALE,
  IN_CROSS_V_OFFSET,
  IN_CROSS_V_SCALE,
  IN_CROSS_V_COMPRESS_IDX,

  // Cross-attention output projection (decoder only)
  IN_CROSS_ATTN_OUT_WEIGHT,
  IN_CROSS_ATTN_OUT_BIAS,
  IN_CROSS_ATTN_OUT_DEQSCALE,
  IN_CROSS_ATTN_OUT_OFFSET,
  IN_CROSS_ATTN_OUT_SCALE,
  IN_CROSS_ATTN_OUT_COMPRESS_IDX,

  // Final layer norm
  IN_FINAL_LAYER_NORM_WEIGHT,
  IN_FINAL_LAYER_NORM_BIAS,
  IN_FINAL_LAYER_NORM_NEW_WEIGHT,
  IN_FINAL_LAYER_NORM_NEW_BIAS,

  // Feed-forward network (gated activation)
  IN_FFN_WI_0_WEIGHT = 61,  // wi_0 (gate projection)
  IN_FFN_WI_0_BIAS,
  IN_FFN_WI_0_DEQSCALE,
  IN_FFN_WI_0_OFFSET,
  IN_FFN_WI_0_SCALE,
  IN_FFN_WI_0_COMPRESS_IDX,

  IN_FFN_WI_1_WEIGHT,  // wi_1 (up projection)
  IN_FFN_WI_1_BIAS,
  IN_FFN_WI_1_DEQSCALE,
  IN_FFN_WI_1_OFFSET,
  IN_FFN_WI_1_SCALE,
  IN_FFN_WI_1_COMPRESS_IDX,

  IN_FFN_WO_WEIGHT,  // wo (down projection)
  IN_FFN_WO_BIAS,
  IN_FFN_WO_DEQSCALE,
  IN_FFN_WO_OFFSET,
  IN_FFN_WO_SCALE,
  IN_FFN_WO_COMPRESS_IDX,
};

constexpr int32_t IN_LAYER_NORM_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_LAYER_NORM_WEIGHT);
constexpr int32_t IN_LAYER_NORM_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_LAYER_NORM_BIAS);
constexpr int32_t IN_INPUT_NORM_NEW_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_INPUT_NORM_NEW_WEIGHT);
constexpr int32_t IN_INPUT_NORM_NEW_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_INPUT_NORM_NEW_BIAS);
constexpr int32_t IN_Q_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_Q_WEIGHT);
constexpr int32_t IN_Q_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_Q_BIAS);
constexpr int32_t IN_Q_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_Q_DEQSCALE);
constexpr int32_t IN_Q_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_Q_OFFSET);
constexpr int32_t IN_Q_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_Q_SCALE);
constexpr int32_t IN_Q_COMPRESS_IDX =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_Q_COMPRESS_IDX);
constexpr int32_t IN_K_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_K_WEIGHT);
constexpr int32_t IN_K_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_K_BIAS);
constexpr int32_t IN_K_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_K_DEQSCALE);
constexpr int32_t IN_K_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_K_OFFSET);
constexpr int32_t IN_K_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_K_SCALE);
constexpr int32_t IN_K_COMPRESS_IDX =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_K_COMPRESS_IDX);
constexpr int32_t IN_V_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_V_WEIGHT);
constexpr int32_t IN_V_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_V_BIAS);
constexpr int32_t IN_V_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_V_DEQSCALE);
constexpr int32_t IN_V_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_V_OFFSET);
constexpr int32_t IN_V_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_V_SCALE);
constexpr int32_t IN_V_COMPRESS_IDX =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_V_COMPRESS_IDX);
constexpr int32_t IN_SELF_ATTN_OUT_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_SELF_ATTN_OUT_WEIGHT);
constexpr int32_t IN_SELF_ATTN_OUT_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_SELF_ATTN_OUT_BIAS);
constexpr int32_t IN_SELF_ATTN_OUT_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_SELF_ATTN_OUT_DEQSCALE);
constexpr int32_t IN_SELF_ATTN_OUT_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_SELF_ATTN_OUT_OFFSET);
constexpr int32_t IN_SELF_ATTN_OUT_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_SELF_ATTN_OUT_SCALE);
constexpr int32_t IN_SELF_ATTN_OUT_COMPRESS_IDX = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_SELF_ATTN_OUT_COMPRESS_IDX);
constexpr int32_t IN_RELATIVE_ATTENTION_BIAS_WEIGHT = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_RELATIVE_ATTENTION_BIAS_WEIGHT);
constexpr int32_t IN_CROSS_LAYER_NORM_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_LAYER_NORM_WEIGHT);
constexpr int32_t IN_CROSS_LAYER_NORM_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_LAYER_NORM_BIAS);
constexpr int32_t IN_CROSS_LAYER_NORM_NEW_WEIGHT = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_CROSS_LAYER_NORM_NEW_WEIGHT);
constexpr int32_t IN_CROSS_LAYER_NORM_NEW_BIAS = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_CROSS_LAYER_NORM_NEW_BIAS);
constexpr int32_t IN_CROSS_Q_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_Q_WEIGHT);
constexpr int32_t IN_CROSS_Q_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_Q_BIAS);
constexpr int32_t IN_CROSS_Q_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_Q_DEQSCALE);
constexpr int32_t IN_CROSS_Q_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_Q_OFFSET);
constexpr int32_t IN_CROSS_Q_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_Q_SCALE);
constexpr int32_t IN_CROSS_Q_COMPRESS_IDX =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_Q_COMPRESS_IDX);
constexpr int32_t IN_CROSS_K_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_K_WEIGHT);
constexpr int32_t IN_CROSS_K_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_K_BIAS);
constexpr int32_t IN_CROSS_K_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_K_DEQSCALE);
constexpr int32_t IN_CROSS_K_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_K_OFFSET);
constexpr int32_t IN_CROSS_K_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_K_SCALE);
constexpr int32_t IN_CROSS_K_COMPRESS_IDX =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_K_COMPRESS_IDX);
constexpr int32_t IN_CROSS_V_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_V_WEIGHT);
constexpr int32_t IN_CROSS_V_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_V_BIAS);
constexpr int32_t IN_CROSS_V_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_V_DEQSCALE);
constexpr int32_t IN_CROSS_V_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_V_OFFSET);
constexpr int32_t IN_CROSS_V_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_V_SCALE);
constexpr int32_t IN_CROSS_V_COMPRESS_IDX =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_V_COMPRESS_IDX);
constexpr int32_t IN_CROSS_ATTN_OUT_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_ATTN_OUT_WEIGHT);
constexpr int32_t IN_CROSS_ATTN_OUT_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_ATTN_OUT_BIAS);
constexpr int32_t IN_CROSS_ATTN_OUT_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_ATTN_OUT_DEQSCALE);
constexpr int32_t IN_CROSS_ATTN_OUT_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_ATTN_OUT_OFFSET);
constexpr int32_t IN_CROSS_ATTN_OUT_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_ATTN_OUT_SCALE);
constexpr int32_t IN_CROSS_ATTN_OUT_COMPRESS_IDX = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_CROSS_ATTN_OUT_COMPRESS_IDX);
constexpr int32_t IN_FINAL_LAYER_NORM_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FINAL_LAYER_NORM_WEIGHT);
constexpr int32_t IN_FINAL_LAYER_NORM_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FINAL_LAYER_NORM_BIAS);
constexpr int32_t IN_FINAL_LAYER_NORM_NEW_WEIGHT = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_FINAL_LAYER_NORM_NEW_WEIGHT);
constexpr int32_t IN_FINAL_LAYER_NORM_NEW_BIAS = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_FINAL_LAYER_NORM_NEW_BIAS);
constexpr int32_t IN_FFN_WI_0_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_0_WEIGHT);
constexpr int32_t IN_FFN_WI_0_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_0_BIAS);
constexpr int32_t IN_FFN_WI_0_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_0_DEQSCALE);
constexpr int32_t IN_FFN_WI_0_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_0_OFFSET);
constexpr int32_t IN_FFN_WI_0_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_0_SCALE);
constexpr int32_t IN_FFN_WI_0_COMPRESS_IDX =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_0_COMPRESS_IDX);
constexpr int32_t IN_FFN_WI_1_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_1_WEIGHT);
constexpr int32_t IN_FFN_WI_1_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_1_BIAS);
constexpr int32_t IN_FFN_WI_1_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_1_DEQSCALE);
constexpr int32_t IN_FFN_WI_1_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_1_OFFSET);
constexpr int32_t IN_FFN_WI_1_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_1_SCALE);
constexpr int32_t IN_FFN_WI_1_COMPRESS_IDX =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_1_COMPRESS_IDX);
constexpr int32_t IN_FFN_WO_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WO_WEIGHT);
constexpr int32_t IN_FFN_WO_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WO_BIAS);
constexpr int32_t IN_FFN_WO_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WO_DEQSCALE);
constexpr int32_t IN_FFN_WO_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WO_OFFSET);
constexpr int32_t IN_FFN_WO_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WO_SCALE);
constexpr int32_t IN_FFN_WO_COMPRESS_IDX =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WO_COMPRESS_IDX);

enum class OneRecMoeBlockLayerTensorId : int32_t {
  // MoE weights (only used when use_moe=true)
  IN_BLOCK_SPARSE_MOE_GATE_WEIGHT = 61,   // routing weights
  IN_BLOCK_SPARSE_MOE_GATE_BIAS = 62,     // routing bias
  IN_BLOCK_SPARSE_MOE_GATE_DESCALE,       // gate descale
  IN_BLOCK_SPARSE_MOE_GATE_OFFSET,        // gate offset
  IN_BLOCK_SPARSE_MOE_GATE_SCALE,         // gate scale
  IN_BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX,  // gate compress index

  // Shared expert weights
  IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,
  IN_MLP_GATEUP_BIAS_SHARED_EXPERT,
  IN_MLP_GATEUP_DESCALE_SHARED_EXPERT,
  IN_MLP_GATEUP_OFFSET_SHARED_EXPERT,
  IN_MLP_GATEUP_SCALE_SHARED_EXPERT,
  IN_MLP_GATEUP_COMPRESS_IDX_SHARED_EXPERT,

  IN_MLP_DOWN_WEIGHT_SHARED_EXPERT,
  IN_MLP_DOWN_BIAS_SHARED_EXPERT,
  IN_MLP_DOWN_DESCALE_SHARED_EXPERT,
  IN_MLP_DOWN_OFFSET_SHARED_EXPERT,
  IN_MLP_DOWN_SCALE_SHARED_EXPERT,
  IN_MLP_DOWN_COMPRESS_IDX_SHARED_EXPERT,

  // Shared expert gate weights
  IN_SHARED_EXPERT_GATE_WEIGHT,
  IN_SHARED_EXPERT_GATE_BIAS,
  IN_SHARED_EXPERT_GATE_DESCALE,
  IN_SHARED_EXPERT_GATE_OFFSET,
  IN_SHARED_EXPERT_GATE_SCALE,
  IN_SHARED_EXPERT_GATE_COMPRESS_IDX,

  // Expert weights
  IN_MLP_GATEUP_WEIGHT_EXPERT,
  IN_MLP_GATEUP_BIAS_EXPERT,
  IN_MLP_GATEUP_DESCALE_EXPERT,
  IN_MLP_GATEUP_OFFSET_EXPERT,
  IN_MLP_GATEUP_SCALE_EXPERT,
  IN_MLP_GATEUP_COMPRESS_IDX_EXPERT,

  IN_MLP_DOWN_WEIGHT_EXPERT,
  IN_MLP_DOWN_BIAS_EXPERT,
  IN_MLP_DOWN_DESCALE_EXPERT,
  IN_MLP_DOWN_OFFSET_EXPERT,
  IN_MLP_DOWN_SCALE_EXPERT,
  IN_MLP_DOWN_COMPRESS_IDX_EXPERT = 96,

  // Runtime tensors (not part of weight tensor array)
  IN_EXPERT_ARRAY = 97,
  IN_EXPERT_GROUP = 98,
  IN_ONE_HOT = 99,
  IN_ZERO_HOT = 100,

  // Legacy aliases for backward compatibility
  IN_MOE_EXPERT_W1_WEIGHT = IN_MLP_GATEUP_WEIGHT_EXPERT,
  IN_MOE_EXPERT_W2_WEIGHT = IN_MLP_DOWN_WEIGHT_EXPERT,
  IN_MOE_EXPERT_W3_WEIGHT = IN_MLP_GATEUP_WEIGHT_EXPERT,
  IN_MOE_SHARED_W1_WEIGHT = IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,
  IN_MOE_SHARED_W2_WEIGHT = IN_MLP_DOWN_WEIGHT_SHARED_EXPERT,
};

constexpr int32_t IN_BLOCK_SPARSE_MOE_GATE_WEIGHT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_BLOCK_SPARSE_MOE_GATE_WEIGHT);
constexpr int32_t IN_BLOCK_SPARSE_MOE_GATE_BIAS = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_BLOCK_SPARSE_MOE_GATE_BIAS);
constexpr int32_t IN_BLOCK_SPARSE_MOE_GATE_DESCALE = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_BLOCK_SPARSE_MOE_GATE_DESCALE);
constexpr int32_t IN_BLOCK_SPARSE_MOE_GATE_OFFSET = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_BLOCK_SPARSE_MOE_GATE_OFFSET);
constexpr int32_t IN_BLOCK_SPARSE_MOE_GATE_SCALE = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_BLOCK_SPARSE_MOE_GATE_SCALE);
constexpr int32_t IN_BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX);
constexpr int32_t IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT);
constexpr int32_t IN_MLP_GATEUP_BIAS_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_BIAS_SHARED_EXPERT);
constexpr int32_t IN_MLP_GATEUP_DESCALE_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_DESCALE_SHARED_EXPERT);
constexpr int32_t IN_MLP_GATEUP_OFFSET_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_OFFSET_SHARED_EXPERT);
constexpr int32_t IN_MLP_GATEUP_SCALE_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_SCALE_SHARED_EXPERT);
constexpr int32_t IN_MLP_GATEUP_COMPRESS_IDX_SHARED_EXPERT =
    static_cast<int32_t>(
        OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_COMPRESS_IDX_SHARED_EXPERT);
constexpr int32_t IN_MLP_DOWN_WEIGHT_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_WEIGHT_SHARED_EXPERT);
constexpr int32_t IN_MLP_DOWN_BIAS_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_BIAS_SHARED_EXPERT);
constexpr int32_t IN_MLP_DOWN_DESCALE_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_DESCALE_SHARED_EXPERT);
constexpr int32_t IN_MLP_DOWN_OFFSET_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_OFFSET_SHARED_EXPERT);
constexpr int32_t IN_MLP_DOWN_SCALE_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_SCALE_SHARED_EXPERT);
constexpr int32_t IN_MLP_DOWN_COMPRESS_IDX_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_COMPRESS_IDX_SHARED_EXPERT);
constexpr int32_t IN_SHARED_EXPERT_GATE_WEIGHT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_SHARED_EXPERT_GATE_WEIGHT);
constexpr int32_t IN_SHARED_EXPERT_GATE_BIAS = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_SHARED_EXPERT_GATE_BIAS);
constexpr int32_t IN_SHARED_EXPERT_GATE_DESCALE = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_SHARED_EXPERT_GATE_DESCALE);
constexpr int32_t IN_SHARED_EXPERT_GATE_OFFSET = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_SHARED_EXPERT_GATE_OFFSET);
constexpr int32_t IN_SHARED_EXPERT_GATE_SCALE = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_SHARED_EXPERT_GATE_SCALE);
constexpr int32_t IN_SHARED_EXPERT_GATE_COMPRESS_IDX = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_SHARED_EXPERT_GATE_COMPRESS_IDX);
constexpr int32_t IN_MLP_GATEUP_WEIGHT_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_WEIGHT_EXPERT);
constexpr int32_t IN_MLP_GATEUP_BIAS_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_BIAS_EXPERT);
constexpr int32_t IN_MLP_GATEUP_DESCALE_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_DESCALE_EXPERT);
constexpr int32_t IN_MLP_GATEUP_OFFSET_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_OFFSET_EXPERT);
constexpr int32_t IN_MLP_GATEUP_SCALE_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_SCALE_EXPERT);
constexpr int32_t IN_MLP_GATEUP_COMPRESS_IDX_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_COMPRESS_IDX_EXPERT);
constexpr int32_t IN_MLP_DOWN_WEIGHT_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_WEIGHT_EXPERT);
constexpr int32_t IN_MLP_DOWN_BIAS_EXPERT =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_BIAS_EXPERT);
constexpr int32_t IN_MLP_DOWN_DESCALE_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_DESCALE_EXPERT);
constexpr int32_t IN_MLP_DOWN_OFFSET_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_OFFSET_EXPERT);
constexpr int32_t IN_MLP_DOWN_SCALE_EXPERT =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_SCALE_EXPERT);
constexpr int32_t IN_MLP_DOWN_COMPRESS_IDX_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_COMPRESS_IDX_EXPERT);
constexpr int32_t IN_EXPERT_ARRAY =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_EXPERT_ARRAY);
constexpr int32_t IN_EXPERT_GROUP =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_EXPERT_GROUP);
constexpr int32_t IN_ONE_HOT =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_ONE_HOT);
constexpr int32_t IN_ZERO_HOT =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_ZERO_HOT);
constexpr int32_t IN_MOE_EXPERT_W1_WEIGHT =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MOE_EXPERT_W1_WEIGHT);
constexpr int32_t IN_MOE_EXPERT_W2_WEIGHT =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MOE_EXPERT_W2_WEIGHT);
constexpr int32_t IN_MOE_EXPERT_W3_WEIGHT =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MOE_EXPERT_W3_WEIGHT);
constexpr int32_t IN_MOE_SHARED_W1_WEIGHT =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MOE_SHARED_W1_WEIGHT);
constexpr int32_t IN_MOE_SHARED_W2_WEIGHT =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MOE_SHARED_W2_WEIGHT);

static const std::unordered_map<std::string, int32_t>
    kOneRecEncoderWeightMapping = {
        {"layer.0.layer_norm.weight", IN_LAYER_NORM_WEIGHT},
        {"layer.0.SelfAttention.q.weight", IN_Q_WEIGHT},
        {"layer.0.SelfAttention.k.weight", IN_K_WEIGHT},
        {"layer.0.SelfAttention.v.weight", IN_V_WEIGHT},
        {"layer.0.SelfAttention.o.weight", IN_SELF_ATTN_OUT_WEIGHT},
        {"layer.0.SelfAttention.relative_attention_bias.weight",
         IN_RELATIVE_ATTENTION_BIAS_WEIGHT},
        {"layer.1.layer_norm.weight", IN_FINAL_LAYER_NORM_WEIGHT},
        {"layer.1.DenseReluDense.wi.weight", IN_FFN_WI_1_WEIGHT},
        {"layer.1.DenseReluDense.wo.weight", IN_FFN_WO_WEIGHT},
        {"layer.1.DenseReluDense.gate_proj.weight", IN_FFN_WI_0_WEIGHT},
        {"layer.1.ffn.wi.weight", IN_FFN_WI_1_WEIGHT},
        {"layer.1.ffn.wo.weight", IN_FFN_WO_WEIGHT},
        {"layer.1.ffn.gate_proj.weight", IN_FFN_WI_0_WEIGHT},
        // Alternative format
        {"0.layer_norm.weight", IN_LAYER_NORM_WEIGHT},
        {"0.SelfAttention.q.weight", IN_Q_WEIGHT},
        {"0.SelfAttention.k.weight", IN_K_WEIGHT},
        {"0.SelfAttention.v.weight", IN_V_WEIGHT},
        {"0.SelfAttention.o.weight", IN_SELF_ATTN_OUT_WEIGHT},
        {"0.SelfAttention.relative_attention_bias.weight",
         IN_RELATIVE_ATTENTION_BIAS_WEIGHT},
        {"1.layer_norm.weight", IN_FINAL_LAYER_NORM_WEIGHT},
        {"1.DenseReluDense.wi.weight", IN_FFN_WI_1_WEIGHT},
        {"1.DenseReluDense.wo.weight", IN_FFN_WO_WEIGHT},
        {"1.DenseReluDense.gate_proj.weight", IN_FFN_WI_0_WEIGHT},
        {"1.ffn.wi.weight", IN_FFN_WI_1_WEIGHT},
        {"1.ffn.wo.weight", IN_FFN_WO_WEIGHT},
        {"1.ffn.gate_proj.weight", IN_FFN_WI_0_WEIGHT},
};

static const std::unordered_map<std::string, int32_t>
    kOneRecDecoderWeightMapping = {
        {"layer.0.layer_norm.weight", IN_LAYER_NORM_WEIGHT},
        {"layer.0.SelfAttention.q.weight", IN_Q_WEIGHT},
        {"layer.0.SelfAttention.k.weight", IN_K_WEIGHT},
        {"layer.0.SelfAttention.v.weight", IN_V_WEIGHT},
        {"layer.0.SelfAttention.o.weight", IN_SELF_ATTN_OUT_WEIGHT},
        {"layer.0.SelfAttention.relative_attention_bias.weight",
         IN_RELATIVE_ATTENTION_BIAS_WEIGHT},
        {"layer.1.layer_norm.weight", IN_CROSS_LAYER_NORM_WEIGHT},
        {"layer.1.EncDecAttention.q.weight", IN_CROSS_Q_WEIGHT},
        {"layer.1.EncDecAttention.k.weight", IN_CROSS_K_WEIGHT},
        {"layer.1.EncDecAttention.v.weight", IN_CROSS_V_WEIGHT},
        {"layer.1.EncDecAttention.o.weight", IN_CROSS_ATTN_OUT_WEIGHT},
        {"layer.2.layer_norm.weight", IN_FINAL_LAYER_NORM_WEIGHT},
        {"layer.2.DenseReluDense.wi.weight", IN_FFN_WI_1_WEIGHT},
        {"layer.2.DenseReluDense.wo.weight", IN_FFN_WO_WEIGHT},
        {"layer.2.DenseReluDense.gate_proj.weight", IN_FFN_WI_0_WEIGHT},
        // Alternative format
        {"0.layer_norm.weight", IN_LAYER_NORM_WEIGHT},
        {"0.SelfAttention.q.weight", IN_Q_WEIGHT},
        {"0.SelfAttention.k.weight", IN_K_WEIGHT},
        {"0.SelfAttention.v.weight", IN_V_WEIGHT},
        {"0.SelfAttention.o.weight", IN_SELF_ATTN_OUT_WEIGHT},
        {"0.SelfAttention.relative_attention_bias.weight",
         IN_RELATIVE_ATTENTION_BIAS_WEIGHT},
        {"1.layer_norm.weight", IN_CROSS_LAYER_NORM_WEIGHT},
        {"1.EncDecAttention.q.weight", IN_CROSS_Q_WEIGHT},
        {"1.EncDecAttention.k.weight", IN_CROSS_K_WEIGHT},
        {"1.EncDecAttention.v.weight", IN_CROSS_V_WEIGHT},
        {"1.EncDecAttention.o.weight", IN_CROSS_ATTN_OUT_WEIGHT},
        {"2.layer_norm.weight", IN_FINAL_LAYER_NORM_WEIGHT},
        {"2.DenseReluDense.wi.weight", IN_FFN_WI_1_WEIGHT},
        {"2.DenseReluDense.wo.weight", IN_FFN_WO_WEIGHT},
        {"2.DenseReluDense.gate_proj.weight", IN_FFN_WI_0_WEIGHT},
        {"2.ffn.wi.weight", IN_FFN_WI_1_WEIGHT},
        {"2.ffn.wo.weight", IN_FFN_WO_WEIGHT},
        {"2.ffn.gate_proj.weight", IN_FFN_WI_0_WEIGHT},
};

static std::unordered_map<std::string, int>
get_onerec_decoder_moe_weight_mapping() {
  std::unordered_map<std::string, int> mapping = kOneRecDecoderWeightMapping;

  mapping.emplace("layer.2.ffn.gate.weight", IN_BLOCK_SPARSE_MOE_GATE_WEIGHT);
  mapping.emplace("2.ffn.gate.weight", IN_BLOCK_SPARSE_MOE_GATE_WEIGHT);

  mapping.emplace("layer.2.ffn.shared_experts.w1.weight",
                  IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT);
  mapping.emplace("layer.2.ffn.shared_experts.w3.weight",
                  IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT);
  mapping.emplace("layer.2.ffn.shared_experts.w2.weight",
                  IN_MLP_DOWN_WEIGHT_SHARED_EXPERT);

  mapping.emplace("layer.2.ffn.shared_expert.gate.weight",
                  IN_SHARED_EXPERT_GATE_WEIGHT);
  mapping.emplace("layer.2.ffn.shared_expert.gate.bias",
                  IN_SHARED_EXPERT_GATE_BIAS);
  mapping.emplace("layer.2.ffn.shared_expert.gate.weight_scale",
                  IN_SHARED_EXPERT_GATE_SCALE);
  mapping.emplace("layer.2.ffn.shared_expert.gate.weight_offset",
                  IN_SHARED_EXPERT_GATE_OFFSET);

  // Expert weights are handled by
  // process_expert_weights()/merge_experts_weights to avoid ambiguous suffix
  // matching and keep deterministic loading.

  return mapping;
}

static const std::unordered_map<std::string, int>
    kOneRecDecoderMoeWeightMapping = get_onerec_decoder_moe_weight_mapping();

static const std::unordered_map<int32_t, int32_t> kOneRecWeightShard = {
    {IN_Q_WEIGHT, 0},
    {IN_K_WEIGHT, 0},
    {IN_V_WEIGHT, 0},
    {IN_SELF_ATTN_OUT_WEIGHT, 1},
    {IN_CROSS_Q_WEIGHT, 0},
    {IN_CROSS_K_WEIGHT, 0},
    {IN_CROSS_V_WEIGHT, 0},
    {IN_CROSS_ATTN_OUT_WEIGHT, 1},
    {IN_FFN_WI_0_WEIGHT, 0},
    {IN_FFN_WI_1_WEIGHT, 0},
    {IN_FFN_WO_WEIGHT, 1},
    // MoE
    {IN_BLOCK_SPARSE_MOE_GATE_WEIGHT, 0},
    {IN_MLP_GATEUP_WEIGHT_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_EXPERT, 1},
    // Shared experts
    {IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT, 0},
    {IN_MLP_GATEUP_OFFSET_SHARED_EXPERT, 0},
    {IN_MLP_GATEUP_SCALE_SHARED_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_SHARED_EXPERT, 1},
    {IN_MLP_DOWN_OFFSET_SHARED_EXPERT, 1},
    {IN_MLP_DOWN_SCALE_SHARED_EXPERT, 1},
    {IN_SHARED_EXPERT_GATE_WEIGHT, 0},
    {IN_SHARED_EXPERT_GATE_BIAS, 0},
    {IN_SHARED_EXPERT_GATE_SCALE, 0},
    {IN_SHARED_EXPERT_GATE_OFFSET, 0},
};

}  // namespace

NpuOneRecBlockLayerImpl::NpuOneRecBlockLayerImpl(const ModelContext& context,
                                                 bool is_decoder,
                                                 int32_t layer_id)
    : BaseLayer(context), is_decoder_(is_decoder), layer_id_(layer_id) {
  const auto& args = context.get_model_args();
  const auto& parallel_args = context.get_parallel_args();
  param_from_args(prefill_param_, args, parallel_args, /*is_prefill=*/true);
  param_from_args(decode_param_, args, parallel_args, /*is_prefill=*/false);

torch::Tensor NpuOneRecBlockLayerImpl::forward(torch::Tensor& hidden_states,
                                               torch::Tensor& attn_mask,
                                               KVCache& kv_cache,
                                               ModelInputParams& input_params,
                                               torch::Tensor* encoder_output,
                                               int32_t node_id,
                                               aclrtEvent* event,
                                               std::atomic<bool>* event_flag) {
  return forward(hidden_states,
                 attn_mask,
                 kv_cache,
                 input_params,
                 encoder_output,
                 node_id,
                 event,
                 event_flag,
                 /*expert_array=*/torch::Tensor());
}

torch::Tensor NpuOneRecBlockLayerImpl::forward(
    torch::Tensor& hidden_states,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    ModelInputParams& input_params,
    torch::Tensor* encoder_output,
    int32_t node_id,
    aclrtEvent* event,
    std::atomic<bool>* event_flag,
    const torch::Tensor& expert_array) {
  (void)attn_mask;
  (void)kv_cache;
  (void)input_params;
  (void)node_id;
  (void)event;
  (void)event_flag;
  (void)expert_array;

  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);

  for (size_t i = 0; i < kOneRecWeightCountPerLayer; ++i) {
    CHECK(node.inTensors.at(i) != nullptr)
        << model_name_ << " inTensor " << i << " is NULL";
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  const int input_tensor_idx = static_cast<int>(kOneRecWeightCountPerLayer);
  const int attention_mask_idx = input_tensor_idx + 1;
  const int token_offset_idx = attention_mask_idx + 1;
  const int layer_id_idx = token_offset_idx + 1;
  const int seq_len_idx = layer_id_idx + 1;

  node.variantPack.inTensors.at(input_tensor_idx) = internal_tensors_;
  node.variantPack.inTensors.at(attention_mask_idx) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);

  node.variantPack.inTensors.at(token_offset_idx) = placeholder_;
  node.variantPack.inTensors.at(token_offset_idx).hostData =
      placeholder_vec_.data();
  node.variantPack.inTensors.at(layer_id_idx) = placeholder_;
  node.variantPack.inTensors.at(layer_id_idx).hostData =
      placeholder_vec_.data();

  const auto* onerec_params = input_params.onerec_params();
  if (onerec_params != nullptr &&
      onerec_params->encoder_seq_lens_tensor.defined()) {
    node.variantPack.inTensors.at(seq_len_idx) =
        atb_speed::Utils::AtTensor2Tensor(
            onerec_params->encoder_seq_lens_tensor);
    node.variantPack.inTensors.at(seq_len_idx).hostData =
        const_cast<int32_t*>(onerec_params->encoder_seq_lens.data());
  } else {
    node.variantPack.inTensors.at(seq_len_idx) = placeholder_;
    node.variantPack.inTensors.at(seq_len_idx).hostData =
        placeholder_vec_.data();
  }

  node.variantPack.outTensors.at(0) = internal_tensors_;
}

void NpuOneRecBlockLayerImpl::build_decoder_moe_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    at::Tensor& attn_mask,
    KVCache& kv_cache,
    ModelInputParams& input_params,
    bool is_prefill,
    torch::Tensor* encoder_output,
    int layer_id,
    const torch::Tensor& expert_array) {
  (void)kv_cache;
  (void)is_prefill;
  (void)layer_id;

  for (size_t i = 0; i < kOneRecMoeWeightCountPerLayer; ++i) {
    CHECK(node.inTensors.at(i) != nullptr)
        << model_name_ << " inTensor " << i << " is NULL";
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  const int moe_tensor_start = static_cast<int>(kOneRecMoeWeightCountPerLayer);
  if (expert_array.defined()) {
    node.variantPack.inTensors.at(moe_tensor_start) =
        atb_speed::Utils::AtTensor2Tensor(expert_array);
  } else {
    node.variantPack.inTensors.at(moe_tensor_start) = placeholder_;
  }

  node.variantPack.inTensors.at(moe_tensor_start + 1) =
      expert_group_.defined() ? atb_speed::Utils::AtTensor2Tensor(expert_group_)
                              : placeholder_;
  node.variantPack.inTensors.at(moe_tensor_start + 2) =
      one_hot_.defined() ? atb_speed::Utils::AtTensor2Tensor(one_hot_)
                         : placeholder_;
  node.variantPack.inTensors.at(moe_tensor_start + 3) =
      zero_hot_.defined() ? atb_speed::Utils::AtTensor2Tensor(zero_hot_)
                          : placeholder_;

  int tensor_idx = setup_common_decoder_tensors(
      node, x, attn_mask, input_params, encoder_output, moe_tensor_start + 4);

  while (tensor_idx < static_cast<int>(node.variantPack.inTensors.size())) {
    node.variantPack.inTensors.at(tensor_idx) = placeholder_;
    node.variantPack.inTensors.at(tensor_idx).hostData =
        placeholder_vec_.data();
    ++tensor_idx;
  }
}

int NpuOneRecBlockLayerImpl::setup_common_decoder_tensors(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    at::Tensor& attn_mask,
    ModelInputParams& input_params,
    torch::Tensor* encoder_output,
    int start_tensor_idx) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);

  int idx = start_tensor_idx;
  node.variantPack.inTensors.at(idx++) = internal_tensors_;
  node.variantPack.inTensors.at(idx++) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);

  // Token offset and layer id placeholders.
  // ATB expects hostData to be valid for these scalar inputs. Keep them as
  // placeholders but always provide hostData to avoid undefined reads.
  node.variantPack.inTensors.at(idx) = placeholder_;
  node.variantPack.inTensors.at(idx++).hostData = placeholder_vec_.data();
  node.variantPack.inTensors.at(idx) = placeholder_;
  node.variantPack.inTensors.at(idx++).hostData = placeholder_vec_.data();

  CHECK(input_params.kv_seq_lens.defined()) << "kv_seq_lens is required.";
  node.variantPack.inTensors.at(idx) =
      atb_speed::Utils::AtTensor2Tensor(input_params.kv_seq_lens);
  node.variantPack.inTensors.at(idx).hostData =
      input_params.kv_seq_lens_vec.data();
  idx++;

  node.variantPack.inTensors.at(idx) = placeholder_;
  node.variantPack.inTensors.at(idx++).hostData = placeholder_vec_.data();
  node.variantPack.inTensors.at(idx) = placeholder_;
  node.variantPack.inTensors.at(idx++).hostData = placeholder_vec_.data();

  if (!FLAGS_enable_rec_prefill_only && input_params.block_tables.defined()) {
    node.variantPack.inTensors.at(idx) =
        atb_speed::Utils::AtTensor2Tensor(input_params.block_tables);
  } else {
    node.variantPack.inTensors.at(idx) = placeholder_;
    node.variantPack.inTensors.at(idx).hostData = placeholder_vec_.data();
  }
  idx++;

  if (!FLAGS_enable_rec_prefill_only &&
      input_params.new_cache_slots.defined()) {
    node.variantPack.inTensors.at(idx) =
        atb_speed::Utils::AtTensor2Tensor(input_params.new_cache_slots);
  } else {
    node.variantPack.inTensors.at(idx) = placeholder_;
    node.variantPack.inTensors.at(idx).hostData = placeholder_vec_.data();
  }
  idx++;

  if (encoder_output != nullptr) {
    encoder_output_contiguous_ = encoder_output->is_contiguous()
                                     ? *encoder_output
                                     : encoder_output->contiguous();
    node.variantPack.inTensors.at(idx) =
        atb_speed::Utils::AtTensor2Tensor(encoder_output_contiguous_);
  } else {
    node.variantPack.inTensors.at(idx) = placeholder_;
  }
  idx++;

  for (int i = 0; i < 3; i++) {
    node.variantPack.inTensors.at(idx) = placeholder_;
    node.variantPack.inTensors.at(idx++).hostData = placeholder_vec_.data();
  }

  const auto* onerec_params = input_params.onerec_params();
  if (onerec_params != nullptr &&
      onerec_params->encoder_seq_lens_tensor.defined()) {
    node.variantPack.inTensors.at(idx) = atb_speed::Utils::AtTensor2Tensor(
        onerec_params->encoder_seq_lens_tensor);
    node.variantPack.inTensors.at(idx++).hostData =
        const_cast<int32_t*>(onerec_params->encoder_seq_lens.data());
  } else {
    node.variantPack.inTensors.at(idx) = placeholder_;
    node.variantPack.inTensors.at(idx++).hostData = placeholder_vec_.data();
  }

  node.variantPack.outTensors.at(0) = internal_tensors_;
  return idx;
}

void NpuOneRecBlockLayerImpl::build_decoder_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    at::Tensor& attn_mask,
    KVCache& kv_cache,
    ModelInputParams& input_params,
    bool is_prefill,
    torch::Tensor* encoder_output,
    int layer_id) {
  (void)kv_cache;
  (void)is_prefill;
  (void)layer_id;

  for (size_t i = 0; i < kOneRecWeightCountPerLayer; ++i) {
    CHECK(node.inTensors.at(i) != nullptr)
        << model_name_ << " inTensor " << i << " is NULL";
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  int tensor_idx = setup_common_decoder_tensors(
      node,
      x,
      attn_mask,
      input_params,
      encoder_output,
      static_cast<int>(kOneRecWeightCountPerLayer));
  while (tensor_idx < static_cast<int>(node.variantPack.inTensors.size())) {
    node.variantPack.inTensors.at(tensor_idx) = placeholder_;
    node.variantPack.inTensors.at(tensor_idx).hostData =
        placeholder_vec_.data();
    ++tensor_idx;
  }
}

void NpuOneRecBlockLayerImpl::resize_experts_weights(
    int num_of_device_experts) {
  experts_weights_["gate_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["up_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["down_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
}

void NpuOneRecBlockLayerImpl::process_expert_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  (void)state_dict;
  std::lock_guard<std::mutex> lock(experts_mutex_);

  int expert_id = extract_expert_index(name);
  if (expert_id < 0) {
    return;
  }

  const int local_index = expert_id % num_experts_per_partition_;
  std::string weight_suffix = extract_endswith(name);

  std::string suffix;
  if (weight_suffix == "gate_proj.weight" || weight_suffix == "w1.weight") {
    suffix = "gate_proj.weight";
  } else if (weight_suffix == "up_proj.weight" ||
             weight_suffix == "w3.weight") {
    suffix = "up_proj.weight";
  } else if (weight_suffix == "down_proj.weight" ||
             weight_suffix == "w2.weight") {
    suffix = "down_proj.weight";
  } else {
    return;
  }

  auto it = experts_weights_.find(suffix);
  if (it == experts_weights_.end() || local_index < 0 ||
      local_index >= static_cast<int>(it->second.size())) {
    LOG(ERROR) << "Invalid OneRec MoE local expert index " << local_index
               << " for " << suffix << " at layer " << layer_id_ << ".";
    return;
  }
  it->second[local_index] = tensor;
}

void NpuOneRecBlockLayerImpl::process_shared_expert_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  (void)state_dict;
  torch::Tensor tmp_tensor = tensor.to(device_);

  std::string canonical_name;
  if (absl::StrContains(name, "gate_proj") || absl::StrContains(name, "w1")) {
    canonical_name = "gate_proj.weight";
  } else if (absl::StrContains(name, "up_proj") ||
             absl::StrContains(name, "w3")) {
    canonical_name = "up_proj.weight";
  } else if (absl::StrContains(name, "down_proj") ||
             absl::StrContains(name, "w2")) {
    canonical_name = "down_proj.weight";
  } else {
    return;
  }

  if (shared_expert_weights_map_.count(canonical_name) > 0) {
    LOG(WARNING) << "Duplicate OneRec shared expert tensor for "
                 << canonical_name << " at layer " << layer_id_
                 << ", overriding previous value.";
  }
  shared_expert_weights_map_[canonical_name] = tmp_tensor;
}

int NpuOneRecBlockLayerImpl::extract_expert_index(const std::string& name) {
  size_t experts_pos = name.find(".experts.");
  if (experts_pos == std::string::npos) {
    return -1;
  }

  size_t start_pos = experts_pos + 9;
  size_t end_pos = name.find(".", start_pos);
  if (end_pos == std::string::npos) {
    return -1;
  }

  try {
    return std::stoi(name.substr(start_pos, end_pos - start_pos));
  } catch (const std::exception&) {
    return -1;
  }
}

std::string NpuOneRecBlockLayerImpl::extract_endswith(
    const std::string& input) {
  size_t experts_pos = input.find(".experts.");
  if (experts_pos == std::string::npos) {
    return "";
  }
  size_t start_pos = experts_pos + 9;
  size_t next_dot = input.find(".", start_pos);
  if (next_dot == std::string::npos) {
    return "";
  }
  return input.substr(next_dot + 1);
}

torch::Tensor NpuOneRecBlockLayerImpl::merge_experts_weights(
    std::vector<torch::Tensor>& experts,
    bool transpose) {
  std::vector<torch::Tensor> valid;
  valid.reserve(experts.size());
  for (auto& t : experts) {
    if (t.defined()) {
      valid.push_back(t.to(device_));
    }
  }
  if (valid.empty()) {
    LOG(ERROR) << "No expert weights to merge at layer " << layer_id_ << ".";
    return torch::Tensor();
  }
  torch::Tensor merged_tensor = torch::stack(valid, 0);
  if (transpose) {
    merged_tensor = merged_tensor.transpose(1, 2);
  }
  return merged_tensor.contiguous();
}

torch::Tensor NpuOneRecBlockLayerImpl::merge_experts_weights(
    std::vector<torch::Tensor>& experts_gate,
    std::vector<torch::Tensor>& experts_up,
    bool transpose) {
  if (experts_gate.size() != experts_up.size()) {
    LOG(ERROR) << "OneRec MoE gate/up expert size mismatch: gate="
               << experts_gate.size() << ", up=" << experts_up.size()
               << ", layer " << layer_id_;
    return torch::Tensor();
  }
  for (size_t i = 0; i < experts_gate.size(); ++i) {
    const bool gate_defined = experts_gate[i].defined();
    const bool up_defined = experts_up[i].defined();
    if (gate_defined != up_defined) {
      LOG(ERROR) << "OneRec MoE gate/up tensor mismatch at local expert " << i
                 << ": gate=" << gate_defined << ", up=" << up_defined
                 << ", layer " << layer_id_;
      return torch::Tensor();
    }
    if (gate_defined) {
      experts_gate[i] = torch::cat({experts_gate[i], experts_up[i]}, 0);
    }
  }
  return merge_experts_weights(experts_gate, transpose);
}

void NpuOneRecBlockLayerImpl::merge_experts_weights() {
  if (experts_weights_.count("gate_proj.weight") == 0 ||
      experts_weights_.count("up_proj.weight") == 0 ||
      experts_weights_.count("down_proj.weight") == 0) {
    return;
  }

  auto merged_gate_up =
      merge_experts_weights(experts_weights_["gate_proj.weight"],
                            experts_weights_["up_proj.weight"],
                            /*transpose=*/false);
  CHECK(merged_gate_up.defined()) << "OneRec MoE gate/up experts merge failed.";
  at_weight_tensors_[IN_MOE_EXPERT_W1_WEIGHT] =
      at_npu::native::npu_format_cast(merged_gate_up, /*format=*/2)
          .contiguous();

  auto merged_down = merge_experts_weights(experts_weights_["down_proj.weight"],
                                           /*transpose=*/false);
  CHECK(merged_down.defined()) << "OneRec MoE down experts merge failed.";
  at_weight_tensors_[IN_MOE_EXPERT_W2_WEIGHT] =
      at_npu::native::npu_format_cast(merged_down, /*format=*/2).contiguous();
}

void NpuOneRecBlockLayerImpl::merge_shared_experts_weights() {
  shared_expert_gate_weights_.clear();
  shared_expert_up_weights_.clear();
  shared_expert_down_weights_.clear();

  if (const auto it = shared_expert_weights_map_.find("gate_proj.weight");
      it != shared_expert_weights_map_.end()) {
    shared_expert_gate_weights_.push_back(it->second);
  }
  if (const auto it = shared_expert_weights_map_.find("up_proj.weight");
      it != shared_expert_weights_map_.end()) {
    shared_expert_up_weights_.push_back(it->second);
  }
  if (const auto it = shared_expert_weights_map_.find("down_proj.weight");
      it != shared_expert_weights_map_.end()) {
    shared_expert_down_weights_.push_back(it->second);
  }

  if (shared_expert_gate_weights_.empty() &&
      shared_expert_up_weights_.empty() &&
      shared_expert_down_weights_.empty()) {
    return;
  }

  if (!shared_expert_gate_weights_.empty() &&
      !shared_expert_up_weights_.empty()) {
    auto merged_gate_up = merge_experts_weights(shared_expert_gate_weights_,
                                                shared_expert_up_weights_,
                                                /*transpose=*/false);
    CHECK(merged_gate_up.defined())
        << "OneRec shared gate/up experts merge failed at layer " << layer_id_;
    at_weight_tensors_[IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT] = merged_gate_up;
  } else if (!shared_expert_gate_weights_.empty()) {
    at_weight_tensors_[IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT] =
        merge_experts_weights(shared_expert_gate_weights_, false);
  }

  if (!shared_expert_down_weights_.empty()) {
    at_weight_tensors_[IN_MLP_DOWN_WEIGHT_SHARED_EXPERT] =
        merge_experts_weights(shared_expert_down_weights_, false);
  }

  shared_expert_gate_weights_.clear();
  shared_expert_up_weights_.clear();
  shared_expert_down_weights_.clear();
  shared_expert_weights_map_.clear();
}

void NpuOneRecBlockLayerImpl::load_state_dict(const StateDict& state_dict) {
  (void)state_dict;
}

void NpuOneRecBlockLayerImpl::verify_loaded_weights(
    const std::string& prefix) const {
  (void)prefix;
}

void NpuOneRecBlockLayerImpl::merge_loaded_weights() {}

}  // namespace layer
}  // namespace xllm
