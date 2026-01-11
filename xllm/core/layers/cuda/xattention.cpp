/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "xattention.h"

#include "flashinfer_planinfo.h"
#include "flashinfer_workspace.h"
#include "kernels/cuda/cuda_ops_api.h"
#include "kernels/ops_api.h"

DECLARE_bool(enable_chunked_prefill);

namespace xllm {
namespace layer {

std::tuple<torch::Tensor, std::optional<torch::Tensor>> XAttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  // LOG(INFO) << "inner XAttentionImpl::forward";
  auto output = torch::empty_like(query);
  auto output_lse = std::nullopt;
  if (attn_metadata.max_seq_len == 0) {
    output = output.view({-1, num_heads_ * head_size_});
    return std::make_tuple(output, output_lse);
  }

  query = query.view({-1, num_heads_, head_size_});
  key = key.view({-1, num_kv_heads_, head_size_});
  value = value.view({-1, num_kv_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v_cache = kv_cache.get_v_cache();

  if (attn_metadata.is_prefill) {
    CHECK(!attn_metadata.is_chunked_prefill)
        << "chunked prefill is not supported";

    // maybe we need to update shared attn state before execute attention,
    // currently we update flashinfer step_wise_attn_state_ at layer 0.
    bool causal = attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
    flashinfer::update_plan_info(
        attn_metadata.plan_info,
        causal ? xllm::kernel::cuda::determine_attention_backend(
                     /*pos_encoding_mode=*/0,
                     /*use_fp16_qk_reduction=*/false,
                     /*use_custom_mask=*/false)
               : "fa2",
        attn_metadata,
        query.scalar_type(),
        key.scalar_type(),
        output.scalar_type(),
        head_size_,
        head_size_,
        num_heads_,
        num_kv_heads_,
        /*block_size*/ k_cache.size(1),
        /*window_size_left*/ sliding_window_,
        /*enable_cuda_graph*/ false,
        /*causal*/ causal,
        /*use_tensor_core*/ true);

    xllm::kernel::cuda::prefill_reshape_and_cache(
        key, value, attn_metadata.full_k_cache, attn_metadata.full_v_cache);
    // LOG(INFO) << "after prefill_reshape_and_cache";
    xllm::kernel::AttentionParams attention_params;
    attention_params.query = query;
    attention_params.output = output;
    attention_params.output_lse = output_lse;
    attention_params.window_size_left = sliding_window_;
    attention_params.scale = scale_;
    attention_params.compute_dtype = attn_metadata.compute_dtype;
    // for flashinfer
    attention_params.float_workspace_buffer =
        FlashinferWorkspace::get_instance().get_float_workspace_buffer();
    attention_params.int_workspace_buffer =
        FlashinferWorkspace::get_instance().get_int_workspace_buffer();
    attention_params.page_locked_int_workspace_buffer =
        FlashinferWorkspace::get_instance()
            .get_page_locked_int_workspace_buffer();
    attention_params.kv_cu_seq_lens = attn_metadata.kv_cu_seq_lens;
    attention_params.q_cu_seq_lens = attn_metadata.q_cu_seq_lens;
    // attention_params.plan_info = attn_metadata.prefill_plan_info;
    attention_params.key = key;
    attention_params.value = value;
    attention_params.uri = attn_metadata.plan_info->uri;
    attention_params.plan_info = attn_metadata.plan_info->plan_info;
    xllm::kernel::batch_prefill(attention_params);
    // LOG(INFO) << "after batch_prefill";
  } else {
    uint32_t batch_size = attn_metadata.kv_cu_seq_lens.size(0) - 1;
    uint32_t total_beam = query.size(0);
    uint32_t beam_size = total_beam / batch_size;

    torch::Tensor full_k_cache = attn_metadata.full_k_cache;
    torch::Tensor full_v_cache = attn_metadata.full_v_cache;

    // [batch_size * beam_size * max_decode_step, num_kv_heads_, head_size_]
    key = key.view({batch_size, beam_size, num_kv_heads_, head_size_})
              .contiguous();
    value = value.view({batch_size, beam_size, num_kv_heads_, head_size_})
                .contiguous();
    int32_t full_kv_len = full_k_cache.size(0);
    int32_t unshared_offset = batch_size * FLAGS_max_token_per_req;
    // int32_t max_decode_step = k_cache.size(2);
    int32_t max_decode_step = FLAGS_max_decode_rounds - 1;
    // LOG(INFO) << "full_kv_len: " << full_kv_len;
    // LOG(INFO) << "unshared_offset: " << unshared_offset;
    auto unshared_k_cache = full_k_cache.slice(0, unshared_offset, full_kv_len);
    auto unshared_v_cache = full_v_cache.slice(0, unshared_offset, full_kv_len);
    // LOG(INFO) << "unshared_k_cache.shape: " << unshared_k_cache.sizes();
    // LOG(INFO) << "unshared_v_cache.shape: " << unshared_v_cache.sizes();
    unshared_k_cache = unshared_k_cache.view(
        {batch_size, beam_size, max_decode_step, num_kv_heads_, head_size_});
    unshared_v_cache = unshared_v_cache.view(
        {batch_size, beam_size, max_decode_step, num_kv_heads_, head_size_});

    xllm::kernel::cuda::decoder_reshape_and_cache(
        key,
        value,
        unshared_k_cache,
        unshared_v_cache,
        attn_metadata.naive_block_table,
        attn_metadata.step);
    // LOG(INFO) << "after decoder_reshape_and_cache";
    full_k_cache = full_k_cache.unsqueeze(1);
    full_v_cache = full_v_cache.unsqueeze(1);
    // LOG(INFO) << "full_k_cache.shape: " << full_k_cache.sizes();
    // LOG(INFO) << "full_v_cache.shape: " << full_v_cache.sizes();
    // maybe we need to update shared attn state before execute attention,
    // currently we update flashinfer step_wise_attn_state_ at layer 0.
    // LOG(INFO) << "attn_metadata.is_prefill: " << attn_metadata.is_prefill;
    bool causal = true;
    flashinfer::update_plan_info(
        attn_metadata.plan_info,
        causal ? xllm::kernel::cuda::determine_attention_backend(
                     /*pos_encoding_mode=*/0,
                     /*use_fp16_qk_reduction=*/false,
                     /*use_custom_mask=*/false)
               : "fa2",
        attn_metadata,
        query.scalar_type(),
        key.scalar_type(),
        output.scalar_type(),
        head_size_,
        head_size_,
        num_heads_,
        num_kv_heads_,
        /*block_size*/ full_k_cache.size(1),
        /*window_size_left*/ sliding_window_,
        /*enable_cuda_graph*/ false,
        /*causal*/ false,
        /*use_tensor_core*/ false);

    xllm::kernel::AttentionParams attention_params;
    auto unshared_lse = std::nullopt;

    attention_params.return_lse = false;
    attention_params.output_lse = unshared_lse;

    attention_params.window_size_left = sliding_window_;
    attention_params.scale = scale_;
    attention_params.compute_dtype = attn_metadata.compute_dtype;
    // for flashinfer
    attention_params.float_workspace_buffer =
        FlashinferWorkspace::get_instance().get_float_workspace_buffer();
    attention_params.int_workspace_buffer =
        FlashinferWorkspace::get_instance().get_int_workspace_buffer();
    attention_params.page_locked_int_workspace_buffer =
        FlashinferWorkspace::get_instance()
            .get_page_locked_int_workspace_buffer();
    // LOG(INFO) << "after attention_params";
    // TODO: support chunked prefill
    CHECK(!attn_metadata.is_chunked_prefill)
        << "chunked prefill is not supported";

    attention_params.query = query;
    attention_params.output = output;

    attention_params.k_cache = full_k_cache;
    attention_params.v_cache = full_v_cache;

    attention_params.paged_kv_indices = attn_metadata.decode_paged_kv_indices;
    attention_params.paged_kv_indptr = attn_metadata.decode_paged_kv_indptr;
    attention_params.paged_kv_last_page_len =
        attn_metadata.decode_paged_kv_last_page_len;
    // LOG(INFO) << "attn_metadata.decode_paged_kv_indices: "
    //           << attn_metadata.decode_paged_kv_indices;
    // LOG(INFO) << "attn_metadata.decode_paged_kv_indptr: "
    //           << attn_metadata.decode_paged_kv_indptr;
    // LOG(INFO) << "attn_metadata.decode_paged_kv_last_page_len: "
    //           << attn_metadata.decode_paged_kv_last_page_len;
    // attention_params.plan_info = attn_metadata.decode_plan_info;
    attention_params.uri = attn_metadata.plan_info->uri;
    attention_params.plan_info = attn_metadata.plan_info->plan_info;
    // LOG(INFO) << "attention_params.uri: " << attention_params.uri;
    attention_params.use_tensor_core = false;
    xllm::kernel::batch_decode(attention_params);
    // LOG(INFO) << "after batch_decode";
  }
  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

}  // namespace layer
}  // namespace xllm