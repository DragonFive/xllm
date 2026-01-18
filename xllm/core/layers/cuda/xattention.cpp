
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

#include <chrono>
#include <cstdio>

#include "core/common/global_flags.h"
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
    KVCache& kv_cache,
    std::optional<torch::Tensor> output) {
  if (!output.has_value()) {
    output = torch::empty_like(query);
  }
  auto output_tensor = output.value();
  auto output_lse = std::nullopt;
  if (attn_metadata.max_seq_len == 0) {
    output_tensor = output_tensor.view({-1, num_heads_ * head_size_});
    return std::make_tuple(output_tensor, output_lse);
  }

  query = query.view({-1, num_heads_, head_size_});
  key = key.view({-1, num_kv_heads_, head_size_});
  value = value.view({-1, num_kv_heads_, head_size_});
  output_tensor = output_tensor.view({-1, num_heads_, head_size_});

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
        output_tensor.scalar_type(),
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
    xllm::kernel::AttentionParams attention_params(attn_metadata);
    attention_params.query = query;
    attention_params.output = output_tensor;
    attention_params.output_lse = output_lse;
    attention_params.window_size_left = sliding_window_;
    attention_params.scale = scale_;
    // for flashinfer
    attention_params.float_workspace_buffer =
        ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
            .get_float_workspace_buffer();
    attention_params.int_workspace_buffer =
        ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
            .get_int_workspace_buffer();
    attention_params.page_locked_int_workspace_buffer =
        ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
            .get_page_locked_int_workspace_buffer();
    // attention_params.kv_cu_seq_lens = attn_metadata.kv_cu_seq_lens;
    // attention_params.q_cu_seq_lens = attn_metadata.q_cu_seq_lens;

    attention_params.key = key;
    attention_params.value = value;
    // attention_params.uri = attn_metadata.plan_info->uri;
    // attention_params.plan_info = attn_metadata.plan_info->plan_info;
    xllm::kernel::batch_prefill(attention_params);
  } else {
    // uint32_t batch_size = attn_metadata.paged_kv_last_page_len.numel();
    uint32_t batch_size = attn_metadata.kv_cu_seq_lens.size(0) - 1;
    uint32_t total_beam = query.size(0);
    uint32_t beam_size = total_beam / batch_size;

    // [batch_size * beam_size * max_decode_step, num_kv_heads_, head_size_]
    key = key.view({batch_size, beam_size, num_kv_heads_, head_size_})
              .contiguous();
    value = value.view({batch_size, beam_size, num_kv_heads_, head_size_})
                .contiguous();
    xllm::kernel::cuda::decoder_reshape_and_cache(
        key,
        value,
        attn_metadata.unshared_k_cache,
        attn_metadata.unshared_v_cache,
        attn_metadata.naive_block_table,
        attn_metadata.step);

    if (FLAGS_enable_xattention_two_stage_decode) {
      if (attn_metadata.enable_cuda_graph) {
        CHECK(attn_metadata.two_stage_decode_cache.has_value())
            << "two_stage_decode_cache must be pre-initialized before CUDA "
               "graph capture/replay. enable_cuda_graph="
            << attn_metadata.enable_cuda_graph << ", batch_size=" << batch_size
            << ", beam_size=" << beam_size;

        // Additional validation for CUDA graph mode
        const auto& cache = attn_metadata.two_stage_decode_cache.value();
        CHECK_EQ(cache.cached_batch_size, static_cast<int32_t>(batch_size))
            << "Cached batch_size mismatch: cached=" << cache.cached_batch_size
            << ", actual=" << batch_size;
        CHECK_EQ(cache.cached_beam_size, static_cast<int32_t>(beam_size))
            << "Cached beam_size mismatch: cached=" << cache.cached_beam_size
            << ", actual=" << beam_size;
      }
      // 判断是否在第一层
      bool is_layer_0 = (attn_metadata.plan_info->layer_id == 0);

      // 初始化或复用缓存（只在第一层初始化）
      if (is_layer_0 && !attn_metadata.enable_cuda_graph) {
        // 检查是否需要重新初始化（参数变化）
        bool need_init =
            !attn_metadata.two_stage_decode_cache.has_value() ||
            attn_metadata.two_stage_decode_cache->cached_batch_size !=
                static_cast<int32_t>(batch_size) ||
            attn_metadata.two_stage_decode_cache->cached_beam_size !=
                static_cast<int32_t>(beam_size) ||
            attn_metadata.two_stage_decode_cache->cached_num_heads !=
                num_heads_ ||
            attn_metadata.two_stage_decode_cache->cached_head_size !=
                head_size_;

        if (need_init) {
          TwoStageDecodeCache cache;

          auto fp32_options = torch::TensorOptions()
                                  .dtype(torch::kFloat32)
                                  .device(query.device());

          // 初始化输出 tensors (3D shape for batch_prefill)
          cache.shared_lse = torch::zeros(
              {batch_size * beam_size, num_heads_, 1}, fp32_options);
          cache.shared_o =
              torch::zeros({batch_size * beam_size, num_heads_, head_size_},
                           query.options());
          cache.unshared_lse =
              torch::zeros({total_beam, num_heads_, 1}, fp32_options);
          cache.unshared_o = torch::zeros({total_beam, num_heads_, head_size_},
                                          query.options());

          // 初始化固定 tensors
          cache.q_cu_seq_lens_shared =
              torch::arange(0,
                            (batch_size + 1) * beam_size,
                            beam_size,
                            torch::TensorOptions()
                                .dtype(torch::kInt32)
                                .device(query.device()));

          cache.paged_kv_indptr_expanded =
              torch::arange(batch_size * beam_size + 1,
                            attn_metadata.paged_kv_indptr.options());

          cache.paged_kv_last_page_len_expanded =
              torch::full({batch_size * beam_size},
                          0,
                          attn_metadata.paged_kv_last_page_len.options());
          cache.paged_kv_last_page_len_expanded.fill_(attn_metadata.step + 1);

          // paged_kv_indices: 每个 (batch, beam) 对应一个 block_id
          cache.paged_kv_indices_expanded = torch::arange(
              batch_size * beam_size, attn_metadata.paged_kv_indices.options());

          // 缓存参数
          cache.cached_batch_size = static_cast<int32_t>(batch_size);
          cache.cached_beam_size = static_cast<int32_t>(beam_size);
          cache.cached_num_heads = num_heads_;
          cache.cached_head_size = head_size_;
          // Use const_cast to modify mutable cache in const attn_metadata
          const_cast<AttentionMetadata&>(attn_metadata).two_stage_decode_cache =
              cache;
        }
      }

      // 获取缓存的 tensor
      auto& cache = attn_metadata.two_stage_decode_cache.value();

      const int64_t unshared_offset =
          static_cast<int64_t>(FLAGS_max_seqs_per_batch) *
          static_cast<int64_t>(FLAGS_max_token_per_req);
      torch::Tensor shared_k_cache =
          attn_metadata.full_k_cache.slice(0, 0, unshared_offset);
      torch::Tensor shared_v_cache =
          attn_metadata.full_v_cache.slice(0, 0, unshared_offset);

      // Step 2: Prepare query (不需要 clone，直接使用 view)
      // shared_q 和 query 的数据相同，只是 shape 不同
      auto shared_q =
          query.view({batch_size * beam_size, num_heads_, head_size_});

      // Step 3: Create temporary AttentionMetadata with q_cu_seq_lens_shared
      AttentionMetadata shared_attn_meta = attn_metadata;
      shared_attn_meta.q_cu_seq_lens = cache.q_cu_seq_lens_shared;

      if (attn_metadata.enable_cuda_graph) {
        CHECK(attn_metadata.plan_info)
            << "plan_info is null when enable_cuda_graph=true";
        CHECK(attn_metadata.plan_info->plan_info.defined())
            << "shared stage plan_info->plan_info should not be null when "
               "enable_cuda_graph is true. plan_info->layer_id="
            << attn_metadata.plan_info->layer_id
            << ", plan_info->uri=" << attn_metadata.plan_info->uri;
      } else {
        flashinfer::update_plan_info(
            attn_metadata.plan_info,
            xllm::kernel::cuda::determine_attention_backend(
                /*pos_encoding_mode=*/0,
                /*use_fp16_qk_reduction=*/false,
                /*use_custom_mask=*/false),
            shared_attn_meta,
            query.scalar_type(),
            shared_k_cache.scalar_type(),
            cache.shared_o.scalar_type(),
            head_size_,
            head_size_,
            num_heads_,
            num_kv_heads_,
            /*block_size*/ 1,
            /*window_size_left*/ sliding_window_,
            /*enable_cuda_graph*/ false,
            /*causal*/ true,
            /*use_tensor_core*/ true);
      }

      xllm::kernel::AttentionParams shared_attention_params(shared_attn_meta);
      shared_attention_params.return_lse = true;
      shared_attention_params.query = shared_q;
      // Use cached tensors directly - they are already in 3D shape
      shared_attention_params.output = cache.shared_o;
      shared_attention_params.output_lse = cache.shared_lse;
      shared_attention_params.window_size_left = sliding_window_;
      shared_attention_params.scale = scale_;
      // CRITICAL FIX: Use dedicated graph workspace if available (CUDA graph
      // mode), otherwise fall back to global workspace (eager mode). This
      // prevents prefill operations from overwriting workspace data that graph
      // replay kernels depend on.
      if (attn_metadata.graph_float_workspace_buffer.has_value()) {
        shared_attention_params.float_workspace_buffer =
            attn_metadata.graph_float_workspace_buffer.value();
        shared_attention_params.int_workspace_buffer =
            attn_metadata.graph_int_workspace_buffer.value();
        shared_attention_params.page_locked_int_workspace_buffer =
            attn_metadata.graph_page_locked_int_workspace_buffer.value();
      } else {
        shared_attention_params.float_workspace_buffer =
            ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
                .get_float_workspace_buffer();
        shared_attention_params.int_workspace_buffer =
            ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
                .get_int_workspace_buffer();
        shared_attention_params.page_locked_int_workspace_buffer =
            ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
                .get_page_locked_int_workspace_buffer();
      }
      shared_attention_params.key = shared_k_cache;
      shared_attention_params.value = shared_v_cache;

      xllm::kernel::batch_prefill(shared_attention_params);
      // batch_prefill writes directly to cache.shared_o and cache.shared_lse
      // through the view, so no need to reassign. The tensors are already in
      // cache, and batch_prefill has modified them in-place.

      // Step 5: Update plan info for unshared stage (decode mode)
      // Use independent unshared_plan_info to avoid overwriting shared stage
      // plan_info

      // Runtime assertion: Validate unshared cache tensors
      CHECK(attn_metadata.unshared_k_cache.defined())
          << "unshared_k_cache must be defined for two-stage decode";
      CHECK(attn_metadata.unshared_v_cache.defined())
          << "unshared_v_cache must be defined for two-stage decode";
      CHECK_GE(attn_metadata.unshared_k_cache.size(2), 1)
          << "unshared_k_cache.size(2) must be >= 1, got: "
          << attn_metadata.unshared_k_cache.size(2)
          << ", tensor sizes: " << attn_metadata.unshared_k_cache.sizes();
      CHECK(attn_metadata.unshared_plan_info)
          << "unshared_plan_info must not be null for two-stage decode";

      int64_t actual_head_dim_qk = query.size(-1);
      int64_t actual_head_dim_vo = attn_metadata.unshared_v_cache.size(-1);
      AttentionMetadata unshared_attn_meta = attn_metadata;
      unshared_attn_meta.plan_info = attn_metadata.unshared_plan_info;
      unshared_attn_meta.paged_kv_indices = cache.paged_kv_indices_expanded;
      unshared_attn_meta.paged_kv_indptr = cache.paged_kv_indptr_expanded;
      unshared_attn_meta.paged_kv_last_page_len =
          cache.paged_kv_last_page_len_expanded;
      unshared_attn_meta.use_tensor_core = false;
      if (attn_metadata.enable_cuda_graph) {
        CHECK(attn_metadata.unshared_plan_info)
            << "unshared_plan_info is null when enable_cuda_graph=true";
        CHECK(attn_metadata.unshared_plan_info->plan_info.defined())
            << "unshared stage plan_info->plan_info should not be null when "
               "enable_cuda_graph is true. unshared_plan_info->layer_id="
            << attn_metadata.unshared_plan_info->layer_id
            << ", unshared_plan_info->uri="
            << attn_metadata.unshared_plan_info->uri
            << ", unshared_k_cache.size(2)="
            << attn_metadata.unshared_k_cache.size(2);
      } else {
        flashinfer::update_plan_info(
            attn_metadata.unshared_plan_info,
            /*backend*/ "fa3",
            unshared_attn_meta,
            query.scalar_type(),
            attn_metadata.unshared_k_cache.scalar_type(),
            cache.unshared_o.scalar_type(),
            actual_head_dim_qk,
            actual_head_dim_vo,
            num_heads_,
            num_kv_heads_,
            /*block_size*/ attn_metadata.unshared_k_cache.size(2),
            /*window_size_left*/ sliding_window_,
            /*enable_cuda_graph*/ false,
            /*causal*/ false,
            /*use_tensor_core*/ false);
      }

      // Step 6: Compute unshared attention using batch_decode
      query = query.view({-1, num_heads_, head_size_});

      int64_t max_decode_step = attn_metadata.unshared_k_cache.size(2);
      torch::Tensor unshared_k = attn_metadata.unshared_k_cache.view(
          {-1, max_decode_step, num_kv_heads_, head_size_});
      torch::Tensor unshared_v = attn_metadata.unshared_v_cache.view(
          {-1, max_decode_step, num_kv_heads_, head_size_});

      xllm::kernel::AttentionParams unshared_attention_params(
          unshared_attn_meta);
      unshared_attention_params.return_lse = true;
      unshared_attention_params.query = query;
      // Use cached tensors directly - batch_decode will write directly to them
      unshared_attention_params.output = cache.unshared_o;
      unshared_attention_params.output_lse = cache.unshared_lse;
      unshared_attention_params.window_size_left = sliding_window_;
      unshared_attention_params.scale = scale_;
      // CRITICAL FIX: Use dedicated graph workspace if available (CUDA graph
      // mode), otherwise fall back to global workspace (eager mode). This
      // prevents prefill operations from overwriting workspace data that graph
      // replay kernels depend on.
      if (attn_metadata.graph_float_workspace_buffer.has_value()) {
        unshared_attention_params.float_workspace_buffer =
            attn_metadata.graph_float_workspace_buffer.value();
        unshared_attention_params.int_workspace_buffer =
            attn_metadata.graph_int_workspace_buffer.value();
        unshared_attention_params.page_locked_int_workspace_buffer =
            attn_metadata.graph_page_locked_int_workspace_buffer.value();
      } else {
        unshared_attention_params.float_workspace_buffer =
            ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
                .get_float_workspace_buffer();
        unshared_attention_params.int_workspace_buffer =
            ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
                .get_int_workspace_buffer();
        unshared_attention_params.page_locked_int_workspace_buffer =
            ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
                .get_page_locked_int_workspace_buffer();
      }
      unshared_attention_params.k_cache = unshared_k;
      unshared_attention_params.v_cache = unshared_v;

      xllm::kernel::batch_decode(unshared_attention_params);
      // unshared_attention_params.output already points to unshared_o,
      // so batch_decode has already written to the cached tensor.

      // Step 7: Combine results
      // shared_o, shared_lse, unshared_o, unshared_lse are already in 3D shape
      xllm::kernel::cuda::lse_combine(output_tensor,
                                      cache.shared_o,
                                      cache.shared_lse,
                                      cache.unshared_o,
                                      cache.unshared_lse);

    } else {
      torch::Tensor full_k_cache = attn_metadata.full_k_cache.unsqueeze(1);
      torch::Tensor full_v_cache = attn_metadata.full_v_cache.unsqueeze(1);

      // maybe we need to update shared attn state before execute attention,
      // currently we update flashinfer step_wise_attn_state_ at layer 0.
      if (attn_metadata.enable_cuda_graph) {
        CHECK(attn_metadata.plan_info->plan_info.defined())
            << "plan_info plan_info should not be null when enable_cuda_graph "
               "is "
               "true";
        VLOG(50) << "no need to update plan_info for CUDA graph";
      } else {
        std::string backend = "fa3";
        flashinfer::update_plan_info(attn_metadata.plan_info,
                                     backend,
                                     attn_metadata,
                                     query.scalar_type(),
                                     key.scalar_type(),
                                     output_tensor.scalar_type(),
                                     head_size_,
                                     head_size_,
                                     num_heads_,
                                     num_kv_heads_,
                                     /*block_size*/ full_k_cache.size(1),
                                     /*window_size_left*/ sliding_window_,
                                     /*enable_cuda_graph*/ false,
                                     /*causal*/ false,
                                     /*use_tensor_core*/ false);
      }

      xllm::kernel::AttentionParams attention_params(attn_metadata);
      auto unshared_lse = std::nullopt;

      attention_params.return_lse = false;
      attention_params.output_lse = unshared_lse;

      attention_params.window_size_left = sliding_window_;
      attention_params.scale = scale_;
      // attention_params.compute_dtype = attn_metadata.compute_dtype;
      // for flashinfer
      attention_params.float_workspace_buffer =
          ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
              .get_float_workspace_buffer();
      attention_params.int_workspace_buffer =
          ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
              .get_int_workspace_buffer();
      attention_params.page_locked_int_workspace_buffer =
          ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
              .get_page_locked_int_workspace_buffer();
      // TODO: support chunked prefill
      CHECK(!attn_metadata.is_chunked_prefill)
          << "chunked prefill is not supported";

      attention_params.query = query;
      attention_params.output = output_tensor;

      attention_params.k_cache = full_k_cache;
      attention_params.v_cache = full_v_cache;

      // attention_params.paged_kv_indices = attn_metadata.paged_kv_indices;
      // attention_params.paged_kv_indptr = attn_metadata.paged_kv_indptr;
      // attention_params.paged_kv_last_page_len =
      //     attn_metadata.paged_kv_last_page_len;
      // attention_params.uri = attn_metadata.plan_info->uri;
      // attention_params.plan_info = attn_metadata.plan_info->plan_info;
      // attention_params.use_tensor_core = false;
      const_cast<AttentionMetadata&>(attention_params.attn_metadata)
          .use_tensor_core = false;
      xllm::kernel::batch_decode(attention_params);
    }
  }
  output_tensor = output_tensor.view({-1, num_heads_ * head_size_});
  return {output_tensor, output_lse};
}

}  // namespace layer
}  // namespace xllm
