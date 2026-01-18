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

#pragma once

#include <ATen/cuda/CUDAGraph.h>
#include <absl/container/flat_hash_map.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

#include "core/common/macros.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_input_params.h"
#include "core/layers/common/attention_metadata.h"
#include "executor_impl.h"
#include "executor_impl_factory.h"
#include "options.h"

namespace xllm {

// Helper class to hold persistent parameters for CUDA graph execution
// Multiple CudaGraph instances can share the same CudaGraphPersistentParam
// object
class CudaGraphPersistentParam {
 public:
  CudaGraphPersistentParam(const ModelArgs& args,
                           const torch::Device& device,
                           const runtime::Options& options);

  ~CudaGraphPersistentParam() = default;

  // Update persistent tensors with new input data
  // If return_capture_params is true, returns a ModelInputParams with
  // persistent buffer references. padded_num_tokens must be > 0 when
  // return_capture_params is true, used for build new ModelInputParams for
  // capture. If return_capture_params is false, only updates persistent buffers
  // and returns std::nullopt.
  std::optional<ModelInputParams> update(const torch::Tensor& tokens,
                                         const torch::Tensor& k_cache,
                                         const torch::Tensor& v_cache,
                                         const torch::Tensor& positions,
                                         const ModelInputParams& params,
                                         uint32_t padded_num_tokens = 0,
                                         bool return_capture_params = false);

  // Getter methods for persistent tensors
  torch::Tensor persistent_tokens(uint32_t actual_tokens) const {
    if (actual_tokens > 0) {
      return persistent_tokens_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return persistent_tokens_;
  }
  torch::Tensor persistent_positions(uint32_t actual_tokens) const {
    if (actual_tokens > 0) {
      return persistent_positions_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return persistent_positions_;
  }
  torch::Tensor persistent_new_cache_slots(uint32_t actual_tokens) const {
    if (actual_tokens > 0) {
      return persistent_new_cache_slots_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return persistent_new_cache_slots_;
  }
  torch::Tensor persistent_block_tables(uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return persistent_block_tables_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return persistent_block_tables_;
  }
  torch::Tensor hidden_states(uint32_t actual_tokens) const {
    if (actual_tokens > 0) {
      return hidden_states_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return hidden_states_;
  }
  // Setter for hidden_states (for assignment)
  void set_hidden_states(const torch::Tensor& value) {
    const uint32_t result_tokens = value.size(0);
    hidden_states_.slice(/*dim=*/0, /*start=*/0, /*end=*/result_tokens)
        .copy_(value, /*non_blocking=*/true);
  }
  torch::Tensor q_seq_lens(uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return q_seq_lens_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return q_seq_lens_;
  }
  torch::Tensor kv_seq_lens(uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return kv_seq_lens_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return kv_seq_lens_;
  }
  torch::Tensor persistent_embedding(uint32_t actual_tokens) const {
    if (actual_tokens > 0) {
      return persistent_embedding_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return persistent_embedding_;
  }
  // FlashInfer decode mode parameters
  torch::Tensor persistent_paged_kv_indptr(uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return persistent_paged_kv_indptr_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1);
    }
    return persistent_paged_kv_indptr_;
  }
  torch::Tensor persistent_paged_kv_indices(uint32_t actual_size) const {
    if (actual_size > 0) {
      return persistent_paged_kv_indices_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_size);
    }
    return persistent_paged_kv_indices_;
  }
  torch::Tensor persistent_paged_kv_last_page_len(
      uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return persistent_paged_kv_last_page_len_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return persistent_paged_kv_last_page_len_;
  }
  torch::Tensor persistent_decode_qo_indptr(uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return persistent_decode_qo_indptr_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1);
    }
    return persistent_decode_qo_indptr_;
  }

  // Getter/setter for persistent two-stage decode cache
  std::optional<layer::TwoStageDecodeCache>&
  persistent_two_stage_decode_cache() {
    return persistent_two_stage_decode_cache_;
  }
  const std::optional<layer::TwoStageDecodeCache>&
  persistent_two_stage_decode_cache() const {
    return persistent_two_stage_decode_cache_;
  }

 private:
  const ModelArgs& args_;
  const torch::Device& device_;
  const runtime::Options& options_;

  // Mutex to protect buffer resize operations from concurrent access
  // Multiple CudaGraph instances may share this persistent_param_ and
  // concurrently call update(), which may trigger buffer resizing
  mutable std::mutex buffer_resize_mutex_;

  // Persistent tensors - basic parameters
  torch::Tensor persistent_tokens_;
  torch::Tensor persistent_positions_;
  torch::Tensor persistent_new_cache_slots_;
  torch::Tensor persistent_block_tables_;
  torch::Tensor hidden_states_;
  torch::Tensor q_seq_lens_;
  torch::Tensor kv_seq_lens_;
  torch::Tensor persistent_embedding_;

  // FlashInfer decode mode parameters
  torch::Tensor persistent_paged_kv_indptr_;
  torch::Tensor persistent_paged_kv_indices_;
  torch::Tensor persistent_paged_kv_last_page_len_;
  torch::Tensor persistent_decode_qo_indptr_;

  // TODO maybe not used. or use q_cu_seq_lens instead.
  torch::Tensor persistent_chunked_prefill_qo_indptr_;

  // CRITICAL FIX: Persistent two-stage decode cache for CUDA graph.
  // This cache MUST survive capture() scope to prevent use-after-free during
  // replay. During capture, xattention kernels record pointers to tensors in
  // this cache. If the cache is destroyed after capture(), replay() will access
  // freed memory, causing GPU hang.
  std::optional<layer::TwoStageDecodeCache> persistent_two_stage_decode_cache_;

  // CRITICAL FIX: Dedicated workspace buffers for CUDA graph mode (isolated
  // from eager mode). These buffers are used during graph capture and replay
  // to avoid conflicts with prefill operations that use the global
  // FlashinferWorkspace. Without this separation, prefill's plan() calls will
  // overwrite workspace data that graph replay kernels depend on, causing
  // kernel hangs.
  torch::Tensor graph_float_workspace_buffer_;
  torch::Tensor graph_int_workspace_buffer_;
  torch::Tensor graph_page_locked_int_workspace_buffer_;

 public:
  // Getters for graph workspace buffers
  torch::Tensor& graph_float_workspace_buffer() {
    return graph_float_workspace_buffer_;
  }
  torch::Tensor& graph_int_workspace_buffer() {
    return graph_int_workspace_buffer_;
  }
  torch::Tensor& graph_page_locked_int_workspace_buffer() {
    return graph_page_locked_int_workspace_buffer_;
  }
  bool has_graph_workspace_buffers() const {
    return graph_float_workspace_buffer_.defined();
  }
};

// CUDA graph executor using libtorch CUDAGraph for memory management
class CudaGraph {
 public:
  // CRITICAL FIX: Accept optional shared_capture_stream from executor to ensure
  // all CudaGraph instances use the same stream, avoiding multi-stream deadlock
  explicit CudaGraph(
      CudaGraphPersistentParam& persistent_param,
      c10::DeviceIndex device_index,
      std::optional<c10::cuda::CUDAStream> shared_stream = std::nullopt)
      : persistent_param_(persistent_param),
        device_index_(device_index),
        capture_stream_(shared_stream) {
    // If shared_stream is provided, use it; otherwise initialize_capture_stream
    // will create a new one on first capture
    if (!capture_stream_.has_value()) {
      initialize_capture_stream(device_index);
    }
  }

  // Capture computation graph for given bucket num_tokens
  bool capture(CausalLM* model,
               const ModelArgs& args,
               const runtime::Options& options,
               const torch::Tensor& tokens,
               const torch::Tensor& positions,
               const ModelInputParams& params,
               std::vector<KVCache>& kv_cache,
               uint32_t bucket_num_tokens,
               const decltype(at::cuda::graph_pool_handle())& pool);

  // Replay captured graph with new input data
  torch::Tensor replay(const torch::Tensor& tokens,
                       const torch::Tensor& positions,
                       std::vector<KVCache>& kv_cache,
                       const ModelInputParams& params);

  // Get the hidden states from the last capture
  torch::Tensor get_hidden_states(uint32_t actual_num_tokens) const {
    return persistent_param_.hidden_states(actual_num_tokens);
  }

 private:
  // Print graph held tensors for debugging
  void print_graph_tensors() const;

  // Initialize capture stream if not already initialized
  void initialize_capture_stream(c10::DeviceIndex device_index);

  // CUDA graph for capturing and replaying
  at::cuda::CUDAGraph graph_;
  uint32_t padded_num_tokens_;

  // Reference to persistent parameters (shared across multiple CudaGraph
  // instances)
  CudaGraphPersistentParam& persistent_param_;

  // Cached capture stream, initialized on first capture
  std::optional<c10::cuda::CUDAStream> capture_stream_;
  // The CUDA stream used during capture (and expected to be the replay stream).
  // This can differ from capture_stream_ if the caller's current stream is
  // already non-default when capturing.
  std::optional<c10::cuda::CUDAStream> graph_stream_;
  c10::DeviceIndex device_index_;

  // CRITICAL FIX: Store attn_metadata to keep plan_info tensors alive during
  // replay. Without this, plan_info and unshared_plan_info are destroyed after
  // capture(), causing replay to access freed memory.
  std::shared_ptr<layer::AttentionMetadata> captured_attn_metadata_;

  // CRITICAL FIX (Layer 5): Store unshared_k_caches and unshared_v_caches to
  // keep them alive during replay. The CUDA graph captures kernel calls that
  // reference these tensors' GPU addresses. Without storing them here, the
  // tensors are destroyed when the first request completes, and replay accesses
  // freed memory causing GPU hang.
  std::vector<torch::Tensor> captured_unshared_k_caches_;
  std::vector<torch::Tensor> captured_unshared_v_caches_;

  // CRITICAL FIX (Layer 6): Store full_k_caches and full_v_caches to keep them
  // alive during replay. Similar to Layer 5, xattention.forward() sets:
  //   attn_metadata.full_k_cache = params.full_k_caches[layer_id_]
  // These tensors are used in the shared stage (batch_prefill) of two-stage
  // decode.
  std::vector<torch::Tensor> captured_full_k_caches_;
  std::vector<torch::Tensor> captured_full_v_caches_;
};

// Executor implementation using CUDA graph optimization
class CudaGraphExecutorImpl : public ExecutorImpl {
 public:
  CudaGraphExecutorImpl(CausalLM* model,
                        const ModelArgs& args,
                        const torch::Device& device,
                        const runtime::Options& options);

  ~CudaGraphExecutorImpl() override = default;

  ForwardInput prepare_inputs(Batch& batch) override;

  // Execute model with graph optimization for decode phase
  torch::Tensor run(const torch::Tensor& tokens,
                    const torch::Tensor& positions,
                    std::vector<KVCache>& kv_caches,
                    const ModelInputParams& params) override;

 private:
  // not own
  CausalLM* model_;

  ModelArgs args_;
  torch::Device device_;
  runtime::Options options_;

  // Lazy-loaded CUDA graphs for different num_tokens
  absl::flat_hash_map<uint64_t, std::unique_ptr<CudaGraph>> graphs_;

  // Persistent parameters shared across all CudaGraph instances
  std::unique_ptr<CudaGraphPersistentParam> persistent_param_;

  // CUDA graph memory pool shared across all CudaGraph instances
  decltype(at::cuda::graph_pool_handle()) graph_pool_;

  // CRITICAL FIX: Shared capture stream for all CudaGraph instances to avoid
  // multi-stream dependency deadlock. Previously, each CudaGraph had its own
  // capture_stream_, causing different rounds to use different streams.
  std::optional<c10::cuda::CUDAStream> shared_capture_stream_;

  // Get bucket num_tokens for given num_tokens
  // For num_tokens < 8: use 1, 2, 4, 8
  // For num_tokens >= 8: use multiples of 8
  uint32_t get_bucket_num_tokens(uint32_t num_tokens) const;

  // Compose graph cache key from bucket size and step-level decode metadata.
  uint64_t make_graph_key(uint32_t bucket_num_tokens,
                          const ModelInputParams& params) const;
};
REGISTER_EXECUTOR("cuda", CudaGraphExecutorImpl);
}  // namespace xllm
