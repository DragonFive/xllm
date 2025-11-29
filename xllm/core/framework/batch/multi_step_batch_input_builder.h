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

// multi_step_batch_input_builder.h
#pragma once

#include <torch/torch.h>

#include <limits>
#include <unordered_set>
#include <vector>

#include "batch_input_builder.h"
#include "framework/request/mm_data.h"
#include "framework/request/sequence.h"
#include "runtime/forward_params.h"
#include "util/threadpool.h"

namespace xllm {

struct ModelArgs;

class MultiStepBatchInputBuilder : public BatchInputBuilder {
 public:
  explicit MultiStepBatchInputBuilder(
      const std::vector<Sequence*>& sequences,
      const std::vector<uint32_t>& allowed_max_tokens,
      const std::vector<torch::Tensor>& input_embeddings_vec,
      const std::vector<MMData>& mm_data_vec,
      // for global kv cache copy block from host to device
      const std::vector<CacheBlockInfo>* copy_in_cache_block_infos,
      // for global kv cache copy block from device to host
      const std::vector<CacheBlockInfo>* copy_out_cache_block_infos,
      // for beam-search
      std::vector<CacheBlockInfo>* swap_cache_block_infos,
      const ModelArgs* args,
      ThreadPool* thread_pool = nullptr);

  ~MultiStepBatchInputBuilder() = default;

 protected:
  // Core building methods - Override base class methods to provide multi-step
  // logic
  void process_single_sequence(
      int32_t seq_index,
      BatchInputBuilder::BuilderState* state_ptr = nullptr,
      std::unordered_set<int32_t>* write_block_ids_ptr = nullptr) override;

 private:
  // Rec-specific state for multi-step
  struct RecMultiStepState {
    // encoder side
    std::vector<int32_t> encoder_seq_lens;
    std::vector<int32_t> encoder_tokens_vec;  // Raw encoder tokens from all sequences
    torch::Tensor encoder_seq_lens_tensor;
    torch::Tensor encoder_token_ids;
    torch::Tensor encoder_positions;
    torch::Tensor encoder_sparse_embedding;
    torch::Tensor decoder_context_embedding;

    // multi-round generated tokens per sequence
    std::vector<std::vector<int32_t>> generated_tokens;

    // stats
    int32_t bs = 0;
    int32_t group_width = 0;
    int32_t seq_len = 0;
    int32_t encoder_max_seq_len = 0;

    // flag to indicate rec model
    bool is_rec = false;
  };

  // State management for MultiStep
  struct MultiStepBuilderState {
    // Base state from parent class
    BatchInputBuilder::BuilderState base_state;

    // Multi-step step tracking data
    std::vector<int32_t> step_tokens_vec;
    std::vector<int32_t> step_positions_vec;
    std::vector<torch::Tensor> step_mrope_positions_vec;

    // Multi-step decode state buffers
    // std::vector<int32_t> decode_flatten_tokens_vec;
    // std::vector<int32_t> decode_flatten_positions_vec;
    // std::vector<int32_t> decode_extra_token_ids;
    // std::vector<int32_t> decode_embedding_ids;
    std::vector<int32_t> decode_selected_token_idxes;
    std::vector<const RequestSamplingParam*> decode_sampling_params;
    std::vector<std::vector<int64_t>> decode_unique_token_ids_vec;
    std::vector<std::vector<int32_t>> decode_unique_token_counts_vec;
    std::vector<int32_t> decode_unique_token_lens_vec;
    std::vector<int32_t> decode_sample_idxes;
#if defined(USE_NPU)
    std::vector<int32_t> decode_seq_lens;
    std::vector<int32_t> decode_q_seq_lens;
#elif defined(USE_MLU)
    std::vector<int32_t> decode_seq_lens = {0};
    std::vector<int32_t> decode_q_seq_lens = {0};
#endif
    std::vector<int32_t> decode_positions_vec;

    // Multi-step specific metadata
    uint32_t total_steps = 0;

    // Rec-specific state for multi-step
    RecMultiStepState rec_state;
  };

  // Enhanced state
  MultiStepBuilderState multi_step_state_;

 private:
  // Override extract_tokens_and_positions to handle multi-step decode logic
  void extract_tokens_and_positions(Sequence* sequence,
                                    uint32_t n_kv_cache_tokens,
                                    uint32_t seq_len,
                                    MultiStepBuilderState* state_ptr);

  // Multi-step specific forward input conversion functions
  ForwardInput state_to_forward_input() override;
  RawForwardInput state_to_raw_forward_input(
      BatchInputBuilder::BuilderState* state_ptr = nullptr) override;

  void setup_kv_cache_info(
      Sequence* sequence,
      uint32_t n_kv_cache_tokens,
      uint32_t seq_len,
      uint32_t q_seq_len,
      BatchInputBuilder::BuilderState* state_ptr,
      std::unordered_set<int32_t>* write_block_ids_ptr) override;

  void setup_continuous_kv_cache_info(
      Sequence* sequence,
      uint32_t n_kv_cache_tokens,
      uint32_t seq_len,
      uint32_t q_seq_len,
      BatchInputBuilder::BuilderState* state_ptr) override;
};

}  // namespace xllm
