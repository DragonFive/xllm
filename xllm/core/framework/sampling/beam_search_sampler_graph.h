/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================*/

#pragma once

#include <ATen/cuda/CUDAGraph.h>
#include <absl/container/flat_hash_map.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>

#include "sampling_params.h"

namespace xllm {

class CausalLM;

// BeamSearch + Sampler CUDA Graph output structure
struct BeamSearchSamplerGraphOutput {
  torch::Tensor logits;
  torch::Tensor top_tokens;
  torch::Tensor top_logprobs;
  torch::Tensor out_acc_logprob;
  torch::Tensor out_token_ids;
  torch::Tensor out_token_index;
  torch::Tensor out_sequence_group;
};

// Persistent Parameters for BeamSearch + Sampler CUDA Graph
class BeamSearchSamplerGraphPersistentParam {
 public:
  BeamSearchSamplerGraphPersistentParam(uint32_t max_batch,
                                        uint32_t max_beam,
                                        uint32_t max_rounds,
                                        uint32_t hidden_size,
                                        torch::ScalarType hidden_dtype,
                                        const torch::Device& device);

  ~BeamSearchSamplerGraphPersistentParam() = default;

  void update(const torch::Tensor& hidden_states,
              const SamplingParameters& params,
              const torch::Tensor& acc_logprob,
              const torch::Tensor& in_sequence_group,
              uint32_t actual_batch,
              uint32_t actual_beam);

  torch::Tensor persistent_hidden_states(uint32_t batch_beam) const {
    if (batch_beam > 0) {
      return persistent_hidden_states_.slice(0, 0, batch_beam);
    }
    return persistent_hidden_states_;
  }

  torch::Tensor persistent_selected_token_idxes(uint32_t batch_beam) const {
    if (batch_beam > 0) {
      return persistent_selected_token_idxes_.slice(0, 0, batch_beam);
    }
    return persistent_selected_token_idxes_;
  }

  torch::Tensor persistent_top_values(uint32_t batch_beam,
                                      uint32_t beam) const {
    if (batch_beam > 0 && beam > 0) {
      return persistent_top_values_.slice(0, 0, batch_beam).slice(1, 0, beam);
    }
    return persistent_top_values_;
  }

  torch::Tensor persistent_top_indices(uint32_t batch_beam,
                                       uint32_t beam) const {
    if (batch_beam > 0 && beam > 0) {
      return persistent_top_indices_.slice(0, 0, batch_beam).slice(1, 0, beam);
    }
    return persistent_top_indices_;
  }

  torch::Tensor persistent_top_logprobs(uint32_t batch_beam,
                                        uint32_t beam) const {
    if (batch_beam > 0 && beam > 0) {
      return persistent_top_logprobs_.slice(0, 0, batch_beam).slice(1, 0, beam);
    }
    return persistent_top_logprobs_;
  }

  torch::Tensor persistent_temperatures(uint32_t batch_beam) const {
    if (batch_beam > 0) {
      return persistent_temperatures_.slice(0, 0, batch_beam);
    }
    return persistent_temperatures_;
  }

  torch::Tensor persistent_acc_logprob(uint32_t batch_beam) const {
    if (batch_beam > 0) {
      return persistent_acc_logprob_.slice(0, 0, batch_beam);
    }
    return persistent_acc_logprob_;
  }

  torch::Tensor persistent_in_sequence_group(uint32_t batch,
                                             uint32_t beam,
                                             uint32_t rounds) const {
    if (batch > 0 && beam > 0 && rounds > 0) {
      return persistent_in_sequence_group_.slice(0, 0, batch)
          .slice(1, 0, beam)
          .slice(2, 0, rounds);
    }
    return persistent_in_sequence_group_;
  }

  torch::Tensor persistent_out_acc_logprob(uint32_t batch_beam) const {
    if (batch_beam > 0) {
      return persistent_out_acc_logprob_.slice(0, 0, batch_beam);
    }
    return persistent_out_acc_logprob_;
  }

  torch::Tensor persistent_out_token_ids(uint32_t batch_beam) const {
    if (batch_beam > 0) {
      return persistent_out_token_ids_.slice(0, 0, batch_beam);
    }
    return persistent_out_token_ids_;
  }

  torch::Tensor persistent_out_token_index(uint32_t batch_beam) const {
    if (batch_beam > 0) {
      return persistent_out_token_index_.slice(0, 0, batch_beam);
    }
    return persistent_out_token_index_;
  }

  torch::Tensor persistent_out_sequence_group(uint32_t batch,
                                              uint32_t beam,
                                              uint32_t rounds) const {
    if (batch > 0 && beam > 0 && rounds > 0) {
      return persistent_out_sequence_group_.slice(0, 0, batch)
          .slice(1, 0, beam)
          .slice(2, 0, rounds);
    }
    return persistent_out_sequence_group_;
  }

 private:
  torch::Device device_;
  uint32_t max_batch_;
  uint32_t max_beam_;
  uint32_t max_rounds_;
  uint32_t hidden_size_;
  uint32_t max_batch_beam_;

  torch::Tensor persistent_hidden_states_;  // [max_batch_beam, hidden_size]
  torch::Tensor persistent_selected_token_idxes_;  // [max_batch_beam]
  torch::Tensor persistent_top_values_;            // [max_batch_beam, max_beam]
  torch::Tensor persistent_top_indices_;           // [max_batch_beam, max_beam]
  torch::Tensor persistent_top_logprobs_;          // [max_batch_beam, max_beam]
  torch::Tensor persistent_temperatures_;          // [max_batch_beam]
  torch::Tensor persistent_acc_logprob_;           // [max_batch_beam, 1]
  torch::Tensor persistent_in_sequence_group_;  // [max_batch, max_beam, rounds]

  torch::Tensor persistent_out_acc_logprob_;  // [max_batch_beam, 1]
  torch::Tensor persistent_out_token_ids_;    // [max_batch_beam, 1]
  torch::Tensor persistent_out_token_index_;  // [max_batch_beam, 1]
  torch::Tensor
      persistent_out_sequence_group_;  // [max_batch, max_beam, rounds]
};

// CUDA Graph for BeamSearch + Sampler
class BeamSearchSamplerGraph {
 public:
  BeamSearchSamplerGraph(
      BeamSearchSamplerGraphPersistentParam& persistent_param,
      c10::DeviceIndex device_index);

  bool capture(CausalLM* model,
               uint32_t batch,
               uint32_t beam,
               uint32_t step,
               uint32_t total_rounds,
               uint32_t bucket_batch,
               const SamplingParameters& params,
               const decltype(at::cuda::graph_pool_handle())& pool);

  BeamSearchSamplerGraphOutput replay(const torch::Tensor& hidden_states,
                                      const SamplingParameters& params,
                                      const torch::Tensor& acc_logprob,
                                      const torch::Tensor& in_sequence_group,
                                      uint32_t batch_size,
                                      uint32_t beam_size,
                                      uint32_t current_step);

 private:
  void initialize_capture_stream(c10::DeviceIndex device_index);

  at::cuda::CUDAGraph graph_;

  BeamSearchSamplerGraphPersistentParam& persistent_param_;

  torch::Tensor logits_buf_;
  torch::Tensor out_beam_count_prefix_sums_;

  uint32_t bucket_batch_;
  uint32_t current_step_;
  uint32_t total_rounds_;

  std::optional<c10::cuda::CUDAStream> capture_stream_;
  c10::DeviceIndex device_index_;
};

// BeamSearch + Sampler CUDA Graph Executor
class BeamSearchSamplerGraphExecutor {
 public:
  BeamSearchSamplerGraphExecutor(CausalLM* model,
                                 uint32_t max_batch,
                                 uint32_t max_beam,
                                 uint32_t max_rounds,
                                 uint32_t hidden_size,
                                 torch::ScalarType hidden_dtype,
                                 const torch::Device& device);

  ~BeamSearchSamplerGraphExecutor() = default;

  std::optional<BeamSearchSamplerGraphOutput> forward(
      const torch::Tensor& hidden_states,
      const SamplingParameters& params,
      const torch::Tensor& acc_logprob,
      const torch::Tensor& in_sequence_group,
      uint32_t batch_size,
      uint32_t beam_size,
      uint32_t current_step);

 private:
  bool should_use_graph(const SamplingParameters& params,
                        uint32_t batch_size,
                        uint32_t beam_size,
                        uint32_t total_rounds,
                        const torch::Tensor& hidden_states) const;

  uint32_t get_bucket_batch_size(uint32_t batch_size) const;

  CausalLM* model_;

  std::unique_ptr<BeamSearchSamplerGraphPersistentParam> persistent_param_;

  absl::flat_hash_map<std::tuple<uint32_t, uint32_t, uint32_t>,
                      std::unique_ptr<BeamSearchSamplerGraph>>
      graphs_;

  decltype(at::cuda::graph_pool_handle()) graph_pool_;

  torch::Device device_;
  uint32_t max_batch_;
  uint32_t max_beam_;
  uint32_t max_rounds_;
  uint32_t hidden_size_;
  torch::ScalarType hidden_dtype_;
};

}  // namespace xllm
