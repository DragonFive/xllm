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

namespace xllm {

// BeamSearch 输出结构
struct BeamSearchOutput {
  torch::Tensor out_acc_logprob;
  torch::Tensor out_token_ids;
  torch::Tensor out_token_index;
  torch::Tensor out_sequence_group;
};

// Persistent Parameters for BeamSearch CUDA Graph
// 持有预分配的最大尺寸张量，通过 slice 返回实际尺寸
class BeamSearchGraphPersistentParam {
 public:
  BeamSearchGraphPersistentParam(uint32_t max_batch,
                                 uint32_t max_beam,
                                 uint32_t max_rounds,
                                 const torch::Device& device);

  ~BeamSearchGraphPersistentParam() = default;

  // 更新 persistent buffers 的数据
  // 将实际输入数据复制到预分配的 persistent buffers 中
  void update(const torch::Tensor& acc_logprob,
              const torch::Tensor& in_sequence_group,
              const torch::Tensor& top_tokens,
              const torch::Tensor& top_logprobs,
              uint32_t actual_batch,
              uint32_t actual_beam);

  // Getter 方法 - 返回 slice 到实际尺寸的张量
  torch::Tensor persistent_acc_logprob(uint32_t batch, uint32_t beam) const {
    if (batch > 0 && beam > 0) {
      return persistent_acc_logprob_.slice(0, 0, batch).slice(1, 0, beam);
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

  torch::Tensor persistent_top_tokens(uint32_t batch_beam,
                                      uint32_t beam) const {
    if (batch_beam > 0 && beam > 0) {
      return persistent_top_tokens_.slice(0, 0, batch_beam).slice(1, 0, beam);
    }
    return persistent_top_tokens_;
  }

  torch::Tensor persistent_top_logprobs(uint32_t batch_beam,
                                        uint32_t beam) const {
    if (batch_beam > 0 && beam > 0) {
      return persistent_top_logprobs_.slice(0, 0, batch_beam).slice(1, 0, beam);
    }
    return persistent_top_logprobs_;
  }

  torch::Tensor persistent_out_acc_logprob(uint32_t batch,
                                           uint32_t beam) const {
    if (batch > 0 && beam > 0) {
      return persistent_out_acc_logprob_.slice(0, 0, batch).slice(1, 0, beam);
    }
    return persistent_out_acc_logprob_;
  }

  torch::Tensor persistent_out_token_ids(uint32_t batch, uint32_t beam) const {
    if (batch > 0 && beam > 0) {
      return persistent_out_token_ids_.slice(0, 0, batch).slice(1, 0, beam);
    }
    return persistent_out_token_ids_;
  }

  torch::Tensor persistent_out_token_index(uint32_t batch,
                                           uint32_t beam) const {
    if (batch > 0 && beam > 0) {
      return persistent_out_token_index_.slice(0, 0, batch).slice(1, 0, beam);
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

  // 获取缓存的 arange 张量（用于替代 torch::arange）
  torch::Tensor cached_batch_range(uint32_t batch, uint32_t beam) const {
    if (batch > 0 && beam > 0) {
      return cached_batch_range_.slice(0, 0, batch).slice(1, 0, beam);
    }
    return cached_batch_range_;
  }

  torch::Tensor cached_beam_range(uint32_t batch, uint32_t beam) const {
    if (batch > 0 && beam > 0) {
      return cached_beam_range_.slice(0, 0, batch).slice(1, 0, beam);
    }
    return cached_beam_range_;
  }

  torch::Tensor cached_indices(uint32_t beam) const {
    if (beam > 0) {
      return cached_indices_.slice(0, 0, beam);
    }
    return cached_indices_;
  }

 private:
  torch::Device device_;
  uint32_t max_batch_;
  uint32_t max_beam_;
  uint32_t max_rounds_;

  // Persistent buffers - 输入张量
  torch::Tensor persistent_acc_logprob_;        // [max_batch, max_beam]
  torch::Tensor persistent_in_sequence_group_;  // [max_batch, max_beam,
                                                // max_rounds]
  torch::Tensor persistent_top_tokens_;    // [max_batch * max_beam, max_beam]
  torch::Tensor persistent_top_logprobs_;  // [max_batch * max_beam, max_beam]

  // Persistent buffers - 输出张量
  torch::Tensor persistent_out_acc_logprob_;  // [max_batch, max_beam]
  torch::Tensor persistent_out_token_ids_;    // [max_batch, max_beam]
  torch::Tensor persistent_out_token_index_;  // [max_batch, max_beam]
  torch::Tensor
      persistent_out_sequence_group_;  // [max_batch, max_beam, max_rounds]

  // 缓存的 arange 结果（避免 torch::arange 动态分配）
  torch::Tensor cached_batch_range_;  // [max_batch, max_beam]
  torch::Tensor cached_beam_range_;   // [max_batch, max_beam]
  torch::Tensor cached_indices_;      // [max_beam]
};

// CUDA Graph for BeamSearch
// 捕获和重放 beam_search 计算图
class BeamSearchGraph {
 public:
  BeamSearchGraph(BeamSearchGraphPersistentParam& persistent_param,
                  c10::DeviceIndex device_index);

  // 捕获 beam_search 计算图
  bool capture(uint32_t batch,
               uint32_t beam,
               uint32_t step,
               uint32_t total_rounds,
               uint32_t bucket_batch,
               const decltype(at::cuda::graph_pool_handle())& pool);

  // 重放捕获的计算图
  BeamSearchOutput replay(const torch::Tensor& acc_logprob,
                          const torch::Tensor& in_sequence_group,
                          const torch::Tensor& top_tokens,
                          const torch::Tensor& top_logprobs,
                          uint32_t batch_size,
                          uint32_t current_step);

 private:
  // 初始化 capture stream
  void initialize_capture_stream(c10::DeviceIndex device_index);

  // CUDA Graph 对象
  at::cuda::CUDAGraph graph_;

  // 引用共享的 persistent parameters
  BeamSearchGraphPersistentParam& persistent_param_;

  // Graph 捕获时的参数
  uint32_t bucket_batch_;
  uint32_t current_step_;

  // Capture stream
  std::optional<c10::cuda::CUDAStream> capture_stream_;
  c10::DeviceIndex device_index_;
};

// BeamSearch CUDA Graph Executor
// 管理多个 BeamSearchGraph 实例，根据 batch_size 和 step 选择合适的 graph
class BeamSearchGraphExecutor {
 public:
  BeamSearchGraphExecutor(uint32_t max_batch,
                          uint32_t max_beam,
                          uint32_t max_rounds,
                          const torch::Device& device);

  ~BeamSearchGraphExecutor() = default;

  // 执行 beam_search（使用 CUDA Graph 或回退到 eager 模式）
  BeamSearchOutput forward(const torch::Tensor& acc_logprob,
                           const torch::Tensor& in_sequence_group,
                           const torch::Tensor& top_tokens,
                           const torch::Tensor& top_logprobs,
                           uint32_t batch_size,
                           uint32_t current_step);

 private:
  // 判断是否应该使用 CUDA Graph
  bool should_use_graph(uint32_t batch_size, uint32_t current_step) const;

  // 获取 bucket batch size（用于减少 graph 数量）
  uint32_t get_bucket_batch_size(uint32_t batch_size) const;

  // Persistent parameters（所有 graph 共享）
  std::unique_ptr<BeamSearchGraphPersistentParam> persistent_param_;

  // CUDA Graph 缓存：(batch_bucket, step) -> graph
  absl::flat_hash_map<std::tuple<uint32_t, uint32_t>,
                      std::unique_ptr<BeamSearchGraph>>
      graphs_;

  // CUDA Graph memory pool
  decltype(at::cuda::graph_pool_handle()) graph_pool_;

  // 配置参数
  torch::Device device_;
  uint32_t max_batch_;
  uint32_t max_beam_;
  uint32_t max_rounds_;
};

}  // namespace xllm
