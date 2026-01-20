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

#include "sampler.h"
#include "sampling_params.h"

namespace xllm {

// Sampler Graph 执行模式
enum class SamplerGraphMode {
  GREEDY_ONLY,            // 纯 greedy：argmax
  BEAM_SEARCH_FAST_PATH,  // BeamSearch 快速路径：AIR TopK + AIR LogSoftmax
  GREEDY_WITH_LOGPROBS,   // Greedy + logprobs：argmax + topk + log_softmax
};

// Persistent Parameters for Sampler CUDA Graph
// 持有预分配的最大尺寸张量，通过 slice 返回实际尺寸
class SamplerGraphPersistentParam {
 public:
  SamplerGraphPersistentParam(uint32_t max_batch,
                              uint32_t max_vocab,
                              uint32_t max_k,
                              const torch::Device& device);

  ~SamplerGraphPersistentParam() = default;

  // 更新 persistent buffers 的数据
  void update(const torch::Tensor& logits,
              const SamplingParameters& params,
              uint32_t actual_batch,
              uint32_t actual_vocab);

  // Getter 方法 - 返回 slice 到实际尺寸的张量
  torch::Tensor persistent_logits(uint32_t batch, uint32_t vocab) const {
    if (batch > 0 && vocab > 0) {
      return persistent_logits_.slice(0, 0, batch).slice(1, 0, vocab);
    }
    return persistent_logits_;
  }

  torch::Tensor persistent_sample_logits(uint32_t batch, uint32_t vocab) const {
    if (batch > 0 && vocab > 0) {
      return persistent_sample_logits_.slice(0, 0, batch).slice(1, 0, vocab);
    }
    return persistent_sample_logits_;
  }

  torch::Tensor persistent_top_values(uint32_t batch, uint32_t k) const {
    if (batch > 0 && k > 0) {
      return persistent_top_values_.slice(0, 0, batch).slice(1, 0, k);
    }
    return persistent_top_values_;
  }

  torch::Tensor persistent_top_indices(uint32_t batch, uint32_t k) const {
    if (batch > 0 && k > 0) {
      return persistent_top_indices_.slice(0, 0, batch).slice(1, 0, k);
    }
    return persistent_top_indices_;
  }

  torch::Tensor persistent_top_logprobs(uint32_t batch, uint32_t k) const {
    if (batch > 0 && k > 0) {
      return persistent_top_logprobs_.slice(0, 0, batch).slice(1, 0, k);
    }
    return persistent_top_logprobs_;
  }

  torch::Tensor persistent_probs(uint32_t batch, uint32_t vocab) const {
    if (batch > 0 && vocab > 0) {
      return persistent_probs_.slice(0, 0, batch).slice(1, 0, vocab);
    }
    return persistent_probs_;
  }

  torch::Tensor persistent_samples(uint32_t batch) const {
    if (batch > 0) {
      return persistent_samples_.slice(0, 0, batch);
    }
    return persistent_samples_;
  }

  torch::Tensor persistent_logprobs(uint32_t batch) const {
    if (batch > 0) {
      return persistent_logprobs_.slice(0, 0, batch);
    }
    return persistent_logprobs_;
  }

  torch::Tensor persistent_temperatures(uint32_t batch) const {
    if (batch > 0) {
      return persistent_temperatures_.slice(0, 0, batch);
    }
    return persistent_temperatures_;
  }

 private:
  torch::Device device_;
  uint32_t max_batch_;
  uint32_t max_vocab_;
  uint32_t max_k_;

  // Persistent buffers - 输入张量
  torch::Tensor persistent_logits_;         // [max_batch, max_vocab]
  torch::Tensor persistent_sample_logits_;  // [max_batch, max_vocab]
  torch::Tensor persistent_temperatures_;   // [max_batch]

  // Persistent buffers - 中间结果
  torch::Tensor persistent_top_values_;    // [max_batch, max_k]
  torch::Tensor persistent_top_indices_;   // [max_batch, max_k]
  torch::Tensor persistent_top_logprobs_;  // [max_batch, max_k]
  torch::Tensor persistent_probs_;         // [max_batch, max_vocab]

  // Persistent buffers - 输出张量
  torch::Tensor persistent_samples_;   // [max_batch]
  torch::Tensor persistent_logprobs_;  // [max_batch]
};

// CUDA Graph for Sampler
// 捕获和重放 sampler 计算图
class SamplerGraph {
 public:
  SamplerGraph(SamplerGraphPersistentParam& persistent_param,
               c10::DeviceIndex device_index);

  // 捕获 sampler 计算图
  bool capture(Sampler* sampler,
               SamplerGraphMode mode,
               uint32_t bucket_batch,
               uint32_t vocab_size,
               const SamplingParameters& params,
               const decltype(at::cuda::graph_pool_handle())& pool);

  // 重放捕获的计算图
  SampleOutput replay(const torch::Tensor& logits,
                      const SamplingParameters& params);

 private:
  // 初始化 capture stream
  void initialize_capture_stream(c10::DeviceIndex device_index);

  // CUDA Graph 对象
  at::cuda::CUDAGraph graph_;

  // 引用共享的 persistent parameters
  SamplerGraphPersistentParam& persistent_param_;

  // Graph 捕获时的参数
  SamplerGraphMode mode_;
  uint32_t bucket_batch_;
  uint32_t vocab_size_;

  // Capture stream
  std::optional<c10::cuda::CUDAStream> capture_stream_;
  c10::DeviceIndex device_index_;
};

// Sampler CUDA Graph Executor
// 管理多个 SamplerGraph 实例，根据 mode 和 batch_size 选择合适的 graph
class SamplerGraphExecutor {
 public:
  SamplerGraphExecutor(uint32_t max_batch,
                       uint32_t max_vocab,
                       uint32_t max_k,
                       const torch::Device& device);

  ~SamplerGraphExecutor() = default;

  // 执行 sampler（使用 CUDA Graph 或回退到 eager 模式）
  SampleOutput forward(torch::Tensor& logits, const SamplingParameters& params);

 private:
  // 判断是否应该使用 CUDA Graph
  bool should_use_graph(const SamplingParameters& params) const;

  // 获取执行模式
  SamplerGraphMode get_mode(const SamplingParameters& params) const;

  // 获取 bucket batch size（用于减少 graph 数量）
  uint32_t get_bucket_batch_size(uint32_t batch_size) const;

  // Eager 模式的 sampler（回退使用）
  std::unique_ptr<Sampler> eager_sampler_;

  // Persistent parameters（所有 graph 共享）
  std::unique_ptr<SamplerGraphPersistentParam> persistent_param_;

  // CUDA Graph 缓存：(mode, batch_bucket) -> graph
  absl::flat_hash_map<std::pair<SamplerGraphMode, uint32_t>,
                      std::unique_ptr<SamplerGraph>>
      graphs_;

  // CUDA Graph memory pool
  decltype(at::cuda::graph_pool_handle()) graph_pool_;

  // 配置参数
  torch::Device device_;
  uint32_t max_batch_;
  uint32_t max_vocab_;
  uint32_t max_k_;
};

}  // namespace xllm
