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

#include "sampler_graph.h"

#include <c10/cuda/CUDAGuard.h>
#include <glog/logging.h>

#include "core/common/global_flags.h"
#include "core/kernels/cuda/air_log_softmax_last_dim.h"
#include "core/kernels/cuda/air_topk_last_dim.h"
#include "logits_utils.h"

namespace xllm {

// ============================================================================
// SamplerGraphPersistentParam Implementation
// ============================================================================

SamplerGraphPersistentParam::SamplerGraphPersistentParam(
    uint32_t max_batch,
    uint32_t max_vocab,
    uint32_t max_k,
    const torch::Device& device)
    : device_(device),
      max_batch_(max_batch),
      max_vocab_(max_vocab),
      max_k_(max_k) {
  // 预分配所有 persistent buffers（最大尺寸）
  auto options = torch::TensorOptions().device(device);

  // 输入张量
  persistent_logits_ =
      torch::empty({max_batch, max_vocab}, options.dtype(torch::kFloat32));
  persistent_sample_logits_ =
      torch::empty({max_batch, max_vocab}, options.dtype(torch::kFloat32));
  persistent_temperatures_ =
      torch::empty({max_batch}, options.dtype(torch::kFloat32));

  // 中间结果
  persistent_top_values_ =
      torch::empty({max_batch, max_k}, options.dtype(torch::kFloat32));
  persistent_top_indices_ =
      torch::empty({max_batch, max_k}, options.dtype(torch::kInt64));
  persistent_top_logprobs_ =
      torch::empty({max_batch, max_k}, options.dtype(torch::kFloat32));
  persistent_probs_ =
      torch::empty({max_batch, max_vocab}, options.dtype(torch::kFloat32));

  // 输出张量
  persistent_samples_ = torch::empty({max_batch}, options.dtype(torch::kInt64));
  persistent_logprobs_ =
      torch::empty({max_batch}, options.dtype(torch::kFloat32));

  LOG(INFO) << "SamplerGraphPersistentParam initialized: max_batch="
            << max_batch << ", max_vocab=" << max_vocab << ", max_k=" << max_k;
}

void SamplerGraphPersistentParam::update(const torch::Tensor& logits,
                                         const SamplingParameters& params,
                                         uint32_t actual_batch,
                                         uint32_t actual_vocab) {
  // 将实际输入数据复制到 persistent buffers 的对应位置
  // 使用 non_blocking=true 以提高性能

  // logits: [actual_batch, actual_vocab]
  persistent_logits_.slice(0, 0, actual_batch)
      .slice(1, 0, actual_vocab)
      .copy_(logits, /*non_blocking=*/true);

  // 如果有 temperatures，也复制进来
  if (params.temperatures.defined()) {
    persistent_temperatures_.slice(0, 0, actual_batch)
        .copy_(params.temperatures, /*non_blocking=*/true);
  }
}

// ============================================================================
// SamplerGraph Implementation
// ============================================================================

SamplerGraph::SamplerGraph(SamplerGraphPersistentParam& persistent_param,
                           c10::DeviceIndex device_index)
    : persistent_param_(persistent_param),
      device_index_(device_index),
      mode_(SamplerGraphMode::GREEDY_ONLY),
      bucket_batch_(0),
      vocab_size_(0) {}

void SamplerGraph::initialize_capture_stream(c10::DeviceIndex device_index) {
  if (!capture_stream_.has_value()) {
    capture_stream_ = at::cuda::getStreamFromPool(
        /*isHighPriority=*/false, device_index);
    LOG(INFO) << "SamplerGraph: Initialized capture stream for device "
              << device_index;
  }
}

bool SamplerGraph::capture(
    Sampler* sampler,
    SamplerGraphMode mode,
    uint32_t bucket_batch,
    uint32_t vocab_size,
    const SamplingParameters& params,
    const decltype(at::cuda::graph_pool_handle())& pool) {
  // 初始化 capture stream
  initialize_capture_stream(device_index_);

  mode_ = mode;
  bucket_batch_ = bucket_batch;
  vocab_size_ = vocab_size;

  LOG(INFO) << "SamplerGraph: Capturing graph for mode="
            << static_cast<int>(mode) << ", bucket_batch=" << bucket_batch
            << ", vocab_size=" << vocab_size;

  // 获取 persistent buffers
  auto logits = persistent_param_.persistent_logits(bucket_batch, vocab_size);
  auto sample_logits =
      persistent_param_.persistent_sample_logits(bucket_batch, vocab_size);

  try {
    // 开始捕获 CUDA Graph
    c10::cuda::CUDAStreamGuard stream_guard(capture_stream_.value());
    graph_.capture_begin(pool);

    // 根据不同模式捕获不同的计算图
    if (mode == SamplerGraphMode::BEAM_SEARCH_FAST_PATH) {
      // BeamSearch 快速路径：AIR TopK + AIR LogSoftmax
      auto top_values = persistent_param_.persistent_top_values(
          bucket_batch, params.max_top_logprobs);
      auto top_indices = persistent_param_.persistent_top_indices(
          bucket_batch, params.max_top_logprobs);
      auto top_logprobs = persistent_param_.persistent_top_logprobs(
          bucket_batch, params.max_top_logprobs);

      // AIR TopK
      if (FLAGS_enable_air_topk && logits.is_cuda()) {
        std::tie(top_values, top_indices) =
            xllm::kernel::cuda::air_topk_last_dim(
                logits,
                static_cast<int32_t>(params.max_top_logprobs),
                /*largest=*/true,
                /*sorted_by_value=*/false,
                /*stable=*/false);

        // AIR LogSoftmax
        torch::Tensor temperatures;
        if (params.temperatures.defined()) {
          temperatures =
              persistent_param_.persistent_temperatures(bucket_batch);
        }
        top_logprobs = xllm::kernel::cuda::air_log_softmax_last_dim(
            top_values, temperatures);
      } else {
        // Fallback to torch::topk + log_softmax
        std::tie(top_values, top_indices) = logits.topk(params.max_top_logprobs,
                                                        /*dim=*/-1,
                                                        /*largest=*/true,
                                                        /*sorted=*/false);

        auto topk_logits = top_values.to(torch::kFloat32);
        if (params.temperatures.defined()) {
          auto temperatures =
              persistent_param_.persistent_temperatures(bucket_batch);
          auto unsqueezed_temperatures = temperatures.unsqueeze(1);
          unsqueezed_temperatures =
              torch::where(unsqueezed_temperatures == 0,
                           torch::ones_like(unsqueezed_temperatures),
                           unsqueezed_temperatures);
          topk_logits.div_(unsqueezed_temperatures);
        }
        top_logprobs = torch::log_softmax(topk_logits, /*dim=*/-1);
      }

    } else if (mode == SamplerGraphMode::GREEDY_ONLY) {
      // 纯 Greedy：argmax
      auto samples = persistent_param_.persistent_samples(bucket_batch);
      samples.copy_(logits.argmax(/*dim=*/-1));

    } else if (mode == SamplerGraphMode::GREEDY_WITH_LOGPROBS) {
      // Greedy + logprobs：argmax + topk + log_softmax
      auto samples = persistent_param_.persistent_samples(bucket_batch);
      auto logprobs_tensor =
          persistent_param_.persistent_logprobs(bucket_batch);

      // Greedy sample
      samples.copy_(logits.argmax(/*dim=*/-1));

      // Compute logprobs
      auto log_probs =
          torch::log_softmax(logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
      auto selected_logprobs =
          log_probs.gather(/*dim=*/-1, samples.view({-1, 1}));
      logprobs_tensor.copy_(selected_logprobs.view({-1}));

      // Top-k logprobs (if needed)
      if (params.max_top_logprobs > 0) {
        auto top_values = persistent_param_.persistent_top_values(
            bucket_batch, params.max_top_logprobs);
        auto top_indices = persistent_param_.persistent_top_indices(
            bucket_batch, params.max_top_logprobs);
        std::tie(top_values, top_indices) =
            log_probs.topk(params.max_top_logprobs, /*dim=*/-1);
      }
    }

    // 结束捕获
    graph_.capture_end();

    LOG(INFO) << "SamplerGraph: Successfully captured graph for mode="
              << static_cast<int>(mode) << ", bucket_batch=" << bucket_batch;
    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "SamplerGraph: Failed to capture graph: " << e.what();
    return false;
  }
}

SampleOutput SamplerGraph::replay(const torch::Tensor& logits,
                                  const SamplingParameters& params) {
  const uint32_t batch_size = logits.size(0);
  const uint32_t vocab_size = logits.size(1);

  // 更新 persistent buffers
  persistent_param_.update(logits, params, batch_size, vocab_size);

  // 重放 CUDA Graph
  graph_.replay();

  // 从 persistent buffers 中提取结果
  SampleOutput output;

  if (mode_ == SamplerGraphMode::BEAM_SEARCH_FAST_PATH) {
    // BeamSearch 快速路径：返回 top_tokens 和 top_logprobs
    output.top_tokens =
        persistent_param_
            .persistent_top_indices(batch_size, params.max_top_logprobs)
            .clone();
    output.top_logprobs =
        persistent_param_
            .persistent_top_logprobs(batch_size, params.max_top_logprobs)
            .clone();

  } else if (mode_ == SamplerGraphMode::GREEDY_ONLY) {
    // 纯 Greedy：返回 next_tokens
    output.next_tokens =
        persistent_param_.persistent_samples(batch_size).clone();

  } else if (mode_ == SamplerGraphMode::GREEDY_WITH_LOGPROBS) {
    // Greedy + logprobs：返回 next_tokens, logprobs, top_logprobs, top_tokens
    output.next_tokens =
        persistent_param_.persistent_samples(batch_size).clone();
    output.logprobs = persistent_param_.persistent_logprobs(batch_size).clone();

    if (params.max_top_logprobs > 0) {
      output.top_logprobs =
          persistent_param_
              .persistent_top_logprobs(batch_size, params.max_top_logprobs)
              .clone();
      output.top_tokens =
          persistent_param_
              .persistent_top_indices(batch_size, params.max_top_logprobs)
              .clone();
    }
  }

  return output;
}

// ============================================================================
// SamplerGraphExecutor Implementation
// ============================================================================

SamplerGraphExecutor::SamplerGraphExecutor(uint32_t max_batch,
                                           uint32_t max_vocab,
                                           uint32_t max_k,
                                           const torch::Device& device)
    : device_(device),
      max_batch_(max_batch),
      max_vocab_(max_vocab),
      max_k_(max_k) {
  // 创建 eager 模式的 sampler（回退使用）
  eager_sampler_ = std::make_unique<Sampler>();

  // 创建共享的 persistent parameters
  persistent_param_ = std::make_unique<SamplerGraphPersistentParam>(
      max_batch, max_vocab, max_k, device);

  // 创建 CUDA Graph memory pool
  graph_pool_ = at::cuda::graph_pool_handle();

  LOG(INFO) << "SamplerGraphExecutor initialized: max_batch=" << max_batch
            << ", max_vocab=" << max_vocab << ", max_k=" << max_k;
}

bool SamplerGraphExecutor::should_use_graph(
    const SamplingParameters& params) const {
  // 检查是否启用了 Sampler Graph
  if (!FLAGS_enable_sampler_graph) {
    return false;
  }

  // 检查是否在 CUDA 设备上
  if (!device_.is_cuda()) {
    return false;
  }

  // 检查是否是支持的模式
  // 不支持 random sampling（multinomial 有随机状态）
  if (params.all_random_sample) {
    return false;
  }

  // 不支持 top_p（动态 mask）
  if (params.top_p.defined()) {
    return false;
  }

  // 不支持 mixed sampling
  if (!params.all_greedy_sample && !params.use_beam_search) {
    return false;
  }

  return true;
}

SamplerGraphMode SamplerGraphExecutor::get_mode(
    const SamplingParameters& params) const {
  // BeamSearch 快速路径
  if (params.use_beam_search && params.logprobs &&
      params.max_top_logprobs > 0 && !params.top_p.defined() &&
      !FLAGS_enable_qwen3_reranker && FLAGS_max_decode_rounds > 0) {
    return SamplerGraphMode::BEAM_SEARCH_FAST_PATH;
  }

  // Greedy + logprobs
  if (params.all_greedy_sample && params.logprobs) {
    return SamplerGraphMode::GREEDY_WITH_LOGPROBS;
  }

  // 纯 Greedy
  if (params.all_greedy_sample) {
    return SamplerGraphMode::GREEDY_ONLY;
  }

  // 默认回退到 eager 模式
  return SamplerGraphMode::GREEDY_ONLY;
}

uint32_t SamplerGraphExecutor::get_bucket_batch_size(
    uint32_t batch_size) const {
  // 复用 CudaGraphExecutorImpl 的 bucket 策略
  if (FLAGS_enable_graph_no_padding) {
    return batch_size;
  }

  if (batch_size <= 1) {
    return 1;
  } else if (batch_size <= 2) {
    return 2;
  } else if (batch_size <= 4) {
    return 4;
  } else if (batch_size <= 8) {
    return 8;
  } else {
    // 向上取整到 16 的倍数
    return ((batch_size + 15) / 16) * 16;
  }
}

SampleOutput SamplerGraphExecutor::forward(torch::Tensor& logits,
                                           const SamplingParameters& params) {
  // 判断是否使用 CUDA Graph
  if (!should_use_graph(params)) {
    // 回退到 eager 模式
    return eager_sampler_->forward(logits, params);
  }

  const uint32_t batch_size = logits.size(0);
  const uint32_t vocab_size = logits.size(1);

  // 获取执行模式
  const SamplerGraphMode mode = get_mode(params);

  // 使用 CUDA Graph
  const uint32_t bucket_batch = get_bucket_batch_size(batch_size);
  const auto graph_key = std::make_pair(mode, bucket_batch);

  // Lazy capture：如果 graph 不存在，则创建并捕获
  auto it = graphs_.find(graph_key);
  if (it == graphs_.end()) {
    LOG(INFO) << "SamplerGraphExecutor: Creating new graph for mode="
              << static_cast<int>(mode) << ", bucket_batch=" << bucket_batch;

    auto graph =
        std::make_unique<SamplerGraph>(*persistent_param_, device_.index());

    // 捕获 graph
    bool success = graph->capture(eager_sampler_.get(),
                                  mode,
                                  bucket_batch,
                                  vocab_size,
                                  params,
                                  graph_pool_);

    if (!success) {
      LOG(WARNING)
          << "SamplerGraphExecutor: Failed to capture graph, falling back to "
             "eager mode";
      // 回退到 eager 模式
      return eager_sampler_->forward(logits, params);
    }

    // 保存 graph
    it = graphs_.emplace(graph_key, std::move(graph)).first;
  }

  // 重放 graph
  return it->second->replay(logits, params);
}

}  // namespace xllm
