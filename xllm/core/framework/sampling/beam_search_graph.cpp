/* Copyright 2025 The xLLM Authors. All Rights Reserved.
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

#include "beam_search_graph.h"

#include <c10/cuda/CUDAGuard.h>
#include <glog/logging.h>

#include "common/global_flags.h"
#include "kernels/cuda/cuda_ops_api.h"

namespace xllm {

// ============================================================================
// BeamSearchGraphPersistentParam Implementation
// ============================================================================

BeamSearchGraphPersistentParam::BeamSearchGraphPersistentParam(
    uint32_t max_batch,
    uint32_t max_beam,
    uint32_t max_rounds,
    const torch::Device& device)
    : device_(device),
      max_batch_(max_batch),
      max_beam_(max_beam),
      max_rounds_(max_rounds) {
  // 预分配所有 persistent buffers（最大尺寸）
  auto options = torch::TensorOptions().device(device);

  // 输入张量
  persistent_acc_logprob_ =
      torch::empty({max_batch, max_beam}, options.dtype(torch::kFloat32));
  persistent_in_sequence_group_ = torch::empty(
      {max_batch, max_beam, max_rounds}, options.dtype(torch::kInt32));
  persistent_top_tokens_ = torch::empty({max_batch * max_beam, max_beam},
                                        options.dtype(torch::kInt32));
  persistent_top_logprobs_ = torch::empty({max_batch * max_beam, max_beam},
                                          options.dtype(torch::kFloat32));

  // 输出张量
  persistent_out_acc_logprob_ =
      torch::empty({max_batch * max_beam, 1}, options.dtype(torch::kFloat32));
  persistent_out_token_ids_ =
      torch::empty({max_batch * max_beam, 1}, options.dtype(torch::kInt32));
  persistent_out_token_index_ =
      torch::empty({max_batch * max_beam, 1}, options.dtype(torch::kInt32));
  persistent_out_sequence_group_ = torch::empty(
      {max_batch, max_beam, max_rounds}, options.dtype(torch::kInt32));

  // 预分配缓存的 arange 张量（避免 torch::arange 动态分配）
  // 这些张量在 capture 时创建，replay 时复用
  cached_batch_range_ =
      torch::empty({max_batch, max_beam}, options.dtype(torch::kInt32));
  cached_beam_range_ =
      torch::empty({max_batch, max_beam}, options.dtype(torch::kInt32));
  cached_indices_ = torch::empty({max_beam}, options.dtype(torch::kInt32));

  // 初始化 cached_batch_range_ 和 cached_beam_range_
  // 这样在 capture 和 replay 时可以直接使用
  auto batch_arange = torch::arange(max_batch, options.dtype(torch::kInt32))
                          .unsqueeze(1)
                          .expand({max_batch, max_beam});
  auto beam_arange = torch::arange(max_beam, options.dtype(torch::kInt32))
                         .unsqueeze(0)
                         .expand({max_batch, max_beam});
  auto indices_arange = torch::arange(max_beam, options.dtype(torch::kInt32));

  cached_batch_range_.copy_(batch_arange);
  cached_beam_range_.copy_(beam_arange);
  cached_indices_.copy_(indices_arange);

  LOG(INFO) << "BeamSearchGraphPersistentParam initialized: max_batch="
            << max_batch << ", max_beam=" << max_beam
            << ", max_rounds=" << max_rounds;
}

void BeamSearchGraphPersistentParam::update(
    const torch::Tensor& acc_logprob,
    const torch::Tensor& in_sequence_group,
    const torch::Tensor& top_tokens,
    const torch::Tensor& top_logprobs,
    uint32_t actual_batch,
    uint32_t actual_beam) {
  // 将实际输入数据复制到 persistent buffers 的对应位置
  // 使用 non_blocking=true 以提高性能

  // acc_logprob: expected [actual_batch, actual_beam], but rec pipeline passes
  // [actual_batch * actual_beam, 1]. Reshape if needed.
  torch::Tensor acc_logprob_view = acc_logprob;
  if (acc_logprob.dim() == 2 && acc_logprob.size(1) == 1 &&
      acc_logprob.size(0) == static_cast<int64_t>(actual_batch * actual_beam)) {
    acc_logprob_view = acc_logprob.view({static_cast<int64_t>(actual_batch),
                                         static_cast<int64_t>(actual_beam)});
  }
  persistent_acc_logprob_.slice(0, 0, actual_batch)
      .slice(1, 0, actual_beam)
      .copy_(acc_logprob_view, /*non_blocking=*/true);

  // in_sequence_group: [actual_batch, actual_beam, rounds]
  const uint32_t rounds = in_sequence_group.size(2);
  persistent_in_sequence_group_.slice(0, 0, actual_batch)
      .slice(1, 0, actual_beam)
      .slice(2, 0, rounds)
      .copy_(in_sequence_group, /*non_blocking=*/true);

  const uint32_t batch_beam = static_cast<uint32_t>(top_tokens.size(0));
  persistent_top_tokens_.slice(0, 0, batch_beam)
      .slice(1, 0, actual_beam)
      .copy_(top_tokens, /*non_blocking=*/true);
  persistent_top_logprobs_.slice(0, 0, batch_beam)
      .slice(1, 0, actual_beam)
      .copy_(top_logprobs, /*non_blocking=*/true);
}

// ============================================================================
// BeamSearchGraph Implementation
// ============================================================================

BeamSearchGraph::BeamSearchGraph(
    BeamSearchGraphPersistentParam& persistent_param,
    c10::DeviceIndex device_index)
    : persistent_param_(persistent_param), device_index_(device_index) {}

void BeamSearchGraph::initialize_capture_stream(c10::DeviceIndex device_index) {
  if (!capture_stream_.has_value()) {
    capture_stream_ = at::cuda::getStreamFromPool(
        /*isHighPriority=*/false, device_index);
    LOG(INFO) << "BeamSearchGraph: Initialized capture stream for device "
              << device_index;
  }
}

bool BeamSearchGraph::capture(
    uint32_t batch,
    uint32_t beam,
    uint32_t step,
    uint32_t total_rounds,
    uint32_t bucket_batch,
    const decltype(at::cuda::graph_pool_handle())& pool) {
  initialize_capture_stream(device_index_);
  bucket_batch_ = bucket_batch;
  current_step_ = step;

  LOG(INFO) << "BeamSearchGraph: Capturing graph for batch=" << batch
            << ", beam=" << beam << ", step=" << step
            << ", total_rounds=" << total_rounds;

  auto acc_logprob =
      persistent_param_.persistent_acc_logprob(bucket_batch, beam);
  auto in_sequence_group = persistent_param_.persistent_in_sequence_group(
      bucket_batch, beam, total_rounds);
  auto top_tokens =
      persistent_param_.persistent_top_tokens(bucket_batch * beam, beam);
  auto top_logprobs =
      persistent_param_.persistent_top_logprobs(bucket_batch * beam, beam);

  auto out_acc_logprob =
      persistent_param_.persistent_out_acc_logprob(bucket_batch, beam);
  auto out_token_ids =
      persistent_param_.persistent_out_token_ids(bucket_batch, beam);
  auto out_token_index =
      persistent_param_.persistent_out_token_index(bucket_batch, beam);
  auto out_sequence_group = persistent_param_.persistent_out_sequence_group(
      bucket_batch, beam, total_rounds);

  auto out_beam_count_prefix_sums = torch::empty(
      {bucket_batch + 1},
      torch::TensorOptions().device(acc_logprob.device()).dtype(torch::kInt32));

  try {
    c10::cuda::CUDAStreamGuard stream_guard(capture_stream_.value());
    graph_.capture_begin(pool);

    xllm::kernel::cuda::beam_search(acc_logprob,
                                    in_sequence_group,
                                    top_tokens,
                                    top_logprobs,
                                    out_acc_logprob,
                                    out_token_ids,
                                    out_token_index,
                                    out_beam_count_prefix_sums,
                                    out_sequence_group,
                                    bucket_batch,
                                    step);

    graph_.capture_end();

    LOG(INFO) << "BeamSearchGraph: Successfully captured graph for batch="
              << batch << ", beam=" << beam << ", step=" << step;
    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "BeamSearchGraph: Failed to capture graph: " << e.what();
    return false;
  }
}

BeamSearchGraphOutput BeamSearchGraph::replay(
    const torch::Tensor& acc_logprob,
    const torch::Tensor& in_sequence_group,
    const torch::Tensor& top_tokens,
    const torch::Tensor& top_logprobs,
    uint32_t batch_size,
    uint32_t current_step) {
  const uint32_t beam_size = in_sequence_group.size(1);

  // 更新 persistent buffers
  persistent_param_.update(acc_logprob,
                           in_sequence_group,
                           top_tokens,
                           top_logprobs,
                           batch_size,
                           beam_size);

  // 重放 CUDA Graph
  graph_.replay();

  // 从 persistent buffers 中提取结果（使用实际 batch_size）
  BeamSearchGraphOutput output;
  output.out_acc_logprob =
      persistent_param_.persistent_out_acc_logprob(batch_size, beam_size)
          .clone();
  output.out_token_ids =
      persistent_param_.persistent_out_token_ids(batch_size, beam_size).clone();
  output.out_token_index =
      persistent_param_.persistent_out_token_index(batch_size, beam_size)
          .clone();
  output.out_sequence_group =
      persistent_param_
          .persistent_out_sequence_group(
              batch_size, beam_size, in_sequence_group.size(2))
          .clone();

  return output;
}

// ============================================================================
// BeamSearchGraphExecutor Implementation
// ============================================================================

BeamSearchGraphExecutor::BeamSearchGraphExecutor(uint32_t max_batch,
                                                 uint32_t max_beam,
                                                 uint32_t max_rounds,
                                                 const torch::Device& device)
    : device_(device),
      max_batch_(max_batch),
      max_beam_(max_beam),
      max_rounds_(max_rounds) {
  // 创建共享的 persistent parameters
  persistent_param_ = std::make_unique<BeamSearchGraphPersistentParam>(
      max_batch, max_beam, max_rounds, device);

  // 创建 CUDA Graph memory pool
  graph_pool_ = at::cuda::graph_pool_handle();

  LOG(INFO) << "BeamSearchGraphExecutor initialized: max_batch=" << max_batch
            << ", max_beam=" << max_beam << ", max_rounds=" << max_rounds;
}

bool BeamSearchGraphExecutor::should_use_graph(uint32_t batch_size,
                                               uint32_t current_step) const {
  // 检查是否启用了 BeamSearch Graph
  if (!FLAGS_enable_sampler_beamsearch_graph) {
    return false;
  }

  // 检查 batch_size 是否在合理范围内
  if (batch_size == 0 || batch_size > max_batch_) {
    return false;
  }

  // 检查是否在 CUDA 设备上
  if (!device_.is_cuda()) {
    return false;
  }

  return true;
}

uint32_t BeamSearchGraphExecutor::get_bucket_batch_size(
    uint32_t batch_size) const {
  // 复用 CudaGraphExecutorImpl 的 bucket 策略
  if (FLAGS_enable_graph_mode_decode_no_padding) {
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

BeamSearchGraphOutput BeamSearchGraphExecutor::forward(
    const torch::Tensor& acc_logprob,
    const torch::Tensor& in_sequence_group,
    const torch::Tensor& top_tokens,
    const torch::Tensor& top_logprobs,
    uint32_t batch_size,
    uint32_t current_step) {
  const uint32_t beam_size = in_sequence_group.size(1);
  const uint32_t total_rounds = in_sequence_group.size(2);

  // 判断是否使用 CUDA Graph
  if (!should_use_graph(batch_size, current_step)) {
    // 回退到 eager 模式（直接调用 beam_search kernel）
    BeamSearchGraphOutput output;
    output.out_acc_logprob =
        torch::empty_like(acc_logprob, acc_logprob.options());
    output.out_token_ids = torch::empty({batch_size, beam_size},
                                        torch::TensorOptions()
                                            .device(acc_logprob.device())
                                            .dtype(torch::kInt32));
    output.out_token_index = torch::empty_like(output.out_token_ids);
    output.out_sequence_group = torch::empty_like(in_sequence_group);

    auto out_beam_count_prefix_sums =
        torch::empty({batch_size + 1},
                     torch::TensorOptions()
                         .device(acc_logprob.device())
                         .dtype(torch::kInt32));

    xllm::kernel::cuda::beam_search(acc_logprob,
                                    in_sequence_group,
                                    top_tokens,
                                    top_logprobs,
                                    output.out_acc_logprob,
                                    output.out_token_ids,
                                    output.out_token_index,
                                    out_beam_count_prefix_sums,
                                    output.out_sequence_group,
                                    batch_size,
                                    current_step);

    return output;
  }

  const uint32_t bucket_batch = get_bucket_batch_size(batch_size);
  const auto graph_key = std::make_tuple(bucket_batch, current_step);

  auto it = graphs_.find(graph_key);
  if (it == graphs_.end()) {
    LOG(INFO) << "BeamSearchGraphExecutor: Creating new graph for bucket_batch="
              << bucket_batch << ", step=" << current_step;

    auto graph =
        std::make_unique<BeamSearchGraph>(*persistent_param_, device_.index());

    bool success = graph->capture(bucket_batch,
                                  beam_size,
                                  current_step,
                                  total_rounds,
                                  bucket_batch,
                                  graph_pool_);

    if (!success) {
      LOG(WARNING)
          << "BeamSearchGraphExecutor: Failed to capture graph, falling back "
             "to eager mode";
      BeamSearchGraphOutput output;
      output.out_acc_logprob =
          torch::empty_like(acc_logprob, acc_logprob.options());
      output.out_token_ids = torch::empty({batch_size, beam_size},
                                          torch::TensorOptions()
                                              .device(acc_logprob.device())
                                              .dtype(torch::kInt32));
      output.out_token_index = torch::empty_like(output.out_token_ids);
      output.out_sequence_group = torch::empty_like(in_sequence_group);

      auto out_beam_count_prefix_sums =
          torch::empty({batch_size + 1},
                       torch::TensorOptions()
                           .device(acc_logprob.device())
                           .dtype(torch::kInt32));

      xllm::kernel::cuda::beam_search(acc_logprob,
                                      in_sequence_group,
                                      top_tokens,
                                      top_logprobs,
                                      output.out_acc_logprob,
                                      output.out_token_ids,
                                      output.out_token_index,
                                      out_beam_count_prefix_sums,
                                      output.out_sequence_group,
                                      batch_size,
                                      current_step);

      return output;
    }

    it = graphs_.emplace(graph_key, std::move(graph)).first;
  }

  return it->second->replay(acc_logprob,
                            in_sequence_group,
                            top_tokens,
                            top_logprobs,
                            batch_size,
                            current_step);
}

}  // namespace xllm
