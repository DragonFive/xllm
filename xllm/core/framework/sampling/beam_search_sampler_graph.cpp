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

#include "beam_search_sampler_graph.h"

#include <c10/cuda/CUDAGuard.h>
#include <glog/logging.h>

#include "common/global_flags.h"
#if defined(USE_CUDA)
#include "kernels/cuda/air_log_softmax_last_dim.h"
#include "kernels/cuda/air_topk_last_dim.h"
#endif
#include "kernels/cuda/cuda_ops_api.h"

namespace xllm {

// ============================================================================
// BeamSearchSamplerGraphPersistentParam Implementation
// ============================================================================

BeamSearchSamplerGraphPersistentParam::BeamSearchSamplerGraphPersistentParam(
    uint32_t max_batch,
    uint32_t max_beam,
    uint32_t max_rounds,
    uint32_t max_vocab,
    const torch::Device& device)
    : device_(device),
      max_batch_(max_batch),
      max_beam_(max_beam),
      max_rounds_(max_rounds),
      max_vocab_(max_vocab),
      max_batch_beam_(max_batch * max_beam) {
  auto options = torch::TensorOptions().device(device);

  persistent_logits_ = torch::empty({max_batch_beam_, max_vocab_},
                                    options.dtype(torch::kFloat32));
  persistent_top_values_ = torch::empty({max_batch_beam_, max_beam_},
                                        options.dtype(torch::kFloat32));
  persistent_top_indices_ =
      torch::empty({max_batch_beam_, max_beam_}, options.dtype(torch::kInt64));
  persistent_top_logprobs_ = torch::empty({max_batch_beam_, max_beam_},
                                          options.dtype(torch::kFloat32));
  persistent_temperatures_ =
      torch::empty({max_batch_beam_}, options.dtype(torch::kFloat32));

  persistent_acc_logprob_ =
      torch::empty({max_batch_beam_, 1}, options.dtype(torch::kFloat32));
  persistent_in_sequence_group_ = torch::empty(
      {max_batch_, max_beam_, max_rounds_}, options.dtype(torch::kInt32));

  persistent_out_acc_logprob_ =
      torch::empty({max_batch_beam_, 1}, options.dtype(torch::kFloat32));
  persistent_out_token_ids_ =
      torch::empty({max_batch_beam_, 1}, options.dtype(torch::kInt32));
  persistent_out_token_index_ =
      torch::empty({max_batch_beam_, 1}, options.dtype(torch::kInt32));
  persistent_out_sequence_group_ = torch::empty(
      {max_batch_, max_beam_, max_rounds_}, options.dtype(torch::kInt32));

  LOG(INFO) << "BeamSearchSamplerGraphPersistentParam initialized: max_batch="
            << max_batch_ << ", max_beam=" << max_beam_
            << ", max_rounds=" << max_rounds_ << ", max_vocab=" << max_vocab_;
}

void BeamSearchSamplerGraphPersistentParam::update(
    const torch::Tensor& logits,
    const SamplingParameters& params,
    const torch::Tensor& acc_logprob,
    const torch::Tensor& in_sequence_group,
    uint32_t actual_batch,
    uint32_t actual_beam) {
  const uint32_t batch_beam = actual_batch * actual_beam;
  const uint32_t vocab = logits.size(1);

  persistent_logits_.slice(0, 0, batch_beam)
      .slice(1, 0, vocab)
      .copy_(logits, /*non_blocking=*/true);

  torch::Tensor acc_logprob_view = acc_logprob;
  if (acc_logprob.dim() == 2 && acc_logprob.size(1) != 1) {
    acc_logprob_view = acc_logprob.view({static_cast<int64_t>(batch_beam), 1});
  }
  persistent_acc_logprob_.slice(0, 0, batch_beam)
      .copy_(acc_logprob_view, /*non_blocking=*/true);

  const uint32_t rounds = in_sequence_group.size(2);
  persistent_in_sequence_group_.slice(0, 0, actual_batch)
      .slice(1, 0, actual_beam)
      .slice(2, 0, rounds)
      .copy_(in_sequence_group, /*non_blocking=*/true);

  if (params.temperatures.defined()) {
    persistent_temperatures_.slice(0, 0, batch_beam)
        .copy_(params.temperatures, /*non_blocking=*/true);
  }
}

// ============================================================================
// BeamSearchSamplerGraph Implementation
// ============================================================================

BeamSearchSamplerGraph::BeamSearchSamplerGraph(
    BeamSearchSamplerGraphPersistentParam& persistent_param,
    c10::DeviceIndex device_index)
    : persistent_param_(persistent_param), device_index_(device_index) {}

void BeamSearchSamplerGraph::initialize_capture_stream(
    c10::DeviceIndex device_index) {
  if (!capture_stream_.has_value()) {
    capture_stream_ = at::cuda::getStreamFromPool(
        /*isHighPriority=*/false, device_index);
    LOG(INFO)
        << "BeamSearchSamplerGraph: Initialized capture stream for device "
        << device_index;
  }
}

bool BeamSearchSamplerGraph::capture(
    uint32_t batch,
    uint32_t beam,
    uint32_t step,
    uint32_t total_rounds,
    uint32_t bucket_batch,
    uint32_t vocab_size,
    const SamplingParameters& params,
    const decltype(at::cuda::graph_pool_handle())& pool) {
  initialize_capture_stream(device_index_);
  bucket_batch_ = bucket_batch;
  current_step_ = step;
  total_rounds_ = total_rounds;

  const uint32_t bucket_batch_beam = bucket_batch * beam;

  LOG(INFO) << "BeamSearchSamplerGraph: Capturing graph for batch=" << batch
            << ", beam=" << beam << ", step=" << step
            << ", total_rounds=" << total_rounds;

  auto logits =
      persistent_param_.persistent_logits(bucket_batch_beam, vocab_size);
  auto acc_logprob =
      persistent_param_.persistent_acc_logprob(bucket_batch_beam);
  auto in_sequence_group = persistent_param_.persistent_in_sequence_group(
      bucket_batch, beam, total_rounds);
  auto top_indices_buf =
      persistent_param_.persistent_top_indices(bucket_batch_beam, beam);
  auto top_logprobs_buf =
      persistent_param_.persistent_top_logprobs(bucket_batch_beam, beam);

  auto out_acc_logprob =
      persistent_param_.persistent_out_acc_logprob(bucket_batch_beam);
  auto out_token_ids =
      persistent_param_.persistent_out_token_ids(bucket_batch_beam);
  auto out_token_index =
      persistent_param_.persistent_out_token_index(bucket_batch_beam);
  auto out_sequence_group = persistent_param_.persistent_out_sequence_group(
      bucket_batch, beam, total_rounds);

  auto out_beam_count_prefix_sums = torch::empty(
      {bucket_batch + 1},
      torch::TensorOptions().device(logits.device()).dtype(torch::kInt32));

  try {
    c10::cuda::CUDAStreamGuard stream_guard(capture_stream_.value());
    graph_.capture_begin(pool);

    torch::Tensor top_values;
    torch::Tensor top_indices;
#if defined(USE_CUDA)
    if (FLAGS_enable_air_topk && logits.is_cuda()) {
      std::tie(top_values, top_indices) = xllm::kernel::cuda::air_topk_last_dim(
          logits,
          static_cast<int32_t>(beam),
          /*largest=*/true,
          /*sorted_by_value=*/FLAGS_enable_topk_sorted);
    } else {
#endif
      auto topk_result = logits.topk(beam,
                                     /*dim=*/-1,
                                     /*largest=*/true,
                                     /*sorted=*/FLAGS_enable_topk_sorted);
      top_values = std::get<0>(topk_result);
      top_indices = std::get<1>(topk_result);
#if defined(USE_CUDA)
    }
#endif

    top_indices_buf.copy_(top_indices);

#if defined(USE_CUDA)
    if (FLAGS_enable_air_topk && top_values.is_cuda()) {
      torch::Tensor temperatures;
      if (params.temperatures.defined()) {
        temperatures =
            persistent_param_.persistent_temperatures(bucket_batch_beam);
      }
      auto top_logprobs = xllm::kernel::cuda::air_log_softmax_last_dim(
          top_values, temperatures);
      top_logprobs_buf.copy_(top_logprobs);
    } else {
#endif
      auto topk_logits = top_values.to(torch::kFloat32);
      if (params.temperatures.defined()) {
        auto temperatures =
            persistent_param_.persistent_temperatures(bucket_batch_beam);
        auto unsqueezed_temperatures = temperatures.unsqueeze(1);
        unsqueezed_temperatures =
            torch::where(unsqueezed_temperatures == 0,
                         torch::ones_like(unsqueezed_temperatures),
                         unsqueezed_temperatures);
        topk_logits.div_(unsqueezed_temperatures);
      }
      auto top_logprobs = torch::log_softmax(topk_logits, /*dim=*/-1);
      top_logprobs_buf.copy_(top_logprobs);
#if defined(USE_CUDA)
    }
#endif

    xllm::kernel::cuda::beam_search(acc_logprob,
                                    in_sequence_group,
                                    top_indices_buf,
                                    top_logprobs_buf,
                                    out_acc_logprob,
                                    out_token_ids,
                                    out_token_index,
                                    out_beam_count_prefix_sums,
                                    out_sequence_group,
                                    bucket_batch,
                                    step);

    graph_.capture_end();

    LOG(INFO) << "BeamSearchSamplerGraph: Successfully captured graph for "
                 "batch="
              << batch << ", beam=" << beam << ", step=" << step;
    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "BeamSearchSamplerGraph: Failed to capture graph: "
               << e.what();
    return false;
  }
}

BeamSearchSamplerGraphOutput BeamSearchSamplerGraph::replay(
    const torch::Tensor& logits,
    const SamplingParameters& params,
    const torch::Tensor& acc_logprob,
    const torch::Tensor& in_sequence_group,
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t current_step) {
  const uint32_t batch_beam = batch_size * beam_size;

  persistent_param_.update(
      logits, params, acc_logprob, in_sequence_group, batch_size, beam_size);

  graph_.replay();

  BeamSearchSamplerGraphOutput output;
  output.top_tokens =
      persistent_param_.persistent_top_indices(batch_beam, beam_size).clone();
  output.top_logprobs =
      persistent_param_.persistent_top_logprobs(batch_beam, beam_size).clone();
  output.out_acc_logprob =
      persistent_param_.persistent_out_acc_logprob(batch_beam).clone();
  output.out_token_ids =
      persistent_param_.persistent_out_token_ids(batch_beam).clone();
  output.out_token_index =
      persistent_param_.persistent_out_token_index(batch_beam).clone();
  output.out_sequence_group =
      persistent_param_
          .persistent_out_sequence_group(
              batch_size, beam_size, in_sequence_group.size(2))
          .clone();

  return output;
}

// ============================================================================
// BeamSearchSamplerGraphExecutor Implementation
// ============================================================================

BeamSearchSamplerGraphExecutor::BeamSearchSamplerGraphExecutor(
    uint32_t max_batch,
    uint32_t max_beam,
    uint32_t max_rounds,
    uint32_t max_vocab,
    const torch::Device& device)
    : device_(device),
      max_batch_(max_batch),
      max_beam_(max_beam),
      max_rounds_(max_rounds),
      max_vocab_(max_vocab) {
  persistent_param_ = std::make_unique<BeamSearchSamplerGraphPersistentParam>(
      max_batch, max_beam, max_rounds, max_vocab, device);

  graph_pool_ = at::cuda::graph_pool_handle();

  LOG(INFO) << "BeamSearchSamplerGraphExecutor initialized: max_batch="
            << max_batch << ", max_beam=" << max_beam
            << ", max_rounds=" << max_rounds << ", max_vocab=" << max_vocab;
}

bool BeamSearchSamplerGraphExecutor::should_use_graph(
    const SamplingParameters& params,
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t total_rounds,
    const torch::Tensor& logits) const {
  if (!FLAGS_enable_beam_search_graph || !FLAGS_enable_sampler_graph) {
    return false;
  }

  if (!device_.is_cuda()) {
    return false;
  }

  if (batch_size == 0 || batch_size > max_batch_) {
    return false;
  }

  if (beam_size == 0 || beam_size > max_beam_) {
    return false;
  }

  if (total_rounds == 0 || total_rounds > max_rounds_) {
    return false;
  }

  // BeamSearch fast path constraints
  if (!params.use_beam_search || !params.logprobs ||
      params.max_top_logprobs <= 0 || params.top_p.defined() ||
      FLAGS_enable_qwen3_reranker || FLAGS_max_decode_rounds <= 0) {
    return false;
  }

  if (static_cast<uint32_t>(params.max_top_logprobs) != beam_size) {
    return false;
  }

  const uint32_t expected_batch_beam = batch_size * beam_size;
  if (logits.size(0) != static_cast<int64_t>(expected_batch_beam)) {
    return false;
  }

  return true;
}

uint32_t BeamSearchSamplerGraphExecutor::get_bucket_batch_size(
    uint32_t batch_size) const {
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
  }

  return ((batch_size + 15) / 16) * 16;
}

std::optional<BeamSearchSamplerGraphOutput>
BeamSearchSamplerGraphExecutor::forward(torch::Tensor& logits,
                                        const SamplingParameters& params,
                                        const torch::Tensor& acc_logprob,
                                        const torch::Tensor& in_sequence_group,
                                        uint32_t batch_size,
                                        uint32_t beam_size,
                                        uint32_t current_step) {
  const uint32_t total_rounds = in_sequence_group.size(2);

  if (!should_use_graph(params, batch_size, beam_size, total_rounds, logits)) {
    return std::nullopt;
  }

  const uint32_t bucket_batch = get_bucket_batch_size(batch_size);
  const auto graph_key =
      std::make_tuple(bucket_batch, current_step, total_rounds);

  auto it = graphs_.find(graph_key);
  if (it == graphs_.end()) {
    LOG(INFO) << "BeamSearchSamplerGraphExecutor: Creating new graph for "
                 "bucket_batch="
              << bucket_batch << ", step=" << current_step
              << ", total_rounds=" << total_rounds;

    auto graph = std::make_unique<BeamSearchSamplerGraph>(*persistent_param_,
                                                          device_.index());

    bool success = graph->capture(bucket_batch,
                                  beam_size,
                                  current_step,
                                  total_rounds,
                                  bucket_batch,
                                  logits.size(1),
                                  params,
                                  graph_pool_);

    if (!success) {
      LOG(WARNING)
          << "BeamSearchSamplerGraphExecutor: Failed to capture graph, "
             "falling back to eager mode";
      return std::nullopt;
    }

    it = graphs_.emplace(graph_key, std::move(graph)).first;
  }

  return it->second->replay(logits,
                            params,
                            acc_logprob,
                            in_sequence_group,
                            batch_size,
                            beam_size,
                            current_step);
}

}  // namespace xllm
