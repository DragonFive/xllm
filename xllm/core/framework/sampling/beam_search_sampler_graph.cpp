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

#include "beam_search_sampler_graph.h"

#include <c10/cuda/CUDAGuard.h>
#include <glog/logging.h>

#include <cstdlib>

#include "common/global_flags.h"
#include "framework/model/causal_lm.h"
#if defined(USE_CUDA)
#include "kernels/cuda/air_log_softmax_last_dim.h"
#include "kernels/cuda/air_topk_last_dim.h"
#endif
#include "kernels/cuda/cuda_ops_api.h"
#include "kernels/cuda/cuda_utils.h"

namespace xllm {

namespace {
// Helper function to check if AIR topk should be used
static inline bool use_air_topk_env() {
  const char* v = std::getenv("XLLM_USE_AIR_TOPK");
  if (!v) {
    return false;
  }
  if (v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N') {
    return false;
  }
  return true;
}
}  // namespace

// ============================================================================
// BeamSearchSamplerGraphPersistentParam Implementation
// ============================================================================

BeamSearchSamplerGraphPersistentParam::BeamSearchSamplerGraphPersistentParam(
    uint32_t max_batch,
    uint32_t max_beam,
    uint32_t max_rounds,
    uint32_t hidden_size,
    torch::ScalarType hidden_dtype,
    const torch::Device& device,
    uint32_t max_prefill_tokens)
    : device_(device),
      max_batch_(max_batch),
      max_beam_(max_beam),
      max_rounds_(max_rounds),
      hidden_size_(hidden_size),
      max_batch_beam_(max_batch * max_beam),
      max_prefill_tokens_(max_prefill_tokens) {
  auto options = torch::TensorOptions().device(device);

  // Decode phase buffers
  persistent_hidden_states_ = torch::empty({max_batch_beam_, hidden_size_},
                                           options.dtype(hidden_dtype));
  persistent_selected_token_idxes_ =
      torch::empty({max_batch_beam_}, options.dtype(torch::kInt32));
  persistent_top_values_ = torch::empty({max_batch_beam_, max_beam_},
                                        options.dtype(torch::kFloat32));
  persistent_top_indices_ =
      torch::empty({max_batch_beam_, max_beam_}, options.dtype(torch::kInt32));
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

  // Prefill phase buffers (only allocate if max_prefill_tokens > 0)
  if (max_prefill_tokens_ > 0) {
    persistent_prefill_hidden_states_ = torch::empty(
        {max_prefill_tokens_, hidden_size_}, options.dtype(hidden_dtype));
    persistent_prefill_selected_idxes_ =
        torch::empty({max_batch_}, options.dtype(torch::kInt32));
  }

  LOG(INFO) << "BeamSearchSamplerGraphPersistentParam initialized: max_batch="
            << max_batch_ << ", max_beam=" << max_beam_
            << ", max_rounds=" << max_rounds_
            << ", hidden_size=" << hidden_size_
            << ", max_prefill_tokens=" << max_prefill_tokens_
            << ", hidden_dtype=" << static_cast<int>(hidden_dtype);
}

void BeamSearchSamplerGraphPersistentParam::update(
    const torch::Tensor& hidden_states,
    const SamplingParameters& params,
    const torch::Tensor& acc_logprob,
    const torch::Tensor& in_sequence_group,
    uint32_t actual_batch,
    uint32_t actual_beam) {
  const uint32_t batch_beam = actual_batch * actual_beam;
  persistent_hidden_states_.slice(0, 0, batch_beam)
      .copy_(hidden_states, /*non_blocking=*/true);
  persistent_selected_token_idxes_.slice(0, 0, batch_beam)
      .copy_(params.selected_token_idxes, /*non_blocking=*/true);

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

void BeamSearchSamplerGraphPersistentParam::update_prefill(
    const torch::Tensor& hidden_states,
    const SamplingParameters& params,
    const torch::Tensor& acc_logprob,
    const torch::Tensor& in_sequence_group,
    uint32_t actual_batch,
    uint32_t actual_beam,
    uint32_t num_prefill_tokens) {
  // Copy prefill hidden states
  persistent_prefill_hidden_states_.slice(0, 0, num_prefill_tokens)
      .copy_(hidden_states, /*non_blocking=*/true);

  // Copy selected token indices (batch_size for prefill)
  persistent_prefill_selected_idxes_.slice(0, 0, actual_batch)
      .copy_(params.selected_token_idxes, /*non_blocking=*/true);

  // Copy acc_logprob and sequence_group (same as decode)
  const uint32_t batch_beam = actual_batch * actual_beam;
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
    // For prefill, temperatures are per-batch, not per-batch-beam
    persistent_temperatures_.slice(0, 0, actual_batch)
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
    CausalLM* model,
    uint32_t batch,
    uint32_t beam,
    uint32_t step,
    uint32_t total_rounds,
    uint32_t bucket_batch,
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

  auto hidden_states =
      persistent_param_.persistent_hidden_states(bucket_batch_beam);
  auto selected_token_idxes =
      persistent_param_.persistent_selected_token_idxes(bucket_batch_beam);
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

  out_beam_count_prefix_sums_ = torch::empty({bucket_batch + 1},
                                             torch::TensorOptions()
                                                 .device(hidden_states.device())
                                                 .dtype(torch::kInt32));

  try {
    c10::cuda::CUDAStreamGuard stream_guard(capture_stream_.value());
    graph_.capture_begin(pool);

    logits_buf_ = model->logits(hidden_states, selected_token_idxes);

    torch::Tensor top_values;
    torch::Tensor top_indices;
#if defined(USE_CUDA)
    if (use_air_topk_env() && logits_buf_.is_cuda()) {
      std::tie(top_values, top_indices) = xllm::kernel::cuda::air_topk_last_dim(
          logits_buf_,
          static_cast<int32_t>(beam),
          /*largest=*/true,
          /*sorted_by_value=*/FLAGS_enable_topk_sorted);
    } else {
#endif
      auto topk_result = logits_buf_.topk(beam,
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
    if (use_air_topk_env() && top_values.is_cuda()) {
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
                                    out_beam_count_prefix_sums_,
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
    const torch::Tensor& hidden_states,
    const SamplingParameters& params,
    const torch::Tensor& acc_logprob,
    const torch::Tensor& in_sequence_group,
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t current_step) {
  const uint32_t batch_beam = batch_size * beam_size;

  persistent_param_.update(hidden_states,
                           params,
                           acc_logprob,
                           in_sequence_group,
                           batch_size,
                           beam_size);

  graph_.replay();

  BeamSearchSamplerGraphOutput output;
  output.logits = logits_buf_.slice(0, 0, batch_beam);
  output.top_tokens =
      persistent_param_.persistent_top_indices(batch_beam, beam_size);
  output.top_logprobs =
      persistent_param_.persistent_top_logprobs(batch_beam, beam_size);
  output.out_acc_logprob =
      persistent_param_.persistent_out_acc_logprob(batch_beam);
  output.out_token_ids = persistent_param_.persistent_out_token_ids(batch_beam);
  output.out_token_index =
      persistent_param_.persistent_out_token_index(batch_beam);
  output.out_sequence_group = persistent_param_.persistent_out_sequence_group(
      batch_size, beam_size, in_sequence_group.size(2));

  return output;
}

// ============================================================================
// BeamSearchSamplerGraphPrefill Implementation
// ============================================================================

BeamSearchSamplerGraphPrefill::BeamSearchSamplerGraphPrefill(
    BeamSearchSamplerGraphPersistentParam& persistent_param,
    c10::DeviceIndex device_index)
    : persistent_param_(persistent_param), device_index_(device_index) {}

void BeamSearchSamplerGraphPrefill::initialize_capture_stream(
    c10::DeviceIndex device_index) {
  if (!capture_stream_.has_value()) {
    capture_stream_ = at::cuda::getStreamFromPool(
        /*isHighPriority=*/false, device_index);
    LOG(INFO) << "BeamSearchSamplerGraphPrefill: Initialized capture stream "
                 "for device "
              << device_index;
  }
}

bool BeamSearchSamplerGraphPrefill::capture(
    CausalLM* model,
    uint32_t bucket_num_tokens,
    uint32_t batch,
    uint32_t beam,
    uint32_t total_rounds,
    const SamplingParameters& params,
    const decltype(at::cuda::graph_pool_handle())& pool) {
  initialize_capture_stream(device_index_);
  bucket_num_tokens_ = bucket_num_tokens;
  bucket_batch_ = batch;
  total_rounds_ = total_rounds;

  LOG(INFO) << "BeamSearchSamplerGraphPrefill: Capturing graph for "
               "bucket_num_tokens="
            << bucket_num_tokens << ", batch=" << batch << ", beam=" << beam
            << ", total_rounds=" << total_rounds;

  // For prefill: hidden_states is [num_tokens, hidden_size]
  // selected_token_idxes is [batch_size]
  auto hidden_states =
      persistent_param_.persistent_prefill_hidden_states(bucket_num_tokens);
  auto selected_token_idxes =
      persistent_param_.persistent_prefill_selected_idxes(batch);
  auto acc_logprob = persistent_param_.persistent_acc_logprob(batch * beam);
  auto in_sequence_group =
      persistent_param_.persistent_in_sequence_group(batch, beam, total_rounds);
  auto top_indices_buf = persistent_param_.persistent_top_indices(batch, beam);
  auto top_logprobs_buf =
      persistent_param_.persistent_top_logprobs(batch, beam);

  auto out_acc_logprob =
      persistent_param_.persistent_out_acc_logprob(batch * beam);
  auto out_token_ids = persistent_param_.persistent_out_token_ids(batch * beam);
  auto out_token_index =
      persistent_param_.persistent_out_token_index(batch * beam);
  auto out_sequence_group = persistent_param_.persistent_out_sequence_group(
      batch, beam, total_rounds);

  out_beam_count_prefix_sums_ = torch::empty({batch + 1},
                                             torch::TensorOptions()
                                                 .device(hidden_states.device())
                                                 .dtype(torch::kInt32));

  try {
    c10::cuda::CUDAStreamGuard stream_guard(capture_stream_.value());
    graph_.capture_begin(pool);

    // For prefill, logits output is [batch_size, vocab_size]
    logits_buf_ = model->logits(hidden_states, selected_token_idxes);

    torch::Tensor top_values;
    torch::Tensor top_indices;
#if defined(USE_CUDA)
    if (use_air_topk_env() && logits_buf_.is_cuda()) {
      std::tie(top_values, top_indices) = xllm::kernel::cuda::air_topk_last_dim(
          logits_buf_,
          static_cast<int32_t>(beam),
          /*largest=*/true,
          /*sorted_by_value=*/FLAGS_enable_topk_sorted);
    } else {
#endif
      auto topk_result = logits_buf_.topk(beam,
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
    if (use_air_topk_env() && top_values.is_cuda()) {
      torch::Tensor temperatures;
      if (params.temperatures.defined()) {
        temperatures = persistent_param_.persistent_temperatures(batch);
      }
      auto top_logprobs = xllm::kernel::cuda::air_log_softmax_last_dim(
          top_values, temperatures);
      top_logprobs_buf.copy_(top_logprobs);
    } else {
#endif
      auto topk_logits = top_values.to(torch::kFloat32);
      if (params.temperatures.defined()) {
        auto temperatures = persistent_param_.persistent_temperatures(batch);
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

    // For prefill (step=0), beam_search uses batch_size as the first dimension
    xllm::kernel::cuda::beam_search(acc_logprob,
                                    in_sequence_group,
                                    top_indices_buf,
                                    top_logprobs_buf,
                                    out_acc_logprob,
                                    out_token_ids,
                                    out_token_index,
                                    out_beam_count_prefix_sums_,
                                    out_sequence_group,
                                    batch,
                                    0);  // step = 0 for prefill

    graph_.capture_end();

    LOG(INFO)
        << "BeamSearchSamplerGraphPrefill: Successfully captured graph for "
           "bucket_num_tokens="
        << bucket_num_tokens << ", batch=" << batch << ", beam=" << beam;
    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "BeamSearchSamplerGraphPrefill: Failed to capture graph: "
               << e.what();
    return false;
  }
}

BeamSearchSamplerGraphOutput BeamSearchSamplerGraphPrefill::replay(
    const torch::Tensor& hidden_states,
    const SamplingParameters& params,
    const torch::Tensor& acc_logprob,
    const torch::Tensor& in_sequence_group,
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t num_prefill_tokens) {
  const uint32_t batch_beam = batch_size * beam_size;

  persistent_param_.update_prefill(hidden_states,
                                   params,
                                   acc_logprob,
                                   in_sequence_group,
                                   batch_size,
                                   beam_size,
                                   num_prefill_tokens);

  graph_.replay();

  BeamSearchSamplerGraphOutput output;
  // For prefill, logits is [batch_size, vocab_size]
  output.logits = logits_buf_.slice(0, 0, batch_size);
  // top_tokens and top_logprobs are [batch_size, beam_size]
  output.top_tokens =
      persistent_param_.persistent_top_indices(batch_size, beam_size);
  output.top_logprobs =
      persistent_param_.persistent_top_logprobs(batch_size, beam_size);
  output.out_acc_logprob =
      persistent_param_.persistent_out_acc_logprob(batch_beam);
  output.out_token_ids = persistent_param_.persistent_out_token_ids(batch_beam);
  output.out_token_index =
      persistent_param_.persistent_out_token_index(batch_beam);
  output.out_sequence_group = persistent_param_.persistent_out_sequence_group(
      batch_size, beam_size, in_sequence_group.size(2));

  return output;
}

// ============================================================================
// BeamSearchSamplerGraphExecutor Implementation
// ============================================================================

BeamSearchSamplerGraphExecutor::BeamSearchSamplerGraphExecutor(
    CausalLM* model,
    uint32_t max_batch,
    uint32_t max_beam,
    uint32_t max_rounds,
    uint32_t hidden_size,
    torch::ScalarType hidden_dtype,
    const torch::Device& device,
    uint32_t max_prefill_tokens,
    uint32_t max_seq_len)
    : model_(model),
      device_(device),
      max_batch_(max_batch),
      max_beam_(max_beam),
      max_rounds_(max_rounds),
      hidden_size_(hidden_size),
      max_prefill_tokens_(max_prefill_tokens),
      max_seq_len_(max_seq_len),
      hidden_dtype_(hidden_dtype) {
  persistent_param_ = std::make_unique<BeamSearchSamplerGraphPersistentParam>(
      max_batch,
      max_beam,
      max_rounds,
      hidden_size,
      hidden_dtype,
      device,
      max_prefill_tokens);

  graph_pool_ = at::cuda::graph_pool_handle();

  LOG(INFO) << "BeamSearchSamplerGraphExecutor initialized: max_batch="
            << max_batch << ", max_beam=" << max_beam
            << ", max_rounds=" << max_rounds << ", hidden_size=" << hidden_size
            << ", max_prefill_tokens=" << max_prefill_tokens
            << ", max_seq_len=" << max_seq_len;
}

bool BeamSearchSamplerGraphExecutor::should_use_graph(
    const SamplingParameters& params,
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t total_rounds,
    uint32_t kv_max_seq_len,
    const torch::Tensor& hidden_states) const {
  if (model_ == nullptr) {
    return false;
  }

  if (!FLAGS_enable_sampler_beamsearch_graph) {
    return false;
  }

  if (!device_.is_cuda()) {
    return false;
  }

  if (!params.selected_token_idxes.defined()) {
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

  if (hidden_states.dim() != 2) {
    return false;
  }

  if (hidden_states.size(1) != static_cast<int64_t>(hidden_size_)) {
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
  if (hidden_states.size(0) != static_cast<int64_t>(expected_batch_beam)) {
    return false;
  }

  if (params.selected_token_idxes.defined() &&
      params.selected_token_idxes.numel() !=
          static_cast<int64_t>(expected_batch_beam)) {
    return false;
  }

  // Check max_seq_len_for_graph_mode
  if (max_seq_len_ > 0 && kv_max_seq_len > max_seq_len_) {
    VLOG(1) << "KV seq len " << kv_max_seq_len
            << " exceeds max_seq_len_for_graph_mode (" << max_seq_len_ << ")";
    return false;
  }

  return true;
}

bool BeamSearchSamplerGraphExecutor::should_use_graph_prefill(
    const SamplingParameters& params,
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t total_rounds,
    uint32_t num_prefill_tokens,
    const torch::Tensor& hidden_states) const {
  if (model_ == nullptr) {
    return false;
  }

  if (!FLAGS_enable_sampler_beamsearch_graph) {
    return false;
  }

  if (!device_.is_cuda()) {
    return false;
  }

  if (!params.selected_token_idxes.defined()) {
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

  // Check max_prefill_tokens
  if (max_prefill_tokens_ == 0) {
    // Prefill graph not enabled
    return false;
  }

  // Check max_tokens_for_graph_mode_prefill
  if (FLAGS_max_tokens_for_graph_mode_prefill > 0 &&
      num_prefill_tokens >
          static_cast<uint32_t>(FLAGS_max_tokens_for_graph_mode_prefill)) {
    VLOG(1) << "Prefill tokens " << num_prefill_tokens
            << " exceeds max_tokens_for_graph_mode_prefill ("
            << FLAGS_max_tokens_for_graph_mode_prefill << ")";
    return false;
  }

  if (num_prefill_tokens > max_prefill_tokens_) {
    VLOG(1) << "Prefill tokens " << num_prefill_tokens
            << " exceeds max_prefill_tokens (" << max_prefill_tokens_ << ")";
    return false;
  }

  if (hidden_states.dim() != 2) {
    return false;
  }

  if (hidden_states.size(1) != static_cast<int64_t>(hidden_size_)) {
    return false;
  }

  // For prefill, hidden_states shape is [num_prefill_tokens, hidden_size]
  if (hidden_states.size(0) != static_cast<int64_t>(num_prefill_tokens)) {
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

  // For prefill, selected_token_idxes is [batch_size]
  if (params.selected_token_idxes.defined() &&
      params.selected_token_idxes.numel() != static_cast<int64_t>(batch_size)) {
    return false;
  }

  return true;
}

uint32_t BeamSearchSamplerGraphExecutor::get_bucket_batch_size(
    uint32_t batch_size) const {
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
  }

  return ((batch_size + 15) / 16) * 16;
}

uint32_t BeamSearchSamplerGraphExecutor::get_prefill_bucket(
    uint32_t num_tokens) const {
  // Bucket strategy consistent with CudaGraphExecutorImpl
  if (num_tokens <= 1) {
    return 1;
  } else if (num_tokens <= 2) {
    return 2;
  } else if (num_tokens <= 4) {
    return 4;
  } else if (num_tokens <= 8) {
    return 8;
  }
  return ((num_tokens + 31) / 32) * 32;
}

std::optional<BeamSearchSamplerGraphOutput>
BeamSearchSamplerGraphExecutor::forward(const torch::Tensor& hidden_states,
                                        const SamplingParameters& params,
                                        const torch::Tensor& acc_logprob,
                                        const torch::Tensor& in_sequence_group,
                                        uint32_t batch_size,
                                        uint32_t beam_size,
                                        uint32_t current_step,
                                        uint32_t kv_max_seq_len) {
  kernel::cuda::NvtxRange range("BeamSearchSamplerGraph.forward");
  const uint32_t total_rounds = in_sequence_group.size(2);

  if (!should_use_graph(params,
                        batch_size,
                        beam_size,
                        total_rounds,
                        kv_max_seq_len,
                        hidden_states)) {
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

    bool success = graph->capture(model_,
                                  bucket_batch,
                                  beam_size,
                                  current_step,
                                  total_rounds,
                                  bucket_batch,
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

  return it->second->replay(hidden_states,
                            params,
                            acc_logprob,
                            in_sequence_group,
                            batch_size,
                            beam_size,
                            current_step);
}

std::optional<BeamSearchSamplerGraphOutput>
BeamSearchSamplerGraphExecutor::forward_prefill(
    const torch::Tensor& hidden_states,
    const SamplingParameters& params,
    const torch::Tensor& acc_logprob,
    const torch::Tensor& in_sequence_group,
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t num_prefill_tokens) {
  kernel::cuda::NvtxRange range("BeamSearchSamplerGraph.forward_prefill");
  const uint32_t total_rounds = in_sequence_group.size(2);

  if (!should_use_graph_prefill(params,
                                batch_size,
                                beam_size,
                                total_rounds,
                                num_prefill_tokens,
                                hidden_states)) {
    return std::nullopt;
  }

  const uint32_t bucket_num_tokens = get_prefill_bucket(num_prefill_tokens);
  const auto graph_key =
      std::make_tuple(bucket_num_tokens, batch_size, total_rounds);

  auto it = prefill_graphs_.find(graph_key);
  if (it == prefill_graphs_.end()) {
    LOG(INFO)
        << "BeamSearchSamplerGraphExecutor: Creating new prefill graph for "
           "bucket_num_tokens="
        << bucket_num_tokens << ", batch=" << batch_size
        << ", total_rounds=" << total_rounds;

    auto graph = std::make_unique<BeamSearchSamplerGraphPrefill>(
        *persistent_param_, device_.index());

    bool success = graph->capture(model_,
                                  bucket_num_tokens,
                                  batch_size,
                                  beam_size,
                                  total_rounds,
                                  params,
                                  graph_pool_);

    if (!success) {
      LOG(WARNING)
          << "BeamSearchSamplerGraphExecutor: Failed to capture prefill graph, "
             "falling back to eager mode";
      return std::nullopt;
    }

    it = prefill_graphs_.emplace(graph_key, std::move(graph)).first;
  }

  return it->second->replay(hidden_states,
                            params,
                            acc_logprob,
                            in_sequence_group,
                            batch_size,
                            beam_size,
                            num_prefill_tokens);
}

}  // namespace xllm
