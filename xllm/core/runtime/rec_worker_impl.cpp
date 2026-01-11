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

#include "rec_worker_impl.h"

#include <glog/logging.h>

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "framework/model/model_input_params.h"
#if defined(USE_CUDA) || defined(USE_ILU)
#include "kernels/cuda/cuda_ops_api.h"
#include "layers/cuda/flashinfer_workspace.h"
#endif
#include "framework/model_loader.h"
#include "models/model_registry.h"
#include "util/env_var.h"
#include "util/timer.h"

namespace xllm {

RecWorkerImpl::LlmRecWorkPipeline::LlmRecWorkPipeline(RecWorkerImpl& worker)
    : worker_(worker) {}

bool RecWorkerImpl::LlmRecWorkPipeline::create_model(RecWorkerImpl& worker,
                                                     ModelContext& context) {
  return worker.LLMWorkerImpl::init_model(context);
}

ForwardInput RecWorkerImpl::LlmRecWorkPipeline::prepare_inputs(Batch& batch) {
  return worker_.WorkerImpl::prepare_inputs(batch);
}

void RecWorkerImpl::LlmRecWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  worker_.WorkerImpl::prepare_work_before_execute(inputs, processed_inputs);
  // LlmRecDefault (pure qwen3) does not process mm_data.
  // For mm_data processing, use LlmRecWithMmDataWorkPipeline.
}

std::optional<ForwardOutput> RecWorkerImpl::LlmRecWorkPipeline::step(
    const ForwardInput& input) {
  return worker_.LLMWorkerImpl::step(input);
}

RecWorkerImpl::OneRecWorkPipeline::OneRecWorkPipeline(RecWorkerImpl& worker)
    : worker_(worker) {}

bool RecWorkerImpl::OneRecWorkPipeline::create_model(RecWorkerImpl& worker,
                                                     ModelContext& context) {
  // OneRec also uses LLM model for now, can be extended to create_rec_model
  // later
  return worker.LLMWorkerImpl::init_model(context);
}

ForwardInput RecWorkerImpl::OneRecWorkPipeline::prepare_inputs(Batch& batch) {
  ThreadPool* thread_pool = worker_.input_builder_thread_pool_
                                ? worker_.input_builder_thread_pool_.get()
                                : nullptr;

  return batch.prepare_rec_forward_input(worker_.options_.num_decoding_tokens(),
                                         /*min_decoding_batch_size=*/0,
                                         worker_.context_.get_model_args(),
                                         thread_pool);
}

void RecWorkerImpl::OneRecWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  worker_.WorkerImpl::prepare_work_before_execute(inputs, processed_inputs);
}

std::optional<ForwardOutput> RecWorkerImpl::OneRecWorkPipeline::step(
    const ForwardInput& input) {
  Timer timer;
  worker_.device_.set_device();

  const auto& sampling_params = input.sampling_params;
  const auto& input_params = input.input_params;

  const auto* onerec_params = input_params.onerec_params();
  CHECK(onerec_params != nullptr) << "OneRec requires rec_params.";

  const OneRecModelInputParams& rec_params = *onerec_params;

  torch::Tensor hidden_states;
  if (rec_params.rec_stage == OneRecModelInputParams::RecStage::PREFILL) {
    if (!rec_params.is_first_prefill) {
      ModelInputParams decoder_params = input_params;
      decoder_params.mutable_onerec_params().is_encoder_forward = false;
      hidden_states = worker_.model_executor_->forward(
          input.token_ids, input.positions, worker_.kv_caches_, decoder_params);
    } else {
      const bool has_sparse_embedding =
          rec_params.encoder_sparse_embedding.defined();
      const bool has_encoder_tokens = rec_params.encoder_token_ids.defined() &&
                                      rec_params.encoder_positions.defined();

      if (!has_sparse_embedding && !has_encoder_tokens) {
        LOG(ERROR) << "OneRec first prefill requires encoder inputs.";
        return std::nullopt;
      }

      ModelInputParams encoder_params = input_params;
      auto& mutable_onerec_params = encoder_params.mutable_onerec_params();
      mutable_onerec_params.is_encoder_forward = true;

      torch::Tensor encoder_tokens;
      if (has_sparse_embedding) {
        mutable_onerec_params.is_hybrid_mode = true;
        encoder_tokens = rec_params.encoder_sparse_embedding;
      } else {
        encoder_tokens = rec_params.encoder_token_ids;
      }

      worker_.model_executor_->forward(encoder_tokens,
                                       rec_params.encoder_positions,
                                       worker_.kv_caches_,
                                       encoder_params);

      ModelInputParams decoder_params = input_params;
      decoder_params.mutable_onerec_params().is_encoder_forward = false;
      hidden_states = worker_.model_executor_->forward(
          input.token_ids, input.positions, worker_.kv_caches_, decoder_params);
    }
  } else {
    ModelInputParams decoder_params = input_params;
    decoder_params.mutable_onerec_params().is_encoder_forward = false;
    hidden_states = worker_.model_executor_->forward(
        input.token_ids, input.positions, worker_.kv_caches_, decoder_params);
  }

  if (!hidden_states.defined()) {
    return std::nullopt;
  }

  if (!worker_.enable_schedule_overlap() && !worker_.driver_ &&
      !worker_.dp_driver_ && !worker_.options_.enable_speculative_decode()) {
    worker_.device_.synchronize_default_stream();
    COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
    DeviceMonitor::get_instance().update_active_activation_memory(
        worker_.device_.index());
    return std::nullopt;
  }

  torch::Tensor logits;
  if (sampling_params.selected_token_idxes.defined()) {
    logits = worker_.model_->logits(hidden_states,
                                    sampling_params.selected_token_idxes);
  }

  ForwardOutput output;

  if (sampling_params.selected_token_idxes.defined()) {
    auto sample_output = worker_.sampler_->forward(logits, sampling_params);
    output.logits = logits;
    output.sample_output = sample_output;
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
  }

  worker_.device_.synchronize_default_stream();
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      worker_.device_.index());

  return output;
}

// ============================================================
// LlmRecWithMmDataWorkPipeline Implementation (qwen3 with embedding)
// ============================================================

RecWorkerImpl::LlmRecWithMmDataWorkPipeline::LlmRecWithMmDataWorkPipeline(
    RecWorkerImpl& worker)
    : worker_(worker) {}

bool RecWorkerImpl::LlmRecWithMmDataWorkPipeline::create_model(
    RecWorkerImpl& worker,
    ModelContext& context) {
  return worker.LLMWorkerImpl::init_model(context);
}

ForwardInput RecWorkerImpl::LlmRecWithMmDataWorkPipeline::prepare_inputs(
    Batch& batch) {
  return worker_.WorkerImpl::prepare_inputs(batch);
}

void RecWorkerImpl::LlmRecWithMmDataWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  worker_.WorkerImpl::prepare_work_before_execute(inputs, processed_inputs);

  if (!inputs.input_params.mm_data.valid()) {
    return;
  }

  torch::Tensor input_embedding;
  torch::Tensor input_tokens_tensor;
  torch::Tensor input_indices_tensor;

  const auto& mm_data = inputs.input_params.mm_data;
  const auto& processed_mm_data = processed_inputs.input_params.mm_data;

  if (auto res = processed_mm_data.get<torch::Tensor>(LLM_REC_INPUT_TOKENS)) {
    input_tokens_tensor = res.value();
  }

  // Input indices are generated on host side.
  if (auto res = mm_data.get<torch::Tensor>(LLM_REC_INPUT_INDICES)) {
    input_indices_tensor = res.value();
  }

  if (auto res =
          processed_mm_data.get<torch::Tensor>(LLM_REC_INPUT_EMBEDDING)) {
    input_embedding = res.value();
  }

  if (input_embedding.defined()) {
    input_embedding = input_embedding.to(worker_.dtype());
  }

  if (input_indices_tensor.defined()) {
    CHECK(input_tokens_tensor.defined())
        << "LLM_REC_INPUT_TOKENS is required when LLM_REC_INPUT_INDICES is "
           "set.";

#if defined(USE_NPU)
    layer::NpuWordEmbedding npu_word_embedding =
        worker_.get_npu_word_embedding();
    torch::Tensor input_tokens_embedding =
        npu_word_embedding(input_tokens_tensor, 0);
#else
    layer::WordEmbedding word_embedding = worker_.get_word_embedding();
    torch::Tensor input_tokens_embedding =
        word_embedding->forward(input_tokens_tensor);
#endif

    if (input_embedding.defined()) {
      torch::Tensor input_indices_cpu =
          input_indices_tensor.to(torch::kCPU).to(torch::kInt64).contiguous();
      const auto* input_indices_ptr = input_indices_cpu.data_ptr<int64_t>();
      std::vector<int64_t> input_indices(
          input_indices_ptr, input_indices_ptr + input_indices_cpu.numel());

      processed_inputs.input_params.input_embedding =
          worker_.merge_embeddings_by_indices(
              input_tokens_embedding, input_embedding, input_indices);
    } else {
      processed_inputs.input_params.input_embedding = input_tokens_embedding;
    }
  } else if (input_embedding.defined()) {
    processed_inputs.input_params.input_embedding = input_embedding;
  }
}

std::optional<ForwardOutput> RecWorkerImpl::LlmRecWithMmDataWorkPipeline::step(
    const ForwardInput& input) {
  return worker_.LLMWorkerImpl::step(input);
}

RecWorkerImpl::LlmRecPureDevicePipeline::LlmRecPureDevicePipeline(
    RecWorkerImpl& worker)
    : worker_(worker) {}

bool RecWorkerImpl::LlmRecPureDevicePipeline::create_model(
    RecWorkerImpl& worker,
    ModelContext& context) {
  // context.print();
  return worker.LLMWorkerImpl::init_model(context);
}

ForwardInput RecWorkerImpl::LlmRecPureDevicePipeline::prepare_inputs(
    Batch& batch) {
  // return worker_.WorkerImpl::prepare_inputs(batch);
  ThreadPool* thread_pool = worker_.input_builder_thread_pool_
                                ? worker_.input_builder_thread_pool_.get()
                                : nullptr;

  return batch.prepare_rec_forward_input(worker_.options_.num_decoding_tokens(),
                                         /*min_decoding_batch_size=*/0,
                                         worker_.context_.get_model_args(),
                                         thread_pool);
}

void RecWorkerImpl::LlmRecPureDevicePipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  auto dtype = worker_.dtype();
  auto device = worker_.device();
  worker_.WorkerImpl::prepare_work_before_execute(inputs, processed_inputs);

#if defined(USE_NPU) || defined(USE_CUDA)
  // step-level decode full cache: allocate/attach by step_uid metadata
  if (FLAGS_max_decode_rounds > 0) {
    auto& mip = processed_inputs.input_params;
    int32_t beam_width = processed_inputs.beam_width;
    int32_t current_round = processed_inputs.current_round;
    int32_t total_round = processed_inputs.total_round;
    const auto& shape = processed_inputs.full_kv_shape;

    CHECK(shape.size() == 3) << "the dims offull_kv_shape should be three.";

    int64_t num_tokens = shape[0];
    int64_t head_num = shape[1];
    int64_t head_dim = shape[2];
    auto kv_cache_options = torch::TensorOptions().dtype(dtype).device(device);
    int32_t num_layers = worker_.context_.get_model_args().n_layers();
    mip.full_k_caches.clear();
    mip.full_v_caches.clear();
    mip.full_k_caches.reserve(num_layers);
    mip.full_v_caches.reserve(num_layers);
    for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
      mip.full_k_caches.emplace_back(
          torch::zeros({num_tokens, head_num, head_dim}, kv_cache_options));
      mip.full_v_caches.emplace_back(
          torch::zeros({num_tokens, head_num, head_dim}, kv_cache_options));
    }

    // scalar metadata tensors (int32 on device)
    {
      auto int_options =
          torch::TensorOptions().dtype(torch::kInt32).device(device);
      mip.beam_width_tensor = torch::tensor({beam_width}, int_options);
      mip.current_round_tensor_list.clear();
      for (int r = 0; r < total_round; ++r) {
        mip.current_round_tensor_list.push_back(
            torch::tensor({r}, int_options));
      }

      // beam batch-level tensors are constructed in WorkerService
    }
    {
      auto int_options =
          torch::TensorOptions().dtype(torch::kInt32).device(device);
      const auto& dec_pos = processed_inputs.decode_positions_vec;
      mip.decode_positions_tensor_list.clear();
      if (!dec_pos.empty() && beam_width > 0 && total_round > 1) {
        const int32_t n = static_cast<int32_t>(dec_pos.size());
        for (int j = 0; j < total_round - 1; ++j) {
          std::vector<int32_t> buf;
          buf.reserve(static_cast<size_t>(n * beam_width));
          for (int i = 0; i < n; ++i) {
            const int32_t base = dec_pos[i] + j;
            for (int b = 0; b < beam_width; ++b) {
              buf.push_back(base);
            }
          }
          mip.decode_positions_tensor_list.push_back(
              torch::tensor(buf, int_options));
        }
      }
    }
  }
#endif
}

std::optional<ForwardOutput> RecWorkerImpl::LlmRecPureDevicePipeline::step(
    const ForwardInput& input) {
  LOG(INFO) << "inner LlmRecPureDevicePipeline::step";
  return step_multi_round(const_cast<ForwardInput&>(input));
}

std::optional<ForwardOutput>
RecWorkerImpl::LlmRecPureDevicePipeline::step_multi_round(ForwardInput& input) {
  auto dtype = worker_.dtype();
  auto device = worker_.device_;
  device.set_device();

  Timer timer;

  int32_t total_rounds = input.total_round;
  int32_t batch = input.input_params.num_sequences;
  int32_t beam_width = input.beam_width;
  // LOG(INFO) << "total_rounds: " << total_rounds;
  std::vector<torch::Tensor> unshared_k_caches;
  std::vector<torch::Tensor> unshared_v_caches;
  auto args = worker_.context_.get_model_args();
  int32_t layer_num = static_cast<int32_t>(args.n_layers());
  int64_t num_heads = worker_.context_.get_model_args().n_heads();
  int64_t head_dim = worker_.context_.get_model_args().head_dim();
  int64_t num_kv_heads =
      worker_.context_.get_model_args().n_kv_heads().value_or(num_heads);

  int32_t full_kv_len = input.input_params.full_k_caches[0].size(0);
  int32_t unshared_offset = batch * FLAGS_max_token_per_req;
  int32_t max_decode_step = total_rounds - 1;

  for (auto i = 0; i < layer_num; ++i) {
    auto full_k_cache = input.input_params.full_k_caches[i];
    auto full_v_cache = input.input_params.full_v_caches[i];

    LOG(INFO) << "full_k_cache.shape: " << full_k_cache.sizes();

    auto unshared_k_cache = full_k_cache.slice(0, unshared_offset, full_kv_len);
    auto unshared_v_cache = full_v_cache.slice(0, unshared_offset, full_kv_len);

    unshared_k_cache = unshared_k_cache.view(
        {batch, beam_width, max_decode_step, num_kv_heads, head_dim});
    unshared_v_cache = unshared_v_cache.view(
        {batch, beam_width, max_decode_step, num_kv_heads, head_dim});
    LOG(INFO) << "unshared_k_cache.shape: " << unshared_k_cache.sizes();
    unshared_k_caches.push_back(unshared_k_cache);
    unshared_v_caches.push_back(unshared_v_cache);
  }

  input.input_params.num_heads = num_heads;
  input.input_params.head_dim = head_dim;

  input.input_params.beam_width = beam_width;
  auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  torch::Tensor sequence_group =
      torch::zeros({batch, beam_width, total_rounds}, int_options);

  int64_t num_seq = batch * beam_width;
  LOG(INFO) << "batch: " << batch;
  LOG(INFO) << "beam_width: " << beam_width;
  LOG(INFO) << "total_rounds: " << total_rounds;
  LOG(INFO) << "num_seq: " << num_seq;
  torch::Tensor acc_logprob = torch::zeros({num_seq, 1}, fp32_options);
  torch::Tensor out_log_probs = torch::zeros({num_seq, 1}, fp32_options);
  torch::Tensor out_token_ids = torch::zeros({num_seq, 1}, int_options);
  torch::Tensor out_token_index = torch::zeros({num_seq, 1}, int_options);
  torch::Tensor out_beam_count_prefix_sums =
      torch::zeros({num_seq, 1}, int_options);
  auto out_seqgroup = sequence_group.clone();

  ForwardOutput output;

#if defined(USE_CUDA)
// input.input_params
//     .prefill_plan_info = kernel::cuda::generate_prefill_plan_info(
//     layer::FlashinferWorkspace::get_instance().get_float_workspace_buffer(),
//     layer::FlashinferWorkspace::get_instance().get_int_workspace_buffer(),
//     layer::FlashinferWorkspace::get_instance()
//         .get_page_locked_int_workspace_buffer(),
//     input.input_params.q_seq_lens,
//     input.input_params.kv_seq_lens,
//     num_heads,
//     num_kv_heads,
//     head_dim,
//     head_dim,
//     dtype,
//     dtype,
//     dtype,
//     /*enable_cuda_graph=*/false);
#endif

  for (int32_t round = 0; round < total_rounds; ++round) {
    const auto& sampling_params =
        round > 0 ? input.decoder_sampling_params : input.sampling_params;
    input.input_params.is_prefill = round == 0;

    if (!input.input_params.current_round_tensor_list.empty() && round >= 0 &&
        round < static_cast<int32_t>(
                    input.input_params.current_round_tensor_list.size())) {
      input.input_params.current_round_tensor =
          input.input_params.current_round_tensor_list[round];

      input.input_params.current_round = round - 1;
    }

    auto hidden_states = worker_.model_executor_->forward(input.token_ids,
                                                          input.positions,
                                                          worker_.kv_caches_,
                                                          input.input_params);
    // LOG(INFO) << "hidden_stats:" << hidden_states;
    if (!hidden_states.defined()) {
      return std::nullopt;
    }

    torch::Tensor logits;
    if (sampling_params.selected_token_idxes.defined()) {
      logits = worker_.model_->logits(hidden_states,
                                      sampling_params.selected_token_idxes);
    }

    if (sampling_params.selected_token_idxes.defined()) {
      auto sample_output = worker_.sampler_->forward(logits, sampling_params);
      torch::Tensor top_tokens =
          sample_output.top_tokens.to(torch::kInt32).reshape({-1, beam_width});
      torch::Tensor top_logprobs =
          sample_output.top_logprobs.reshape({-1, beam_width});

#if defined(USE_NPU)
      xllm_ops::beam_search(acc_logprob,
                            top_tokens,
                            top_logprobs,
                            sequence_group,
                            round,
                            out_token_ids,
                            out_token_index,
                            out_log_probs,
                            out_beam_count_prefix_sums,
                            out_seqgroup);
#elif defined(USE_CUDA)
      xllm::kernel::cuda::beam_search(acc_logprob,
                                      sequence_group,
                                      top_tokens,
                                      top_logprobs,
                                      out_log_probs,
                                      out_token_ids,
                                      out_token_index,
                                      out_beam_count_prefix_sums,
                                      out_seqgroup,
                                      batch,
                                      round);

#endif
      sequence_group.copy_(out_seqgroup);
      acc_logprob.copy_(out_log_probs);

      if (round < total_rounds - 1) {
        update_input_for_next_round(input,
                                    round,
                                    sample_output,
                                    out_token_ids,
                                    batch,
                                    beam_width,
                                    unshared_k_caches,
                                    unshared_v_caches,
                                    num_heads,
                                    num_kv_heads,
                                    head_dim);
        if (round > 0) {
#if defined(USE_NPU)
          xllm_ops::cache_select(out_token_index,
                                 unshared_k_caches,
                                 unshared_v_caches,
                                 input.input_params.block_tables,
                                 out_beam_count_prefix_sums,
                                 round,
                                 beam_width,
                                 layer_num);
#elif defined(USE_CUDA)
          torch::Tensor naive_block_table =
              torch::arange(batch, int_options).unsqueeze(1);
          xllm::kernel::cuda::cache_select(out_token_index,
                                           unshared_k_caches,
                                           unshared_v_caches,
                                           naive_block_table,
                                           out_beam_count_prefix_sums,
                                           round - 1,  // 对应第0步decode
                                           beam_width,
                                           layer_num);
#endif
        }
      }

      // LOG(FATAL) << "after cache_select.";
      // update output at the last round.
      if (round == total_rounds - 1) {
        // LOG(INFO) << "inner round == total_rounds - 1.";
        output.logits = logits;
        output.sample_output = sample_output;
        output.do_sample = sampling_params.do_sample;
        output.logprobs = sampling_params.logprobs;
        output.max_top_logprobs = sampling_params.max_top_logprobs;
        output.beam_search_output.src_seq_idxes = out_token_index.reshape({-1});
        output.beam_search_output.out_tokens = out_token_ids.reshape({-1});
        output.beam_search_output.out_logprobs = out_log_probs.reshape({-1});
        output.beam_sequence_group = sequence_group;
      }
    }
  }

  auto ret = device.synchronize_default_stream();
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(device.index());
  return output;
}

void RecWorkerImpl::LlmRecPureDevicePipeline::update_input_for_next_round(
    ForwardInput& input,
    int32_t round,
    const SampleOutput& sample_output,
    const torch::Tensor& out_token_ids,
    int32_t batch,
    int32_t beam_width,
    const std::vector<torch::Tensor>& unshared_k_caches,
    const std::vector<torch::Tensor>& unshared_v_caches,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim) {
  if (round == 0) {
    input.token_ids = sample_output.top_tokens.to(torch::kInt32).reshape({-1});
  } else {
    input.token_ids = out_token_ids.clone().reshape({-1});
  }

  // update next round positions.
  if (!input.input_params.decode_positions_tensor_list.empty() && round >= 0 &&
      round < static_cast<int32_t>(
                  input.input_params.decode_positions_tensor_list.size())) {
    input.positions = input.input_params.decode_positions_tensor_list[round];
  }

  // LOG(INFO) << "input.positions: " << input.positions;

  // 强制改为decode模式
  input.input_params.batch_forward_type = BatchForwardType(2);

  // 在 decode 阶段（round > 0）计算 paged_kv 相关参数，这些值在所有层都是相同的

  // 获取必要的维度信息
  int32_t batch_size = batch;
  int32_t beam_size = beam_width;
  int32_t current_step = round;  // round 0 是 prefill，round 1 是 step 0

  // 从第一层的 cache 获取维度信息（假设所有层相同）
  auto unshared_k_cache_first = unshared_k_caches[0];
  uint32_t shared_kv_len = FLAGS_max_token_per_req;
  LOG(INFO) << "shared_kv_len: " << shared_kv_len;
  uint32_t max_decode_step = unshared_k_cache_first.size(2);
  LOG(INFO) << "max_decode_step: " << max_decode_step;
  // 获取必要的 tensor
  auto kv_cu_seq_lens = input.input_params.kv_seq_lens;
  LOG(INFO) << "kv_cu_seq_lens: " << kv_cu_seq_lens;
  // 计算 batch_shared_kv_lens
  // [batch_size]
  auto batch_shared_kv_lens = torch::diff(kv_cu_seq_lens);
  LOG(INFO) << "batch_shared_kv_lens: " << batch_shared_kv_lens;
  auto paged_options =
      torch::TensorOptions().dtype(torch::kInt32).device(worker_.device_);

  // LOG(INFO) << "paged_options: " << paged_options;
  // 计算 shared_kv_indices（与 attention.cpp 中的逻辑相同）
  // [[13, 13, 13], [15, 15, 15], [16, 16, 16], ...]
  auto beam_shared_kv_expanded =
      batch_shared_kv_lens.unsqueeze(1).expand({-1, shared_kv_len});
  // LOG(INFO) << "beam_shared_kv_expanded: " << beam_shared_kv_expanded;
  auto shared_kv_len_offsets = torch::arange(0, shared_kv_len, paged_options);
  shared_kv_len_offsets =
      shared_kv_len_offsets.unsqueeze(0).expand({batch_size, -1});
  // LOG(INFO) << "shared_kv_len_offsets: " << shared_kv_len_offsets;
  auto shared_mask = shared_kv_len_offsets < beam_shared_kv_expanded;
  shared_mask = shared_mask.unsqueeze(1).expand({-1, beam_size, -1});
  // LOG(INFO) << "shared_mask: " << shared_mask;

  // auto batch_offsets = torch::arange(0, batch_size, paged_options);
  // auto shared_batch_offsets = batch_offsets.unsqueeze(1).expand({-1,
  // shared_kv_len});
  // [[0, 0, 0], [0, 0, 0], [0, 0, 0], ...]
  auto shared_batch_offsets =
      torch::zeros({batch_size, shared_kv_len}, paged_options);
  // LOG(INFO) << "shared_batch_offsets: " << shared_batch_offsets;
  // 这个tensor是确定每个请求的shared_kv的基址的，现在是按照shared_kv_len直接均匀划分的
  // 但是也可以参考batch_shared_kv_lens，按照请求的真实长度划分，这样prefill_reshape_and_cache就很简单了
  // shared_batch_offsets = shared_batch_offsets * shared_kv_len;
  // kv_cu_seq_lens: [0, 13, 28, 44]
  // kv_cu_seq_lens.slice(0, 0, -1): [0, 13, 28]
  // shared_batch_offsets: [[0, 0, 0], [0, 0, 0], [0, 0, 0], ...]
  shared_batch_offsets =
      shared_batch_offsets +
      kv_cu_seq_lens.slice(0, 0, -1).unsqueeze(1).expand({-1, shared_kv_len});
  // LOG(INFO) << "shared_batch_offsets: " << shared_batch_offsets;
  // shared_batch_offsets: [[0, 0, 0], [13, 13, 13], [28, 28, 28], ...]
  // shared_kv_len_offsets: [[0, 1, 2], [0, 1, 2], [0, 1, 2], ...]
  auto shared_kv_indices = shared_batch_offsets + shared_kv_len_offsets;
  shared_kv_indices =
      shared_kv_indices.unsqueeze(1).expand({-1, beam_size, -1});
  // LOG(INFO) << "shared_kv_indices: " << shared_kv_indices;
  // shared_kv_indices: [[0, 1, 2], [13, 14, 15], [28, 29, 30], ...]

  // shared_kv_indices = shared_kv_indices.masked_fill(~mask, 0);
  // LOG(INFO) << "shared_kv_indices: " << shared_kv_indices;

  // 计算 unshared_kv_indices
  uint32_t unshared_begin_index = shared_kv_len * batch_size;
  // auto batch_ids = input.input_params.paged_kv_indices;
  auto batch_ids = torch::arange(0, batch_size, paged_options);
  batch_ids = batch_ids.unsqueeze(1)
                  .expand({-1, beam_size})
                  .unsqueeze(2)
                  .expand({-1, -1, max_decode_step});
  batch_ids = batch_ids * beam_size * max_decode_step;
  // LOG(INFO) << "batch_ids: " << batch_ids;
  auto beams_ids = torch::arange(0, beam_size, paged_options);
  beams_ids = beams_ids.unsqueeze(0)
                  .expand({batch_size, -1})
                  .unsqueeze(2)
                  .expand({-1, -1, max_decode_step});
  beams_ids = beams_ids * max_decode_step;
  // LOG(INFO) << "beams_ids: " << beams_ids;
  auto max_decode_step_ids = torch::arange(0, max_decode_step, paged_options);
  max_decode_step_ids = max_decode_step_ids.unsqueeze(0)
                            .expand({batch_size, -1})
                            .unsqueeze(1)
                            .expand({-1, beam_size, -1});
  // LOG(INFO) << "max_decode_step_ids: " << max_decode_step_ids;
  auto unshared_kv_offsets = batch_ids + beams_ids + max_decode_step_ids;
  // LOG(INFO) << "unshared_kv_offsets: " << unshared_kv_offsets;
  auto unshared_kv_indices = unshared_kv_offsets + unshared_begin_index;
  // unshared_kv_indices = unshared_kv_indices.view({batch_size, -1});
  // LOG(INFO) << "unshared_kv_indices: " << unshared_kv_indices;
  // // 合并 shared 和 unshared indices
  // shared_kv_indices = shared_kv_indices.unsqueeze(1).expand({-1, beam_size,
  // -1}); auto full_kv_indices = torch::cat({shared_kv_indices,
  // unshared_kv_indices}, 2); LOG(INFO) << "full_kv_indices: " <<
  // full_kv_indices;
  // // 计算 mask
  // auto shared_mask = mask.unsqueeze(1).expand({-1, beam_size, -1});
  auto unshared_mask = max_decode_step_ids <= current_step;
  // unshared_mask = unshared_mask.view({batch_size, -1});
  // LOG(INFO) << "unshared_mask: " << unshared_mask;
  auto full_mask = torch::cat({shared_mask, unshared_mask}, 2);
  // LOG(INFO) << "full_mask: " << full_mask;

  // torch::Tensor shared_kv_indices = torch::arange(0, shared_kv_len,
  // paged_options); shared_kv_indices =
  // shared_kv_indices.unsqueeze(0).expand({batch_size, -1}); shared_kv_indices
  // = shared_kv_indices.unsqueeze(1).expand({-1, beam_size, -1}); LOG(INFO) <<
  // "shared_kv_indices: " << shared_kv_indices;
  torch::Tensor full_kv_indices =
      torch::cat({shared_kv_indices, unshared_kv_indices}, 2);
  // LOG(INFO) << "full_kv_indices: " << full_kv_indices;
  full_kv_indices = full_kv_indices.masked_select(full_mask);
  // LOG(INFO) << "full_kv_indices: " << full_kv_indices;
  // LOG(INFO) << "shared_kv_indices: " << shared_kv_indices;
  // LOG(FATAL) << "after.";
  // auto full_mask = torch::cat({shared_mask, unshared_mask}, 2);
  // LOG(INFO) << "full_mask: " << full_mask;
  // // 过滤 indices
  // auto paged_kv_indices = full_kv_indices.masked_select(full_mask);
  auto paged_kv_indices = full_kv_indices;
  // LOG(INFO) << "paged_kv_indices: " << paged_kv_indices;
  // 计算 paged_kv_indptr
  auto batch_beam_shared_kv_lens =
      batch_shared_kv_lens.unsqueeze(1).expand({-1, beam_size});
  uint32_t unshared_kv_len = current_step + 1;
  batch_beam_shared_kv_lens = batch_beam_shared_kv_lens + unshared_kv_len;
  auto flattened = batch_beam_shared_kv_lens.flatten();
  auto cumsum_result = torch::cumsum(flattened, 0);
  auto paged_kv_indptr = torch::cat(
      {torch::zeros({1}, paged_options), cumsum_result.to(paged_options)}, 0);

  // 计算 paged_kv_last_page_len
  auto paged_kv_last_page_len =
      torch::ones({batch_size * beam_size}, paged_options);

  // 设置到 input_params 中
  input.input_params.decode_paged_kv_indices = paged_kv_indices;
  input.input_params.decode_paged_kv_indptr = paged_kv_indptr;
  input.input_params.decode_paged_kv_last_page_len = paged_kv_last_page_len;
  LOG(INFO) << "input.input_params.decode_paged_kv_indices: "
            << paged_kv_indices;
  LOG(INFO) << "input.input_params.decode_paged_kv_indptr: " << paged_kv_indptr;
  LOG(INFO) << "input.input_params.decode_paged_kv_last_page_len: "
            << paged_kv_last_page_len;
  // LOG(INFO) << "input.input_params.decode_paged_kv_indices: " <<
  // input.input_params.decode_paged_kv_indices; LOG(INFO) <<
  // "input.input_params.decode_paged_kv_indptr: " <<
  // input.input_params.decode_paged_kv_indptr; LOG(INFO) <<
  // "input.input_params.decode_paged_kv_last_page_len: " <<
  // input.input_params.decode_paged_kv_last_page_len; LOG(FATAL) << "after
  // update_input_for_decode.";
  // #if defined(USE_CUDA)
  // // 创建一个 dummy query tensor 用于获取维度信息
  // // query shape: [batch * beam, num_heads, head_dim]
  // auto dummy_query = torch::empty(
  //     {batch_size * beam_size, num_heads, head_dim},
  //     torch::TensorOptions().dtype(worker_.dtype()).device(worker_.device()));
  // //
  // 由于里面用的是unshared_k_cache_first.size(2)，size(1)代表block_size,代表num_kv_heads，所以需要先view一下
  // unshared_k_cache_first =
  //     unshared_k_cache_first.view({-1, 1, num_kv_heads, head_dim});
  // input.input_params.decode_plan_info =
  // kernel::cuda::generate_decode_plan_info(
  //     layer::FlashinferWorkspace::get_instance().get_float_workspace_buffer(),
  //     layer::FlashinferWorkspace::get_instance().get_int_workspace_buffer(),
  //     layer::FlashinferWorkspace::get_instance()
  //         .get_page_locked_int_workspace_buffer(),
  //     paged_kv_indptr,
  //     paged_kv_last_page_len,
  //     dummy_query,
  //     unshared_k_cache_first,
  //     unshared_v_caches[0],
  //     /*window_left=*/0,  // TODO: 从 input_params 获取
  //     /*enable_cuda_graph=*/false);
  // #endif
}

RecWorkerImpl::RecWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : LLMWorkerImpl(parallel_args, device, options) {
  if (!is_driver()) {
    return;
  }

  const int64_t num_threads = std::max<int64_t>(
      1, util::get_int_env("XLLM_REC_INPUT_BUILDER_THREADS", 16));
  input_builder_thread_pool_ =
      std::make_shared<ThreadPool>(static_cast<size_t>(num_threads));
}

bool RecWorkerImpl::init_model(ModelContext& context) {
  const auto& model_type = context.get_model_args().model_type();
  rec_model_kind_ = get_rec_model_kind(model_type);
  CHECK(rec_model_kind_ != RecModelKind::kNone)
      << "Unsupported rec model_type: " << model_type;

  // Create work pipeline first
  auto pipeline_type = get_rec_pipeline_type(rec_model_kind_);
  work_pipeline_ = create_pipeline(pipeline_type, *this);

  // Let pipeline create model
  return work_pipeline_->create_model(*this, context);
}

ForwardInput RecWorkerImpl::prepare_inputs(Batch& batch) {
  CHECK(work_pipeline_ != nullptr) << "RecWorkerImpl is not initialized.";
  return work_pipeline_->prepare_inputs(batch);
}

void RecWorkerImpl::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  CHECK(work_pipeline_ != nullptr) << "RecWorkerImpl is not initialized.";
  work_pipeline_->prepare_work_before_execute(inputs, processed_inputs);
}

torch::Tensor RecWorkerImpl::merge_embeddings_by_indices(
    const torch::Tensor& input_tokens_embedding,
    const torch::Tensor& input_embedding,
    const std::vector<int64_t>& input_indices) {
  CHECK_EQ(input_embedding.dim(), 2);
  CHECK_EQ(input_tokens_embedding.dim(), 2);
  CHECK_EQ(input_tokens_embedding.size(1), input_embedding.size(1));
  CHECK_EQ(input_tokens_embedding.dtype(), input_embedding.dtype());
  CHECK_EQ(input_tokens_embedding.device(), input_embedding.device());

  const int64_t total_rows =
      input_tokens_embedding.size(0) + input_embedding.size(0);
  const int64_t cols = input_embedding.size(1);

  torch::Device device = input_embedding.device();
  torch::Tensor merged = torch::empty(
      {total_rows, cols}, torch::dtype(input_embedding.dtype()).device(device));

  std::vector<int64_t> input_embedding_indices;
  for (int64_t i = 0; i < total_rows; ++i) {
    if (std::find(input_indices.begin(), input_indices.end(), i) ==
        input_indices.end()) {
      input_embedding_indices.push_back(i);
    }
  }

  CHECK_EQ(input_embedding_indices.size(), input_embedding.size(0));

  torch::Tensor input_embedding_indices_tensor =
      torch::tensor(input_embedding_indices, torch::kInt64).to(device);
  merged.index_put_({input_embedding_indices_tensor, torch::indexing::Ellipsis},
                    input_embedding);

  torch::Tensor input_indices_tensor =
      torch::tensor(input_indices, torch::kInt64).to(device);
  merged.index_put_({input_indices_tensor, torch::indexing::Ellipsis},
                    input_tokens_embedding);

  return merged;
}

std::optional<ForwardOutput> RecWorkerImpl::step(const ForwardInput& input) {
  CHECK(work_pipeline_ != nullptr) << "RecWorkerImpl is not initialized.";
  return work_pipeline_->step(input);
}

// ============================================================
// RecWorkerImpl pipeline factory (static method)
// ============================================================
std::unique_ptr<RecWorkerImpl::RecWorkPipeline> RecWorkerImpl::create_pipeline(
    RecPipelineType type,
    RecWorkerImpl& worker) {
  switch (type) {
    case RecPipelineType::kLlmRecDefault:
      return std::make_unique<LlmRecWorkPipeline>(worker);
    case RecPipelineType::kLlmRecWithMmData:
      return std::make_unique<LlmRecWithMmDataWorkPipeline>(worker);
    case RecPipelineType::kOneRecDefault:
      return std::make_unique<OneRecWorkPipeline>(worker);
    case RecPipelineType::kLlmRecPureDevicePipeline:
      return std::make_unique<LlmRecPureDevicePipeline>(worker);
    default:
      LOG(FATAL) << "Unknown RecWorkerImpl pipeline type: "
                 << static_cast<int>(type);
      return nullptr;
  }
}

}  // namespace xllm
