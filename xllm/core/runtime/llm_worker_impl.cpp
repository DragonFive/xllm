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

#include "llm_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/mspti_helper.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/sampling/rec_constrained_decoding.h"
#include "framework/state_dict/state_dict.h"
#include "models/model_registry.h"
#include "util/threadpool.h"
#include "util/timer.h"
#include "util/utils.h"
#if defined(USE_NPU)
#include <tuple>

#include "kernels/npu/xllm_ops/beam_search_group.h"
#include "kernels/npu/xllm_ops/cache_select.h"
#endif

namespace xllm {

LLMWorkerImpl::LLMWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {}

bool LLMWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";
  device_.set_device();

  // Try to create a causal LM model
  model_ = create_llm_model(context);

  // Dont find model in causal models
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_executor_ = std::make_unique<Executor>(
      model_.get(), context.get_model_args(), device_, options_);

  if (FLAGS_enable_eplb) {
    eplb_executor_ = std::make_unique<EplbExecutor>(model_.get(), device_);
  }

  if (FLAGS_enable_beam_search_kernel) {
    beam_searcher_ = std::make_unique<BeamSearcher>();
  }

  // Initialize constrained decoding for rec backend
  if (FLAGS_enable_constrained_decoding && FLAGS_backend == "rec") {
    const auto& model_args = context.get_model_args();
    const auto vocab_size = model_args.vocab_size();
    if (vocab_size > 0) {
      std::string model_version = model_args.model_version();
      if (model_version.empty()) {
        model_version = "0";
        VLOG(1) << "model_version is empty, defaulting to 0 for constrained "
                   "decoding";
      }
      constrained_decoding_ = std::make_unique<RecConstrainedDecoding>(
          model_version,
          static_cast<int32_t>(vocab_size),
          dtype_,
          device_.unwrap());
      // Build mask cache
      if (!constrained_decoding_->build_mask_cache()) {
        LOG(WARNING) << "Failed to build constrained decoding mask cache";
      } else {
        LOG(INFO) << "Constrained decoding initialized with model_version: "
                  << model_version << ", vocab_size: " << vocab_size;
      }
    } else {
      LOG(WARNING) << "Constrained decoding enabled but vocab_size not "
                      "available";
    }
  }

  return true;
}

std::optional<ForwardOutput> LLMWorkerImpl::step(
    const BatchedForwardInputs& inputs) {
  device_.set_device();
  Timer timer;
  if (!inputs.micro_inputs.empty() && inputs.micro_inputs[0].total_round > 0) {
    return step_multi_round(inputs);
  }
  std::vector<torch::Tensor> flatten_tokens_micro_batches;
  std::vector<torch::Tensor> flatten_positions_micro_batches;
  std::vector<ModelInputParams> input_params_micro_batches;
  auto& concated_sampling_params = inputs.concated_sampling_params;

  std::vector<folly::SemiFuture<bool>> futures;

  for (auto i = 0; i < inputs.micro_inputs.size(); ++i) {
    flatten_tokens_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].token_ids));
    flatten_positions_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].positions));
    input_params_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].input_params));

    if (options_.kv_cache_transfer_mode() == "PUSH" &&
        !inputs.micro_inputs[i].transfer_kv_infos.empty()) {
#if defined(USE_NPU)
      std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
          std::make_shared<NPULayerSynchronizerImpl>(
              context_.get_model_args().n_layers());
      const_cast<ModelInputParams*>(&(input_params_micro_batches[i]))
          ->layer_synchronizer = layer_synchronizer;

      futures.emplace_back(kv_cache_transfer_->push_kv_blocks_async(
          inputs.micro_inputs[i].transfer_kv_infos,
          context_.get_parallel_args(),
          layer_synchronizer,
          is_spec_draft_));
#endif
    }
  }
  if (FLAGS_enable_eplb) {
    eplb_executor_->eplb_execute(inputs.micro_inputs[0].eplb_info);
  }

  // temporarily use [0], will be adapted in next pr
  // call model executor forward to get hidden states
  auto hidden_states = model_executor_->forward(flatten_tokens_micro_batches,
                                                flatten_positions_micro_batches,
                                                kv_caches_,
                                                input_params_micro_batches);
  if (!hidden_states.defined()) {
    return std::nullopt;
  }

  torch::Tensor logits;
  if (concated_sampling_params.selected_token_idxes.defined()) {
    logits = model_->logits(hidden_states,
                            concated_sampling_params.selected_token_idxes);
  }

  ForwardOutput output;
  if (FLAGS_enable_eplb) {
    output.expert_load_data = expert_load_data_;
    output.prepared_layer_id = eplb_executor_->get_ready_layer_id();
    if (output.prepared_layer_id != -1) {
      eplb_executor_->reset_ready_layer_id();
    }
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_ &&
      !options_.enable_speculative_decode()) {
    auto ret = device_.synchronize_default_stream();
    // in p-d disaggregation scene, all micro batches should be in same
    // prefill/decode stage, so, to judge transfer_kv_infos.empty,
    // just use micro inputs.micro_inputs[0] here
    if (options_.kv_cache_transfer_mode() == "PUSH" &&
        !inputs.micro_inputs[0].transfer_kv_infos.empty()) {
      auto results =
          folly::collectAll(futures).within(std::chrono::seconds(60)).get();
      for (const auto& result : results) {
        if (!result.value()) {
          LOG(ERROR) << "kv_cache_transfer_ failed";
          return std::nullopt;
        }
      }
    }
    if (FLAGS_enable_eplb) {
      return output;
    }
    return std::nullopt;
  }

  // driver prepare model output
  SampleOutput sample_output;
  if (concated_sampling_params.selected_token_idxes.defined()) {
    sample_output = sampler_->forward(logits, concated_sampling_params);
    output.logits = logits;

    // beam search kernel
    BeamSearchOutput beam_search_output;
    if (concated_sampling_params.use_beam_search &&
        inputs.acc_logprob.defined() && inputs.acc_logprob.numel() > 0) {
      beam_search_output = beam_searcher_->forward(inputs.acc_logprob,
                                                   sample_output.top_tokens,
                                                   sample_output.top_logprobs);
    }

    // set sample output to output
    output.sample_output = sample_output;
    // carry over the sampling params
    output.do_sample = concated_sampling_params.do_sample;
    output.logprobs = concated_sampling_params.logprobs;
    output.max_top_logprobs = concated_sampling_params.max_top_logprobs;
    // set beam search output to output
    output.beam_search_output = beam_search_output;
  }

  // if running in multi_stream_parallel step, all micro batches
  // should be in same prefill stage, so, to judge empty_kv_cache,
  // just use micro batch 0 here
  if (options_.enable_speculative_decode() && !is_spec_draft_) {
    if (input_params_micro_batches[0].q_seq_lens_vec[0] > 1) {
      output.sample_output.embeddings = hidden_states;
    } else if (concated_sampling_params.sample_idxes.defined()) {
      // auto sample_idxes =
      //     concated_sampling_params.selected_token_idxes.index_select(
      //         /*dim=*/0, concated_sampling_params.sample_idxes);
      auto embeddings = hidden_states.index_select(
          /*dim=*/0, concated_sampling_params.sample_idxes);
      output.sample_output.embeddings = embeddings;
    }
  }

  // if running in multi_stream_parallel step, all micro batches
  // should be in same prefill stage, so, to judge empty_kv_cache,
  // just use micro batch 0 here
  if (options_.enable_speculative_decode() && !is_spec_draft_) {
    if (input_params_micro_batches[0].q_seq_lens_vec[0] > 1) {
      output.sample_output.embeddings = hidden_states;
    } else if (concated_sampling_params.sample_idxes.defined()) {
      // auto sample_idxes =
      //     concated_sampling_params.selected_token_idxes.index_select(
      //         /*dim=*/0, concated_sampling_params.sample_idxes);
      auto embeddings = hidden_states.index_select(
          /*dim=*/0, concated_sampling_params.sample_idxes);
      output.sample_output.embeddings = embeddings;
    }
  }

  auto ret = device_.synchronize_default_stream();

  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !inputs.micro_inputs[0].transfer_kv_infos.empty()) {
    auto results =
        folly::collectAll(futures).within(std::chrono::seconds(60)).get();
    for (const auto& result : results) {
      if (!result.value()) {
        LOG(ERROR) << "kv_cache_transfer_ failed";
        return std::nullopt;
      }
    }
  }

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      device_.index());

  return output;
}

std::optional<ForwardOutput> LLMWorkerImpl::step_multi_round(
    const BatchedForwardInputs& inputs) {
  device_.set_device();
  Timer timer;

  if (!inputs.micro_inputs.empty()) {
    const auto& first_micro = inputs.micro_inputs[0];
    if (!first_micro.token_ids.defined() || !first_micro.positions.defined()) {
      LOG(FATAL) << "Empty token_ids or positions in multi-step input";
    }
  }

  std::vector<torch::Tensor> flatten_tokens_micro_batches;
  std::vector<torch::Tensor> flatten_positions_micro_batches;
  std::vector<ModelInputParams> input_params_micro_batches;
  std::vector<folly::SemiFuture<bool>> futures;

  for (auto i = 0; i < inputs.micro_inputs.size(); ++i) {
    flatten_tokens_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].token_ids));
    flatten_positions_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].positions));
    input_params_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].input_params));
  }

  int32_t total_rounds = inputs.micro_inputs[0].total_round;
  ForwardOutput output;

  std::vector<torch::Tensor> unshared_k_cache;
  std::vector<torch::Tensor> unshared_v_cache;
  auto args = context_.get_model_args();
  int32_t layer_num = static_cast<int32_t>(args.n_layers());
  for (auto i = 0; i < layer_num; ++i) {
    unshared_k_cache.push_back(kv_caches_[i].get_k_cache());
    unshared_v_cache.push_back(kv_caches_[i].get_v_cache());
  }
  int32_t batch = input_params_micro_batches.empty()
                      ? 0
                      : input_params_micro_batches[0].num_sequences;
  int32_t beam_width_init = inputs.micro_inputs[0].beam_width;
  auto int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device_);
  auto fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device_);
  torch::Tensor sequence_group =
      torch::zeros({batch, beam_width_init, total_rounds}, int_options);
  // preallocate outputs and cached inputs
  int64_t num_seq = batch * beam_width_init;
  torch::Tensor acc_logprob = torch::zeros({num_seq, 1}, fp32_options);
  torch::Tensor out_token_ids = torch::empty({num_seq, 1}, int_options);
  torch::Tensor out_token_index = torch::empty({num_seq, 1}, int_options);
  torch::Tensor out_beam_count_prefix_sums =
      torch::empty({num_seq, 1}, int_options);

  // ========== Rec backend: detect and setup ==========
  const bool is_rec_backend = (FLAGS_backend == "rec");
  bool has_encoder_inputs = false;

  // Initialize generated_tokens_cpu for constrained decoding
  // Shape will be set lazily per round to match current sampling rows
  std::vector<std::vector<int32_t>> generated_tokens_cpu;
  const bool use_constrained_decoding =
      is_rec_backend && constrained_decoding_ != nullptr;

  // Async state for constrained decoding optimization
  std::optional<folly::SemiFuture<torch::Tensor>> mask_future;
  std::optional<folly::SemiFuture<folly::Unit>> update_future;

  if (is_rec_backend && !input_params_micro_batches.empty() &&
      input_params_micro_batches[0].rec_params.has_value()) {
    auto& rec_params = input_params_micro_batches[0].rec_params.value();

    // Check for encoder inputs
    if ((rec_params.encoder_token_ids.defined() &&
         rec_params.encoder_positions.defined()) ||
        rec_params.encoder_sparse_embedding.defined()) {
      has_encoder_inputs = true;

      // Set hybrid mode if sparse embedding is defined
      if (rec_params.encoder_sparse_embedding.defined()) {
        input_params_micro_batches[0].rec_params->is_hybrid_mode = true;
      }
    }
    // Convert decoder_context_embedding to model dtype if defined
    // (serialization converts to fp32, need to restore to model dtype)
    if (rec_params.decoder_context_embedding.defined()) {
      input_params_micro_batches[0].rec_params->decoder_context_embedding =
          rec_params.decoder_context_embedding.to(dtype_).to(device_);
    }
  } else {
    // no rec params present
  }

  for (int32_t round = 0; round < total_rounds; ++round) {
    const auto& concated_sampling_params =
        round > 0 ? inputs.concated_decoder_sampling_params
                  : inputs.concated_sampling_params;

    // Ensure generated_tokens_cpu rows match current sampling rows (first
    // round). Prefill uses batch rows; later rounds use beam-expanded rows.
    if (use_constrained_decoding && generated_tokens_cpu.empty()) {
      int64_t sample_rows = (round == 0) ? batch : num_seq;
      generated_tokens_cpu.resize(static_cast<size_t>(sample_rows));
    }

    // Wait for previous round's generated_tokens_cpu update to complete
    if (update_future.has_value()) {
      std::move(update_future.value()).get();
      update_future.reset();
    }

    // Async: start generate_mask before model forward (CPU/NPU parallel)
    if (use_constrained_decoding) {
      folly::Promise<torch::Tensor> promise;
      mask_future = promise.getSemiFuture();
      // Capture generated_tokens_cpu by value to avoid race condition
      auto tokens_snapshot = generated_tokens_cpu;
      general_threadpool_.schedule(
          [this,
           tokens_snapshot = std::move(tokens_snapshot),
           promise = std::move(promise)]() mutable {
            auto filter_mask =
                constrained_decoding_->generate_mask(tokens_snapshot);
            promise.setValue(std::move(filter_mask));
          });
    }

    for (auto i = 0; i < input_params_micro_batches.size(); ++i) {
      auto& mip = input_params_micro_batches[i];
      mip.is_prefill = round == 0;
      if (!mip.current_round_tensor_list.empty() && round >= 0 &&
          round < static_cast<int32_t>(mip.current_round_tensor_list.size())) {
        mip.current_round_tensor = mip.current_round_tensor_list[round];
      }
    }

    torch::Tensor hidden_states;

    // ========== Rec backend: handle encoder/decoder forward ==========
    if (is_rec_backend && round == 0 && has_encoder_inputs) {
      // Prefill (round == 0): run encoder first, then decoder
      auto& rec_params = input_params_micro_batches[0].rec_params.value();

      // 1. Run encoder forward
      auto encoder_input_params = input_params_micro_batches;
      encoder_input_params[0].rec_params->is_encoder_forward = true;

      std::vector<torch::Tensor> encoder_tokens;
      std::vector<torch::Tensor> encoder_positions;

      if (rec_params.is_hybrid_mode &&
          rec_params.encoder_sparse_embedding.defined()) {
        // Hybrid mode: use sparse embedding, positions not needed
        // Convert sparse embedding to model dtype and move to device
        // (serialization converts to fp32, need to restore to model dtype)
        auto sparse_embedding =
            rec_params.encoder_sparse_embedding.to(dtype_).to(device_);
        encoder_tokens.push_back(sparse_embedding);
        // Create a placeholder positions tensor for hybrid mode
        // The model will ignore this in hybrid mode
        auto positions_placeholder = torch::zeros(
            {sparse_embedding.size(0)},
            torch::TensorOptions().dtype(torch::kInt32).device(device_));
        encoder_positions.push_back(positions_placeholder);
      } else if (rec_params.encoder_token_ids.defined()) {
        encoder_tokens.push_back(rec_params.encoder_token_ids);
        if (rec_params.encoder_positions.defined()) {
          encoder_positions.push_back(rec_params.encoder_positions);
        }
      }

      // Run encoder
      if (!encoder_tokens.empty() && !encoder_positions.empty()) {
        hidden_states = model_executor_->forward(encoder_tokens,
                                                 encoder_positions,
                                                 kv_caches_,
                                                 encoder_input_params);
      }

      // 2. Run decoder forward
      encoder_input_params[0].rec_params->is_encoder_forward = false;
      hidden_states = model_executor_->forward(flatten_tokens_micro_batches,
                                               flatten_positions_micro_batches,
                                               kv_caches_,
                                               encoder_input_params);
    } else {
      // Standard LLM forward or rec decode rounds (round > 0)

      // For rec model decode rounds: ensure positions are available
      if (is_rec_backend && round > 0 &&
          flatten_positions_micro_batches.empty()) {
        int32_t num_seqs = batch * beam_width_init;
        std::vector<int32_t> positions_vec(num_seqs);
        for (int32_t s = 0; s < num_seqs; ++s) {
          // rec model decoder: position = prompt_len (1 for BOS) + round
          positions_vec[s] = 1 + round;
        }
        auto positions_tensor = torch::tensor(
            positions_vec,
            torch::TensorOptions().dtype(torch::kInt32).device(device_));
        flatten_positions_micro_batches.push_back(positions_tensor);
      }

      hidden_states = model_executor_->forward(flatten_tokens_micro_batches,
                                               flatten_positions_micro_batches,
                                               kv_caches_,
                                               input_params_micro_batches);
    }

    if (!hidden_states.defined()) {
      return std::nullopt;
    }

    torch::Tensor logits;

    if (concated_sampling_params.selected_token_idxes.defined()) {
      logits = model_->logits(hidden_states,
                              concated_sampling_params.selected_token_idxes);
    }
    if (concated_sampling_params.selected_token_idxes.defined()) {
      // Create a mutable copy of sampling params for filter mask
      SamplingParameters sampling_params_with_mask = concated_sampling_params;

      // Wait for async filter mask (started before model forward)
      if (use_constrained_decoding && mask_future.has_value()) {
        torch::Tensor filter_mask = std::move(mask_future.value()).get();
        mask_future.reset();
        if (filter_mask.defined()) {
          // Move filter_mask to device and convert to model dtype
          filter_mask = filter_mask.to(dtype_).to(device_);

          // Sanity check mask rows vs expected rows (round0=batch,
          // others=beam-expanded)
          int64_t expected_rows = (round == 0) ? batch : num_seq;
          if (filter_mask.size(0) !=
              expected_rows) {  // allow single-row broadcast
            LOG(ERROR) << "filter_mask rows (" << filter_mask.size(0)
                       << ") mismatch expected rows (" << expected_rows
                       << ") in round " << round
                       << "; this may broadcast unexpectedly.";
          }

          sampling_params_with_mask.filter_mask = filter_mask;
        }
      }

      auto sample_output = sampler_->forward(logits, sampling_params_with_mask);
      torch::Tensor top_tokens;
      torch::Tensor top_logprobs;
      int32_t beam_width = inputs.micro_inputs[0].beam_width;

      if (round == 0) {
        top_tokens =
            sample_output.top_tokens.to(torch::kInt32).reshape({-1, 1});
        top_logprobs = sample_output.top_logprobs.reshape({-1, 1});
      } else {
        top_tokens = sample_output.top_tokens.to(torch::kInt32)
                         .reshape({-1, beam_width});
        top_logprobs = sample_output.top_logprobs.reshape({-1, beam_width});
      }
      {
        LLM_MSTX_RANGE();
        xllm_ops::beam_search(acc_logprob,
                              top_tokens,
                              top_logprobs,
                              sequence_group,
                              round,
                              out_token_ids,
                              out_token_index,
                              acc_logprob,
                              out_beam_count_prefix_sums,
                              sequence_group);
      }

      // Async update generated_tokens_cpu for constrained decoding
      // This runs in parallel with next round's preparation, if not last round
      if (use_constrained_decoding && round < total_rounds - 1) {
        folly::Promise<folly::Unit> update_promise;
        update_future = update_promise.getSemiFuture();

        if (round == 0) {
          // Round 0: directly append top_tokens to each sequence
          // Copy tensor to CPU first, then update async
          auto tokens_cpu =
              sample_output.top_tokens.to(torch::kCPU).to(torch::kInt32);
          general_threadpool_.schedule(
              [&generated_tokens_cpu,
               tokens_cpu = std::move(tokens_cpu),
               num_seq,
               update_promise = std::move(update_promise)]() mutable {
                auto tokens_accessor = tokens_cpu.accessor<int32_t, 1>();
                for (int64_t i = 0; i < num_seq; ++i) {
                  generated_tokens_cpu[i].push_back(tokens_accessor[i]);
                }
                update_promise.setValue(folly::Unit{});
              });
        } else {
          // Round > 0: reorder based on beam selection and append new tokens
          auto token_ids_cpu = out_token_ids.to(torch::kCPU).to(torch::kInt32);
          auto token_index_cpu =
              out_token_index.to(torch::kCPU).to(torch::kInt32);
          general_threadpool_.schedule(
              [&generated_tokens_cpu,
               token_ids_cpu = std::move(token_ids_cpu),
               token_index_cpu = std::move(token_index_cpu),
               num_seq,
               update_promise = std::move(update_promise)]() mutable {
                auto ids_accessor = token_ids_cpu.accessor<int32_t, 2>();
                auto idx_accessor = token_index_cpu.accessor<int32_t, 2>();

                // Create new generated_tokens based on beam selection
                std::vector<std::vector<int32_t>> new_generated_tokens(num_seq);
                for (int64_t i = 0; i < num_seq; ++i) {
                  int32_t src_idx = idx_accessor[i][0];  // beam source index
                  new_generated_tokens[i] = generated_tokens_cpu[src_idx];
                  new_generated_tokens[i].push_back(ids_accessor[i][0]);
                }
                generated_tokens_cpu = std::move(new_generated_tokens);
                update_promise.setValue(folly::Unit{});
              });
        }
      } else if (use_constrained_decoding && round == total_rounds - 1) {
        // Last round: update synchronously (no next round to wait for)
        if (round == 0) {
          auto tokens_cpu =
              sample_output.top_tokens.to(torch::kCPU).to(torch::kInt32);
          auto tokens_accessor = tokens_cpu.accessor<int32_t, 1>();
          for (int64_t i = 0; i < num_seq; ++i) {
            generated_tokens_cpu[i].push_back(tokens_accessor[i]);
          }
        } else {
          auto token_ids_cpu = out_token_ids.to(torch::kCPU).to(torch::kInt32);
          auto token_index_cpu =
              out_token_index.to(torch::kCPU).to(torch::kInt32);
          auto ids_accessor = token_ids_cpu.accessor<int32_t, 2>();
          auto idx_accessor = token_index_cpu.accessor<int32_t, 2>();
          std::vector<std::vector<int32_t>> new_generated_tokens(num_seq);
          for (int64_t i = 0; i < num_seq; ++i) {
            int32_t src_idx = idx_accessor[i][0];
            new_generated_tokens[i] = generated_tokens_cpu[src_idx];
            new_generated_tokens[i].push_back(ids_accessor[i][0]);
          }
          generated_tokens_cpu = std::move(new_generated_tokens);
        }
      }

      // keep group offset contiguous across rounds (already in out_* tensors)
      // update next round tokens.
      if (round == 0) {
        flatten_tokens_micro_batches[0] =
            sample_output.top_tokens.to(torch::kInt32).reshape({-1});
      } else {
        flatten_tokens_micro_batches[0] = out_token_ids.clone().reshape({-1});
      }

      // update next round positions.
      flatten_positions_micro_batches.clear();
      for (auto i = 0; i < inputs.micro_inputs.size(); ++i) {
        auto& mip = input_params_micro_batches[i];
        if (!mip.decode_positions_tensor_list.empty() && round >= 0 &&
            round <
                static_cast<int32_t>(mip.decode_positions_tensor_list.size())) {
          flatten_positions_micro_batches.push_back(
              mip.decode_positions_tensor_list[round]);
        }
      }
      // For rec model: if decode_positions_tensor_list is empty, generate
      // positions based on prompt_len + round + 1 (next round's position)
      if (is_rec_backend && flatten_positions_micro_batches.empty() &&
          round < total_rounds - 1) {
        // Get prompt length from decode_positions_vec (stored prompt_len
        // values)
        int32_t next_round = round + 1;
        int32_t num_seqs = batch * beam_width_init;
        // Position for decode round is prompt_len + round
        // For rec model, we use a simple incrementing position
        std::vector<int32_t> positions_vec(num_seqs);
        for (int32_t s = 0; s < num_seqs; ++s) {
          // Use the stored prompt length + current decode step
          int32_t prompt_len = 1;  // rec model decoder starts with BOS token
          positions_vec[s] = prompt_len + next_round;
        }
        auto positions_tensor = torch::tensor(
            positions_vec,
            torch::TensorOptions().dtype(torch::kInt32).device(device_));
        flatten_positions_micro_batches.push_back(positions_tensor);
      }
      // update output at the last round.
      if (round == total_rounds - 1) {
        output.logits = logits;
        output.sample_output = sample_output;
        output.do_sample = concated_sampling_params.do_sample;
        output.logprobs = concated_sampling_params.logprobs;
        output.max_top_logprobs = concated_sampling_params.max_top_logprobs;
        output.beam_search_output.src_seq_idxes = out_token_index.reshape({-1});
        output.beam_search_output.out_tokens = out_token_ids.reshape({-1});
        output.beam_search_output.out_logprobs = acc_logprob.reshape({-1});
        output.beam_search_output.group_offset =
            out_beam_count_prefix_sums.reshape({-1});
        output.beam_sequence_group = sequence_group;
      }

#if defined(USE_NPU)
      if (beam_width > 1 && round > 0 && round < total_rounds - 1) {
        LLM_MSTX_RANGE();
        xllm_ops::cache_select(out_token_index,
                               unshared_k_cache,
                               unshared_v_cache,
                               inputs.concated_block_tables,
                               out_beam_count_prefix_sums,
                               round,
                               beam_width,
                               layer_num);
      }
#endif
    }
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_ &&
      !options_.enable_speculative_decode()) {
    auto ret = device_.synchronize_default_stream();
    if (options_.instance_role() == InstanceRole::PREFILL &&
        options_.kv_cache_transfer_mode() == "PUSH" &&
        !inputs.micro_inputs[0].transfer_kv_infos.empty()) {
      auto results =
          folly::collectAll(futures).within(std::chrono::seconds(60)).get();
      for (const auto& result : results) {
        if (!result.value()) {
          return std::nullopt;
        }
      }
    }
    if (FLAGS_enable_eplb) {
      return output;
    }
    return std::nullopt;
  }

  auto ret = device_.synchronize_default_stream();
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      device_.index());
  return output;
}

}  // namespace xllm
