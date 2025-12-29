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

#include <c10/core/StreamGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <optional>
#include <vector>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "framework/model/model_input_params.h"
#include "framework/model_loader.h"
#include "models/model_registry.h"
#include "util/env_var.h"
#include "util/timer.h"

namespace xllm {

RecWorkerImpl::LlmRecWorkPipeline::LlmRecWorkPipeline(RecWorkerImpl& worker)
    : worker_(worker) {}

ForwardInput RecWorkerImpl::LlmRecWorkPipeline::prepare_inputs(Batch& batch) {
  return worker_.WorkerImpl::prepare_inputs(batch);
}

void RecWorkerImpl::LlmRecWorkPipeline::prepare_work_before_execute(
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

    layer::WordEmbedding word_embedding = worker_.get_word_embedding();
#if defined(USE_NPU)
    torch::Tensor input_tokens_embedding =
        word_embedding(input_tokens_tensor, 0);
#else
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

std::optional<ForwardOutput> RecWorkerImpl::LlmRecWorkPipeline::step(
    const ForwardInput& input) {
  return worker_.LLMWorkerImpl::step(input);
}

RecWorkerImpl::OneRecWorkPipeline::OneRecWorkPipeline(RecWorkerImpl& worker)
    : worker_(worker) {}

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

RecWorkerImpl::~RecWorkerImpl() {
  if (!concurrent_llmrec_enabled_) {
    return;
  }

  model_.release();
  model_executor_.release();
}

bool RecWorkerImpl::init_model(ModelContext& context) {
  const auto& model_type = context.get_model_args().model_type();
  rec_model_kind_ = get_rec_model_kind(model_type);
  CHECK(rec_model_kind_ != RecModelKind::kNone)
      << "Unsupported rec model_type: " << model_type;

  max_concurrency_ = std::max<uint32_t>(FLAGS_llm_worker_max_concurrency, 1);
  concurrent_llmrec_enabled_ =
      rec_model_kind_ == RecModelKind::kLlmRec && max_concurrency_ > 1;

  if (concurrent_llmrec_enabled_) {
    CHECK(model_ == nullptr) << "Model is already initialized.";

    device_.set_device();

    step_threadpool_ = std::make_unique<ThreadPool>(
        static_cast<size_t>(max_concurrency_),
        [this]() mutable { device_.set_device(); });

    model_instances_.reserve(max_concurrency_);
    executor_instances_.reserve(max_concurrency_);
    execute_streams_.reserve(max_concurrency_);
    context_instances_.reserve(max_concurrency_);

    for (uint32_t i = 0; i < max_concurrency_; ++i) {
      auto stream = device_.get_stream_from_pool();
      execute_streams_.push_back(std::move(stream));

      auto stream_guard = execute_streams_[i]->set_stream_guard();
      ModelContext instance_context(context.get_parallel_args(),
                                    context.get_model_args(),
                                    context.get_quant_args(),
                                    context.get_tensor_options());
      context_instances_.push_back(std::move(instance_context));

      auto model_instance = create_llm_model(context_instances_[i]);
      CHECK(model_instance != nullptr)
          << "Failed to create model instance " << i;
      model_instances_.push_back(std::move(model_instance));

      auto executor =
          std::make_unique<Executor>(model_instances_[i].get(),
                                     context_instances_[i].get_model_args(),
                                     device_,
                                     options_);
      executor_instances_.push_back(std::move(executor));
    }

    model_.reset(model_instances_[0].get());
    model_executor_.reset(executor_instances_[0].get());

    if (FLAGS_enable_eplb) {
      eplb_executor_ = std::make_unique<EplbExecutor>(model_.get(), device_);
    }

    if (FLAGS_enable_beam_search_kernel) {
      beam_searcher_ = std::make_unique<BeamSearcher>();
    }
  } else {
    if (!LLMWorkerImpl::init_model(context)) {
      return false;
    }
  }

  if (rec_model_kind_ == RecModelKind::kLlmRec) {
    work_pipeline_ = std::make_unique<LlmRecWorkPipeline>(*this);
  } else if (rec_model_kind_ == RecModelKind::kOneRec) {
    work_pipeline_ = std::make_unique<OneRecWorkPipeline>(*this);
  }

  return true;
}

void RecWorkerImpl::load_model(std::unique_ptr<ModelLoader> loader) {
  if (!concurrent_llmrec_enabled_) {
    WorkerImpl::load_model(std::move(loader));
    return;
  }

  CHECK(!model_instances_.empty())
      << "Model instances are not initialized. Call init_model() first.";

  std::string model_weights_path = loader->model_weights_path();

  model_instances_[0]->load_model(std::move(loader));

  for (size_t i = 1; i < model_instances_.size(); ++i) {
    auto model_loader = ModelLoader::create(model_weights_path);
    CHECK(model_loader != nullptr)
        << "Failed to create ModelLoader for model instance " << i;
    model_instances_[i]->load_model(std::move(model_loader));
  }
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

folly::SemiFuture<std::optional<ForwardOutput>> RecWorkerImpl::step_async(
    const ForwardInput& inputs) {
  if (!concurrent_llmrec_enabled_) {
    return WorkerImpl::step_async(inputs);
  }

  CHECK(step_threadpool_ != nullptr)
      << "step_threadpool_ is not initialized. Call init_model() first.";

  ForwardInput input_on_device;
  prepare_work_before_execute(inputs, input_on_device);

  folly::Promise<std::optional<ForwardOutput>> promise;
  auto future = promise.getSemiFuture();

  step_threadpool_->schedule([this,
                              input = std::move(input_on_device),
                              promise = std::move(promise)]() mutable {
    if (hierarchy_kv_cache_transfer_ != nullptr) {
      hierarchy_kv_cache_transfer_->set_layer_synchronizer(input.input_params);
    }

    const auto output = this->step(input);

    if (!enable_schedule_overlap()) {
      promise.setValue(output);
      return;
    }

    if (last_step_output_valid_ && !input.input_params.empty_kv_cache) {
      input = update_input_by_last_step_output(input);
    }

    const auto output_overlap = this->step(input);
    if (output_overlap.has_value()) {
      if (is_driver() || FLAGS_enable_eplb) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !is_recorded_; });
        update_last_step_output(output_overlap);
        is_recorded_ = true;
        cv_.notify_one();
      } else {
        update_last_step_output(output_overlap);
      }
    } else {
      if (is_driver() || FLAGS_enable_eplb) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !is_recorded_; });
        last_step_output_valid_ = false;
        is_recorded_ = true;
        cv_.notify_one();
      } else {
        last_step_output_valid_ = false;
      }
    }

    promise.setValue(output_overlap);
  });

  return future;
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

void RecWorkerImpl::allocate_instance_id_for_current_thread() {
  std::thread::id current_thread_id = std::this_thread::get_id();

  std::lock_guard<std::mutex> lock(allocation_mutex_);

  auto it = thread_id_to_instance_id_.find(current_thread_id);
  if (it != thread_id_to_instance_id_.end()) {
    return;
  }

  const size_t kInvalidInstanceId = static_cast<size_t>(-1);
  size_t instance_id = kInvalidInstanceId;
  for (size_t i = 0; i < static_cast<size_t>(max_concurrency_); ++i) {
    if (!allocated_instance_ids_.contains(i)) {
      instance_id = i;
      break;
    }
  }

  CHECK_NE(instance_id, kInvalidInstanceId)
      << "No available instance id, all " << max_concurrency_
      << " instance ids are allocated";

  thread_id_to_instance_id_[current_thread_id] = instance_id;
  allocated_instance_ids_.insert(instance_id);
}

void RecWorkerImpl::get_thread_model_instance(CausalLM*& model,
                                              Executor*& executor,
                                              Stream*& execute_stream,
                                              ModelContext*& context) {
  std::thread::id current_thread_id = std::this_thread::get_id();

  auto it = thread_id_to_instance_id_.find(current_thread_id);
  if (it == thread_id_to_instance_id_.end()) {
    allocate_instance_id_for_current_thread();
    it = thread_id_to_instance_id_.find(current_thread_id);
  }

  CHECK(it != thread_id_to_instance_id_.end())
      << "Failed to find instance id for thread " << current_thread_id;
  const size_t instance_id = it->second;

  CHECK_LT(instance_id, model_instances_.size())
      << "Thread model index " << instance_id
      << " exceeds model instances size " << model_instances_.size();

  model = model_instances_[instance_id].get();
  executor = executor_instances_[instance_id].get();
  execute_stream = execute_streams_[instance_id].get();
  context = &context_instances_[instance_id];
}

void RecWorkerImpl::update_last_step_output(
    const std::optional<ForwardOutput>& output) {
  if (!output.has_value()) {
    last_step_output_valid_ = false;
    return;
  }

  if (output.value().sample_output.next_tokens.defined()) {
    last_step_output_ = output.value();
    last_step_output_valid_ = true;
    return;
  }

  if (FLAGS_enable_eplb) {
    last_step_output_ = output.value();
  }
  last_step_output_valid_ = false;
}

std::optional<ForwardOutput> RecWorkerImpl::step(const ForwardInput& input) {
  CHECK(work_pipeline_ != nullptr) << "RecWorkerImpl is not initialized.";
  if (!concurrent_llmrec_enabled_ || rec_model_kind_ != RecModelKind::kLlmRec) {
    return work_pipeline_->step(input);
  }

  Timer timer;
  auto& sampling_params = input.sampling_params;

  CausalLM* model = nullptr;
  Executor* executor = nullptr;
  Stream* execute_stream = nullptr;
  ModelContext* context = nullptr;
  get_thread_model_instance(model, executor, execute_stream, context);

  c10::StreamGuard stream_guard = execute_stream->set_stream_guard();

  std::vector<folly::SemiFuture<bool>> futures;

  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
#if defined(USE_NPU)
    std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<NPULayerSynchronizerImpl>(
            context->get_model_args().n_layers());
    const_cast<ModelInputParams*>(&(input.input_params))->layer_synchronizer =
        layer_synchronizer;

    futures.emplace_back(
        kv_cache_transfer_->push_kv_blocks_async(input.transfer_kv_infos,
                                                 context->get_parallel_args(),
                                                 layer_synchronizer,
                                                 is_spec_draft_));
#endif
  }

  if (FLAGS_enable_eplb) {
    eplb_executor_->eplb_execute(input.eplb_info);
  }

  auto hidden_states = executor->forward(
      input.token_ids, input.positions, kv_caches_, input.input_params);
  if (!hidden_states.defined()) {
    return std::nullopt;
  }

  torch::Tensor logits;
  if (sampling_params.selected_token_idxes.defined()) {
    logits = model->logits(hidden_states, sampling_params.selected_token_idxes);
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
    execute_stream->synchronize();

    if (options_.kv_cache_transfer_mode() == "PUSH" &&
        !input.transfer_kv_infos.empty()) {
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

  SampleOutput sample_output;
  if (sampling_params.selected_token_idxes.defined()) {
    sample_output = sampler_->forward(logits, sampling_params);
    output.logits = logits;

    BeamSearchOutput beam_search_output;
    if (sampling_params.use_beam_search && beam_searcher_ != nullptr &&
        input.acc_logprob.defined() && input.acc_logprob.numel() > 0) {
      beam_search_output = beam_searcher_->forward(input.acc_logprob,
                                                   sample_output.top_tokens,
                                                   sample_output.top_logprobs);
    }

    output.sample_output = sample_output;
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
    output.beam_search_output = beam_search_output;
  }

  if (options_.enable_speculative_decode()) {
    if (!input.input_params.batch_forward_type.is_decode() && !is_spec_draft_) {
      output.sample_output.embeddings = hidden_states;
    } else if (sampling_params.selected_token_idxes.defined()) {
      auto embeddings = hidden_states.index_select(
          /*dim=*/0, sampling_params.selected_token_idxes);
      output.sample_output.embeddings = embeddings;
    }
  }

  execute_stream->synchronize();

  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
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

}  // namespace xllm
