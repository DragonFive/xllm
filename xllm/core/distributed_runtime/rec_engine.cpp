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

#include "rec_engine.h"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "framework/model/model_args.h"
#include "framework/model_loader.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/request/rec_type.h"
#include "master.h"  // For MasterStatus::WAKEUP constant
#include "util/env_var.h"
#include "util/net.h"
#include "util/pretty_print.h"
#include "util/rec_model_utils.h"
#include "util/scope_guard.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
namespace {

constexpr int64_t kMinimalOneRecMetadataKVBlocks = 2;
constexpr const char* kRecMultiRoundTpPerPipelineAtbCommEnv =
    "XLLM_REC_MULTIROUND_TP_PER_PIPELINE_ATB_COMM";
constexpr const char* kRecMultiRoundTpSingleModelPipelineEnv =
    "XLLM_REC_MULTIROUND_TP_SINGLE_MODEL_PIPELINE";
constexpr const char* kRecMultiRoundTpSerializeShapeFirstUseEnv =
    "XLLM_REC_MULTIROUND_TP_SERIALIZE_SHAPE_FIRST_USE";
constexpr const char* kRecMultiRoundEngineTimingEnv =
    "XLLM_DEBUG_REC_MULTIROUND_ENGINE_TIMING";

}  // namespace

namespace {

bool enable_rec_multiround_tp_per_pipeline_atb_comm() {
  static const bool enabled =
      util::get_bool_env(kRecMultiRoundTpPerPipelineAtbCommEnv, false);
  return enabled;
}

bool enable_rec_multiround_tp_single_model_pipeline() {
  static const bool enabled =
      util::get_bool_env(kRecMultiRoundTpSingleModelPipelineEnv, false);
  return enabled;
}

bool serialize_rec_multiround_tp_shape_first_use() {
  static const bool enabled =
      util::get_bool_env(kRecMultiRoundTpSerializeShapeFirstUseEnv, true);
  return enabled;
}

bool enable_rec_multiround_engine_timing() {
  static const bool enabled =
      util::get_bool_env(kRecMultiRoundEngineTimingEnv, false);
  return enabled;
}

const char* rec_model_kind_name(RecModelKind kind) {
  switch (kind) {
    case RecModelKind::kOneRec:
      return "OneRec";
    case RecModelKind::kLlmRec:
      return "LlmRec";
    case RecModelKind::kNone:
      return "None";
  }
  return "Unknown";
}

const char* rec_pipeline_type_name(RecPipelineType type) {
  switch (type) {
    case RecPipelineType::kLlmRecDefault:
      return "LlmRecDefault";
    case RecPipelineType::kLlmRecWithMmData:
      return "LlmRecWithMmData";
    case RecPipelineType::kOneRecDefault:
      return "OneRecDefault";
    case RecPipelineType::kLlmRecMultiRoundPipeline:
      return "LlmRecMultiRoundPipeline";
    case RecPipelineType::kOneRecXAttentionPipeline:
      return "OneRecXAttentionPipeline";
  }
  return "Unknown";
}

std::string device_list_to_string(const std::vector<torch::Device>& devices) {
  std::string result;
  for (size_t i = 0; i < devices.size(); ++i) {
    if (i > 0) {
      result += ",";
    }
    result += devices[i].str();
  }
  return result;
}

void validate_local_rec_tp_options(const runtime::Options& options) {
  const int32_t local_world_size =
      static_cast<int32_t>(options.devices().size());
  CHECK_EQ(options.nnodes(), 1)
      << "backend=rec local TP currently supports nnodes=1 only.";
  CHECK_EQ(options.dp_size(), 1)
      << "backend=rec local TP currently supports dp_size=1 only.";
  CHECK_EQ(options.cp_size(), 1)
      << "backend=rec local TP currently supports cp_size=1 only.";
  CHECK(options.tp_size() == 1 || options.tp_size() == local_world_size)
      << "backend=rec local TP uses --devices as the local TP world. "
      << "Set --tp_size=" << local_world_size << " or adjust --devices.";
  if (options.tp_size() == 1 && local_world_size > 1) {
    LOG(WARNING) << "backend=rec local TP is enabled by --devices size "
                 << local_world_size
                 << " while --tp_size keeps the default value 1.";
  }
  CHECK(local_world_size == 1 || local_world_size == 2)
      << "backend=rec local TP MVP supports one device or TP2 only, got "
      << local_world_size << " devices.";
}

void rethrow_worker_exception(
    size_t worker_index,
    const folly::Try<std::optional<ForwardOutput>>& result) {
  if (!result.hasException()) {
    return;
  }

  try {
    static_cast<void>(result.value());
  } catch (const std::exception& e) {
    throw std::runtime_error("Worker " + std::to_string(worker_index) +
                             " failed with exception: " + e.what());
  } catch (...) {
    throw std::runtime_error("Worker " + std::to_string(worker_index) +
                             " failed with unknown exception");
  }
}

void validate_local_rec_worker_results(
    const std::vector<folly::Try<std::optional<ForwardOutput>>>& results,
    const char* pipeline_name) {
  CHECK(!results.empty()) << pipeline_name << " requires at least one worker.";
  for (size_t i = 0; i < results.size(); ++i) {
    rethrow_worker_exception(i, results[i]);
    if (i == 0) {
      CHECK(results[i].value().has_value())
          << pipeline_name
          << " driver worker failed to execute model and returned no output.";
      continue;
    }
    if (!results[i].value().has_value()) {
      if (util::get_bool_env("XLLM_DEBUG_REC_PIPELINE_CONCURRENCY", false)) {
        LOG(INFO) << pipeline_name << " non-driver worker " << i
                  << " returned no user-visible output.";
      }
      continue;
    }
    LOG(INFO) << pipeline_name << " non-driver worker " << i
              << " returned output; engine will ignore it.";
  }
}

}  // namespace

// ============================================================
// RecEngine Implementation
// ============================================================

RecEngine::RecEngine(const runtime::Options& options,
                     std::shared_ptr<DistManager> dist_manager)
    : options_(options), dist_manager_(dist_manager) {
  const auto& devices = options_.devices();
  CHECK_GT(devices.size(), 0) << "At least one device is required";

  CHECK(!devices[0].is_cpu()) << "CPU device is not supported";
  const auto device_type = devices[0].type();
  for (const auto device : devices) {
    CHECK_EQ(device.type(), device_type)
        << "All devices should be the same type";
  }
}

bool RecEngine::init() {
  if (!init_model()) {
    LOG(ERROR) << "Failed to init model from: " << options_.model_path();
    return false;
  }

  auto kv_cache_cap = estimate_kv_cache_capacity();

  if (!allocate_kv_cache(kv_cache_cap)) {
    LOG(ERROR) << "Failed to allocate kv cache";
    return false;
  }

  return true;
}

bool RecEngine::init_model() {
  const std::string& model_path = options_.model_path();
  auto model_loader = ModelLoader::create(model_path);

  tokenizer_ = model_loader->tokenizer();
  CHECK(tokenizer_ != nullptr);

  args_ = model_loader->model_args();
  quant_args_ = model_loader->quant_args();
  tokenizer_args_ = model_loader->tokenizer_args();
  // Determine rec model kind and create pipeline via factory
  rec_model_kind_ = get_rec_model_kind(args_.model_type());
  CHECK(rec_model_kind_ != RecModelKind::kNone)
      << "Unsupported rec model_type: " << args_.model_type();
  if (rec_model_kind_ == RecModelKind::kOneRec &&
      !onerec_batch_input_builder_cache_) {
    onerec_batch_input_builder_cache_ =
        std::make_unique<OneRecBatchInputBuilderCache>();
  }
  auto pipeline_type = get_rec_pipeline_type(rec_model_kind_);
  validate_local_rec_tp_options(options_);
  LOG(INFO) << "REC local execution config, model_type=" << args_.model_type()
            << ", rec_model_kind=" << rec_model_kind_name(rec_model_kind_)
            << ", pipeline_type=" << rec_pipeline_type_name(pipeline_type)
            << ", devices=" << device_list_to_string(options_.devices())
            << ", local_tp_size=" << options_.devices().size()
            << ", tp_size=" << options_.tp_size()
            << ", dp_size=" << options_.dp_size()
            << ", cp_size=" << options_.cp_size()
            << ", nnodes=" << options_.nnodes()
            << ", rec_worker_max_concurrency="
            << options_.rec_worker_max_concurrency()
            << ", beam_width=" << options_.beam_width()
            << ", enable_graph=" << options_.enable_graph();
  pipeline_ = create_pipeline(pipeline_type, *this);
  // LlmRec-specific initialization
  if (rec_model_kind_ == RecModelKind::kLlmRec) {
#if defined(USE_NPU)
    FLAGS_enable_atb_comm_multiprocess =
        options_.enable_offline_inference() || (options_.nnodes() > 1);
#endif

    auto master_node_addr = options_.master_node_addr().value_or("");
    CHECK(!master_node_addr.empty())
        << "REC(kLlmRec) need to set master node addr, "
           "Please set --master_node_addr.";
  }
  // Pipeline-specific setup
  pipeline_->setup_workers();
  pipeline_->process_group_test();

  if (!threadpool_) {
    threadpool_ = std::make_unique<ThreadPool>(
        16, true, __FILE__, __LINE__, "rec_engine_pool");
  }
  // Compute KV cache config (shared logic)
  const int32_t world_size = static_cast<int32_t>(options_.devices().size());
  const int64_t n_heads = args_.n_heads();
  const int64_t n_kv_heads = args_.n_kv_heads().value_or(n_heads);
  n_local_kv_heads_ = std::max<int64_t>(1, n_kv_heads / world_size);
  head_dim_ = args_.head_dim();
  dtype_ = xllm::util::parse_dtype(args_.dtype(), options_.devices()[0]);

  LOG(INFO) << "Block info, block_size: " << options_.block_size()
            << ", n_local_kv_heads: " << n_local_kv_heads_
            << ", head_dim: " << head_dim_ << ", n_layers: " << args_.n_layers()
            << ", dtype: " << dtype_;
  LOG(INFO) << "Initializing model with " << args_;
  LOG(INFO) << "Initializing model with quant args: " << quant_args_;
  LOG(INFO) << "Initializing model with tokenizer args: " << tokenizer_args_;

  // Pipeline-specific model initialization
  return pipeline_->init_model_workers(model_path);
}

Engine::KVCacheCapacity RecEngine::estimate_kv_cache_capacity() {
  const int64_t max_cache_size = options_.max_cache_size();
  const double max_memory_utilization = options_.max_memory_utilization();

  // compute kv cache slot size
  const int64_t dtype_size = torch::scalarTypeToTypeMeta(dtype_).itemsize();
  const int64_t slot_size = 2 * dtype_size * head_dim_ * n_local_kv_heads_;

  KVCacheCapacity kv_cache_cap;
  kv_cache_cap.slot_size = slot_size;
  kv_cache_cap.n_layers = args_.n_layers();

  const int32_t block_size = options_.block_size();
  const int64_t block_size_in_bytes = block_size * slot_size;
  const int64_t cache_block_size_in_bytes =
      args_.n_layers() * block_size_in_bytes;
  CHECK_GT(cache_block_size_in_bytes, 0)
      << "cache block size must be positive.";

  const int64_t minimal_kv_cache_blocks = pipeline_->minimal_kv_cache_blocks();
  if (minimal_kv_cache_blocks > 0) {
    int64_t n_blocks = minimal_kv_cache_blocks;
    if (max_cache_size > 0) {
      const int64_t max_cache_blocks =
          max_cache_size / cache_block_size_in_bytes;
      CHECK_GE(max_cache_blocks, minimal_kv_cache_blocks)
          << "max_cache_size is too small for OneRec metadata kv cache. It "
             "must fit the minimal metadata blocks, "
             "max_cache_size="
          << readable_size(max_cache_size)
          << ", block_bytes=" << readable_size(cache_block_size_in_bytes);
      n_blocks = std::min(n_blocks, max_cache_blocks);
    }
    kv_cache_cap.n_blocks = n_blocks;
    kv_cache_cap.cache_size_in_bytes = n_blocks * cache_block_size_in_bytes;
    LOG(INFO) << "OneRec uses minimal metadata kv cache, blocks: "
              << kv_cache_cap.n_blocks
              << ", bytes: " << readable_size(kv_cache_cap.cache_size_in_bytes)
              << ", max_memory_utilization: " << max_memory_utilization
              << ", max_cache_size: " << readable_size(max_cache_size);
    return kv_cache_cap;
  }

  int64_t cache_size_in_bytes = pipeline_->estimate_min_available_memory();

  // apply memory cap from config
  if (max_memory_utilization < 1.0 || max_cache_size > 0) {
    // Re-estimate with caps applied (pipeline returns raw available memory)
    // The caps are applied in estimate_min_available_memory
  }

  kv_cache_cap.cache_size_in_bytes = std::max(cache_size_in_bytes, int64_t(0));
  CHECK_GT(kv_cache_cap.cache_size_in_bytes, 0)
      << "Available kv cache size must be greater than 0";

  kv_cache_cap.n_blocks = kv_cache_cap.cache_size_in_bytes /
                          (args_.n_layers() * block_size_in_bytes);
  CHECK_GT(kv_cache_cap.n_blocks, 0) << "no n_blocks for kv cache";

  return kv_cache_cap;
}

bool RecEngine::allocate_kv_cache(const Engine::KVCacheCapacity& kv_cache_cap) {
  LOG(INFO) << "kv cache capacity: "
            << "bytes: " << kv_cache_cap.cache_size_in_bytes
            << ", blocks: " << kv_cache_cap.n_blocks
            << ", slot_size: " << kv_cache_cap.slot_size;

  const int32_t block_size = options_.block_size();

  // init kv cache for each worker
  std::vector<std::vector<int64_t>> kv_cache_shape;
  kv_cache_shape.reserve(2);
  kv_cache_shape.emplace_back(std::vector<int64_t>{
      kv_cache_cap.n_blocks, block_size, n_local_kv_heads_, head_dim_});
  kv_cache_shape.emplace_back(std::vector<int64_t>{
      kv_cache_cap.n_blocks, block_size, n_local_kv_heads_, head_dim_});
#if defined(USE_MLU)
  for (auto& shape : kv_cache_shape) {
    std::swap(shape[1], shape[2]);
  }
#endif

  LOG(INFO) << "Initializing k cache with shape: [" << kv_cache_shape[0] << "]";
  LOG(INFO) << "Initializing v cache with shape: [" << kv_cache_shape[1] << "]";

  // initialize block manager
  BlockManagerPool::Options options;
  options.num_blocks(kv_cache_cap.n_blocks)
      .host_num_blocks(0)
      .block_size(block_size)
      .enable_prefix_cache(options_.enable_prefix_cache())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_cache_upload(options_.enable_cache_upload());
  kv_cache_manager_ = std::make_unique<BlockManagerPool>(options, dp_size_);

  return pipeline_->allocate_kv_cache(kv_cache_shape);
}

ForwardOutput RecEngine::step(std::vector<Batch>& batches) {
  return pipeline_->step(batches);
}

void RecEngine::update_last_step_result(std::vector<Batch>& batch) {
  UNUSED_PARAMETER(batch);
}

std::vector<int64_t> RecEngine::get_active_activation_memory() const {
  return pipeline_->get_active_activation_memory();
}

// ============================================================
// LlmRecEnginePipeline Implementation
// ============================================================

RecEngine::LlmRecEnginePipeline::LlmRecEnginePipeline(RecEngine& engine)
    : RecEnginePipeline(engine) {}

void RecEngine::LlmRecEnginePipeline::setup_workers() {
  if (!engine_.dist_manager_) {
    engine_.dist_manager_ = std::make_shared<DistManager>(engine_.options_);
  }
  engine_.worker_clients_ = engine_.dist_manager_->get_worker_clients();
  engine_.dp_size_ = engine_.options_.dp_size();
  engine_.worker_clients_num_ = engine_.worker_clients_.size();
  engine_.dp_local_tp_size_ = engine_.worker_clients_num_ / engine_.dp_size_;
}

void RecEngine::LlmRecEnginePipeline::process_group_test() {
#if !defined(USE_NPU)
  if (engine_.worker_clients_num_ > 1) {
    std::vector<folly::SemiFuture<folly::Unit>> futures;
    futures.reserve(engine_.worker_clients_num_);
    for (auto& worker : engine_.worker_clients_) {
      futures.emplace_back(worker->process_group_test_async());
    }
    const int32_t timeout_seconds =
        util::get_process_group_test_timeout_seconds();
    folly::collectAll(futures)
        .within(std::chrono::seconds(timeout_seconds))
        .get();
  }
#endif
}

bool RecEngine::LlmRecEnginePipeline::init_model_workers(
    const std::string& model_path) {
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(engine_.worker_clients_num_);
  for (auto& worker : engine_.worker_clients_) {
    futures.emplace_back(worker->init_model_async(
        model_path, FLAGS_random_seed, MasterStatus::WAKEUP));
  }
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }
  return true;
}

int64_t RecEngine::LlmRecEnginePipeline::estimate_min_available_memory() {
  const int64_t max_cache_size = engine_.options_.max_cache_size();
  const double max_memory_utilization =
      engine_.options_.max_memory_utilization();

  std::vector<folly::SemiFuture<std::tuple<int64_t, int64_t>>> futures;
  futures.reserve(engine_.worker_clients_.size());
  for (auto& worker : engine_.worker_clients_) {
    futures.emplace_back(worker->estimate_kv_cache_capacity_async());
  }

  int64_t cache_size_in_bytes = std::numeric_limits<int64_t>::max();
  auto results = folly::collectAll(futures).get();
  for (size_t i = 0; i < results.size(); ++i) {
    if (!results[i].hasValue()) {
      LOG(ERROR) << "Failed to profile memory usage for worker: " << i;
      continue;
    }
    auto [available_memory, total_memory] = results[i].value();
    LOG(INFO) << "worker #" << i
              << ": available memory: " << readable_size(available_memory)
              << ", total memory: " << readable_size(total_memory)
              << ". Using max_memory_utilization: " << max_memory_utilization
              << ", max_cache_size: " << readable_size(max_cache_size);
    if (max_memory_utilization < 1.0) {
      const int64_t buffer_memory =
          total_memory * (1.0 - max_memory_utilization);
      available_memory -= buffer_memory;
    }
    if (max_cache_size > 0) {
      available_memory = std::min(available_memory, max_cache_size);
    }
    cache_size_in_bytes = std::min(cache_size_in_bytes, available_memory);
  }
  return cache_size_in_bytes;
}

bool RecEngine::LlmRecEnginePipeline::allocate_kv_cache(
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(engine_.worker_clients_.size());
  for (auto& worker : engine_.worker_clients_) {
    futures.emplace_back(worker->allocate_kv_cache_async(kv_cache_shape));
  }
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }
  return true;
}

size_t RecEngine::LlmRecEnginePipeline::num_workers() const {
  if (engine_.dp_size_ > 1) {
    return engine_.dp_local_tp_size_;
  }
  return engine_.worker_clients_.size();
}

std::vector<RawForwardInput> RecEngine::LlmRecEnginePipeline::prepare_inputs(
    std::vector<Batch>& batch) {
  std::vector<RawForwardInput> batched_inputs;
  batched_inputs.reserve(engine_.dp_size_);

  // some dp related variables
  std::vector<int32_t> dp_global_token_nums(engine_.dp_size_);
  std::vector<int32_t> dp_is_decode(engine_.dp_size_, 0);
  // when enable dp, we need to check the forward type of each batch
  // and set the empty forward type of each batch to the same value as the first
  // batch
  BatchForwardType batch_forward_type;

  for (int32_t dp_rank = 0; dp_rank < engine_.dp_size_; ++dp_rank) {
    // kLlmRec needs refresh_forward_type for correct dp_is_decode
    batch[dp_rank].refresh_forward_type();

    batched_inputs.emplace_back(std::move(batch[dp_rank].prepare_forward_input(
        engine_.args_, engine_.threadpool_.get())));
    dp_global_token_nums[dp_rank] =
        batched_inputs[dp_rank].flatten_tokens_vec.size();
    if (batch_forward_type.is_empty() &&
        !batched_inputs[dp_rank].batch_forward_type.is_empty()) {
      batch_forward_type = batched_inputs[dp_rank].batch_forward_type;
    }
    dp_is_decode[dp_rank] = batch_forward_type.is_decode() &&
                            batched_inputs[dp_rank].q_max_seq_len == 1;
  }

  for (int32_t dp_rank = 0; dp_rank < engine_.dp_size_; ++dp_rank) {
    batched_inputs[dp_rank].dp_global_token_nums = dp_global_token_nums;
    batched_inputs[dp_rank].dp_is_decode = dp_is_decode;
    if (batched_inputs[dp_rank].batch_forward_type.is_empty()) {
      batched_inputs[dp_rank].batch_forward_type = batch_forward_type;
    }
  }

  return batched_inputs;
}

ForwardOutput RecEngine::LlmRecEnginePipeline::step(
    std::vector<Batch>& batches) {
  if (engine_.worker_clients_.empty()) {
    return {};
  }

  DCHECK(engine_.dp_size_ == static_cast<int32_t>(batches.size()))
      << "Split DP batch failed with dp_size as " << engine_.dp_size_
      << " and actual batch size as " << batches.size() << ".";

  auto run_one_step = [this, &batches](int step_idx) -> bool {
    Timer timer;
    auto raw_forward_inputs = prepare_inputs(batches);
    COUNTER_ADD(prepare_input_latency_microseconds,
                timer.elapsed_microseconds());

    const bool all_empty =
        std::all_of(raw_forward_inputs.begin(),
                    raw_forward_inputs.end(),
                    [](const RawForwardInput& input) {
                      return input.flatten_tokens_vec.empty();
                    });
    if (all_empty) {
      return false;
    }

    std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
    futures.reserve(engine_.worker_clients_num_);

    timer.reset();
    for (size_t worker_rank = 0; worker_rank < engine_.worker_clients_num_;
         ++worker_rank) {
      auto dp_rank = worker_rank / engine_.dp_local_tp_size_;
      futures.emplace_back(engine_.worker_clients_[worker_rank]->step_async(
          raw_forward_inputs[dp_rank]));
    }
    auto results = folly::collectAll(futures).get();

    if (step_idx == 0) {
      COUNTER_ADD(rec_first_token_latency_microseconds,
                  timer.elapsed_microseconds());
    } else if (step_idx == 1) {
      COUNTER_ADD(rec_second_token_latency_microseconds,
                  timer.elapsed_microseconds());
    } else if (step_idx == 2) {
      COUNTER_ADD(rec_third_token_latency_microseconds,
                  timer.elapsed_microseconds());
    }

    timer.reset();
    size_t dp_rank = 0;
    for (size_t worker_rank = 0; worker_rank < engine_.worker_clients_num_;
         worker_rank += engine_.dp_local_tp_size_) {
      auto result = results[worker_rank].value();
      if (!result.has_value()) {
        LOG(FATAL) << "Failed to execute model, result has no value";
      }
      if (result.value().src_seq_idxes.empty()) {
        batches[dp_rank].process_sample_output(result.value(), false);
      } else {
        batches[dp_rank].process_beam_search_output(result.value(), false);
        // Transfer src_blocks_ to blocks_ for beam search sequences
        // RecEngine doesn't have Scheduler/BlockManagerPool to trigger this
        for (size_t i = 0; i < batches[dp_rank].size(); ++i) {
          auto* seq = batches[dp_rank][i];
          if (seq->check_beam_search() &&
              !seq->kv_state().src_blocks().empty()) {
            seq->kv_state().process_beam_search(std::nullopt);
          }
        }
      }
      // Refresh sequences_ from sequence_groups_ after beam search processing.
      // This is needed because SequencesGroup::process_beam_search() replaces
      // its internal sequences_, invalidating pointers in Batch::sequences_.
      batches[dp_rank].refresh_sequences_from_groups();
      ++dp_rank;
    }
    COUNTER_ADD(rec_sampling_latency_microseconds,
                timer.elapsed_microseconds());
    return true;
  };

  // Get dynamic max steps from batch (based on max_tokens in requests)
  const size_t max_steps = get_max_steps_from_batch(batches);

  for (size_t step_idx = 0; step_idx < max_steps; ++step_idx) {
    if (!run_one_step(step_idx)) {
      break;
    }
  }

  for (auto& batch : batches) {
    batch.finish();
  }
  return {};
}

std::vector<int64_t>
RecEngine::LlmRecEnginePipeline::get_active_activation_memory() const {
  std::vector<folly::SemiFuture<int64_t>> futures;
  futures.reserve(engine_.worker_clients_.size());
  for (auto& worker : engine_.worker_clients_) {
    futures.emplace_back(worker->get_active_activation_memory_async());
  }

  auto results = folly::collectAll(futures).get();
  std::vector<int64_t> active_activation_memories;
  active_activation_memories.reserve(futures.size());
  for (auto& result : results) {
    active_activation_memories.emplace_back(result.value());
  }
  return active_activation_memories;
}

size_t RecEngine::LlmRecEnginePipeline::get_max_steps_from_batch(
    std::vector<Batch>& batches) const {
  size_t max_steps = 0;
  bool has_stopping_checker = false;
  for (auto& batch : batches) {
    // Use get_sequences() to handle both sequences_ and sequence_groups_
    // This ensures compatibility with both LlmRec and OneRec scenarios
    auto sequences = batch.get_sequences();
    for (auto* seq : sequences) {
      const auto* stopping_checker = seq->stopping_checker();
      if (stopping_checker) {
        has_stopping_checker = true;
        max_steps =
            std::max(max_steps, stopping_checker->get_max_generated_tokens());
      }
    }
  }
  // If has stopping_checker, use max_tokens from it;
  // otherwise fall back to kRecDecodeSteps for OneRec compatibility
  if (has_stopping_checker && max_steps > 0) {
    return max_steps;
  }
  return kRecDecodeSteps;
}

// ============================================================
// OneRecLocalEnginePipeline Implementation
// ============================================================

RecEngine::OneRecLocalEnginePipeline::OneRecLocalEnginePipeline(
    RecEngine& engine)
    : RecEnginePipeline(engine) {}

void RecEngine::OneRecLocalEnginePipeline::setup_workers() {
  // OneRec uses local workers, no DistManager setup needed
}

void RecEngine::OneRecLocalEnginePipeline::process_group_test() {
  if (engine_.workers_.size() > 1) {
    std::vector<folly::SemiFuture<folly::Unit>> futures;
    futures.reserve(engine_.workers_.size());
    for (auto& worker : engine_.workers_) {
      futures.emplace_back(worker->process_group_test_async());
    }
    const int32_t timeout_seconds =
        util::get_process_group_test_timeout_seconds();
    folly::collectAll(futures)
        .within(std::chrono::seconds(timeout_seconds))
        .get();
  }
}

bool RecEngine::OneRecLocalEnginePipeline::init_model_workers(
    const std::string& model_path) {
  const auto& devices = engine_.options_.devices();
  const int32_t world_size = static_cast<int32_t>(devices.size());

  // OneRec local workers still expect valid TP group metadata even on a
  // single device. For world_size == 1, only rank/world_size metadata is
  // needed, so avoid creating a real communication backend or extra streams.
  // For multi-device local workers use a ProcessGroup-backed path. On NPU this
  // avoids direct HcclCommInitAll, which is not reliable for 910C sibling dies.
  if (world_size == 1) {
    engine_.process_groups_.clear();
    engine_.process_groups_.emplace_back(
        std::make_unique<ProcessGroup>(/*rank=*/0, world_size, devices[0]));
  } else {
    engine_.process_groups_ =
        parallel_state::create_local_process_groups(devices, engine_.options_);
  }

  engine_.workers_.clear();
  WorkerType worker_type = WorkerType::REC;
  for (int32_t rank = 0; rank < world_size; ++rank) {
    ProcessGroup* pg = engine_.process_groups_[rank].get();
    ParallelArgs parallel_args(rank, world_size, pg);
    parallel_args.tp_group_ = pg;
    engine_.workers_.emplace_back(std::make_unique<Worker>(
        parallel_args, devices[rank], engine_.options_, worker_type));
  }

  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(engine_.workers_.size());
  for (auto& worker : engine_.workers_) {
    futures.emplace_back(worker->init_model_async(
        model_path, FLAGS_random_seed, MasterStatus::WAKEUP));
  }
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }
  return true;
}

int64_t RecEngine::OneRecLocalEnginePipeline::estimate_min_available_memory() {
  const int64_t max_cache_size = engine_.options_.max_cache_size();
  const double max_memory_utilization =
      engine_.options_.max_memory_utilization();

  std::vector<folly::SemiFuture<std::tuple<int64_t, int64_t>>> futures;
  futures.reserve(engine_.workers_.size());
  for (auto& worker : engine_.workers_) {
    futures.emplace_back(worker->estimate_kv_cache_capacity_async());
  }

  int64_t cache_size_in_bytes = std::numeric_limits<int64_t>::max();
  auto results = folly::collectAll(futures).get();
  for (size_t i = 0; i < results.size(); ++i) {
    if (!results[i].hasValue()) {
      LOG(ERROR) << "Failed to profile memory usage for worker: " << i;
      continue;
    }
    auto [available_memory, total_memory] = results[i].value();
    LOG(INFO) << "worker #" << i
              << ": available memory: " << readable_size(available_memory)
              << ", total memory: " << readable_size(total_memory)
              << ". Using max_memory_utilization: " << max_memory_utilization
              << ", max_cache_size: " << readable_size(max_cache_size);
    if (max_memory_utilization < 1.0) {
      const int64_t buffer_memory =
          total_memory * (1.0 - max_memory_utilization);
      available_memory -= buffer_memory;
    }
    if (max_cache_size > 0) {
      available_memory = std::min(available_memory, max_cache_size);
    }
    cache_size_in_bytes = std::min(cache_size_in_bytes, available_memory);
  }
  return cache_size_in_bytes;
}

bool RecEngine::OneRecLocalEnginePipeline::allocate_kv_cache(
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(engine_.workers_.size());
  for (auto& worker : engine_.workers_) {
    futures.emplace_back(worker->allocate_kv_cache_async(kv_cache_shape));
  }
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }
  return true;
}

size_t RecEngine::OneRecLocalEnginePipeline::num_workers() const {
  return engine_.workers_.size();
}

// ============================================================
// OneRecPrefillOnlyEnginePipeline Implementation
// ============================================================

RecEngine::OneRecPrefillOnlyEnginePipeline::OneRecPrefillOnlyEnginePipeline(
    RecEngine& engine)
    : OneRecLocalEnginePipeline(engine) {}

int64_t RecEngine::OneRecPrefillOnlyEnginePipeline::minimal_kv_cache_blocks()
    const {
  return use_legacy_onerec_prefill_only_contract()
             ? kMinimalOneRecMetadataKVBlocks
             : 0;
}

ForwardOutput RecEngine::OneRecPrefillOnlyEnginePipeline::step(
    std::vector<Batch>& batches) {
  if (engine_.workers_.empty()) {
    return {};
  }
  CHECK(engine_.onerec_batch_input_builder_cache_ != nullptr)
      << "OneRec batch cache is not initialized.";
  CHECK(!batches.empty()) << "OneRec engine requires at least one batch.";

  Timer timer;
  Timer timer_total;
  // OneRec does not need refresh_forward_type
  batches[0].set_onerec_batch_input_builder_cache(
      engine_.onerec_batch_input_builder_cache_.get());
  auto forward_inputs = engine_.workers_[0]->prepare_inputs(batches[0]);
  COUNTER_ADD(prepare_input_latency_microseconds, timer.elapsed_microseconds());

  if (!forward_inputs.token_ids.defined()) {
    return {};
  }

  timer.reset();
  const auto& prefill_output = get_model_output(forward_inputs);
  COUNTER_ADD(rec_first_token_latency_microseconds,
              timer.elapsed_microseconds());

  timer.reset();
  batches[0].process_sample_output(prefill_output.sample_output, false);
  COUNTER_ADD(rec_sampling_latency_microseconds, timer.elapsed_microseconds());

  ForwardOutput decode_output;
  for (size_t i = 0; i < kRecDecodeSteps; ++i) {
    timer.reset();
    // OneRec does not need refresh_forward_type
    batches[0].set_onerec_batch_input_builder_cache(
        engine_.onerec_batch_input_builder_cache_.get());
    forward_inputs = engine_.workers_[0]->prepare_inputs(batches[0]);
    COUNTER_ADD(prepare_input_latency_microseconds,
                timer.elapsed_microseconds());

    timer.reset();
    decode_output = get_model_output(forward_inputs);
    if (i == 0) {
      COUNTER_ADD(rec_second_token_latency_microseconds,
                  timer.elapsed_microseconds());
    } else if (i == 1) {
      COUNTER_ADD(rec_third_token_latency_microseconds,
                  timer.elapsed_microseconds());
    }

    timer.reset();
    batches[0].process_sample_output(
        decode_output.sample_output,
        false,
        /*force_requested_beam_result_size=*/i + 1 == kRecDecodeSteps);
    COUNTER_ADD(rec_sampling_latency_microseconds,
                timer.elapsed_microseconds());
  }

  batches[0].finish();

  VLOG(1) << "OneRec batch size " << batches.size() << ", sequence size "
          << batches[0].size() << ", infer took "
          << timer_total.elapsed_milliseconds() << "ms.";

  return decode_output;
}

ForwardOutput RecEngine::OneRecPrefillOnlyEnginePipeline::get_model_output(
    const ForwardInput& model_inputs) {
  std::vector<folly::SemiFuture<std::optional<ForwardOutput>>> futures;
  futures.reserve(engine_.workers_.size());
  for (auto& worker : engine_.workers_) {
    futures.emplace_back(worker->step_async(model_inputs));
  }
  auto results = folly::collectAll(futures).get();

  validate_local_rec_worker_results(results, "OneRec prefill-only");

  auto forward_output = results.front().value();
  CHECK(forward_output.has_value()) << "Failed to execute model";

  auto& output = forward_output.value();
  auto& sample_output = output.sample_output;

  if (sample_output.embeddings.defined()) {
    sample_output.embeddings = safe_to(
        sample_output.embeddings,
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32),
        /*non_blocking=*/true);
  }

  if (sample_output.next_tokens.defined()) {
    sample_output.next_tokens =
        safe_to(sample_output.next_tokens, torch::kCPU, /*non_blocking=*/true);
    if (sample_output.logprobs.defined()) {
      sample_output.logprobs =
          safe_to(sample_output.logprobs, torch::kCPU, true);
    }
    if (sample_output.top_tokens.defined()) {
      sample_output.top_tokens =
          safe_to(sample_output.top_tokens, torch::kCPU, true);
    }
    if (sample_output.top_logprobs.defined()) {
      sample_output.top_logprobs =
          safe_to(sample_output.top_logprobs, torch::kCPU, true);
    }
  }
  Device(engine_.workers_[0]->device()).synchronize_default_stream();

  return output;
}

std::vector<int64_t>
RecEngine::OneRecLocalEnginePipeline::get_active_activation_memory() const {
  std::vector<folly::SemiFuture<int64_t>> futures;
  futures.reserve(engine_.workers_.size());
  for (auto& worker : engine_.workers_) {
    futures.emplace_back(worker->get_active_activation_memory_async());
  }

  auto results = folly::collectAll(futures).get();
  std::vector<int64_t> active_activation_memories;
  active_activation_memories.reserve(futures.size());
  for (auto& result : results) {
    active_activation_memories.emplace_back(result.value());
  }
  return active_activation_memories;
}

// ============================================================
// OneRecXAttentionEnginePipeline Implementation
// ============================================================

RecEngine::OneRecXAttentionEnginePipeline::OneRecXAttentionEnginePipeline(
    RecEngine& engine)
    : OneRecLocalEnginePipeline(engine) {}

int64_t RecEngine::OneRecXAttentionEnginePipeline::minimal_kv_cache_blocks()
    const {
  return kMinimalOneRecMetadataKVBlocks;
}

ForwardOutput RecEngine::OneRecXAttentionEnginePipeline::step(
    std::vector<Batch>& batches) {
  if (engine_.workers_.empty()) {
    return {};
  }
  CHECK(engine_.onerec_batch_input_builder_cache_ != nullptr)
      << "OneRec batch cache is not initialized.";
  CHECK(!batches.empty()) << "OneRec engine requires at least one batch.";

  Timer timer;
  batches[0].set_onerec_batch_input_builder_cache(
      engine_.onerec_batch_input_builder_cache_.get());
  auto forward_inputs = engine_.workers_[0]->prepare_inputs(batches[0]);
  COUNTER_ADD(prepare_input_latency_microseconds, timer.elapsed_microseconds());

  if (!forward_inputs.token_ids.defined()) {
    return {};
  }

  timer.reset();
  const auto& output = get_model_output(forward_inputs);
  COUNTER_ADD(rec_first_token_latency_microseconds,
              timer.elapsed_microseconds());

  timer.reset();
  if (output.beam_sequence_group.defined() &&
      output.beam_sequence_group.numel() > 0) {
    batches[0].process_beam_sequence_group(output);
  } else {
    batches[0].process_sample_output(output.sample_output, false);
  }
  COUNTER_ADD(rec_sampling_latency_microseconds, timer.elapsed_microseconds());

  batches[0].finish();
  return output;
}

ForwardOutput RecEngine::OneRecXAttentionEnginePipeline::get_model_output(
    const ForwardInput& model_inputs) {
  const bool trace_engine_output =
      util::get_bool_env("XLLM_DEBUG_ONEREC_ENGINE_TRACE", false);
  const bool trace_stage_timing =
      util::get_bool_env("XLLM_DEBUG_ONEREC_XATTN_STAGE_TIMING", false);
  Timer engine_timer;
  auto log_engine_stage = [&](const char* stage_name,
                              const torch::Tensor& tensor = torch::Tensor()) {
    if (!trace_engine_output) {
      return;
    }
    LOG(INFO) << "OneRec xattention engine stage=" << stage_name
              << ", tensor_defined=" << tensor.defined() << ", tensor_shape="
              << (tensor.defined() ? tensor.sizes() : c10::IntArrayRef{});
  };
  auto log_engine_timing = [&](const char* stage_name) {
    if (!trace_stage_timing) {
      return;
    }
    LOG(INFO) << "OneRec xattention engine timing, stage=" << stage_name
              << ", elapsed_us=" << engine_timer.elapsed_microseconds();
    engine_timer.reset();
  };
  std::vector<folly::SemiFuture<std::optional<ForwardOutput>>> futures;
  futures.reserve(engine_.workers_.size());
  for (auto& worker : engine_.workers_) {
    futures.emplace_back(worker->step_async(model_inputs));
  }
  auto results = folly::collectAll(futures).get();
  log_engine_timing("worker_step_collect");
  log_engine_stage("after_collect_all");

  validate_local_rec_worker_results(results, "OneRec xattention");

  auto forward_output = results.front().value();
  CHECK(forward_output.has_value()) << "Failed to execute model";

  auto& output = forward_output.value();
  auto& sample_output = output.sample_output;
  const bool has_beam_output = output.beam_sequence_group.defined() &&
                               output.beam_sequence_group.numel() > 0;

  if (!has_beam_output && sample_output.embeddings.defined()) {
    log_engine_stage("before_embeddings_to_cpu", sample_output.embeddings);
    sample_output.embeddings = safe_to(
        sample_output.embeddings,
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32),
        /*non_blocking=*/true);
    log_engine_stage("after_embeddings_to_cpu", sample_output.embeddings);
  }

  if (!has_beam_output && sample_output.next_tokens.defined()) {
    log_engine_stage("before_next_tokens_to_cpu", sample_output.next_tokens);
    sample_output.next_tokens =
        safe_to(sample_output.next_tokens, torch::kCPU, /*non_blocking=*/true);
    log_engine_stage("after_next_tokens_to_cpu", sample_output.next_tokens);
    if (sample_output.logprobs.defined()) {
      log_engine_stage("before_logprobs_to_cpu", sample_output.logprobs);
      sample_output.logprobs =
          safe_to(sample_output.logprobs, torch::kCPU, true);
      log_engine_stage("after_logprobs_to_cpu", sample_output.logprobs);
    }
    if (sample_output.top_tokens.defined()) {
      log_engine_stage("before_top_tokens_to_cpu", sample_output.top_tokens);
      sample_output.top_tokens =
          safe_to(sample_output.top_tokens, torch::kCPU, true);
      log_engine_stage("after_top_tokens_to_cpu", sample_output.top_tokens);
    }
    if (sample_output.top_logprobs.defined()) {
      log_engine_stage("before_top_logprobs_to_cpu",
                       sample_output.top_logprobs);
      sample_output.top_logprobs =
          safe_to(sample_output.top_logprobs, torch::kCPU, true);
      log_engine_stage("after_top_logprobs_to_cpu", sample_output.top_logprobs);
    }
  }
  if (has_beam_output) {
    log_engine_stage("before_beam_sequence_group_to_cpu",
                     output.beam_sequence_group);
    output.beam_sequence_group =
        safe_to(output.beam_sequence_group, torch::kCPU, true);
    log_engine_stage("after_beam_sequence_group_to_cpu",
                     output.beam_sequence_group);
  }
  if (output.beam_search_output.out_logprobs.defined() &&
      output.beam_search_output.out_logprobs.numel() > 0) {
    log_engine_stage("before_beam_out_logprobs_to_cpu",
                     output.beam_search_output.out_logprobs);
    output.beam_search_output.out_logprobs =
        safe_to(output.beam_search_output.out_logprobs, torch::kCPU, true);
    log_engine_stage("after_beam_out_logprobs_to_cpu",
                     output.beam_search_output.out_logprobs);
  }
  log_engine_timing("output_d2h_submit");
  log_engine_stage("before_default_stream_sync");
  Device(engine_.workers_[0]->device()).synchronize_default_stream();
  log_engine_timing("default_stream_sync");
  log_engine_stage("after_default_stream_sync");

  return output;
}

// ============================================================
// RecMultiRoundEnginePipeline Implementation
// ============================================================

RecEngine::RecMultiRoundEnginePipeline::RecMultiRoundEnginePipeline(
    RecEngine& engine)
    : RecEnginePipeline(engine) {}

void RecEngine::RecMultiRoundEnginePipeline::setup_workers() {
  // RecMultiRound uses local workers, no DistManager setup needed
}

void RecEngine::RecMultiRoundEnginePipeline::process_group_test() {
  if (engine_.workers_.size() > 1) {
    std::vector<folly::SemiFuture<folly::Unit>> futures;
    futures.reserve(engine_.workers_.size());
    for (auto& worker : engine_.workers_) {
      futures.emplace_back(worker->process_group_test_async());
    }
    const int32_t timeout_seconds =
        util::get_process_group_test_timeout_seconds();
    folly::collectAll(futures)
        .within(std::chrono::seconds(timeout_seconds))
        .get();
  }
}

bool RecEngine::RecMultiRoundEnginePipeline::init_model_workers(
    const std::string& model_path) {
  const auto& devices = engine_.options_.devices();
  const int32_t world_size = static_cast<int32_t>(devices.size());

  // Single-card REC multi-round still needs a non-null tp_group_ during
  // model/layer construction. For NPU, construct a real backend process group
  // through the same ProcessGroup-backed path used by local TP.
#if defined(USE_NPU)
  if (world_size == 1) {
    std::string host;
    int port;
    net::parse_host_port_from_addr(
        engine_.options_.master_node_addr().value(), host, port);
    engine_.process_groups_.clear();
    engine_.process_groups_.emplace_back(create_process_group(
        /*rank=*/0,
        /*world_size=*/1,
        /*rank_size=*/1,
        /*port=*/port,
        /*trans=*/false,
        host,
        /*group_name=*/"rec_single_local_pg",
        devices[0]));
  } else {
    engine_.process_groups_ =
        parallel_state::create_local_process_groups(devices, engine_.options_);
  }
#else
  engine_.process_groups_ =
      parallel_state::create_local_process_groups(devices, engine_.options_);
#endif

  engine_.workers_.clear();
  WorkerType worker_type = WorkerType::REC;
  for (int32_t rank = 0; rank < world_size; ++rank) {
    ProcessGroup* pg = engine_.process_groups_[rank].get();
    ParallelArgs parallel_args(rank, world_size, pg);
    // Set tp_group_ = process_group_ for TP parallelism
    parallel_args.tp_group_ = pg;
    LOG(INFO) << "REC multi-round local worker init, rank=" << rank
              << ", world_size=" << world_size << ", device=" << devices[rank]
              << ", driver=" << (rank == 0) << ", rec_worker_max_concurrency="
              << engine_.options_.rec_worker_max_concurrency();
    engine_.workers_.emplace_back(std::make_unique<Worker>(
        parallel_args, devices[rank], engine_.options_, worker_type));
  }

  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(engine_.workers_.size());
  for (auto& worker : engine_.workers_) {
    futures.emplace_back(worker->init_model_async(
        model_path, FLAGS_random_seed, MasterStatus::WAKEUP));
  }
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }
  initialize_shared_pipeline_indices();
  return true;
}

bool RecEngine::RecMultiRoundEnginePipeline::use_shared_pipeline_index() const {
  return engine_.workers_.size() > 1 &&
         engine_.options_.rec_worker_max_concurrency() > 1;
}

void RecEngine::RecMultiRoundEnginePipeline::
    initialize_shared_pipeline_indices() {
  if (shared_pipeline_indices_initialized_ || !use_shared_pipeline_index()) {
    return;
  }
  // Default to multiple slots only when the per-pipeline ATB TP domain path is
  // enabled. Without that path, all pipelines share one TP communicator and
  // must stay serialized unless the operator explicitly opts in.
  const bool per_pipeline_atb_comm =
      enable_rec_multiround_tp_per_pipeline_atb_comm();
  const bool force_single_tp_pipeline = util::get_bool_env(
      "XLLM_REC_MULTIROUND_TP_FORCE_SINGLE_PIPELINE", !per_pipeline_atb_comm);
  const size_t pipeline_count =
      !enable_rec_multiround_tp_single_model_pipeline() &&
              !force_single_tp_pipeline && per_pipeline_atb_comm
          ? static_cast<size_t>(engine_.options_.rec_worker_max_concurrency())
          : 1;
  shared_pipeline_count_ = pipeline_count;
  shared_pipeline_in_flight_.assign(pipeline_count, false);
  initialize_rec_tp_control_groups(pipeline_count);
  shared_pipeline_first_use_serializing_ =
      per_pipeline_atb_comm && pipeline_count > 1;
  if (shared_pipeline_first_use_serializing_) {
    // ATB/HCCL lazily creates communicator state on the first real request.
    // Serializing the first use of each pipeline avoids concurrent root-info
    // initialization while preserving full pipeline concurrency after warmup.
    shared_pipeline_indices_.enqueue(0);
  } else {
    for (size_t i = 0; i < pipeline_count; ++i) {
      shared_pipeline_indices_.enqueue(i);
    }
  }
  shared_pipeline_indices_initialized_ = true;
  LOG(INFO) << "REC multi-round TP shared pipeline coordinator initialized, "
            << "pipeline_count=" << pipeline_count << ", first_use_serialized="
            << shared_pipeline_first_use_serializing_
            << ", configured_rec_worker_max_concurrency="
            << engine_.options_.rec_worker_max_concurrency()
            << ", single_model_pipeline="
            << enable_rec_multiround_tp_single_model_pipeline()
            << ", local_tp_size=" << engine_.workers_.size();
}

void RecEngine::RecMultiRoundEnginePipeline::initialize_rec_tp_control_groups(
    size_t pipeline_count) {
  if (pipeline_count <= 1 ||
      !enable_rec_multiround_tp_per_pipeline_atb_comm()) {
    return;
  }
  const auto& devices = engine_.options_.devices();
  const int32_t world_size = static_cast<int32_t>(devices.size());
  CHECK_GT(world_size, 1)
      << "per-pipeline REC TP control groups require local TP.";

  std::string host;
  int port;
  net::parse_host_port_from_addr(
      engine_.options_.master_node_addr().value(), host, port);
  host = "127.0.0.1";

  rec_tp_control_groups_by_pipeline_.clear();
  rec_tp_control_groups_by_pipeline_.reserve(pipeline_count);
  constexpr int kRecTpControlPortBaseOffset = 100;
  for (size_t pipeline_index = 0; pipeline_index < pipeline_count;
       ++pipeline_index) {
    std::vector<std::unique_ptr<ProcessGroup>> groups;
    groups.reserve(devices.size());
    const int control_port =
        port + kRecTpControlPortBaseOffset + static_cast<int>(pipeline_index);
    const std::string group_name =
        "rec_tp_control_pipeline_" + std::to_string(pipeline_index);
    for (int32_t rank = 0; rank < world_size; ++rank) {
      groups.emplace_back(create_process_group(rank,
                                               world_size,
                                               world_size,
                                               control_port,
                                               /*trans=*/false,
                                               host,
                                               group_name,
                                               devices[rank]));
    }
    rec_tp_control_groups_by_pipeline_.emplace_back(std::move(groups));
  }

  for (int32_t rank = 0; rank < world_size; ++rank) {
    for (size_t pipeline_index = 0; pipeline_index < pipeline_count;
         ++pipeline_index) {
      engine_.workers_[rank]->set_rec_pipeline_control_group(
          pipeline_index,
          rec_tp_control_groups_by_pipeline_[pipeline_index][rank].get());
    }
  }
  LOG(INFO) << "REC multi-round TP per-pipeline control groups initialized, "
            << "pipeline_count=" << pipeline_count
            << ", world_size=" << world_size
            << ", base_port=" << port + kRecTpControlPortBaseOffset;
}

size_t RecEngine::RecMultiRoundEnginePipeline::lease_shared_pipeline_index() {
  size_t pipeline_index;
  shared_pipeline_indices_.wait_dequeue(pipeline_index);
  CHECK_LT(pipeline_index, shared_pipeline_in_flight_.size())
      << "REC multi-round TP leased invalid pipeline index=" << pipeline_index
      << ", pipeline_count=" << shared_pipeline_in_flight_.size();
  {
    std::lock_guard<std::mutex> lock(shared_pipeline_lease_mutex_);
    CHECK(!shared_pipeline_in_flight_[pipeline_index])
        << "REC multi-round TP leased pipeline already in-flight, pipeline="
        << pipeline_index;
    shared_pipeline_in_flight_[pipeline_index] = true;
  }
  if (util::get_bool_env("XLLM_DEBUG_REC_PIPELINE_CONCURRENCY", false)) {
    LOG(INFO) << "REC multi-round TP leased shared pipeline=" << pipeline_index;
  }
  return pipeline_index;
}

std::vector<size_t>
RecEngine::RecMultiRoundEnginePipeline::lease_all_shared_pipeline_indices() {
  CHECK_GT(shared_pipeline_count_, 0);
  std::vector<size_t> leased_pipeline_indices;
  leased_pipeline_indices.reserve(shared_pipeline_count_);
  for (size_t i = 0; i < shared_pipeline_count_; ++i) {
    leased_pipeline_indices.emplace_back(lease_shared_pipeline_index());
  }
  CHECK(std::find(leased_pipeline_indices.begin(),
                  leased_pipeline_indices.end(),
                  0) != leased_pipeline_indices.end())
      << "REC multi-round TP shape first-use serialization requires pipeline 0 "
         "to be leased.";
  return leased_pipeline_indices;
}

int64_t RecEngine::RecMultiRoundEnginePipeline::rec_multiround_shape_key(
    const ForwardInput& model_inputs) const {
  const StepDecodeMeta* step_meta = model_inputs.step_meta();
  if (step_meta == nullptr) {
    return -1;
  }
  CHECK_GT(step_meta->batch_size, 0);
  CHECK_GT(step_meta->beam_width, 0);
  CHECK_GT(step_meta->total_round, 0);
  int64_t key = step_meta->batch_size;
  key = key * 1024 + step_meta->beam_width;
  key = key * 1024 + step_meta->total_round;
  return key;
}

bool RecEngine::RecMultiRoundEnginePipeline::should_serialize_shape_first_use(
    const ForwardInput& model_inputs) {
  if (!serialize_rec_multiround_tp_shape_first_use() ||
      shared_pipeline_count_ <= 1 ||
      !enable_rec_multiround_tp_per_pipeline_atb_comm()) {
    return false;
  }
  const int64_t shape_key = rec_multiround_shape_key(model_inputs);
  if (shape_key < 0) {
    return false;
  }
  bool inserted = false;
  {
    std::lock_guard<std::mutex> lock(shared_pipeline_shape_init_mutex_);
    inserted = initialized_shape_keys_.insert(shape_key).second;
  }
  if (inserted) {
    {
      std::lock_guard<std::mutex> lock(shared_pipeline_init_mutex_);
      if (shared_pipeline_first_use_serializing_) {
        // The existing pipeline first-use gate is already serializing this
        // request. Remember the shape, but do not add a second serialization.
        return false;
      }
    }
    LOG(INFO) << "REC multi-round TP serializes first use of shape_key="
              << shape_key
              << " to avoid concurrent ATB/HCCL lazy initialization.";
  }
  return inserted;
}

void RecEngine::RecMultiRoundEnginePipeline::release_shared_pipeline_index(
    size_t pipeline_index) {
  CHECK_LT(pipeline_index, shared_pipeline_in_flight_.size())
      << "REC multi-round TP released invalid pipeline index=" << pipeline_index
      << ", pipeline_count=" << shared_pipeline_in_flight_.size();
  {
    std::lock_guard<std::mutex> lock(shared_pipeline_lease_mutex_);
    CHECK(shared_pipeline_in_flight_[pipeline_index])
        << "REC multi-round TP released pipeline that is not in-flight, "
           "pipeline="
        << pipeline_index;
    shared_pipeline_in_flight_[pipeline_index] = false;
  }
  if (shared_pipeline_first_use_serializing_) {
    std::lock_guard<std::mutex> lock(shared_pipeline_init_mutex_);
    if (pipeline_index == shared_pipeline_next_init_index_) {
      ++shared_pipeline_next_init_index_;
      if (shared_pipeline_next_init_index_ < shared_pipeline_count_) {
        shared_pipeline_indices_.enqueue(shared_pipeline_next_init_index_);
      } else {
        shared_pipeline_first_use_serializing_ = false;
        for (size_t i = 0; i < shared_pipeline_count_; ++i) {
          shared_pipeline_indices_.enqueue(i);
        }
        LOG(INFO)
            << "REC multi-round TP shared pipeline first-use initialization "
            << "completed, pipeline_count=" << shared_pipeline_count_;
      }
      return;
    }
  }
  shared_pipeline_indices_.enqueue(pipeline_index);
  if (util::get_bool_env("XLLM_DEBUG_REC_PIPELINE_CONCURRENCY", false)) {
    LOG(INFO) << "REC multi-round TP released shared pipeline="
              << pipeline_index;
  }
}

uint64_t RecEngine::RecMultiRoundEnginePipeline::next_rec_tp_step_id() {
  return rec_tp_step_id_.fetch_add(1, std::memory_order_relaxed);
}

int64_t
RecEngine::RecMultiRoundEnginePipeline::estimate_min_available_memory() {
  const int64_t max_cache_size = engine_.options_.max_cache_size();
  const double max_memory_utilization =
      engine_.options_.max_memory_utilization();

  std::vector<folly::SemiFuture<std::tuple<int64_t, int64_t>>> futures;
  futures.reserve(engine_.workers_.size());
  for (auto& worker : engine_.workers_) {
    futures.emplace_back(worker->estimate_kv_cache_capacity_async());
  }

  int64_t cache_size_in_bytes = std::numeric_limits<int64_t>::max();
  auto results = folly::collectAll(futures).get();
  for (size_t i = 0; i < results.size(); ++i) {
    if (!results[i].hasValue()) {
      LOG(ERROR) << "Failed to profile memory usage for worker: " << i;
      continue;
    }
    auto [available_memory, total_memory] = results[i].value();
    LOG(INFO) << "worker #" << i
              << ": available memory: " << readable_size(available_memory)
              << ", total memory: " << readable_size(total_memory)
              << ". Using max_memory_utilization: " << max_memory_utilization
              << ", max_cache_size: " << readable_size(max_cache_size);
    if (max_memory_utilization < 1.0) {
      const int64_t buffer_memory =
          total_memory * (1.0 - max_memory_utilization);
      available_memory -= buffer_memory;
    }
    if (max_cache_size > 0) {
      available_memory = std::min(available_memory, max_cache_size);
    }
    cache_size_in_bytes = std::min(cache_size_in_bytes, available_memory);
  }
  return cache_size_in_bytes;
}

bool RecEngine::RecMultiRoundEnginePipeline::allocate_kv_cache(
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(engine_.workers_.size());
  for (auto& worker : engine_.workers_) {
    futures.emplace_back(worker->allocate_kv_cache_async(kv_cache_shape));
  }
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }
  return true;
}

size_t RecEngine::RecMultiRoundEnginePipeline::num_workers() const {
  return engine_.workers_.size();
}

ForwardOutput RecEngine::RecMultiRoundEnginePipeline::step(
    std::vector<Batch>& batches) {
  if (engine_.workers_.empty()) {
    return {};
  }

  const bool trace_engine_timing = enable_rec_multiround_engine_timing();
  Timer timer;
  // Call worker's prepare_inputs (multi-round logic is inside worker)
  auto forward_inputs = engine_.workers_[0]->prepare_inputs(batches[0]);
  COUNTER_ADD(prepare_input_latency_microseconds, timer.elapsed_microseconds());
  if (trace_engine_timing) {
    LOG(INFO) << "REC multi-round engine timing, stage=prepare_inputs"
              << ", batch_size=" << batches[0].size()
              << ", elapsed_us=" << timer.elapsed_microseconds();
  }

  if (!forward_inputs.token_ids.defined()) {
    return {};
  }

  timer.reset();
  // Execute model inference (only one step, multi-round handled by worker)
  const auto& output = get_model_output(forward_inputs);
  COUNTER_ADD(rec_first_token_latency_microseconds,
              timer.elapsed_microseconds());
  if (trace_engine_timing) {
    LOG(INFO) << "REC multi-round engine timing, stage=get_model_output"
              << ", batch_size=" << batches[0].size()
              << ", elapsed_us=" << timer.elapsed_microseconds();
  }

  timer.reset();
  // Use process_beam_sequence_group for multi-round beam search results
  // instead of process_sample_output which would call append_token()
  batches[0].process_beam_sequence_group(output);
  COUNTER_ADD(rec_sampling_latency_microseconds, timer.elapsed_microseconds());
  if (trace_engine_timing) {
    LOG(INFO) << "REC multi-round engine timing, stage=process_beam_sequence"
              << ", batch_size=" << batches[0].size()
              << ", elapsed_us=" << timer.elapsed_microseconds();
  }

  batches[0].finish();
  return output;
}

ForwardOutput RecEngine::RecMultiRoundEnginePipeline::get_model_output(
    const ForwardInput& model_inputs) {
  const bool trace_engine_output =
      util::get_bool_env("XLLM_DEBUG_ONEREC_ENGINE_TRACE", false);
  auto log_engine_stage = [&](const char* stage_name,
                              const torch::Tensor& tensor = torch::Tensor()) {
    if (!trace_engine_output) {
      return;
    }
    LOG(INFO) << "REC multi-round engine stage=" << stage_name
              << ", tensor_defined=" << tensor.defined() << ", tensor_shape="
              << (tensor.defined() ? tensor.sizes() : c10::IntArrayRef{})
              << ", tensor_device="
              << (tensor.defined() ? tensor.device().str() : "<undefined>");
  };

  std::vector<folly::SemiFuture<std::optional<ForwardOutput>>> futures;
  futures.reserve(engine_.workers_.size());
  std::optional<size_t> shared_pipeline_index;
  std::vector<size_t> leased_pipeline_indices;
  const bool trace_engine_timing = enable_rec_multiround_engine_timing();
  Timer stage_timer;
  if (use_shared_pipeline_index()) {
    std::lock_guard<std::mutex> lock(shared_pipeline_acquire_mutex_);
    if (should_serialize_shape_first_use(model_inputs)) {
      leased_pipeline_indices = lease_all_shared_pipeline_indices();
      shared_pipeline_index = 0;
    } else {
      shared_pipeline_index = lease_shared_pipeline_index();
      leased_pipeline_indices.emplace_back(shared_pipeline_index.value());
    }
  }
  if (trace_engine_timing) {
    LOG(INFO) << "REC multi-round engine timing, stage=lease_pipeline"
              << ", shared_pipeline="
              << (shared_pipeline_index.has_value()
                      ? std::to_string(shared_pipeline_index.value())
                      : "none")
              << ", leased_count=" << leased_pipeline_indices.size()
              << ", elapsed_us=" << stage_timer.elapsed_microseconds();
  }
  stage_timer.reset();
  const uint64_t rec_tp_step_id =
      engine_.workers_.size() > 1 ? next_rec_tp_step_id() : 0;
  auto pipeline_index_guard = xllm::ScopeGuard([&] {
    for (size_t pipeline_index : leased_pipeline_indices) {
      release_shared_pipeline_index(pipeline_index);
    }
  });
  for (auto& worker : engine_.workers_) {
    if (shared_pipeline_index.has_value()) {
      futures.emplace_back(worker->rec_step_async_with_pipeline_index(
          model_inputs, shared_pipeline_index.value(), rec_tp_step_id));
    } else if (rec_tp_step_id != 0) {
      futures.emplace_back(worker->rec_step_async_with_pipeline_index(
          model_inputs, /*pipeline_index=*/0, rec_tp_step_id));
    } else {
      futures.emplace_back(worker->step_async(model_inputs));
    }
  }
  if (trace_engine_timing) {
    LOG(INFO) << "REC multi-round engine timing, stage=schedule_workers"
              << ", rec_tp_step_id=" << rec_tp_step_id
              << ", futures=" << futures.size()
              << ", elapsed_us=" << stage_timer.elapsed_microseconds();
  }
  stage_timer.reset();
  auto results = folly::collectAll(futures).get();
  if (trace_engine_timing) {
    LOG(INFO) << "REC multi-round engine timing, stage=collect_workers"
              << ", rec_tp_step_id=" << rec_tp_step_id
              << ", elapsed_us=" << stage_timer.elapsed_microseconds();
  }
  stage_timer.reset();

  validate_local_rec_worker_results(results, "REC multi-round");
  if (trace_engine_timing) {
    LOG(INFO) << "REC multi-round engine timing, stage=validate_results"
              << ", rec_tp_step_id=" << rec_tp_step_id
              << ", elapsed_us=" << stage_timer.elapsed_microseconds();
  }
  stage_timer.reset();

  auto forward_output = results.front().value();
  CHECK(forward_output.has_value()) << "Failed to execute model";

  // D2H transfer for beam_sequence_group (multi-round results)
  auto& output = forward_output.value();
  Device(engine_.workers_[0]->device()).set_device();
  // TODO. uncomment this in next pr.
  log_engine_stage("before_beam_sequence_group_to_cpu",
                   output.beam_sequence_group);
  output.beam_sequence_group = safe_to(output.beam_sequence_group, torch::kCPU);
  log_engine_stage("after_beam_sequence_group_to_cpu",
                   output.beam_sequence_group);
  if (trace_engine_timing) {
    LOG(INFO) << "REC multi-round engine timing, stage=beam_sequence_to_cpu"
              << ", rec_tp_step_id=" << rec_tp_step_id
              << ", elapsed_us=" << stage_timer.elapsed_microseconds();
  }
  stage_timer.reset();
  if (output.beam_search_output.out_logprobs.defined()) {
    log_engine_stage("before_beam_out_logprobs_to_cpu",
                     output.beam_search_output.out_logprobs);
    output.beam_search_output.out_logprobs =
        safe_to(output.beam_search_output.out_logprobs, torch::kCPU);
    log_engine_stage("after_beam_out_logprobs_to_cpu",
                     output.beam_search_output.out_logprobs);
    if (trace_engine_timing) {
      LOG(INFO) << "REC multi-round engine timing, stage=out_logprobs_to_cpu"
                << ", rec_tp_step_id=" << rec_tp_step_id
                << ", elapsed_us=" << stage_timer.elapsed_microseconds();
    }
    stage_timer.reset();
  }

  return output;
}

std::vector<int64_t>
RecEngine::RecMultiRoundEnginePipeline::get_active_activation_memory() const {
  std::vector<folly::SemiFuture<int64_t>> futures;
  futures.reserve(engine_.workers_.size());
  for (auto& worker : engine_.workers_) {
    futures.emplace_back(worker->get_active_activation_memory_async());
  }

  auto results = folly::collectAll(futures).get();
  std::vector<int64_t> active_activation_memories;
  active_activation_memories.reserve(futures.size());
  for (auto& result : results) {
    active_activation_memories.emplace_back(result.value());
  }
  return active_activation_memories;
}

// ============================================================
// RecEngine pipeline factory (static method)
// ============================================================
std::unique_ptr<RecEngine::RecEnginePipeline> RecEngine::create_pipeline(
    RecPipelineType type,
    RecEngine& engine) {
  switch (type) {
    case RecPipelineType::kLlmRecDefault:
      return std::make_unique<LlmRecEnginePipeline>(engine);
    case RecPipelineType::kLlmRecMultiRoundPipeline:
      return std::make_unique<RecMultiRoundEnginePipeline>(engine);
    case RecPipelineType::kOneRecDefault:
      return std::make_unique<OneRecPrefillOnlyEnginePipeline>(engine);
    case RecPipelineType::kOneRecXAttentionPipeline:
      return std::make_unique<OneRecXAttentionEnginePipeline>(engine);
    default:
      LOG(FATAL) << "Unknown RecEngine pipeline type: "
                 << static_cast<int>(type);
      return nullptr;
  }
}

}  // namespace xllm
