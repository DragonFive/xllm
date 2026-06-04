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
#include <chrono>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <thread>
#include <tuple>
#include <vector>

#include "common/device_monitor.h"
#include "common/global_flags.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "framework/model/model_input_params.h"
#include "util/rec_model_utils.h"
#if defined(USE_CUDA)
#include "kernels/cuda/cuda_ops_api.h"
#include "kernels/cuda/xattention/xattention_ops_api.h"
#include "layers/cuda/flashinfer_workspace.h"
#include "layers/cuda/xattention_workspace.h"
#include "platform/cuda/device_capture_lock.h"
#endif
#if defined(USE_NPU)
#include "kernels/npu/npu_ops_api.h"
#include "kernels/npu/xllm_ops/xllm_ops_api.h"
#include "platform/npu/device_capture_lock.h"
#endif
#include "common/version_singleton.h"
#include "framework/model_loader.h"
#include "framework/sampling/rec_constrained_decoding.h"
#include "framework/sampling/rec_sampler.h"
#include "framework/state_dict/rec_vocab_dict.h"
#include "models/model_registry.h"
#include "runtime/rec_beam_utils.h"
#include "util/env_var.h"
#include "util/scope_guard.h"
#include "util/timer.h"

namespace xllm {

namespace {

constexpr const char* kRecMultiRoundDebugDeviceSyncEnv =
    "XLLM_DEBUG_REC_MULTIROUND_DEVICE_SYNC";
constexpr const char* kRecMultiRoundStageTimingEnv =
    "XLLM_DEBUG_REC_MULTIROUND_STAGE_TIMING";
constexpr const char* kRecMultiRoundSyncLogitsAfterLmHeadEnv =
    "XLLM_REC_MULTIROUND_SYNC_LOGITS_AFTER_LMHEAD";
constexpr const char* kRecMultiRoundSyncLogitsPolicyEnv =
    "XLLM_REC_MULTIROUND_SYNC_LOGITS_POLICY";
constexpr const char* kRecMultiRoundTpPerPipelineAtbCommEnv =
    "XLLM_REC_MULTIROUND_TP_PER_PIPELINE_ATB_COMM";
constexpr const char* kRecMultiRoundTpRank0ControlEnv =
    "XLLM_REC_MULTIROUND_TP_RANK0_CONTROL";
constexpr const char* kRecMultiRoundTpPackedRank0ControlEnv =
    "XLLM_REC_MULTIROUND_TP_PACKED_RANK0_CONTROL";
constexpr const char* kRecMultiRoundTpHostSharedControlEnv =
    "XLLM_REC_MULTIROUND_TP_HOST_SHARED_CONTROL";
constexpr const char* kRecMultiRoundTpNpuIntermediateBeamEnv =
    "XLLM_REC_MULTIROUND_TP_NPU_INTERMEDIATE_BEAM";
constexpr const char* kRecMultiRoundTpSerializeCacheSelectEnv =
    "XLLM_REC_MULTIROUND_TP_SERIALIZE_CACHE_SELECT";
constexpr const char* kRecMultiRoundTpSingleModelPipelineEnv =
    "XLLM_REC_MULTIROUND_TP_SINGLE_MODEL_PIPELINE";

RecVocabDict* get_onerec_vocab_dict(const std::string& model_weights_path) {
  if (model_weights_path.empty()) {
    return nullptr;
  }
  const std::string model_version =
      std::filesystem::path(model_weights_path).filename().string();
  return VersionSingleton<RecVocabDict>::GetInstance(model_version);
}

int32_t get_onerec_decode_round(const OneRecXAttentionParams& params) {
  if (params.rec_stage != OneRecModelInputParams::RecStage::DECODE ||
      params.generated_tokens.empty()) {
    return 0;
  }
  return std::max<int32_t>(
      static_cast<int32_t>(params.generated_tokens.front().size()) - 1, 0);
}

bool enable_onerec_selected_token_cpu_check() {
  return util::get_bool_env("XLLM_DEBUG_ONEREC_SELECTED_TOKEN_CPU_CHECK",
                            false) &&
         !util::get_bool_env("XLLM_DEBUG_ONEREC_SKIP_SELECTED_TOKEN_CPU_CHECK",
                             false);
}

bool enable_onerec_xattention_stage_timing() {
  return util::get_bool_env("XLLM_DEBUG_ONEREC_XATTN_STAGE_TIMING", false);
}

bool enable_rec_pipeline_concurrency_debug() {
  return util::get_bool_env("XLLM_DEBUG_REC_PIPELINE_CONCURRENCY", false);
}

bool enable_rec_multiround_debug_device_sync() {
  return util::get_bool_env(kRecMultiRoundDebugDeviceSyncEnv, false);
}

bool enable_rec_multiround_stage_timing() {
  return util::get_bool_env(kRecMultiRoundStageTimingEnv, false);
}

bool sync_rec_multiround_logits_after_lmhead() {
  return util::get_bool_env(kRecMultiRoundSyncLogitsAfterLmHeadEnv, true);
}

enum class RecMultiRoundLogitsSyncPolicy {
  kAll,
  kRank0Only,
  kNonDriverOnly,
  kFinalOnly,
  kNonFinalOnly,
  kNone,
};

RecMultiRoundLogitsSyncPolicy rec_multiround_logits_sync_policy() {
  if (!sync_rec_multiround_logits_after_lmhead()) {
    return RecMultiRoundLogitsSyncPolicy::kNone;
  }
  const char* policy_env = std::getenv(kRecMultiRoundSyncLogitsPolicyEnv);
  if (policy_env == nullptr || std::string(policy_env).empty() ||
      std::string(policy_env) == "all") {
    return RecMultiRoundLogitsSyncPolicy::kAll;
  }
  const std::string policy(policy_env);
  if (policy == "rank0_only") {
    return RecMultiRoundLogitsSyncPolicy::kRank0Only;
  }
  if (policy == "non_driver_only") {
    return RecMultiRoundLogitsSyncPolicy::kNonDriverOnly;
  }
  if (policy == "final_only") {
    return RecMultiRoundLogitsSyncPolicy::kFinalOnly;
  }
  if (policy == "non_final_only") {
    return RecMultiRoundLogitsSyncPolicy::kNonFinalOnly;
  }
  if (policy == "none") {
    return RecMultiRoundLogitsSyncPolicy::kNone;
  }
  LOG(FATAL) << "Unsupported " << kRecMultiRoundSyncLogitsPolicyEnv << "="
             << policy
             << ". Expected one of all, rank0_only, non_driver_only, "
                "final_only, non_final_only, none.";
  return RecMultiRoundLogitsSyncPolicy::kAll;
}

bool enable_rec_multiround_tp_per_pipeline_atb_comm() {
  return util::get_bool_env(kRecMultiRoundTpPerPipelineAtbCommEnv, false);
}

bool enable_rec_multiround_tp_rank0_control() {
  return util::get_bool_env(kRecMultiRoundTpRank0ControlEnv, false);
}

bool enable_rec_multiround_tp_packed_rank0_control() {
  return util::get_bool_env(kRecMultiRoundTpPackedRank0ControlEnv, false);
}

bool enable_rec_multiround_tp_host_shared_control() {
  return util::get_bool_env(kRecMultiRoundTpHostSharedControlEnv, false);
}

bool enable_rec_multiround_tp_npu_intermediate_beam() {
  return util::get_bool_env(kRecMultiRoundTpNpuIntermediateBeamEnv, false);
}

bool serialize_rec_multiround_tp_cache_select() {
  return util::get_bool_env(kRecMultiRoundTpSerializeCacheSelectEnv, true);
}

bool enable_rec_multiround_tp_single_model_pipeline() {
  return util::get_bool_env(kRecMultiRoundTpSingleModelPipelineEnv, false);
}

std::mutex& rec_multiround_tp_cache_select_mutex() {
  static std::mutex mutex;
  return mutex;
}

uint64_t rec_multiround_control_key(uint64_t rec_tp_step_id, int32_t round) {
  CHECK_GT(rec_tp_step_id, 0);
  CHECK_GE(round, 0);
  constexpr uint64_t kMaxRecMultiRoundControlRounds = 1024;
  CHECK_LT(static_cast<uint64_t>(round), kMaxRecMultiRoundControlRounds);
  return rec_tp_step_id * kMaxRecMultiRoundControlRounds +
         static_cast<uint64_t>(round);
}

void log_rec_multiround_stage_timing(int32_t rank,
                                     int32_t round,
                                     const char* stage,
                                     const Timer& timer) {
  if (!enable_rec_multiround_stage_timing()) {
    return;
  }
  LOG(INFO) << "REC multi-round host timing, rank=" << rank
            << ", round=" << round << ", stage=" << stage
            << ", elapsed_ms=" << timer.elapsed_milliseconds();
}

#if defined(USE_NPU)
bool should_sync_rec_multiround_tp_stage(int32_t world_size,
                                         const torch::Device& device,
                                         bool is_driver,
                                         bool final_round) {
  if (world_size <= 1 || device.type() != torch::kPrivateUse1) {
    return false;
  }
  switch (rec_multiround_logits_sync_policy()) {
    case RecMultiRoundLogitsSyncPolicy::kAll:
      return true;
    case RecMultiRoundLogitsSyncPolicy::kRank0Only:
      return is_driver;
    case RecMultiRoundLogitsSyncPolicy::kNonDriverOnly:
      return !is_driver;
    case RecMultiRoundLogitsSyncPolicy::kFinalOnly:
      return final_round;
    case RecMultiRoundLogitsSyncPolicy::kNonFinalOnly:
      return !final_round;
    case RecMultiRoundLogitsSyncPolicy::kNone:
      return false;
  }
  return true;
}

void debug_sync_rec_multiround_device(const torch::Device& device,
                                      int32_t rank,
                                      int32_t round,
                                      const char* stage) {
  if (!enable_rec_multiround_debug_device_sync()) {
    return;
  }
  Device device_guard(device);
  device_guard.set_device();
  const auto ret = aclrtSynchronizeDevice();
  LOG(INFO) << "REC multi-round debug device sync, rank=" << rank
            << ", round=" << round << ", stage=" << stage
            << ", device=" << device << ", ret=" << ret;
  CHECK_EQ(ret, ACL_SUCCESS) << "REC multi-round debug device sync failed"
                             << ", rank=" << rank << ", round=" << round
                             << ", stage=" << stage << ", device=" << device;
}
#endif

std::string tensor_debug_shape(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return "<undefined>";
  }
  std::string shape = "[";
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    if (i > 0) {
      shape += ",";
    }
    shape += std::to_string(tensor.size(i));
  }
  shape += "]";
  return shape;
}

#if defined(USE_NPU)
torch::Tensor int32_vector_to_device_tensor(const std::vector<int32_t>& values,
                                            const torch::Device& device) {
  auto cpu_options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  torch::Tensor cpu_tensor = values.empty()
                                 ? torch::empty({0}, cpu_options)
                                 : torch::tensor(values, cpu_options);
  return cpu_tensor.to(device, /*non_blocking=*/false);
}

torch::Tensor int64_vector_to_device_tensor(const std::vector<int64_t>& values,
                                            const torch::Device& device) {
  auto cpu_options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  torch::Tensor cpu_tensor = values.empty()
                                 ? torch::empty({0}, cpu_options)
                                 : torch::tensor(values, cpu_options);
  return cpu_tensor.to(device, /*non_blocking=*/false);
}
#endif  // defined(USE_NPU)

int32_t get_requested_beam_result_width(const SamplingParameters& params,
                                        int32_t beam_width) {
  return params.num_return_sequences > 0 ? params.num_return_sequences
                                         : beam_width;
}

struct OneRecBeamSearchTensors {
  torch::Tensor sequence_group;
  torch::Tensor acc_logprob;
  torch::Tensor out_log_probs;
  torch::Tensor out_token_ids;
  torch::Tensor out_token_index;
  torch::Tensor out_beam_count_prefix_sums;
  torch::Tensor out_seqgroup;
};

OneRecBeamSearchTensors prepare_onerec_beam_search_tensors(
    int32_t batch_size,
    int32_t beam_width,
    int32_t total_rounds,
    const torch::Device& device) {
  const torch::TensorOptions int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  const torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const int64_t num_seq = static_cast<int64_t>(batch_size) * beam_width;

  OneRecBeamSearchTensors tensors;
  tensors.sequence_group =
      torch::zeros({batch_size, beam_width, total_rounds}, int_options);
  tensors.acc_logprob = torch::zeros({num_seq, 1}, fp32_options);
  tensors.out_log_probs = torch::zeros({num_seq, 1}, fp32_options);
  tensors.out_token_ids = torch::zeros({num_seq, 1}, int_options);
  tensors.out_token_index = torch::zeros({num_seq, 1}, int_options);
  tensors.out_beam_count_prefix_sums = torch::zeros({num_seq, 1}, int_options);
  tensors.out_seqgroup = torch::zeros_like(tensors.sequence_group);
  return tensors;
}

struct OneRecBeamSearchOutputTensors {
  torch::Tensor out_log_probs;
  torch::Tensor out_token_ids;
  torch::Tensor out_token_index;
  torch::Tensor out_beam_count_prefix_sums;
  torch::Tensor out_seqgroup;
};

OneRecBeamSearchOutputTensors prepare_onerec_beam_search_output_tensors(
    int32_t batch_size,
    int32_t output_width,
    int32_t total_rounds,
    const torch::Device& device) {
  const torch::TensorOptions int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  const torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const int64_t num_seq = static_cast<int64_t>(batch_size) * output_width;

  OneRecBeamSearchOutputTensors tensors;
  tensors.out_log_probs = torch::zeros({num_seq, 1}, fp32_options);
  tensors.out_token_ids = torch::zeros({num_seq, 1}, int_options);
  tensors.out_token_index = torch::zeros({num_seq, 1}, int_options);
  tensors.out_beam_count_prefix_sums = torch::zeros({num_seq, 1}, int_options);
  tensors.out_seqgroup =
      torch::zeros({batch_size, output_width, total_rounds}, int_options);
  return tensors;
}

#if defined(USE_NPU)
bool can_use_beam_search_rec_final_select(int32_t batch_size,
                                          const torch::Tensor& top_tokens,
                                          int32_t result_width) {
  // Gate for using the NPU fused final-step path for `num_return_sequences`.
  // If any shape/alignment requirement is not met, we fall back to the
  // torch-based host implementation (`select_final_onerec_beam_results`) for
  // correctness.
  constexpr int32_t kMaxFinalSelectRequestNum = 48;
  constexpr int32_t kMaxFinalSelectTopK = 2048;
  constexpr int32_t kMaxFinalSelectMergeWidth = 2048;

  // batch_size is the number of independent requests in the micro-batch.
  // The fused kernel is only validated up to kMaxFinalSelectRequestNum.
  if (batch_size <= 0 || batch_size > kMaxFinalSelectRequestNum) {
    return false;
  }

  // top_tokens is expected to be a 2D tensor: [batch_size * beam_width, top_k].
  if (top_tokens.dim() != 2) {
    return false;
  }

  // The row dimension should be divisible by batch_size so we can infer an
  // integer beam_width per request.
  if (top_tokens.size(0) % batch_size != 0) {
    return false;
  }

  // candidate_top_k is the number of per-parent candidates produced by the
  // sampler on the final round (top-k per active beam).
  const int64_t candidate_top_k = top_tokens.size(1);

  // Fused `num_return_sequences` final-step requirements:
  // 1) candidate_top_k >= result_width:
  //    The final selection must be able to pick result_width beams; if each
  //    parent beam only provides fewer candidates, the fused path can miss
  //    valid top results.
  // 2) candidate_top_k % 8 == 0:
  //    Implementation constraint for vectorized loads / alignment on NPU.
  // 3) result_width % 32 == 0:
  //    Implementation constraint for output tiling / alignment.
  // 4) candidate_top_k <= kMaxFinalSelectTopK:
  //    Hard cap for internal buffers / workspace.
  // 5) result_width * 2 <= kMaxFinalSelectMergeWidth:
  //    The kernel uses an internal merge width (often about 2x result_width)
  //    and must stay within its supported limit.
  return candidate_top_k >= result_width && candidate_top_k % 8 == 0 &&
         result_width % 32 == 0 && candidate_top_k <= kMaxFinalSelectTopK &&
         result_width * 2 <= kMaxFinalSelectMergeWidth;
}
#endif  // defined(USE_NPU)

void select_final_onerec_beam_results(
    const torch::Tensor& acc_logprob,
    const torch::Tensor& sequence_group,
    const torch::Tensor& top_tokens,
    const torch::Tensor& top_logprobs,
    int32_t batch_size,
    int32_t beam_width,
    int32_t result_width,
    int32_t total_rounds,
    int32_t current_step,
    OneRecBeamSearchOutputTensors& output_tensors) {
  CHECK_GT(result_width, 0);
  CHECK_EQ(acc_logprob.dim(), 2) << "acc_logprob must be [batch * beam, 1]";
  CHECK_EQ(sequence_group.dim(), 3)
      << "sequence_group must be [batch, beam, total_rounds]";
  CHECK_EQ(top_tokens.dim(), 2) << "top_tokens must be [batch * beam, top_k]";
  CHECK_EQ(top_logprobs.dim(), 2)
      << "top_logprobs must be [batch * beam, top_k]";
  CHECK_EQ(top_tokens.sizes(), top_logprobs.sizes())
      << "top_tokens/top_logprobs shape mismatch";
  CHECK_EQ(top_tokens.size(0), static_cast<int64_t>(batch_size) * beam_width)
      << "top_tokens rows must equal batch * beam";

  const int64_t top_k = top_tokens.size(1);
  CHECK_GT(top_k, 0);
  CHECK_LE(result_width, beam_width * top_k)
      << "num_return_sequences exceeds available final beam candidates";

  const torch::Device device = top_tokens.device();
  const torch::Tensor batch_offsets =
      torch::arange(batch_size,
                    torch::TensorOptions().dtype(torch::kLong).device(device))
          .unsqueeze(1) *
      beam_width;
  const torch::Tensor top_tokens_view =
      top_tokens.view({batch_size, beam_width, top_k});
  const torch::Tensor top_logprobs_view =
      top_logprobs.view({batch_size, beam_width, top_k});
  const torch::Tensor acc_logprob_view =
      acc_logprob.view({batch_size, beam_width, 1});
  const torch::Tensor combined_probs =
      (acc_logprob_view + top_logprobs_view)
          .view({batch_size, beam_width * top_k});

  torch::Tensor new_probs;
  torch::Tensor new_indices;
  std::tie(new_probs, new_indices) = combined_probs.topk(result_width,
                                                         /*dim=*/-1,
                                                         /*largest=*/true,
                                                         /*sorted=*/true);

  const torch::Tensor parent_beam = torch::floor_divide(new_indices, top_k);
  const torch::Tensor token_in_beam = torch::remainder(new_indices, top_k);
  const torch::Tensor batch_idx =
      torch::arange(batch_size,
                    torch::TensorOptions().dtype(torch::kLong).device(device))
          .unsqueeze(1)
          .expand_as(parent_beam);

  using torch::indexing::Slice;
  using torch::indexing::TensorIndex;
  const torch::Tensor new_tokens =
      top_tokens_view.index({TensorIndex(batch_idx),
                             TensorIndex(parent_beam),
                             TensorIndex(token_in_beam)});

  output_tensors.out_log_probs.view({batch_size, result_width})
      .copy_(new_probs);
  output_tensors.out_token_index.view({batch_size, result_width})
      .copy_((parent_beam + batch_offsets).to(torch::kInt32));
  output_tensors.out_token_ids.view({batch_size, result_width})
      .copy_(new_tokens);

  const torch::Tensor batch_range =
      torch::arange(batch_size,
                    torch::TensorOptions().dtype(torch::kLong).device(device))
          .unsqueeze(1)
          .expand({batch_size, result_width});
  output_tensors.out_seqgroup.slice(/*dim=*/2, /*start=*/0, current_step)
      .copy_(sequence_group.index({TensorIndex(batch_range),
                                   TensorIndex(parent_beam),
                                   Slice(0, current_step)}));
  output_tensors.out_seqgroup
      .slice(/*dim=*/2, /*start=*/current_step, /*end=*/current_step + 1)
      .copy_(new_tokens.unsqueeze(2));
}

void select_intermediate_onerec_beam_results(
    const torch::Tensor& acc_logprob,
    const torch::Tensor& sequence_group,
    const torch::Tensor& top_tokens,
    const torch::Tensor& top_logprobs,
    int32_t batch_size,
    int32_t beam_width,
    int32_t current_step,
    torch::Tensor& out_token_ids,
    torch::Tensor& out_token_index,
    torch::Tensor& out_log_probs,
    torch::Tensor& out_beam_count_prefix_sums,
    torch::Tensor& out_sequence) {
#if defined(USE_NPU)
  if (top_tokens.device().is_privateuseone()) {
    auto acc_logprob_cpu = acc_logprob.to(torch::kCPU);
    auto sequence_group_cpu = sequence_group.to(torch::kCPU);
    auto top_tokens_cpu = top_tokens.to(torch::kCPU);
    auto top_logprobs_cpu = top_logprobs.to(torch::kCPU);
    auto out_token_ids_cpu = torch::empty_like(out_token_ids, torch::kCPU);
    auto out_token_index_cpu = torch::empty_like(out_token_index, torch::kCPU);
    auto out_log_probs_cpu = torch::empty_like(out_log_probs, torch::kCPU);
    auto out_beam_count_prefix_sums_cpu =
        torch::empty_like(out_beam_count_prefix_sums, torch::kCPU);
    auto out_sequence_cpu = torch::empty_like(out_sequence, torch::kCPU);

    select_intermediate_onerec_beam_results(acc_logprob_cpu,
                                            sequence_group_cpu,
                                            top_tokens_cpu,
                                            top_logprobs_cpu,
                                            batch_size,
                                            beam_width,
                                            current_step,
                                            out_token_ids_cpu,
                                            out_token_index_cpu,
                                            out_log_probs_cpu,
                                            out_beam_count_prefix_sums_cpu,
                                            out_sequence_cpu);

    out_token_ids.copy_(out_token_ids_cpu.to(top_tokens.device()));
    out_token_index.copy_(out_token_index_cpu.to(top_tokens.device()));
    out_log_probs.copy_(out_log_probs_cpu.to(top_tokens.device()));
    out_beam_count_prefix_sums.copy_(
        out_beam_count_prefix_sums_cpu.to(top_tokens.device()));
    out_sequence.copy_(out_sequence_cpu.to(top_tokens.device()));
    return;
  }
#endif

  CHECK_EQ(acc_logprob.dim(), 2) << "acc_logprob must be [batch * beam, 1]";
  CHECK_EQ(sequence_group.dim(), 3)
      << "sequence_group must be [batch, beam, total_rounds]";
  CHECK_EQ(top_tokens.dim(), 2) << "top_tokens must be [batch * beam, top_k]";
  CHECK_EQ(top_logprobs.dim(), 2)
      << "top_logprobs must be [batch * beam, top_k]";
  CHECK_EQ(top_tokens.sizes(), top_logprobs.sizes())
      << "top_tokens/top_logprobs shape mismatch";
  CHECK_EQ(top_tokens.size(0), static_cast<int64_t>(batch_size) * beam_width)
      << "top_tokens rows must equal batch * beam";
  CHECK_EQ(out_sequence.dim(), 3)
      << "out_sequence must be [batch, beam, total_rounds]";

  const int64_t candidate_top_k = top_tokens.size(1);
  CHECK_GT(candidate_top_k, 0);

  const torch::Device device = top_tokens.device();
  const torch::Tensor top_tokens_view =
      top_tokens.view({batch_size, beam_width, candidate_top_k});
  const torch::Tensor top_logprobs_view =
      top_logprobs.view({batch_size, beam_width, candidate_top_k});
  const torch::Tensor acc_logprob_view =
      acc_logprob.view({batch_size, beam_width, 1});
  const torch::Tensor combined_probs =
      (acc_logprob_view + top_logprobs_view)
          .view({batch_size, beam_width * candidate_top_k});

  torch::Tensor new_probs;
  torch::Tensor new_indices;
  std::tie(new_probs, new_indices) = combined_probs.topk(beam_width,
                                                         /*dim=*/-1,
                                                         /*largest=*/true,
                                                         /*sorted=*/true);

  const torch::Tensor parent_beam =
      torch::floor_divide(new_indices, candidate_top_k);
  const torch::Tensor token_in_beam =
      torch::remainder(new_indices, candidate_top_k);
  const torch::Tensor batch_idx =
      torch::arange(batch_size,
                    torch::TensorOptions().dtype(torch::kLong).device(device))
          .unsqueeze(1)
          .expand_as(parent_beam);

  using torch::indexing::Slice;
  using torch::indexing::TensorIndex;
  const torch::Tensor new_tokens =
      top_tokens_view.index({TensorIndex(batch_idx),
                             TensorIndex(parent_beam),
                             TensorIndex(token_in_beam)});

  const torch::Tensor beam_range = torch::arange(
      beam_width, torch::TensorOptions().dtype(torch::kLong).device(device));
  const torch::Tensor parent_one_hot =
      parent_beam.unsqueeze(-1).eq(beam_range).to(torch::kLong);
  const torch::Tensor parent_counts = parent_one_hot.sum(/*dim=*/1);
  const torch::Tensor parent_prefix = parent_counts.cumsum(/*dim=*/1);
  const torch::Tensor parent_starts = parent_prefix - parent_counts;
  const torch::Tensor rank_in_parent =
      parent_one_hot.cumsum(/*dim=*/1)
          .gather(/*dim=*/2, parent_beam.unsqueeze(-1))
          .squeeze(-1) -
      1;
  const torch::Tensor output_pos =
      parent_starts.gather(/*dim=*/1, parent_beam) + rank_in_parent;

  torch::Tensor sorted_tokens =
      torch::empty({batch_size, beam_width}, top_tokens.options());
  torch::Tensor sorted_probs =
      torch::empty({batch_size, beam_width}, top_logprobs.options());
  torch::Tensor sorted_parent =
      torch::empty({batch_size, beam_width},
                   torch::TensorOptions().dtype(torch::kLong).device(device));
  sorted_tokens.scatter_(/*dim=*/1, output_pos, new_tokens);
  sorted_probs.scatter_(/*dim=*/1, output_pos, new_probs);
  sorted_parent.scatter_(/*dim=*/1, output_pos, parent_beam);

  const torch::Tensor batch_offsets =
      torch::arange(batch_size,
                    torch::TensorOptions().dtype(torch::kLong).device(device))
          .unsqueeze(1) *
      beam_width;
  out_token_ids.view({batch_size, beam_width}).copy_(sorted_tokens);
  out_token_index.view({batch_size, beam_width})
      .copy_((sorted_parent + batch_offsets).to(torch::kInt32));
  out_log_probs.view({batch_size, beam_width}).copy_(sorted_probs);
  out_beam_count_prefix_sums.view({batch_size, beam_width})
      .copy_((parent_prefix + batch_offsets).to(torch::kInt32));

  const torch::Tensor batch_range =
      torch::arange(batch_size,
                    torch::TensorOptions().dtype(torch::kLong).device(device))
          .unsqueeze(1)
          .expand({batch_size, beam_width});
  out_sequence.slice(/*dim=*/2, /*start=*/0, current_step)
      .copy_(sequence_group.index({TensorIndex(batch_range),
                                   TensorIndex(sorted_parent),
                                   Slice(0, current_step)}));
  out_sequence
      .slice(/*dim=*/2, /*start=*/current_step, /*end=*/current_step + 1)
      .copy_(sorted_tokens.unsqueeze(2));
}

}  // namespace

// ============================================================
// RecWorkerImpl Implementation (base)
// ============================================================

void RecWorkerImpl::RecWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
#if defined(USE_NPU)
  // Without device_capture_lock, ACL graph capture will be interrupted by the
  // synchronization H2D of data update streams asynchronously scheduled by
  // other threads, even if the capture and synchronization streams are not the
  // same, and even if capture_mode is set to
  // ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL.
  // The possible reason is that ACL graph capture may use additional auxiliary
  // streams, and these auxiliary streams might be the same as the
  // asynchronously scheduled data update streams.

  std::optional<std::unique_lock<std::mutex>> lock_guard;
  if (FLAGS_enable_graph) {
    auto& capture_lock =
        ::xllm::npu::DeviceCaptureLock::get_instance().get_lock(
            runtime_.worker.device().index());
    lock_guard.emplace(capture_lock);
  }
#endif
  processed_inputs =
      inputs.to(runtime_.worker.device(), runtime_.worker.dtype());
  auto& input_params = processed_inputs.input_params;
#if defined(USE_NPU)
  if (input_params.swap_blocks.size() > 0 && !FLAGS_enable_block_copy_kernel) {
    auto& swap_blocks = input_params.swap_blocks;

    // collect src and dst indices
    std::vector<int64_t> src_indices, dst_indices;
    src_indices.reserve(swap_blocks.size());
    dst_indices.reserve(swap_blocks.size());

    for (const auto& block : swap_blocks) {
      src_indices.push_back(block.src_block_id);
      dst_indices.push_back(block.dst_block_id);
    }

    // batch select keys and values
    auto src_tensor = torch::tensor(
        src_indices,
        torch::dtype(torch::kLong).device(runtime_.worker.device_));
    auto dst_tensor = torch::tensor(
        dst_indices,
        torch::dtype(torch::kLong).device(runtime_.worker.device_));
    const int64_t num_layers = runtime_.context->get_model_args().n_layers();
    for (int layer_id = 0; layer_id < num_layers; layer_id++) {
      runtime_.worker.kv_caches_[layer_id].swap_blocks(src_tensor, dst_tensor);
    }
  }
  if (runtime_.context->get_model_args().enable_mla() &&
      input_params.batch_forward_type.is_chunked_prefill()) {
    runtime_.worker.prepare_mla_prefixcache_inputs(input_params);
  }

  if (!runtime_.context->get_parallel_args().mapping_data().empty() &&
      (runtime_.context->get_parallel_args().dp_size() > 1 ||
       runtime_.context->get_parallel_args().ep_size() > 1)) {
    torch::Tensor token_size_per_dp_group =
        torch::tensor(processed_inputs.input_params.dp_global_token_nums,
                      torch::TensorOptions()
                          .device(torch::kCPU)
                          .dtype(torch::kInt32)
                          .pinned_memory(true));
    bool is_prefill =
        processed_inputs.input_params.batch_forward_type.is_prefill();
    DpEpPadding dp_ep_padding(
        token_size_per_dp_group,
        runtime_.context->get_model_args().num_experts_per_tok(),
        runtime_.context->get_parallel_args().mapping_data(),
        runtime_.worker.device(),
        runtime_.worker.dtype(),
        is_prefill);
    processed_inputs.input_params.dp_ep_padding_data = dp_ep_padding.build();
  }
#endif
}

ForwardInput RecWorkerImpl::RecWorkPipeline::prepare_inputs(Batch& batch) {
  return runtime_.worker.WorkerImpl::prepare_inputs(batch);
}

std::optional<ForwardOutput> RecWorkerImpl::RecWorkPipeline::step(
    const ForwardInput& input) {
  Timer timer;
  auto& sampling_params = input.sampling_params;

  std::vector<folly::SemiFuture<bool>> futures;

  if (runtime_.worker.options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
#if defined(USE_NPU)
    std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<NPULayerSynchronizerImpl>(
            runtime_.context->get_model_args().n_layers());
    const_cast<ModelInputParams*>(&(input.input_params))->layer_synchronizer =
        layer_synchronizer;

    futures.emplace_back(
        runtime_.worker.kv_cache_transfer_->push_kv_blocks_async(
            input.transfer_kv_infos,
            runtime_.context->get_parallel_args(),
            layer_synchronizer,
            runtime_.worker.is_spec_draft_));
#endif
  }

  if (FLAGS_enable_eplb) {
    runtime_.eplb_executor->eplb_execute(input.eplb_info);
  }

  // temporarily use [0], will be adapted in next pr
  // call model executor forward to get hidden states
  auto model_output = runtime_.executor->forward(input.token_ids,
                                                 input.positions,
                                                 runtime_.worker.kv_caches_,
                                                 input.input_params);
  if (!model_output.hidden_states.defined()) {
    return std::nullopt;
  }

  torch::Tensor logits;
  if (sampling_params.selected_token_idxes.defined()) {
    logits = runtime_.model->logits(model_output.hidden_states,
                                    sampling_params.selected_token_idxes);
  }

  ForwardOutput output;
  if (FLAGS_enable_eplb) {
    output.expert_load_data = runtime_.expert_load_data;
    output.prepared_layer_id = runtime_.eplb_executor->get_ready_layer_id();
    if (output.prepared_layer_id != -1) {
      runtime_.eplb_executor->reset_ready_layer_id();
    }
  }

  if (!runtime_.worker.driver_ && !runtime_.worker.dp_driver_ &&
      !runtime_.worker.options_.enable_speculative_decode()) {
    auto ret = runtime_.stream->synchronize();
    // in p-d disaggregation scene, all micro batches should be in same
    // prefill/decode stage, so, to judge transfer_kv_infos.empty,
    if (runtime_.worker.options_.kv_cache_transfer_mode() == "PUSH" &&
        !input.transfer_kv_infos.empty()) {
      auto results =
          folly::collectAll(futures).within(std::chrono::seconds(60)).get();
      for (const auto& result : results) {
        // TODO: Add error handling
        if (!result.value()) {
          LOG(ERROR) << "kv_cache_transfer_ failed";
          break;
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
  if (sampling_params.selected_token_idxes.defined()) {
    sample_output = runtime_.worker.sampler_->forward(logits, sampling_params);
    output.logits = logits;

    // beam search kernel
    BeamSearchOutput beam_search_output;
    if (sampling_params.use_beam_search && input.acc_logprob.defined() &&
        input.acc_logprob.numel() > 0) {
      beam_search_output =
          runtime_.worker.beam_searcher_->forward(input.acc_logprob,
                                                  sample_output.top_tokens,
                                                  sample_output.top_logprobs);
    }

    // set sample output to output
    output.sample_output = sample_output;
    // carry over the sampling params
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
    // set beam search output to output
    output.beam_search_output = beam_search_output;
  }

  if (runtime_.worker.options_.enable_speculative_decode()) {
    if (!input.input_params.batch_forward_type.is_decode() &&
        !runtime_.worker.is_spec_draft_) {
      output.sample_output.embeddings = model_output.hidden_states;
    } else if (sampling_params.selected_token_idxes.defined()) {
      auto embeddings = model_output.hidden_states.index_select(
          /*dim=*/0, sampling_params.selected_token_idxes);
      output.sample_output.embeddings = embeddings;
    }
  }

  auto ret = runtime_.stream->synchronize();

  if (runtime_.worker.options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
    auto results =
        folly::collectAll(futures).within(std::chrono::seconds(60)).get();
    for (const auto& result : results) {
      // TODO: Add error handling
      if (!result.value()) {
        LOG(ERROR) << "kv_cache_transfer_ failed";
        break;
      }
    }
  }

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      runtime_.worker.device_.index());

  return output;
}

void RecWorkerImpl::LlmRecWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  RecWorkPipeline::prepare_work_before_execute(inputs, processed_inputs);

  runtime_.worker.prepare_multi_modal_data(processed_inputs);
}

RecWorkerImpl::OneRecWorkPipeline::OneRecWorkPipeline(
    RecPipelineRuntime& runtime,
    RecPipelineType pipeline_type)
    : RecWorkPipeline(runtime),
      rec_sampler_(std::make_unique<RecSampler>(pipeline_type)),
      filter_mask_threadpool_(std::make_unique<ThreadPool>(1)) {
  if (!FLAGS_enable_constrained_decoding) {
    return;
  }

  auto* vocab_dict = get_onerec_vocab_dict(runtime_.worker.model_weights_path_);
  CHECK(vocab_dict != nullptr)
      << "Failed to get RecVocabDict for OneRec constrained decoding, "
      << "model_path=" << runtime_.worker.model_weights_path_;

  const int32_t vocab_size =
      static_cast<int32_t>(runtime_.context->get_model_args().vocab_size());
  constrained_decoding_ =
      std::make_unique<RecConstrainedDecoding>(vocab_dict,
                                               vocab_size,
                                               runtime_.worker.dtype(),
                                               runtime_.worker.device(),
                                               /*use_gen_threadpool=*/false);
  CHECK(constrained_decoding_->build_mask_cache())
      << "Failed to build OneRec constrained decoding cache, vocab_size="
      << vocab_size;
}

ForwardInput RecWorkerImpl::OneRecWorkPipeline::prepare_inputs(Batch& batch) {
  MPMCThreadPool* thread_pool =
      runtime_.worker.input_builder_thread_pool_
          ? runtime_.worker.input_builder_thread_pool_.get()
          : nullptr;

  return batch.prepare_rec_forward_input(
      runtime_.worker.options_.num_decoding_tokens(),
      /*min_decoding_batch_size=*/0,
      runtime_.context->get_model_args(),
      thread_pool);
}

void RecWorkerImpl::OneRecWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  RecWorkPipeline::prepare_work_before_execute(inputs, processed_inputs);

  auto& onerec_params = processed_inputs.input_params.mutable_onerec_params();
  if (!onerec_params.decoder_context_embedding.defined()) {
    return;
  }

  if (onerec_params.decoder_context_embedding.scalar_type() ==
      runtime_.worker.dtype()) {
    return;
  }

  onerec_params.decoder_context_embedding =
      onerec_params.decoder_context_embedding.to(runtime_.worker.dtype());
}

folly::SemiFuture<torch::Tensor>
RecWorkerImpl::OneRecWorkPipeline::prepare_filter_mask_async(
    const std::vector<std::vector<int32_t>>& generated_tokens) {
  folly::Promise<torch::Tensor> promise;
  auto future = promise.getSemiFuture();

  if (!constrained_decoding_ || !filter_mask_threadpool_ ||
      generated_tokens.empty()) {
    promise.setValue(torch::Tensor());
    return future;
  }

  filter_mask_threadpool_->schedule(
      [this, generated_tokens, promise = std::move(promise)]() mutable {
        try {
          auto filter_mask =
              constrained_decoding_->generate_mask(generated_tokens);
          promise.setValue(filter_mask);
        } catch (const std::exception& e) {
          const int32_t batch = static_cast<int32_t>(generated_tokens.size());
          const int32_t seq =
              batch > 0 ? static_cast<int32_t>(generated_tokens[0].size()) : 0;
          LOG(ERROR) << "Failed to generate OneRec filter mask, batch=" << batch
                     << ", seq=" << seq << ", error=" << e.what();
          promise.setValue(torch::Tensor());
        } catch (...) {
          const int32_t batch = static_cast<int32_t>(generated_tokens.size());
          const int32_t seq =
              batch > 0 ? static_cast<int32_t>(generated_tokens[0].size()) : 0;
          LOG(ERROR) << "Failed to generate OneRec filter mask, batch=" << batch
                     << ", seq=" << seq << ", error=unknown";
          promise.setValue(torch::Tensor());
        }
      });

  return future;
}

std::optional<ForwardOutput> RecWorkerImpl::OneRecWorkPipeline::step(
    const ForwardInput& input) {
  Timer timer;
  runtime_.worker.device_.set_device();

  ForwardInput& mutable_input = const_cast<ForwardInput&>(input);
  const auto& sampling_params = mutable_input.sampling_params;

  const auto* onerec_params = mutable_input.input_params.onerec_params();
  CHECK(onerec_params != nullptr) << "OneRec requires rec_params.";

  const OneRecModelInputParams& rec_params = *onerec_params;
  OneRecModelInputParams& mutable_onerec_params =
      mutable_input.input_params.mutable_onerec_params();
  const bool has_decoder_context =
      rec_params.decoder_context_embedding.defined();
  const bool has_encoder_context =
      rec_params.has_encoder_output || has_decoder_context;
  const bool has_encoder_output = rec_params.has_encoder_output;
  auto run_onerec_forward = [&](const torch::Tensor& token_ids,
                                const torch::Tensor& positions,
                                bool is_encoder_forward,
                                bool forward_has_encoder_output,
                                bool is_hybrid_mode) {
    mutable_onerec_params.is_encoder_forward = is_encoder_forward;
    mutable_onerec_params.has_encoder_output = forward_has_encoder_output;
    mutable_onerec_params.is_hybrid_mode = is_hybrid_mode;
    return runtime_.executor->forward(token_ids,
                                      positions,
                                      runtime_.worker.kv_caches_,
                                      mutable_input.input_params);
  };
  std::optional<folly::SemiFuture<torch::Tensor>> filter_mask_future;
  if ((runtime_.worker.driver_ || runtime_.worker.dp_driver_) &&
      FLAGS_enable_constrained_decoding && constrained_decoding_ != nullptr &&
      sampling_params.selected_token_idxes.defined()) {
    filter_mask_future = prepare_filter_mask_async(rec_params.generated_tokens);
  }

  torch::Tensor hidden_states;
  if (rec_params.rec_stage == OneRecModelInputParams::RecStage::PREFILL) {
    if (!rec_params.is_first_prefill) {
      if (!has_encoder_context) {
        LOG(ERROR) << "OneRec prefill requires encoder context.";
        return std::nullopt;
      }
      auto model_output = run_onerec_forward(mutable_input.token_ids,
                                             mutable_input.positions,
                                             /*is_encoder_forward=*/false,
                                             /*forward_has_encoder_output=*/
                                             has_encoder_output,
                                             /*is_hybrid_mode=*/false);
      hidden_states = model_output.hidden_states;
    } else {
      const bool has_sparse_embedding =
          rec_params.encoder_sparse_embedding.defined();
      const bool has_encoder_tokens = rec_params.encoder_token_ids.defined() &&
                                      rec_params.encoder_positions.defined();

      if (!has_sparse_embedding && !has_encoder_tokens) {
        LOG(ERROR) << "OneRec first prefill requires encoder inputs.";
        return std::nullopt;
      }

      torch::Tensor encoder_tokens;
      if (has_sparse_embedding) {
        encoder_tokens = rec_params.encoder_sparse_embedding;
      } else {
        encoder_tokens = rec_params.encoder_token_ids;
      }

      auto encoder_output =
          run_onerec_forward(encoder_tokens,
                             rec_params.encoder_positions,
                             /*is_encoder_forward=*/true,
                             /*forward_has_encoder_output=*/
                             has_encoder_output,
                             /*is_hybrid_mode=*/has_sparse_embedding);

      const bool decoder_has_encoder_output =
          encoder_output.hidden_states.defined();
      auto model_output = run_onerec_forward(mutable_input.token_ids,
                                             mutable_input.positions,
                                             /*is_encoder_forward=*/false,
                                             /*forward_has_encoder_output=*/
                                             decoder_has_encoder_output,
                                             /*is_hybrid_mode=*/false);
      hidden_states = model_output.hidden_states;
    }
  } else {
    if (!has_encoder_context) {
      LOG(ERROR) << "OneRec decode requires encoder context.";
      return std::nullopt;
    }
    auto model_output =
        run_onerec_forward(mutable_input.token_ids,
                           mutable_input.positions,
                           /*is_encoder_forward=*/false,
                           /*forward_has_encoder_output=*/has_encoder_output,
                           /*is_hybrid_mode=*/false);
    hidden_states = model_output.hidden_states;
  }

  if (!hidden_states.defined()) {
    return std::nullopt;
  }

  if (!runtime_.worker.driver_ && !runtime_.worker.dp_driver_ &&
      !runtime_.worker.options_.enable_speculative_decode()) {
    runtime_.stream->synchronize();
    COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
    DeviceMonitor::get_instance().update_active_activation_memory(
        runtime_.worker.device_.index());
    return std::nullopt;
  }

  torch::Tensor logits;
  if (sampling_params.selected_token_idxes.defined()) {
    logits = runtime_.model->logits(hidden_states,
                                    sampling_params.selected_token_idxes);
  }

  ForwardOutput output;

  if (sampling_params.selected_token_idxes.defined()) {
    torch::Tensor filter_mask;
    if (filter_mask_future.has_value()) {
      filter_mask = std::move(filter_mask_future.value()).get();
    }
    auto sample_output =
        rec_sampler_->forward(logits, sampling_params, filter_mask);
    output.logits = logits;
    output.sample_output = sample_output;
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
  }

  runtime_.stream->synchronize();
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      runtime_.worker.device_.index());

  return output;
}

RecWorkerImpl::OneRecXAttentionWorkPipeline::OneRecXAttentionWorkPipeline(
    RecPipelineRuntime& runtime)
    : RecWorkPipeline(runtime),
      rec_sampler_(std::make_unique<RecSampler>(
          RecPipelineType::kOneRecXAttentionPipeline)),
      filter_mask_threadpool_(std::make_unique<ThreadPool>(1)) {
  max_seqs_per_batch_ = runtime_.worker.options_.max_seqs_per_batch();
  beam_width_ = std::max<int32_t>(1, runtime_.worker.options_.beam_width());
  max_decode_step_ =
      std::max<int32_t>(0, get_rec_multi_round_decode_rounds() - 1);
  allocate_unshared_kv_caches();

  if (!FLAGS_enable_constrained_decoding) {
    return;
  }
  auto* vocab_dict = get_onerec_vocab_dict(runtime_.worker.model_weights_path_);
  CHECK(vocab_dict != nullptr)
      << "Failed to get RecVocabDict for OneRec xattention constrained "
         "decoding, model_path="
      << runtime_.worker.model_weights_path_;

  const int32_t vocab_size =
      static_cast<int32_t>(runtime_.context->get_model_args().vocab_size());
#if defined(USE_NPU)
  initialize_constraint_device_tensors(
      vocab_dict->build_constraint_tables(vocab_size));
#endif
  constrained_decoding_ =
      std::make_unique<RecConstrainedDecoding>(vocab_dict,
                                               vocab_size,
                                               runtime_.worker.dtype(),
                                               runtime_.worker.device(),
                                               /*use_gen_threadpool=*/false);
  CHECK(constrained_decoding_->build_mask_cache())
      << "Failed to build OneRec xattention constrained decoding cache, "
      << "vocab_size=" << vocab_size;
}

void RecWorkerImpl::OneRecXAttentionWorkPipeline::
    initialize_constraint_device_tensors(const RecConstraintTables& tables) {
#if defined(USE_NPU)
  CHECK_GT(tables.vocab_size, 0);
  CHECK(!tables.first_token_ids.empty())
      << "OneRec constrained decoding requires non-empty first token table.";
  CHECK_EQ(tables.prefix1_pair_keys.size(), tables.prefix1_values.size());
  CHECK_EQ(tables.prefix2_value_offsets.size(),
           tables.prefix1_values.size() + 1);

  const torch::Device& device = runtime_.worker.device();
  constraint_device_tensors_.first_token_ids =
      int32_vector_to_device_tensor(tables.first_token_ids, device);
  constraint_device_tensors_.prefix1_offsets =
      int32_vector_to_device_tensor(tables.prefix1_offsets, device);
  constraint_device_tensors_.prefix1_values =
      int32_vector_to_device_tensor(tables.prefix1_values, device);
  constraint_device_tensors_.prefix1_pair_keys =
      int64_vector_to_device_tensor(tables.prefix1_pair_keys, device);
  constraint_device_tensors_.prefix2_value_offsets =
      int32_vector_to_device_tensor(tables.prefix2_value_offsets, device);
  constraint_device_tensors_.prefix2_values =
      int32_vector_to_device_tensor(tables.prefix2_values, device);
  constraint_device_tensors_.max_prefix1_degree = tables.max_prefix1_degree;
  constraint_device_tensors_.max_prefix2_degree = tables.max_prefix2_degree;
  constraint_device_tensors_.initialized = true;

  LOG(INFO) << "Build OneRec xattention constraint device tables, "
            << "first_tokens=" << tables.first_token_ids.size()
            << ", prefix1_edges=" << tables.prefix1_values.size()
            << ", prefix2_edges=" << tables.prefix2_values.size()
            << ", max_prefix1_degree=" << tables.max_prefix1_degree
            << ", max_prefix2_degree=" << tables.max_prefix2_degree;
#else
  UNUSED_PARAMETER(tables);
#endif
}

bool RecWorkerImpl::OneRecXAttentionWorkPipeline::can_use_device_constraints(
    const SamplingParameters& sampling_params,
    int32_t current_step,
    int32_t beam_width) const {
#if defined(USE_NPU)
  return FLAGS_enable_constrained_decoding &&
         constraint_device_tensors_.initialized && current_step >= 0 &&
         current_step < REC_TOKEN_SIZE && beam_width > 0 &&
         sampling_params.selected_token_idxes.defined() &&
         sampling_params.sample_idxes.defined() &&
         sampling_params.selected_token_idxes.numel() ==
             sampling_params.sample_idxes.numel() &&
         sampling_params.do_sample.defined() &&
         sampling_params.all_greedy_sample &&
         !sampling_params.all_random_sample &&
         sampling_params.use_beam_search && sampling_params.logprobs &&
         sampling_params.max_top_logprobs > 0 &&
         sampling_params.max_top_logprobs == beam_width &&
         !sampling_params.frequency_penalties.defined() &&
         !sampling_params.presence_penalties.defined() &&
         !sampling_params.repetition_penalties.defined() &&
         !sampling_params.top_k.defined() && !sampling_params.top_p.defined();
#else
  UNUSED_PARAMETER(sampling_params);
  UNUSED_PARAMETER(current_step);
  UNUSED_PARAMETER(beam_width);
  return false;
#endif
}

SampleOutput
RecWorkerImpl::OneRecXAttentionWorkPipeline::sample_with_device_constraints(
    torch::Tensor& logits,
    const SamplingParameters& sampling_params,
    const torch::Tensor& sequence_group,
    int32_t current_step) const {
  SampleOutput output;
#if defined(USE_NPU)
  std::tie(output.top_tokens, output.top_logprobs) =
      xllm::kernel::npu::rec_constrained_topk(
          logits,
          sequence_group,
          constraint_device_tensors_.first_token_ids,
          constraint_device_tensors_.prefix1_offsets,
          constraint_device_tensors_.prefix1_values,
          constraint_device_tensors_.prefix1_pair_keys,
          constraint_device_tensors_.prefix2_value_offsets,
          constraint_device_tensors_.prefix2_values,
          sampling_params.temperatures,
          static_cast<int64_t>(current_step),
          sampling_params.max_top_logprobs,
          constraint_device_tensors_.max_prefix1_degree,
          constraint_device_tensors_.max_prefix2_degree);
  output.next_tokens =
      output.top_tokens.select(/*dim=*/1, /*index=*/0).to(torch::kLong);
  output.logprobs =
      output.top_logprobs.select(/*dim=*/1, /*index=*/0).contiguous();
  return output;
#else
  UNUSED_PARAMETER(logits);
  UNUSED_PARAMETER(sampling_params);
  UNUSED_PARAMETER(sequence_group);
  UNUSED_PARAMETER(current_step);
  LOG(FATAL) << "OneRec xattention device constraints require USE_NPU.";
  return output;
#endif
}

void RecWorkerImpl::OneRecXAttentionWorkPipeline::
    allocate_unshared_kv_caches() {
  if (max_seqs_per_batch_ <= 0 || beam_width_ <= 0 || max_decode_step_ <= 0) {
    return;
  }

  const auto& args = runtime_.context->get_model_args();
  const auto& parallel_args = runtime_.context->get_parallel_args();
  const int32_t num_layers = static_cast<int32_t>(args.n_layers());
  const int64_t decoder_kv_heads = args.decoder_n_kv_heads().value_or(
      args.n_kv_heads().value_or(args.decoder_n_heads()));
  const int64_t local_kv_heads =
      decoder_kv_heads / std::max<int64_t>(parallel_args.world_size(), 1);
  const int64_t head_dim = args.decoder_head_dim();
  auto cache_options = torch::TensorOptions()
                           .dtype(runtime_.worker.dtype())
                           .device(runtime_.worker.device());

  cached_unshared_k_caches_.resize(num_layers);
  cached_unshared_v_caches_.resize(num_layers);
  for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    cached_unshared_k_caches_[layer_id] =
        torch::zeros({static_cast<int64_t>(max_seqs_per_batch_),
                      static_cast<int64_t>(beam_width_),
                      local_kv_heads,
                      static_cast<int64_t>(max_decode_step_),
                      head_dim},
                     cache_options);
    cached_unshared_v_caches_[layer_id] =
        torch::zeros({static_cast<int64_t>(max_seqs_per_batch_),
                      static_cast<int64_t>(beam_width_),
                      local_kv_heads,
                      static_cast<int64_t>(max_decode_step_),
                      head_dim},
                     cache_options);
  }
}

void RecWorkerImpl::OneRecXAttentionWorkPipeline::
    prepare_unshared_kv_caches_for_input(
        const ForwardInput& inputs,
        OneRecXAttentionParams& onerec_params) {
  const int32_t request_beam_width =
      inputs.step_meta() != nullptr
          ? std::max<int32_t>(1, inputs.step_meta()->beam_width)
          : std::max<int32_t>(1,
                              onerec_params.group_width > 0
                                  ? onerec_params.group_width
                                  : runtime_.worker.options_.beam_width());
  if (request_beam_width > beam_width_ || cached_unshared_k_caches_.empty() ||
      cached_unshared_v_caches_.empty()) {
    beam_width_ = request_beam_width;
    allocate_unshared_kv_caches();
  }
  if (cached_unshared_k_caches_.empty() || cached_unshared_v_caches_.empty()) {
    return;
  }

  const int32_t batch_size =
      inputs.step_meta() != nullptr
          ? std::max<int32_t>(1, inputs.step_meta()->batch_size)
          : std::max<int32_t>(1, onerec_params.bs);
  onerec_params.unshared_k_caches.clear();
  onerec_params.unshared_v_caches.clear();
  onerec_params.unshared_k_caches.reserve(cached_unshared_k_caches_.size());
  onerec_params.unshared_v_caches.reserve(cached_unshared_v_caches_.size());
  for (size_t layer_id = 0; layer_id < cached_unshared_k_caches_.size();
       ++layer_id) {
    auto unshared_k = cached_unshared_k_caches_[layer_id]
                          .slice(0, 0, batch_size)
                          .slice(1, 0, request_beam_width);
    auto unshared_v = cached_unshared_v_caches_[layer_id]
                          .slice(0, 0, batch_size)
                          .slice(1, 0, request_beam_width);
    unshared_k.zero_();
    unshared_v.zero_();
    onerec_params.unshared_k_caches.emplace_back(std::move(unshared_k));
    onerec_params.unshared_v_caches.emplace_back(std::move(unshared_v));
  }
}

void RecWorkerImpl::OneRecXAttentionWorkPipeline::execute_cache_select(
    const torch::Tensor& out_token_index,
    const torch::Tensor& out_beam_count_prefix_sums,
    OneRecXAttentionParams& onerec_params,
    int32_t round,
    int32_t batch_size,
    int32_t beam_width,
    int32_t num_layers) {
  if (round <= 0 || onerec_params.unshared_k_caches.empty() ||
      onerec_params.unshared_v_caches.empty()) {
    return;
  }
#if defined(USE_NPU)
  auto device = runtime_.worker.device();
  auto int32_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto batch_offsets = torch::arange(batch_size, int32_options) * beam_width;
  auto batch_offsets_2d = batch_offsets.unsqueeze(1);

  auto beam_index_global = out_token_index.reshape({batch_size, beam_width});
  auto beam_index_local = beam_index_global - batch_offsets_2d;
  auto group_prefix_global =
      out_beam_count_prefix_sums.reshape({batch_size, beam_width});
  auto group_prefix_local = group_prefix_global - batch_offsets_2d;
  auto block_table = torch::arange(batch_size, int32_options);

  xllm::kernel::npu::select_unshared_kv(
      /*beam_index=*/beam_index_local.reshape({-1}),
      /*x_key_block=*/onerec_params.unshared_k_caches,
      /*x_value_block=*/onerec_params.unshared_v_caches,
      /*block_table=*/block_table,
      /*group_offset=*/group_prefix_local.reshape({-1}),
      /*decode_step=*/static_cast<int64_t>(round),
      /*beam_size=*/beam_width,
      /*layer_num=*/num_layers);
#elif defined(USE_CUDA)
  auto block_table = torch::arange(batch_size,
                                   torch::TensorOptions()
                                       .dtype(torch::kInt32)
                                       .device(runtime_.worker.device()))
                         .view({batch_size, 1});
  xllm::kernel::cuda::cache_select(out_token_index,
                                   onerec_params.unshared_k_caches,
                                   onerec_params.unshared_v_caches,
                                   block_table,
                                   round - 1,
                                   beam_width,
                                   num_layers);
#else
  UNUSED_PARAMETER(out_token_index);
  UNUSED_PARAMETER(out_beam_count_prefix_sums);
  UNUSED_PARAMETER(onerec_params);
  UNUSED_PARAMETER(round);
  UNUSED_PARAMETER(batch_size);
  UNUSED_PARAMETER(beam_width);
  UNUSED_PARAMETER(num_layers);
#endif
}

ForwardInput RecWorkerImpl::OneRecXAttentionWorkPipeline::prepare_inputs(
    Batch& batch) {
  MPMCThreadPool* thread_pool =
      runtime_.worker.input_builder_thread_pool_
          ? runtime_.worker.input_builder_thread_pool_.get()
          : nullptr;

  return batch.prepare_rec_forward_input(
      runtime_.worker.options_.num_decoding_tokens(),
      /*min_decoding_batch_size=*/0,
      runtime_.context->get_model_args(),
      thread_pool);
}

void RecWorkerImpl::OneRecXAttentionWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  const bool trace_stage_timing = enable_onerec_xattention_stage_timing();
  Timer prepare_timer;
  auto log_prepare_timing = [&](const char* stage_name) {
    if (!trace_stage_timing) {
      return;
    }
    runtime_.stream->synchronize();
    LOG(INFO) << "OneRec xattention prepare timing, stage=" << stage_name
              << ", elapsed_us=" << prepare_timer.elapsed_microseconds();
    prepare_timer.reset();
  };
  RecWorkPipeline::prepare_work_before_execute(inputs, processed_inputs);
  log_prepare_timing("base_to_device");

#if defined(USE_NPU)
  if (enable_onerec_selected_token_cpu_check()) {
    auto validate_selected_token_idxes_roundtrip =
        [](const torch::Tensor& host_tensor,
           const torch::Tensor& device_tensor,
           const char* tensor_name) {
          if (!host_tensor.defined() || !device_tensor.defined()) {
            return;
          }
          auto host_cpu = host_tensor.to(torch::kCPU, /*non_blocking=*/false);
          auto device_cpu =
              device_tensor.to(torch::kCPU, /*non_blocking=*/false);
          CHECK(torch::equal(host_cpu, device_cpu))
              << "OneRec xattention " << tensor_name
              << " changed during H2D round-trip, host=" << host_cpu
              << ", device_roundtrip=" << device_cpu;
        };
    validate_selected_token_idxes_roundtrip(
        inputs.sampling_params.selected_token_idxes,
        processed_inputs.sampling_params.selected_token_idxes,
        "sampling_params.selected_token_idxes");
    validate_selected_token_idxes_roundtrip(
        inputs.decoder_sampling_params.selected_token_idxes,
        processed_inputs.decoder_sampling_params.selected_token_idxes,
        "decoder_sampling_params.selected_token_idxes");
  }
#endif

  auto& onerec_params =
      processed_inputs.input_params.mutable_onerec_xattention_params();
  const auto& args = runtime_.context->get_model_args();
  const auto& parallel_args = runtime_.context->get_parallel_args();
  const int64_t decoder_kv_heads = args.decoder_n_kv_heads().value_or(
      args.n_kv_heads().value_or(args.decoder_n_heads()));
  const int64_t local_kv_heads =
      decoder_kv_heads / std::max<int64_t>(parallel_args.world_size(), 1);
  const int64_t head_dim = args.decoder_head_dim();
  const int64_t batch_size = std::max<int64_t>(
      onerec_params.bs > 0 ? onerec_params.bs
                           : inputs.input_params.num_sequences,
      1);
  int64_t shared_kv_tokens = 0;
  if (onerec_params.decoder_context_embedding.defined()) {
    const int64_t hidden_size =
        std::max<int64_t>(onerec_params.decoder_context_embedding.size(-1), 1);
    shared_kv_tokens =
        onerec_params.decoder_context_embedding.numel() / hidden_size;
  } else if (processed_inputs.token_ids.defined()) {
    shared_kv_tokens = processed_inputs.token_ids.numel();
  }
  if (shared_kv_tokens <= 0) {
    shared_kv_tokens =
        batch_size *
        std::max<int64_t>(processed_inputs.input_params.q_max_seq_len, 1);
  }
  const int32_t decoder_layers = static_cast<int32_t>(args.n_layers());
  auto fp_options = torch::TensorOptions()
                        .dtype(runtime_.worker.dtype())
                        .device(runtime_.worker.device());

  onerec_params.shared_k_caches.clear();
  onerec_params.shared_v_caches.clear();
  onerec_params.shared_k_caches.reserve(decoder_layers);
  onerec_params.shared_v_caches.reserve(decoder_layers);
  for (int32_t layer_id = 0; layer_id < decoder_layers; ++layer_id) {
    onerec_params.shared_k_caches.emplace_back(
        torch::zeros({shared_kv_tokens,
                      std::max<int64_t>(local_kv_heads, 1),
                      std::max<int64_t>(head_dim, 1)},
                     fp_options));
    onerec_params.shared_v_caches.emplace_back(
        torch::zeros({shared_kv_tokens,
                      std::max<int64_t>(local_kv_heads, 1),
                      std::max<int64_t>(head_dim, 1)},
                     fp_options));
  }
  prepare_unshared_kv_caches_for_input(inputs, onerec_params);
  processed_inputs.input_params.block_tables =
      torch::arange(batch_size,
                    torch::TensorOptions()
                        .dtype(torch::kInt32)
                        .device(runtime_.worker.device()));
  log_prepare_timing("cache_prepare");

  const int32_t beam_width =
      inputs.step_meta() != nullptr
          ? std::max<int32_t>(1, inputs.step_meta()->beam_width)
          : std::max<int32_t>(1, runtime_.worker.options_.beam_width());
  const auto int_options = torch::TensorOptions()
                               .dtype(torch::kInt32)
                               .device(runtime_.worker.device());
  onerec_params.beam_width_tensor = torch::tensor({beam_width}, int_options);
  onerec_params.current_round_tensor =
      torch::tensor({get_onerec_decode_round(onerec_params)}, int_options);
  if (enable_onerec_selected_token_cpu_check() &&
      processed_inputs.sampling_params.selected_token_idxes.defined()) {
    onerec_params.debug_selected_token_idxes =
        processed_inputs.sampling_params.selected_token_idxes;
    auto selected_cpu = inputs.sampling_params.selected_token_idxes.to(
        torch::kCPU, /*non_blocking=*/false);
    auto selected_cpu_i64 = selected_cpu.to(torch::kInt64).contiguous();
    const int64_t* ptr = selected_cpu_i64.data_ptr<int64_t>();
    onerec_params.debug_selected_token_idxes_expected.assign(
        ptr, ptr + selected_cpu_i64.numel());
  } else {
    onerec_params.debug_selected_token_idxes = torch::Tensor();
    onerec_params.debug_selected_token_idxes_expected.clear();
  }

  if (!onerec_params.decoder_context_embedding.defined()) {
    log_prepare_timing("metadata_prepare");
    return;
  }

  if (onerec_params.decoder_context_embedding.scalar_type() ==
      runtime_.worker.dtype()) {
    log_prepare_timing("metadata_prepare");
    return;
  }

  onerec_params.decoder_context_embedding =
      onerec_params.decoder_context_embedding.to(runtime_.worker.dtype());
  log_prepare_timing("metadata_prepare");
}

folly::SemiFuture<torch::Tensor>
RecWorkerImpl::OneRecXAttentionWorkPipeline::prepare_filter_mask_async(
    const std::vector<std::vector<int32_t>>& generated_tokens) {
  folly::Promise<torch::Tensor> promise;
  auto future = promise.getSemiFuture();

  if (!constrained_decoding_ || !filter_mask_threadpool_ ||
      generated_tokens.empty()) {
    promise.setValue(torch::Tensor());
    return future;
  }

  filter_mask_threadpool_->schedule(
      [this, generated_tokens, promise = std::move(promise)]() mutable {
        try {
          auto filter_mask =
              constrained_decoding_->generate_mask(generated_tokens);
          promise.setValue(filter_mask);
        } catch (const std::exception& e) {
          const int32_t batch = static_cast<int32_t>(generated_tokens.size());
          const int32_t seq =
              batch > 0 ? static_cast<int32_t>(generated_tokens[0].size()) : 0;
          LOG(ERROR) << "Failed to generate OneRec xattention filter mask, "
                     << "batch=" << batch << ", seq=" << seq
                     << ", error=" << e.what();
          promise.setValue(torch::Tensor());
        } catch (...) {
          const int32_t batch = static_cast<int32_t>(generated_tokens.size());
          const int32_t seq =
              batch > 0 ? static_cast<int32_t>(generated_tokens[0].size()) : 0;
          LOG(ERROR) << "Failed to generate OneRec xattention filter mask, "
                     << "batch=" << batch << ", seq=" << seq
                     << ", error=unknown";
          promise.setValue(torch::Tensor());
        }
      });

  return future;
}

std::optional<ForwardOutput> RecWorkerImpl::OneRecXAttentionWorkPipeline::step(
    const ForwardInput& input) {
  Timer timer;
  runtime_.worker.device_.set_device();
  const bool trace_stage_timing = enable_onerec_xattention_stage_timing();
  auto log_stage_timing =
      [&](const char* stage_name, int32_t round, Timer& stage_timer) {
        if (!trace_stage_timing) {
          return;
        }
        runtime_.stream->synchronize();
        LOG(INFO) << "OneRec xattention stage timing, stage=" << stage_name
                  << ", round=" << round
                  << ", elapsed_us=" << stage_timer.elapsed_microseconds();
        stage_timer.reset();
      };

  ForwardInput mutable_input = input;
  CHECK(mutable_input.input_params.onerec_xattention_params() != nullptr)
      << "OneRec xattention pipeline requires onerec_xattention_params.";

  struct RoundResult {
    torch::Tensor logits;
    SampleOutput sample_output;
    SamplingParameters sampling_params;
  };

  auto run_single_round =
      [&](const SamplingParameters& sampling_params,
          int32_t current_step,
          const torch::Tensor& sequence_group,
          int32_t request_beam_width) -> std::optional<RoundResult> {
    auto* round_params = mutable_input.input_params.onerec_xattention_params();
    CHECK(round_params != nullptr)
        << "OneRec xattention pipeline requires onerec_xattention_params.";

    const bool has_decoder_context =
        round_params->decoder_context_embedding.defined();
    const bool has_encoder_context =
        round_params->has_encoder_output || has_decoder_context;
    const bool selected_token_cpu_check =
        enable_onerec_selected_token_cpu_check();
    const int32_t stage_round = get_onerec_decode_round(*round_params);
    Timer stage_timer;

    const bool use_device_constraints = can_use_device_constraints(
        sampling_params, current_step, request_beam_width);
#if defined(USE_NPU)
    if (FLAGS_enable_constrained_decoding &&
        sampling_params.selected_token_idxes.defined() &&
        !use_device_constraints) {
      LOG_FIRST_N(WARNING, 8)
          << "Unsupported OneRec xattention constrained decoding request for "
             "device constraints, falling back to CPU mask generation. "
          << "current_step=" << current_step
          << ", beam_width=" << request_beam_width
          << ", max_top_logprobs=" << sampling_params.max_top_logprobs
          << ", use_beam_search=" << sampling_params.use_beam_search
          << ", logprobs=" << sampling_params.logprobs
          << ", selected_token_idxes_numel="
          << (sampling_params.selected_token_idxes.defined()
                  ? sampling_params.selected_token_idxes.numel()
                  : 0)
          << ", sample_idxes_defined=" << sampling_params.sample_idxes.defined()
          << ", sample_idxes_numel="
          << (sampling_params.sample_idxes.defined()
                  ? sampling_params.sample_idxes.numel()
                  : 0)
          << ", do_sample_defined=" << sampling_params.do_sample.defined()
          << ", all_greedy_sample=" << sampling_params.all_greedy_sample
          << ", all_random_sample=" << sampling_params.all_random_sample
          << ", has_frequency_penalties="
          << sampling_params.frequency_penalties.defined()
          << ", has_presence_penalties="
          << sampling_params.presence_penalties.defined()
          << ", has_repetition_penalties="
          << sampling_params.repetition_penalties.defined()
          << ", has_top_k=" << sampling_params.top_k.defined()
          << ", has_top_p=" << sampling_params.top_p.defined()
          << ", constraint_tables_initialized="
          << constraint_device_tensors_.initialized;
    }
#endif

    std::optional<folly::SemiFuture<torch::Tensor>> filter_mask_future;
    if ((runtime_.worker.driver_ || runtime_.worker.dp_driver_) &&
        FLAGS_enable_constrained_decoding && constrained_decoding_ != nullptr &&
        sampling_params.selected_token_idxes.defined() &&
        !use_device_constraints) {
      filter_mask_future =
          prepare_filter_mask_async(round_params->generated_tokens);
    }

    torch::Tensor selected_token_idxes_for_logits;
    torch::Tensor selected_token_idxes_before_cpu;
    if (sampling_params.selected_token_idxes.defined()) {
#if defined(USE_NPU)
      selected_token_idxes_for_logits =
          sampling_params.selected_token_idxes.to(torch::kInt64).contiguous();
      if (selected_token_cpu_check) {
        selected_token_idxes_before_cpu = selected_token_idxes_for_logits.to(
            torch::kCPU, /*non_blocking=*/false);
        if (selected_token_idxes_before_cpu.numel() > 0) {
          const int64_t min_idx_before =
              selected_token_idxes_before_cpu.min().item<int64_t>();
          CHECK_GE(min_idx_before, 0)
              << "OneRec xattention selected_token_idxes already negative "
                 "before model forward, min_idx="
              << min_idx_before
              << ", values=" << selected_token_idxes_before_cpu
              << ", rec_stage=" << static_cast<int32_t>(round_params->rec_stage)
              << ", is_first_prefill=" << round_params->is_first_prefill;
        }
      }
      selected_token_idxes_for_logits = selected_token_idxes_for_logits.clone();
#else
      selected_token_idxes_for_logits = sampling_params.selected_token_idxes;
#endif
    }

#if defined(USE_NPU)
    auto validate_selected_token_idxes_stage = [&](const char* stage_name) {
      if (!selected_token_cpu_check ||
          !sampling_params.selected_token_idxes.defined()) {
        return;
      }
      auto selected_token_idxes_stage_cpu = sampling_params.selected_token_idxes
                                                .to(torch::kCPU,
                                                    /*non_blocking=*/false)
                                                .to(torch::kInt64);
      CHECK(torch::equal(selected_token_idxes_stage_cpu,
                         selected_token_idxes_before_cpu))
          << "OneRec xattention selected_token_idxes changed after "
          << stage_name << ", before=" << selected_token_idxes_before_cpu
          << ", after=" << selected_token_idxes_stage_cpu
          << ", rec_stage=" << static_cast<int32_t>(round_params->rec_stage)
          << ", is_first_prefill=" << round_params->is_first_prefill;
    };
#endif

    torch::Tensor hidden_states;
    if (round_params->rec_stage == OneRecModelInputParams::RecStage::PREFILL) {
      if (!round_params->is_first_prefill) {
        if (!has_encoder_context) {
          LOG(ERROR) << "OneRec xattention prefill requires encoder context.";
          return std::nullopt;
        }
        ModelInputParams decoder_params = mutable_input.input_params;
        auto& decoder_onerec_params =
            decoder_params.mutable_onerec_xattention_params();
        decoder_onerec_params.is_encoder_forward = false;
        decoder_onerec_params.has_encoder_output =
            round_params->has_encoder_output;
        auto model_output =
            runtime_.executor->forward(mutable_input.token_ids,
                                       mutable_input.positions,
                                       runtime_.worker.kv_caches_,
                                       decoder_params);
#if defined(USE_NPU)
        validate_selected_token_idxes_stage("decoder_forward");
#endif
        hidden_states = model_output.hidden_states;
      } else {
        const bool has_sparse_embedding =
            round_params->encoder_sparse_embedding.defined();
        const bool has_encoder_tokens =
            round_params->encoder_token_ids.defined() &&
            round_params->encoder_positions.defined();

        if (!has_sparse_embedding && !has_encoder_tokens) {
          LOG(ERROR) << "OneRec xattention first prefill requires encoder "
                        "inputs.";
          return std::nullopt;
        }

        ModelInputParams encoder_params = mutable_input.input_params;
        auto& encoder_onerec_params =
            encoder_params.mutable_onerec_xattention_params();
        encoder_onerec_params.is_encoder_forward = true;
        encoder_onerec_params.is_hybrid_mode = has_sparse_embedding;

        torch::Tensor encoder_tokens;
        if (has_sparse_embedding) {
          encoder_tokens = round_params->encoder_sparse_embedding;
        } else {
          encoder_onerec_params.is_hybrid_mode = false;
          encoder_tokens = round_params->encoder_token_ids;
        }

        auto encoder_output =
            runtime_.executor->forward(encoder_tokens,
                                       round_params->encoder_positions,
                                       runtime_.worker.kv_caches_,
                                       encoder_params);
#if defined(USE_NPU)
        validate_selected_token_idxes_stage("encoder_forward");
#endif

        ModelInputParams decoder_params = mutable_input.input_params;
        auto& decoder_onerec_params =
            decoder_params.mutable_onerec_xattention_params();
        decoder_onerec_params.is_encoder_forward = false;
        decoder_onerec_params.has_encoder_output =
            encoder_output.hidden_states.defined();
        auto model_output =
            runtime_.executor->forward(mutable_input.token_ids,
                                       mutable_input.positions,
                                       runtime_.worker.kv_caches_,
                                       decoder_params);
#if defined(USE_NPU)
        validate_selected_token_idxes_stage("decoder_forward");
#endif
        hidden_states = model_output.hidden_states;
      }
    } else {
      if (!has_encoder_context) {
        LOG(ERROR) << "OneRec xattention decode requires encoder context.";
        return std::nullopt;
      }
      ModelInputParams decoder_params = mutable_input.input_params;
      auto& decoder_onerec_params =
          decoder_params.mutable_onerec_xattention_params();
      decoder_onerec_params.is_encoder_forward = false;
      decoder_onerec_params.has_encoder_output =
          round_params->has_encoder_output;
      auto model_output = runtime_.executor->forward(mutable_input.token_ids,
                                                     mutable_input.positions,
                                                     runtime_.worker.kv_caches_,
                                                     decoder_params);
#if defined(USE_NPU)
      validate_selected_token_idxes_stage("decode_forward");
#endif
      hidden_states = model_output.hidden_states;
    }

    log_stage_timing("forward", stage_round, stage_timer);
    if (!hidden_states.defined()) {
      return std::nullopt;
    }

    RoundResult result;
    result.sampling_params = sampling_params;
    if (sampling_params.selected_token_idxes.defined()) {
      torch::Tensor selected_token_idxes = selected_token_idxes_for_logits;
#if defined(USE_NPU)
      if (selected_token_cpu_check) {
        auto selected_token_idxes_after_cpu =
            sampling_params.selected_token_idxes
                .to(torch::kCPU,
                    /*non_blocking=*/false)
                .to(torch::kInt64);
        CHECK(torch::equal(selected_token_idxes_after_cpu,
                           selected_token_idxes_before_cpu))
            << "OneRec xattention selected_token_idxes changed during model "
               "forward, before="
            << selected_token_idxes_before_cpu
            << ", after=" << selected_token_idxes_after_cpu
            << ", rec_stage=" << static_cast<int32_t>(round_params->rec_stage)
            << ", is_first_prefill=" << round_params->is_first_prefill;
      }
      if (selected_token_idxes.scalar_type() != torch::kInt64) {
        selected_token_idxes = selected_token_idxes.to(torch::kInt64);
      }
      selected_token_idxes = selected_token_idxes.contiguous();
      if (selected_token_cpu_check) {
        CHECK_EQ(selected_token_idxes_before_cpu.dim(), 1)
            << "OneRec xattention selected_token_idxes must be 1-D, got "
            << selected_token_idxes_before_cpu.dim();
        if (selected_token_idxes_before_cpu.numel() > 0) {
          const int64_t hidden_rows =
              hidden_states.dim() > 0 ? hidden_states.size(0) : 0;
          const int64_t min_idx =
              selected_token_idxes_before_cpu.min().item<int64_t>();
          const int64_t max_idx =
              selected_token_idxes_before_cpu.max().item<int64_t>();
          CHECK_GE(min_idx, 0)
              << "OneRec xattention selected_token_idxes contains negative "
                 "index, min_idx="
              << min_idx << ", max_idx=" << max_idx
              << ", hidden_rows=" << hidden_rows
              << ", rec_stage=" << static_cast<int32_t>(round_params->rec_stage)
              << ", is_first_prefill=" << round_params->is_first_prefill;
          CHECK_LT(max_idx, hidden_rows)
              << "OneRec xattention selected_token_idxes out of range, "
              << "min_idx=" << min_idx << ", max_idx=" << max_idx
              << ", hidden_rows=" << hidden_rows
              << ", rec_stage=" << static_cast<int32_t>(round_params->rec_stage)
              << ", is_first_prefill=" << round_params->is_first_prefill;
        }
      }
#endif
      result.logits =
          runtime_.model->logits(hidden_states, selected_token_idxes);
      log_stage_timing("logits", stage_round, stage_timer);
      torch::Tensor filter_mask;
      if (filter_mask_future.has_value()) {
        filter_mask = std::move(filter_mask_future.value()).get();
      }
      RecSamplingContext sampling_context;
      sampling_context.sequence_group = sequence_group;
      sampling_context.current_step = current_step;
      sampling_context.beam_width = request_beam_width;
      if (use_device_constraints) {
        sampling_context.device_constrained_sampler =
            [this](torch::Tensor& logits,
                   const SamplingParameters& params,
                   const torch::Tensor& sequence_group,
                   int32_t current_step,
                   int32_t beam_width) -> std::optional<SampleOutput> {
          if (!can_use_device_constraints(params, current_step, beam_width)) {
            return std::nullopt;
          }
          return sample_with_device_constraints(
              logits, params, sequence_group, current_step);
        };
      }
      result.sample_output = rec_sampler_->forward(
          result.logits, sampling_params, filter_mask, &sampling_context);
      log_stage_timing("sampler", stage_round, stage_timer);
    }
    return result;
  };

  auto prepare_decode_round_input =
      [&](int32_t round,
          int32_t batch_size,
          int32_t beam_width,
          const std::vector<int32_t>& decode_positions_vec,
          const torch::Tensor& sequence_group) {
        auto& round_params =
            mutable_input.input_params.mutable_onerec_xattention_params();
        const int32_t decode_step = std::max(round - 1, 0);

        round_params.rec_stage = OneRecModelInputParams::RecStage::DECODE;
        round_params.is_first_prefill = false;
        round_params.is_encoder_forward = false;
        if (round_params.current_round_tensor.defined()) {
          round_params.current_round_tensor.fill_(round);
        }
        if (round_params.beam_width_tensor.defined()) {
          round_params.beam_width_tensor.fill_(beam_width);
        }

        if (sequence_group.defined()) {
          mutable_input.token_ids =
              sequence_group.select(/*dim=*/2, /*index=*/decode_step)
                  .contiguous()
                  .reshape({-1});
        }

        round_params.decoder_context_embedding = torch::Tensor();
        round_params.bs = batch_size;
        round_params.group_width = beam_width;
        round_params.seq_len = 1;

        std::vector<int32_t> positions_host;
        positions_host.reserve(static_cast<size_t>(batch_size * beam_width));
        std::vector<int32_t> selected_token_idxes;
        selected_token_idxes.reserve(
            static_cast<size_t>(batch_size * beam_width));
        for (int32_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
          for (int32_t beam_idx = 0; beam_idx < beam_width; ++beam_idx) {
            positions_host.emplace_back(
                decode_positions_vec.at(static_cast<size_t>(seq_idx)) +
                decode_step);
            selected_token_idxes.emplace_back(seq_idx * beam_width + beam_idx);
          }
        }
        auto int_options = torch::TensorOptions()
                               .dtype(torch::kInt32)
                               .device(runtime_.worker.device());
        mutable_input.positions = torch::tensor(positions_host, int_options);
        mutable_input.decoder_sampling_params.selected_token_idxes =
            torch::tensor(selected_token_idxes, int_options);
        mutable_input.decoder_sampling_params.num_return_sequences =
            mutable_input.sampling_params.num_return_sequences;
        mutable_input.input_params.batch_forward_type =
            BatchForwardType::DECODE;
        mutable_input.input_params.num_sequences = batch_size * beam_width;
        mutable_input.input_params.input_embedding = torch::Tensor();
        mutable_input.input_params.attn_metadata = nullptr;
      };

  auto step_meta = mutable_input.step_meta();
  const bool use_multi_round =
      step_meta != nullptr && step_meta->total_round > 1 &&
      step_meta->beam_width > 1 &&
      mutable_input.decoder_sampling_params.selected_token_idxes.defined();

  if (!use_multi_round) {
    const int32_t request_beam_width =
        step_meta != nullptr
            ? std::max<int32_t>(1, step_meta->beam_width)
            : std::max<int32_t>(1, runtime_.worker.options_.beam_width());
    auto result = run_single_round(mutable_input.sampling_params,
                                   /*current_step=*/0,
                                   torch::Tensor(),
                                   request_beam_width);
    if (!result.has_value()) {
      return std::nullopt;
    }

    if (!runtime_.worker.driver_ && !runtime_.worker.dp_driver_ &&
        !runtime_.worker.options_.enable_speculative_decode()) {
      runtime_.stream->synchronize();
      COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
      DeviceMonitor::get_instance().update_active_activation_memory(
          runtime_.worker.device_.index());
      return std::nullopt;
    }

    ForwardOutput output;
    output.logits = result->logits;
    output.sample_output = result->sample_output;
    output.do_sample = result->sampling_params.do_sample;
    output.logprobs = result->sampling_params.logprobs;
    output.max_top_logprobs = result->sampling_params.max_top_logprobs;

    runtime_.stream->synchronize();
    COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
    DeviceMonitor::get_instance().update_active_activation_memory(
        runtime_.worker.device_.index());
    return output;
  }

  const int32_t batch_size = step_meta->batch_size;
  const int32_t beam_width = step_meta->beam_width;
  const int32_t total_rounds = step_meta->total_round;
  OneRecBeamSearchTensors beam_tensors = prepare_onerec_beam_search_tensors(
      batch_size, beam_width, total_rounds, runtime_.worker.device());

  ForwardOutput output;
  torch::Tensor top_tokens;
  torch::Tensor top_logprobs;

  for (int32_t round = 0; round < total_rounds; ++round) {
    if (round > 0) {
      Timer prepare_round_timer;
      prepare_decode_round_input(round,
                                 batch_size,
                                 beam_width,
                                 step_meta->decode_positions_vec,
                                 beam_tensors.sequence_group);
      log_stage_timing(
          "prepare_decode_round_input", round, prepare_round_timer);
    }

    const auto& sampling_params = round == 0
                                      ? mutable_input.sampling_params
                                      : mutable_input.decoder_sampling_params;
    SamplingParameters round_sampling_params = sampling_params;
    const int32_t requested_result_width =
        get_requested_beam_result_width(round_sampling_params, beam_width);
    const bool final_round = round == total_rounds - 1;
    const bool output_logprobs = sampling_params.logprobs;
    const int64_t output_max_top_logprobs = sampling_params.max_top_logprobs;
    if (final_round && requested_result_width != beam_width) {
      round_sampling_params.max_top_logprobs = std::max<int64_t>(
          round_sampling_params.max_top_logprobs, requested_result_width);
      round_sampling_params.logprobs = true;
    }
    auto result = run_single_round(
        round_sampling_params, round, beam_tensors.sequence_group, beam_width);
    if (!result.has_value()) {
      return std::nullopt;
    }
    if (final_round && requested_result_width != beam_width &&
        util::get_bool_env("XLLM_DEBUG_ONEREC_ENGINE_TRACE", false)) {
      LOG(INFO) << "OneRec xattention final round sampling shapes: "
                << "requested_result_width=" << requested_result_width
                << ", beam_width=" << beam_width
                << ", logprobs=" << round_sampling_params.logprobs
                << ", max_top_logprobs="
                << round_sampling_params.max_top_logprobs
                << ", top_tokens_shape="
                << (result->sample_output.top_tokens.defined()
                        ? result->sample_output.top_tokens.sizes()
                        : c10::IntArrayRef{})
                << ", top_logprobs_shape="
                << (result->sample_output.top_logprobs.defined()
                        ? result->sample_output.top_logprobs.sizes()
                        : c10::IntArrayRef{});
    }
    if (!result->sample_output.top_tokens.defined() ||
        !result->sample_output.top_logprobs.defined()) {
      output.do_sample = result->sampling_params.do_sample;
      output.logprobs = result->sampling_params.logprobs;
      output.max_top_logprobs = result->sampling_params.max_top_logprobs;
      continue;
    }

#if defined(USE_NPU)
    Timer beam_timer;
    if (round == 0) {
      top_tokens =
          result->sample_output.top_tokens.to(torch::kInt32).reshape({-1, 1});
      top_logprobs = result->sample_output.top_logprobs.reshape({-1, 1});
      runtime::detail::write_first_round_beam_outputs(
          top_tokens,
          top_logprobs,
          batch_size,
          beam_tensors.out_token_ids,
          beam_tensors.out_log_probs,
          beam_tensors.out_seqgroup);
    } else if (final_round && requested_result_width != beam_width) {
      top_tokens = result->sample_output.top_tokens.to(torch::kInt32);
      top_logprobs = result->sample_output.top_logprobs;
      OneRecBeamSearchOutputTensors final_tensors =
          prepare_onerec_beam_search_output_tensors(batch_size,
                                                    requested_result_width,
                                                    total_rounds,
                                                    runtime_.worker.device());
      if (can_use_beam_search_rec_final_select(
              batch_size, top_tokens, requested_result_width)) {
        xllm::kernel::npu::beam_search_rec(
            /*logprobs=*/beam_tensors.acc_logprob,
            /*top_tokens=*/top_tokens,
            /*top_logprobs=*/top_logprobs,
            /*sequence_group=*/beam_tensors.sequence_group,
            /*current_step=*/static_cast<int64_t>(round),
            /*result_width=*/requested_result_width,
            /*out_token_ids=*/final_tensors.out_token_ids,
            /*out_token_index=*/final_tensors.out_token_index,
            /*out_log_probs=*/final_tensors.out_log_probs,
            /*out_beam_count_prefix_sums=*/
            final_tensors.out_beam_count_prefix_sums,
            /*out_sequence=*/final_tensors.out_seqgroup);
      } else {
        select_final_onerec_beam_results(beam_tensors.acc_logprob,
                                         beam_tensors.sequence_group,
                                         top_tokens,
                                         top_logprobs,
                                         batch_size,
                                         beam_width,
                                         requested_result_width,
                                         total_rounds,
                                         round,
                                         final_tensors);
      }
      beam_tensors.out_token_ids = std::move(final_tensors.out_token_ids);
      beam_tensors.out_token_index = std::move(final_tensors.out_token_index);
      beam_tensors.out_beam_count_prefix_sums =
          std::move(final_tensors.out_beam_count_prefix_sums);
      beam_tensors.out_log_probs = std::move(final_tensors.out_log_probs);
      beam_tensors.out_seqgroup = std::move(final_tensors.out_seqgroup);
    } else {
      top_tokens = result->sample_output.top_tokens.to(torch::kInt32);
      top_logprobs = result->sample_output.top_logprobs;
      xllm::kernel::npu::beam_search_rec(
          /*logprobs=*/beam_tensors.acc_logprob,
          /*top_tokens=*/top_tokens,
          /*top_logprobs=*/top_logprobs,
          /*sequence_group=*/beam_tensors.sequence_group,
          /*current_step=*/static_cast<int64_t>(round),
          /*out_token_ids=*/beam_tensors.out_token_ids,
          /*out_token_index=*/beam_tensors.out_token_index,
          /*out_log_probs=*/beam_tensors.out_log_probs,
          /*out_beam_count_prefix_sums=*/
          beam_tensors.out_beam_count_prefix_sums,
          /*out_sequence=*/beam_tensors.out_seqgroup);
    }
    log_stage_timing("beam_search", round, beam_timer);
#elif defined(USE_CUDA)
    Timer beam_timer;
    if (final_round && requested_result_width != beam_width) {
      top_tokens =
          result->sample_output.top_tokens.to(torch::kInt32)
              .reshape({-1, result->sample_output.top_tokens.size(-1)});
      top_logprobs = result->sample_output.top_logprobs.reshape(
          {-1, result->sample_output.top_logprobs.size(-1)});
      OneRecBeamSearchOutputTensors final_tensors =
          prepare_onerec_beam_search_output_tensors(batch_size,
                                                    requested_result_width,
                                                    total_rounds,
                                                    runtime_.worker.device());
      select_final_onerec_beam_results(beam_tensors.acc_logprob,
                                       beam_tensors.sequence_group,
                                       top_tokens,
                                       top_logprobs,
                                       batch_size,
                                       beam_width,
                                       requested_result_width,
                                       total_rounds,
                                       round,
                                       final_tensors);
      beam_tensors.out_token_ids = std::move(final_tensors.out_token_ids);
      beam_tensors.out_token_index = std::move(final_tensors.out_token_index);
      beam_tensors.out_beam_count_prefix_sums =
          std::move(final_tensors.out_beam_count_prefix_sums);
      beam_tensors.out_log_probs = std::move(final_tensors.out_log_probs);
      beam_tensors.out_seqgroup = std::move(final_tensors.out_seqgroup);
    } else {
      top_tokens =
          result->sample_output.top_tokens.to(torch::kInt32)
              .reshape({-1, result->sample_output.top_tokens.size(-1)});
      top_logprobs = result->sample_output.top_logprobs.reshape(
          {-1, result->sample_output.top_logprobs.size(-1)});
      xllm::kernel::cuda::beam_search(beam_tensors.acc_logprob,
                                      beam_tensors.sequence_group,
                                      top_tokens,
                                      top_logprobs,
                                      beam_tensors.out_log_probs,
                                      beam_tensors.out_token_ids,
                                      beam_tensors.out_token_index,
                                      beam_tensors.out_beam_count_prefix_sums,
                                      beam_tensors.out_seqgroup,
                                      batch_size,
                                      requested_result_width,
                                      round);
    }
    log_stage_timing("beam_search", round, beam_timer);
#else
    LOG(FATAL) << "OneRec xattention beam search requires NPU or CUDA.";
#endif
    std::swap(beam_tensors.sequence_group, beam_tensors.out_seqgroup);
    std::swap(beam_tensors.acc_logprob, beam_tensors.out_log_probs);
    if (round > 0 && round < total_rounds - 1) {
      auto& round_params =
          mutable_input.input_params.mutable_onerec_xattention_params();
      execute_cache_select(
          beam_tensors.out_token_index,
          beam_tensors.out_beam_count_prefix_sums,
          round_params,
          round,
          batch_size,
          beam_width,
          static_cast<int32_t>(runtime_.context->get_model_args().n_layers()));
    }

    if (round == total_rounds - 1) {
      output.do_sample = result->sampling_params.do_sample;
      output.logprobs = output_logprobs;
      output.max_top_logprobs = output_max_top_logprobs;
      output.beam_search_output.src_seq_idxes =
          beam_tensors.out_token_index.reshape({-1});
      output.beam_search_output.out_tokens =
          beam_tensors.out_token_ids.reshape({-1});
      output.beam_search_output.out_logprobs =
          beam_tensors.acc_logprob.reshape({-1});
      output.beam_sequence_group = beam_tensors.sequence_group;
    }
  }

  runtime_.stream->synchronize();
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      runtime_.worker.device_.index());
  return output;
}

// ============================================================
// LlmRecMultiRoundPipeline Implementation (qwen3 with embedding)
// ============================================================

RecWorkerImpl::LlmRecMultiRoundPipeline::LlmRecMultiRoundPipeline(
    RecPipelineRuntime& runtime)
    : RecWorkPipeline(runtime),
      rec_sampler_(std::make_unique<RecSampler>(
          RecPipelineType::kLlmRecMultiRoundPipeline)) {
  max_seqs_per_batch_ = runtime_.worker.options_.max_seqs_per_batch();
  max_tokens_per_batch_ = runtime_.worker.options_.max_tokens_per_batch();
  max_token_per_req_ = max_seqs_per_batch_ > 0
                           ? (max_tokens_per_batch_ / max_seqs_per_batch_)
                           : 0;
  beam_width_ = runtime_.worker.options_.beam_width();

  full_kv_cache_offsets_ = std::make_unique<FullKvCacheOffsets>(this);
  allocate_kv_caches_related();

  static auto shared_control_state = std::make_shared<SharedControlState>();
  shared_control_state_ = shared_control_state;
}

ForwardInput RecWorkerImpl::LlmRecMultiRoundPipeline::prepare_inputs(
    Batch& batch) {
  MPMCThreadPool* thread_pool =
      runtime_.worker.input_builder_thread_pool_
          ? runtime_.worker.input_builder_thread_pool_.get()
          : nullptr;

  return batch.prepare_rec_forward_input(
      runtime_.worker.options_.num_decoding_tokens(),
      /*min_decoding_batch_size=*/0,
      runtime_.context->get_model_args(),
      thread_pool);
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  RecWorkPipeline::prepare_work_before_execute(inputs, processed_inputs);

  runtime_.worker.prepare_multi_modal_data(processed_inputs);

#if defined(USE_NPU) || defined(USE_CUDA)
  prepare_kv_caches_related_for_input(inputs, processed_inputs);
#endif
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::allocate_kv_caches_related() {
  auto dtype = runtime_.worker.dtype();
  auto device = runtime_.worker.device();
  auto kv_cache_options = torch::TensorOptions().dtype(dtype).device(device);
  auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  int32_t num_layers = runtime_.context->get_model_args().n_layers();

  int32_t full_kv_len =
      max_tokens_per_batch_ + max_seqs_per_batch_ * beam_width_ *
                                  (get_rec_multi_round_decode_rounds() - 1);
  int64_t num_kv_heads =
      runtime_.context->get_model_args().n_kv_heads().value_or(
          runtime_.context->get_model_args().n_heads());
  int64_t head_dim = runtime_.context->get_model_args().head_dim();

  cached_full_k_caches_.resize(num_layers);
  cached_full_v_caches_.resize(num_layers);

  for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
#if defined(USE_NPU)
    const int64_t full_kv_elems =
        static_cast<int64_t>(full_kv_len) * num_kv_heads * head_dim;
    auto target_layer_full_k_cache =
        torch::zeros({full_kv_elems}, kv_cache_options);
    auto target_layer_full_v_cache =
        torch::zeros({full_kv_elems}, kv_cache_options);
#else
    auto target_layer_full_k_cache =
        torch::zeros({full_kv_len, num_kv_heads, head_dim}, kv_cache_options);
    auto target_layer_full_v_cache =
        torch::zeros({full_kv_len, num_kv_heads, head_dim}, kv_cache_options);
#endif

    cached_full_k_caches_[layer_id] = target_layer_full_k_cache;
    cached_full_v_caches_[layer_id] = target_layer_full_v_cache;
  }

#if defined(USE_NPU)
  cached_naive_block_table_ = torch::arange(max_seqs_per_batch_, int_options);
#else
  cached_naive_block_table_ =
      torch::arange(max_seqs_per_batch_ * beam_width_, int_options)
          .unsqueeze(1);
#endif
  cached_current_round_tensor_ = torch::zeros({1}, int_options);
  cached_beam_width_tensor_ = torch::zeros({1}, int_options);

  if (FLAGS_enable_xattention_one_stage) {
    return;
  }

  const int64_t num_heads = runtime_.context->get_model_args().n_heads();
  const int64_t max_total_beam =
      static_cast<int64_t>(max_seqs_per_batch_) * beam_width_;
  auto fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  cached_two_stage_shared_lse_ =
      torch::zeros({max_total_beam, num_heads, 1}, fp32_options);
  cached_two_stage_shared_o_ =
      torch::zeros({max_total_beam, num_heads, head_dim}, kv_cache_options);
  cached_two_stage_unshared_lse_ =
      torch::zeros({max_total_beam, num_heads, 1}, fp32_options);
  cached_two_stage_unshared_o_ =
      torch::zeros({max_total_beam, num_heads, head_dim}, kv_cache_options);
  cached_two_stage_q_cu_seq_lens_shared_ =
      torch::zeros({max_seqs_per_batch_ + 1}, int_options);
  cached_two_stage_qo_indptr_expanded_ =
      torch::zeros({max_total_beam + 1}, int_options);
  cached_two_stage_paged_kv_indptr_expanded_ =
      torch::zeros({max_total_beam + 1}, int_options);
  cached_two_stage_paged_kv_indices_expanded_ =
      torch::zeros({max_total_beam}, int_options);
  cached_two_stage_paged_kv_last_page_len_expanded_ =
      torch::zeros({max_total_beam}, int_options);
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::
    prepare_kv_caches_related_for_input(const ForwardInput& inputs,
                                        ForwardInput& processed_inputs) {
  auto device = runtime_.worker.device();
  auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto& input_params = processed_inputs.input_params;
  auto& llm_rec_params = input_params.mutable_llmrec_params();

  const auto* step_meta = inputs.step_meta();
  CHECK(step_meta != nullptr)
      << "step_meta is required for rec multi-round mode";
  int32_t batch_size = step_meta->batch_size;
  int32_t beam_width = step_meta->beam_width;
  int32_t total_round = step_meta->total_round;
  llm_rec_params.batch_size = batch_size;
  llm_rec_params.beam_width = beam_width;
  llm_rec_params.total_round = total_round;
  const auto& shape = step_meta->full_kv_shape;
  CHECK(shape.size() == 3) << "the dims of full_kv_shape should be three.";
  int32_t full_kv_len = shape[0];
  int64_t num_kv_heads = shape[1];
  int64_t head_dim = shape[2];
  int32_t num_layers = runtime_.context->get_model_args().n_layers();
  int32_t max_decode_step = total_round - 1;
  int32_t unshared_offset = max_tokens_per_batch_;

  if (!cached_full_k_caches_.empty() && cached_full_k_caches_[0].defined()) {
    llm_rec_params.full_k_caches.reserve(num_layers);
    llm_rec_params.full_v_caches.reserve(num_layers);
    llm_rec_params.unshared_k_caches.reserve(num_layers);
    llm_rec_params.unshared_v_caches.reserve(num_layers);
    llm_rec_params.shared_k_caches.reserve(num_layers);
    llm_rec_params.shared_v_caches.reserve(num_layers);

    for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
#if defined(USE_NPU)
      auto layer_full_k_cache_flat = cached_full_k_caches_[layer_id];
      auto layer_full_v_cache_flat = cached_full_v_caches_[layer_id];

      const int64_t shared_kv_tokens = static_cast<int64_t>(unshared_offset);
      const int64_t shared_kv_elems =
          shared_kv_tokens * num_kv_heads * head_dim;
      const int64_t full_kv_elems =
          static_cast<int64_t>(full_kv_len) * num_kv_heads * head_dim;

      auto layer_full_k_cache =
          layer_full_k_cache_flat.view({full_kv_len, num_kv_heads, head_dim});
      auto layer_full_v_cache =
          layer_full_v_cache_flat.view({full_kv_len, num_kv_heads, head_dim});

      auto layer_shared_k_cache =
          layer_full_k_cache_flat.narrow(0, 0, shared_kv_elems)
              .view({shared_kv_tokens, num_kv_heads, head_dim});
      auto layer_shared_v_cache =
          layer_full_v_cache_flat.narrow(0, 0, shared_kv_elems)
              .view({shared_kv_tokens, num_kv_heads, head_dim});

      // unshared view: [block_num, beam, kv_head, max_decode_step, head_dim]
      auto layer_unshared_k_cache =
          layer_full_k_cache_flat
              .narrow(0, shared_kv_elems, full_kv_elems - shared_kv_elems)
              .view({static_cast<int64_t>(max_seqs_per_batch_),
                     static_cast<int64_t>(beam_width),
                     num_kv_heads,
                     static_cast<int64_t>(max_decode_step),
                     head_dim})
              .slice(0, 0, batch_size);
      auto layer_unshared_v_cache =
          layer_full_v_cache_flat
              .narrow(0, shared_kv_elems, full_kv_elems - shared_kv_elems)
              .view({static_cast<int64_t>(max_seqs_per_batch_),
                     static_cast<int64_t>(beam_width),
                     num_kv_heads,
                     static_cast<int64_t>(max_decode_step),
                     head_dim})
              .slice(0, 0, batch_size);
      llm_rec_params.shared_k_caches.emplace_back(layer_shared_k_cache);
      llm_rec_params.shared_v_caches.emplace_back(layer_shared_v_cache);
#else
      auto layer_full_k_cache = cached_full_k_caches_[layer_id];
      auto layer_full_v_cache = cached_full_v_caches_[layer_id];

      auto layer_unshared_k_cache =
          layer_full_k_cache.slice(0, unshared_offset, full_kv_len);
      auto layer_unshared_v_cache =
          layer_full_v_cache.slice(0, unshared_offset, full_kv_len);

      layer_unshared_k_cache =
          layer_unshared_k_cache
              .view({static_cast<int64_t>(max_seqs_per_batch_),
                     static_cast<int64_t>(beam_width_),
                     static_cast<int64_t>(max_decode_step),
                     num_kv_heads,
                     head_dim})
              .slice(0, 0, batch_size);
      layer_unshared_v_cache =
          layer_unshared_v_cache
              .view({static_cast<int64_t>(max_seqs_per_batch_),
                     static_cast<int64_t>(beam_width_),
                     static_cast<int64_t>(max_decode_step),
                     num_kv_heads,
                     head_dim})
              .slice(0, 0, batch_size);
#endif

      llm_rec_params.full_k_caches.emplace_back(layer_full_k_cache);
      llm_rec_params.full_v_caches.emplace_back(layer_full_v_cache);
      llm_rec_params.unshared_k_caches.emplace_back(layer_unshared_k_cache);
      llm_rec_params.unshared_v_caches.emplace_back(layer_unshared_v_cache);
    }
  }

#if defined(USE_NPU)
  input_params.block_tables = cached_naive_block_table_.slice(0, 0, batch_size);
#else
  input_params.block_tables =
      cached_naive_block_table_.slice(0, 0, batch_size * beam_width);
#endif

  const auto& decode_positions = step_meta->decode_positions_vec;
  llm_rec_params.decode_positions_tensor_list.clear();
  if (!decode_positions.empty() && beam_width > 0 && total_round > 1) {
    const int32_t num_sequences = static_cast<int32_t>(decode_positions.size());
    std::vector<int32_t> position_buffer;
    position_buffer.reserve(static_cast<size_t>(num_sequences * beam_width));
    for (int32_t round_idx = 0; round_idx < total_round - 1; ++round_idx) {
      position_buffer.clear();
      for (int32_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
        const int32_t base_position = decode_positions[seq_idx] + round_idx;
        for (int32_t beam_idx = 0; beam_idx < beam_width; ++beam_idx) {
          position_buffer.emplace_back(base_position);
        }
      }
      llm_rec_params.decode_positions_tensor_list.emplace_back(
          torch::tensor(position_buffer, int_options));
    }
  }
}

std::optional<ForwardOutput> RecWorkerImpl::LlmRecMultiRoundPipeline::step(
    const ForwardInput& input) {
  Timer timer;
  auto device = runtime_.worker.device_;
  device.set_device();

  ForwardInput& mutable_input = const_cast<ForwardInput&>(input);

  const auto* step_meta = mutable_input.step_meta();
  CHECK(step_meta != nullptr)
      << "step_meta is required for rec multi-round mode";
  int32_t total_rounds = step_meta->total_round;
  int32_t max_decode_step = total_rounds - 1;
  int32_t batch_size = step_meta->batch_size;
  int32_t beam_width = step_meta->beam_width;
  int32_t num_layers =
      static_cast<int32_t>(runtime_.context->get_model_args().n_layers());
  const int32_t rank = runtime_.context->get_parallel_args().rank();
  const bool use_rank0_control =
      enable_rec_multiround_tp_rank0_control() &&
      runtime_.context->get_parallel_args().world_size() > 1 &&
      input.rec_tp_step_id > 0;
  const bool use_host_shared_control =
      use_rank0_control && enable_rec_multiround_tp_host_shared_control();
  const bool is_driver = runtime_.worker.is_driver();

  CHECK_GT(runtime_.worker.kv_caches_.size(), 0)
      << "KV caches are not initialized.";

  BeamSearchTensors beam_tensors =
      prepare_beam_search_tensors(batch_size, beam_width, total_rounds, device);

  ForwardOutput output;
  torch::Tensor logits;
  SampleOutput sample_output;
  torch::Tensor top_tokens;
  torch::Tensor top_logprobs;
  std::optional<folly::SemiFuture<NextRoundInputResults>>
      next_round_async_result;

  const bool trace_pipeline_debug = enable_rec_pipeline_concurrency_debug();
  for (int32_t round = 0; round < total_rounds; ++round) {
    Timer stage_timer;
    const auto& sampling_params = round > 0
                                      ? mutable_input.decoder_sampling_params
                                      : mutable_input.sampling_params;
    SamplingParameters round_sampling_params = sampling_params;
    const bool final_round = round == total_rounds - 1;
    const int32_t requested_result_width =
        get_requested_beam_result_width(round_sampling_params, beam_width);
    const bool output_logprobs = sampling_params.logprobs;
    const int64_t output_max_top_logprobs = sampling_params.max_top_logprobs;
    if (trace_pipeline_debug) {
      LOG(INFO) << "REC multi-round stage, rank="
                << runtime_.context->get_parallel_args().rank()
                << ", round=" << round << ", stage=round_begin"
                << ", total_rounds=" << total_rounds
                << ", final_round=" << final_round
                << ", requested_result_width=" << requested_result_width
                << ", beam_width=" << beam_width << ", token_ids="
                << (mutable_input.token_ids.defined()
                        ? tensor_debug_shape(mutable_input.token_ids)
                        : "<undefined>")
                << ", positions="
                << (mutable_input.positions.defined()
                        ? tensor_debug_shape(mutable_input.positions)
                        : "<undefined>");
    }
    if (final_round && requested_result_width != beam_width) {
      round_sampling_params.max_top_logprobs = std::max<int64_t>(
          round_sampling_params.max_top_logprobs, requested_result_width);
      round_sampling_params.logprobs = true;
    }

    // Prepare round input according to the active backend.
#if defined(USE_NPU)
    prepare_round_input_for_npu(mutable_input, round, top_tokens, beam_tensors);
#else
    prepare_round_input_and_schedule_next(mutable_input,
                                          round,
                                          total_rounds,
                                          batch_size,
                                          beam_width,
                                          max_decode_step,
                                          top_tokens,
                                          beam_tensors,
                                          next_round_async_result);
#endif
    log_rec_multiround_stage_timing(
        rank, round, "prepare_round_input", stage_timer);
    stage_timer.reset();
    if (trace_pipeline_debug) {
      LOG(INFO) << "REC multi-round stage, rank="
                << runtime_.context->get_parallel_args().rank()
                << ", round=" << round << ", stage=prepare_done"
                << ", token_ids="
                << (mutable_input.token_ids.defined()
                        ? tensor_debug_shape(mutable_input.token_ids)
                        : "<undefined>")
                << ", positions="
                << (mutable_input.positions.defined()
                        ? tensor_debug_shape(mutable_input.positions)
                        : "<undefined>");
    }

    auto model_output = runtime_.executor->forward(mutable_input.token_ids,
                                                   mutable_input.positions,
                                                   runtime_.worker.kv_caches_,
                                                   mutable_input.input_params);
    log_rec_multiround_stage_timing(
        rank, round, "executor_forward", stage_timer);
    stage_timer.reset();
    if (trace_pipeline_debug) {
      LOG(INFO) << "REC multi-round stage, rank="
                << runtime_.context->get_parallel_args().rank()
                << ", round=" << round << ", stage=forward_done"
                << ", hidden_states="
                << (model_output.hidden_states.defined()
                        ? tensor_debug_shape(model_output.hidden_states)
                        : "<undefined>");
    }
    if (!model_output.hidden_states.defined()) {
      return std::nullopt;
    }
#if defined(USE_NPU)
    debug_sync_rec_multiround_device(device, rank, round, "forward_done");
#endif
    torch::Tensor hidden_states = model_output.hidden_states;

    if (sampling_params.selected_token_idxes.defined()) {
      if (trace_pipeline_debug) {
        LOG(INFO) << "REC multi-round stage, rank="
                  << runtime_.context->get_parallel_args().rank()
                  << ", round=" << round << ", stage=logits_begin"
                  << ", selected_token_idxes="
                  << tensor_debug_shape(sampling_params.selected_token_idxes);
      }
      logits = runtime_.model->logits(hidden_states,
                                      sampling_params.selected_token_idxes);
      if (trace_pipeline_debug) {
        LOG(INFO) << "REC multi-round stage, rank="
                  << runtime_.context->get_parallel_args().rank()
                  << ", round=" << round << ", stage=logits_done"
                  << ", logits="
                  << (logits.defined() ? tensor_debug_shape(logits)
                                       : "<undefined>");
      }
#if defined(USE_NPU)
      if (should_sync_rec_multiround_tp_stage(
              runtime_.context->get_parallel_args().world_size(),
              runtime_.worker.device(),
              is_driver,
              final_round)) {
        const int sync_ret = runtime_.stream->synchronize();
        CHECK_EQ(sync_ret, 0)
            << "REC multi-round TP logits stream sync failed"
            << ", rank=" << rank << ", round=" << round << ", ret=" << sync_ret;
      }
      debug_sync_rec_multiround_device(device, rank, round, "logits_done");
#endif
      log_rec_multiround_stage_timing(
          rank, round, "logits_and_sync", stage_timer);
      stage_timer.reset();
    }

    if (sampling_params.selected_token_idxes.defined() &&
        !(use_rank0_control && !is_driver)) {
      sample_output = rec_sampler_->forward(logits, round_sampling_params);
      log_rec_multiround_stage_timing(
          rank, round, "sampler_forward", stage_timer);
      stage_timer.reset();
      if (trace_pipeline_debug) {
        LOG(INFO) << "REC multi-round stage, rank="
                  << runtime_.context->get_parallel_args().rank()
                  << ", round=" << round << ", stage=sampler_done"
                  << ", top_tokens="
                  << (sample_output.top_tokens.defined()
                          ? tensor_debug_shape(sample_output.top_tokens)
                          : "<undefined>")
                  << ", top_logprobs="
                  << (sample_output.top_logprobs.defined()
                          ? tensor_debug_shape(sample_output.top_logprobs)
                          : "<undefined>");
      }
#if defined(USE_NPU)
      debug_sync_rec_multiround_device(device, rank, round, "sampler_done");
#endif
    } else if (use_rank0_control && !is_driver) {
      Timer sync_control_timer;
      if (use_host_shared_control) {
        if (!final_round) {
          const SharedControlTensors control =
              wait_rank0_control(input.rec_tp_step_id, round);
          apply_shared_control(
              control, round, beam_tensors, top_tokens, top_logprobs);
        }
      } else {
        const int64_t top_count =
            std::max<int64_t>(1, round_sampling_params.max_top_logprobs);
        synchronize_rank0_control_with_allreduce(round,
                                                 final_round,
                                                 batch_size,
                                                 beam_width,
                                                 requested_result_width,
                                                 total_rounds,
                                                 top_count,
                                                 beam_tensors,
                                                 top_tokens,
                                                 top_logprobs);
      }
      log_rec_multiround_stage_timing(
          rank, round, "sync_rank0_control", sync_control_timer);
      if (trace_pipeline_debug) {
        LOG(INFO) << "REC multi-round stage, rank="
                  << runtime_.context->get_parallel_args().rank()
                  << ", round=" << round << ", stage=rank0_control_applied"
                  << ", top_tokens="
                  << (top_tokens.defined() ? tensor_debug_shape(top_tokens)
                                           : "<undefined>")
                  << ", sequence_group="
                  << (beam_tensors.sequence_group.defined()
                          ? tensor_debug_shape(beam_tensors.sequence_group)
                          : "<undefined>");
      }
    }

    if (use_rank0_control && !is_driver) {
      if (!final_round && round > 0 && round < total_rounds - 1) {
        execute_cache_select(
            beam_tensors, mutable_input, round, beam_width, num_layers);
        log_rec_multiround_stage_timing(
            rank, round, "cache_select_from_rank0_control", stage_timer);
        stage_timer.reset();
      }
    } else if (sample_output.top_tokens.defined() &&
               sample_output.top_logprobs.defined()) {
      int64_t top_tokens_numel = sample_output.top_tokens.numel();
      int64_t top_logprobs_numel = sample_output.top_logprobs.numel();
      CHECK_EQ(top_tokens_numel % beam_width, 0)
          << "top_tokens numel (" << top_tokens_numel
          << ") must be divisible by beam_width (" << step_meta->beam_width
          << ")";
      CHECK_EQ(top_logprobs_numel % beam_width, 0)
          << "top_logprobs numel (" << top_logprobs_numel
          << ") must be divisible by beam_width (" << step_meta->beam_width
          << ")";

#if defined(USE_NPU)
      top_tokens = sample_output.top_tokens.to(torch::kInt32);
      top_logprobs = sample_output.top_logprobs;
#else
      const int64_t candidate_top_k = sample_output.top_tokens.size(-1);
      top_tokens = sample_output.top_tokens.to(torch::kInt32)
                       .reshape({-1, candidate_top_k});
      top_logprobs = sample_output.top_logprobs.reshape({-1, candidate_top_k});
#endif
      if (final_round && requested_result_width != beam_width) {
        execute_final_beam_search(top_tokens,
                                  top_logprobs,
                                  beam_tensors,
                                  round,
                                  batch_size,
                                  beam_width,
                                  requested_result_width);
      } else {
        execute_beam_search(top_tokens,
                            top_logprobs,
                            beam_tensors,
                            round,
                            batch_size,
                            requested_result_width,
                            total_rounds);
      }
      log_rec_multiround_stage_timing(rank, round, "beam_search", stage_timer);
      stage_timer.reset();
      if (use_rank0_control) {
        Timer sync_control_timer;
        if (use_host_shared_control) {
          if (!final_round) {
            publish_rank0_control(input.rec_tp_step_id,
                                  round,
                                  final_round,
                                  top_tokens,
                                  top_logprobs,
                                  beam_tensors);
          }
        } else {
          const int64_t top_count =
              std::max<int64_t>(1, round_sampling_params.max_top_logprobs);
          synchronize_rank0_control_with_allreduce(round,
                                                   final_round,
                                                   batch_size,
                                                   beam_width,
                                                   requested_result_width,
                                                   total_rounds,
                                                   top_count,
                                                   beam_tensors,
                                                   top_tokens,
                                                   top_logprobs);
        }
        log_rec_multiround_stage_timing(
            rank, round, "sync_rank0_control", sync_control_timer);
      }
      if (trace_pipeline_debug) {
        LOG(INFO) << "REC multi-round stage, rank="
                  << runtime_.context->get_parallel_args().rank()
                  << ", round=" << round << ", stage=beam_search_done"
                  << ", out_token_ids="
                  << (beam_tensors.out_token_ids.defined()
                          ? tensor_debug_shape(beam_tensors.out_token_ids)
                          : "<undefined>")
                  << ", sequence_group="
                  << (beam_tensors.sequence_group.defined()
                          ? tensor_debug_shape(beam_tensors.sequence_group)
                          : "<undefined>");
      }
#if defined(USE_NPU)
      debug_sync_rec_multiround_device(device, rank, round, "beam_search_done");
#endif

      if (round > 0 && round < total_rounds - 1) {
        execute_cache_select(
            beam_tensors, mutable_input, round, beam_width, num_layers);
        log_rec_multiround_stage_timing(
            rank, round, "cache_select", stage_timer);
        stage_timer.reset();
        if (trace_pipeline_debug) {
          LOG(INFO) << "REC multi-round stage, rank="
                    << runtime_.context->get_parallel_args().rank()
                    << ", round=" << round << ", stage=cache_select_done";
        }
#if defined(USE_NPU)
        debug_sync_rec_multiround_device(
            device, rank, round, "cache_select_done");
#endif
      }

      if (final_round) {
        build_final_output(
            logits, sample_output, sampling_params, beam_tensors, output);
        output.logprobs = output_logprobs;
        output.max_top_logprobs = output_max_top_logprobs;
        log_rec_multiround_stage_timing(
            rank, round, "build_final_output", stage_timer);
        stage_timer.reset();
        if (trace_pipeline_debug) {
          LOG(INFO) << "REC multi-round stage, rank="
                    << runtime_.context->get_parallel_args().rank()
                    << ", round=" << round << ", stage=final_output_done"
                    << ", beam_sequence_group="
                    << (output.beam_sequence_group.defined()
                            ? tensor_debug_shape(output.beam_sequence_group)
                            : "<undefined>");
        }
#if defined(USE_NPU)
        debug_sync_rec_multiround_device(
            device, rank, round, "final_output_done");
#endif
      }
    }
  }

  if (trace_pipeline_debug) {
    LOG(INFO) << "REC multi-round stage, rank="
              << runtime_.context->get_parallel_args().rank()
              << ", stage=stream_synchronize_begin";
  }
  Timer final_sync_timer;
  runtime_.stream->synchronize();
  log_rec_multiround_stage_timing(
      rank, total_rounds, "final_stream_sync", final_sync_timer);
  if (trace_pipeline_debug) {
    LOG(INFO) << "REC multi-round stage, rank="
              << runtime_.context->get_parallel_args().rank()
              << ", stage=stream_synchronize_done";
  }

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(device.index());
  if (use_rank0_control && !is_driver) {
    return std::nullopt;
  }
  return output;
}

RecWorkerImpl::LlmRecMultiRoundPipeline::BeamSearchTensors
RecWorkerImpl::LlmRecMultiRoundPipeline::prepare_beam_search_tensors(
    int32_t batch_size,
    int32_t beam_width,
    int32_t total_rounds,
    const torch::Device& device) {
  auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  BeamSearchTensors tensors;
  tensors.sequence_group =
      torch::zeros({batch_size, beam_width, total_rounds}, int_options);
  int64_t num_seq = batch_size * beam_width;
  tensors.acc_logprob = torch::zeros({num_seq, 1}, fp32_options);
  tensors.out_log_probs = torch::zeros({num_seq, 1}, fp32_options);
  tensors.out_token_ids = torch::zeros({num_seq, 1}, int_options);
  tensors.out_token_index = torch::zeros({num_seq, 1}, int_options);
  tensors.out_beam_count_prefix_sums = torch::zeros({num_seq, 1}, int_options);
  tensors.out_seqgroup = torch::zeros_like(tensors.sequence_group);
  return tensors;
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::execute_beam_search(
    const torch::Tensor& top_tokens,
    const torch::Tensor& top_logprobs,
    BeamSearchTensors& beam_tensors,
    int32_t round,
    int32_t batch_size,
    int32_t requested_result_width,
    int32_t total_rounds) {
#if defined(USE_NPU)
  (void)requested_result_width;
  (void)total_rounds;
  if (round == 0) {
    runtime::detail::write_first_round_beam_outputs(
        top_tokens.reshape({-1, 1}),
        top_logprobs.reshape({-1, 1}),
        batch_size,
        beam_tensors.out_token_ids,
        beam_tensors.out_log_probs,
        beam_tensors.out_seqgroup);
  } else if (runtime_.context->get_parallel_args().world_size() > 1 &&
             !enable_rec_multiround_tp_npu_intermediate_beam()) {
    select_intermediate_onerec_beam_results(
        /*acc_logprob=*/beam_tensors.acc_logprob,
        /*sequence_group=*/beam_tensors.sequence_group,
        /*top_tokens=*/top_tokens,
        /*top_logprobs=*/top_logprobs,
        /*batch_size=*/batch_size,
        /*beam_width=*/
        static_cast<int32_t>(beam_tensors.sequence_group.size(1)),
        /*current_step=*/round,
        /*out_token_ids=*/beam_tensors.out_token_ids,
        /*out_token_index=*/beam_tensors.out_token_index,
        /*out_log_probs=*/beam_tensors.out_log_probs,
        /*out_beam_count_prefix_sums=*/
        beam_tensors.out_beam_count_prefix_sums,
        /*out_sequence=*/beam_tensors.out_seqgroup);
  } else {
    xllm::kernel::npu::beam_search_rec(
        /*logprobs=*/beam_tensors.acc_logprob,
        /*top_tokens=*/top_tokens,
        /*top_logprobs=*/top_logprobs,
        /*sequence_group=*/beam_tensors.sequence_group,
        /*current_step=*/static_cast<int64_t>(round),
        /*out_token_ids=*/beam_tensors.out_token_ids,
        /*out_token_index=*/beam_tensors.out_token_index,
        /*out_log_probs=*/beam_tensors.out_log_probs,
        /*out_beam_count_prefix_sums=*/
        beam_tensors.out_beam_count_prefix_sums,
        /*out_sequence=*/beam_tensors.out_seqgroup);
  }
#elif defined(USE_CUDA)
  (void)total_rounds;
  xllm::kernel::cuda::beam_search(beam_tensors.acc_logprob,
                                  beam_tensors.sequence_group,
                                  top_tokens,
                                  top_logprobs,
                                  beam_tensors.out_log_probs,
                                  beam_tensors.out_token_ids,
                                  beam_tensors.out_token_index,
                                  beam_tensors.out_beam_count_prefix_sums,
                                  beam_tensors.out_seqgroup,
                                  batch_size,
                                  requested_result_width,
                                  round);
#endif
  std::swap(beam_tensors.sequence_group, beam_tensors.out_seqgroup);
  std::swap(beam_tensors.acc_logprob, beam_tensors.out_log_probs);
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::execute_final_beam_search(
    const torch::Tensor& top_tokens,
    const torch::Tensor& top_logprobs,
    BeamSearchTensors& beam_tensors,
    int32_t round,
    int32_t batch_size,
    int32_t beam_width,
    int32_t requested_result_width) {
  OneRecBeamSearchOutputTensors final_tensors =
      prepare_onerec_beam_search_output_tensors(
          batch_size,
          requested_result_width,
          static_cast<int32_t>(beam_tensors.sequence_group.size(2)),
          runtime_.worker.device());

#if defined(USE_NPU)
  if (runtime_.context->get_parallel_args().world_size() == 1 &&
      can_use_beam_search_rec_final_select(
          batch_size, top_tokens, requested_result_width)) {
    xllm::kernel::npu::beam_search_rec(
        /*logprobs=*/beam_tensors.acc_logprob,
        /*top_tokens=*/top_tokens,
        /*top_logprobs=*/top_logprobs,
        /*sequence_group=*/beam_tensors.sequence_group,
        /*current_step=*/static_cast<int64_t>(round),
        /*result_width=*/requested_result_width,
        /*out_token_ids=*/final_tensors.out_token_ids,
        /*out_token_index=*/final_tensors.out_token_index,
        /*out_log_probs=*/final_tensors.out_log_probs,
        /*out_beam_count_prefix_sums=*/final_tensors.out_beam_count_prefix_sums,
        /*out_sequence=*/final_tensors.out_seqgroup);
  } else {
    select_final_onerec_beam_results(
        beam_tensors.acc_logprob,
        beam_tensors.sequence_group,
        top_tokens,
        top_logprobs,
        batch_size,
        beam_width,
        requested_result_width,
        static_cast<int32_t>(beam_tensors.sequence_group.size(2)),
        round,
        final_tensors);
  }
#elif defined(USE_CUDA)
  select_final_onerec_beam_results(
      beam_tensors.acc_logprob,
      beam_tensors.sequence_group,
      top_tokens,
      top_logprobs,
      batch_size,
      beam_width,
      requested_result_width,
      static_cast<int32_t>(beam_tensors.sequence_group.size(2)),
      round,
      final_tensors);
#else
  LOG(FATAL) << "Rec multi-round final beam search requires NPU or CUDA.";
#endif

  beam_tensors.out_token_ids = std::move(final_tensors.out_token_ids);
  beam_tensors.out_token_index = std::move(final_tensors.out_token_index);
  beam_tensors.out_beam_count_prefix_sums =
      std::move(final_tensors.out_beam_count_prefix_sums);
  beam_tensors.out_log_probs = std::move(final_tensors.out_log_probs);
  beam_tensors.out_seqgroup = std::move(final_tensors.out_seqgroup);

  std::swap(beam_tensors.sequence_group, beam_tensors.out_seqgroup);
  std::swap(beam_tensors.acc_logprob, beam_tensors.out_log_probs);
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::execute_cache_select(
    const BeamSearchTensors& beam_tensors,
    ForwardInput& input,
    int32_t round,
    int32_t beam_width,
    int32_t num_layers) {
#if defined(USE_NPU)
  auto device = runtime_.worker.device();
  auto int32_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  const int32_t batch_size =
      static_cast<int32_t>(beam_tensors.sequence_group.size(0));
  auto batch_offsets = torch::arange(batch_size, int32_options) * beam_width;
  auto batch_offsets_2d = batch_offsets.unsqueeze(1);

  auto beam_index_global =
      beam_tensors.out_token_index.reshape({batch_size, beam_width});
  auto beam_index_local = beam_index_global - batch_offsets_2d;
  auto group_prefix_global =
      beam_tensors.out_beam_count_prefix_sums.reshape({batch_size, beam_width});
  auto group_prefix_local = group_prefix_global - batch_offsets_2d;

  auto block_table = torch::arange(batch_size, int32_options);

  const auto& unshared_k_caches =
      input.input_params.mutable_llmrec_params().unshared_k_caches;
  const auto& unshared_v_caches =
      input.input_params.mutable_llmrec_params().unshared_v_caches;

  auto run_cache_select = [&]() {
    xllm::kernel::npu::select_unshared_kv(
        /*beam_index=*/beam_index_local.reshape({-1}),
        /*x_key_block=*/unshared_k_caches,
        /*x_value_block=*/unshared_v_caches,
        /*block_table=*/block_table,
        /*group_offset=*/group_prefix_local.reshape({-1}),
        /*decode_step=*/static_cast<int64_t>(round),
        /*beam_size=*/beam_width,
        /*layer_num=*/num_layers);
  };
  if (runtime_.context->get_parallel_args().world_size() > 1 &&
      serialize_rec_multiround_tp_cache_select()) {
    std::lock_guard<std::mutex> lock(rec_multiround_tp_cache_select_mutex());
    run_cache_select();
  } else {
    run_cache_select();
  }
#elif defined(USE_CUDA)
  xllm::kernel::cuda::cache_select(
      beam_tensors.out_token_index,
      input.input_params.mutable_llmrec_params().unshared_k_caches,
      input.input_params.mutable_llmrec_params().unshared_v_caches,
      input.input_params.block_tables,
      round - 1,
      beam_width,
      num_layers);
#endif
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::publish_rank0_control(
    uint64_t rec_tp_step_id,
    int32_t round,
    bool final_round,
    const torch::Tensor& top_tokens,
    const torch::Tensor& top_logprobs,
    const BeamSearchTensors& beam_tensors) {
  if (rec_tp_step_id == 0 || shared_control_state_ == nullptr) {
    return;
  }

  SharedControlTensors control;
  control.top_tokens_cpu =
      top_tokens.defined() ? top_tokens.to(torch::kCPU) : torch::Tensor();
  control.top_logprobs_cpu =
      top_logprobs.defined() ? top_logprobs.to(torch::kCPU) : torch::Tensor();
  control.sequence_group_cpu = beam_tensors.sequence_group.defined()
                                   ? beam_tensors.sequence_group.to(torch::kCPU)
                                   : torch::Tensor();
  control.acc_logprob_cpu = beam_tensors.acc_logprob.defined()
                                ? beam_tensors.acc_logprob.to(torch::kCPU)
                                : torch::Tensor();
  control.out_token_ids_cpu = beam_tensors.out_token_ids.defined()
                                  ? beam_tensors.out_token_ids.to(torch::kCPU)
                                  : torch::Tensor();
  control.out_token_index_cpu =
      beam_tensors.out_token_index.defined()
          ? beam_tensors.out_token_index.to(torch::kCPU)
          : torch::Tensor();
  control.out_beam_count_prefix_sums_cpu =
      beam_tensors.out_beam_count_prefix_sums.defined()
          ? beam_tensors.out_beam_count_prefix_sums.to(torch::kCPU)
          : torch::Tensor();
  control.final_round = final_round;

  const uint64_t key = rec_multiround_control_key(rec_tp_step_id, round);
  std::shared_ptr<SharedControlSlot> slot;
  {
    std::lock_guard<std::mutex> state_lock(shared_control_state_->slots_mutex);
    auto& slot_ref = shared_control_state_->slots[key];
    if (slot_ref == nullptr) {
      slot_ref = std::make_shared<SharedControlSlot>();
    }
    slot = slot_ref;
  }

  {
    std::lock_guard<std::mutex> slot_lock(slot->mutex);
    slot->tensors = std::move(control);
  }
  slot->cv.notify_all();
}

RecWorkerImpl::LlmRecMultiRoundPipeline::SharedControlTensors
RecWorkerImpl::LlmRecMultiRoundPipeline::wait_rank0_control(
    uint64_t rec_tp_step_id,
    int32_t round) {
  CHECK_GT(rec_tp_step_id, 0);
  CHECK(shared_control_state_ != nullptr);
  const uint64_t key = rec_multiround_control_key(rec_tp_step_id, round);

  std::shared_ptr<SharedControlSlot> slot;
  {
    std::lock_guard<std::mutex> state_lock(shared_control_state_->slots_mutex);
    auto& slot_ref = shared_control_state_->slots[key];
    if (slot_ref == nullptr) {
      slot_ref = std::make_shared<SharedControlSlot>();
    }
    slot = slot_ref;
  }

  std::unique_lock<std::mutex> slot_lock(slot->mutex);
  const bool ready =
      slot->cv.wait_for(slot_lock, std::chrono::seconds(60), [&] {
        return slot->tensors.has_value();
      });
  CHECK(ready) << "Timed out waiting for rank0 REC multi-round TP control, "
               << "step_id=" << rec_tp_step_id << ", round=" << round;
  return slot->tensors.value();
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::erase_rank0_control(
    uint64_t rec_tp_step_id) {
  if (rec_tp_step_id == 0 || shared_control_state_ == nullptr ||
      !runtime_.worker.is_driver()) {
    return;
  }
  std::lock_guard<std::mutex> state_lock(shared_control_state_->slots_mutex);
  const uint64_t first_key = rec_multiround_control_key(rec_tp_step_id, 0);
  const uint64_t last_key = rec_multiround_control_key(rec_tp_step_id + 1, 0);
  auto it = shared_control_state_->slots.lower_bound(first_key);
  while (it != shared_control_state_->slots.end() && it->first < last_key) {
    it = shared_control_state_->slots.erase(it);
  }
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::apply_shared_control(
    const SharedControlTensors& control,
    int32_t round,
    BeamSearchTensors& beam_tensors,
    torch::Tensor& top_tokens,
    torch::Tensor& top_logprobs) {
  auto device = runtime_.worker.device();
  if (control.top_tokens_cpu.defined()) {
    top_tokens = control.top_tokens_cpu.to(device, /*non_blocking=*/false);
  }
  if (control.top_logprobs_cpu.defined()) {
    top_logprobs = control.top_logprobs_cpu.to(device, /*non_blocking=*/false);
  }
  if (control.sequence_group_cpu.defined()) {
    beam_tensors.sequence_group =
        control.sequence_group_cpu.to(device, /*non_blocking=*/false);
  }
  if (control.acc_logprob_cpu.defined()) {
    beam_tensors.acc_logprob =
        control.acc_logprob_cpu.to(device, /*non_blocking=*/false);
  }
  if (!control.final_round && round > 0) {
    if (control.out_token_ids_cpu.defined()) {
      beam_tensors.out_token_ids =
          control.out_token_ids_cpu.to(device, /*non_blocking=*/false);
    }
    if (control.out_token_index_cpu.defined()) {
      beam_tensors.out_token_index =
          control.out_token_index_cpu.to(device, /*non_blocking=*/false);
    }
    if (control.out_beam_count_prefix_sums_cpu.defined()) {
      beam_tensors.out_beam_count_prefix_sums =
          control.out_beam_count_prefix_sums_cpu.to(device,
                                                    /*non_blocking=*/false);
    }
  }
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::
    synchronize_rank0_control_with_allreduce(int32_t round,
                                             bool final_round,
                                             int32_t batch_size,
                                             int32_t beam_width,
                                             int32_t requested_result_width,
                                             int32_t total_rounds,
                                             int64_t top_count,
                                             BeamSearchTensors& beam_tensors,
                                             torch::Tensor& top_tokens,
                                             torch::Tensor& top_logprobs) {
  const ParallelArgs& parallel_args = runtime_.context->get_parallel_args();
  ProcessGroup* tp_group = runtime_.rec_tp_control_group != nullptr
                               ? runtime_.rec_tp_control_group
                               : parallel_args.tp_group_;
  CHECK(tp_group != nullptr)
      << "REC multi-round rank0-control requires a TP process group.";
  CHECK_GT(tp_group->world_size(), 1)
      << "REC multi-round rank0-control should only run for TP world_size > 1.";
  CHECK_EQ(tp_group->rank(), parallel_args.rank())
      << "TP group rank must match worker rank for rank0-control.";
  CHECK_GT(batch_size, 0);
  CHECK_GT(beam_width, 0);
  CHECK_GT(total_rounds, 0);
  CHECK_GT(top_count, 0);

  const torch::Device device = runtime_.worker.device();
  const bool is_driver = runtime_.worker.is_driver();
  const int64_t top_rows = round == 0
                               ? static_cast<int64_t>(batch_size)
                               : static_cast<int64_t>(batch_size) * beam_width;
  const int32_t output_width =
      final_round && requested_result_width != beam_width
          ? requested_result_width
          : beam_width;
  CHECK_GT(output_width, 0);
  const int64_t output_rows = static_cast<int64_t>(batch_size) * output_width;

  auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  struct ControlTensorPlan {
    torch::Tensor* tensor;
    std::vector<int64_t> shape;
    const char* name;
  };

  auto shape_numel = [](const std::vector<int64_t>& shape) {
    int64_t numel = 1;
    for (int64_t dim : shape) {
      CHECK_GT(dim, 0);
      numel *= dim;
    }
    return numel;
  };

  auto check_driver_tensor = [&](torch::Tensor& tensor,
                                 const std::vector<int64_t>& shape,
                                 const char* name) {
    CHECK(tensor.defined()) << "rank0 control tensor is undefined: " << name;
    CHECK_EQ(tensor.sizes(), torch::IntArrayRef(shape))
        << "rank0 control tensor shape mismatch for " << name
        << ", expected=" << torch::IntArrayRef(shape)
        << ", actual=" << tensor.sizes();
  };

  std::vector<ControlTensorPlan> int_plans = {
      {&top_tokens, {top_rows, top_count}, "top_tokens"},
      {&beam_tensors.sequence_group,
       {batch_size, output_width, total_rounds},
       "sequence_group"},
      {&beam_tensors.out_token_ids, {output_rows, 1}, "out_token_ids"},
      {&beam_tensors.out_token_index, {output_rows, 1}, "out_token_index"},
      {&beam_tensors.out_beam_count_prefix_sums,
       {output_rows, 1},
       "out_beam_count_prefix_sums"},
  };
  std::vector<ControlTensorPlan> fp32_plans = {
      {&top_logprobs, {top_rows, top_count}, "top_logprobs"},
      {&beam_tensors.acc_logprob, {output_rows, 1}, "acc_logprob"},
  };

  if (enable_rec_multiround_tp_packed_rank0_control()) {
    auto pack_allreduce_unpack = [&](std::vector<ControlTensorPlan>& plans,
                                     const torch::TensorOptions& options,
                                     torch::Dtype dtype) {
      int64_t total_numel = 0;
      for (const auto& plan : plans) {
        total_numel += shape_numel(plan.shape);
      }
      torch::Tensor buffer = torch::zeros({total_numel}, options);

      int64_t offset = 0;
      for (auto& plan : plans) {
        const int64_t numel = shape_numel(plan.shape);
        if (is_driver) {
          check_driver_tensor(*plan.tensor, plan.shape, plan.name);
          buffer.narrow(/*dim=*/0, offset, numel)
              .copy_(plan.tensor->to(dtype).contiguous().view({numel}));
        }
        offset += numel;
      }

      tp_group->allreduce(buffer);

      offset = 0;
      for (auto& plan : plans) {
        const int64_t numel = shape_numel(plan.shape);
        *plan.tensor = buffer.narrow(/*dim=*/0, offset, numel).view(plan.shape);
        offset += numel;
      }
    };

    // Preserve identical collective order across TP ranks while reducing the
    // rank0-control fan-out from seven small HCCL allreduces to two.
    pack_allreduce_unpack(int_plans, int_options, torch::kInt32);
    pack_allreduce_unpack(fp32_plans, fp32_options, torch::kFloat32);
    return;
  }

  auto prepare_int_tensor =
      [&](torch::Tensor& tensor, std::vector<int64_t> shape, const char* name) {
        if (is_driver) {
          check_driver_tensor(tensor, shape, name);
          tensor = tensor.to(torch::kInt32).contiguous();
        } else {
          tensor = torch::zeros(shape, int_options);
        }
        tp_group->allreduce(tensor);
      };

  auto prepare_fp32_tensor =
      [&](torch::Tensor& tensor, std::vector<int64_t> shape, const char* name) {
        if (is_driver) {
          check_driver_tensor(tensor, shape, name);
          tensor = tensor.to(torch::kFloat32).contiguous();
        } else {
          tensor = torch::zeros(shape, fp32_options);
        }
        tp_group->allreduce(tensor);
      };

  // All ranks must issue HCCL collectives in the same order with identical
  // shape/dtype. Rank0 contributes real control tensors; other ranks
  // contribute zeros and receive rank0's values via SUM allreduce.
  prepare_int_tensor(top_tokens, {top_rows, top_count}, "top_tokens");
  prepare_fp32_tensor(top_logprobs, {top_rows, top_count}, "top_logprobs");
  prepare_int_tensor(beam_tensors.sequence_group,
                     {batch_size, output_width, total_rounds},
                     "sequence_group");
  prepare_fp32_tensor(
      beam_tensors.acc_logprob, {output_rows, 1}, "acc_logprob");
  prepare_int_tensor(
      beam_tensors.out_token_ids, {output_rows, 1}, "out_token_ids");
  prepare_int_tensor(
      beam_tensors.out_token_index, {output_rows, 1}, "out_token_index");
  prepare_int_tensor(beam_tensors.out_beam_count_prefix_sums,
                     {output_rows, 1},
                     "out_beam_count_prefix_sums");
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::build_final_output(
    const torch::Tensor& logits,
    const SampleOutput& sample_output,
    const SamplingParameters& sampling_params,
    const BeamSearchTensors& beam_tensors,
    ForwardOutput& output) {
  output.logits = logits;
  output.sample_output = sample_output;
  output.do_sample = sampling_params.do_sample;
  output.logprobs = sampling_params.logprobs;
  output.max_top_logprobs = sampling_params.max_top_logprobs;
  output.beam_search_output.src_seq_idxes =
      beam_tensors.out_token_index.reshape({-1});
  output.beam_search_output.out_tokens =
      beam_tensors.out_token_ids.reshape({-1});
  output.beam_search_output.out_logprobs =
      beam_tensors.acc_logprob.reshape({-1});
  output.beam_sequence_group = beam_tensors.sequence_group;
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::prepare_two_stage_round_input(
    ForwardInput& input,
    int32_t round,
    const torch::Tensor& top_tokens,
    const BeamSearchTensors& beam_tensors) {
#if defined(USE_NPU)
// TODO: implement prepare_two_stage_round_input for NPU
#elif defined(USE_CUDA)
  auto& llm_rec_params = input.input_params.mutable_llmrec_params();
  CHECK_EQ(FLAGS_enable_xattention_one_stage, false)
      << "prepare_two_stage_round_input should only be called when "
         "two-stage decode is enabled";

  input.input_params.paged_kv_indices = torch::Tensor();
  input.input_params.paged_kv_indptr = torch::Tensor();
  input.input_params.paged_kv_last_page_len = torch::Tensor();
  input.input_params.num_sequences =
      llm_rec_params.batch_size *
      std::max<int32_t>(llm_rec_params.beam_width, 1);

  // previous_step corresponds to the decode step that produced tokens for
  // this round.
  const int32_t previous_step = round - 1;
  if (previous_step == 0) {
    // First decode step uses top_tokens from prefill.
    if (top_tokens.defined()) {
      input.token_ids = top_tokens.reshape({-1});
    }
  } else if (previous_step > 0) {
    // Later steps use beam search output tokens.
    input.token_ids = beam_tensors.out_token_ids.reshape({-1});
  }

  if (!llm_rec_params.decode_positions_tensor_list.empty() &&
      previous_step >= 0 &&
      previous_step < static_cast<int32_t>(
                          llm_rec_params.decode_positions_tensor_list.size())) {
    input.positions =
        llm_rec_params.decode_positions_tensor_list[previous_step];
  }

  input.input_params.batch_forward_type = BatchForwardType(2);
  input.input_params.input_embedding = torch::Tensor();
  cached_current_round_tensor_.fill_(previous_step);
  llm_rec_params.current_round_tensor = cached_current_round_tensor_;

  const int32_t batch_size = std::max<int32_t>(llm_rec_params.batch_size, 0);
  const int32_t beam_width = std::max<int32_t>(llm_rec_params.beam_width, 1);
  const int64_t total_beam = static_cast<int64_t>(batch_size) * beam_width;

  CHECK_LE(total_beam, cached_two_stage_shared_lse_.size(0))
      << "two-stage cache total_beam overflow";
  CHECK_LE(batch_size + 1, cached_two_stage_q_cu_seq_lens_shared_.size(0))
      << "two-stage q_cu_seq_lens cache overflow";
  CHECK_LE(total_beam + 1, cached_two_stage_qo_indptr_expanded_.size(0))
      << "two-stage qo_indptr cache overflow";

  llm_rec_params.two_stage_shared_lse =
      cached_two_stage_shared_lse_.slice(0, 0, total_beam);
  llm_rec_params.two_stage_shared_o =
      cached_two_stage_shared_o_.slice(0, 0, total_beam);
  llm_rec_params.two_stage_unshared_lse =
      cached_two_stage_unshared_lse_.slice(0, 0, total_beam);
  llm_rec_params.two_stage_unshared_o =
      cached_two_stage_unshared_o_.slice(0, 0, total_beam);
  llm_rec_params.two_stage_q_cu_seq_lens_shared =
      cached_two_stage_q_cu_seq_lens_shared_.slice(0, 0, batch_size + 1);
  llm_rec_params.two_stage_qo_indptr_expanded =
      cached_two_stage_qo_indptr_expanded_.slice(0, 0, total_beam + 1);
  llm_rec_params.two_stage_paged_kv_indptr_expanded =
      cached_two_stage_paged_kv_indptr_expanded_.slice(0, 0, total_beam + 1);
  llm_rec_params.two_stage_paged_kv_indices_expanded =
      cached_two_stage_paged_kv_indices_expanded_.slice(0, 0, total_beam);
  llm_rec_params.two_stage_paged_kv_last_page_len_expanded =
      cached_two_stage_paged_kv_last_page_len_expanded_.slice(0, 0, total_beam);

  auto int_options = torch::TensorOptions()
                         .dtype(torch::kInt32)
                         .device(runtime_.worker.device());
  auto q_cu_seq_lens_values =
      torch::arange(0, (batch_size + 1) * beam_width, beam_width, int_options);
  llm_rec_params.two_stage_q_cu_seq_lens_shared.copy_(q_cu_seq_lens_values,
                                                      /*non_blocking=*/true);

  // The unshared two-stage decode path packs one query row per expanded beam,
  // so qo_indptr is the prefix sum of per-beam query lengths rather than a
  // paged-kv layout descriptor.
  auto qo_indptr_values = torch::arange(total_beam + 1, int_options);
  llm_rec_params.two_stage_qo_indptr_expanded.copy_(qo_indptr_values,
                                                    /*non_blocking=*/true);

  auto paged_kv_indptr_values = torch::arange(total_beam + 1, int_options);
  llm_rec_params.two_stage_paged_kv_indptr_expanded.copy_(
      paged_kv_indptr_values, /*non_blocking=*/true);

  if (input.input_params.block_tables.defined() &&
      input.input_params.block_tables.numel() >= total_beam) {
    llm_rec_params.two_stage_paged_kv_indices_expanded.copy_(
        input.input_params.block_tables.view({-1}).slice(0, 0, total_beam),
        /*non_blocking=*/true);
  } else {
    auto paged_kv_indices_values = torch::arange(total_beam, int_options);
    llm_rec_params.two_stage_paged_kv_indices_expanded.copy_(
        paged_kv_indices_values, /*non_blocking=*/true);
  }

  llm_rec_params.two_stage_paged_kv_last_page_len_expanded.fill_(previous_step +
                                                                 1);
  input.input_params.attn_metadata = nullptr;
#endif
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::prepare_round_input_for_npu(
    ForwardInput& input,
    int32_t round,
    const torch::Tensor& top_tokens,
    const BeamSearchTensors& beam_tensors) {
  auto& llm_rec_params = input.input_params.mutable_llmrec_params();
  CHECK(cached_current_round_tensor_.defined());
  CHECK(cached_beam_width_tensor_.defined());

  cached_beam_width_tensor_.fill_(llm_rec_params.beam_width);
  llm_rec_params.beam_width_tensor = cached_beam_width_tensor_;
  cached_current_round_tensor_.fill_(round);
  llm_rec_params.current_round_tensor = cached_current_round_tensor_;
  input.input_params.attn_metadata = nullptr;

  if (round > 0) {
    if (round == 1) {
      if (top_tokens.defined()) {
        input.token_ids = top_tokens.reshape({-1});
      }
    } else {
      input.token_ids = beam_tensors.out_token_ids.reshape({-1});
    }

    const int32_t decode_step = round - 1;
    if (!llm_rec_params.decode_positions_tensor_list.empty() &&
        decode_step < static_cast<int32_t>(
                          llm_rec_params.decode_positions_tensor_list.size())) {
      input.positions =
          llm_rec_params.decode_positions_tensor_list[decode_step];
    }

    input.input_params.batch_forward_type = BatchForwardType::DECODE;
    input.input_params.input_embedding = torch::Tensor();
  }
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::prepare_input_for_current_round(
    ForwardInput& input,
    const NextRoundInputResults& results,
    int32_t round,
    const torch::Tensor& top_tokens,
    const BeamSearchTensors& beam_tensors) {
#if defined(USE_CUDA)
  if (FLAGS_enable_xattention_one_stage) {
    input.input_params.paged_kv_indices = results.paged_kv_indices;
    input.input_params.paged_kv_indptr = results.paged_kv_indptr;
    input.input_params.paged_kv_last_page_len = results.paged_kv_last_page_len;
    input.input_params.num_sequences =
        input.input_params.paged_kv_last_page_len.numel();
  } else {
    prepare_two_stage_round_input(input, round, top_tokens, beam_tensors);
    return;
  }
#endif
  // previous_step corresponds to the decode step that produced tokens for
  // this round.
  const int32_t previous_step = round - 1;
  if (previous_step == 0) {
    // First decode step uses top_tokens from prefill.
    if (top_tokens.defined()) {
      input.token_ids = top_tokens.reshape({-1});
    }
  } else if (previous_step > 0) {
    // Later steps use beam search output tokens.
    input.token_ids = beam_tensors.out_token_ids.reshape({-1});
  }

  auto& llm_rec_params = input.input_params.mutable_llmrec_params();
  if (!llm_rec_params.decode_positions_tensor_list.empty() &&
      previous_step >= 0 &&
      previous_step < static_cast<int32_t>(
                          llm_rec_params.decode_positions_tensor_list.size())) {
    input.positions =
        llm_rec_params.decode_positions_tensor_list[previous_step];
  }

  input.input_params.batch_forward_type = BatchForwardType(2);
  input.input_params.input_embedding = torch::Tensor();
  cached_current_round_tensor_.fill_(previous_step);
  llm_rec_params.current_round_tensor = cached_current_round_tensor_;
  input.input_params.attn_metadata = nullptr;
}

folly::SemiFuture<
    RecWorkerImpl::LlmRecMultiRoundPipeline::NextRoundInputResults>
RecWorkerImpl::LlmRecMultiRoundPipeline::compute_next_round_input_async(
    const torch::Tensor& kv_seq_lens,
    int32_t current_step,
    int32_t batch_size,
    int32_t beam_width,
    int32_t max_decode_step) {
  folly::Promise<NextRoundInputResults> promise;
  auto future = promise.getSemiFuture();

#if defined(USE_CUDA)
  if (FLAGS_enable_xattention_one_stage) {
    // Capture necessary data for async computation
    auto full_kv_offsets = full_kv_cache_offsets_->full_kv_offsets;
    auto full_kv_mask = full_kv_cache_offsets_->full_kv_mask;
    auto full_kv_indices = full_kv_cache_offsets_->full_kv_indices;
    auto unshared_full_kv_offsets = full_kv_cache_offsets_->unshared_offsets;
    auto real_max_decode_step_ids = full_kv_cache_offsets_->max_decode_step_ids;
    uint32_t unshared_kv_begin_offset = max_tokens_per_batch_;

    // Launch async computation in thread pool (can overlap with GPU execution)
    threadpool_.schedule([=, this, promise = std::move(promise)]() mutable {
      auto device = runtime_.worker.device();
      auto int32_device_options =
          torch::TensorOptions().dtype(torch::kInt32).device(device);
      // Protect CUDA graph capture from conflicting GPU work submitted on
      // prepare_stream_ while capture is in progress. Use shared lock to allow
      // multiple prepare operations to run concurrently, but prevent conflicts
      // with capture operations. This mirrors the NPU DeviceCaptureLock usage
      // in WorkerImpl::prepare_work_before_execute.
      std::optional<std::shared_lock<std::shared_mutex>> lock_guard;
      if (runtime_.worker.options_.enable_graph()) {
        auto& replay_lock =
            ::xllm::cuda::DeviceCaptureLock::get_instance().get_read_lock(
                runtime_.worker.device_.index());
        lock_guard.emplace(replay_lock);
      }

      c10::StreamGuard streamGuard =
          runtime_.worker.prepare_stream_->set_stream_guard();
      auto shared_kv_offsets = full_kv_offsets.slice(2, 0, max_token_per_req_)
                                   .slice(0, 0, batch_size);

      auto shared_kv_lens_each_batch = torch::diff(kv_seq_lens);

      auto shared_kv_lens_each_batch_broadcast =
          shared_kv_lens_each_batch.unsqueeze(1).unsqueeze(1);

      auto shared_mask =
          full_kv_mask.slice(2, 0, max_token_per_req_).slice(0, 0, batch_size);

      shared_mask.copy_(shared_kv_offsets <
                        shared_kv_lens_each_batch_broadcast);

      auto kv_lens_batch_offsets = kv_seq_lens.slice(0, 0, -1);

      auto kv_lens_batch_offsets_broadcast =
          kv_lens_batch_offsets.unsqueeze(1).unsqueeze(1);

      auto shared_kv_indices = full_kv_indices.slice(2, 0, max_token_per_req_)
                                   .slice(0, 0, batch_size);

      shared_kv_indices.copy_(kv_lens_batch_offsets_broadcast +
                              shared_kv_offsets);

      auto unshared_kv_offsets =
          unshared_full_kv_offsets.slice(0, 0, batch_size);
      int32_t unshared_kv_len = beam_width * max_decode_step;
      auto unshared_kv_indices =
          full_kv_indices
              .slice(
                  2, max_token_per_req_, max_token_per_req_ + unshared_kv_len)
              .slice(0, 0, batch_size);
      unshared_kv_indices.copy_(unshared_kv_offsets + unshared_kv_begin_offset);

      auto unshared_mask =
          full_kv_mask
              .slice(
                  2, max_token_per_req_, max_token_per_req_ + unshared_kv_len)
              .slice(0, 0, batch_size);
      auto real_max_decode_step_ids_slice =
          real_max_decode_step_ids.slice(0, 0, batch_size);
      unshared_mask.copy_(real_max_decode_step_ids_slice <= current_step);

      unshared_kv_len = current_step + 1;

      auto batch_beam_shared_kv_lens =
          (shared_kv_lens_each_batch.unsqueeze(1).expand({-1, beam_width}) +
           unshared_kv_len)
              .flatten();
      auto cumsum_result = torch::cumsum(batch_beam_shared_kv_lens, 0);
      auto paged_kv_indptr =
          torch::cat({torch::zeros({1}, int32_device_options),
                      cumsum_result.to(int32_device_options)},
                     0);
      auto paged_kv_indices = full_kv_indices.masked_select(full_kv_mask);
      auto paged_kv_last_page_len =
          torch::ones({batch_size * beam_width}, int32_device_options);
      runtime_.worker.prepare_stream_->synchronize();

      NextRoundInputResults results;
      results.paged_kv_indices = paged_kv_indices;
      results.paged_kv_indptr = paged_kv_indptr;
      results.paged_kv_last_page_len = paged_kv_last_page_len;
      promise.setValue(results);
    });
  } else {
    promise.setValue(NextRoundInputResults{});
  }
#endif
  return future;
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::
    prepare_round_input_and_schedule_next(
        ForwardInput& input,
        int32_t round,
        int32_t total_rounds,
        int32_t batch_size,
        int32_t beam_width,
        int32_t max_decode_step,
        const torch::Tensor& top_tokens,
        const BeamSearchTensors& beam_tensors,
        std::optional<folly::SemiFuture<NextRoundInputResults>>&
            next_round_async_result) {
  // Phase A: consume async result for the current round (prepared in last
  // round).
  if (next_round_async_result.has_value()) {
    auto results = std::move(next_round_async_result.value()).get();
    prepare_input_for_current_round(
        input, results, round, top_tokens, beam_tensors);

    // Ensure this future is not consumed twice.
    next_round_async_result.reset();
  }

  // Phase B: schedule async computation for the next round, if any.
  if (round < total_rounds - 1) {
    next_round_async_result =
        compute_next_round_input_async(input.input_params.kv_seq_lens,
                                       round,
                                       batch_size,
                                       beam_width,
                                       max_decode_step);
  }
}

RecWorkerImpl::LlmRecMultiRoundPipeline::FullKvCacheOffsets::FullKvCacheOffsets(
    LlmRecMultiRoundPipeline* multi_round_pipeline) {
#if defined(USE_NPU)
// TODO: implement FullKvCacheOffsets for NPU
#elif defined(USE_CUDA)
  auto device = multi_round_pipeline->runtime().worker.device();
  auto int32_device_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  int32_t max_decode_step = get_rec_multi_round_decode_rounds() - 1;
  full_kv_offsets =
      torch::arange(0,
                    multi_round_pipeline->max_token_per_req_ + max_decode_step,
                    int32_device_options)
          .unsqueeze(0)
          .expand({multi_round_pipeline->max_seqs_per_batch_, -1})
          .unsqueeze(1)
          .expand({-1, multi_round_pipeline->beam_width_, -1});
  full_kv_mask =
      torch::zeros({multi_round_pipeline->max_seqs_per_batch_,
                    multi_round_pipeline->beam_width_,
                    multi_round_pipeline->max_token_per_req_ + max_decode_step},
                   int32_device_options)
          .to(torch::kBool);
  full_kv_indices = torch::zeros_like(full_kv_offsets);

  auto batch_ids =
      torch::arange(
          0, multi_round_pipeline->max_seqs_per_batch_, int32_device_options)
          .unsqueeze(1)
          .unsqueeze(2)
          .expand({-1, multi_round_pipeline->beam_width_, max_decode_step}) *
      (multi_round_pipeline->beam_width_ * max_decode_step);

  auto beams_ids =
      torch::arange(0, multi_round_pipeline->beam_width_, int32_device_options)
          .unsqueeze(0)
          .unsqueeze(2)
          .expand({multi_round_pipeline->max_seqs_per_batch_,
                   -1,
                   max_decode_step}) *
      max_decode_step;

  max_decode_step_ids = torch::arange(0, max_decode_step, int32_device_options)
                            .unsqueeze(0)
                            .unsqueeze(1)
                            .expand({multi_round_pipeline->max_seqs_per_batch_,
                                     multi_round_pipeline->beam_width_,
                                     -1});
  unshared_offsets = batch_ids + beams_ids + max_decode_step_ids;
#endif
}

// ============================================================
// RecWorkerImpl Implementation
// ============================================================

void RecWorkerImpl::initialize_xattention_workspace() {
#if defined(USE_CUDA)
  if (FLAGS_enable_xattention_one_stage) {
    return;
  }
  ::xllm::layer::xattention::XAttentionWorkspace::get_instance().initialize(
      device_);
#endif
}

RecWorkerImpl::RecWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : LLMWorkerImpl(parallel_args, device, options) {
  initialize_xattention_workspace();

  step_threadpool_ = std::make_unique<ThreadPool>(
      options_.rec_worker_max_concurrency(), [this]() mutable {
        device_.set_device();
#if defined(USE_CUDA)
        ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
            .initialize(device_);
        initialize_xattention_workspace();
#endif
      });

  LOG(INFO) << "RecWorkerImpl constructor, rank=" << parallel_args.rank()
            << ", world_size=" << parallel_args.world_size()
            << ", device=" << device << ", is_driver=" << is_driver()
            << ", rec_worker_max_concurrency="
            << options_.rec_worker_max_concurrency();
  if (is_driver()) {
    const int64_t num_threads = std::max<int64_t>(
        1, util::get_int_env("XLLM_REC_INPUT_BUILDER_THREADS", 16));
    input_builder_thread_pool_ =
        std::make_shared<MPMCThreadPool>(static_cast<size_t>(num_threads));
  }
}

RecWorkerImpl::~RecWorkerImpl() {
  // Release model_, model_executor_, eplb_executor_ in destructor to avoid
  // double deletion. Ownership actually belongs to work_pipelines_[0].
  model_.release();
  model_executor_.release();

  if (FLAGS_enable_eplb) {
    eplb_executor_.release();
  }
}

bool RecWorkerImpl::init_model(const std::string& model_weights_path,
                               int32_t random_seed,
                               MasterStatus master_status) {
  if (!WorkerImpl::init_model(model_weights_path, random_seed, master_status)) {
    return false;
  }

  if (FLAGS_enable_eplb) {
    work_pipelines_[0]->runtime().expert_load_data = expert_load_data_;

    for (size_t i = 1; i < work_pipelines_.size(); ++i) {
      work_pipelines_[i]->runtime().expert_load_data =
          work_pipelines_[0]->runtime().expert_load_data.clone();
    }
  }

  return true;
}

bool RecWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";

  // Determine rec model kind and pipeline type
  const auto& model_type = context.get_model_args().model_type();
  rec_model_kind_ = get_rec_model_kind(model_type);
  CHECK(rec_model_kind_ != RecModelKind::kNone)
      << "Unsupported rec model_type: " << model_type;

  // Create concurrent pipeline (not base class pipeline)
  auto pipeline_type = get_rec_pipeline_type(rec_model_kind_);

  size_t pipeline_count =
      static_cast<size_t>(options_.rec_worker_max_concurrency());
  if (pipeline_type == RecPipelineType::kLlmRecMultiRoundPipeline &&
      parallel_args_.world_size() > 1 &&
      enable_rec_multiround_tp_single_model_pipeline()) {
    pipeline_count = 1;
    LOG(INFO) << "REC multi-round local TP uses single model pipeline per rank"
              << ", rank=" << parallel_args_.rank()
              << ", world_size=" << parallel_args_.world_size()
              << ", configured_rec_worker_max_concurrency="
              << options_.rec_worker_max_concurrency()
              << ", effective_pipeline_count=" << pipeline_count;
  }

  // Reserve space for model instances
  work_pipelines_.reserve(pipeline_count);
  for (size_t i = 0; i < pipeline_count; ++i) {
    RecPipelineRuntime runtime(*this);
    runtime.pipeline_index = i;
    auto stream = device_.get_stream_from_pool();
    runtime.stream = std::move(stream);
    auto stream_guard = runtime.stream->set_stream_guard();

    ParallelArgs pipeline_parallel_args = context.get_parallel_args();
#if defined(USE_NPU)
    if (enable_rec_multiround_tp_per_pipeline_atb_comm() &&
        pipeline_type == RecPipelineType::kLlmRecMultiRoundPipeline &&
        pipeline_parallel_args.world_size() > 1) {
      pipeline_parallel_args.atb_tp_comm_domain(std::to_string(i));
      LOG(INFO) << "REC pipeline uses experimental ATB TP comm domain"
                << ", rank=" << pipeline_parallel_args.rank()
                << ", world_size=" << pipeline_parallel_args.world_size()
                << ", pipeline=" << i << ", comm_domain="
                << pipeline_parallel_args.atb_tp_comm_domain();
    }
#endif

    runtime.context =
        std::make_unique<ModelContext>(pipeline_parallel_args,
                                       context.get_model_args(),
                                       context.get_quant_args(),
                                       context.get_tensor_options());

    if (rec_model_kind_ == RecModelKind::kOneRec) {
      runtime.model = create_rec_model(*runtime.context.get());
    } else {
      runtime.model = create_llm_model(*runtime.context.get());
    }
    CHECK(runtime.model != nullptr) << "Failed to create model instance " << i;

    runtime.executor =
        std::make_unique<Executor>(runtime.model.get(),
                                   runtime.context->get_model_args(),
                                   runtime.worker.device(),
                                   runtime.worker.options_);

    if (FLAGS_enable_eplb) {
      runtime.eplb_executor = std::make_unique<EplbExecutor>(
          runtime.model.get(), runtime.worker.device());
    }

    work_pipelines_.emplace_back(create_pipeline(pipeline_type, runtime));
    index_queue_.enqueue(i);
  }

  model_.reset(work_pipelines_[0]->runtime().model.get());
  model_executor_.reset(work_pipelines_[0]->runtime().executor.get());

  // Complete other initialization (EPLB, BeamSearcher, etc.)
  if (FLAGS_enable_beam_search_kernel) {
    beam_searcher_ = std::make_unique<BeamSearcher>();
  }

  if (FLAGS_enable_eplb) {
    eplb_executor_.reset(work_pipelines_[0]->runtime().eplb_executor.get());
  }

  LOG(INFO) << "Created " << work_pipelines_.size()
            << " pipelines for concurrent execution";
  return true;
}

void RecWorkerImpl::load_model(std::unique_ptr<ModelLoader> loader) {
  CHECK(!work_pipelines_.empty())
      << "Model instances are not initialized. Call init_model() first.";

  // Save model weights path to create new loaders for other instances
  std::string model_weights_path = loader->model_weights_path();

  // Load weights for the first model instance (using the original loader)
  work_pipelines_[0]->runtime().model->load_model(std::move(loader));
  LOG(INFO) << "Loaded weights for model instance 0";

  // Create new loaders and load weights for other model instances
  for (size_t i = 1; i < work_pipelines_.size(); ++i) {
    auto model_loader = ModelLoader::create(model_weights_path);
    CHECK(model_loader != nullptr)
        << "Failed to create ModelLoader for model instance " << i;
    work_pipelines_[i]->runtime().model->load_model(std::move(model_loader));
    LOG(INFO) << "Loaded weights for model instance " << i;
  }

  LOG(INFO) << "Loaded weights for all " << work_pipelines_.size() << " models";
}

bool RecWorkerImpl::init_onerec_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";
  device_.set_device();

  model_ = create_rec_model(context);
  CHECK(model_ != nullptr) << "Failed to create rec model.";
  model_executor_ = std::make_unique<Executor>(
      model_.get(), context.get_model_args(), device_, options_);

  if (FLAGS_enable_eplb) {
    eplb_executor_ = std::make_unique<EplbExecutor>(model_.get(), device_);
  }
  return true;
}

ForwardInput RecWorkerImpl::prepare_inputs(Batch& batch) {
  CHECK(!work_pipelines_.empty()) << "RecWorkerImpl is not initialized.";
  return work_pipelines_[0]->prepare_inputs(batch);
}

void RecWorkerImpl::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  LOG(FATAL)
      << "RecWorkerImpl::prepare_work_before_execute should not be called.";
}

void RecWorkerImpl::prepare_multi_modal_data(ForwardInput& processed_inputs) {
  if (!processed_inputs.input_params.mm_data.valid()) {
    return;
  }

  torch::Tensor multi_modal_values;
  torch::Tensor multi_modal_indices;

  const auto& processed_mm_data = processed_inputs.input_params.mm_data;
  if (auto res = processed_mm_data.get<torch::Tensor>("MULTI_MODAL_VALUES")) {
    multi_modal_values = res.value();
  }

  if (auto res = processed_mm_data.get<torch::Tensor>("MULTI_MODAL_INDICES")) {
    multi_modal_indices = res.value();
  }

  if (!multi_modal_values.defined() || !multi_modal_indices.defined()) {
    return;
  }

#if defined(USE_NPU)
  layer::NpuWordEmbedding npu_word_embedding = get_npu_word_embedding();
  torch::Tensor input_tokens_embedding =
      npu_word_embedding(processed_inputs.token_ids, 0);
#else
  layer::WordEmbedding word_embedding = get_word_embedding();
  torch::Tensor input_tokens_embedding =
      word_embedding->forward(processed_inputs.token_ids);
#endif

  std::vector<torch::indexing::TensorIndex> indices = {
      torch::indexing::TensorIndex(multi_modal_indices),
      torch::indexing::Slice()};

  input_tokens_embedding.index_put_(indices, multi_modal_values);
  processed_inputs.input_params.input_embedding = input_tokens_embedding;
}

std::optional<ForwardOutput> RecWorkerImpl::step(const ForwardInput& input) {
  LOG(FATAL) << "RecWorkerImpl::step should not be called.";
  return std::nullopt;
}

folly::SemiFuture<std::optional<ForwardOutput>>
RecWorkerImpl::schedule_step_async(const ForwardInput& input,
                                   size_t index,
                                   bool return_pipeline_index) {
  folly::Promise<std::optional<ForwardOutput>> promise;

  CHECK_LT(index, work_pipelines_.size())
      << "REC pipeline index out of range, index=" << index
      << ", pipelines=" << work_pipelines_.size();
  auto future = promise.getSemiFuture();
  // Copy the input because the scheduled task may run after step_async returns.
  ForwardInput input_copy = input;

  // Use schedule() to assign tasks, letting ThreadPool automatically select
  // idle threads The logic for allocating instance_id happens when the task
  // executes (see lambda below)
  step_threadpool_->schedule_with_tid(
      [this,
       input = std::move(input_copy),
       index,
       return_pipeline_index,
       promise = std::move(promise)]() mutable {
        xllm::ScopeGuard index_guard([&] {
          if (return_pipeline_index) {
            index_queue_.enqueue(index);
          }
        });
        try {
          if (enable_rec_pipeline_concurrency_debug()) {
            LOG(INFO) << "REC pipeline execute begin, rank="
                      << parallel_args_.rank()
                      << ", world_size=" << parallel_args_.world_size()
                      << ", device=" << device_.unwrap()
                      << ", pipeline=" << index
                      << ", thread_id=" << std::this_thread::get_id();
          }
          auto stream_guard =
              work_pipelines_[index]->runtime().stream->set_stream_guard();

          ForwardInput input_on_device;
          work_pipelines_[index]->prepare_work_before_execute(input,
                                                              input_on_device);

          if (hierarchy_kv_cache_transfer_ != nullptr) {
            hierarchy_kv_cache_transfer_->set_layer_synchronizer(
                input_on_device.input_params);
          }

          const auto output = work_pipelines_[index]->step(input_on_device);
          if (enable_rec_pipeline_concurrency_debug()) {
            LOG(INFO) << "REC pipeline execute end, rank="
                      << parallel_args_.rank()
                      << ", device=" << device_.unwrap()
                      << ", pipeline=" << index
                      << ", has_output=" << output.has_value();
          }
          promise.setValue(output);
        } catch (const std::exception& e) {
          LOG(ERROR) << "RecWorkerImpl::step_async failed on pipeline " << index
                     << ": " << e.what();
          promise.setException(
              folly::exception_wrapper(std::current_exception()));
        } catch (...) {
          LOG(ERROR) << "RecWorkerImpl::step_async failed on pipeline " << index
                     << ": unknown exception";
          promise.setException(
              folly::exception_wrapper(std::current_exception()));
        }
      },
      index);

  return future;
}

folly::SemiFuture<std::optional<ForwardOutput>> RecWorkerImpl::step_async(
    const ForwardInput& input) {
  size_t index;
  index_queue_.wait_dequeue(index);
  return schedule_step_async(input, index, /*return_pipeline_index=*/true);
}

folly::SemiFuture<std::optional<ForwardOutput>>
RecWorkerImpl::step_async_with_pipeline_index(const ForwardInput& input,
                                              size_t pipeline_index) {
  return schedule_step_async(
      input, pipeline_index, /*return_pipeline_index=*/false);
}

void RecWorkerImpl::set_pipeline_control_group(size_t pipeline_index,
                                               ProcessGroup* group) {
  CHECK_LT(pipeline_index, work_pipelines_.size())
      << "REC pipeline control group index out of range, index="
      << pipeline_index << ", pipelines=" << work_pipelines_.size();
  CHECK(group != nullptr) << "REC pipeline control group must not be null.";
  work_pipelines_[pipeline_index]->runtime().rec_tp_control_group = group;
  LOG(INFO) << "REC pipeline control group bound, rank="
            << parallel_args_.rank()
            << ", world_size=" << parallel_args_.world_size()
            << ", device=" << device_.unwrap()
            << ", pipeline=" << pipeline_index
            << ", control_group_rank=" << group->rank()
            << ", control_group_world_size=" << group->world_size();
}

// ============================================================
// RecWorkerImpl pipeline factory (static method)
// ============================================================

std::unique_ptr<RecWorkerImpl::RecWorkPipeline> RecWorkerImpl::create_pipeline(
    RecPipelineType type,
    RecPipelineRuntime& runtime) {
  switch (type) {
    case RecPipelineType::kLlmRecDefault:
      return std::make_unique<LlmRecWorkPipeline>(runtime);
    case RecPipelineType::kOneRecDefault:
      return std::make_unique<OneRecWorkPipeline>(runtime);
    case RecPipelineType::kLlmRecMultiRoundPipeline:
      return std::make_unique<LlmRecMultiRoundPipeline>(runtime);
    case RecPipelineType::kOneRecXAttentionPipeline:
      return std::make_unique<OneRecXAttentionWorkPipeline>(runtime);
    default:
      LOG(FATAL) << "Unknown RecWorkerImpl pipeline type: "
                 << static_cast<int>(type);
      return nullptr;
  }
}

}  // namespace xllm
