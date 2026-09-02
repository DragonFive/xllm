/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "framework/kv_cache_transfer/kv_cache_transfer.h"

#include <glog/logging.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <tuple>
#include <unordered_set>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/kv_cache_transfer/push_route.h"
#include "core/util/verbose_trace_logger.h"

#if defined(USE_NPU)
#include "framework/kv_cache_transfer/llm_data_dist_transfer.h"
#endif
#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
#include "framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"
#endif

namespace xllm {

namespace detail {

void append_kv_transfer_trace_request(
    std::vector<KVTransferTraceRequest>* requests,
    const TransferKVInfo& info) {
  CHECK(requests != nullptr);
  auto request_it =
      std::find_if(requests->begin(),
                   requests->end(),
                   [&info](const KVTransferTraceRequest& request) {
                     return request.request_id == info.request_id;
                   });
  if (request_it == requests->end()) {
    requests->emplace_back();
    request_it = std::prev(requests->end());
    request_it->request_id = info.request_id;
    request_it->groups.reserve(info.mappings.size());
  }

  for (const KVTransferMapping& mapping : info.mappings) {
    auto group_it = std::find_if(request_it->groups.begin(),
                                 request_it->groups.end(),
                                 [&mapping](const KVTransferTraceGroup& group) {
                                   return group.group_id == mapping.group_id;
                                 });
    if (group_it == request_it->groups.end()) {
      request_it->groups.emplace_back();
      group_it = std::prev(request_it->groups.end());
      group_it->group_id = mapping.group_id;
    }
    group_it->local_block_count += mapping.local_ids.size();
    group_it->remote_block_count += mapping.remote_ids.size();
    group_it->local_block_ids.insert(group_it->local_block_ids.end(),
                                     mapping.local_ids.begin(),
                                     mapping.local_ids.end());
    group_it->remote_block_ids.insert(group_it->remote_block_ids.end(),
                                      mapping.remote_ids.begin(),
                                      mapping.remote_ids.end());
  }
}

bool operator<(const KVTransferCoverageKey& lhs,
               const KVTransferCoverageKey& rhs) {
  return std::tie(lhs.request_id,
                  lhs.owner_rank,
                  lhs.layer_id,
                  lhs.group_id,
                  lhs.cache_role,
                  lhs.destination_physical_block_id) <
         std::tie(rhs.request_id,
                  rhs.owner_rank,
                  rhs.layer_id,
                  rhs.group_id,
                  rhs.cache_role,
                  rhs.destination_physical_block_id);
}

KVTransferCoverageLedger::KVTransferCoverageLedger(
    std::vector<KVTransferCoverageKey> expected) {
  for (const KVTransferCoverageKey& contribution : expected) {
    if (!expected_.emplace(contribution).second) {
      ++duplicate_count_;
    }
  }
}

KVTransferCoverageRecordResult KVTransferCoverageLedger::record(
    const KVTransferCoverageKey& contribution) {
  if (expected_.find(contribution) == expected_.end()) {
    ++unexpected_count_;
    return KVTransferCoverageRecordResult::UNEXPECTED;
  }
  if (!received_.emplace(contribution).second) {
    ++duplicate_count_;
    return KVTransferCoverageRecordResult::DUPLICATE;
  }
  return KVTransferCoverageRecordResult::RECORDED;
}

bool KVTransferCoverageLedger::is_ready() const {
  return duplicate_count_ == 0 && unexpected_count_ == 0 &&
         received_.size() == expected_.size();
}

std::vector<KVTransferCoverageKey> KVTransferCoverageLedger::missing() const {
  std::vector<KVTransferCoverageKey> missing;
  missing.reserve(expected_.size() - received_.size());
  std::set_difference(expected_.begin(),
                      expected_.end(),
                      received_.begin(),
                      received_.end(),
                      std::back_inserter(missing));
  return missing;
}

std::optional<std::string> validate_llm_data_dist_capability(
    const LlmDataDistCapability& capability) {
  if (!capability.is_npu_backend) {
    return std::string("LlmDataDist requires an NPU build");
  }
  if (capability.transfer_mode != "PUSH") {
    return std::string("LlmDataDist requires kv_cache_transfer_mode=PUSH");
  }
  if (capability.model_type != "glm_moe_dsa") {
    return std::string("LlmDataDist supports only model_type=glm_moe_dsa");
  }
  if (!capability.has_lightning_indexer) {
    return std::string("LlmDataDist requires Lightning Indexer cache");
  }
  if (capability.kv_cache_dtype != "auto") {
    return std::string("LlmDataDist requires kv_cache_dtype=auto");
  }
  if (capability.enable_xtensor) {
    return std::string("LlmDataDist does not support XTensor cache");
  }
  if (capability.has_linear_attention_cache) {
    return std::string("LlmDataDist does not support linear-attention cache");
  }
  if (capability.has_grouped_cache_layout) {
    return std::string("LlmDataDist does not support grouped cache layout");
  }
  if (capability.is_spec_draft) {
    return std::string(
        "LlmDataDist does not support speculative or MTP draft cache");
  }

  const bool supported_prefill =
      capability.instance_role == InstanceRole::PREFILL &&
      capability.dp_size == 1 && capability.cp_size >= 1 &&
      capability.kv_split_size == 1;
  const bool supported_decode =
      capability.instance_role == InstanceRole::DECODE &&
      capability.dp_size == 1 && capability.cp_size == 1 &&
      capability.kv_split_size == 1;
  if (!supported_prefill && !supported_decode) {
    return "LlmDataDist supports only PREFILL(dp=1,cp>=1,kv_split=1) or "
           "DECODE(dp=1,cp=1,kv_split=1); got role=" +
           capability.instance_role.to_string() +
           ",dp=" + std::to_string(capability.dp_size) +
           ",cp=" + std::to_string(capability.cp_size) +
           ",kv_split=" + std::to_string(capability.kv_split_size);
  }
  return std::nullopt;
}

}  // namespace detail

KVCacheLayoutQueryResult KVCacheTransfer::get_kv_cache_layout() { return {}; }

KVTransferNotificationDrainResult
KVCacheTransfer::drain_kv_transfer_notifications(size_t max_notifications) {
  (void)max_notifications;
  return {};
}

bool KVCacheTransfer::validate_transfer_mappings(
    const std::vector<KVTransferMapping>& mappings,
    const std::string& request_id,
    int32_t kv_split_size) {
  if (kv_split_size < 1) {
    LOG(ERROR) << "KV cache transfer requires kv_split_size >= 1, request_id="
               << request_id << ", kv_split_size=" << kv_split_size;
    return false;
  }

  std::unordered_set<int32_t> group_ids;
  group_ids.reserve(mappings.size());
  for (const KVTransferMapping& mapping : mappings) {
    if (!group_ids.emplace(mapping.group_id).second) {
      LOG(ERROR) << "Duplicate KV cache transfer mapping, request_id="
                 << request_id << ", group_id=" << mapping.group_id;
      return false;
    }

    const std::optional<BlockType> block_type =
        block_type_from_cache_group_id(mapping.group_id);
    const bool validate_full_kv_split_coverage =
        kv_split_size > 1 && block_type.has_value() &&
        is_kv_split_cache_block_type(block_type.value());
    const std::unordered_set<uint64_t> local_ids(mapping.local_ids.begin(),
                                                 mapping.local_ids.end());
    if (local_ids.size() != mapping.local_ids.size()) {
      LOG(ERROR) << "Duplicate local KV cache block id, request_id="
                 << request_id << ", group_id=" << mapping.group_id;
      return false;
    }
    const std::unordered_set<uint64_t> remote_ids(mapping.remote_ids.begin(),
                                                  mapping.remote_ids.end());
    if (remote_ids.size() != mapping.remote_ids.size()) {
      LOG(ERROR) << "Duplicate remote KV cache block id, request_id="
                 << request_id << ", group_id=" << mapping.group_id;
      return false;
    }
    const bool has_logical_metadata = !mapping.logical_block_ordinals.empty() ||
                                      !mapping.valid_tokens.empty();
    if (has_logical_metadata &&
        (mapping.remote_ids.size() != mapping.logical_block_ordinals.size() ||
         mapping.remote_ids.size() != mapping.valid_tokens.size())) {
      LOG(ERROR) << "KV cache logical metadata size mismatch, request_id="
                 << request_id << ", group_id=" << mapping.group_id
                 << ", remote=" << mapping.remote_ids.size()
                 << ", logical=" << mapping.logical_block_ordinals.size()
                 << ", valid_tokens=" << mapping.valid_tokens.size();
      return false;
    }
    std::unordered_set<uint64_t> receipt_remote_ids;
    receipt_remote_ids.reserve(mapping.receipt_remote_ids.size());
    for (uint64_t receipt_remote_id : mapping.receipt_remote_ids) {
      if (!receipt_remote_ids.emplace(receipt_remote_id).second ||
          remote_ids.find(receipt_remote_id) == remote_ids.end()) {
        LOG(ERROR) << "Invalid KV receipt destination block, request_id="
                   << request_id << ", group_id=" << mapping.group_id
                   << ", remote_id=" << receipt_remote_id;
        return false;
      }
    }
    if (!mapping.receipt_remote_ids.empty() && !has_logical_metadata) {
      LOG(ERROR) << "KV receipt metadata is missing logical identity, "
                 << "request_id=" << request_id
                 << ", group_id=" << mapping.group_id;
      return false;
    }
    if (!validate_full_kv_split_coverage) {
      if (mapping.local_ids.size() != mapping.remote_ids.size()) {
        LOG(ERROR) << "KV cache transfer mapping size mismatch, request_id="
                   << request_id << ", group_id=" << mapping.group_id
                   << ", local=" << mapping.local_ids.size()
                   << ", remote=" << mapping.remote_ids.size();
        return false;
      }
      continue;
    }

    const size_t local_count = mapping.local_ids.size();
    const size_t remote_count = mapping.remote_ids.size();
    if (local_count == 0) {
      if (remote_count != 0) {
        LOG(ERROR) << "KV-split mapping has remote ids without local ids, "
                   << "request_id=" << request_id
                   << ", group_id=" << mapping.group_id
                   << ", remote=" << remote_count;
        return false;
      }
      continue;
    }

    const size_t split_size = static_cast<size_t>(kv_split_size);
    if (has_logical_metadata) {
      const std::vector<uint64_t>& ordinals = mapping.logical_block_ordinals;
      if (ordinals.front() < static_cast<uint64_t>(mapping.remote_shared_num)) {
        LOG(ERROR) << "KV-split mapping includes a shared destination block, "
                   << "request_id=" << request_id
                   << ", group_id=" << mapping.group_id
                   << ", first_ordinal=" << ordinals.front()
                   << ", remote_shared_num=" << mapping.remote_shared_num;
        return false;
      }
      for (size_t i = 1; i < ordinals.size(); ++i) {
        if (ordinals[i - 1] == std::numeric_limits<uint64_t>::max() ||
            ordinals[i] != ordinals[i - 1] + 1) {
          LOG(ERROR) << "KV-split logical block ordinals are not contiguous, "
                     << "request_id=" << request_id
                     << ", group_id=" << mapping.group_id
                     << ", previous=" << ordinals[i - 1]
                     << ", current=" << ordinals[i];
          return false;
        }
      }

      const uint64_t first_source_ordinal =
          ordinals.front() / static_cast<uint64_t>(split_size);
      const uint64_t last_source_ordinal =
          ordinals.back() / static_cast<uint64_t>(split_size);
      const uint64_t expected_local_count =
          last_source_ordinal - first_source_ordinal + 1;
      if (expected_local_count != static_cast<uint64_t>(local_count)) {
        LOG(ERROR) << "KV-split logical source coverage mismatch, request_id="
                   << request_id << ", group_id=" << mapping.group_id
                   << ", local=" << local_count
                   << ", expected_local=" << expected_local_count
                   << ", first_ordinal=" << ordinals.front()
                   << ", last_ordinal=" << ordinals.back()
                   << ", kv_split_size=" << kv_split_size;
        return false;
      }
      continue;
    }

    if (local_count > std::numeric_limits<size_t>::max() / split_size) {
      LOG(ERROR) << "KV-split mapping coverage size overflow, request_id="
                 << request_id << ", group_id=" << mapping.group_id
                 << ", local=" << local_count
                 << ", kv_split_size=" << kv_split_size;
      return false;
    }
    const size_t max_remote_count = local_count * split_size;
    const size_t min_remote_count = max_remote_count - split_size + 1;
    if (remote_count < min_remote_count || remote_count > max_remote_count) {
      LOG(ERROR) << "KV-split mapping remote coverage mismatch, request_id="
                 << request_id << ", group_id=" << mapping.group_id
                 << ", local=" << local_count << ", remote=" << remote_count
                 << ", kv_split_size=" << kv_split_size
                 << ", expected_remote_range=[" << min_remote_count << ", "
                 << max_remote_count << "]";
      return false;
    }
  }
  return true;
}

bool KVCacheTransfer::validate_transfer_mappings(
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    int32_t kv_split_size) {
  for (const TransferKVInfo& info : transfer_kv_infos) {
    const bool strict_readiness = !info.decode_kv_manifest.empty();
    if (strict_readiness != (info.attempt_epoch > 0) ||
        strict_readiness != (info.allocation_generation > 0)) {
      LOG(ERROR) << "Incomplete Decode KV readiness identity, request_id="
                 << info.request_id;
      return false;
    }
    if (!validate_transfer_mappings(
            info.mappings, info.request_id, kv_split_size)) {
      return false;
    }
  }
  return true;
}

folly::SemiFuture<bool> KVCacheTransfer::pull_kv_blocks_async(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<KVTransferMapping>& mappings) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  if (!validate_transfer_mappings(
          mappings, /*request_id=*/"PULL", /*kv_split_size=*/1)) {
    promise.setValue(false);
    return future;
  }
  threadpool_.schedule([this,
                        src_cluster_id,
                        src_addr,
                        mappings,
                        promise = std::move(promise)]() mutable {
    const bool success = pull_kv_blocks(src_cluster_id, src_addr, mappings);
    promise.setValue(success);
  });
  return future;
}

// In KV-split mode, destination logical ordinals identify both the source
// logical block and its owner shard. Legacy mappings without ordinals retain
// the positional stride behavior for aligned handoffs only.
std::vector<TransferKVInfo> filter_kv_split_infos(
    int32_t kv_split_rank,
    int32_t kv_split_size,
    const std::vector<TransferKVInfo>& kv_infos) {
  CHECK_GT(kv_split_size, 0);
  CHECK_GE(kv_split_rank, 0);
  CHECK_LT(kv_split_rank, kv_split_size);
  std::vector<TransferKVInfo> filtered_kv_infos;
  for (const TransferKVInfo& kv_info : kv_infos) {
    TransferKVInfo filtered = kv_info;
    for (KVTransferMapping& mapping : filtered.mappings) {
      const std::optional<BlockType> block_type =
          block_type_from_cache_group_id(mapping.group_id);
      if (!block_type.has_value() ||
          !is_kv_split_cache_block_type(block_type.value())) {
        continue;
      }
      const std::vector<uint64_t> local_ids = mapping.local_ids;
      const std::vector<uint64_t> remote_ids = mapping.remote_ids;
      const std::vector<uint64_t> logical_block_ordinals =
          mapping.logical_block_ordinals;
      const std::vector<uint32_t> valid_tokens = mapping.valid_tokens;
      const bool has_logical_metadata =
          !logical_block_ordinals.empty() || !valid_tokens.empty();
      if (has_logical_metadata) {
        CHECK_EQ(remote_ids.size(), logical_block_ordinals.size());
        CHECK_EQ(remote_ids.size(), valid_tokens.size());
      }
      const std::unordered_set<uint64_t> receipt_remote_ids(
          mapping.receipt_remote_ids.begin(), mapping.receipt_remote_ids.end());
      mapping.local_ids.clear();
      mapping.remote_ids.clear();
      mapping.logical_block_ordinals.clear();
      mapping.valid_tokens.clear();
      mapping.receipt_remote_ids.clear();
      mapping.local_ids.reserve(local_ids.size());
      mapping.remote_ids.reserve(local_ids.size());
      mapping.logical_block_ordinals.reserve(local_ids.size());
      mapping.valid_tokens.reserve(local_ids.size());
      mapping.receipt_remote_ids.reserve(local_ids.size());

      if (has_logical_metadata) {
        const uint64_t split_size = static_cast<uint64_t>(kv_split_size);
        const uint64_t source_origin =
            logical_block_ordinals.front() / split_size;
        for (size_t remote_idx = 0; remote_idx < remote_ids.size();
             ++remote_idx) {
          const uint64_t logical_ordinal = logical_block_ordinals[remote_idx];
          if (logical_ordinal % split_size !=
              static_cast<uint64_t>(kv_split_rank)) {
            continue;
          }
          const uint64_t source_ordinal = logical_ordinal / split_size;
          CHECK_GE(source_ordinal, source_origin);
          const uint64_t local_idx = source_ordinal - source_origin;
          CHECK_LT(local_idx, static_cast<uint64_t>(local_ids.size()));
          mapping.local_ids.emplace_back(
              local_ids[static_cast<size_t>(local_idx)]);
          mapping.remote_ids.emplace_back(remote_ids[remote_idx]);
          mapping.logical_block_ordinals.emplace_back(logical_ordinal);
          mapping.valid_tokens.emplace_back(valid_tokens[remote_idx]);
          if (receipt_remote_ids.find(remote_ids[remote_idx]) !=
              receipt_remote_ids.end()) {
            mapping.receipt_remote_ids.emplace_back(remote_ids[remote_idx]);
          }
        }
      } else {
        for (size_t k = 0; k < local_ids.size(); ++k) {
          const size_t remote_idx = static_cast<size_t>(kv_split_rank) +
                                    k * static_cast<size_t>(kv_split_size);
          if (remote_idx >= remote_ids.size()) {
            break;
          }
          mapping.local_ids.emplace_back(local_ids[k]);
          mapping.remote_ids.emplace_back(remote_ids[remote_idx]);
          if (receipt_remote_ids.find(remote_ids[remote_idx]) !=
              receipt_remote_ids.end()) {
            mapping.receipt_remote_ids.emplace_back(remote_ids[remote_idx]);
          }
        }
      }
    }
    const bool has_mapping = std::any_of(filtered.mappings.begin(),
                                         filtered.mappings.end(),
                                         [](const KVTransferMapping& mapping) {
                                           return !mapping.local_ids.empty() &&
                                                  !mapping.remote_ids.empty();
                                         });
    if (has_mapping) {
      filtered_kv_infos.push_back(std::move(filtered));
    }
  }
  return filtered_kv_infos;
}

std::vector<std::string> KVCacheTransfer::rotate_dst_rank(
    const std::vector<std::string>& keys,
    int32_t kv_split_rank) {
  int32_t offset = kv_split_rank;
  std::vector<std::string> rotated_keys;
  auto sorted_keys = keys;
  std::sort(sorted_keys.begin(), sorted_keys.end());
  for (int32_t i = 0; i < keys.size(); i++) {
    rotated_keys.emplace_back(sorted_keys[(i + offset) % sorted_keys.size()]);
  }
  return rotated_keys;
}

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
folly::SemiFuture<bool> KVCacheTransfer::push_kv_blocks_async(
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    const ParallelArgs& parallel_args,
    std::shared_ptr<KVPushSynchronizerImpl> layer_synchronizer,
    bool is_spec_draft) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
#if defined(USE_NPU)
  if (layer_synchronizer != nullptr &&
      VerboseTraceLogger::get_instance().enabled()) {
    std::vector<detail::KVTransferTraceRequest> trace_requests;
    for (const TransferKVInfo& info : transfer_kv_infos) {
      detail::append_kv_transfer_trace_request(&trace_requests, info);
    }
    LayerSynchronizerTraceContext trace_context;
    trace_context.request_ids.reserve(trace_requests.size());
    for (const detail::KVTransferTraceRequest& request : trace_requests) {
      trace_context.request_ids.emplace_back(request.request_id);
    }
    trace_context.source_rank = parallel_args.rank();
    trace_context.cp_rank = parallel_args.cp_rank();
    trace_context.kv_split_rank = parallel_args.kv_split_rank();
    trace_context.kv_split_size = parallel_args.kv_split_size_effective();
    layer_synchronizer->set_trace_context(std::move(trace_context));
  }
#endif
  threadpool_.schedule([this,
                        transfer_kv_infos,
                        parallel_args,
                        layer_synchronizer,
                        is_spec_draft,
                        promise = std::move(promise)]() mutable {
    std::unordered_map<std::string, KVCacheInfo> merged_kv_infos;
    std::vector<TransferKVInfo> filtered_kv_infos;
    const std::vector<TransferKVInfo>* kv_infos = &transfer_kv_infos;
    // Filter when KV is actually sharded across ranks. When
    // kv_split_size==1 (each CP rank holds a full KV replica) the filter
    // degenerates to a copy, so we skip it and let each rank consume
    // remote_ids 1:1.
    const int32_t kv_split_size = parallel_args.kv_split_size_effective();
    if (!validate_transfer_mappings(*kv_infos, kv_split_size)) {
      promise.setValue(false);
      return;
    }
    if (!is_kv_push_representative(
            parallel_args.cp_rank(), parallel_args.cp_size(), kv_split_size)) {
      if (VerboseTraceLogger::get_instance().enabled()) {
        std::vector<detail::KVTransferTraceRequest> trace_requests;
        for (const TransferKVInfo& info : *kv_infos) {
          detail::append_kv_transfer_trace_request(&trace_requests, info);
        }
        for (const detail::KVTransferTraceRequest& request : trace_requests) {
          XLLM_VERBOSE_TRACE()
              << "event=kv_push_replica_skipped request-id="
              << request.request_id << " source-rank=" << parallel_args.rank()
              << " cp-rank=" << parallel_args.cp_rank()
              << " kv-shard=" << parallel_args.kv_split_rank()
              << " kv-split-size=" << kv_split_size;
        }
      }
      promise.setValue(true);
      return;
    }
    if (kv_split_size > 1) {
      filtered_kv_infos = filter_kv_split_infos(
          parallel_args.kv_split_rank(), kv_split_size, *kv_infos);
      kv_infos = &filtered_kv_infos;
      if (kv_infos->empty()) {
        promise.setValue(true);
        return;
      }
    }
    if (!validate_transfer_mappings(*kv_infos, /*kv_split_size=*/1)) {
      promise.setValue(false);
      return;
    }
    merge_kv_blocks(merged_kv_infos, *kv_infos, parallel_args);
    bool success = true;
    if (!merged_kv_infos.empty()) {
      success = this->push_kv_blocks(merged_kv_infos,
                                     layer_synchronizer,
                                     is_spec_draft,
                                     parallel_args.kv_split_rank(),
                                     parallel_args.kv_split_size_effective());
    }
    promise.setValue(success);
  });
  return future;
}
#endif

void KVCacheTransfer::merge_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    const ParallelArgs& parallel_args) {
  // Obtain the parallel parameters of the source instance.
  // When CP is enabled on the P side, the per-DP worker count is
  // cp_size * tp_size. We need the *actual* TP size (excluding CP) so that
  // src_dp_local_tp_rank correctly reflects only the TP dimension.
  // Using cp_size * tp_size here would make CP rank > 0 workers appear to
  // have a tp_rank >= dst_world_size, causing the linked_dp_ranks filter to
  // skip all requests for those workers.
  int32_t src_rank = parallel_args.rank();
  int32_t src_dp_size = parallel_args.dp_size();
  int32_t src_world_size = parallel_args.world_size();
  int32_t src_cp_size = parallel_args.cp_size();
  CHECK_GT(src_dp_size, 0);
  CHECK_GT(src_cp_size, 0);
  CHECK_EQ(src_world_size % (src_dp_size * src_cp_size), 0)
      << "Invalid Prefill DP/CP topology for KV PUSH routing";
  int32_t src_tp_size = src_world_size / src_dp_size / src_cp_size;
  int32_t src_dp_local_tp_rank = src_rank % src_tp_size;
  auto append_mappings = [](std::vector<KVTransferMapping>& dst,
                            const std::vector<KVTransferMapping>& src) {
    for (const KVTransferMapping& src_mapping : src) {
      auto it = std::find_if(dst.begin(),
                             dst.end(),
                             [&src_mapping](const KVTransferMapping& mapping) {
                               return mapping.group_id == src_mapping.group_id;
                             });
      if (it == dst.end()) {
        dst.emplace_back(src_mapping);
        continue;
      }
      it->local_ids.insert(it->local_ids.end(),
                           src_mapping.local_ids.begin(),
                           src_mapping.local_ids.end());
      it->remote_ids.insert(it->remote_ids.end(),
                            src_mapping.remote_ids.begin(),
                            src_mapping.remote_ids.end());
      it->logical_block_ordinals.insert(
          it->logical_block_ordinals.end(),
          src_mapping.logical_block_ordinals.begin(),
          src_mapping.logical_block_ordinals.end());
      it->valid_tokens.insert(it->valid_tokens.end(),
                              src_mapping.valid_tokens.begin(),
                              src_mapping.valid_tokens.end());
      it->receipt_remote_ids.insert(it->receipt_remote_ids.end(),
                                    src_mapping.receipt_remote_ids.begin(),
                                    src_mapping.receipt_remote_ids.end());
    }
  };
  for (auto& info : transfer_kv_infos) {
    // Obtain the parallel parameters of the destination instance.
    int32_t dst_dp_rank = info.dp_rank;
    int32_t dst_dp_size = info.remote_instance_info.dp_size;
    int32_t dst_world_size = info.remote_instance_info.cluster_ids.size();
    int32_t dst_tp_size = dst_world_size / dst_dp_size;
    // Get the DP groups of the destination instance connected to the current
    // worker.
    std::unordered_set<int32_t> linked_dp_ranks;
    for (int32_t i = src_dp_local_tp_rank; i < dst_world_size;
         i += src_tp_size) {
      int32_t linked_dp_rank = i / dst_tp_size;
      linked_dp_ranks.emplace(linked_dp_rank);
    }
    // If the target DP rank of the request is not linked to the current worker,
    // skip the request.
    if (linked_dp_ranks.find(dst_dp_rank) == linked_dp_ranks.end()) {
      continue;
    }
    // The current worker needs to push the KV Cache to all workers in the
    // destination DP group it is connected to.
    for (int32_t i =
             src_dp_local_tp_rank % dst_tp_size + dst_tp_size * dst_dp_rank;
         i < dst_tp_size * (dst_dp_rank + 1);
         i += src_tp_size) {
      uint64_t dst_cluster_id = info.remote_instance_info.cluster_ids[i];
      auto& dst_addr = info.remote_instance_info.addrs[i];
      std::string key = std::to_string(dst_cluster_id) + "_" + dst_addr;
      // Merge all kv blocks with the same destination worker into a single
      // vector.
      if (merged_kv_infos.find(key) == merged_kv_infos.end()) {
        KVCacheInfo kv_info;
        kv_info.dst_cluster_id = dst_cluster_id;
        kv_info.dst_addr = dst_addr;
        kv_info.source_worker_rank = src_rank;
        kv_info.destination_worker_rank = i;
        append_mappings(kv_info.mappings, info.mappings);
        if (!info.decode_kv_manifest.empty()) {
          kv_info.receipt_infos.emplace_back(info);
        }

        // XTensor mode: copy destination offsets
        if (!info.dst_xtensor_layer_offsets.empty()) {
          kv_info.dst_xtensor_layer_offsets = info.dst_xtensor_layer_offsets;
        }
        detail::append_kv_transfer_trace_request(&kv_info.trace_requests, info);
        merged_kv_infos[key] = std::move(kv_info);
      } else {
        CHECK_EQ(merged_kv_infos[key].source_worker_rank, src_rank);
        CHECK_EQ(merged_kv_infos[key].destination_worker_rank, i);
        append_mappings(merged_kv_infos[key].mappings, info.mappings);
        if (!info.decode_kv_manifest.empty()) {
          merged_kv_infos[key].receipt_infos.emplace_back(info);
        }

        // XTensor mode: merge destination offsets (append to each layer)
        if (!info.dst_xtensor_layer_offsets.empty()) {
          auto& existing = merged_kv_infos[key].dst_xtensor_layer_offsets;
          // Initialize if not already done
          if (existing.empty()) {
            existing = info.dst_xtensor_layer_offsets;
          } else {
            // Append offsets for each layer
            for (size_t layer = 0;
                 layer < info.dst_xtensor_layer_offsets.size() &&
                 layer < existing.size();
                 ++layer) {
              existing[layer].k_offsets.insert(
                  existing[layer].k_offsets.end(),
                  info.dst_xtensor_layer_offsets[layer].k_offsets.begin(),
                  info.dst_xtensor_layer_offsets[layer].k_offsets.end());
              existing[layer].v_offsets.insert(
                  existing[layer].v_offsets.end(),
                  info.dst_xtensor_layer_offsets[layer].v_offsets.begin(),
                  info.dst_xtensor_layer_offsets[layer].v_offsets.end());
            }
          }
        }
        detail::append_kv_transfer_trace_request(
            &merged_kv_infos[key].trace_requests, info);
      }
    }
  }
}

std::shared_ptr<KVCacheTransfer> KVCacheTransferFactory::create(
    const std::string& transfer_type,
    uint16_t transfer_listen_port,
    InstanceRole instance_role,
    const Device& device,
    bool enable_lighting_indexer,
    const std::string& model_type,
    const std::string& model_id) {
  std::shared_ptr<KVCacheTransfer> transfer;

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
  LOG(INFO) << "Create KVCacheTransfer backend: " << transfer_type;
  if (transfer_type == "LlmDataDist") {
#if defined(USE_NPU)
    transfer = std::make_shared<LlmDataDistTransfer>(
        transfer_listen_port, instance_role, enable_lighting_indexer);
#else
    LOG(ERROR) << "LlmDataDist KV cache transfer requires an NPU build.";
#endif
  } else if (transfer_type == "Mooncake") {
    const int32_t device_id = device.index();
    std::shared_ptr<MooncakeKVCacheTransferBase> mooncake_transfer;
#if defined(USE_NPU)
    if (::xllm::KVCacheConfig::get_instance().enable_xtensor()) {
      auto xtensor_transfer = std::make_shared<MooncakeKVCacheTransferXTensor>(
          device_id, transfer_listen_port, device);
      if (!model_id.empty()) {
        xtensor_transfer->set_model_id(model_id);
        LOG(INFO)
            << "XTensor mode enabled for MooncakeKVCacheTransfer, model_id="
            << model_id;
      }
      mooncake_transfer = xtensor_transfer;
    } else {
      mooncake_transfer = std::make_shared<MooncakeKVCacheTransferDefault>(
          device_id, transfer_listen_port, device, model_type);
    }
#else
    mooncake_transfer = std::make_shared<MooncakeKVCacheTransferDefault>(
        device_id, transfer_listen_port, device, model_type);
#endif
    transfer = mooncake_transfer;
  } else {
    LOG(ERROR) << "Unsupported KVCacheTransfer backend: " << transfer_type;
  }
#else
  LOG(ERROR) << "KV cache transfer backend " << transfer_type
             << " is not available in this build.";
#endif

  return transfer;
}

}  // namespace xllm
