/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/kv_cache_transfer/llm_data_dist_transfer.h"

#include <glog/logging.h>

#include <algorithm>
#include <map>
#include <utility>

#include "common/macros.h"
#include "util/net.h"
#include "util/verbose_trace_logger.h"

namespace xllm {
namespace {

const std::map<torch::ScalarType, ge::DataType> kScalarTypeToDtype = {
    {torch::kBool, ge::DT_BOOL},
    {torch::kByte, ge::DT_UINT8},
    {torch::kChar, ge::DT_INT8},
    {torch::kShort, ge::DT_INT16},
    {torch::kInt, ge::DT_INT32},
    {torch::kLong, ge::DT_INT64},
    {torch::kBFloat16, ge::DT_BF16},
    {torch::kHalf, ge::DT_FLOAT16},
    {torch::kFloat, ge::DT_FLOAT},
    {torch::kDouble, ge::DT_DOUBLE},
};

ge::DataType dtype_to_ge_dtype(torch::ScalarType dtype) {
  const auto it = kScalarTypeToDtype.find(dtype);
  CHECK(it != kScalarTypeToDtype.cend()) << "Unsupported data type: " << dtype;
  return it->second;
}

bool is_link_success(llm_datadist::Status status) {
  return status == llm_datadist::LLM_SUCCESS ||
         status == llm_datadist::LLM_ALREADY_LINK ||
         status == llm_datadist::LLM_EXIST_LINK;
}

bool is_unlink_success(llm_datadist::Status status) {
  return status == llm_datadist::LLM_SUCCESS ||
         status == llm_datadist::LLM_NOT_YET_LINK;
}

const char* coverage_record_result_to_string(
    detail::KVTransferCoverageRecordResult result) {
  switch (result) {
    case detail::KVTransferCoverageRecordResult::RECORDED:
      return "recorded";
    case detail::KVTransferCoverageRecordResult::DUPLICATE:
      return "duplicate";
    case detail::KVTransferCoverageRecordResult::UNEXPECTED:
      return "unexpected";
  }
  return "unknown";
}

}  // namespace

LlmDataDistTransfer::LlmDataDistTransfer(uint16_t listen_port,
                                         InstanceRole instance_role,
                                         bool enable_lighting_indexer)
    : listen_port_(listen_port) {
  CHECK(enable_lighting_indexer)
      << "LlmDataDist requires a model with Lightning Indexer cache.";
  if (instance_role == InstanceRole::PREFILL) {
    role_ = llm_datadist::LlmRole::kPrompt;
  } else if (instance_role == InstanceRole::DECODE) {
    role_ = llm_datadist::LlmRole::kDecoder;
  } else {
    LOG(FATAL) << "LlmDataDist requires PREFILL or DECODE instance_role, got "
               << instance_role.to_string();
  }

  host_ip_ = net::get_local_ip_addr();
  CHECK(!host_ip_.empty()) << "Failed to get host IP for LlmDataDist.";
  cluster_id_ = net::convert_ip_port_to_uint64(host_ip_, listen_port_);
  llm_data_dist_ =
      std::make_shared<llm_datadist::LlmDataDist>(cluster_id_, role_);
}

LlmDataDistTransfer::~LlmDataDistTransfer() { finalize(); }

void LlmDataDistTransfer::initialize(int32_t device_id) {
  std::map<llm_datadist::AscendString, llm_datadist::AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = std::to_string(device_id).c_str();
  if (role_ == llm_datadist::LlmRole::kPrompt) {
    const std::string local_ip_info =
        host_ip_ + ":" + std::to_string(listen_port_);
    options[llm_datadist::OPTION_LISTEN_IP_INFO] = local_ip_info.c_str();
  }

  const llm_datadist::Status status = llm_data_dist_->Initialize(options);
  CHECK_EQ(status, llm_datadist::LLM_SUCCESS)
      << "Initialize LlmDataDist failed, status=" << std::hex << status;
  initialized_ = true;
  LOG(INFO) << "Initialized LlmDataDist, cluster_id=" << cluster_id_
            << ", role="
            << (role_ == llm_datadist::LlmRole::kPrompt ? "PREFILL" : "DECODE");
}

void LlmDataDistTransfer::finalize() {
  if (!initialized_) {
    return;
  }
  llm_data_dist_->Finalize();
  initialized_ = false;
  linked_cluster_ids_.clear();
}

void LlmDataDistTransfer::register_kv_cache(std::vector<KVCache>& kv_caches,
                                            const KVCacheShape& kv_cache_shape,
                                            torch::ScalarType dtype) {
  UNUSED_PARAMETER(kv_cache_shape);
  UNUSED_PARAMETER(dtype);
  register_layer_caches(kv_caches);
}

void LlmDataDistTransfer::register_kv_cache_spec(
    std::vector<KVCache>& kv_caches,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
  UNUSED_PARAMETER(kv_caches);
  UNUSED_PARAMETER(kv_cache_shape);
  UNUSED_PARAMETER(dtype);
  LOG(FATAL) << "LlmDataDist does not support speculative or MTP draft cache.";
}

void LlmDataDistTransfer::free_kv_cache() { layer_registered_caches_.clear(); }

void LlmDataDistTransfer::get_cache_info(uint64_t& cluster_id,
                                         std::string& addr) {
  cluster_id = cluster_id_;
  addr = host_ip_;
}

bool LlmDataDistTransfer::link_clusters(
    const std::vector<uint64_t>& cluster_ids,
    const std::vector<std::string>& remote_addrs,
    const std::vector<uint16_t>& ports) {
  if (cluster_ids.size() != remote_addrs.size() ||
      cluster_ids.size() != ports.size()) {
    LOG(ERROR) << "LlmDataDist cluster endpoint size mismatch: cluster_ids="
               << cluster_ids.size() << ", addrs=" << remote_addrs.size()
               << ", ports=" << ports.size();
    return false;
  }

  std::vector<llm_datadist::ClusterInfo> clusters;
  std::vector<size_t> source_indices;
  std::unordered_set<uint64_t> pending_cluster_ids = linked_cluster_ids_;
  clusters.reserve(cluster_ids.size());
  source_indices.reserve(cluster_ids.size());
  for (size_t i = 0; i < cluster_ids.size(); ++i) {
    if (!pending_cluster_ids.insert(cluster_ids[i]).second) {
      continue;
    }
    clusters.emplace_back(
        create_cluster_info(cluster_ids[i], remote_addrs[i], ports[i]));
    source_indices.emplace_back(i);
  }
  if (clusters.empty()) {
    return true;
  }

  std::vector<llm_datadist::Status> statuses;
  const llm_datadist::Status status = llm_data_dist_->LinkLlmClusters(
      clusters, statuses, /*timeout_in_millis=*/60000);
  if (status != llm_datadist::LLM_SUCCESS) {
    LOG(ERROR) << "LinkLlmClusters failed, status=" << std::hex << status;
    return false;
  }
  if (statuses.size() != clusters.size()) {
    LOG(ERROR) << "LinkLlmClusters returned " << statuses.size()
               << " per-cluster statuses for " << clusters.size()
               << " clusters.";
    return false;
  }

  bool success = true;
  for (size_t i = 0; i < statuses.size(); ++i) {
    const size_t source_index = source_indices[i];
    if (!is_link_success(statuses[i])) {
      LOG(ERROR) << "LinkLlmClusters failed for cluster_id="
                 << cluster_ids[source_index]
                 << ", addr=" << remote_addrs[source_index]
                 << ", port=" << ports[source_index] << ", status=" << std::hex
                 << statuses[i];
      success = false;
      continue;
    }
    linked_cluster_ids_.insert(cluster_ids[source_index]);
  }
  return success;
}

bool LlmDataDistTransfer::unlink_cluster(const uint64_t& cluster_id,
                                         const std::string& remote_addr,
                                         uint16_t port,
                                         bool force_flag) {
  if (linked_cluster_ids_.find(cluster_id) == linked_cluster_ids_.end()) {
    return true;
  }

  std::vector<llm_datadist::ClusterInfo> clusters = {
      create_cluster_info(cluster_id, remote_addr, port)};
  std::vector<llm_datadist::Status> statuses;
  const llm_datadist::Status status = llm_data_dist_->UnlinkLlmClusters(
      clusters, statuses, /*timeout_in_millis=*/1000, force_flag);
  if (status != llm_datadist::LLM_SUCCESS) {
    LOG(ERROR) << "UnlinkLlmClusters failed for cluster_id=" << cluster_id
               << ", status=" << std::hex << status;
    return false;
  }
  if (statuses.size() != 1 || !is_unlink_success(statuses.front())) {
    LOG(ERROR) << "UnlinkLlmClusters failed for cluster_id=" << cluster_id
               << ", per-cluster status count=" << statuses.size()
               << (statuses.empty()
                       ? std::string()
                       : ", status=" + std::to_string(statuses.front()));
    return false;
  }

  linked_cluster_ids_.erase(cluster_id);
  return true;
}

bool LlmDataDistTransfer::pull_kv_blocks(
    uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<KVTransferMapping>& mappings) {
  UNUSED_PARAMETER(src_cluster_id);
  UNUSED_PARAMETER(src_addr);
  UNUSED_PARAMETER(mappings);
  LOG(ERROR) << "LlmDataDist backend supports PUSH mode only; PULL is not "
                "supported.";
  return false;
}

bool LlmDataDistTransfer::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
    bool is_spec_draft,
    int32_t kv_split_rank,
    int32_t kv_split_size) {
  if (is_spec_draft) {
    LOG(ERROR) << "LlmDataDist does not support speculative or MTP draft "
                  "cache PUSH.";
    return false;
  }
  return push_layer_caches(layer_registered_caches_,
                           merged_kv_infos,
                           layer_synchronizer,
                           kv_split_rank,
                           kv_split_size);
}

LlmDataDistTransfer::RegisteredCache LlmDataDistTransfer::register_cache_tensor(
    int64_t layer_id,
    const KVCacheTensor& cache_tensor) {
  const torch::Tensor& tensor = cache_tensor.tensor;
  CHECK(tensor.defined() && tensor.numel() > 0)
      << cache_tensor.role.to_string() << " cache is not allocated at layer "
      << layer_id;

  const uintptr_t tensor_addr = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  const std::vector<uint64_t> addrs = {static_cast<uint64_t>(tensor_addr)};
  RegisteredCache registered_cache{
      cache_tensor.role, cache_tensor.group_id, llm_datadist::Cache{}, tensor};
  registered_cache.cache.tensor_addrs = {tensor_addr};

  llm_datadist::CacheDesc& desc = registered_cache.cache.cache_desc;
  desc.num_tensors = 1;
  desc.data_type = dtype_to_ge_dtype(tensor.scalar_type());
  desc.shape = tensor.sizes().vec();

  const llm_datadist::Status status = llm_data_dist_->RegisterKvCache(
      desc, addrs, {}, registered_cache.cache.cache_id);
  CHECK_EQ(status, llm_datadist::LLM_SUCCESS)
      << "Register " << cache_tensor.role.to_string()
      << " cache failed at layer " << layer_id << ", status=" << std::hex
      << status;
  VLOG(5) << "Registered LlmDataDist cache: layer=" << layer_id
          << ", role=" << cache_tensor.role.to_string()
          << ", group_id=" << cache_tensor.group_id
          << ", cache_id=" << registered_cache.cache.cache_id
          << ", shape=" << tensor.sizes();
  return registered_cache;
}

void LlmDataDistTransfer::register_layer_caches(
    std::vector<KVCache>& kv_caches) {
  CHECK(!kv_caches.empty()) << "KV caches must be allocated before register.";
  layer_registered_caches_.clear();
  layer_registered_caches_.resize(kv_caches.size());

  for (size_t layer_id = 0; layer_id < kv_caches.size(); ++layer_id) {
    for (const KVCacheTensor& cache_tensor :
         kv_caches[layer_id].get_cache_tensors()) {
      layer_registered_caches_[layer_id].emplace_back(
          register_cache_tensor(static_cast<int64_t>(layer_id), cache_tensor));
    }
    CHECK(!layer_registered_caches_[layer_id].empty())
        << "No cache tensor registered at layer " << layer_id;
  }
}

bool LlmDataDistTransfer::push_layer_caches(
    const LayerRegisteredCaches& layer_registered_caches,
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
    int32_t kv_split_rank,
    int32_t kv_split_size) {
  if (layer_synchronizer == nullptr) {
    LOG(ERROR) << "LlmDataDist PUSH requires a layer synchronizer.";
    return false;
  }

  std::vector<std::string> keys;
  keys.reserve(merged_kv_infos.size());
  for (const auto& [key, unused] : merged_kv_infos) {
    UNUSED_PARAMETER(unused);
    keys.emplace_back(key);
  }
  if (kv_split_size > 1) {
    keys = rotate_dst_rank(keys, kv_split_rank);
  }

  bool result = true;
  const bool trace_enabled = VerboseTraceLogger::get_instance().enabled();
  using CoverageLedgerKey = std::pair<std::string, std::string>;
  std::map<CoverageLedgerKey, detail::KVTransferCoverageLedger>
      coverage_ledgers;
  for (const std::string& key : keys) {
    const KVCacheInfo& kv_info = merged_kv_infos.at(key);
    for (const detail::KVTransferTraceRequest& request :
         kv_info.trace_requests) {
      std::vector<detail::KVTransferCoverageKey> expected;
      for (size_t layer_index = 0; layer_index < layer_registered_caches.size();
           ++layer_index) {
        for (const RegisteredCache& registered_cache :
             layer_registered_caches[layer_index]) {
          const auto group_it = std::find_if(
              request.groups.begin(),
              request.groups.end(),
              [&registered_cache](const detail::KVTransferTraceGroup& group) {
                return group.group_id == registered_cache.group_id;
              });
          if (group_it == request.groups.end()) {
            continue;
          }
          for (uint64_t destination_physical_block_id :
               group_it->remote_block_ids) {
            expected.emplace_back(detail::KVTransferCoverageKey{
                request.request_id,
                kv_split_rank,
                static_cast<int64_t>(layer_index),
                registered_cache.group_id,
                static_cast<KVCacheTensorRole::Value>(registered_cache.role),
                destination_physical_block_id});
          }
        }
      }
      const CoverageLedgerKey ledger_key{key, request.request_id};
      const bool inserted =
          coverage_ledgers
              .emplace(ledger_key,
                       detail::KVTransferCoverageLedger(std::move(expected)))
              .second;
      if (!inserted) {
        LOG(ERROR) << "Duplicate LlmDataDist request coverage manifest, "
                   << "request_id=" << request.request_id
                   << ", destination_cluster_id=" << kv_info.dst_cluster_id;
        result = false;
      }
    }
  }

  for (size_t layer_index = 0; layer_index < layer_registered_caches.size();
       ++layer_index) {
    if (!layer_synchronizer->synchronize_layer(
            static_cast<int64_t>(layer_index))) {
      result = false;
      continue;
    }
    for (const std::string& key : keys) {
      const KVCacheInfo& kv_info = merged_kv_infos.at(key);
      for (const RegisteredCache& registered_cache :
           layer_registered_caches[layer_index]) {
        const auto mapping_it =
            std::find_if(kv_info.mappings.begin(),
                         kv_info.mappings.end(),
                         [&registered_cache](const KVTransferMapping& mapping) {
                           return mapping.group_id == registered_cache.group_id;
                         });
        if (mapping_it == kv_info.mappings.end()) {
          LOG(ERROR) << "Missing KV transfer mapping, layer=" << layer_index
                     << ", role=" << registered_cache.role.to_string()
                     << ", group_id=" << registered_cache.group_id;
          result = false;
          continue;
        }
        if (mapping_it->local_ids.empty()) {
          continue;
        }
        if (mapping_it->local_ids.size() != mapping_it->remote_ids.size()) {
          LOG(ERROR) << "KV transfer mapping size mismatch, layer="
                     << layer_index
                     << ", role=" << registered_cache.role.to_string()
                     << ", group_id=" << registered_cache.group_id
                     << ", local=" << mapping_it->local_ids.size()
                     << ", remote=" << mapping_it->remote_ids.size();
          result = false;
          continue;
        }

        llm_datadist::CacheIndex cache_index{kv_info.dst_cluster_id,
                                             registered_cache.cache.cache_id};
        llm_datadist::KvCacheExtParam ext_param{};
        ext_param.src_layer_range = {0, 0};
        ext_param.dst_layer_range = {0, 0};
        ext_param.tensor_num_per_layer = 1;
        const llm_datadist::Status status =
            llm_data_dist_->PushKvBlocks(registered_cache.cache,
                                         cache_index,
                                         mapping_it->local_ids,
                                         mapping_it->remote_ids,
                                         ext_param);
        if (status != llm_datadist::LLM_SUCCESS) {
          LOG(ERROR) << "PushKvBlocks failed, layer=" << layer_index
                     << ", role=" << registered_cache.role.to_string()
                     << ", group_id=" << registered_cache.group_id
                     << ", destination_cluster_id=" << kv_info.dst_cluster_id
                     << ", status=" << std::hex << status;
          result = false;
          continue;
        }

        for (const detail::KVTransferTraceRequest& request :
             kv_info.trace_requests) {
          const auto group_it = std::find_if(
              request.groups.begin(),
              request.groups.end(),
              [&registered_cache](const detail::KVTransferTraceGroup& group) {
                return group.group_id == registered_cache.group_id;
              });
          if (group_it == request.groups.end()) {
            continue;
          }
          if (group_it->local_block_ids.size() !=
              group_it->remote_block_ids.size()) {
            LOG(ERROR) << "LlmDataDist source coverage mapping size mismatch, "
                       << "request_id=" << request.request_id
                       << ", layer=" << layer_index
                       << ", role=" << registered_cache.role.to_string()
                       << ", local=" << group_it->local_block_ids.size()
                       << ", remote=" << group_it->remote_block_ids.size();
            result = false;
            continue;
          }

          const CoverageLedgerKey ledger_key{key, request.request_id};
          auto ledger_it = coverage_ledgers.find(ledger_key);
          if (ledger_it == coverage_ledgers.end()) {
            LOG(ERROR) << "Missing LlmDataDist request coverage ledger, "
                       << "request_id=" << request.request_id
                       << ", destination_cluster_id=" << kv_info.dst_cluster_id;
            result = false;
            continue;
          }
          for (size_t block_index = 0;
               block_index < group_it->remote_block_ids.size();
               ++block_index) {
            const uint64_t destination_physical_block_id =
                group_it->remote_block_ids[block_index];
            const detail::KVTransferCoverageKey contribution{
                request.request_id,
                kv_split_rank,
                static_cast<int64_t>(layer_index),
                registered_cache.group_id,
                static_cast<KVCacheTensorRole::Value>(registered_cache.role),
                destination_physical_block_id};
            const detail::KVTransferCoverageRecordResult coverage_result =
                ledger_it->second.record(contribution);
            if (coverage_result !=
                detail::KVTransferCoverageRecordResult::RECORDED) {
              LOG(ERROR) << "Invalid LlmDataDist coverage contribution, "
                         << "request_id=" << request.request_id
                         << ", owner-rank=" << kv_split_rank
                         << ", layer=" << layer_index
                         << ", role=" << registered_cache.role.to_string()
                         << ", destination-physical-block="
                         << destination_physical_block_id << ", result="
                         << coverage_record_result_to_string(coverage_result);
              result = false;
              continue;
            }
            if (trace_enabled) {
              const LayerSynchronizerTraceContext& trace_context =
                  layer_synchronizer->trace_context();
              XLLM_VERBOSE_TRACE()
                  << "event=kv_cache_block_push_api_success request-id="
                  << request.request_id
                  << " source-rank=" << trace_context.source_rank
                  << " owner-rank=" << kv_split_rank
                  << " cp-rank=" << trace_context.cp_rank
                  << " kv-split-size=" << kv_split_size
                  << " layer=" << layer_index
                  << " cache-group=" << registered_cache.group_id
                  << " cache-role=" << registered_cache.role.to_string()
                  << " destination-physical-block="
                  << destination_physical_block_id
                  << " local-block=" << group_it->local_block_ids[block_index]
                  << " destination-cluster-id=" << kv_info.dst_cluster_id
                  << " destination=" << kv_info.dst_addr
                  << " remote-visibility=unverified"
                  << " transfer-backend=LlmDataDist";
            }
          }
        }
      }
    }
  }

  for (const auto& [coverage_key, ledger] : coverage_ledgers) {
    const KVCacheInfo& kv_info = merged_kv_infos.at(coverage_key.first);
    if (!ledger.is_ready()) {
      const std::vector<detail::KVTransferCoverageKey> missing =
          ledger.missing();
      LOG(ERROR) << "Incomplete LlmDataDist source submission coverage, "
                 << "request_id=" << coverage_key.second
                 << ", owner-rank=" << kv_split_rank
                 << ", destination_cluster_id=" << kv_info.dst_cluster_id
                 << ", expected=" << ledger.expected_count()
                 << ", received=" << ledger.received_count()
                 << ", missing=" << missing.size()
                 << ", duplicate=" << ledger.duplicate_count()
                 << ", unexpected=" << ledger.unexpected_count();
      result = false;
      continue;
    }
    if (trace_enabled) {
      const LayerSynchronizerTraceContext& trace_context =
          layer_synchronizer->trace_context();
      XLLM_VERBOSE_TRACE()
          << "event=source_submission_coverage_complete request-id="
          << coverage_key.second << " source-rank=" << trace_context.source_rank
          << " owner-rank=" << kv_split_rank
          << " cp-rank=" << trace_context.cp_rank
          << " kv-split-size=" << kv_split_size
          << " expected-contributions=" << ledger.expected_count()
          << " received-contributions=" << ledger.received_count()
          << " destination-cluster-id=" << kv_info.dst_cluster_id
          << " destination=" << kv_info.dst_addr
          << " coverage-scope=source-submission"
          << " receiver-coverage-verified=false"
          << " transfer-backend=LlmDataDist";
    }
  }
  return result;
}

llm_datadist::ClusterInfo LlmDataDistTransfer::create_cluster_info(
    uint64_t cluster_id,
    const std::string& remote_ip,
    uint16_t remote_port) const {
  llm_datadist::IpInfo local_ip_info{};
  local_ip_info.ip = host_ip_.c_str();
  local_ip_info.port = listen_port_;

  llm_datadist::IpInfo remote_ip_info{};
  remote_ip_info.ip = remote_ip.c_str();
  remote_ip_info.port = remote_port;

  llm_datadist::ClusterInfo cluster_info{};
  cluster_info.remote_cluster_id = cluster_id;
  cluster_info.local_ip_infos.emplace_back(std::move(local_ip_info));
  cluster_info.remote_ip_infos.emplace_back(std::move(remote_ip_info));
  return cluster_info;
}

}  // namespace xllm
