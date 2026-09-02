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

#pragma once

#include <llm_datadist/llm_datadist.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "framework/kv_cache_transfer/kv_cache_transfer.h"

namespace xllm {

class LlmDataDistTransfer final : public KVCacheTransfer {
 public:
  LlmDataDistTransfer(uint16_t listen_port,
                      InstanceRole instance_role,
                      bool enable_lighting_indexer);
  ~LlmDataDistTransfer() override;

  void initialize(int32_t device_id) override;
  void finalize() override;

  void register_kv_cache(std::vector<KVCache>& kv_caches,
                         const KVCacheShape& kv_cache_shape,
                         torch::ScalarType dtype) override;
  void register_kv_cache_spec(std::vector<KVCache>& kv_caches,
                              const KVCacheShape& kv_cache_shape,
                              torch::ScalarType dtype) override;
  void free_kv_cache() override;

  void get_cache_info(uint64_t& cluster_id, std::string& addr) override;
  bool link_clusters(const std::vector<uint64_t>& cluster_ids,
                     const std::vector<std::string>& remote_addrs,
                     const std::vector<uint16_t>& ports) override;
  bool unlink_cluster(const uint64_t& cluster_id,
                      const std::string& remote_addr,
                      uint16_t port,
                      bool force_flag = true) override;

  bool pull_kv_blocks(uint64_t src_cluster_id,
                      const std::string& src_addr,
                      const std::vector<KVTransferMapping>& mappings) override;
  bool push_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
      bool is_spec_draft,
      int32_t kv_split_rank,
      int32_t kv_split_size) override;

 private:
  struct RegisteredCache {
    KVCacheTensorRole role;
    int32_t group_id = 0;
    llm_datadist::Cache cache;
    torch::Tensor tensor;
  };

  using LayerRegisteredCaches = std::vector<std::vector<RegisteredCache>>;

  [[nodiscard]] llm_datadist::ClusterInfo create_cluster_info(
      uint64_t cluster_id,
      const std::string& remote_ip,
      uint16_t remote_port) const;
  RegisteredCache register_cache_tensor(int64_t layer_id,
                                        const KVCacheTensor& cache_tensor);
  void register_layer_caches(std::vector<KVCache>& kv_caches);
  bool push_layer_caches(
      const LayerRegisteredCaches& layer_registered_caches,
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
      int32_t kv_split_rank,
      int32_t kv_split_size);

  uint64_t cluster_id_ = 0;
  std::string host_ip_;
  uint16_t listen_port_ = 0;
  llm_datadist::LlmRole role_ = llm_datadist::LlmRole::kMix;
  bool initialized_ = false;
  std::unordered_set<uint64_t> linked_cluster_ids_;
  std::shared_ptr<llm_datadist::LlmDataDist> llm_data_dist_;
  LayerRegisteredCaches layer_registered_caches_;
};

}  // namespace xllm
