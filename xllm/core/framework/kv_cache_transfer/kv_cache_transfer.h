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

#pragma once

#include <folly/futures/Future.h>

#include <cstddef>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "common/types.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#if defined(USE_NPU)
#include "platform/npu/npu_layer_synchronizer.h"
#endif
#if defined(USE_MLU)
#include "platform/mlu/mlu_layer_synchronizer.h"
#endif
#if defined(USE_DCU)
#include "platform/dcu/dcu_layer_synchronizer.h"
#endif
#include "framework/parallel_state/parallel_args.h"
#include "platform/device.h"
#include "util/threadpool.h"

namespace xllm {

struct KVCacheLayoutQueryResult final {
  bool ok = true;
  bool supported = false;
  std::string serialized_manifest;
};

struct KVTransferNotificationDrainResult final {
  bool ok = true;
  bool supported = false;
  bool more_available = false;
  std::vector<std::string> payloads;
};

namespace detail {

struct KVTransferTraceGroup {
  int32_t group_id = 0;
  size_t local_block_count = 0;
  size_t remote_block_count = 0;
  std::vector<uint64_t> local_block_ids;
  std::vector<uint64_t> remote_block_ids;
};

struct KVTransferTraceRequest {
  std::string request_id;
  std::vector<KVTransferTraceGroup> groups;
};

void append_kv_transfer_trace_request(
    std::vector<KVTransferTraceRequest>* requests,
    const TransferKVInfo& info);

// One source-side direct-write submission for a Decode request. The destination
// physical block id is allocated by Decode and passed to the PUSH API. It is
// not a request logical-block ordinal, and recording it does not prove that the
// receiver can observe the submitted data.
struct KVTransferCoverageKey {
  std::string request_id;
  int32_t owner_rank = 0;
  int64_t layer_id = 0;
  int32_t group_id = 0;
  KVCacheTensorRole::Value cache_role = KVCacheTensorRole::INVALID;
  uint64_t destination_physical_block_id = 0;
};

bool operator<(const KVTransferCoverageKey& lhs,
               const KVTransferCoverageKey& rhs);

enum class KVTransferCoverageRecordResult : int8_t {
  RECORDED = 0,
  DUPLICATE = 1,
  UNEXPECTED = 2,
};

// Fail-closed exact-once accounting for request-local source submissions. A
// ledger is ready only after every expected PUSH API submission has been
// recorded once and no duplicate or unexpected contribution was seen. This
// does not establish receiver visibility or Decode readiness.
class KVTransferCoverageLedger final {
 public:
  explicit KVTransferCoverageLedger(
      std::vector<KVTransferCoverageKey> expected);

  KVTransferCoverageRecordResult record(
      const KVTransferCoverageKey& contribution);
  bool is_ready() const;
  size_t expected_count() const { return expected_.size(); }
  size_t received_count() const { return received_.size(); }
  size_t duplicate_count() const { return duplicate_count_; }
  size_t unexpected_count() const { return unexpected_count_; }
  std::vector<KVTransferCoverageKey> missing() const;

 private:
  std::set<KVTransferCoverageKey> expected_;
  std::set<KVTransferCoverageKey> received_;
  size_t duplicate_count_ = 0;
  size_t unexpected_count_ = 0;
};

struct LlmDataDistCapability {
  bool is_npu_backend = false;
  InstanceRole instance_role = InstanceRole::DEFAULT;
  std::string transfer_mode = "PUSH";
  std::string model_type;
  bool has_lightning_indexer = false;
  std::string kv_cache_dtype = "auto";
  bool enable_xtensor = false;
  bool has_linear_attention_cache = false;
  bool has_grouped_cache_layout = false;
  bool is_spec_draft = false;
  int32_t dp_size = 1;
  int32_t cp_size = 1;
  int32_t kv_split_size = 1;
};

std::optional<std::string> validate_llm_data_dist_capability(
    const LlmDataDistCapability& capability);

}  // namespace detail

#if defined(USE_NPU)
using KVPushSynchronizerImpl = NPULayerSynchronizerImpl;
#elif defined(USE_MLU)
using KVPushSynchronizerImpl = MLULayerSynchronizerImpl;
#elif defined(USE_DCU)
using KVPushSynchronizerImpl = DCULayerSynchronizerImpl;
#endif

// In KV-split mode, filters and remaps each block-scoped cache mapping's
// remote_ids so that every KV-split rank sees only the destination blocks
// assigned to it. This includes ordinary KV and grouped SWA/C4/C128 caches.
// When `kv_split_size == 1` the caller should skip this entirely (every rank
// holds the full KV replica and remote_ids is 1:1 with local_ids).
//
// Note: prior to the KV-split / CP decoupling refactor this was named
// filter_cp_kv_infos and gated on cp_size>1. The behavior is identical when
// kv_split_size == cp_size (the legacy default), so callers that pass cp_rank
// / cp_size keep working byte-for-byte.
std::vector<TransferKVInfo> filter_kv_split_infos(
    int32_t kv_split_rank,
    int32_t kv_split_size,
    const std::vector<TransferKVInfo>& kv_infos);

class KVCacheTransfer {
 public:
  struct KVCacheInfo {
    uint64_t dst_cluster_id;
    std::string dst_addr;
    int32_t source_worker_rank = -1;
    int32_t destination_worker_rank = -1;
    std::vector<KVTransferMapping> mappings;

    // Strict readiness metadata remains request-scoped even when transfer
    // mappings from several requests are merged for one destination worker.
    std::vector<TransferKVInfo> receipt_infos;

    // XTensor mode: destination offsets from D-node (per-layer)
    // dst_xtensor_layer_offsets[layer_id] = {k_offsets, v_offsets}
    std::vector<XTensorLayerOffsets> dst_xtensor_layer_offsets;

    // Preserved for source-side submission accounting. Verbose tracing only
    // controls emission of successful-submission records.
    std::vector<detail::KVTransferTraceRequest> trace_requests;
  };

  static std::vector<std::string> rotate_dst_rank(
      const std::vector<std::string>& keys,
      int32_t kv_split_rank);

  KVCacheTransfer() = default;
  virtual ~KVCacheTransfer() = default;

  virtual void initialize(int32_t device_id) {};

  virtual void finalize() {};

  virtual void free_kv_cache() {};

  virtual void configure_cache_layout(const ParallelArgs& parallel_args,
                                      const ModelArgs& model_args,
                                      int32_t block_token_capacity,
                                      bool is_spec_draft) {}

  virtual void register_kv_cache(std::vector<xllm::KVCache>& kv_caches,
                                 const KVCacheShape& kv_cache_shape,
                                 const torch::ScalarType dtype) {};

  virtual void register_kv_cache_spec(std::vector<xllm::KVCache>& kv_caches,
                                      const KVCacheShape& kv_cache_shape,
                                      const torch::ScalarType dtype) {
    NOT_IMPLEMENTED();
  };

  virtual void get_cache_info(uint64_t& cluster_id, std::string& addr) = 0;

  virtual KVCacheLayoutQueryResult get_kv_cache_layout();

  virtual KVTransferNotificationDrainResult drain_kv_transfer_notifications(
      size_t max_notifications);

  virtual bool link_clusters(const std::vector<uint64_t>& cluster_ids,
                             const std::vector<std::string>& remote_addrs,
                             const std::vector<uint16_t>& ports) = 0;

  virtual bool unlink_cluster(const uint64_t& cluster_id,
                              const std::string& remote_addr,
                              const uint16_t port,
                              bool force_flag = true) = 0;

  virtual bool pull_kv_blocks(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<KVTransferMapping>& mappings) = 0;

  virtual folly::SemiFuture<bool> pull_kv_blocks_async(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<KVTransferMapping>& mappings);

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
  virtual folly::SemiFuture<bool> push_kv_blocks_async(
      const std::vector<TransferKVInfo>& transfer_kv_infos,
      const ParallelArgs& parallel_args,
      std::shared_ptr<KVPushSynchronizerImpl> layer_synchronizer,
      bool is_spec_draft);
#endif

  virtual void merge_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      const std::vector<TransferKVInfo>& transfer_kv_infos,
      const ParallelArgs& parallel_args);

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
  virtual bool push_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
      bool is_spec_draft,
      int32_t kv_split_rank,
      int32_t kv_split_size) = 0;
#endif

 protected:
  static bool validate_transfer_mappings(
      const std::vector<KVTransferMapping>& mappings,
      const std::string& request_id,
      int32_t kv_split_size);

  static bool validate_transfer_mappings(
      const std::vector<TransferKVInfo>& transfer_kv_infos,
      int32_t kv_split_size);

  // working thread
  ThreadPool threadpool_{/*num_threads=*/1,
                         /*cpu_binding=*/false,
                         /*pool_name=*/"KVCacheTransfer.async"};
};

class KVCacheTransferFactory {
 public:
  static std::shared_ptr<KVCacheTransfer> create(
      const std::string& transfer_type,
      uint16_t transfer_listen_port,
      InstanceRole instance_role,
      const Device& device,
      bool enable_lighting_indexer,
      const std::string& model_type = "",
      const std::string& model_id = "");
};

}  // namespace xllm
