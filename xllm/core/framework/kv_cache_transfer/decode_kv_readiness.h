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

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "common/types.h"
#include "framework/kv_cache_transfer/cache_layout.h"

namespace xllm {

inline constexpr uint32_t kDecodeKVReadinessSchemaVersion = 1;
inline constexpr char kMooncakeDecodeKVNotificationName[] =
    "xllm.decode-kv-readiness.v1";

// Only REMOTE_VISIBLE proves that Decode can safely consume the contribution.
enum class DecodeKVCompletionLevel : int8_t {
  SUBMITTED = 0,
  SOURCE_COMPLETE = 1,
  REMOTE_VISIBLE = 2,
};

// Identifies one logical Decode KV contribution and its destination allocation.
struct DecodeKVContributionKey {
  int32_t source_worker_rank = 0;
  int32_t destination_worker_rank = 0;
  int64_t layer_id = 0;
  int32_t group_id = 0;
  int32_t cache_role = 0;
  uint64_t logical_block_ordinal = 0;
  uint64_t destination_physical_block_id = 0;
};

bool operator<(const DecodeKVContributionKey& lhs,
               const DecodeKVContributionKey& rhs);

struct DecodeKVExpectedContribution {
  DecodeKVContributionKey key;
  uint32_t valid_tokens = 0;
};

struct DecodeKVExpectedManifest {
  std::string request_id;
  uint64_t attempt_epoch = 0;
  uint64_t allocation_generation = 0;
  std::vector<DecodeKVExpectedContribution> contributions;
};

struct DecodeKVSourceTopology {
  int32_t world_size = 0;
  int32_t dp_size = 1;
  int32_t cp_size = 1;
  int32_t kv_split_size = 1;
  int32_t dp_rank = 0;
};

struct DecodeKVWorkerLayout {
  int32_t worker_rank = 0;
  WorkerCacheLayoutManifest manifest;
};

struct DecodeKVReceipt {
  std::string request_id;
  DecodeKVContributionKey key;
  std::string submission_id;
  uint64_t attempt_epoch = 0;
  uint64_t allocation_generation = 0;
  uint32_t valid_tokens = 0;
  DecodeKVCompletionLevel completion_level = DecodeKVCompletionLevel::SUBMITTED;
};

struct MooncakeDecodeKVNotification {
  uint32_t schema_version = kDecodeKVReadinessSchemaVersion;
  std::string submission_id;
  uint32_t batch_index = 0;
  uint32_t batch_count = 1;
  std::vector<DecodeKVReceipt> receipts;
};

struct DecodeKVReadinessPrepareResult final {
  bool ok = false;
  std::string serialized_manifest;
  uint64_t attempt_epoch = 0;
  uint64_t allocation_generation = 0;
  std::string error;
};

struct DecodeKVReadinessPollResult final {
  bool ok = false;
  bool complete = false;
  std::string error;
};

struct DecodeKVReadinessSnapshot final {
  bool found = false;
  bool ready = false;
  bool poisoned = false;
  bool published = false;
  uint64_t attempt_epoch = 0;
  uint64_t allocation_generation = 0;
  std::string failure_reason;
};

enum class DecodeKVPayloadResult : int8_t {
  OK = 0,
  MALFORMED = 1,
  VERSION_MISMATCH = 2,
  INVALID_ENVELOPE = 3,
  INVALID_RECEIPT = 4,
};

// Returns true when the Decode allocation contains at least one receiver-side
// block contribution that Prefill must publish. Malformed mapping metadata also
// returns true so the strict manifest builder can reject it fail closed.
bool has_decode_kv_receiver_contributions(
    const std::vector<KVTransferMapping>& mappings);

Status build_decode_kv_expected_contributions(
    const DecodeKVSourceTopology& source_topology,
    int32_t destination_dp_rank,
    const std::vector<KVTransferMapping>& mappings,
    const std::vector<DecodeKVWorkerLayout>& destination_layouts,
    std::vector<DecodeKVExpectedContribution>* contributions);

bool serialize_decode_kv_expected_manifest(
    const DecodeKVExpectedManifest& manifest,
    std::string* serialized,
    std::string* error);

DecodeKVPayloadResult deserialize_decode_kv_expected_manifest(
    const std::string& serialized,
    DecodeKVExpectedManifest* manifest,
    std::string* error);

bool serialize_mooncake_decode_kv_notification(
    const MooncakeDecodeKVNotification& notification,
    std::string* serialized,
    std::string* error);

// If protobuf parsing succeeds, notification is populated before semantic
// validation. Callers can therefore quarantine referenced requests on a
// version mismatch or malformed envelope without treating the payload as a
// valid receipt.
DecodeKVPayloadResult deserialize_mooncake_decode_kv_notification(
    const std::string& serialized,
    MooncakeDecodeKVNotification* notification,
    std::string* error);

enum class DecodeKVReceiptRecordResult : int8_t {
  RECORDED = 0,
  COMPLETION_ADVANCED = 1,
  IDEMPOTENT_REPLAY = 2,
  LEDGER_POISONED = 3,
  REQUEST_MISMATCH = 4,
  STALE_ATTEMPT = 5,
  STALE_ALLOCATION_GENERATION = 6,
  EMPTY_SUBMISSION_ID = 7,
  UNEXPECTED_CONTRIBUTION = 8,
  VALID_TOKEN_MISMATCH = 9,
  CONFLICTING_DUPLICATE = 10,
  COMPLETION_REGRESSION = 11,
  INVALID_COMPLETION_LEVEL = 12,
};

// Request-local, backend-neutral readiness accounting. This ledger consumes a
// Decode-owned expected manifest and receiver-visibility receipts; source-side
// PUSH submission alone is intentionally insufficient for readiness.
class DecodeKVReadinessLedger final {
 public:
  explicit DecodeKVReadinessLedger(DecodeKVExpectedManifest manifest);

  DecodeKVReceiptRecordResult record(const DecodeKVReceipt& receipt);
  const std::string& request_id() const { return request_id_; }
  uint64_t attempt_epoch() const { return attempt_epoch_; }
  uint64_t allocation_generation() const { return allocation_generation_; }
  bool is_ready() const;
  bool is_poisoned() const;
  bool was_published() const;
  std::string failure_reason() const;
  std::vector<DecodeKVExpectedContribution> missing() const;

  void mark_poisoned(std::string reason);

  // Marks readiness as published once. All public state transitions are safe
  // for concurrent receiver callbacks.
  bool try_publish();

 private:
  bool is_ready_locked() const;
  void poison(std::string reason);

  std::string request_id_;
  uint64_t attempt_epoch_ = 0;
  uint64_t allocation_generation_ = 0;

  // Ordered maps keep missing-contribution diagnostics deterministic.
  std::map<DecodeKVContributionKey, DecodeKVExpectedContribution> expected_;
  std::map<DecodeKVContributionKey, DecodeKVReceipt> receipts_;
  mutable std::mutex mutex_;
  bool poisoned_ = false;
  bool published_ = false;
  std::string failure_reason_;
};

}  // namespace xllm
