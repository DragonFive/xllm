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

#include "core/framework/kv_cache_transfer/decode_kv_readiness.h"

#include <cstdint>
#include <limits>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "disagg_pd.pb.h"

namespace xllm {
namespace {

Status invalid(std::string message) {
  return Status(StatusCode::INVALID_ARGUMENT, std::move(message));
}

void contribution_key_to_proto(const DecodeKVContributionKey& key,
                               proto::DecodeKVContributionKey* proto_key) {
  proto_key->set_source_worker_rank(key.source_worker_rank);
  proto_key->set_destination_worker_rank(key.destination_worker_rank);
  proto_key->set_layer_id(key.layer_id);
  proto_key->set_group_id(key.group_id);
  proto_key->set_cache_role(key.cache_role);
  proto_key->set_logical_block_ordinal(key.logical_block_ordinal);
  proto_key->set_destination_physical_block_id(
      key.destination_physical_block_id);
}

DecodeKVContributionKey contribution_key_from_proto(
    const proto::DecodeKVContributionKey& proto_key) {
  DecodeKVContributionKey key;
  key.source_worker_rank = proto_key.source_worker_rank();
  key.destination_worker_rank = proto_key.destination_worker_rank();
  key.layer_id = proto_key.layer_id();
  key.group_id = proto_key.group_id();
  key.cache_role = proto_key.cache_role();
  key.logical_block_ordinal = proto_key.logical_block_ordinal();
  key.destination_physical_block_id = proto_key.destination_physical_block_id();
  return key;
}

void expected_contribution_to_proto(
    const DecodeKVExpectedContribution& contribution,
    proto::DecodeKVExpectedContribution* proto_contribution) {
  contribution_key_to_proto(contribution.key,
                            proto_contribution->mutable_key());
  proto_contribution->set_valid_tokens(contribution.valid_tokens);
}

DecodeKVExpectedContribution expected_contribution_from_proto(
    const proto::DecodeKVExpectedContribution& proto_contribution) {
  DecodeKVExpectedContribution contribution;
  contribution.key = contribution_key_from_proto(proto_contribution.key());
  contribution.valid_tokens = proto_contribution.valid_tokens();
  return contribution;
}

bool validate_receipt(const DecodeKVReceipt& receipt, std::string* error) {
  if (receipt.request_id.empty()) {
    *error = "Decode KV receipt request_id must not be empty";
    return false;
  }
  if (receipt.attempt_epoch == 0 || receipt.allocation_generation == 0) {
    *error = "Decode KV receipt identity generations must be positive";
    return false;
  }
  if (receipt.valid_tokens == 0) {
    *error = "Decode KV receipt valid_tokens must be positive";
    return false;
  }
  return true;
}

bool is_valid_completion_level(DecodeKVCompletionLevel completion_level) {
  switch (completion_level) {
    case DecodeKVCompletionLevel::SUBMITTED:
    case DecodeKVCompletionLevel::SOURCE_COMPLETE:
    case DecodeKVCompletionLevel::REMOTE_VISIBLE:
      return true;
  }
  return false;
}

}  // namespace

bool has_decode_kv_receiver_contributions(
    const std::vector<KVTransferMapping>& mappings) {
  for (const KVTransferMapping& mapping : mappings) {
    if (mapping.remote_ids.size() != mapping.logical_block_ordinals.size() ||
        mapping.remote_ids.size() != mapping.valid_tokens.size()) {
      return true;
    }
    for (size_t index = 0; index < mapping.remote_ids.size(); ++index) {
      if (mapping.remote_ids[index] != std::numeric_limits<uint64_t>::max() &&
          mapping.valid_tokens[index] > 0 &&
          mapping.logical_block_ordinals[index] >= mapping.remote_shared_num) {
        return true;
      }
    }
  }
  return false;
}

Status build_decode_kv_expected_contributions(
    const DecodeKVSourceTopology& source_topology,
    int32_t destination_dp_rank,
    const std::vector<KVTransferMapping>& mappings,
    const std::vector<DecodeKVWorkerLayout>& destination_layouts,
    std::vector<DecodeKVExpectedContribution>* contributions) {
  if (contributions == nullptr) {
    return invalid("Decode KV contribution output must not be null");
  }
  contributions->clear();
  if (source_topology.world_size <= 0 || source_topology.dp_size <= 0 ||
      source_topology.cp_size <= 0 || source_topology.kv_split_size <= 1 ||
      source_topology.kv_split_size > source_topology.cp_size ||
      source_topology.cp_size % source_topology.kv_split_size != 0 ||
      source_topology.world_size %
              (source_topology.dp_size * source_topology.cp_size) !=
          0 ||
      source_topology.dp_rank < 0 ||
      source_topology.dp_rank >= source_topology.dp_size) {
    return invalid("Invalid strict Decode KV source topology");
  }
  if (destination_layouts.empty()) {
    return invalid("Decode KV destination layout set is empty");
  }

  std::unordered_map<int32_t, const KVTransferMapping*> mappings_by_group;
  mappings_by_group.reserve(mappings.size());
  for (const KVTransferMapping& mapping : mappings) {
    if (mapping.remote_ids.size() != mapping.logical_block_ordinals.size() ||
        mapping.remote_ids.size() != mapping.valid_tokens.size()) {
      return invalid("Decode KV mapping metadata is not aligned");
    }
    if (!mappings_by_group.emplace(mapping.group_id, &mapping).second) {
      return invalid("Decode KV mapping contains a duplicate group");
    }
  }

  const int32_t source_tp_size =
      source_topology.world_size /
      (source_topology.dp_size * source_topology.cp_size);
  const ParallelCoordinates& destination_coordinates =
      destination_layouts.front().manifest.coordinates;
  if (destination_coordinates.dp_size <= 0 ||
      destination_coordinates.tp_size <= 0 || destination_dp_rank < 0 ||
      destination_dp_rank >= destination_coordinates.dp_size) {
    return invalid("Invalid Decode KV destination topology");
  }
  if (source_tp_size > destination_coordinates.tp_size) {
    return invalid(
        "Strict Decode KV readiness does not yet represent multiple source "
        "TP shards for one destination worker");
  }
  if (destination_coordinates.tp_size % source_tp_size != 0) {
    return invalid(
        "Strict Decode KV readiness requires destination TP size to be a "
        "multiple of source TP size");
  }

  const int32_t replicas_per_owner =
      source_topology.cp_size / source_topology.kv_split_size;
  const int32_t destination_replicas_per_source_owner =
      destination_coordinates.tp_size / source_tp_size;
  std::set<DecodeKVContributionKey> unique_contributions;
  for (const DecodeKVWorkerLayout& worker_layout : destination_layouts) {
    const WorkerCacheLayoutManifest& layout = worker_layout.manifest;
    const Status layout_status = validate_worker_cache_layout(layout);
    if (!layout_status.ok()) {
      return invalid("Invalid Decode worker cache layout: " +
                     layout_status.message());
    }
    if (layout.coordinates.dp_rank != destination_dp_rank) {
      continue;
    }

    for (const CacheTensorManifest& tensor : layout.tensors) {
      if (tensor.cache_namespace != CacheNamespace::MAIN) {
        continue;
      }
      if (tensor.shard.resource_scope != CacheResourceScope::BLOCK) {
        return invalid(
            "Strict Decode KV readiness does not support sequence-scoped "
            "cache resources");
      }
      const int32_t source_tp_owner = tensor.shard.spans.front().owner_tp_rank /
                                      destination_replicas_per_source_owner;
      for (const LogicalSpan& span : tensor.shard.spans) {
        if (span.owner_tp_rank / destination_replicas_per_source_owner !=
            source_tp_owner) {
          return invalid(
              "Decode KV tensor spans map to multiple source TP owners");
        }
      }
      const auto mapping_it = mappings_by_group.find(tensor.group_id);
      if (mapping_it == mappings_by_group.end()) {
        return invalid("Decode cache layout has no request mapping for group " +
                       std::to_string(tensor.group_id));
      }
      const KVTransferMapping& mapping = *mapping_it->second;
      for (size_t index = 0; index < mapping.remote_ids.size(); ++index) {
        const uint64_t destination_block = mapping.remote_ids[index];
        const uint32_t valid_tokens = mapping.valid_tokens[index];
        const uint64_t logical_ordinal = mapping.logical_block_ordinals[index];
        if (destination_block == std::numeric_limits<uint64_t>::max() ||
            valid_tokens == 0 ||
            logical_ordinal <
                static_cast<uint64_t>(mapping.remote_shared_num)) {
          continue;
        }
        if (destination_block >= tensor.resource_count) {
          return invalid("Decode KV destination block exceeds cache layout");
        }

        const int32_t kv_owner = static_cast<int32_t>(
            logical_ordinal %
            static_cast<uint64_t>(source_topology.kv_split_size));
        const int32_t source_cp_rank = kv_owner * replicas_per_owner;
        const int32_t source_worker_rank =
            source_topology.dp_rank *
                (source_topology.cp_size * source_tp_size) +
            source_cp_rank * source_tp_size + source_tp_owner;

        DecodeKVExpectedContribution contribution;
        contribution.key.source_worker_rank = source_worker_rank;
        contribution.key.destination_worker_rank = worker_layout.worker_rank;
        contribution.key.layer_id = tensor.layer_id;
        contribution.key.group_id = tensor.group_id;
        contribution.key.cache_role = tensor.role;
        contribution.key.logical_block_ordinal = logical_ordinal;
        contribution.key.destination_physical_block_id = destination_block;
        contribution.valid_tokens = valid_tokens;
        if (!unique_contributions.emplace(contribution.key).second) {
          return invalid("Decode KV manifest contains duplicate contribution");
        }
        contributions->emplace_back(std::move(contribution));
      }
    }
  }
  if (contributions->empty()) {
    return invalid("Decode KV manifest has no receiver contributions");
  }
  return Status();
}

bool serialize_decode_kv_expected_manifest(
    const DecodeKVExpectedManifest& manifest,
    std::string* serialized,
    std::string* error) {
  if (serialized == nullptr || error == nullptr) {
    return false;
  }
  if (manifest.request_id.empty() || manifest.attempt_epoch == 0 ||
      manifest.allocation_generation == 0 || manifest.contributions.empty()) {
    *error = "Decode KV manifest is incomplete";
    return false;
  }
  proto::DecodeKVExpectedManifest proto_manifest;
  proto_manifest.set_schema_version(kDecodeKVReadinessSchemaVersion);
  proto_manifest.set_request_id(manifest.request_id);
  proto_manifest.set_attempt_epoch(manifest.attempt_epoch);
  proto_manifest.set_allocation_generation(manifest.allocation_generation);
  for (const DecodeKVExpectedContribution& contribution :
       manifest.contributions) {
    expected_contribution_to_proto(contribution,
                                   proto_manifest.add_contributions());
  }
  if (!proto_manifest.SerializeToString(serialized)) {
    *error = "Failed to serialize Decode KV manifest";
    return false;
  }
  return true;
}

DecodeKVPayloadResult deserialize_decode_kv_expected_manifest(
    const std::string& serialized,
    DecodeKVExpectedManifest* manifest,
    std::string* error) {
  if (manifest == nullptr || error == nullptr) {
    return DecodeKVPayloadResult::MALFORMED;
  }
  proto::DecodeKVExpectedManifest proto_manifest;
  if (!proto_manifest.ParseFromString(serialized)) {
    *error = "Malformed Decode KV manifest protobuf";
    return DecodeKVPayloadResult::MALFORMED;
  }
  manifest->request_id = proto_manifest.request_id();
  manifest->attempt_epoch = proto_manifest.attempt_epoch();
  manifest->allocation_generation = proto_manifest.allocation_generation();
  manifest->contributions.clear();
  manifest->contributions.reserve(proto_manifest.contributions_size());
  for (const proto::DecodeKVExpectedContribution& proto_contribution :
       proto_manifest.contributions()) {
    manifest->contributions.emplace_back(
        expected_contribution_from_proto(proto_contribution));
  }
  if (proto_manifest.schema_version() != kDecodeKVReadinessSchemaVersion) {
    *error = "Unsupported Decode KV manifest schema version";
    return DecodeKVPayloadResult::VERSION_MISMATCH;
  }
  DecodeKVReadinessLedger validator(*manifest);
  if (validator.is_poisoned()) {
    *error = validator.failure_reason();
    return DecodeKVPayloadResult::INVALID_ENVELOPE;
  }
  return DecodeKVPayloadResult::OK;
}

bool serialize_mooncake_decode_kv_notification(
    const MooncakeDecodeKVNotification& notification,
    std::string* serialized,
    std::string* error) {
  if (serialized == nullptr || error == nullptr) {
    return false;
  }
  if (notification.schema_version != kDecodeKVReadinessSchemaVersion ||
      notification.submission_id.empty() || notification.batch_count == 0 ||
      notification.batch_index >= notification.batch_count) {
    *error = "Invalid Mooncake Decode KV notification envelope";
    return false;
  }
  if (!notification.receipts.empty() &&
      notification.batch_index + 1 != notification.batch_count) {
    *error = "Mooncake Decode KV receipts are allowed only on the final batch";
    return false;
  }
  proto::MooncakeKVNotification proto_notification;
  proto_notification.set_schema_version(notification.schema_version);
  proto_notification.set_submission_id(notification.submission_id);
  proto_notification.set_batch_index(notification.batch_index);
  proto_notification.set_batch_count(notification.batch_count);
  for (const DecodeKVReceipt& receipt : notification.receipts) {
    if (!validate_receipt(receipt, error)) {
      return false;
    }
    proto::DecodeKVReceipt* proto_receipt = proto_notification.add_receipts();
    proto_receipt->set_request_id(receipt.request_id);
    contribution_key_to_proto(receipt.key, proto_receipt->mutable_key());
    proto_receipt->set_attempt_epoch(receipt.attempt_epoch);
    proto_receipt->set_allocation_generation(receipt.allocation_generation);
    proto_receipt->set_valid_tokens(receipt.valid_tokens);
  }
  if (!proto_notification.SerializeToString(serialized)) {
    *error = "Failed to serialize Mooncake Decode KV notification";
    return false;
  }
  return true;
}

DecodeKVPayloadResult deserialize_mooncake_decode_kv_notification(
    const std::string& serialized,
    MooncakeDecodeKVNotification* notification,
    std::string* error) {
  if (notification == nullptr || error == nullptr) {
    return DecodeKVPayloadResult::MALFORMED;
  }
  proto::MooncakeKVNotification proto_notification;
  if (!proto_notification.ParseFromString(serialized)) {
    *error = "Malformed Mooncake Decode KV notification protobuf";
    return DecodeKVPayloadResult::MALFORMED;
  }
  notification->schema_version = proto_notification.schema_version();
  notification->submission_id = proto_notification.submission_id();
  notification->batch_index = proto_notification.batch_index();
  notification->batch_count = proto_notification.batch_count();
  notification->receipts.clear();
  notification->receipts.reserve(proto_notification.receipts_size());
  for (const proto::DecodeKVReceipt& proto_receipt :
       proto_notification.receipts()) {
    DecodeKVReceipt receipt;
    receipt.request_id = proto_receipt.request_id();
    receipt.key = contribution_key_from_proto(proto_receipt.key());
    receipt.submission_id = notification->submission_id;
    receipt.attempt_epoch = proto_receipt.attempt_epoch();
    receipt.allocation_generation = proto_receipt.allocation_generation();
    receipt.valid_tokens = proto_receipt.valid_tokens();
    receipt.completion_level = DecodeKVCompletionLevel::REMOTE_VISIBLE;
    notification->receipts.emplace_back(std::move(receipt));
  }
  if (notification->schema_version != kDecodeKVReadinessSchemaVersion) {
    *error = "Unsupported Mooncake Decode KV notification schema version";
    return DecodeKVPayloadResult::VERSION_MISMATCH;
  }
  if (notification->submission_id.empty() || notification->batch_count == 0 ||
      notification->batch_index >= notification->batch_count) {
    *error = "Invalid Mooncake Decode KV notification envelope";
    return DecodeKVPayloadResult::INVALID_ENVELOPE;
  }
  if (!notification->receipts.empty() &&
      notification->batch_index + 1 != notification->batch_count) {
    *error = "Mooncake Decode KV receipts are allowed only on the final batch";
    return DecodeKVPayloadResult::INVALID_ENVELOPE;
  }
  for (const DecodeKVReceipt& receipt : notification->receipts) {
    if (!validate_receipt(receipt, error)) {
      return DecodeKVPayloadResult::INVALID_RECEIPT;
    }
  }
  return DecodeKVPayloadResult::OK;
}

bool operator<(const DecodeKVContributionKey& lhs,
               const DecodeKVContributionKey& rhs) {
  return std::tie(lhs.source_worker_rank,
                  lhs.destination_worker_rank,
                  lhs.layer_id,
                  lhs.group_id,
                  lhs.cache_role,
                  lhs.logical_block_ordinal,
                  lhs.destination_physical_block_id) <
         std::tie(rhs.source_worker_rank,
                  rhs.destination_worker_rank,
                  rhs.layer_id,
                  rhs.group_id,
                  rhs.cache_role,
                  rhs.logical_block_ordinal,
                  rhs.destination_physical_block_id);
}

DecodeKVReadinessLedger::DecodeKVReadinessLedger(
    DecodeKVExpectedManifest manifest)
    : request_id_(std::move(manifest.request_id)),
      attempt_epoch_(manifest.attempt_epoch),
      allocation_generation_(manifest.allocation_generation) {
  using LogicalContributionIdentity =
      std::tuple<int32_t, int64_t, int32_t, int32_t, uint64_t>;
  using DestinationAllocationIdentity =
      std::tuple<int32_t, int64_t, int32_t, int32_t, uint64_t>;

  if (request_id_.empty()) {
    poison("Decode KV manifest request_id must not be empty");
  }
  if (attempt_epoch_ == 0) {
    poison("Decode KV manifest attempt_epoch must be positive");
  }
  if (allocation_generation_ == 0) {
    poison("Decode KV manifest allocation_generation must be positive");
  }
  if (manifest.contributions.empty()) {
    poison("Decode KV manifest must contain at least one contribution");
  }

  std::set<LogicalContributionIdentity> logical_contributions;
  std::set<DestinationAllocationIdentity> destination_allocations;
  for (DecodeKVExpectedContribution& contribution : manifest.contributions) {
    if (contribution.valid_tokens == 0) {
      poison("Decode KV manifest contribution valid_tokens must be positive");
    }

    const DecodeKVContributionKey& key = contribution.key;
    const LogicalContributionIdentity logical_identity =
        std::make_tuple(key.destination_worker_rank,
                        key.layer_id,
                        key.group_id,
                        key.cache_role,
                        key.logical_block_ordinal);
    if (!logical_contributions.emplace(logical_identity).second) {
      poison(
          "Decode KV manifest has multiple owners for one logical "
          "contribution");
    }

    const DestinationAllocationIdentity destination_identity =
        std::make_tuple(key.destination_worker_rank,
                        key.layer_id,
                        key.group_id,
                        key.cache_role,
                        key.destination_physical_block_id);
    if (!destination_allocations.emplace(destination_identity).second) {
      poison(
          "Decode KV manifest maps multiple logical blocks to one "
          "destination allocation");
    }

    if (!expected_.emplace(contribution.key, std::move(contribution)).second) {
      poison("Decode KV manifest contains a duplicate transport key");
    }
  }
}

DecodeKVReceiptRecordResult DecodeKVReadinessLedger::record(
    const DecodeKVReceipt& receipt) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (poisoned_) {
    return DecodeKVReceiptRecordResult::LEDGER_POISONED;
  }
  if (receipt.request_id != request_id_) {
    poison("Decode KV receipt request_id does not match the manifest");
    return DecodeKVReceiptRecordResult::REQUEST_MISMATCH;
  }
  if (receipt.attempt_epoch != attempt_epoch_) {
    poison("Decode KV receipt attempt epoch does not match the manifest");
    return DecodeKVReceiptRecordResult::STALE_ATTEMPT;
  }
  if (receipt.allocation_generation != allocation_generation_) {
    poison(
        "Decode KV receipt allocation generation does not match the manifest");
    return DecodeKVReceiptRecordResult::STALE_ALLOCATION_GENERATION;
  }
  if (receipt.submission_id.empty()) {
    poison("Decode KV receipt submission_id must not be empty");
    return DecodeKVReceiptRecordResult::EMPTY_SUBMISSION_ID;
  }
  if (!is_valid_completion_level(receipt.completion_level)) {
    poison("Decode KV receipt has an invalid completion level");
    return DecodeKVReceiptRecordResult::INVALID_COMPLETION_LEVEL;
  }

  const auto expected_it = expected_.find(receipt.key);
  if (expected_it == expected_.end()) {
    poison("Decode KV receipt does not belong to the expected manifest");
    return DecodeKVReceiptRecordResult::UNEXPECTED_CONTRIBUTION;
  }
  if (receipt.valid_tokens != expected_it->second.valid_tokens) {
    poison("Decode KV receipt valid_tokens does not match the manifest");
    return DecodeKVReceiptRecordResult::VALID_TOKEN_MISMATCH;
  }

  auto receipt_it = receipts_.find(receipt.key);
  if (receipt_it == receipts_.end()) {
    receipts_.emplace(receipt.key, receipt);
    return DecodeKVReceiptRecordResult::RECORDED;
  }

  DecodeKVReceipt& recorded_receipt = receipt_it->second;
  if (recorded_receipt.submission_id != receipt.submission_id) {
    poison("Decode KV contribution has conflicting submission identities");
    return DecodeKVReceiptRecordResult::CONFLICTING_DUPLICATE;
  }
  if (recorded_receipt.completion_level == receipt.completion_level) {
    return DecodeKVReceiptRecordResult::IDEMPOTENT_REPLAY;
  }
  if (static_cast<int8_t>(receipt.completion_level) <
      static_cast<int8_t>(recorded_receipt.completion_level)) {
    poison("Decode KV receipt completion level regressed");
    return DecodeKVReceiptRecordResult::COMPLETION_REGRESSION;
  }

  recorded_receipt = receipt;
  return DecodeKVReceiptRecordResult::COMPLETION_ADVANCED;
}

bool DecodeKVReadinessLedger::is_ready() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return is_ready_locked();
}

bool DecodeKVReadinessLedger::is_poisoned() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return poisoned_;
}

bool DecodeKVReadinessLedger::was_published() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return published_;
}

std::string DecodeKVReadinessLedger::failure_reason() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return failure_reason_;
}

bool DecodeKVReadinessLedger::is_ready_locked() const {
  if (poisoned_ || expected_.empty()) {
    return false;
  }
  for (const auto& expected_entry : expected_) {
    const auto receipt_it = receipts_.find(expected_entry.first);
    if (receipt_it == receipts_.end() ||
        receipt_it->second.completion_level !=
            DecodeKVCompletionLevel::REMOTE_VISIBLE) {
      return false;
    }
  }
  return true;
}

std::vector<DecodeKVExpectedContribution> DecodeKVReadinessLedger::missing()
    const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<DecodeKVExpectedContribution> missing;
  missing.reserve(expected_.size());
  for (const auto& expected_entry : expected_) {
    const auto receipt_it = receipts_.find(expected_entry.first);
    if (receipt_it != receipts_.end() &&
        receipt_it->second.completion_level ==
            DecodeKVCompletionLevel::REMOTE_VISIBLE) {
      continue;
    }
    missing.emplace_back(expected_entry.second);
  }
  return missing;
}

void DecodeKVReadinessLedger::mark_poisoned(std::string reason) {
  std::lock_guard<std::mutex> lock(mutex_);
  poison(std::move(reason));
}

bool DecodeKVReadinessLedger::try_publish() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (published_ || !is_ready_locked()) {
    return false;
  }
  published_ = true;
  return true;
}

void DecodeKVReadinessLedger::poison(std::string reason) {
  poisoned_ = true;
  if (failure_reason_.empty()) {
    failure_reason_ = std::move(reason);
  }
}

}  // namespace xllm
