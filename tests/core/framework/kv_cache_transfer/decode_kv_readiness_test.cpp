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

#include <gtest/gtest.h>

#include <cstdint>
#include <future>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "disagg_pd.pb.h"
#include "framework/block/block.h"
#include "framework/kv_cache/kv_cache_tensor_role.h"

namespace xllm {
namespace {

constexpr uint64_t kAttemptEpoch = 7;
constexpr uint64_t kAllocationGeneration = 11;

DecodeKVContributionKey make_key(int64_t layer_id,
                                 uint64_t logical_block_ordinal,
                                 uint64_t destination_physical_block_id) {
  DecodeKVContributionKey key;
  key.source_worker_rank = 1;
  key.destination_worker_rank = 3;
  key.layer_id = layer_id;
  key.group_id = 5;
  key.cache_role = 2;
  key.logical_block_ordinal = logical_block_ordinal;
  key.destination_physical_block_id = destination_physical_block_id;
  return key;
}

DecodeKVExpectedContribution make_expected(const DecodeKVContributionKey& key,
                                           uint32_t valid_tokens = 16) {
  DecodeKVExpectedContribution contribution;
  contribution.key = key;
  contribution.valid_tokens = valid_tokens;
  return contribution;
}

DecodeKVExpectedManifest make_manifest(
    std::vector<DecodeKVExpectedContribution> contributions) {
  DecodeKVExpectedManifest manifest;
  manifest.request_id = "request-a";
  manifest.attempt_epoch = kAttemptEpoch;
  manifest.allocation_generation = kAllocationGeneration;
  manifest.contributions = std::move(contributions);
  return manifest;
}

DecodeKVReceipt make_receipt(const DecodeKVContributionKey& key,
                             const std::string& submission_id,
                             DecodeKVCompletionLevel completion_level,
                             uint32_t valid_tokens = 16) {
  DecodeKVReceipt receipt;
  receipt.request_id = "request-a";
  receipt.key = key;
  receipt.submission_id = submission_id;
  receipt.attempt_epoch = kAttemptEpoch;
  receipt.allocation_generation = kAllocationGeneration;
  receipt.valid_tokens = valid_tokens;
  receipt.completion_level = completion_level;
  return receipt;
}

TEST(DecodeKVReadinessSelectionTest, FullPrefixHitHasNoReceiverContributions) {
  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.remote_shared_num = 4;

  EXPECT_FALSE(has_decode_kv_receiver_contributions({mapping}));
}

TEST(DecodeKVReadinessSelectionTest,
     SharedGroupedBlocksAndInvalidPlaceholdersAreIgnored) {
  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::SWA);
  mapping.remote_ids = {10, std::numeric_limits<uint64_t>::max(), 12};
  mapping.logical_block_ordinals = {0, 1, 2};
  mapping.valid_tokens = {16, 0, 16};
  mapping.remote_shared_num = 3;

  EXPECT_FALSE(has_decode_kv_receiver_contributions({mapping}));
}

TEST(DecodeKVReadinessSelectionTest,
     UnsharedSuffixRequiresReceiverContributions) {
  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.remote_ids = {42};
  mapping.logical_block_ordinals = {4};
  mapping.valid_tokens = {16};
  mapping.remote_shared_num = 4;

  EXPECT_TRUE(has_decode_kv_receiver_contributions({mapping}));
}

TEST(DecodeKVReadinessSelectionTest, MalformedMetadataStillFailsClosed) {
  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.remote_ids = {42};

  EXPECT_TRUE(has_decode_kv_receiver_contributions({mapping}));
}

DecodeKVWorkerLayout make_replicated_worker_layout(int32_t worker_rank,
                                                   int32_t tp_size) {
  DecodeKVWorkerLayout worker_layout;
  worker_layout.worker_rank = worker_rank;
  WorkerCacheLayoutManifest& manifest = worker_layout.manifest;
  manifest.incarnation_id = "decode-" + std::to_string(worker_rank);
  manifest.layout_generation = 1;
  manifest.fingerprint = "glm-mla-layout";
  manifest.backend = "Mooncake";
  manifest.layout_family = "replicated-mla";
  manifest.cluster_id = static_cast<uint64_t>(worker_rank + 1);
  manifest.addr = "decode-addr-" + std::to_string(worker_rank);
  manifest.listen_port = static_cast<uint16_t>(20000 + worker_rank);
  manifest.coordinates.tp_rank = worker_rank;
  manifest.coordinates.tp_size = tp_size;

  CacheTensorManifest tensor;
  tensor.cache_namespace = CacheNamespace::MAIN;
  tensor.layer_id = 0;
  tensor.role = static_cast<int32_t>(KVCacheTensorRole::KEY);
  tensor.group_id = 5;
  tensor.mooncake_buffer_id = 0;
  tensor.scalar_type = 0;
  tensor.element_bytes = 1;
  tensor.shape = {64, 1};
  tensor.stride = {1, 1};
  tensor.contiguous = true;
  tensor.resource_count = 64;
  tensor.resource_stride_bytes = 1;
  tensor.buffer_bytes = 64;
  tensor.block_token_capacity = 128;
  tensor.shard.kind = LogicalShardKind::REPLICATED;
  tensor.shard.resource_scope = CacheResourceScope::BLOCK;
  LogicalSpan span;
  span.logical_tensor = "key";
  span.bytes_per_region = 1;
  span.owner_tp_rank = 0;
  tensor.shard.spans.emplace_back(std::move(span));
  manifest.tensors.emplace_back(std::move(tensor));
  return worker_layout;
}

TEST(DecodeKVExpectedContributionTest,
     ReplicatedMlaUsesStaticSourceOwnerAcrossDestinationTpReplicas) {
  DecodeKVSourceTopology source_topology;
  source_topology.world_size = 16;
  source_topology.dp_size = 1;
  source_topology.cp_size = 4;
  source_topology.kv_split_size = 2;

  KVTransferMapping mapping;
  mapping.group_id = 5;
  mapping.remote_ids = {41, 42};
  mapping.logical_block_ordinals = {0, 1};
  mapping.valid_tokens = {128, 22};

  std::vector<DecodeKVWorkerLayout> destination_layouts;
  destination_layouts.reserve(16);
  for (int32_t worker_rank = 0; worker_rank < 16; ++worker_rank) {
    destination_layouts.emplace_back(
        make_replicated_worker_layout(worker_rank, /*tp_size=*/16));
  }

  std::vector<DecodeKVExpectedContribution> contributions;
  const Status status =
      build_decode_kv_expected_contributions(source_topology,
                                             /*destination_dp_rank=*/0,
                                             {mapping},
                                             destination_layouts,
                                             &contributions);

  ASSERT_TRUE(status.ok()) << status.message();
  ASSERT_EQ(contributions.size(), 32U);
  for (int32_t destination_worker_rank = 0; destination_worker_rank < 16;
       ++destination_worker_rank) {
    const DecodeKVExpectedContribution& first =
        contributions[static_cast<size_t>(destination_worker_rank) * 2];
    const DecodeKVExpectedContribution& second =
        contributions[static_cast<size_t>(destination_worker_rank) * 2 + 1];
    EXPECT_EQ(first.key.source_worker_rank, 0);
    EXPECT_EQ(second.key.source_worker_rank, 8);
    EXPECT_EQ(first.key.destination_worker_rank, destination_worker_rank);
    EXPECT_EQ(second.key.destination_worker_rank, destination_worker_rank);
    EXPECT_EQ(first.key.group_id, 5);
    EXPECT_EQ(second.key.group_id, 5);
    EXPECT_EQ(first.key.logical_block_ordinal, 0U);
    EXPECT_EQ(second.key.logical_block_ordinal, 1U);
    EXPECT_EQ(first.key.destination_physical_block_id, 41U);
    EXPECT_EQ(second.key.destination_physical_block_id, 42U);
    EXPECT_EQ(first.valid_tokens, 128U);
    EXPECT_EQ(second.valid_tokens, 22U);
  }
}

TEST(DecodeKVExpectedContributionTest,
     RejectsTensorWithMultipleSourceTpOwners) {
  DecodeKVSourceTopology source_topology;
  source_topology.world_size = 16;
  source_topology.dp_size = 1;
  source_topology.cp_size = 4;
  source_topology.kv_split_size = 2;

  KVTransferMapping mapping;
  mapping.group_id = 5;
  mapping.remote_ids = {41};
  mapping.logical_block_ordinals = {0};
  mapping.valid_tokens = {22};

  DecodeKVWorkerLayout worker_layout =
      make_replicated_worker_layout(/*worker_rank=*/0, /*tp_size=*/16);
  CacheTensorManifest& tensor = worker_layout.manifest.tensors.front();
  tensor.shape = {64, 2};
  tensor.stride = {2, 1};
  tensor.resource_stride_bytes = 2;
  tensor.buffer_bytes = 128;
  tensor.shard.kind = LogicalShardKind::COMPOSITE;
  tensor.shard.spans.clear();
  LogicalSpan first;
  first.logical_tensor = "key-first";
  first.bytes_per_region = 1;
  first.owner_tp_rank = 0;
  tensor.shard.spans.emplace_back(std::move(first));
  LogicalSpan second;
  second.logical_tensor = "key-second";
  second.physical_offset_bytes = 1;
  second.bytes_per_region = 1;
  second.owner_tp_rank = 8;
  tensor.shard.spans.emplace_back(std::move(second));

  std::vector<DecodeKVExpectedContribution> contributions;
  const Status status =
      build_decode_kv_expected_contributions(source_topology,
                                             /*destination_dp_rank=*/0,
                                             {mapping},
                                             {worker_layout},
                                             &contributions);

  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("multiple source TP owners"),
            std::string::npos);
  EXPECT_TRUE(contributions.empty());
}

TEST(DecodeKVExpectedContributionTest,
     RejectsNonDivisibleDestinationTpTopology) {
  DecodeKVSourceTopology source_topology;
  source_topology.world_size = 12;
  source_topology.dp_size = 1;
  source_topology.cp_size = 4;
  source_topology.kv_split_size = 2;

  KVTransferMapping mapping;
  mapping.group_id = 5;
  mapping.remote_ids = {41};
  mapping.logical_block_ordinals = {0};
  mapping.valid_tokens = {22};

  const DecodeKVWorkerLayout worker_layout =
      make_replicated_worker_layout(/*worker_rank=*/0, /*tp_size=*/8);
  std::vector<DecodeKVExpectedContribution> contributions;
  const Status status =
      build_decode_kv_expected_contributions(source_topology,
                                             /*destination_dp_rank=*/0,
                                             {mapping},
                                             {worker_layout},
                                             &contributions);

  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("destination TP size"), std::string::npos);
  EXPECT_TRUE(contributions.empty());
}

TEST(DecodeKVReadinessPayloadTest, ManifestRoundTripPreservesIdentity) {
  const DecodeKVContributionKey key =
      make_key(/*layer_id=*/3,
               /*logical_block_ordinal=*/4,
               /*destination_physical_block_id=*/101);
  const DecodeKVExpectedManifest manifest =
      make_manifest({make_expected(key, /*valid_tokens=*/7)});
  std::string serialized;
  std::string error;

  ASSERT_TRUE(
      serialize_decode_kv_expected_manifest(manifest, &serialized, &error))
      << error;

  DecodeKVExpectedManifest round_trip;
  EXPECT_EQ(
      deserialize_decode_kv_expected_manifest(serialized, &round_trip, &error),
      DecodeKVPayloadResult::OK)
      << error;
  EXPECT_EQ(round_trip.request_id, manifest.request_id);
  EXPECT_EQ(round_trip.attempt_epoch, manifest.attempt_epoch);
  EXPECT_EQ(round_trip.allocation_generation, manifest.allocation_generation);
  ASSERT_EQ(round_trip.contributions.size(), 1u);
  const DecodeKVExpectedContribution& contribution =
      round_trip.contributions.front();
  EXPECT_EQ(contribution.key.source_worker_rank, key.source_worker_rank);
  EXPECT_EQ(contribution.key.destination_worker_rank,
            key.destination_worker_rank);
  EXPECT_EQ(contribution.key.layer_id, key.layer_id);
  EXPECT_EQ(contribution.key.group_id, key.group_id);
  EXPECT_EQ(contribution.key.cache_role, key.cache_role);
  EXPECT_EQ(contribution.key.logical_block_ordinal, key.logical_block_ordinal);
  EXPECT_EQ(contribution.key.destination_physical_block_id,
            key.destination_physical_block_id);
  EXPECT_EQ(contribution.valid_tokens, 7u);
}

TEST(DecodeKVReadinessPayloadTest,
     ManifestRejectsSchemaMismatchMalformedAndZeroIdentity) {
  const DecodeKVContributionKey key =
      make_key(/*layer_id=*/0,
               /*logical_block_ordinal=*/0,
               /*destination_physical_block_id=*/100);
  std::string serialized;
  std::string error;
  ASSERT_TRUE(serialize_decode_kv_expected_manifest(
      make_manifest({make_expected(key)}), &serialized, &error));

  proto::DecodeKVExpectedManifest proto_manifest;
  ASSERT_TRUE(proto_manifest.ParseFromString(serialized));
  proto_manifest.set_schema_version(kDecodeKVReadinessSchemaVersion + 1);
  ASSERT_TRUE(proto_manifest.SerializeToString(&serialized));
  DecodeKVExpectedManifest parsed;
  EXPECT_EQ(
      deserialize_decode_kv_expected_manifest(serialized, &parsed, &error),
      DecodeKVPayloadResult::VERSION_MISMATCH);

  const std::string malformed(1, static_cast<char>(0x80));
  EXPECT_EQ(deserialize_decode_kv_expected_manifest(malformed, &parsed, &error),
            DecodeKVPayloadResult::MALFORMED);

  proto_manifest.set_schema_version(kDecodeKVReadinessSchemaVersion);
  proto_manifest.set_attempt_epoch(0);
  ASSERT_TRUE(proto_manifest.SerializeToString(&serialized));
  EXPECT_EQ(
      deserialize_decode_kv_expected_manifest(serialized, &parsed, &error),
      DecodeKVPayloadResult::INVALID_ENVELOPE);
  EXPECT_NE(error.find("attempt_epoch"), std::string::npos);

  proto_manifest.set_attempt_epoch(kAttemptEpoch);
  proto_manifest.set_allocation_generation(0);
  ASSERT_TRUE(proto_manifest.SerializeToString(&serialized));
  EXPECT_EQ(
      deserialize_decode_kv_expected_manifest(serialized, &parsed, &error),
      DecodeKVPayloadResult::INVALID_ENVELOPE);
  EXPECT_NE(error.find("allocation_generation"), std::string::npos);
}

TEST(DecodeKVReadinessPayloadTest,
     MooncakeNotificationRoundTripPreservesFinalReceipts) {
  const DecodeKVContributionKey key =
      make_key(/*layer_id=*/2,
               /*logical_block_ordinal=*/3,
               /*destination_physical_block_id=*/103);
  MooncakeDecodeKVNotification notification;
  notification.submission_id = "submission-final";
  notification.batch_index = 1;
  notification.batch_count = 2;
  notification.receipts.emplace_back(
      make_receipt(key,
                   notification.submission_id,
                   DecodeKVCompletionLevel::REMOTE_VISIBLE,
                   /*valid_tokens=*/5));
  std::string serialized;
  std::string error;

  ASSERT_TRUE(serialize_mooncake_decode_kv_notification(
      notification, &serialized, &error))
      << error;

  MooncakeDecodeKVNotification round_trip;
  EXPECT_EQ(deserialize_mooncake_decode_kv_notification(
                serialized, &round_trip, &error),
            DecodeKVPayloadResult::OK)
      << error;
  EXPECT_EQ(round_trip.schema_version, kDecodeKVReadinessSchemaVersion);
  EXPECT_EQ(round_trip.submission_id, notification.submission_id);
  EXPECT_EQ(round_trip.batch_index, 1u);
  EXPECT_EQ(round_trip.batch_count, 2u);
  ASSERT_EQ(round_trip.receipts.size(), 1u);
  EXPECT_EQ(round_trip.receipts.front().request_id, "request-a");
  EXPECT_EQ(round_trip.receipts.front().key.logical_block_ordinal, 3u);
  EXPECT_EQ(round_trip.receipts.front().valid_tokens, 5u);
  EXPECT_EQ(round_trip.receipts.front().completion_level,
            DecodeKVCompletionLevel::REMOTE_VISIBLE);
}

TEST(DecodeKVReadinessPayloadTest,
     MooncakeNotificationRejectsInvalidWireEnvelopes) {
  const DecodeKVContributionKey key =
      make_key(/*layer_id=*/2,
               /*logical_block_ordinal=*/3,
               /*destination_physical_block_id=*/103);
  MooncakeDecodeKVNotification notification;
  notification.submission_id = "submission-final";
  notification.batch_index = 1;
  notification.batch_count = 2;
  notification.receipts.emplace_back(
      make_receipt(key,
                   notification.submission_id,
                   DecodeKVCompletionLevel::REMOTE_VISIBLE));
  std::string serialized;
  std::string error;
  ASSERT_TRUE(serialize_mooncake_decode_kv_notification(
      notification, &serialized, &error));

  proto::MooncakeKVNotification proto_notification;
  ASSERT_TRUE(proto_notification.ParseFromString(serialized));
  proto_notification.set_schema_version(kDecodeKVReadinessSchemaVersion + 1);
  ASSERT_TRUE(proto_notification.SerializeToString(&serialized));
  MooncakeDecodeKVNotification parsed;
  EXPECT_EQ(
      deserialize_mooncake_decode_kv_notification(serialized, &parsed, &error),
      DecodeKVPayloadResult::VERSION_MISMATCH);

  const std::string malformed(1, static_cast<char>(0x80));
  EXPECT_EQ(
      deserialize_mooncake_decode_kv_notification(malformed, &parsed, &error),
      DecodeKVPayloadResult::MALFORMED);

  notification.batch_index = 0;
  EXPECT_FALSE(serialize_mooncake_decode_kv_notification(
      notification, &serialized, &error));
  EXPECT_NE(error.find("final batch"), std::string::npos);

  proto_notification.set_schema_version(kDecodeKVReadinessSchemaVersion);
  proto_notification.set_batch_index(0);
  ASSERT_TRUE(proto_notification.SerializeToString(&serialized));
  EXPECT_EQ(
      deserialize_mooncake_decode_kv_notification(serialized, &parsed, &error),
      DecodeKVPayloadResult::INVALID_ENVELOPE);
  EXPECT_NE(error.find("final batch"), std::string::npos);
}

TEST(DecodeKVReadinessLedgerTest, ZeroIdentityGenerationsFailClosed) {
  const DecodeKVContributionKey key =
      make_key(/*layer_id=*/0,
               /*logical_block_ordinal=*/0,
               /*destination_physical_block_id=*/100);
  DecodeKVExpectedManifest zero_attempt = make_manifest({make_expected(key)});
  zero_attempt.attempt_epoch = 0;
  DecodeKVReadinessLedger zero_attempt_ledger(std::move(zero_attempt));
  EXPECT_TRUE(zero_attempt_ledger.is_poisoned());
  EXPECT_NE(zero_attempt_ledger.failure_reason().find("attempt_epoch"),
            std::string::npos);

  DecodeKVExpectedManifest zero_allocation =
      make_manifest({make_expected(key)});
  zero_allocation.allocation_generation = 0;
  DecodeKVReadinessLedger zero_allocation_ledger(std::move(zero_allocation));
  EXPECT_TRUE(zero_allocation_ledger.is_poisoned());
  EXPECT_NE(
      zero_allocation_ledger.failure_reason().find("allocation_generation"),
      std::string::npos);
}

TEST(DecodeKVReadinessLedgerTest, EmptyManifestFailsClosedWithDiagnostic) {
  DecodeKVReadinessLedger ledger(make_manifest({}));

  EXPECT_TRUE(ledger.is_poisoned());
  EXPECT_FALSE(ledger.is_ready());
  EXPECT_FALSE(ledger.try_publish());
  EXPECT_NE(ledger.failure_reason().find("at least one contribution"),
            std::string::npos);
}

TEST(DecodeKVReadinessLedgerTest, RejectsMultipleOwnersForLogicalBlock) {
  const DecodeKVContributionKey first_key =
      make_key(/*layer_id=*/0,
               /*logical_block_ordinal=*/0,
               /*destination_physical_block_id=*/100);
  DecodeKVContributionKey second_key =
      make_key(/*layer_id=*/0,
               /*logical_block_ordinal=*/0,
               /*destination_physical_block_id=*/101);
  second_key.source_worker_rank = 2;
  DecodeKVReadinessLedger ledger(
      make_manifest({make_expected(first_key), make_expected(second_key)}));

  EXPECT_TRUE(ledger.is_poisoned());
  EXPECT_FALSE(ledger.is_ready());
  EXPECT_NE(ledger.failure_reason().find("multiple owners"), std::string::npos);
}

TEST(DecodeKVReadinessLedgerTest,
     RejectsDestinationAllocationCollisionAcrossLogicalBlocks) {
  const DecodeKVContributionKey first_key =
      make_key(/*layer_id=*/0,
               /*logical_block_ordinal=*/0,
               /*destination_physical_block_id=*/100);
  const DecodeKVContributionKey second_key =
      make_key(/*layer_id=*/0,
               /*logical_block_ordinal=*/1,
               /*destination_physical_block_id=*/100);
  DecodeKVReadinessLedger ledger(
      make_manifest({make_expected(first_key), make_expected(second_key)}));

  EXPECT_TRUE(ledger.is_poisoned());
  EXPECT_FALSE(ledger.is_ready());
  EXPECT_NE(ledger.failure_reason().find("destination allocation"),
            std::string::npos);
}

TEST(DecodeKVReadinessLedgerTest, MissingContributionNeverBecomesReady) {
  const DecodeKVContributionKey first_key =
      make_key(/*layer_id=*/0,
               /*logical_block_ordinal=*/0,
               /*destination_physical_block_id=*/100);
  const DecodeKVContributionKey missing_key =
      make_key(/*layer_id=*/0,
               /*logical_block_ordinal=*/1,
               /*destination_physical_block_id=*/101);
  DecodeKVReadinessLedger ledger(
      make_manifest({make_expected(first_key),
                     make_expected(missing_key, /*valid_tokens=*/3)}));

  EXPECT_EQ(ledger.request_id(), "request-a");
  EXPECT_EQ(
      ledger.record(make_receipt(first_key,
                                 /*submission_id=*/"submission-first",
                                 DecodeKVCompletionLevel::REMOTE_VISIBLE)),
      DecodeKVReceiptRecordResult::RECORDED);
  EXPECT_FALSE(ledger.is_ready());
  EXPECT_FALSE(ledger.try_publish());

  const std::vector<DecodeKVExpectedContribution> missing = ledger.missing();
  ASSERT_EQ(missing.size(), 1U);
  EXPECT_EQ(missing[0].key.logical_block_ordinal,
            missing_key.logical_block_ordinal);
  EXPECT_EQ(missing[0].valid_tokens, 3U);
}

TEST(DecodeKVReadinessLedgerTest, ConflictingDuplicatePoisonsLedger) {
  const DecodeKVContributionKey first_key =
      make_key(/*layer_id=*/0,
               /*logical_block_ordinal=*/0,
               /*destination_physical_block_id=*/100);
  const DecodeKVContributionKey second_key =
      make_key(/*layer_id=*/0,
               /*logical_block_ordinal=*/1,
               /*destination_physical_block_id=*/101);
  DecodeKVReadinessLedger ledger(
      make_manifest({make_expected(first_key), make_expected(second_key)}));

  EXPECT_EQ(
      ledger.record(make_receipt(first_key,
                                 /*submission_id=*/"submission-original",
                                 DecodeKVCompletionLevel::REMOTE_VISIBLE)),
      DecodeKVReceiptRecordResult::RECORDED);
  EXPECT_EQ(
      ledger.record(make_receipt(first_key,
                                 /*submission_id=*/"submission-conflict",
                                 DecodeKVCompletionLevel::REMOTE_VISIBLE)),
      DecodeKVReceiptRecordResult::CONFLICTING_DUPLICATE);
  EXPECT_EQ(
      ledger.record(make_receipt(second_key,
                                 /*submission_id=*/"submission-second",
                                 DecodeKVCompletionLevel::REMOTE_VISIBLE)),
      DecodeKVReceiptRecordResult::LEDGER_POISONED);

  EXPECT_FALSE(ledger.is_ready());
  EXPECT_FALSE(ledger.try_publish());
}

TEST(DecodeKVReadinessLedgerTest, IdenticalSubmissionReplayIsIdempotent) {
  const DecodeKVContributionKey key =
      make_key(/*layer_id=*/1,
               /*logical_block_ordinal=*/0,
               /*destination_physical_block_id=*/200);
  DecodeKVReadinessLedger ledger(make_manifest({make_expected(key)}));
  const DecodeKVReceipt receipt =
      make_receipt(key,
                   /*submission_id=*/"submission-replayed",
                   DecodeKVCompletionLevel::REMOTE_VISIBLE);

  EXPECT_EQ(ledger.record(receipt), DecodeKVReceiptRecordResult::RECORDED);
  EXPECT_EQ(ledger.record(receipt),
            DecodeKVReceiptRecordResult::IDEMPOTENT_REPLAY);
  EXPECT_TRUE(ledger.is_ready());
}

TEST(DecodeKVReadinessLedgerTest, RejectsStaleAttemptAndAllocationGeneration) {
  const DecodeKVContributionKey key =
      make_key(/*layer_id=*/2,
               /*logical_block_ordinal=*/0,
               /*destination_physical_block_id=*/300);
  DecodeKVReadinessLedger ledger(make_manifest({make_expected(key)}));

  DecodeKVReceipt stale_attempt =
      make_receipt(key,
                   /*submission_id=*/"submission-stale-attempt",
                   DecodeKVCompletionLevel::REMOTE_VISIBLE);
  stale_attempt.attempt_epoch = kAttemptEpoch - 1;
  EXPECT_EQ(ledger.record(stale_attempt),
            DecodeKVReceiptRecordResult::STALE_ATTEMPT);

  DecodeKVReceipt stale_allocation =
      make_receipt(key,
                   /*submission_id=*/"submission-stale-allocation",
                   DecodeKVCompletionLevel::REMOTE_VISIBLE);
  stale_allocation.allocation_generation = kAllocationGeneration - 1;
  EXPECT_EQ(ledger.record(stale_allocation),
            DecodeKVReceiptRecordResult::LEDGER_POISONED);

  EXPECT_EQ(
      ledger.record(make_receipt(key,
                                 /*submission_id=*/"submission-current",
                                 DecodeKVCompletionLevel::REMOTE_VISIBLE)),
      DecodeKVReceiptRecordResult::LEDGER_POISONED);
  EXPECT_FALSE(ledger.is_ready());
  EXPECT_FALSE(ledger.try_publish());
}

TEST(DecodeKVReadinessLedgerTest, BindsRequestAndPartialBlockCoverage) {
  const DecodeKVContributionKey key =
      make_key(/*layer_id=*/2,
               /*logical_block_ordinal=*/0,
               /*destination_physical_block_id=*/300);
  DecodeKVReadinessLedger request_mismatch_ledger(
      make_manifest({make_expected(key, /*valid_tokens=*/3)}));
  DecodeKVReceipt request_mismatch =
      make_receipt(key,
                   /*submission_id=*/"submission-request-mismatch",
                   DecodeKVCompletionLevel::REMOTE_VISIBLE,
                   /*valid_tokens=*/3);
  request_mismatch.request_id = "request-b";
  EXPECT_EQ(request_mismatch_ledger.record(request_mismatch),
            DecodeKVReceiptRecordResult::REQUEST_MISMATCH);
  EXPECT_FALSE(request_mismatch_ledger.is_ready());

  DecodeKVReadinessLedger coverage_mismatch_ledger(
      make_manifest({make_expected(key, /*valid_tokens=*/3)}));
  EXPECT_EQ(coverage_mismatch_ledger.record(
                make_receipt(key,
                             /*submission_id=*/"submission-coverage-mismatch",
                             DecodeKVCompletionLevel::REMOTE_VISIBLE,
                             /*valid_tokens=*/16)),
            DecodeKVReceiptRecordResult::VALID_TOKEN_MISMATCH);
  EXPECT_FALSE(coverage_mismatch_ledger.is_ready());

  DecodeKVReadinessLedger complete_ledger(
      make_manifest({make_expected(key, /*valid_tokens=*/3)}));
  EXPECT_EQ(complete_ledger.record(
                make_receipt(key,
                             /*submission_id=*/"submission-complete",
                             DecodeKVCompletionLevel::REMOTE_VISIBLE,
                             /*valid_tokens=*/3)),
            DecodeKVReceiptRecordResult::RECORDED);
  EXPECT_TRUE(complete_ledger.is_ready());
}

TEST(DecodeKVReadinessLedgerTest, DelayedFinalRemoteVisibleReceiptMakesReady) {
  const DecodeKVContributionKey first_key =
      make_key(/*layer_id=*/3,
               /*logical_block_ordinal=*/0,
               /*destination_physical_block_id=*/400);
  const DecodeKVContributionKey delayed_key =
      make_key(/*layer_id=*/3,
               /*logical_block_ordinal=*/1,
               /*destination_physical_block_id=*/401);
  DecodeKVReadinessLedger ledger(
      make_manifest({make_expected(first_key), make_expected(delayed_key)}));

  EXPECT_EQ(ledger.record(make_receipt(first_key,
                                       /*submission_id=*/"submission-first",
                                       DecodeKVCompletionLevel::SUBMITTED)),
            DecodeKVReceiptRecordResult::RECORDED);
  EXPECT_FALSE(ledger.is_ready());
  EXPECT_EQ(
      ledger.record(make_receipt(first_key,
                                 /*submission_id=*/"submission-first",
                                 DecodeKVCompletionLevel::REMOTE_VISIBLE)),
      DecodeKVReceiptRecordResult::COMPLETION_ADVANCED);
  EXPECT_EQ(
      ledger.record(make_receipt(delayed_key,
                                 /*submission_id=*/"submission-delayed",
                                 DecodeKVCompletionLevel::SOURCE_COMPLETE)),
      DecodeKVReceiptRecordResult::RECORDED);
  EXPECT_FALSE(ledger.is_ready());

  EXPECT_EQ(
      ledger.record(make_receipt(delayed_key,
                                 /*submission_id=*/"submission-delayed",
                                 DecodeKVCompletionLevel::REMOTE_VISIBLE)),
      DecodeKVReceiptRecordResult::COMPLETION_ADVANCED);
  EXPECT_TRUE(ledger.is_ready());
  EXPECT_TRUE(ledger.missing().empty());
}

TEST(DecodeKVReadinessLedgerTest, PublishesExactlyOnce) {
  const DecodeKVContributionKey key =
      make_key(/*layer_id=*/4,
               /*logical_block_ordinal=*/0,
               /*destination_physical_block_id=*/500);
  DecodeKVReadinessLedger ledger(make_manifest({make_expected(key)}));

  EXPECT_FALSE(ledger.try_publish());
  EXPECT_EQ(
      ledger.record(make_receipt(key,
                                 /*submission_id=*/"submission-ready",
                                 DecodeKVCompletionLevel::REMOTE_VISIBLE)),
      DecodeKVReceiptRecordResult::RECORDED);
  EXPECT_TRUE(ledger.try_publish());
  EXPECT_FALSE(ledger.try_publish());
  EXPECT_TRUE(ledger.was_published());

  EXPECT_EQ(
      ledger.record(make_receipt(key,
                                 /*submission_id=*/"submission-ready",
                                 DecodeKVCompletionLevel::REMOTE_VISIBLE)),
      DecodeKVReceiptRecordResult::IDEMPOTENT_REPLAY);
  EXPECT_TRUE(ledger.is_ready());
}

TEST(DecodeKVReadinessLedgerTest, ConflictingReceiptAfterPublishPoisonsLedger) {
  const DecodeKVContributionKey key =
      make_key(/*layer_id=*/4,
               /*logical_block_ordinal=*/0,
               /*destination_physical_block_id=*/501);
  DecodeKVReadinessLedger ledger(make_manifest({make_expected(key)}));

  EXPECT_EQ(
      ledger.record(make_receipt(key,
                                 /*submission_id=*/"submission-original",
                                 DecodeKVCompletionLevel::REMOTE_VISIBLE)),
      DecodeKVReceiptRecordResult::RECORDED);
  ASSERT_TRUE(ledger.try_publish());
  EXPECT_EQ(
      ledger.record(make_receipt(key,
                                 /*submission_id=*/"submission-conflict",
                                 DecodeKVCompletionLevel::REMOTE_VISIBLE)),
      DecodeKVReceiptRecordResult::CONFLICTING_DUPLICATE);
  EXPECT_TRUE(ledger.was_published());
  EXPECT_TRUE(ledger.is_poisoned());
  EXPECT_FALSE(ledger.is_ready());
}

TEST(DecodeKVReadinessLedgerTest, ConcurrentCallbacksPublishExactlyOnce) {
  const DecodeKVContributionKey first_key =
      make_key(/*layer_id=*/5,
               /*logical_block_ordinal=*/0,
               /*destination_physical_block_id=*/600);
  const DecodeKVContributionKey second_key =
      make_key(/*layer_id=*/5,
               /*logical_block_ordinal=*/1,
               /*destination_physical_block_id=*/601);
  DecodeKVReadinessLedger ledger(
      make_manifest({make_expected(first_key), make_expected(second_key)}));

  std::future<DecodeKVReceiptRecordResult> first_record =
      std::async(std::launch::async, [&ledger, &first_key]() {
        return ledger.record(
            make_receipt(first_key,
                         /*submission_id=*/"submission-first",
                         DecodeKVCompletionLevel::REMOTE_VISIBLE));
      });
  std::future<DecodeKVReceiptRecordResult> second_record =
      std::async(std::launch::async, [&ledger, &second_key]() {
        return ledger.record(
            make_receipt(second_key,
                         /*submission_id=*/"submission-second",
                         DecodeKVCompletionLevel::REMOTE_VISIBLE));
      });
  EXPECT_EQ(first_record.get(), DecodeKVReceiptRecordResult::RECORDED);
  EXPECT_EQ(second_record.get(), DecodeKVReceiptRecordResult::RECORDED);
  ASSERT_TRUE(ledger.is_ready());

  std::vector<std::future<bool>> publish_results;
  publish_results.reserve(8);
  for (int32_t index = 0; index < 8; ++index) {
    publish_results.emplace_back(std::async(
        std::launch::async, [&ledger]() { return ledger.try_publish(); }));
  }

  int32_t publish_count = 0;
  for (std::future<bool>& result : publish_results) {
    if (result.get()) {
      ++publish_count;
    }
  }
  EXPECT_EQ(publish_count, 1);
  EXPECT_TRUE(ledger.was_published());
}

}  // namespace
}  // namespace xllm
