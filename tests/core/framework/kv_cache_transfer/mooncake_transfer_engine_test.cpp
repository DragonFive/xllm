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

#include "framework/kv_cache_transfer/mooncake_transfer_engine.h"

#include <brpc/controller.h>
#include <gtest/gtest.h>

#if defined(USE_NPU)
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "framework/kv_cache/kv_cache_capacity.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/kv_cache_transfer/kv_cache_transfer.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "util/net.h"
#include "worker.pb.h"

#define private public
#define protected public
#include "framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"
#undef private
#undef protected

namespace xllm {

namespace {

constexpr size_t kScaleBlockBytes = 96 * sizeof(float);

#if defined(USE_NPU)
constexpr std::chrono::seconds kTransferCompletionTimeout{5};

bool round_trip_legacy_mooncake_notification(
    const TransferMetadata::NotifyDesc& sent,
    TransferMetadata::NotifyDesc* received) {
  Json::Value outbound;
  outbound["name"] = sent.name;
  outbound["notify_msg"] = sent.notify_msg;
  const std::string json = Json::FastWriter{}.write(outbound);

  int sockets[2] = {-1, -1};
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
    return false;
  }
  const int32_t write_result = mooncake::writeString(
      sockets[0], mooncake::HandShakeRequestType::Notify, json);
  const auto [request_type, received_json] = mooncake::readString(sockets[1]);
  close(sockets[0]);
  close(sockets[1]);
  if (write_result != 0 ||
      request_type != mooncake::HandShakeRequestType::Notify) {
    return false;
  }

  Json::CharReaderBuilder builder;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  Json::Value inbound;
  std::string error;
  if (!reader->parse(received_json.data(),
                     received_json.data() + received_json.size(),
                     &inbound,
                     &error)) {
    return false;
  }
  received->name = inbound["name"].asString();
  received->notify_msg = inbound["notify_msg"].asString();
  return true;
}
#endif

TransferKVInfo make_info(int32_t dst_dp_size,
                         int32_t dst_tp_size,
                         int32_t dst_dp_rank) {
  TransferKVInfo info;
  info.request_id = "req";
  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.local_ids = {11, 12};
  mapping.remote_ids = {21, 22};
  info.mappings.emplace_back(std::move(mapping));
  info.dp_rank = dst_dp_rank;
  info.remote_instance_info.dp_size = dst_dp_size;

  int32_t dst_world_size = dst_dp_size * dst_tp_size;
  for (int32_t i = 0; i < dst_world_size; ++i) {
    info.remote_instance_info.cluster_ids.emplace_back(
        static_cast<uint64_t>(100 + i));
    info.remote_instance_info.addrs.emplace_back("addr_" + std::to_string(i));
  }

  return info;
}

ParallelArgs make_args(int32_t rank, int32_t world_size, int32_t dp_size) {
  return ParallelArgs(rank, world_size, dp_size, nullptr);
}

class GenericKVCacheTransferForTest final : public KVCacheTransfer {
 public:
  void get_cache_info(uint64_t& cluster_id, std::string& addr) override {
    cluster_id = 0;
    addr.clear();
  }

  bool link_clusters(const std::vector<uint64_t>& cluster_ids,
                     const std::vector<std::string>& remote_addrs,
                     const std::vector<uint16_t>& ports) override {
    return cluster_ids.size() == remote_addrs.size() &&
           cluster_ids.size() == ports.size();
  }

  bool unlink_cluster(const uint64_t& cluster_id,
                      const std::string& remote_addr,
                      uint16_t port,
                      bool force_flag) override {
    return cluster_id == 0 && remote_addr.empty() && port == 0 && force_flag;
  }

  bool pull_kv_blocks(const uint64_t src_cluster_id,
                      const std::string& src_addr,
                      const std::vector<KVTransferMapping>& mappings) override {
    return src_cluster_id == 0 && src_addr.empty() && mappings.empty();
  }

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
  bool push_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
      bool is_spec_draft,
      int32_t kv_split_rank,
      int32_t kv_split_size) override {
    return merged_kv_infos.empty() && layer_synchronizer == nullptr &&
           !is_spec_draft && kv_split_rank == 0 && kv_split_size == 1;
  }
#endif
};

DecodeKVExpectedContribution make_expected_contribution(
    int32_t source_worker_rank,
    int32_t destination_worker_rank,
    int64_t layer_id,
    int32_t group_id,
    KVCacheTensorRole::Value cache_role,
    uint64_t logical_block_ordinal,
    uint64_t destination_physical_block_id,
    uint32_t valid_tokens) {
  DecodeKVExpectedContribution contribution;
  contribution.key.source_worker_rank = source_worker_rank;
  contribution.key.destination_worker_rank = destination_worker_rank;
  contribution.key.layer_id = layer_id;
  contribution.key.group_id = group_id;
  contribution.key.cache_role = cache_role;
  contribution.key.logical_block_ordinal = logical_block_ordinal;
  contribution.key.destination_physical_block_id =
      destination_physical_block_id;
  contribution.valid_tokens = valid_tokens;
  return contribution;
}

TransferKVInfo make_strict_receipt_info(
    const std::string& request_id,
    uint64_t attempt_epoch,
    uint64_t allocation_generation,
    std::vector<KVTransferMapping> mappings,
    std::vector<DecodeKVExpectedContribution> contributions) {
  DecodeKVExpectedManifest manifest;
  manifest.request_id = request_id;
  manifest.attempt_epoch = attempt_epoch;
  manifest.allocation_generation = allocation_generation;
  manifest.contributions = std::move(contributions);

  TransferKVInfo info;
  info.request_id = request_id;
  info.attempt_epoch = attempt_epoch;
  info.allocation_generation = allocation_generation;
  info.mappings = std::move(mappings);
  std::string error;
  EXPECT_TRUE(serialize_decode_kv_expected_manifest(
      manifest, &info.decode_kv_manifest, &error))
      << error;
  return info;
}

TEST(MooncakeTransferEngineTest,
     PreparesLayerReceiptsOnceWithAllCacheRolesAndIdentity) {
  constexpr int32_t kSourceWorkerRank = 2;
  constexpr int32_t kDestinationWorkerRank = 3;
  constexpr int32_t kGroupId = 17;
  constexpr uint64_t kAttemptEpoch = 7;
  constexpr uint64_t kAllocationGeneration = 11;

  KVTransferMapping mapping;
  mapping.group_id = kGroupId;
  mapping.remote_ids = {41, 42};
  mapping.logical_block_ordinals = {5, 6};
  mapping.valid_tokens = {16, 3};
  mapping.receipt_remote_ids = {42};

  std::vector<DecodeKVExpectedContribution> contributions;
  for (int64_t layer_id = 0; layer_id < 2; ++layer_id) {
    contributions.emplace_back(
        make_expected_contribution(kSourceWorkerRank,
                                   kDestinationWorkerRank,
                                   layer_id,
                                   kGroupId,
                                   KVCacheTensorRole::KEY,
                                   /*logical_block_ordinal=*/5,
                                   /*destination_physical_block_id=*/41,
                                   /*valid_tokens=*/16));
    contributions.emplace_back(
        make_expected_contribution(kSourceWorkerRank,
                                   kDestinationWorkerRank,
                                   layer_id,
                                   kGroupId,
                                   KVCacheTensorRole::KEY,
                                   /*logical_block_ordinal=*/6,
                                   /*destination_physical_block_id=*/42,
                                   /*valid_tokens=*/3));
    contributions.emplace_back(
        make_expected_contribution(kSourceWorkerRank,
                                   kDestinationWorkerRank,
                                   layer_id,
                                   kGroupId,
                                   KVCacheTensorRole::VALUE,
                                   /*logical_block_ordinal=*/6,
                                   /*destination_physical_block_id=*/42,
                                   /*valid_tokens=*/3));
  }

  KVCacheTransfer::KVCacheInfo kv_info;
  kv_info.source_worker_rank = kSourceWorkerRank;
  kv_info.destination_worker_rank = kDestinationWorkerRank;
  kv_info.receipt_infos.emplace_back(
      make_strict_receipt_info("request-a",
                               kAttemptEpoch,
                               kAllocationGeneration,
                               {mapping},
                               std::move(contributions)));

  detail::PreparedMooncakeDecodeKVNotifications prepared;
  std::string error;
  ASSERT_TRUE(detail::prepare_mooncake_decode_kv_notifications(
      kv_info, /*num_layers=*/2, &prepared, &error))
      << error;
  ASSERT_EQ(prepared.receipts_by_layer.size(), 2u);
  for (int64_t layer_id = 0; layer_id < 2; ++layer_id) {
    const std::vector<DecodeKVReceipt>& receipts =
        prepared.receipts_by_layer[static_cast<size_t>(layer_id)];
    ASSERT_EQ(receipts.size(), 2u);
    EXPECT_EQ(receipts[0].request_id, "request-a");
    EXPECT_EQ(receipts[0].attempt_epoch, kAttemptEpoch);
    EXPECT_EQ(receipts[0].allocation_generation, kAllocationGeneration);
    EXPECT_EQ(receipts[0].key.layer_id, layer_id);
    EXPECT_EQ(receipts[0].key.cache_role, KVCacheTensorRole::KEY);
    EXPECT_EQ(receipts[1].key.cache_role, KVCacheTensorRole::VALUE);
    for (const DecodeKVReceipt& receipt : receipts) {
      EXPECT_EQ(receipt.key.logical_block_ordinal, 6u);
      EXPECT_EQ(receipt.key.destination_physical_block_id, 42u);
      EXPECT_EQ(receipt.valid_tokens, 3u);
      EXPECT_EQ(receipt.completion_level,
                DecodeKVCompletionLevel::REMOTE_VISIBLE);
    }
  }
}

TEST(MooncakeTransferEngineTest,
     PreparedReceiptsPreserveFirstDuplicateRemoteMapping) {
  constexpr int32_t kGroupId = 17;
  KVTransferMapping mapping;
  mapping.group_id = kGroupId;
  mapping.remote_ids = {42, 42};
  mapping.logical_block_ordinals = {7, 8};
  mapping.valid_tokens = {16, 4};
  mapping.receipt_remote_ids = {42};

  std::vector<DecodeKVExpectedContribution> contributions;
  for (int64_t layer_id = 0; layer_id < 2; ++layer_id) {
    contributions.emplace_back(make_expected_contribution(
        /*source_worker_rank=*/2,
        /*destination_worker_rank=*/3,
        layer_id,
        kGroupId,
        KVCacheTensorRole::KEY,
        /*logical_block_ordinal=*/7,
        /*destination_physical_block_id=*/42,
        /*valid_tokens=*/16));
  }

  KVCacheTransfer::KVCacheInfo kv_info;
  kv_info.source_worker_rank = 2;
  kv_info.destination_worker_rank = 3;
  kv_info.receipt_infos.emplace_back(
      make_strict_receipt_info("request-a",
                               /*attempt_epoch=*/7,
                               /*allocation_generation=*/11,
                               {mapping},
                               std::move(contributions)));

  detail::PreparedMooncakeDecodeKVNotifications prepared;
  std::string error;
  ASSERT_TRUE(detail::prepare_mooncake_decode_kv_notifications(
      kv_info, /*num_layers=*/2, &prepared, &error))
      << error;
  ASSERT_EQ(prepared.receipts_by_layer.size(), 2u);
  for (const std::vector<DecodeKVReceipt>& receipts :
       prepared.receipts_by_layer) {
    ASSERT_EQ(receipts.size(), 1u);
    EXPECT_EQ(receipts[0].key.logical_block_ordinal, 7u);
    EXPECT_EQ(receipts[0].valid_tokens, 16u);
  }
}

TEST(MooncakeTransferEngineTest, PreparedReceiptsRejectIdentityAndTokenDrift) {
  constexpr int32_t kGroupId = 17;
  KVTransferMapping mapping;
  mapping.group_id = kGroupId;
  mapping.remote_ids = {42};
  mapping.logical_block_ordinals = {7};
  mapping.valid_tokens = {16};
  mapping.receipt_remote_ids = {42};
  std::vector<DecodeKVExpectedContribution> contributions = {
      make_expected_contribution(/*source_worker_rank=*/2,
                                 /*destination_worker_rank=*/3,
                                 /*layer_id=*/0,
                                 kGroupId,
                                 KVCacheTensorRole::KEY,
                                 /*logical_block_ordinal=*/7,
                                 /*destination_physical_block_id=*/42,
                                 /*valid_tokens=*/16)};

  KVCacheTransfer::KVCacheInfo kv_info;
  kv_info.source_worker_rank = 2;
  kv_info.destination_worker_rank = 3;
  kv_info.receipt_infos.emplace_back(
      make_strict_receipt_info("request-a",
                               /*attempt_epoch=*/7,
                               /*allocation_generation=*/11,
                               {mapping},
                               contributions));

  detail::PreparedMooncakeDecodeKVNotifications prepared;
  std::string error;
  kv_info.receipt_infos[0].attempt_epoch = 8;
  EXPECT_FALSE(detail::prepare_mooncake_decode_kv_notifications(
      kv_info, /*num_layers=*/1, &prepared, &error));
  EXPECT_NE(error.find("identity"), std::string::npos);

  kv_info.receipt_infos[0].attempt_epoch = 7;
  kv_info.receipt_infos[0].mappings[0].valid_tokens[0] = 15;
  EXPECT_FALSE(detail::prepare_mooncake_decode_kv_notifications(
      kv_info, /*num_layers=*/1, &prepared, &error));
  EXPECT_NE(error.find("valid-token"), std::string::npos);
}

TEST(MooncakeTransferEngineTest,
     RegionBatchingPublishesReceiptsOnlyOnFinalBatch) {
  EXPECT_EQ(detail::mooncake_transfer_batch_count(/*region_count=*/0), 0u);
  EXPECT_EQ(detail::mooncake_transfer_batch_count(/*region_count=*/4096), 1u);
  EXPECT_EQ(detail::mooncake_transfer_batch_count(/*region_count=*/4097), 2u);

  MooncakeDecodeKVNotification notification;
  notification.submission_id = "submission-a";
  DecodeKVReceipt receipt;
  receipt.request_id = "request-a";
  receipt.attempt_epoch = 7;
  receipt.allocation_generation = 11;
  receipt.valid_tokens = 3;
  receipt.completion_level = DecodeKVCompletionLevel::REMOTE_VISIBLE;
  notification.receipts.emplace_back(std::move(receipt));

  const MooncakeDecodeKVNotification first_batch =
      detail::make_mooncake_batch_notification(
          notification, /*batch_index=*/0, /*batch_count=*/2);
  EXPECT_EQ(first_batch.batch_index, 0u);
  EXPECT_EQ(first_batch.batch_count, 2u);
  EXPECT_TRUE(first_batch.receipts.empty());

  const MooncakeDecodeKVNotification final_batch =
      detail::make_mooncake_batch_notification(
          notification, /*batch_index=*/1, /*batch_count=*/2);
  EXPECT_EQ(final_batch.batch_index, 1u);
  EXPECT_EQ(final_batch.batch_count, 2u);
  ASSERT_EQ(final_batch.receipts.size(), 1u);
  EXPECT_EQ(final_batch.receipts.front().request_id, "request-a");
}

TEST(MooncakeTransferEngineTest, PendingNotificationDrainIsBoundedAndFiltered) {
  const std::string first =
      detail::frame_mooncake_notification_payload("first");
  const std::string second =
      detail::frame_mooncake_notification_payload("second");
  const std::string third =
      detail::frame_mooncake_notification_payload("third");
  std::deque<TransferMetadata::NotifyDesc> pending = {
      {"other", "unrelated"},
      {kMooncakeDecodeKVNotificationName, first},
      {kMooncakeDecodeKVNotificationName, second},
      {kMooncakeDecodeKVNotificationName, third}};
  std::vector<std::string> payloads;
  bool more_available = false;
  std::string error;

  ASSERT_TRUE(detail::drain_mooncake_pending_notifications(
      &pending,
      kMooncakeDecodeKVNotificationName,
      /*max_notifications=*/2,
      &payloads,
      &more_available,
      &error))
      << error;
  EXPECT_EQ(payloads, (std::vector<std::string>{"first", "second"}));
  EXPECT_TRUE(more_available);
  ASSERT_EQ(pending.size(), 2u);
  EXPECT_EQ(pending.front().name, "other");

  ASSERT_TRUE(detail::drain_mooncake_pending_notifications(
      &pending,
      kMooncakeDecodeKVNotificationName,
      /*max_notifications=*/2,
      &payloads,
      &more_available,
      &error))
      << error;
  EXPECT_EQ(payloads, (std::vector<std::string>{"third"}));
  EXPECT_FALSE(more_available);
  ASSERT_EQ(pending.size(), 1u);
  EXPECT_EQ(pending.front().notify_msg, "unrelated");
}

TEST(MooncakeTransferEngineTest,
     NotificationCarrierFramePreservesEmptyReceiptPayload) {
  MooncakeDecodeKVNotification notification;
  notification.submission_id = "submission-empty";

  std::string serialized;
  std::string error;
  ASSERT_TRUE(serialize_mooncake_decode_kv_notification(
      notification, &serialized, &error))
      << error;

  std::string decoded;
  ASSERT_TRUE(detail::unframe_mooncake_notification_payload(
      serialized, &decoded, &error))
      << error;
  EXPECT_EQ(decoded, serialized);

  const std::string framed =
      detail::frame_mooncake_notification_payload(serialized);
  EXPECT_TRUE(std::all_of(framed.begin(), framed.end(), [](char byte) {
    return static_cast<unsigned char>(byte) < 0x80;
  }));

  decoded.clear();
  ASSERT_TRUE(
      detail::unframe_mooncake_notification_payload(framed, &decoded, &error))
      << error;
  EXPECT_EQ(decoded, serialized);

  MooncakeDecodeKVNotification parsed;
  ASSERT_EQ(
      deserialize_mooncake_decode_kv_notification(decoded, &parsed, &error),
      DecodeKVPayloadResult::OK)
      << error;
  EXPECT_EQ(parsed.submission_id, notification.submission_id);
  EXPECT_TRUE(parsed.receipts.empty());
}

TEST(MooncakeTransferEngineTest,
     NotificationCarrierFrameRejectsMalformedInputsAndPartialDrain) {
  std::string decoded = "stale";
  std::string error;
  ASSERT_TRUE(detail::unframe_mooncake_notification_payload(
      "legacy-raw-payload", &decoded, &error));
  EXPECT_EQ(decoded, "legacy-raw-payload");

  decoded = "stale";
  error.clear();
  EXPECT_FALSE(detail::unframe_mooncake_notification_payload(
      std::string(detail::kMooncakeDecodeKVNotificationFramePrefix) + "1:%",
      &decoded,
      &error));
  EXPECT_TRUE(decoded.empty());
  EXPECT_NE(error.find("Base64"), std::string::npos);

  decoded = "stale";
  error.clear();
  const std::string length_mismatch =
      std::string(detail::kMooncakeDecodeKVNotificationFramePrefix) + "2:YQ==";
  EXPECT_FALSE(detail::unframe_mooncake_notification_payload(
      length_mismatch, &decoded, &error));
  EXPECT_TRUE(decoded.empty());
  EXPECT_NE(error.find("length mismatch"), std::string::npos);

  std::deque<TransferMetadata::NotifyDesc> pending = {
      {kMooncakeDecodeKVNotificationName,
       detail::frame_mooncake_notification_payload("valid")},
      {kMooncakeDecodeKVNotificationName, length_mismatch}};
  std::vector<std::string> payloads = {"stale"};
  bool more_available = true;
  error.clear();
  EXPECT_FALSE(detail::drain_mooncake_pending_notifications(
      &pending,
      kMooncakeDecodeKVNotificationName,
      /*max_notifications=*/2,
      &payloads,
      &more_available,
      &error));
  EXPECT_TRUE(payloads.empty());
  EXPECT_FALSE(more_available);
  EXPECT_NE(error.find("length mismatch"), std::string::npos);
}

#if defined(USE_NPU)
TEST(MooncakeTransferEngineTest,
     FramedCarrierPreservesMultiReceiptProtobufPayload) {
  MooncakeDecodeKVNotification notification;
  notification.submission_id = "submission-b38";
  for (uint64_t logical_block_ordinal = 0; logical_block_ordinal < 3;
       ++logical_block_ordinal) {
    for (KVCacheTensorRole::Value cache_role :
         {KVCacheTensorRole::KEY,
          KVCacheTensorRole::INDEX,
          KVCacheTensorRole::INDEX_SCALE}) {
      DecodeKVReceipt receipt;
      receipt.request_id =
          "chatcmpl-17447825819546509822-6MTPgjcwuPrzo4DDYZwuzm";
      receipt.key.source_worker_rank = 0;
      receipt.key.destination_worker_rank = 0;
      receipt.key.layer_id = 0;
      receipt.key.group_id = cache_group_id(BlockType::KV);
      receipt.key.cache_role = cache_role;
      receipt.key.logical_block_ordinal = logical_block_ordinal;
      receipt.key.destination_physical_block_id = 128 + logical_block_ordinal;
      receipt.attempt_epoch = 1788014343;
      receipt.allocation_generation = 1;
      receipt.valid_tokens = logical_block_ordinal == 2 ? 37 : 128;
      receipt.completion_level = DecodeKVCompletionLevel::REMOTE_VISIBLE;
      notification.receipts.emplace_back(std::move(receipt));
    }
  }

  std::string serialized;
  std::string error;
  ASSERT_TRUE(serialize_mooncake_decode_kv_notification(
      notification, &serialized, &error))
      << error;
  ASSERT_TRUE(std::any_of(serialized.begin(), serialized.end(), [](char byte) {
    return static_cast<unsigned char>(byte) >= 0x80;
  }));

  TransferMetadata::NotifyDesc raw_sent{kMooncakeDecodeKVNotificationName,
                                        serialized};
  TransferMetadata::NotifyDesc raw_received;
  ASSERT_TRUE(round_trip_legacy_mooncake_notification(raw_sent, &raw_received));
  EXPECT_NE(raw_received.notify_msg, serialized);
  MooncakeDecodeKVNotification corrupted;
  EXPECT_EQ(deserialize_mooncake_decode_kv_notification(
                raw_received.notify_msg, &corrupted, &error),
            DecodeKVPayloadResult::MALFORMED);

  const std::string framed =
      detail::frame_mooncake_notification_payload(serialized);
  ASSERT_TRUE(std::all_of(framed.begin(), framed.end(), [](char byte) {
    return static_cast<unsigned char>(byte) < 0x80;
  }));
  TransferMetadata::NotifyDesc sent{kMooncakeDecodeKVNotificationName, framed};
  TransferMetadata::NotifyDesc received;
  ASSERT_TRUE(round_trip_legacy_mooncake_notification(sent, &received));
  ASSERT_EQ(received.name, sent.name);

  std::deque<TransferMetadata::NotifyDesc> pending = {std::move(received)};
  std::vector<std::string> payloads;
  bool more_available = false;
  ASSERT_TRUE(detail::drain_mooncake_pending_notifications(
      &pending,
      kMooncakeDecodeKVNotificationName,
      /*max_notifications=*/1,
      &payloads,
      &more_available,
      &error))
      << error;
  ASSERT_EQ(payloads.size(), 1u);
  EXPECT_EQ(payloads.front(), serialized);

  MooncakeDecodeKVNotification parsed;
  ASSERT_EQ(deserialize_mooncake_decode_kv_notification(
                payloads.front(), &parsed, &error),
            DecodeKVPayloadResult::OK)
      << error;
  EXPECT_EQ(parsed.submission_id, notification.submission_id);
  EXPECT_EQ(parsed.receipts.size(), notification.receipts.size());
}
#endif

TEST(KVTransferObservabilityTest, AggregatesBlockCountsByRequestAndGroup) {
  TransferKVInfo first;
  first.request_id = "request-a";

  KVTransferMapping kv_mapping;
  kv_mapping.group_id = cache_group_id(BlockType::KV);
  kv_mapping.local_ids = {1, 2};
  kv_mapping.remote_ids = {11, 12};
  first.mappings.emplace_back(std::move(kv_mapping));

  KVTransferMapping linear_mapping;
  linear_mapping.group_id = cache_group_id(BlockType::LINEAR);
  linear_mapping.local_ids = {3};
  linear_mapping.remote_ids = {13};
  first.mappings.emplace_back(std::move(linear_mapping));

  TransferKVInfo second = first;
  second.mappings.resize(1);
  second.mappings[0].local_ids = {4};
  second.mappings[0].remote_ids = {14};

  TransferKVInfo third = second;
  third.request_id = "request-b";

  std::vector<detail::KVTransferTraceRequest> requests;
  detail::append_kv_transfer_trace_request(&requests, first);
  detail::append_kv_transfer_trace_request(&requests, second);
  detail::append_kv_transfer_trace_request(&requests, third);

  ASSERT_EQ(requests.size(), 2U);
  EXPECT_EQ(requests[0].request_id, "request-a");
  ASSERT_EQ(requests[0].groups.size(), 2U);
  EXPECT_EQ(requests[0].groups[0].group_id, cache_group_id(BlockType::KV));
  EXPECT_EQ(requests[0].groups[0].local_block_count, 3U);
  EXPECT_EQ(requests[0].groups[0].remote_block_count, 3U);
  EXPECT_EQ(requests[0].groups[0].local_block_ids,
            (std::vector<uint64_t>{1, 2, 4}));
  EXPECT_EQ(requests[0].groups[0].remote_block_ids,
            (std::vector<uint64_t>{11, 12, 14}));
  EXPECT_EQ(requests[0].groups[1].group_id, cache_group_id(BlockType::LINEAR));
  EXPECT_EQ(requests[0].groups[1].local_block_count, 1U);
  EXPECT_EQ(requests[0].groups[1].remote_block_count, 1U);
  EXPECT_EQ(requests[0].groups[1].local_block_ids, (std::vector<uint64_t>{3}));
  EXPECT_EQ(requests[0].groups[1].remote_block_ids,
            (std::vector<uint64_t>{13}));

  EXPECT_EQ(requests[1].request_id, "request-b");
  ASSERT_EQ(requests[1].groups.size(), 1U);
  EXPECT_EQ(requests[1].groups[0].group_id, cache_group_id(BlockType::KV));
  EXPECT_EQ(requests[1].groups[0].local_block_count, 1U);
  EXPECT_EQ(requests[1].groups[0].remote_block_count, 1U);
  EXPECT_EQ(requests[1].groups[0].local_block_ids, (std::vector<uint64_t>{4}));
  EXPECT_EQ(requests[1].groups[0].remote_block_ids,
            (std::vector<uint64_t>{14}));
}

TEST(KVTransferObservabilityTest, AggregatesFilteredPartialTailOwnerCounts) {
  TransferKVInfo info;
  info.request_id = "partial-tail";
  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.local_ids = {1, 2};
  mapping.remote_ids = {11, 12, 13};
  info.mappings.emplace_back(std::move(mapping));

  const std::vector<TransferKVInfo> owner_zero =
      filter_kv_split_infos(/*kv_split_rank=*/0, /*kv_split_size=*/2, {info});
  const std::vector<TransferKVInfo> owner_one =
      filter_kv_split_infos(/*kv_split_rank=*/1, /*kv_split_size=*/2, {info});
  ASSERT_EQ(owner_zero.size(), 1U);
  ASSERT_EQ(owner_one.size(), 1U);

  std::vector<detail::KVTransferTraceRequest> owner_zero_trace;
  std::vector<detail::KVTransferTraceRequest> owner_one_trace;
  detail::append_kv_transfer_trace_request(&owner_zero_trace, owner_zero[0]);
  detail::append_kv_transfer_trace_request(&owner_one_trace, owner_one[0]);

  ASSERT_EQ(owner_zero_trace.size(), 1U);
  ASSERT_EQ(owner_zero_trace[0].groups.size(), 1U);
  EXPECT_EQ(owner_zero_trace[0].groups[0].local_block_count, 2U);
  EXPECT_EQ(owner_zero_trace[0].groups[0].remote_block_count, 2U);
  EXPECT_EQ(owner_zero_trace[0].groups[0].local_block_ids,
            (std::vector<uint64_t>{1, 2}));
  EXPECT_EQ(owner_zero_trace[0].groups[0].remote_block_ids,
            (std::vector<uint64_t>{11, 13}));
  ASSERT_EQ(owner_one_trace.size(), 1U);
  ASSERT_EQ(owner_one_trace[0].groups.size(), 1U);
  EXPECT_EQ(owner_one_trace[0].groups[0].local_block_count, 1U);
  EXPECT_EQ(owner_one_trace[0].groups[0].remote_block_count, 1U);
  EXPECT_EQ(owner_one_trace[0].groups[0].local_block_ids,
            (std::vector<uint64_t>{1}));
  EXPECT_EQ(owner_one_trace[0].groups[0].remote_block_ids,
            (std::vector<uint64_t>{12}));
}

class RecordingMooncakeTransferEngine final : public MooncakeTransferEngine {
 public:
  struct MoveCall {
    std::string remote_addr;
    std::vector<BufferTransferMapping> mappings;
    MoveOpcode opcode;
  };

  RecordingMooncakeTransferEngine(uint16_t listen_port,
                                  const torch::Device& device)
      : MooncakeTransferEngine(listen_port, device) {}

  bool register_memory(std::vector<void*> addrs,
                       std::vector<size_t> lens,
                       std::vector<uint64_t> buf_bytes) override {
    registered_addrs.emplace_back(std::move(addrs));
    registered_lens.emplace_back(std::move(lens));
    registered_block_bytes.emplace_back(std::move(buf_bytes));
    return true;
  }

  bool move_memory_groups(const std::string& remote_addr,
                          const std::vector<BufferTransferMapping>& mappings,
                          MoveOpcode opcode) override {
    move_calls.emplace_back(MoveCall{remote_addr, mappings, opcode});
    move_call_count_.fetch_add(/*arg=*/1, std::memory_order_release);
    return move_result;
  }

  bool has_reshard_plan(const std::string& remote_addr) const override {
    return planned_addrs.find(remote_addr) != planned_addrs.end();
  }

  bool has_outgoing_plan(const std::string& remote_addr,
                         CacheNamespace cache_namespace) const override {
    return cache_namespace == CacheNamespace::MAIN &&
           outgoing_addrs.find(remote_addr) != outgoing_addrs.end();
  }

  bool move_result = true;
  std::unordered_set<std::string> planned_addrs;
  std::unordered_set<std::string> outgoing_addrs;
  std::vector<std::vector<void*>> registered_addrs;
  std::vector<std::vector<size_t>> registered_lens;
  std::vector<std::vector<uint64_t>> registered_block_bytes;
  std::vector<MoveCall> move_calls;
  std::atomic<size_t> move_call_count_{0};
};

#if defined(USE_NPU)
bool wait_for_exact_move_call_count(
    const RecordingMooncakeTransferEngine& engine,
    size_t expected_count) {
  const std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + kTransferCompletionTimeout;
  while (engine.move_call_count_.load(std::memory_order_acquire) <
             expected_count &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  return engine.move_call_count_.load(std::memory_order_acquire) ==
         expected_count;
}
#endif

TEST(MooncakeKVCacheTransferDefaultTest,
     NegotiatedPlansFilterDestinationDpGroup) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  engine->planned_addrs = {"addr_1", "addr_3"};
  engine->outgoing_addrs = {"addr_1", "addr_3"};
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  const TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                        /*dst_tp_size=*/4,
                                        /*dst_dp_rank=*/0);
  const ParallelArgs parallel_args = make_args(/*rank=*/0,
                                               /*world_size=*/2,
                                               /*dp_size=*/1);
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged_kv_infos;

  transfer.merge_kv_blocks(merged_kv_infos, {info}, parallel_args);

  ASSERT_EQ(merged_kv_infos.size(), 2U);
  EXPECT_NE(merged_kv_infos.find("101_addr_1"), merged_kv_infos.end());
  EXPECT_NE(merged_kv_infos.find("103_addr_3"), merged_kv_infos.end());
}

TEST(MooncakeKVCacheTransferDefaultTest,
     NegotiatedEmptyOutgoingPlansDoNotUseLegacyFanout) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  engine->planned_addrs = {"addr_0", "addr_1", "addr_2", "addr_3"};
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                  /*dst_tp_size=*/4,
                                  /*dst_dp_rank=*/0);
  info.decode_kv_manifest = "strict-manifest";
  const ParallelArgs parallel_args = make_args(/*rank=*/1,
                                               /*world_size=*/4,
                                               /*dp_size=*/1);
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged_kv_infos;

  transfer.merge_kv_blocks(merged_kv_infos, {info}, parallel_args);

  EXPECT_TRUE(merged_kv_infos.empty());
}

TEST(MooncakeKVCacheTransferDefaultTest,
     MissingNegotiationPreservesLegacyDestinationFanout) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  const TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                        /*dst_tp_size=*/4,
                                        /*dst_dp_rank=*/0);
  const ParallelArgs parallel_args = make_args(/*rank=*/0,
                                               /*world_size=*/2,
                                               /*dp_size=*/1);
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged_kv_infos;

  transfer.merge_kv_blocks(merged_kv_infos, {info}, parallel_args);

  EXPECT_EQ(merged_kv_infos.size(), 4U);
}

TEST(KVCacheTransferBatchMergeTest,
     GenericMergeAllowsRequestScopedSharedPrefixLengths) {
  GenericKVCacheTransferForTest transfer;
  TransferKVInfo first = make_info(/*dst_dp_size=*/1,
                                   /*dst_tp_size=*/1,
                                   /*dst_dp_rank=*/0);
  first.request_id = "short-prefix";
  first.mappings[0].remote_shared_num = 1;
  TransferKVInfo second = first;
  second.request_id = "long-prefix";
  second.mappings[0].local_ids = {13};
  second.mappings[0].remote_ids = {23};
  second.mappings[0].remote_shared_num = 2;
  const ParallelArgs parallel_args = make_args(/*rank=*/0,
                                               /*world_size=*/1,
                                               /*dp_size=*/1);
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged_kv_infos;

  transfer.merge_kv_blocks(merged_kv_infos, {first, second}, parallel_args);

  ASSERT_EQ(merged_kv_infos.size(), 1U);
  const auto& mappings = merged_kv_infos.begin()->second.mappings;
  ASSERT_EQ(mappings.size(), 1U);
  EXPECT_EQ(mappings[0].local_ids, (std::vector<uint64_t>{11, 12, 13}));
  EXPECT_EQ(mappings[0].remote_ids, (std::vector<uint64_t>{21, 22, 23}));
}

TEST(MooncakeKVCacheTransferDefaultTest,
     MergeAllowsRequestScopedSharedPrefixLengths) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  TransferKVInfo first = make_info(/*dst_dp_size=*/1,
                                   /*dst_tp_size=*/1,
                                   /*dst_dp_rank=*/0);
  first.request_id = "short-prefix";
  first.mappings[0].remote_shared_num = 1;
  TransferKVInfo second = first;
  second.request_id = "long-prefix";
  second.mappings[0].local_ids = {13};
  second.mappings[0].remote_ids = {23};
  second.mappings[0].remote_shared_num = 2;
  const ParallelArgs parallel_args = make_args(/*rank=*/0,
                                               /*world_size=*/1,
                                               /*dp_size=*/1);
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged_kv_infos;

  transfer.merge_kv_blocks(merged_kv_infos, {first, second}, parallel_args);

  ASSERT_EQ(merged_kv_infos.size(), 1U);
  const auto& mappings = merged_kv_infos.begin()->second.mappings;
  ASSERT_EQ(mappings.size(), 1U);
  EXPECT_EQ(mappings[0].local_ids, (std::vector<uint64_t>{11, 12, 13}));
  EXPECT_EQ(mappings[0].remote_ids, (std::vector<uint64_t>{21, 22, 23}));
}

#if defined(USE_NPU)
constexpr int32_t kValidatePushCommand = 1;
constexpr int32_t kPreparePullCommand = 2;
constexpr int32_t kStopChildCommand = 3;
constexpr char kPeerCommandFdEnv[] = "XLLM_MOONCAKE_TEST_PEER_COMMAND_FD";
constexpr char kPeerStatusFdEnv[] = "XLLM_MOONCAKE_TEST_PEER_STATUS_FD";
constexpr char kPeerListenPortEnv[] = "XLLM_MOONCAKE_TEST_PEER_LISTEN_PORT";
constexpr char kPeerDeviceIndexEnv[] = "XLLM_MOONCAKE_TEST_PEER_DEVICE_INDEX";
constexpr char kControllerProcessEnv[] =
    "XLLM_MOONCAKE_TEST_CONTROLLER_PROCESS";

bool write_all(int fd, const void* data, size_t size) {
  const char* cursor = static_cast<const char*>(data);
  while (size > 0) {
    const ssize_t written = write(fd, cursor, size);
    if (written < 0 && errno == EINTR) {
      continue;
    }
    if (written <= 0) {
      return false;
    }
    cursor += written;
    size -= static_cast<size_t>(written);
  }
  return true;
}

bool read_all(int fd, void* data, size_t size) {
  char* cursor = static_cast<char*>(data);
  while (size > 0) {
    const ssize_t received = read(fd, cursor, size);
    if (received < 0 && errno == EINTR) {
      continue;
    }
    if (received <= 0) {
      return false;
    }
    cursor += received;
    size -= static_cast<size_t>(received);
  }
  return true;
}

bool write_endpoint(int fd,
                    uint64_t cluster_id,
                    uint16_t listen_port,
                    const std::string& addr) {
  const uint32_t addr_size = static_cast<uint32_t>(addr.size());
  return write_all(fd, &cluster_id, sizeof(cluster_id)) &&
         write_all(fd, &listen_port, sizeof(listen_port)) &&
         write_all(fd, &addr_size, sizeof(addr_size)) &&
         write_all(fd, addr.data(), addr_size);
}

bool read_endpoint(int fd,
                   uint64_t* cluster_id,
                   uint16_t* listen_port,
                   std::string* addr) {
  uint32_t addr_size = 0;
  if (!read_all(fd, cluster_id, sizeof(*cluster_id)) ||
      !read_all(fd, listen_port, sizeof(*listen_port)) ||
      !read_all(fd, &addr_size, sizeof(addr_size)) || addr_size == 0 ||
      addr_size > 1024) {
    return false;
  }
  addr->resize(addr_size);
  return read_all(fd, addr->data(), addr_size);
}

class ChildProcessGuard final {
 public:
  explicit ChildProcessGuard(pid_t pid) : pid_(pid) {}
  ~ChildProcessGuard() {
    if (pid_ > 0) {
      kill(pid_, SIGKILL);
      while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {
      }
    }
  }

  void release() { pid_ = -1; }

 private:
  pid_t pid_;
};

class ScopedSigpipeIgnore final {
 public:
  ScopedSigpipeIgnore() : previous_handler_(signal(SIGPIPE, SIG_IGN)) {}
  ~ScopedSigpipeIgnore() {
    if (previous_handler_ != SIG_ERR) {
      signal(SIGPIPE, previous_handler_);
    }
  }

 private:
  using SignalHandler = void (*)(int);
  SignalHandler previous_handler_;
};

class ScopedEnvironmentVariable final {
 public:
  explicit ScopedEnvironmentVariable(const char* name) : name_(name) {
    const char* value = std::getenv(name);
    if (value != nullptr) {
      original_value_ = value;
    }
  }

  ~ScopedEnvironmentVariable() {
    if (original_value_.has_value()) {
      setenv(name_.c_str(), original_value_->c_str(), /*overwrite=*/1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  bool set(const char* value) {
    return setenv(name_.c_str(), value, /*overwrite=*/1) == 0;
  }

 private:
  std::string name_;
  std::optional<std::string> original_value_;
};

struct NpuMixedTransferCaches {
  torch::Tensor backing;
  torch::Tensor conv;
  torch::Tensor ssm;
  torch::Tensor key;
  torch::Tensor value;
  torch::Tensor index;
  torch::Tensor index_scale;
  std::vector<KVCache> caches;
};

NpuMixedTransferCaches make_npu_mixed_transfer_caches(
    const torch::Device& device) {
  NpuMixedTransferCaches tensors;
  tensors.backing = torch::zeros({6, 2, 1024, 512},
                                 torch::dtype(torch::kBFloat16).device(device));
  tensors.conv = tensors.backing.index({0});
  tensors.ssm = tensors.backing.index({1});
  tensors.key = tensors.backing.index({2});
  tensors.value = tensors.backing.index({3});
  tensors.index = tensors.backing.index({4});
  tensors.index_scale = tensors.backing.index({5});
  tensors.caches.emplace_back(
      LinearAttentionKVCacheTensors{tensors.conv, tensors.ssm});
  tensors.caches.emplace_back(
      IndexedKVCacheTensors{KVCacheTensors{tensors.key, tensors.value},
                            tensors.index,
                            tensors.index_scale});
  return tensors;
}

void fill_mixed_transfer_block(NpuMixedTransferCaches* tensors,
                               int64_t block_id,
                               bool pull_pattern) {
  const double offset = pull_pattern ? 4.0 : 0.0;
  tensors->conv.index({block_id}).fill_(1.25 + offset);
  tensors->ssm.index({block_id}).fill_(-2.5 - offset);
  tensors->key.index({block_id}).fill_(3.5 + offset);
  tensors->value.index({block_id}).fill_(-4.5 - offset);
  tensors->index.index({block_id}).fill_(pull_pattern ? 17 : 42);
  tensors->index_scale.index({block_id}).fill_(pull_pattern ? 0.25 : 0.125);
}

void zero_mixed_transfer_block(NpuMixedTransferCaches* tensors,
                               int64_t block_id) {
  tensors->conv.index({block_id}).zero_();
  tensors->ssm.index({block_id}).zero_();
  tensors->key.index({block_id}).zero_();
  tensors->value.index({block_id}).zero_();
  tensors->index.index({block_id}).zero_();
  tensors->index_scale.index({block_id}).zero_();
}

bool tensor_block_has_value(const torch::Tensor& tensor,
                            int64_t block_id,
                            double value) {
  const torch::Tensor block = tensor.index({block_id}).cpu();
  return torch::equal(block, torch::full_like(block, value));
}

bool mixed_transfer_block_matches(const NpuMixedTransferCaches& tensors,
                                  int64_t block_id,
                                  bool pull_pattern) {
  const double offset = pull_pattern ? 4.0 : 0.0;
  return tensor_block_has_value(tensors.conv, block_id, 1.25 + offset) &&
         tensor_block_has_value(tensors.ssm, block_id, -2.5 - offset) &&
         tensor_block_has_value(tensors.key, block_id, 3.5 + offset) &&
         tensor_block_has_value(tensors.value, block_id, -4.5 - offset) &&
         tensor_block_has_value(
             tensors.index, block_id, pull_pattern ? 17 : 42) &&
         tensor_block_has_value(
             tensors.index_scale, block_id, pull_pattern ? 0.25 : 0.125);
}

int run_npu_round_trip_peer(int command_fd,
                            int status_fd,
                            uint16_t listen_port,
                            int32_t device_index) {
  Device remote_device(device_index);
  remote_device.set_device();
  remote_device.init_device_context();
  const torch::Device remote_torch_device = remote_device.unwrap();
  MooncakeKVCacheTransferDefault remote_transfer(device_index,
                                                 listen_port,
                                                 remote_torch_device,
                                                 /*model_type=*/"test");
  remote_transfer.initialize(device_index);
  NpuMixedTransferCaches remote_caches =
      make_npu_mixed_transfer_caches(remote_torch_device);
  remote_transfer.register_kv_cache(
      remote_caches.caches, KVCacheShape(), torch::kBFloat16);

  const auto& layers = remote_transfer.main_layout_.layers;
  const bool layout_matches =
      layers.size() == 2 && layers[0].size() == 2 && layers[1].size() == 4 &&
      layers[0][0].role == KVCacheTensorRole::CONV &&
      layers[0][1].role == KVCacheTensorRole::SSM &&
      layers[0][0].group_id == cache_group_id(BlockType::LINEAR) &&
      layers[0][1].group_id == cache_group_id(BlockType::LINEAR) &&
      layers[1][0].role == KVCacheTensorRole::KEY &&
      layers[1][1].role == KVCacheTensorRole::VALUE &&
      layers[1][2].role == KVCacheTensorRole::INDEX &&
      layers[1][3].role == KVCacheTensorRole::INDEX_SCALE &&
      layers[1][0].group_id == cache_group_id(BlockType::KV) &&
      layers[1][1].group_id == cache_group_id(BlockType::KV) &&
      layers[1][2].group_id == cache_group_id(BlockType::KV) &&
      layers[1][3].group_id == cache_group_id(BlockType::KV);
  if (!layout_matches) {
    return 10;
  }

  uint64_t remote_cluster_id = 0;
  std::string remote_addr;
  remote_transfer.get_cache_info(remote_cluster_id, remote_addr);
  if (remote_addr.empty() ||
      !write_endpoint(status_fd, remote_cluster_id, listen_port, remote_addr)) {
    return 11;
  }

  while (true) {
    int32_t command = 0;
    if (!read_all(command_fd, &command, sizeof(command))) {
      return 12;
    }

    uint8_t success = 0;
    if (command == kValidatePushCommand) {
      remote_device.set_device();
      success = remote_device.synchronize_default_stream() == 0 &&
                        mixed_transfer_block_matches(remote_caches,
                                                     /*block_id=*/1,
                                                     /*pull_pattern=*/false)
                    ? 1
                    : 0;
    } else if (command == kPreparePullCommand) {
      remote_device.set_device();
      fill_mixed_transfer_block(
          &remote_caches, /*block_id=*/1, /*pull_pattern=*/true);
      success = remote_device.synchronize_default_stream() == 0 ? 1 : 0;
    } else if (command == kStopChildCommand) {
      close(command_fd);
      close(status_fd);
      // The peer is an exec-isolated test process. The transfer and remote
      // session have already been verified and closed by the parent before
      // this command. Bypass third-party process-global teardown, which can
      // terminate on a still-joinable TransferEngine thread.
      _exit(0);
    } else {
      return 13;
    }

    if (!write_all(status_fd, &success, sizeof(success))) {
      return 14;
    }
  }
}
#endif

#if defined(USE_MLU)
KVCacheShape make_indexer_int8_transfer_shape() {
  proto::KVCacheShape proto_shape;
  for (int64_t dim : std::vector<int64_t>{2, 1, 1, 16}) {
    proto_shape.add_key_cache_shape(dim);
    proto_shape.add_value_cache_shape(dim);
  }
  for (int64_t dim : std::vector<int64_t>{2, 96, 1, 8}) {
    proto_shape.add_index_cache_shape(dim);
  }
  for (int64_t dim : std::vector<int64_t>{2, 96, 1}) {
    proto_shape.add_index_cache_scale_shape(dim);
  }
  return KVCacheShape::from_proto(proto_shape);
}

std::vector<KVCache> make_mixed_transfer_caches(const torch::Device& device) {
  std::shared_ptr<KVCacheTensorAllocator> allocator =
      default_kv_tensor_allocator();
  auto make_full_cache = [&allocator, &device]() {
    torch::Tensor key = allocator->allocate(
        KVCacheTensorRole::KEY, {2, 1, 1, 16}, torch::kBFloat16, device);
    torch::Tensor index = allocator->allocate(
        KVCacheTensorRole::INDEX, {2, 96, 1, 8}, torch::kChar, device);
    torch::Tensor index_scale = allocator->allocate(
        KVCacheTensorRole::INDEX_SCALE, {2, 96, 1}, torch::kFloat32, device);
    return KVCache(IndexedKVCacheTensors{
        KVCacheTensors{key, torch::Tensor()}, index, index_scale});
  };
  auto make_shared_cache = [&allocator, &device]() {
    torch::Tensor key = allocator->allocate(
        KVCacheTensorRole::KEY, {2, 1, 1, 16}, torch::kChar, device);
    torch::Tensor key_scale = allocator->allocate(
        KVCacheTensorRole::KEY_SCALE, {2, 1, 1}, torch::kFloat32, device);
    return KVCache(QuantizedKVCacheTensors{
        KVCacheTensors{key, torch::Tensor()}, key_scale, torch::Tensor()});
  };

  std::vector<KVCache> caches;
  caches.reserve(4);
  caches.emplace_back(make_full_cache());
  caches.emplace_back(make_shared_cache());
  caches.emplace_back(make_full_cache());
  caches.emplace_back(make_shared_cache());
  return caches;
}
#endif

}  // namespace

TEST(MooncakeTransferEngineServiceTest, OpenSessionRejectsMissingAddr) {
  MooncakeTransferEngineService service;
  proto::SessionInfo request;
  proto::Status response;
  brpc::Controller cntl;

  service.OpenSession(&cntl, &request, &response, nullptr);

  EXPECT_FALSE(response.ok());
}

TEST(MooncakeTransferEngineServiceTest, CloseSessionRejectsMissingAddr) {
  MooncakeTransferEngineService service;
  proto::SessionInfo request;
  proto::Status response;
  brpc::Controller cntl;

  service.CloseSession(&cntl, &request, &response, nullptr);

  EXPECT_FALSE(response.ok());
}

TEST(MooncakeTransferEngineServiceTest, CloseSessionWithoutHandleReturnsTrue) {
  MooncakeTransferEngineService service;
  proto::SessionInfo request;
  request.set_addr("127.0.0.1:5001");
  proto::Status response;
  brpc::Controller cntl;

  service.CloseSession(&cntl, &request, &response, nullptr);

  EXPECT_TRUE(response.ok());
}

TEST(MooncakeKVCacheTransferDefaultTest,
     PullUsesGroupSpecificMappingsForKvAndLinearBuffers) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  caches.emplace_back(LinearAttentionKVCacheTensors{torch::zeros({1, 3}),
                                                    torch::zeros({1, 5})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  KVTransferMapping kv_mapping;
  kv_mapping.group_id = cache_group_id(BlockType::KV);
  kv_mapping.local_ids = {1, 2};
  kv_mapping.remote_ids = {11, 12};
  KVTransferMapping linear_mapping;
  linear_mapping.group_id = cache_group_id(BlockType::LINEAR);
  linear_mapping.local_ids = {0};
  linear_mapping.remote_ids = {7};

  ASSERT_TRUE(transfer.pull_kv_blocks(
      /*src_cluster_id=*/1, "remote", {kv_mapping, linear_mapping}));
  ASSERT_EQ(engine_observer->move_calls.size(), 1U);
  const RecordingMooncakeTransferEngine::MoveCall& call =
      engine_observer->move_calls[0];
  EXPECT_EQ(call.remote_addr, "remote");
  EXPECT_EQ(call.opcode, MooncakeTransferEngine::MoveOpcode::READ);
  ASSERT_EQ(call.mappings.size(), 4U);
  EXPECT_EQ(call.mappings[0].buf_id, 0);
  EXPECT_EQ(call.mappings[1].buf_id, 1);
  EXPECT_EQ(call.mappings[0].local_ids, kv_mapping.local_ids);
  EXPECT_EQ(call.mappings[1].remote_ids, kv_mapping.remote_ids);
  EXPECT_EQ(call.mappings[2].buf_id, 2);
  EXPECT_EQ(call.mappings[3].buf_id, 3);
  EXPECT_EQ(call.mappings[2].local_ids, linear_mapping.local_ids);
  EXPECT_EQ(call.mappings[3].remote_ids, linear_mapping.remote_ids);
}

TEST(MooncakeKVCacheTransferDefaultTest,
     RegistersCheckpointedSsmAsLogicalSequenceSlots) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"qwen3_5",
                                          std::move(engine));
  transfer.addr_ = "local";
  ModelArgs model_args;
  model_args.model_type("qwen3_5")
      .n_layers(1)
      .n_heads(4)
      .linear_num_key_heads(2)
      .linear_num_value_heads(2)
      .linear_key_head_dim(1)
      .linear_value_head_dim(1);
  transfer.configure_cache_layout(make_args(/*rank=*/0,
                                            /*world_size=*/1,
                                            /*dp_size=*/1),
                                  model_args,
                                  /*block_token_capacity=*/0,
                                  /*is_spec_draft=*/false);

  proto::KVCacheShape proto_shape;
  for (int64_t dim : {2, 1, 6}) {
    proto_shape.add_conv_cache_shape(dim);
  }
  for (int64_t dim : {6, 2, 1, 1}) {
    proto_shape.add_ssm_cache_shape(dim);
  }
  const KVCacheShape shape = KVCacheShape::from_proto(proto_shape);
  std::vector<KVCache> caches;
  caches.emplace_back(LinearAttentionKVCacheTensors{
      torch::zeros({2, 1, 6}), torch::zeros({6, 2, 1, 1})});

  transfer.register_kv_cache(caches, shape, torch::kFloat32);

  ASSERT_EQ(engine_observer->registered_block_bytes.size(), 1U);
  ASSERT_EQ(engine_observer->registered_block_bytes[0].size(), 2U);
  EXPECT_EQ(engine_observer->registered_block_bytes[0][0], 24U);
  EXPECT_EQ(engine_observer->registered_block_bytes[0][1], 24U);
  ASSERT_EQ(transfer.local_cache_layout_.tensors.size(), 2U);
  const auto ssm_it = std::find_if(
      transfer.local_cache_layout_.tensors.begin(),
      transfer.local_cache_layout_.tensors.end(),
      [](const CacheTensorManifest& tensor) {
        return tensor.role == static_cast<int32_t>(KVCacheTensorRole::SSM);
      });
  ASSERT_NE(ssm_it, transfer.local_cache_layout_.tensors.end());
  EXPECT_EQ(ssm_it->resource_count, 2U);
  EXPECT_EQ(ssm_it->physical_rows_per_resource, 3U);
  EXPECT_EQ(ssm_it->resource_stride_bytes, 24U);
  ASSERT_EQ(ssm_it->shard.spans.size(), 2U);
  EXPECT_EQ(ssm_it->shard.spans[0].repeat_count, 3U);
}

TEST(MooncakeKVCacheTransferDefaultTest,
     RegistersTp1SpecDraftBesideTp2MainCache) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"qwen3_5",
                                          std::move(engine));
  transfer.addr_ = "local-draft-body-tp1";
  ModelArgs model_args;
  model_args.model_type("qwen3_5")
      .n_layers(1)
      .n_heads(4)
      .n_kv_heads(4)
      .head_dim(1);
  KVCacheCapacity cache_capacity;
  cache_capacity.n_blocks(4).block_size(3);

  transfer.configure_cache_layout(make_args(/*rank=*/0,
                                            /*world_size=*/2,
                                            /*dp_size=*/1),
                                  model_args,
                                  /*block_token_capacity=*/3,
                                  /*is_spec_draft=*/false);
  const KVCacheShape main_shape(cache_capacity, model_args, /*world_size=*/2);
  std::vector<KVCache> main_caches;
  main_caches.emplace_back(
      KVCacheTensors{torch::zeros(main_shape.key_cache_shape()),
                     torch::zeros(main_shape.value_cache_shape())});
  transfer.register_kv_cache(main_caches, main_shape, torch::kFloat32);

  transfer.configure_cache_layout(make_args(/*rank=*/0,
                                            /*world_size=*/1,
                                            /*dp_size=*/1),
                                  model_args,
                                  /*block_token_capacity=*/3,
                                  /*is_spec_draft=*/true);
  const KVCacheShape draft_shape(cache_capacity, model_args, /*world_size=*/1);
  std::vector<KVCache> draft_caches;
  draft_caches.emplace_back(
      KVCacheTensors{torch::zeros(draft_shape.key_cache_shape()),
                     torch::zeros(draft_shape.value_cache_shape())});
  transfer.register_kv_cache_spec(draft_caches, draft_shape, torch::kFloat32);

  EXPECT_EQ(transfer.local_cache_layout_.coordinates.tp_size, 2);
  ASSERT_EQ(transfer.local_cache_layout_.tensors.size(), 4U);
  const CacheTensorManifest& spec_key = transfer.local_cache_layout_.tensors[2];
  EXPECT_EQ(spec_key.cache_namespace, CacheNamespace::SPEC_DRAFT);
  ASSERT_EQ(spec_key.shard.spans.size(), 4U);
  EXPECT_TRUE(std::all_of(
      spec_key.shard.spans.begin(),
      spec_key.shard.spans.end(),
      [](const LogicalSpan& span) { return span.owner_tp_rank == 0; }));
}

TEST(MooncakeKVCacheTransferDefaultTest,
     GroupedPullUsesSwaAndCompressedMappings) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"deepseek_v4",
                                          std::move(engine));
  DeepSeekV4KVCacheTensors tensors;
  tensors.key_cache = torch::zeros({4, 2});
  tensors.swa_cache = torch::zeros({4, 3});
  tensors.compressed_block_type = BlockType::C4;
  std::vector<KVCache> caches;
  caches.emplace_back(tensors);
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  KVTransferMapping swa_mapping;
  swa_mapping.group_id = cache_group_id(BlockType::SWA);
  swa_mapping.local_ids = {1};
  swa_mapping.remote_ids = {11};
  KVTransferMapping c4_mapping;
  c4_mapping.group_id = cache_group_id(BlockType::C4);
  c4_mapping.local_ids = {2};
  c4_mapping.remote_ids = {12};

  ASSERT_TRUE(transfer.pull_kv_blocks(
      /*src_cluster_id=*/1, "remote", {c4_mapping, swa_mapping}));
  ASSERT_EQ(engine_observer->move_calls.size(), 1U);
  const std::vector<MooncakeTransferEngine::BufferTransferMapping>& mappings =
      engine_observer->move_calls[0].mappings;
  ASSERT_EQ(mappings.size(), 2U);
  EXPECT_EQ(mappings[0].buf_id, 0);
  EXPECT_EQ(mappings[0].local_ids, swa_mapping.local_ids);
  EXPECT_EQ(mappings[1].buf_id, 1);
  EXPECT_EQ(mappings[1].remote_ids, c4_mapping.remote_ids);
}

TEST(MooncakeKVCacheTransferDefaultTest,
     PullCoversMainAndSpecLayoutsInOneRead) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> main_caches;
  main_caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  std::vector<KVCache> spec_caches;
  spec_caches.emplace_back(LinearAttentionKVCacheTensors{torch::zeros({1, 3}),
                                                         torch::zeros({1, 5})});
  transfer.register_kv_cache(main_caches, KVCacheShape(), torch::kFloat32);
  transfer.register_kv_cache_spec(spec_caches, KVCacheShape(), torch::kFloat32);

  KVTransferMapping kv_mapping;
  kv_mapping.group_id = cache_group_id(BlockType::KV);
  kv_mapping.local_ids = {1};
  kv_mapping.remote_ids = {11};
  KVTransferMapping linear_mapping;
  linear_mapping.group_id = cache_group_id(BlockType::LINEAR);
  linear_mapping.local_ids = {0};
  linear_mapping.remote_ids = {7};

  ASSERT_TRUE(transfer.pull_kv_blocks(
      /*src_cluster_id=*/1, "remote", {kv_mapping, linear_mapping}));
  ASSERT_EQ(engine_observer->move_calls.size(), 1U);
  const std::vector<MooncakeTransferEngine::BufferTransferMapping>& mappings =
      engine_observer->move_calls[0].mappings;
  ASSERT_EQ(mappings.size(), 4U);
  EXPECT_EQ(mappings[0].buf_id, 0);
  EXPECT_EQ(mappings[1].buf_id, 1);
  EXPECT_EQ(mappings[2].buf_id, 2);
  EXPECT_EQ(mappings[3].buf_id, 3);
}

TEST(MooncakeKVCacheTransferDefaultTest, PullRejectsInvalidMappings) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  KVTransferMapping mismatched;
  mismatched.group_id = cache_group_id(BlockType::KV);
  mismatched.local_ids = {1, 2};
  mismatched.remote_ids = {11};
  EXPECT_FALSE(
      transfer.pull_kv_blocks(/*src_cluster_id=*/1, "remote", {mismatched}));

  KVTransferMapping duplicate;
  duplicate.group_id = cache_group_id(BlockType::KV);
  duplicate.local_ids = {1};
  duplicate.remote_ids = {11};
  EXPECT_FALSE(transfer.pull_kv_blocks(
      /*src_cluster_id=*/1, "remote", {duplicate, duplicate}));

  KVTransferMapping wrong_group;
  wrong_group.group_id = cache_group_id(BlockType::LINEAR);
  wrong_group.local_ids = {0};
  wrong_group.remote_ids = {7};
  EXPECT_FALSE(
      transfer.pull_kv_blocks(/*src_cluster_id=*/1, "remote", {wrong_group}));
  EXPECT_TRUE(engine_observer->move_calls.empty());
}

#if defined(USE_NPU)
TEST(MooncakeKVCacheTransferDefaultTest,
     PushRejectsDuplicateMappingsBeforeMerge) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                  /*dst_tp_size=*/1,
                                  /*dst_dp_rank=*/0);
  info.mappings.emplace_back(info.mappings[0]);
  const ParallelArgs parallel_args = make_args(/*rank=*/0,
                                               /*world_size=*/1,
                                               /*dp_size=*/1);
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer;

  folly::SemiFuture<bool> future = transfer.push_kv_blocks_async(
      {info}, parallel_args, synchronizer, /*is_spec_draft=*/false);
  EXPECT_FALSE(std::move(future).get());
  EXPECT_TRUE(engine_observer->move_calls.empty());
}

TEST(MooncakeKVCacheTransferDefaultTest,
     PushRejectsPartialStrictReadinessIdentityBeforeMerge) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  const ParallelArgs parallel_args = make_args(/*rank=*/0,
                                               /*world_size=*/1,
                                               /*dp_size=*/1);
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer;

  TransferKVInfo attempt_only = make_info(/*dst_dp_size=*/1,
                                          /*dst_tp_size=*/1,
                                          /*dst_dp_rank=*/0);
  attempt_only.attempt_epoch = 7;
  folly::SemiFuture<bool> attempt_future = transfer.push_kv_blocks_async(
      {attempt_only}, parallel_args, synchronizer, /*is_spec_draft=*/false);
  EXPECT_FALSE(std::move(attempt_future).get());

  TransferKVInfo allocation_only = make_info(/*dst_dp_size=*/1,
                                             /*dst_tp_size=*/1,
                                             /*dst_dp_rank=*/0);
  allocation_only.allocation_generation = 11;
  folly::SemiFuture<bool> allocation_future = transfer.push_kv_blocks_async(
      {allocation_only}, parallel_args, synchronizer, /*is_spec_draft=*/false);
  EXPECT_FALSE(std::move(allocation_future).get());
  EXPECT_TRUE(engine_observer->move_calls.empty());
}

TEST(MooncakeKVCacheTransferDefaultTest,
     KvSplitFilterAcceptsCompleteAndPartialFinalCoverage) {
  TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                  /*dst_tp_size=*/1,
                                  /*dst_dp_rank=*/0);
  info.mappings[0].remote_ids = {21, 22, 23};

  std::vector<TransferKVInfo> rank_zero_infos = filter_kv_split_infos(
      /*kv_split_rank=*/0, /*kv_split_size=*/2, {info});
  ASSERT_EQ(rank_zero_infos.size(), 1U);
  ASSERT_EQ(rank_zero_infos[0].mappings.size(), 1U);
  EXPECT_EQ(rank_zero_infos[0].mappings[0].local_ids,
            (std::vector<uint64_t>{11, 12}));
  EXPECT_EQ(rank_zero_infos[0].mappings[0].remote_ids,
            (std::vector<uint64_t>{21, 23}));

  std::vector<TransferKVInfo> rank_one_infos = filter_kv_split_infos(
      /*kv_split_rank=*/1, /*kv_split_size=*/2, {info});
  ASSERT_EQ(rank_one_infos.size(), 1U);
  ASSERT_EQ(rank_one_infos[0].mappings.size(), 1U);
  EXPECT_EQ(rank_one_infos[0].mappings[0].local_ids,
            (std::vector<uint64_t>{11}));
  EXPECT_EQ(rank_one_infos[0].mappings[0].remote_ids,
            (std::vector<uint64_t>{22}));
}

TEST(MooncakeKVCacheTransferDefaultTest,
     KvSplitFilterUsesOrdinalsForPartialFirstAndLastBlocks) {
  TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                  /*dst_tp_size=*/1,
                                  /*dst_dp_rank=*/0);
  info.mappings[0].local_ids = {31, 32, 33};
  info.mappings[0].remote_ids = {103, 104, 105, 106};
  info.mappings[0].logical_block_ordinals = {3, 4, 5, 6};
  info.mappings[0].valid_tokens = {16, 16, 16, 5};
  info.mappings[0].receipt_remote_ids = {103, 105, 106};
  info.mappings[0].remote_shared_num = 3;

  const std::vector<TransferKVInfo> rank_zero_infos = filter_kv_split_infos(
      /*kv_split_rank=*/0, /*kv_split_size=*/2, {info});
  ASSERT_EQ(rank_zero_infos.size(), 1U);
  ASSERT_EQ(rank_zero_infos[0].mappings.size(), 1U);
  EXPECT_EQ(rank_zero_infos[0].mappings[0].local_ids,
            (std::vector<uint64_t>{32, 33}));
  EXPECT_EQ(rank_zero_infos[0].mappings[0].remote_ids,
            (std::vector<uint64_t>{104, 106}));
  EXPECT_EQ(rank_zero_infos[0].mappings[0].logical_block_ordinals,
            (std::vector<uint64_t>{4, 6}));
  EXPECT_EQ(rank_zero_infos[0].mappings[0].valid_tokens,
            (std::vector<uint32_t>{16, 5}));
  EXPECT_EQ(rank_zero_infos[0].mappings[0].receipt_remote_ids,
            (std::vector<uint64_t>{106}));

  const std::vector<TransferKVInfo> rank_one_infos = filter_kv_split_infos(
      /*kv_split_rank=*/1, /*kv_split_size=*/2, {info});
  ASSERT_EQ(rank_one_infos.size(), 1U);
  ASSERT_EQ(rank_one_infos[0].mappings.size(), 1U);
  EXPECT_EQ(rank_one_infos[0].mappings[0].local_ids,
            (std::vector<uint64_t>{31, 32}));
  EXPECT_EQ(rank_one_infos[0].mappings[0].remote_ids,
            (std::vector<uint64_t>{103, 105}));
  EXPECT_EQ(rank_one_infos[0].mappings[0].logical_block_ordinals,
            (std::vector<uint64_t>{3, 5}));
  EXPECT_EQ(rank_one_infos[0].mappings[0].valid_tokens,
            (std::vector<uint32_t>{16, 16}));
  EXPECT_EQ(rank_one_infos[0].mappings[0].receipt_remote_ids,
            (std::vector<uint64_t>{103, 105}));
}

TEST(MooncakeKVCacheTransferDefaultTest,
     KvSplitFilterSupportsFourOwnersWithPartialFirstAndLastBlocks) {
  TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                  /*dst_tp_size=*/1,
                                  /*dst_dp_rank=*/0);
  info.mappings[0].local_ids = {31, 32, 33};
  info.mappings[0].remote_ids = {103, 104, 105, 106, 107, 108, 109};
  info.mappings[0].logical_block_ordinals = {3, 4, 5, 6, 7, 8, 9};
  info.mappings[0].valid_tokens = {32, 32, 32, 32, 32, 32, 5};
  info.mappings[0].receipt_remote_ids = {103, 105, 108, 109};
  info.mappings[0].remote_shared_num = 3;

  struct ExpectedOwnerMapping {
    std::vector<uint64_t> local_ids;
    std::vector<uint64_t> remote_ids;
    std::vector<uint64_t> logical_block_ordinals;
    std::vector<uint32_t> valid_tokens;
    std::vector<uint64_t> receipt_remote_ids;
  };
  const std::vector<ExpectedOwnerMapping> expected = {
      {{32, 33}, {104, 108}, {4, 8}, {32, 32}, {108}},
      {{32, 33}, {105, 109}, {5, 9}, {32, 5}, {105, 109}},
      {{32}, {106}, {6}, {32}, {}},
      {{31, 32}, {103, 107}, {3, 7}, {32, 32}, {103}},
  };

  for (int32_t owner_rank = 0; owner_rank < 4; ++owner_rank) {
    SCOPED_TRACE(owner_rank);
    const std::vector<TransferKVInfo> owner_infos =
        filter_kv_split_infos(owner_rank, /*kv_split_size=*/4, {info});
    ASSERT_EQ(owner_infos.size(), 1U);
    ASSERT_EQ(owner_infos[0].mappings.size(), 1U);
    const KVTransferMapping& mapping = owner_infos[0].mappings[0];
    const ExpectedOwnerMapping& owner_expected =
        expected[static_cast<size_t>(owner_rank)];
    EXPECT_EQ(mapping.local_ids, owner_expected.local_ids);
    EXPECT_EQ(mapping.remote_ids, owner_expected.remote_ids);
    EXPECT_EQ(mapping.logical_block_ordinals,
              owner_expected.logical_block_ordinals);
    EXPECT_EQ(mapping.valid_tokens, owner_expected.valid_tokens);
    EXPECT_EQ(mapping.receipt_remote_ids, owner_expected.receipt_remote_ids);
  }
}

TEST(MooncakeKVCacheTransferDefaultTest,
     KvSplitValidationAcceptsPartialEndsAndRejectsOrdinalDrift) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  ParallelArgs parallel_args(/*rank=*/1,
                             /*world_size=*/4,
                             /*dp_size=*/1,
                             /*cp_size=*/4,
                             /*process_group=*/nullptr,
                             /*ep_size=*/1);
  parallel_args.kv_split_size(2);
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer;

  TransferKVInfo partial_ends = make_info(/*dst_dp_size=*/1,
                                          /*dst_tp_size=*/1,
                                          /*dst_dp_rank=*/0);
  partial_ends.mappings[0].local_ids = {31, 32, 33};
  partial_ends.mappings[0].remote_ids = {103, 104, 105, 106};
  partial_ends.mappings[0].logical_block_ordinals = {3, 4, 5, 6};
  partial_ends.mappings[0].valid_tokens = {16, 16, 16, 5};
  partial_ends.mappings[0].remote_shared_num = 3;
  folly::SemiFuture<bool> accepted = transfer.push_kv_blocks_async(
      {partial_ends}, parallel_args, synchronizer, /*is_spec_draft=*/false);
  EXPECT_TRUE(std::move(accepted).get());

  const auto expect_rejected = [&](std::vector<uint64_t> ordinals,
                                   std::vector<uint32_t> valid_tokens,
                                   std::vector<uint64_t> local_ids) {
    TransferKVInfo malformed = partial_ends;
    malformed.mappings[0].logical_block_ordinals = std::move(ordinals);
    malformed.mappings[0].valid_tokens = std::move(valid_tokens);
    malformed.mappings[0].local_ids = std::move(local_ids);
    folly::SemiFuture<bool> rejected = transfer.push_kv_blocks_async(
        {malformed}, parallel_args, synchronizer, /*is_spec_draft=*/false);
    EXPECT_FALSE(std::move(rejected).get());
  };

  expect_rejected(/*ordinals=*/{3, 4, 5},
                  /*valid_tokens=*/{16, 16, 16, 5},
                  /*local_ids=*/{31, 32});
  expect_rejected(/*ordinals=*/{3, 3, 4, 5},
                  /*valid_tokens=*/{16, 16, 16, 5},
                  /*local_ids=*/{31, 32});
  expect_rejected(/*ordinals=*/{3, 5, 6, 7},
                  /*valid_tokens=*/{16, 16, 16, 5},
                  /*local_ids=*/{31, 32, 33});
  expect_rejected(/*ordinals=*/{4, 3, 5, 6},
                  /*valid_tokens=*/{16, 16, 16, 5},
                  /*local_ids=*/{31, 32});
  EXPECT_TRUE(engine_observer->move_calls.empty());
}

TEST(MooncakeKVCacheTransferDefaultTest,
     KvSplitFilterRemapsGroupedAttentionCaches) {
  TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                  /*dst_tp_size=*/1,
                                  /*dst_dp_rank=*/0);
  info.mappings[0].group_id = cache_group_id(BlockType::C4);
  info.mappings[0].remote_ids = {21, 22, 23, 24};
  KVTransferMapping linear_mapping;
  linear_mapping.group_id = cache_group_id(BlockType::LINEAR);
  linear_mapping.local_ids = {31};
  linear_mapping.remote_ids = {41};
  info.mappings.emplace_back(std::move(linear_mapping));

  std::vector<TransferKVInfo> rank_one_infos = filter_kv_split_infos(
      /*kv_split_rank=*/1, /*kv_split_size=*/2, {info});

  ASSERT_EQ(rank_one_infos.size(), 1U);
  ASSERT_EQ(rank_one_infos[0].mappings.size(), 2U);
  EXPECT_EQ(rank_one_infos[0].mappings[0].local_ids,
            (std::vector<uint64_t>{11, 12}));
  EXPECT_EQ(rank_one_infos[0].mappings[0].remote_ids,
            (std::vector<uint64_t>{22, 24}));
  EXPECT_EQ(rank_one_infos[0].mappings[1].local_ids,
            (std::vector<uint64_t>{31}));
  EXPECT_EQ(rank_one_infos[0].mappings[1].remote_ids,
            (std::vector<uint64_t>{41}));
}

TEST(MooncakeKVCacheTransferDefaultTest,
     PushRejectsIncompleteKvSplitCoverageBeforeFilter) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                  /*dst_tp_size=*/1,
                                  /*dst_dp_rank=*/0);
  info.mappings[0].remote_ids = {21, 22};
  ParallelArgs parallel_args = make_args(/*rank=*/0,
                                         /*world_size=*/2,
                                         /*dp_size=*/1);
  parallel_args.kv_split_size(2);
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer;

  folly::SemiFuture<bool> future = transfer.push_kv_blocks_async(
      {info}, parallel_args, synchronizer, /*is_spec_draft=*/false);
  EXPECT_FALSE(std::move(future).get());
  EXPECT_TRUE(engine_observer->move_calls.empty());
}

TEST(MooncakeKVCacheTransferDefaultTest,
     PushRejectsDuplicateKvSplitDestinationBeforeFilter) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                  /*dst_tp_size=*/1,
                                  /*dst_dp_rank=*/0);
  info.mappings[0].remote_ids = {21, 21, 23};
  ParallelArgs parallel_args = make_args(/*rank=*/0,
                                         /*world_size=*/2,
                                         /*dp_size=*/1);
  parallel_args.kv_split_size(2);
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer;

  folly::SemiFuture<bool> future = transfer.push_kv_blocks_async(
      {info}, parallel_args, synchronizer, /*is_spec_draft=*/false);
  EXPECT_FALSE(std::move(future).get());
  EXPECT_TRUE(engine_observer->move_calls.empty());
}

TEST(MooncakeKVCacheTransferDefaultTest,
     PushRejectsExcessKvSplitCoverageBeforeFilter) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                  /*dst_tp_size=*/1,
                                  /*dst_dp_rank=*/0);
  info.mappings[0].remote_ids = {21, 22, 23, 24, 25};
  ParallelArgs parallel_args = make_args(/*rank=*/0,
                                         /*world_size=*/2,
                                         /*dp_size=*/1);
  parallel_args.kv_split_size(2);
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer;

  folly::SemiFuture<bool> future = transfer.push_kv_blocks_async(
      {info}, parallel_args, synchronizer, /*is_spec_draft=*/false);
  EXPECT_FALSE(std::move(future).get());
  EXPECT_TRUE(engine_observer->move_calls.empty());
}

TEST(MooncakeKVCacheTransferDefaultTest,
     DISABLED_NpuLinearIndexerScaleRoundTripPeerProcess) {
  const char* command_fd_env = std::getenv(kPeerCommandFdEnv);
  const char* status_fd_env = std::getenv(kPeerStatusFdEnv);
  const char* listen_port_env = std::getenv(kPeerListenPortEnv);
  const char* device_index_env = std::getenv(kPeerDeviceIndexEnv);
  ASSERT_NE(command_fd_env, nullptr);
  ASSERT_NE(status_fd_env, nullptr);
  ASSERT_NE(listen_port_env, nullptr);
  ASSERT_NE(device_index_env, nullptr);

  const int command_fd = std::atoi(command_fd_env);
  const int status_fd = std::atoi(status_fd_env);
  const int listen_port = std::atoi(listen_port_env);
  const int device_index = std::atoi(device_index_env);
  ASSERT_GE(command_fd, 0);
  ASSERT_GE(status_fd, 0);
  ASSERT_GT(listen_port, 0);
  ASSERT_LE(listen_port, UINT16_MAX);
  ASSERT_GE(device_index, 0);
  EXPECT_EQ(run_npu_round_trip_peer(command_fd,
                                    status_fd,
                                    static_cast<uint16_t>(listen_port),
                                    device_index),
            0);
}

TEST(MooncakeKVCacheTransferDefaultTest,
     NpuLinearIndexerScalePushAndPullRoundTrip) {
  if (std::getenv(kControllerProcessEnv) == nullptr) {
    ScopedEnvironmentVariable controller_process(kControllerProcessEnv);
    ASSERT_TRUE(controller_process.set("1"));

    const pid_t controller_pid = fork();
    ASSERT_GE(controller_pid, 0);
    if (controller_pid == 0) {
      execl("/proc/self/exe",
            "mooncake_transfer_engine_test",
            "--gtest_filter=MooncakeKVCacheTransferDefaultTest."
            "NpuLinearIndexerScalePushAndPullRoundTrip",
            "--gtest_color=no",
            static_cast<char*>(nullptr));
      _exit(127);
    }

    ChildProcessGuard controller_guard(controller_pid);
    int controller_status = 0;
    pid_t waited_pid = -1;
    do {
      waited_pid = waitpid(controller_pid, &controller_status, 0);
    } while (waited_pid < 0 && errno == EINTR);
    if (waited_pid == controller_pid) {
      controller_guard.release();
    }
    ASSERT_EQ(waited_pid, controller_pid);
    ASSERT_TRUE(WIFEXITED(controller_status));
    EXPECT_EQ(WEXITSTATUS(controller_status), 0);
    return;
  }

  const int32_t device_count = Platform::device_count();
  if (device_count < 2) {
    GTEST_SKIP() << "Two NPU devices are required for Mooncake memory "
                    "transfer.";
  }
  const int32_t remote_device_index = device_count > 4 ? 4 : 1;

  const int32_t local_listen_port = net::get_local_free_port();
  int32_t remote_listen_port = net::get_local_free_port();
  while (remote_listen_port == local_listen_port) {
    remote_listen_port = net::get_local_free_port();
  }
  ASSERT_GT(local_listen_port, 0);
  ASSERT_GT(remote_listen_port, 0);

  int parent_to_child[2];
  int child_to_parent[2];
  ASSERT_EQ(pipe(parent_to_child), 0);
  ASSERT_EQ(pipe(child_to_parent), 0);
  ScopedSigpipeIgnore sigpipe_guard;
  ScopedEnvironmentVariable hccl_base_port("HCCL_IF_BASE_PORT");
  ASSERT_TRUE(hccl_base_port.set("35439"));

  ASSERT_EQ(setenv(kPeerCommandFdEnv,
                   std::to_string(parent_to_child[0]).c_str(),
                   /*overwrite=*/1),
            0);
  ASSERT_EQ(setenv(kPeerStatusFdEnv,
                   std::to_string(child_to_parent[1]).c_str(),
                   /*overwrite=*/1),
            0);
  ASSERT_EQ(setenv(kPeerListenPortEnv,
                   std::to_string(remote_listen_port).c_str(),
                   /*overwrite=*/1),
            0);
  ASSERT_EQ(setenv(kPeerDeviceIndexEnv,
                   std::to_string(remote_device_index).c_str(),
                   /*overwrite=*/1),
            0);

  const pid_t child_pid = fork();
  ASSERT_GE(child_pid, 0);
  if (child_pid == 0) {
    close(parent_to_child[1]);
    close(child_to_parent[0]);
    execl("/proc/self/exe",
          "mooncake_transfer_engine_test",
          "--gtest_filter=MooncakeKVCacheTransferDefaultTest."
          "DISABLED_NpuLinearIndexerScaleRoundTripPeerProcess",
          "--gtest_also_run_disabled_tests",
          "--gtest_color=no",
          static_cast<char*>(nullptr));
    _exit(127);
  }

  ASSERT_TRUE(hccl_base_port.set("34439"));

  unsetenv(kPeerCommandFdEnv);
  unsetenv(kPeerStatusFdEnv);
  unsetenv(kPeerListenPortEnv);
  unsetenv(kPeerDeviceIndexEnv);

  close(parent_to_child[0]);
  close(child_to_parent[1]);
  ChildProcessGuard child_guard(child_pid);

  Device local_device(/*device_id=*/0);
  local_device.set_device();
  local_device.init_device_context();
  const torch::Device local_torch_device = local_device.unwrap();
  MooncakeKVCacheTransferDefault local_transfer(
      /*device_id=*/0,
      static_cast<uint16_t>(local_listen_port),
      local_torch_device,
      /*model_type=*/"test");
  local_transfer.initialize(/*device_id=*/0);
  NpuMixedTransferCaches local_caches =
      make_npu_mixed_transfer_caches(local_torch_device);
  local_transfer.register_kv_cache(
      local_caches.caches, KVCacheShape(), torch::kBFloat16);

  ASSERT_EQ(local_transfer.main_layout_.layers.size(), 2U);
  ASSERT_EQ(local_transfer.main_layout_.layers[0].size(), 2U);
  ASSERT_EQ(local_transfer.main_layout_.layers[1].size(), 4U);
  EXPECT_EQ(local_transfer.main_layout_.layers[0][0].role,
            KVCacheTensorRole::CONV);
  EXPECT_EQ(local_transfer.main_layout_.layers[0][1].role,
            KVCacheTensorRole::SSM);
  EXPECT_EQ(local_transfer.main_layout_.layers[1][0].role,
            KVCacheTensorRole::KEY);
  EXPECT_EQ(local_transfer.main_layout_.layers[1][1].role,
            KVCacheTensorRole::VALUE);
  EXPECT_EQ(local_transfer.main_layout_.layers[1][2].role,
            KVCacheTensorRole::INDEX);
  EXPECT_EQ(local_transfer.main_layout_.layers[1][3].role,
            KVCacheTensorRole::INDEX_SCALE);
  for (const auto& buffer : local_transfer.main_layout_.layers[0]) {
    EXPECT_EQ(buffer.group_id, cache_group_id(BlockType::LINEAR));
  }
  for (const auto& buffer : local_transfer.main_layout_.layers[1]) {
    EXPECT_EQ(buffer.group_id, cache_group_id(BlockType::KV));
  }

  uint64_t local_cluster_id = 0;
  std::string local_addr;
  local_transfer.get_cache_info(local_cluster_id, local_addr);
  ASSERT_FALSE(local_addr.empty());

  uint64_t remote_cluster_id = 0;
  uint16_t received_remote_port = 0;
  std::string remote_addr;
  ASSERT_TRUE(read_endpoint(child_to_parent[0],
                            &remote_cluster_id,
                            &received_remote_port,
                            &remote_addr));
  ASSERT_EQ(received_remote_port, static_cast<uint16_t>(remote_listen_port));
  ASSERT_FALSE(remote_addr.empty());
  ASSERT_TRUE(local_transfer.mooncake_te_->open_session(remote_cluster_id,
                                                        remote_addr));

  KVTransferMapping linear_mapping;
  linear_mapping.group_id = cache_group_id(BlockType::LINEAR);
  linear_mapping.local_ids = {0};
  linear_mapping.remote_ids = {1};
  KVTransferMapping kv_mapping;
  kv_mapping.group_id = cache_group_id(BlockType::KV);
  kv_mapping.local_ids = {0};
  kv_mapping.remote_ids = {1};

  local_device.set_device();
  fill_mixed_transfer_block(
      &local_caches, /*block_id=*/0, /*pull_pattern=*/false);
  ASSERT_EQ(local_device.synchronize_default_stream(), 0);

  KVCacheTransfer::KVCacheInfo info;
  info.dst_cluster_id = remote_cluster_id;
  info.dst_addr = remote_addr;
  info.mappings = {linear_mapping, kv_mapping};
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged_infos;
  merged_infos.emplace(remote_addr, std::move(info));
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer =
      std::make_shared<KVPushSynchronizerImpl>(/*num_layers=*/2);
  ASSERT_TRUE(synchronizer->record_event(/*layer_index=*/0,
                                         /*device_index=*/0));
  ASSERT_TRUE(synchronizer->record_event(/*layer_index=*/1,
                                         /*device_index=*/0));
  ASSERT_TRUE(local_transfer.push_kv_blocks(merged_infos,
                                            synchronizer,
                                            /*is_spec_draft=*/false,
                                            /*kv_split_rank=*/0,
                                            /*kv_split_size=*/1));

  int32_t command = kValidatePushCommand;
  ASSERT_TRUE(write_all(parent_to_child[1], &command, sizeof(command)));
  uint8_t child_success = 0;
  ASSERT_TRUE(
      read_all(child_to_parent[0], &child_success, sizeof(child_success)));
  ASSERT_EQ(child_success, 1);

  command = kPreparePullCommand;
  ASSERT_TRUE(write_all(parent_to_child[1], &command, sizeof(command)));
  ASSERT_TRUE(
      read_all(child_to_parent[0], &child_success, sizeof(child_success)));
  ASSERT_EQ(child_success, 1);

  local_device.set_device();
  zero_mixed_transfer_block(&local_caches, /*block_id=*/0);
  ASSERT_EQ(local_device.synchronize_default_stream(), 0);
  ASSERT_TRUE(local_transfer.pull_kv_blocks(
      remote_cluster_id, remote_addr, {linear_mapping, kv_mapping}));
  ASSERT_EQ(local_device.synchronize_default_stream(), 0);
  EXPECT_TRUE(mixed_transfer_block_matches(
      local_caches, /*block_id=*/0, /*pull_pattern=*/true));

  ASSERT_TRUE(local_transfer.unlink_cluster(remote_cluster_id,
                                            remote_addr,
                                            received_remote_port,
                                            /*force_flag=*/true));
  command = kStopChildCommand;
  ASSERT_TRUE(write_all(parent_to_child[1], &command, sizeof(command)));
  close(parent_to_child[1]);
  close(child_to_parent[0]);

  int child_status = 0;
  pid_t waited_pid = -1;
  do {
    waited_pid = waitpid(child_pid, &child_status, 0);
  } while (waited_pid < 0 && errno == EINTR);
  if (waited_pid == child_pid) {
    child_guard.release();
  }
  ASSERT_EQ(waited_pid, child_pid);
  ASSERT_TRUE(WIFEXITED(child_status));
  EXPECT_EQ(WEXITSTATUS(child_status), 0);

  const int32_t exit_status = ::testing::Test::HasFailure() ? 1 : 0;
  _exit(exit_status);
}

TEST(MooncakeKVCacheTransferDefaultTest, PushFutureWaitsForDelayedLayerEvent) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "NPU device is required for delayed layer event test.";
  }
  Device device(/*device_id=*/0);
  device.set_device();
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.reserve(2);
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  const TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                        /*dst_tp_size=*/1,
                                        /*dst_dp_rank=*/0);
  const ParallelArgs parallel_args = make_args(/*rank=*/0,
                                               /*world_size=*/1,
                                               /*dp_size=*/1);
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer =
      std::make_shared<KVPushSynchronizerImpl>(/*num_layers=*/2,
                                               /*timeout=*/2000);
  ASSERT_TRUE(synchronizer->record_event(/*layer_index=*/0,
                                         /*device_index=*/0));

  folly::Future<bool> future =
      transfer
          .push_kv_blocks_async(
              {info}, parallel_args, synchronizer, /*is_spec_draft=*/false)
          .toUnsafeFuture();
  const bool first_layer_pushed =
      wait_for_exact_move_call_count(*engine_observer, /*expected_count=*/1);
  if (!first_layer_pushed) {
    synchronizer->abort();
  }
  ASSERT_TRUE(first_layer_pushed);
  EXPECT_FALSE(future.isReady());

  const bool event_recorded =
      synchronizer->record_event(/*layer_index=*/1, /*device_index=*/0);
  if (!event_recorded) {
    synchronizer->abort();
  }
  ASSERT_TRUE(event_recorded);
  future.wait(kTransferCompletionTimeout);
  ASSERT_TRUE(future.isReady());
  EXPECT_TRUE(std::move(future).get());
  ASSERT_EQ(engine_observer->move_calls.size(), 2U);
  EXPECT_EQ(engine_observer->move_calls[0].opcode,
            MooncakeTransferEngine::MoveOpcode::WRITE);
  EXPECT_EQ(engine_observer->move_calls[1].opcode,
            MooncakeTransferEngine::MoveOpcode::WRITE);
}

TEST(MooncakeKVCacheTransferDefaultTest,
     PushFutureReturnsFalseWhenPendingLayerIsAborted) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "NPU device is required for pending layer abort test.";
  }
  Device device(/*device_id=*/0);
  device.set_device();
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.reserve(2);
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  const TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                        /*dst_tp_size=*/1,
                                        /*dst_dp_rank=*/0);
  const ParallelArgs parallel_args = make_args(/*rank=*/0,
                                               /*world_size=*/1,
                                               /*dp_size=*/1);
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer =
      std::make_shared<KVPushSynchronizerImpl>(/*num_layers=*/2,
                                               /*timeout=*/2000);
  ASSERT_TRUE(synchronizer->record_event(/*layer_index=*/0,
                                         /*device_index=*/0));

  folly::Future<bool> future =
      transfer
          .push_kv_blocks_async(
              {info}, parallel_args, synchronizer, /*is_spec_draft=*/false)
          .toUnsafeFuture();
  const bool first_layer_pushed =
      wait_for_exact_move_call_count(*engine_observer, /*expected_count=*/1);
  if (!first_layer_pushed) {
    synchronizer->abort();
  }
  ASSERT_TRUE(first_layer_pushed);
  EXPECT_FALSE(future.isReady());

  synchronizer->abort();
  future.wait(kTransferCompletionTimeout);
  ASSERT_TRUE(future.isReady());
  EXPECT_FALSE(std::move(future).get());
  EXPECT_EQ(engine_observer->move_calls.size(), 1U);
}

TEST(MooncakeKVCacheTransferDefaultTest,
     PushPropagatesSynchronizeLayerFailure) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "NPU device is required for synchronizer failure test.";
  }
  Device device(/*device_id=*/0);
  device.set_device();
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.local_ids = {1};
  mapping.remote_ids = {2};
  KVCacheTransfer::KVCacheInfo info;
  info.dst_cluster_id = 1;
  info.dst_addr = "remote";
  info.mappings.emplace_back(std::move(mapping));
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged_infos;
  merged_infos.emplace("remote", std::move(info));
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer =
      std::make_shared<KVPushSynchronizerImpl>(/*num_layers=*/1);
  synchronizer->abort();

  EXPECT_FALSE(transfer.push_kv_blocks(merged_infos,
                                       synchronizer,
                                       /*is_spec_draft=*/false,
                                       /*kv_split_rank=*/0,
                                       /*kv_split_size=*/1));
  EXPECT_TRUE(engine_observer->move_calls.empty());
}

#endif

#if defined(USE_MLU)
TEST(MooncakeKVCacheTransferDefaultTest,
     AddBufUsesLogicalLengthWithoutChangingBlockBytes) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for Mooncake registration tests.";
  }
  Device device(/*device_id=*/0);
  device.set_device();
  const torch::Device torch_device = device.unwrap();
  MooncakeKVCacheTransferDefault transfer(
      /*device_id=*/0,
      /*listen_port=*/0,
      torch_device,
      /*model_type=*/"test");
  const torch::Tensor tensor = torch::zeros(
      {2, 96, 1}, torch::dtype(torch::kFloat32).device(torch_device));
  std::vector<void*> addrs;
  std::vector<size_t> lens;
  std::vector<uint64_t> block_bytes;

  transfer.add_buf(tensor, addrs, lens, block_bytes);

  ASSERT_EQ(addrs.size(), 1U);
  EXPECT_EQ(addrs[0], tensor.data_ptr());
  EXPECT_EQ(lens[0], tensor.nbytes());
  EXPECT_EQ(block_bytes[0], kScaleBlockBytes);
}

TEST(MooncakeKVCacheTransferDefaultTest,
     RegistersMixedLayersFromProtocolRolesInStableOrder) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for Mooncake registration tests.";
  }
  Device device(/*device_id=*/0);
  device.set_device();
  const torch::Device torch_device = device.unwrap();
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch_device);
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch_device,
                                          /*model_type=*/"glm_moe_dsa",
                                          std::move(engine));
  std::vector<KVCache> caches = make_mixed_transfer_caches(torch_device);
  const KVCacheShape shape = make_indexer_int8_transfer_shape();

  transfer.register_kv_cache(caches, shape, torch::kBFloat16);

  ASSERT_EQ(engine_observer->registered_addrs.size(), 1U);
  ASSERT_EQ(engine_observer->registered_addrs[0].size(), 10U);
  const std::vector<void*> expected_addrs = {
      caches[0].get_k_cache().data_ptr(),
      caches[0].get_index_cache().data_ptr(),
      caches[0].get_indexer_cache_scale()->data_ptr(),
      caches[1].get_k_cache().data_ptr(),
      caches[1].get_k_cache_scale()->data_ptr(),
      caches[2].get_k_cache().data_ptr(),
      caches[2].get_index_cache().data_ptr(),
      caches[2].get_indexer_cache_scale()->data_ptr(),
      caches[3].get_k_cache().data_ptr(),
      caches[3].get_k_cache_scale()->data_ptr()};
  EXPECT_EQ(engine_observer->registered_addrs[0], expected_addrs);
  EXPECT_EQ(engine_observer->registered_lens[0][2],
            caches[0].get_indexer_cache_scale()->nbytes());
  EXPECT_EQ(engine_observer->registered_block_bytes[0][2],
            caches[0].get_indexer_cache_scale()->nbytes() / 2);
  EXPECT_EQ(engine_observer->registered_lens[0][4],
            caches[1].get_k_cache_scale()->nbytes());
  EXPECT_EQ(engine_observer->registered_block_bytes[0][4],
            caches[1].get_k_cache_scale()->nbytes() / 2);

  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.local_ids = {1};
  mapping.remote_ids = {0};
  EXPECT_TRUE(transfer.pull_kv_blocks(
      /*src_cluster_id=*/1, /*src_addr=*/"remote", {mapping}));
  ASSERT_EQ(engine_observer->move_calls.size(), 1U);
  ASSERT_EQ(engine_observer->move_calls[0].mappings.size(), 10U);
  for (size_t index = 0; index < engine_observer->move_calls[0].mappings.size();
       ++index) {
    EXPECT_EQ(engine_observer->move_calls[0].mappings[index].buf_id,
              static_cast<int64_t>(index));
  }
}

TEST(MooncakeKVCacheTransferDefaultTest,
     SpecRegistrationStartsAfterActualMainBufferCount) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for Mooncake registration tests.";
  }
  Device device(/*device_id=*/0);
  device.set_device();
  const torch::Device torch_device = device.unwrap();
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch_device);
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch_device,
                                          /*model_type=*/"glm_moe_dsa",
                                          std::move(engine));
  std::vector<KVCache> main_caches = make_mixed_transfer_caches(torch_device);
  std::vector<KVCache> draft_source = make_mixed_transfer_caches(torch_device);
  std::vector<KVCache> draft_caches;
  draft_caches.reserve(2);
  draft_caches.emplace_back(std::move(draft_source[1]));
  draft_caches.emplace_back(std::move(draft_source[0]));
  const KVCacheShape shape = make_indexer_int8_transfer_shape();

  transfer.register_kv_cache(main_caches, shape, torch::kBFloat16);
  transfer.register_kv_cache_spec(draft_caches, shape, torch::kBFloat16);

  ASSERT_EQ(engine_observer->registered_addrs.size(), 2U);
  EXPECT_EQ(engine_observer->registered_addrs[0].size(), 10U);
  EXPECT_EQ(engine_observer->registered_addrs[1].size(), 5U);
  ASSERT_EQ(transfer.spec_layout_.layers.size(), 2U);
  ASSERT_EQ(transfer.spec_layout_.layers[0].size(), 2U);
  ASSERT_EQ(transfer.spec_layout_.layers[1].size(), 3U);
  EXPECT_EQ(transfer.spec_layout_.layers[0][0].buf_id, 10);
  EXPECT_EQ(transfer.spec_layout_.layers[0][1].buf_id, 11);
  EXPECT_EQ(transfer.spec_layout_.layers[1][0].buf_id, 12);
  EXPECT_EQ(transfer.spec_layout_.layers[1][1].buf_id, 13);
  EXPECT_EQ(transfer.spec_layout_.layers[1][2].buf_id, 14);
}

TEST(MooncakeKVCacheTransferDefaultTest, AddBufRejectsNonContiguousTensor) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  MooncakeKVCacheTransferDefault transfer(
      /*device_id=*/0,
      /*listen_port=*/0,
      torch::Device(torch::kCPU),
      /*model_type=*/"test");
  torch::Tensor tensor = torch::zeros({2, 96, 2}, torch::kFloat32)
                             .transpose(/*dim0=*/1, /*dim1=*/2);
  std::vector<void*> addrs;
  std::vector<size_t> lens;
  std::vector<uint64_t> block_bytes;

  EXPECT_DEATH(transfer.add_buf(tensor, addrs, lens, block_bytes),
               "contiguous");
}

TEST(MooncakeKVCacheTransferDefaultTest,
     IndexScaleRegistersAndRoundTripsWithKvBlocks) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for Mooncake memory transfer.";
  }

  Device device(/*device_id=*/0);
  device.set_device();
  const torch::Device torch_device = device.unwrap();
  const int32_t listen_port = net::get_local_free_port();
  ASSERT_GT(listen_port, 0);

  MooncakeKVCacheTransferDefault transfer(
      /*device_id=*/0,
      static_cast<uint16_t>(listen_port),
      torch_device,
      /*model_type=*/"deepseek_v32");
  transfer.initialize(/*device_id=*/0);

  const KVCacheShape shape = make_indexer_int8_transfer_shape();
  KVCacheCreateOptions options;
  options.device(torch_device)
      .dtype(torch::kBFloat16)
      .num_layers(1)
      .model_type("deepseek_v32")
      .enable_lighting_indexer(true)
      .enable_indexer_cache_quant(true);
  std::vector<KVCache> caches;
  allocate_kv_caches(caches, shape, options);
  ASSERT_EQ(caches.size(), 1U);

  KVCache& cache = caches[0];
  torch::Tensor key_cache = cache.get_k_cache();
  torch::Tensor value_cache = cache.get_v_cache();
  torch::Tensor index_cache = cache.get_index_cache();
  std::optional<torch::Tensor> index_scale = cache.get_indexer_cache_scale();
  ASSERT_TRUE(index_scale.has_value());
  ASSERT_EQ(index_cache.scalar_type(), torch::kChar);
  ASSERT_EQ(index_scale->scalar_type(), torch::kFloat32);
  EXPECT_EQ(index_scale->nbytes(), 2 * kScaleBlockBytes);
  EXPECT_EQ(index_scale->storage().nbytes(), index_scale->nbytes());

  key_cache.index({0}).fill_(1.25);
  key_cache.index({1}).zero_();
  value_cache.index({0}).fill_(-2.5);
  value_cache.index({1}).zero_();
  index_cache.index({0}).fill_(42);
  index_cache.index({1}).zero_();
  index_scale->index({0}).fill_(0.125);
  index_scale->index({1}).zero_();
  device.synchronize_default_stream();

  transfer.register_kv_cache(caches, shape, torch::kBFloat16);

  ASSERT_EQ(transfer.main_layout_.layers.size(), 1U);
  ASSERT_EQ(transfer.main_layout_.layers[0].size(), 4U);
  for (size_t index = 0; index < transfer.main_layout_.layers[0].size();
       ++index) {
    EXPECT_EQ(transfer.main_layout_.layers[0][index].buf_id,
              static_cast<int64_t>(index));
  }

  uint64_t cluster_id = 0;
  std::string addr;
  transfer.get_cache_info(cluster_id, addr);
  ASSERT_FALSE(addr.empty());
  ASSERT_TRUE(transfer.mooncake_te_->open_session(/*cluster_id=*/0, addr));
  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.local_ids = {1};
  mapping.remote_ids = {0};
  ASSERT_TRUE(transfer.pull_kv_blocks(
      /*src_cluster_id=*/0, addr, {mapping}));
  device.synchronize_default_stream();

  EXPECT_TRUE(torch::equal(key_cache.index({1}), key_cache.index({0})));
  EXPECT_TRUE(torch::equal(value_cache.index({1}), value_cache.index({0})));
  EXPECT_TRUE(torch::equal(index_cache.index({1}), index_cache.index({0})));
  EXPECT_TRUE(torch::equal(index_scale->index({1}), index_scale->index({0})));

  EXPECT_TRUE(transfer.unlink_cluster(
      /*cluster_id=*/0,
      addr,
      static_cast<uint16_t>(listen_port),
      /*force_flag=*/true));
}
#endif

}  // namespace xllm
