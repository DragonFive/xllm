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

#include "scheduler/disagg_pd_scheduler.h"

#include <brpc/closure_guard.h>
#include <brpc/server.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "common/metrics.h"
#include "distributed_runtime/comm_channel.h"
#include "distributed_runtime/llm_engine.h"
#include "framework/block/block_manager_impl.h"
#include "framework/block/block_manager_pool.h"
#include "framework/kv_cache_transfer/decode_kv_readiness.h"
#include "framework/model/model_args.h"
#include "framework/request/request.h"
#include "framework/request/request_state.h"
#include "framework/tokenizer/tokenizer.h"

namespace xllm {
namespace {

constexpr std::chrono::seconds kSlowReadinessRpcDelay{2};

class SlowDecodeKVReadinessService final : public proto::DistributeWorker {
 public:
  void GetKVCacheLayout(::google::protobuf::RpcController* /*controller*/,
                        const proto::Empty* /*request*/,
                        proto::KVCacheLayoutResponse* response,
                        ::google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    std::this_thread::sleep_for(kSlowReadinessRpcDelay);
    response->set_ok(true);
    response->set_supported(true);
    response->set_serialized_manifest("layout");
  }

  void DrainKVTransferNotifications(
      ::google::protobuf::RpcController* /*controller*/,
      const proto::DrainKVTransferNotificationsRequest* /*request*/,
      proto::DrainKVTransferNotificationsResponse* response,
      ::google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    std::this_thread::sleep_for(kSlowReadinessRpcDelay);
    response->set_ok(true);
    response->set_supported(true);
  }
};

class FakeTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view& /*text*/,
              std::vector<int32_t>* /*ids*/,
              bool /*add_special_tokens*/) const override {
    NOT_IMPLEMENTED();
  }

  std::string decode(const Slice<int32_t>& /*ids*/,
                     bool /*skip_special_tokens*/) const override {
    NOT_IMPLEMENTED();
  }

  std::optional<int32_t> token_to_id(
      const std::string_view& /*token*/) const override {
    NOT_IMPLEMENTED();
  }

  std::string id_to_token(int32_t /*id*/) const override { NOT_IMPLEMENTED(); }

  size_t vocab_size() const override { NOT_IMPLEMENTED(); }

  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>();
  }
};

class FakeEngine final : public Engine {
 public:
  FakeEngine(int32_t num_blocks,
             int32_t block_size,
             int32_t num_speculative_tokens = 0) {
    BlockManagerPool::Options options;
    options.num_blocks(num_blocks)
        .block_size(block_size)
        .enable_prefix_cache(true)
        .enable_disagg_pd(true)
        .num_speculative_tokens(num_speculative_tokens)
        .num_embedding_blocks(num_blocks);
    tokenizer_ = std::make_unique<FakeTokenizer>();
    block_manager_ = std::make_unique<BlockManagerPool>(options, /*dp_size=*/1);
  }

  ForwardOutput step(std::vector<Batch>& /*batch*/) override {
    NOT_IMPLEMENTED();
  }

  void update_last_step_result(std::vector<Batch>& /*batch*/) override {
    NOT_IMPLEMENTED();
  }

  const Tokenizer* tokenizer() const override { return tokenizer_.get(); }

  BlockManagerPool* block_manager_pool() const override {
    return block_manager_.get();
  }

  const ModelArgs& model_args() const override { return model_args_; }

  const TokenizerArgs& tokenizer_args() const override { NOT_IMPLEMENTED(); }

  std::vector<int64_t> get_active_activation_memory() const override {
    NOT_IMPLEMENTED();
  }

  bool init() override { return true; }

  bool pull_kv_blocks(int32_t /*src_dp_size*/,
                      int32_t /*src_dp_rank*/,
                      const std::vector<uint64_t>& /*src_cluster_ids*/,
                      const std::vector<std::string>& /*src_addrs*/,
                      int32_t /*dst_dp_rank*/,
                      const std::vector<KVTransferMapping>& mappings) override {
    pulled_mappings = mappings;
    return true;
  }

  DecodeKVReadinessPollResult poll_decode_kv_readiness(
      size_t /*max_notifications_per_worker*/) override {
    ++readiness_poll_count;
    if (serialized_readiness_enabled_) {
      std::vector<KVTransferNotificationDrainResult> drains;
      drains.swap(serialized_readiness_drains_);
      if (drains.empty()) {
        KVTransferNotificationDrainResult drain;
        drain.supported = true;
        drains.emplace_back(std::move(drain));
      }
      return serialized_readiness_coordinator_.poll(
          drains.size(), [&drains](size_t worker_rank) {
            return std::move(drains[worker_rank]);
          });
    }
    if (!readiness_poll_result.ok && poison_on_poll_failure &&
        readiness_snapshot.found) {
      readiness_snapshot.poisoned = true;
      readiness_snapshot.failure_reason = readiness_poll_result.error;
    }
    return readiness_poll_result;
  }

  DecodeKVReadinessSnapshot get_decode_kv_readiness(
      const std::string& request_id) const override {
    if (serialized_readiness_enabled_) {
      return serialized_readiness_coordinator_.snapshot(request_id);
    }
    if (request_id != "req") {
      return {};
    }
    return readiness_snapshot;
  }

  bool try_publish_decode_kv_readiness(const std::string& request_id) override {
    ++readiness_publish_count;
    if (serialized_readiness_enabled_) {
      return serialized_readiness_coordinator_.try_publish(request_id);
    }
    if (request_id != "req" || !readiness_snapshot.found ||
        !readiness_snapshot.ready || readiness_snapshot.poisoned ||
        readiness_snapshot.published) {
      return false;
    }
    readiness_snapshot.published = true;
    return true;
  }

  void discard_decode_kv_readiness(const std::string& request_id) override {
    if (serialized_readiness_enabled_) {
      if (serialized_readiness_coordinator_.discard(request_id)) {
        ++readiness_discard_count;
      }
      return;
    }
    if (request_id == "req" && readiness_snapshot.found) {
      ++readiness_discard_count;
      readiness_snapshot.found = false;
    }
  }

  bool unlink_cluster(const std::vector<uint64_t>& /*cluster_ids*/,
                      const std::vector<std::string>& /*addrs*/,
                      const std::vector<uint16_t>& /*ports*/,
                      int32_t /*src_dp_size*/,
                      int32_t /*src_kv_split_size*/) override {
    ++unlink_count;
    return true;
  }

  void configure_strict_readiness(uint64_t attempt_epoch,
                                  uint64_t allocation_generation) {
    readiness_snapshot = {};
    readiness_snapshot.found = true;
    readiness_snapshot.attempt_epoch = attempt_epoch;
    readiness_snapshot.allocation_generation = allocation_generation;
    readiness_poll_result.ok = true;
    readiness_poll_result.complete = true;
    readiness_poll_result.error.clear();
    poison_on_poll_failure = false;
  }

  void configure_serialized_readiness(DecodeKVExpectedManifest manifest) {
    serialized_readiness_enabled_ = true;
    auto ledger =
        std::make_shared<DecodeKVReadinessLedger>(std::move(manifest));
    CHECK(!ledger->is_poisoned()) << ledger->failure_reason();
    CHECK(serialized_readiness_coordinator_.register_ledger(std::move(ledger)));
  }

  void queue_serialized_readiness_payload(std::string payload) {
    if (serialized_readiness_drains_.empty()) {
      KVTransferNotificationDrainResult drain;
      drain.supported = true;
      serialized_readiness_drains_.emplace_back(std::move(drain));
    }
    serialized_readiness_drains_.front().payloads.emplace_back(
        std::move(payload));
  }

  void queue_serialized_readiness_drains(
      std::vector<KVTransferNotificationDrainResult> drains) {
    CHECK(serialized_readiness_drains_.empty());
    serialized_readiness_drains_ = std::move(drains);
  }

  void mark_readiness_ready() { readiness_snapshot.ready = true; }

  void fail_readiness_poll(const std::string& error) {
    readiness_poll_result.ok = false;
    readiness_poll_result.complete = false;
    readiness_poll_result.error = error;
    poison_on_poll_failure = true;
  }

  std::vector<KVTransferMapping> pulled_mappings;
  DecodeKVReadinessSnapshot readiness_snapshot;
  DecodeKVReadinessPollResult readiness_poll_result;
  bool poison_on_poll_failure = false;
  int32_t readiness_poll_count = 0;
  int32_t readiness_publish_count = 0;
  int32_t readiness_discard_count = 0;
  int32_t unlink_count = 0;

 private:
  bool serialized_readiness_enabled_ = false;
  detail::DecodeKVReadinessCoordinator serialized_readiness_coordinator_;
  std::vector<KVTransferNotificationDrainResult> serialized_readiness_drains_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<BlockManagerPool> block_manager_;
  ModelArgs model_args_;
};

class TestDisaggPDScheduler final : public DisaggPDScheduler {
 public:
  TestDisaggPDScheduler(Engine* engine, const Options& options)
      : DisaggPDScheduler(engine, options) {}

  void cache_prefill_blocks_for_test(Request* request) {
    cache_prefill_blocks(request);
  }

  bool pop_decode_request_for_test(std::shared_ptr<Request>* request) {
    return request_queue_.read(*request);
  }

  bool enqueue_ready_request_for_test(std::shared_ptr<Request> request) {
    return enqueue_ready_request(std::move(request));
  }

  void do_permanent_rejection_for_test(const std::shared_ptr<Request>& request,
                                       int32_t status_code) {
    do_permanent_rejection(request, status_code);
  }

  void poll_decode_kv_readiness_for_test() { poll_decode_kv_readiness(); }

  void expire_quarantine_drain_for_test() {
    quarantine_drain_deadline_ = std::chrono::steady_clock::now();
  }

  void wait_for_responses() { response_processor_->wait_completion(); }

  bool is_quarantined_for_test(const std::string& request_id) const {
    return quarantined_requests_.find(request_id) !=
           quarantined_requests_.end();
  }

  const Request* quarantined_request_for_test(
      const std::string& request_id) const {
    const auto it = quarantined_requests_.find(request_id);
    return it == quarantined_requests_.end() ? nullptr : it->second.get();
  }

  size_t strict_pending_count_for_test() const {
    return strict_decode_pending_states_.size();
  }

  static int64_t amortized_token_latency_for_test(int64_t latency,
                                                  size_t num_tokens) {
    return amortized_token_latency(latency, num_tokens);
  }

  void update_metrics(std::vector<Sequence*>& sequences) {
    update_token_latency_metrics(sequences);
  }
};

DisaggPDScheduler::Options make_options() {
  DisaggPDScheduler::Options options;
  options.enable_pd_ooc(true)
      .enable_disagg_pd(true)
      .enable_schedule_overlap(false)
      .instance_role(InstanceRole::PREFILL)
      .max_tokens_per_batch(32)
      .max_seqs_per_batch(4)
      .max_tokens_per_chunk_for_prefill(32)
      .dp_size(1);
  return options;
}

DisaggPDScheduler::Options make_mtp_decode_options() {
  DisaggPDScheduler::Options options = make_options();
  options.instance_role(InstanceRole::DECODE).num_speculative_tokens(1);
  return options;
}

DisaggPDScheduler::Options make_decode_options() {
  DisaggPDScheduler::Options options = make_options();
  options.instance_role(InstanceRole::DECODE);
  return options;
}

std::shared_ptr<Request> make_request(
    const std::vector<int32_t>& prompt_token_ids,
    const std::string& request_id = "req") {
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  stopping_checker.set_max_context_len(64);
  stopping_checker.set_ignore_eos(true);

  RequestState state("prompt",
                     prompt_token_ids,
                     sampling_param,
                     scheduler_param,
                     stopping_checker,
                     prompt_token_ids.size() + 8,
                     /*n=*/1,
                     /*best_of=*/1,
                     /*stream=*/false,
                     /*echo=*/false,
                     /*logprobs=*/false,
                     /*skip_special_tokens=*/false,
                     /*include_usage=*/false,
                     /*mm_data=*/nullptr,
                     /*service_request_id=*/nullptr);

  return std::make_shared<Request>(
      request_id, "x-request-id", "x-request-time", state, "service-req");
}

void finish_prefill(Sequence* sequence) {
  CHECK(sequence != nullptr);
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  sequence->append_token(Token(999));
}

size_t first_cache_size(const BlockManagerPool& block_manager) {
  const std::vector<size_t> cache_sizes =
      block_manager.num_blocks_in_prefix_cache();
  CHECK(!cache_sizes.empty());
  return cache_sizes[0];
}

size_t first_free_block_count(const BlockManagerPool& block_manager) {
  const std::vector<size_t> free_block_counts = block_manager.num_free_blocks();
  CHECK(!free_block_counts.empty());
  return free_block_counts[0];
}

void release_prefix_cache(BlockManagerPool* block_manager) {
  CHECK(block_manager != nullptr);
  const size_t num_data_blocks = block_manager->num_blocks() - 1;
  std::vector<int32_t> token_ids;
  token_ids.reserve(num_data_blocks * block_manager->block_size());
  for (size_t i = 0; i < num_data_blocks * block_manager->block_size(); ++i) {
    token_ids.push_back(static_cast<int32_t>(1000 + i));
  }

  std::shared_ptr<Request> request = make_request(token_ids);
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(block_manager->allocate(sequence));
  block_manager->deallocate(sequence);
  EXPECT_EQ(first_cache_size(*block_manager), 0u);
}

bool recv_first_generation(DisaggPDScheduler* scheduler,
                           const torch::Tensor& mtp_embedding,
                           int32_t num_cached_tokens = 0) {
  return scheduler->decode_recv_first_generation(
      "req",
      /*token_id=*/42,
      /*has_logprob=*/false,
      /*logprob=*/0.0f,
      /*time_to_first_token_latency_seconds=*/0.1,
      /*top_tokens=*/{},
      /*top_logprobs=*/{},
      /*kv_cache_transfer_mode=*/"PUSH",
      /*src_cluster_ids=*/{},
      /*src_addrs=*/{},
      /*source_mappings=*/{},
      /*src_dp_size=*/1,
      /*src_dp_rank=*/0,
      mtp_embedding,
      num_cached_tokens);
}

bool recv_strict_first_generation(
    DisaggPDScheduler* scheduler,
    uint64_t attempt_epoch,
    uint64_t allocation_generation,
    double time_to_first_token_latency_seconds = 0.1,
    std::vector<KVTransferMapping> source_mappings = {},
    torch::Tensor mtp_embedding = torch::Tensor(),
    const std::string& request_id = "req") {
  return scheduler->decode_recv_first_generation(
      request_id,
      /*token_id=*/42,
      /*has_logprob=*/true,
      /*logprob=*/-0.25f,
      time_to_first_token_latency_seconds,
      /*top_tokens=*/{42, 43},
      /*top_logprobs=*/{-0.25f, -1.0f},
      /*kv_cache_transfer_mode=*/"PUSH",
      /*src_cluster_ids=*/{101, 102},
      /*src_addrs=*/{"source-a", "source-b"},
      std::move(source_mappings),
      /*src_dp_size=*/2,
      /*src_dp_rank=*/1,
      std::move(mtp_embedding),
      /*num_cached_tokens=*/2,
      attempt_epoch,
      allocation_generation);
}

DecodeKVContributionKey make_readiness_key(
    uint64_t logical_block_ordinal,
    uint64_t destination_physical_block_id) {
  DecodeKVContributionKey key;
  key.source_worker_rank = 0;
  key.destination_worker_rank = 0;
  key.layer_id = 0;
  key.group_id = cache_group_id(BlockType::KV);
  key.cache_role = 0;
  key.logical_block_ordinal = logical_block_ordinal;
  key.destination_physical_block_id = destination_physical_block_id;
  return key;
}

DecodeKVExpectedManifest make_readiness_manifest(
    const std::string& request_id,
    uint64_t attempt_epoch,
    uint64_t allocation_generation,
    const std::vector<DecodeKVContributionKey>& keys) {
  DecodeKVExpectedManifest manifest;
  manifest.request_id = request_id;
  manifest.attempt_epoch = attempt_epoch;
  manifest.allocation_generation = allocation_generation;
  manifest.contributions.reserve(keys.size());
  for (const DecodeKVContributionKey& key : keys) {
    DecodeKVExpectedContribution contribution;
    contribution.key = key;
    contribution.valid_tokens = 2;
    manifest.contributions.emplace_back(std::move(contribution));
  }
  return manifest;
}

DecodeKVReceipt make_readiness_receipt(const std::string& request_id,
                                       const DecodeKVContributionKey& key,
                                       const std::string& submission_id,
                                       uint64_t attempt_epoch,
                                       uint64_t allocation_generation) {
  DecodeKVReceipt receipt;
  receipt.request_id = request_id;
  receipt.key = key;
  receipt.submission_id = submission_id;
  receipt.attempt_epoch = attempt_epoch;
  receipt.allocation_generation = allocation_generation;
  receipt.valid_tokens = 2;
  receipt.completion_level = DecodeKVCompletionLevel::REMOTE_VISIBLE;
  return receipt;
}

std::string serialize_readiness_notification(
    const std::string& submission_id,
    std::vector<DecodeKVReceipt> receipts) {
  MooncakeDecodeKVNotification notification;
  notification.submission_id = submission_id;
  notification.receipts = std::move(receipts);
  std::string serialized;
  std::string error;
  CHECK(serialize_mooncake_decode_kv_notification(
      notification, &serialized, &error))
      << error;
  return serialized;
}

KVTransferNotificationDrainResult make_readiness_drain(
    std::vector<std::string> payloads = {},
    bool more_available = false,
    bool ok = true,
    bool supported = true) {
  KVTransferNotificationDrainResult drain;
  drain.ok = ok;
  drain.supported = supported;
  drain.more_available = more_available;
  drain.payloads = std::move(payloads);
  return drain;
}

OutputFunc capture_failure_output(int32_t* callback_count,
                                  std::optional<Status>* callback_status) {
  CHECK(callback_count != nullptr);
  CHECK(callback_status != nullptr);
  return [callback_count, callback_status](const RequestOutput& output) {
    ++(*callback_count);
    *callback_status = output.status;
    return true;
  };
}

}  // namespace

TEST(CommChannelTest, DecodeKVReadinessRpcsUseFiniteTimeouts) {
  SlowDecodeKVReadinessService service;
  brpc::Server server;
  ASSERT_EQ(server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE), 0);
  ASSERT_EQ(server.Start("127.0.0.1:0", nullptr), 0);

  CommChannel channel;
  const std::string server_address =
      "127.0.0.1:" + std::to_string(server.listen_address().port);
  ASSERT_TRUE(channel.init_brpc(server_address));

  const std::chrono::steady_clock::time_point layout_start =
      std::chrono::steady_clock::now();
  const KVCacheLayoutQueryResult layout = channel.get_kv_cache_layout();
  const std::chrono::steady_clock::duration layout_elapsed =
      std::chrono::steady_clock::now() - layout_start;
  EXPECT_FALSE(layout.ok);
  EXPECT_LT(layout_elapsed, kSlowReadinessRpcDelay);

  const std::chrono::steady_clock::time_point drain_start =
      std::chrono::steady_clock::now();
  const KVTransferNotificationDrainResult drain =
      channel.drain_kv_transfer_notifications(/*max_notifications=*/1);
  const std::chrono::steady_clock::duration drain_elapsed =
      std::chrono::steady_clock::now() - drain_start;
  EXPECT_FALSE(drain.ok);
  EXPECT_LT(drain_elapsed, kSlowReadinessRpcDelay);

  EXPECT_EQ(server.Stop(/*closewait_ms=*/0), 0);
  EXPECT_EQ(server.Join(), 0);
}

TEST(DisaggPDSchedulerTest, CachesPrefillBlocksBeforeRelease) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  BlockManagerPool* block_manager = engine.block_manager_pool();

  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(block_manager->allocate(sequence));
  finish_prefill(sequence);

  scheduler.cache_prefill_blocks_for_test(request.get());
  EXPECT_EQ(first_cache_size(*block_manager), 2u);

  block_manager->deallocate(request.get());
  EXPECT_EQ(sequence->kv_state().num_blocks(BlockType::KV), 0u);
  EXPECT_EQ(first_cache_size(*block_manager), 2u);

  std::shared_ptr<Request> matched_request = make_request({1, 2, 3, 4, 5});
  Sequence* matched_sequence = matched_request->sequences()[0].get();
  block_manager->allocate_shared(matched_sequence);

  EXPECT_EQ(matched_sequence->kv_state().shared_blocks_num(BlockType::KV), 2u);
  block_manager->deallocate(matched_sequence);
  release_prefix_cache(block_manager);
}

TEST(DisaggPDSchedulerTest, CacheSkipsExistingSharedBlocks) {
  FakeEngine engine(/*num_blocks=*/10, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  BlockManagerPool* block_manager = engine.block_manager_pool();

  std::shared_ptr<Request> seed_request = make_request({1, 2, 3, 4});
  Sequence* seed_sequence = seed_request->sequences()[0].get();
  ASSERT_TRUE(block_manager->allocate(seed_sequence));
  finish_prefill(seed_sequence);
  scheduler.cache_prefill_blocks_for_test(seed_request.get());
  block_manager->deallocate(seed_request.get());
  ASSERT_EQ(first_cache_size(*block_manager), 2u);

  std::shared_ptr<Request> extended_request = make_request({1, 2, 3, 4, 5, 6});
  Sequence* extended_sequence = extended_request->sequences()[0].get();
  block_manager->allocate_shared(extended_sequence);
  ASSERT_EQ(extended_sequence->kv_state().shared_blocks_num(BlockType::KV), 2u);
  ASSERT_TRUE(block_manager->allocate(extended_sequence,
                                      extended_sequence->num_prompt_tokens()));
  finish_prefill(extended_sequence);

  scheduler.cache_prefill_blocks_for_test(extended_request.get());
  EXPECT_EQ(first_cache_size(*block_manager), 3u);

  block_manager->deallocate(extended_request.get());
  EXPECT_EQ(first_cache_size(*block_manager), 3u);
  release_prefix_cache(block_manager);
}

TEST(DisaggPDSchedulerTest, MtpFirstGenerationRequiresBootstrapBeforeQueue) {
  FakeEngine engine(/*num_blocks=*/8,
                    /*block_size=*/2,
                    /*num_speculative_tokens=*/1);
  TestDisaggPDScheduler scheduler(&engine, make_mtp_decode_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  ASSERT_TRUE(
      engine.block_manager_pool()->allocate(request->sequences()[0].get()));
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  EXPECT_FALSE(recv_first_generation(&scheduler, torch::Tensor()));
  std::shared_ptr<Request> queued;
  EXPECT_FALSE(scheduler.pop_decode_request_for_test(&queued));
}

TEST(DisaggPDSchedulerTest, MtpFirstGenerationStoresBootstrapThenQueues) {
  FakeEngine engine(/*num_blocks=*/8,
                    /*block_size=*/2,
                    /*num_speculative_tokens=*/1);
  TestDisaggPDScheduler scheduler(&engine, make_mtp_decode_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  ASSERT_GE(sequence->get_embedding_block_id(), 0);
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  torch::Tensor embedding = torch::tensor({1.0f, 2.0f});
  EXPECT_TRUE(recv_first_generation(&scheduler, embedding));

  std::shared_ptr<Request> queued;
  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  EXPECT_EQ(queued->request_id(), "req");
  EXPECT_EQ(queued->sequences()[0]->tokens().back(), 42);
  EXPECT_TRUE(torch::equal(
      queued->sequences()[0]->get_mtp_bootstrap_embedding(), embedding));
}

TEST(DisaggPDSchedulerTest, GroupedPullAlignsActiveSwaSuffix) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_decode_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());

  BlockManager::Options swa_options;
  swa_options.num_blocks(8).block_size(2);
  BlockManagerImpl swa_manager(swa_options);
  std::vector<Block> live_swa_blocks = swa_manager.allocate(2);
  std::vector<Block> logical_swa_blocks(2);
  logical_swa_blocks.insert(
      logical_swa_blocks.end(), live_swa_blocks.begin(), live_swa_blocks.end());
  sequence->add_blocks(BlockType::SWA, logical_swa_blocks);
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  KVTransferMapping source_mapping;
  source_mapping.group_id = cache_group_id(BlockType::SWA);
  source_mapping.remote_ids = {101, 102};
  ASSERT_TRUE(scheduler.decode_recv_first_generation(
      "req",
      /*token_id=*/42,
      /*has_logprob=*/false,
      /*logprob=*/0.0f,
      /*time_to_first_token_latency_seconds=*/0.1,
      /*top_tokens=*/{},
      /*top_logprobs=*/{},
      /*kv_cache_transfer_mode=*/"PULL",
      /*src_cluster_ids=*/{1},
      /*src_addrs=*/{"remote"},
      /*source_mappings=*/{source_mapping},
      /*src_dp_size=*/1,
      /*src_dp_rank=*/0));

  ASSERT_EQ(engine.pulled_mappings.size(), 1U);
  EXPECT_EQ(engine.pulled_mappings[0].group_id, cache_group_id(BlockType::SWA));
  EXPECT_EQ(
      engine.pulled_mappings[0].local_ids,
      (std::vector<uint64_t>{static_cast<uint64_t>(live_swa_blocks[0].id()),
                             static_cast<uint64_t>(live_swa_blocks[1].id())}));
  EXPECT_EQ(engine.pulled_mappings[0].remote_ids,
            (std::vector<uint64_t>{101, 102}));

  std::shared_ptr<Request> queued;
  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  engine.block_manager_pool()->deallocate(queued.get());
  queued->sequences()[0]->kv_state().erase_blocks(BlockType::SWA);
}

TEST(DisaggPDSchedulerTest, FirstDecodeTokenLatencyIsNonNegative) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  EXPECT_TRUE(recv_first_generation(&scheduler, torch::Tensor()));

  std::shared_ptr<Request> queued;
  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  // Base rebuilt in decode_recv_first_generation must not sit in the future:
  // pre-fix it was created_time + ttft (~now+100ms), yielding a negative ITL.
  int64_t first_itl = queued->sequences()[0]->tbt(absl::Now());
  EXPECT_GE(first_itl, 0);
}

TEST(DisaggPDSchedulerTest, PreservesPrefillCachedTokensOnDecodeRequest) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  ASSERT_TRUE(recv_first_generation(
      &scheduler, torch::Tensor(), /*num_cached_tokens=*/2));

  std::shared_ptr<Request> queued;
  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  EXPECT_EQ(queued->num_prefix_cache_tokens(), 2u);
}

TEST(DisaggPDSchedulerTest,
     StrictReadinessTokenFirstPublishesAndEnqueuesExactlyOnce) {
  constexpr uint64_t kAttemptEpoch = 7;
  constexpr uint64_t kAllocationGeneration = 11;
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  engine.configure_strict_readiness(kAttemptEpoch, kAllocationGeneration);
  TestDisaggPDScheduler scheduler(&engine, make_decode_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  ASSERT_TRUE(scheduler.try_allocate(request->sequences()[0].get()));
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.local_ids = {1};
  mapping.remote_ids = {101};
  mapping.logical_block_ordinals = {2};
  mapping.valid_tokens = {1};
  mapping.receipt_remote_ids = {101};
  mapping.remote_shared_num = 2;
  const torch::Tensor embedding = torch::tensor({1.0f, 2.0f});
  EXPECT_TRUE(recv_strict_first_generation(&scheduler,
                                           kAttemptEpoch,
                                           kAllocationGeneration,
                                           /*ttft_seconds=*/0.1,
                                           {mapping},
                                           embedding));
  EXPECT_TRUE(recv_strict_first_generation(&scheduler,
                                           kAttemptEpoch,
                                           kAllocationGeneration,
                                           /*ttft_seconds=*/0.1,
                                           {mapping},
                                           embedding.clone()));
  EXPECT_EQ(scheduler.strict_pending_count_for_test(), 1u);
  std::shared_ptr<Request> queued;
  EXPECT_FALSE(scheduler.pop_decode_request_for_test(&queued));

  engine.mark_readiness_ready();
  scheduler.poll_decode_kv_readiness_for_test();
  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  EXPECT_EQ(queued->sequences()[0]->tokens().back(), 42);
  EXPECT_EQ(engine.readiness_publish_count, 1);
  EXPECT_EQ(engine.readiness_discard_count, 1);

  scheduler.poll_decode_kv_readiness_for_test();
  EXPECT_FALSE(scheduler.pop_decode_request_for_test(&request));
  EXPECT_EQ(engine.readiness_publish_count, 1);
  engine.block_manager_pool()->deallocate(queued.get());
}

TEST(DisaggPDSchedulerTest, StrictReadinessReceiptFirstWaitsForToken) {
  constexpr uint64_t kAttemptEpoch = 7;
  constexpr uint64_t kAllocationGeneration = 11;
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  engine.configure_strict_readiness(kAttemptEpoch, kAllocationGeneration);
  TestDisaggPDScheduler scheduler(&engine, make_decode_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  ASSERT_TRUE(scheduler.try_allocate(request->sequences()[0].get()));
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  engine.mark_readiness_ready();
  scheduler.poll_decode_kv_readiness_for_test();
  std::shared_ptr<Request> queued;
  EXPECT_FALSE(scheduler.pop_decode_request_for_test(&queued));
  EXPECT_EQ(engine.readiness_publish_count, 0);

  ASSERT_TRUE(recv_strict_first_generation(
      &scheduler, kAttemptEpoch, kAllocationGeneration));
  EXPECT_FALSE(scheduler.pop_decode_request_for_test(&queued));
  scheduler.poll_decode_kv_readiness_for_test();
  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  EXPECT_EQ(engine.readiness_publish_count, 1);
  engine.block_manager_pool()->deallocate(queued.get());
}

TEST(DisaggPDSchedulerTest,
     StrictReadinessWaitsForCurrentReceiverDrainToComplete) {
  constexpr uint64_t kAttemptEpoch = 7;
  constexpr uint64_t kAllocationGeneration = 11;
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  engine.configure_strict_readiness(kAttemptEpoch, kAllocationGeneration);
  TestDisaggPDScheduler scheduler(&engine, make_decode_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  ASSERT_TRUE(scheduler.try_allocate(request->sequences()[0].get()));
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));
  ASSERT_TRUE(recv_strict_first_generation(
      &scheduler, kAttemptEpoch, kAllocationGeneration));

  engine.mark_readiness_ready();
  engine.readiness_poll_result.complete = false;
  scheduler.poll_decode_kv_readiness_for_test();

  std::shared_ptr<Request> queued;
  EXPECT_FALSE(scheduler.pop_decode_request_for_test(&queued));
  EXPECT_EQ(engine.readiness_publish_count, 0);

  engine.readiness_poll_result.complete = true;
  scheduler.poll_decode_kv_readiness_for_test();
  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  EXPECT_EQ(engine.readiness_publish_count, 1);
  engine.block_manager_pool()->deallocate(queued.get());
}

TEST(DisaggPDSchedulerTest,
     SerializedNotificationDelayedFinalReceiptEnqueuesOnlyAfterComplete) {
  constexpr uint64_t kAttemptEpoch = 7;
  constexpr uint64_t kAllocationGeneration = 11;
  const std::string request_id = "delayed-request";
  const DecodeKVContributionKey first_key =
      make_readiness_key(/*logical_block_ordinal=*/0,
                         /*destination_physical_block_id=*/100);
  const DecodeKVContributionKey delayed_key =
      make_readiness_key(/*logical_block_ordinal=*/1,
                         /*destination_physical_block_id=*/101);

  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  engine.configure_serialized_readiness(
      make_readiness_manifest(request_id,
                              kAttemptEpoch,
                              kAllocationGeneration,
                              {first_key, delayed_key}));
  TestDisaggPDScheduler scheduler(&engine, make_decode_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4}, request_id);
  ASSERT_TRUE(scheduler.try_allocate(request->sequences()[0].get()));
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));
  ASSERT_TRUE(recv_strict_first_generation(&scheduler,
                                           kAttemptEpoch,
                                           kAllocationGeneration,
                                           /*ttft_seconds=*/0.1,
                                           /*source_mappings=*/{},
                                           torch::Tensor(),
                                           request_id));

  engine.queue_serialized_readiness_drains(
      {make_readiness_drain({serialize_readiness_notification(
           "submission-first",
           {make_readiness_receipt(request_id,
                                   first_key,
                                   "submission-first",
                                   kAttemptEpoch,
                                   kAllocationGeneration)})}),
       make_readiness_drain()});
  scheduler.poll_decode_kv_readiness_for_test();

  std::shared_ptr<Request> queued;
  EXPECT_FALSE(scheduler.pop_decode_request_for_test(&queued));
  EXPECT_EQ(scheduler.strict_pending_count_for_test(), 1u);
  EXPECT_EQ(engine.readiness_publish_count, 0);

  engine.queue_serialized_readiness_drains(
      {make_readiness_drain(
           {serialize_readiness_notification(
               "submission-delayed",
               {make_readiness_receipt(request_id,
                                       delayed_key,
                                       "submission-delayed",
                                       kAttemptEpoch,
                                       kAllocationGeneration)})},
           /*more_available=*/true),
       make_readiness_drain()});
  scheduler.poll_decode_kv_readiness_for_test();

  EXPECT_FALSE(scheduler.pop_decode_request_for_test(&queued));
  EXPECT_EQ(scheduler.strict_pending_count_for_test(), 1u);
  EXPECT_EQ(engine.readiness_publish_count, 0);

  engine.queue_serialized_readiness_drains(
      {make_readiness_drain(), make_readiness_drain()});
  scheduler.poll_decode_kv_readiness_for_test();

  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  EXPECT_EQ(queued->request_id(), request_id);
  EXPECT_EQ(queued->sequences()[0]->tokens().back(), 42);
  EXPECT_EQ(engine.readiness_publish_count, 1);
  EXPECT_EQ(engine.readiness_discard_count, 1);
  engine.block_manager_pool()->deallocate(queued.get());
}

TEST(DisaggPDSchedulerTest,
     SerializedNotificationFailuresQuarantineAndIsolateNextRequest) {
  enum class FailureKind : int8_t {
    MISSING_RECEIPT = 0,
    CONFLICTING_DUPLICATE = 1,
    WORKER_DRAIN = 2,
  };
  struct FailureCase {
    std::string name;
    FailureKind kind;
    std::string expected_message;
  };
  const std::vector<FailureCase> failure_cases = {
      {"missing", FailureKind::MISSING_RECEIPT, "receiver receipts"},
      {"conflicting-duplicate",
       FailureKind::CONFLICTING_DUPLICATE,
       "conflicting"},
      {"worker-drain", FailureKind::WORKER_DRAIN, "failed to drain"}};

  for (const FailureCase& failure_case : failure_cases) {
    SCOPED_TRACE(failure_case.name);
    constexpr uint64_t kAttemptEpoch = 7;
    constexpr uint64_t kAllocationGeneration = 11;
    const std::string failed_request_id = "failed-" + failure_case.name;
    const std::string healthy_request_id = "healthy-" + failure_case.name;
    const DecodeKVContributionKey first_key =
        make_readiness_key(/*logical_block_ordinal=*/0,
                           /*destination_physical_block_id=*/100);
    const DecodeKVContributionKey missing_key =
        make_readiness_key(/*logical_block_ordinal=*/1,
                           /*destination_physical_block_id=*/101);

    FakeEngine engine(/*num_blocks=*/16, /*block_size=*/2);
    DisaggPDScheduler::Options options = make_decode_options();
    options.decode_kv_readiness_timeout_ms(0);
    int32_t callback_count = 0;
    std::optional<Status> callback_status;
    TestDisaggPDScheduler scheduler(&engine, options);
    engine.configure_serialized_readiness(
        make_readiness_manifest(failed_request_id,
                                kAttemptEpoch,
                                kAllocationGeneration,
                                {first_key, missing_key}));
    std::shared_ptr<Request> failed_request =
        make_request({1, 2, 3, 4}, failed_request_id);
    failed_request->state().output_func =
        capture_failure_output(&callback_count, &callback_status);
    ASSERT_TRUE(scheduler.try_allocate(failed_request->sequences()[0].get()));
    ASSERT_TRUE(scheduler.decode_schedule(failed_request, "prefill"));
    ASSERT_TRUE(recv_strict_first_generation(&scheduler,
                                             kAttemptEpoch,
                                             kAllocationGeneration,
                                             /*ttft_seconds=*/0.1,
                                             /*source_mappings=*/{},
                                             torch::Tensor(),
                                             failed_request_id));

    const std::string original_payload = serialize_readiness_notification(
        "submission-original",
        {make_readiness_receipt(failed_request_id,
                                first_key,
                                "submission-original",
                                kAttemptEpoch,
                                kAllocationGeneration)});
    if (failure_case.kind == FailureKind::WORKER_DRAIN) {
      engine.queue_serialized_readiness_drains(
          {make_readiness_drain({original_payload}),
           make_readiness_drain(/*payloads=*/{},
                                /*more_available=*/false,
                                /*ok=*/false)});
    } else {
      engine.queue_serialized_readiness_payload(original_payload);
    }
    if (failure_case.kind == FailureKind::CONFLICTING_DUPLICATE) {
      engine.queue_serialized_readiness_payload(
          serialize_readiness_notification(
              "submission-conflict",
              {make_readiness_receipt(failed_request_id,
                                      first_key,
                                      "submission-conflict",
                                      kAttemptEpoch,
                                      kAllocationGeneration)}));
    }
    scheduler.poll_decode_kv_readiness_for_test();
    scheduler.wait_for_responses();

    EXPECT_TRUE(scheduler.is_quarantined_for_test(failed_request_id));
    EXPECT_EQ(callback_count, 1);
    ASSERT_TRUE(callback_status.has_value());
    EXPECT_NE(callback_status->message().find(failure_case.expected_message),
              std::string::npos);
    std::shared_ptr<Request> queued;
    EXPECT_FALSE(scheduler.pop_decode_request_for_test(&queued));

    const DecodeKVContributionKey healthy_key =
        make_readiness_key(/*logical_block_ordinal=*/2,
                           /*destination_physical_block_id=*/102);
    engine.configure_serialized_readiness(
        make_readiness_manifest(healthy_request_id,
                                kAttemptEpoch + 1,
                                kAllocationGeneration + 1,
                                {healthy_key}));
    std::shared_ptr<Request> healthy_request =
        make_request({5, 6, 7, 8}, healthy_request_id);
    ASSERT_TRUE(scheduler.try_allocate(healthy_request->sequences()[0].get()));
    ASSERT_TRUE(scheduler.decode_schedule(healthy_request, "prefill"));
    ASSERT_TRUE(recv_strict_first_generation(&scheduler,
                                             kAttemptEpoch + 1,
                                             kAllocationGeneration + 1,
                                             /*ttft_seconds=*/0.1,
                                             /*source_mappings=*/{},
                                             torch::Tensor(),
                                             healthy_request_id));
    engine.queue_serialized_readiness_payload(serialize_readiness_notification(
        "submission-healthy",
        {make_readiness_receipt(healthy_request_id,
                                healthy_key,
                                "submission-healthy",
                                kAttemptEpoch + 1,
                                kAllocationGeneration + 1)}));
    scheduler.poll_decode_kv_readiness_for_test();

    ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
    EXPECT_EQ(queued->request_id(), healthy_request_id);
    EXPECT_EQ(queued->sequences()[0]->tokens().back(), 42);
    EXPECT_FALSE(scheduler.is_quarantined_for_test(healthy_request_id));
    EXPECT_EQ(engine.readiness_publish_count, 1);
    EXPECT_EQ(engine.readiness_discard_count, 2);
    engine.block_manager_pool()->deallocate(queued.get());
  }
}

TEST(DisaggPDSchedulerTest,
     SerializedNotificationRoutesReceiptToMatchingRequestLedger) {
  constexpr uint64_t kAttemptEpoch = 7;
  constexpr uint64_t kAllocationGeneration = 11;
  const std::string first_request_id = "route-first";
  const std::string second_request_id = "route-second";
  const DecodeKVContributionKey first_key =
      make_readiness_key(/*logical_block_ordinal=*/0,
                         /*destination_physical_block_id=*/100);
  const DecodeKVContributionKey second_key =
      make_readiness_key(/*logical_block_ordinal=*/1,
                         /*destination_physical_block_id=*/101);

  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  engine.configure_serialized_readiness(make_readiness_manifest(
      first_request_id, kAttemptEpoch, kAllocationGeneration, {first_key}));
  engine.configure_serialized_readiness(
      make_readiness_manifest(second_request_id,
                              kAttemptEpoch + 1,
                              kAllocationGeneration + 1,
                              {second_key}));
  engine.queue_serialized_readiness_drains(
      {make_readiness_drain({serialize_readiness_notification(
           "submission-second",
           {make_readiness_receipt(second_request_id,
                                   second_key,
                                   "submission-second",
                                   kAttemptEpoch + 1,
                                   kAllocationGeneration + 1)})}),
       make_readiness_drain()});

  const DecodeKVReadinessPollResult poll =
      engine.poll_decode_kv_readiness(/*max_notifications_per_worker=*/256);

  EXPECT_TRUE(poll.ok);
  EXPECT_TRUE(poll.complete);
  const DecodeKVReadinessSnapshot first =
      engine.get_decode_kv_readiness(first_request_id);
  const DecodeKVReadinessSnapshot second =
      engine.get_decode_kv_readiness(second_request_id);
  EXPECT_FALSE(first.ready);
  EXPECT_TRUE(second.ready);
  EXPECT_FALSE(engine.try_publish_decode_kv_readiness(first_request_id));
  EXPECT_TRUE(engine.try_publish_decode_kv_readiness(second_request_id));
  engine.discard_decode_kv_readiness(first_request_id);
  engine.discard_decode_kv_readiness(second_request_id);
}

TEST(DecodeKVReadinessCoordinatorTest,
     LedgerRegisteredDuringDrainWaitsForNextCompletePoll) {
  constexpr uint64_t kAttemptEpoch = 7;
  constexpr uint64_t kAllocationGeneration = 11;
  const std::string request_id = "registered-during-drain";
  const DecodeKVContributionKey key =
      make_readiness_key(/*logical_block_ordinal=*/0,
                         /*destination_physical_block_id=*/100);
  DecodeKVExpectedManifest manifest = make_readiness_manifest(
      request_id, kAttemptEpoch, kAllocationGeneration, {key});
  const std::string payload = serialize_readiness_notification(
      "submission",
      {make_readiness_receipt(request_id,
                              key,
                              "submission",
                              kAttemptEpoch,
                              kAllocationGeneration)});
  detail::DecodeKVReadinessCoordinator coordinator;
  bool registered = false;

  const DecodeKVReadinessPollResult first_poll = coordinator.poll(
      /*worker_count=*/1,
      [&coordinator, &manifest, &payload, &registered](size_t worker_rank) {
        EXPECT_EQ(worker_rank, 0u);
        std::shared_ptr<DecodeKVReadinessLedger> ledger =
            std::make_shared<DecodeKVReadinessLedger>(std::move(manifest));
        registered = coordinator.register_ledger(std::move(ledger));
        return make_readiness_drain({payload});
      });

  EXPECT_TRUE(registered);
  EXPECT_TRUE(first_poll.ok);
  EXPECT_TRUE(first_poll.complete);
  const DecodeKVReadinessSnapshot first_snapshot =
      coordinator.snapshot(request_id);
  EXPECT_TRUE(first_snapshot.found);
  EXPECT_FALSE(first_snapshot.ready);
  EXPECT_FALSE(first_snapshot.poisoned);
  EXPECT_FALSE(coordinator.try_publish(request_id));

  const DecodeKVReadinessPollResult second_poll = coordinator.poll(
      /*worker_count=*/1, [](size_t worker_rank) {
        EXPECT_EQ(worker_rank, 0u);
        return make_readiness_drain();
      });

  EXPECT_TRUE(second_poll.ok);
  EXPECT_TRUE(second_poll.complete);
  EXPECT_TRUE(coordinator.snapshot(request_id).ready);
  EXPECT_TRUE(coordinator.try_publish(request_id));
  EXPECT_FALSE(coordinator.try_publish(request_id));
  EXPECT_TRUE(coordinator.discard(request_id));
}

TEST(DecodeKVReadinessCoordinatorTest,
     WorkerDrainFailurePoisonsOnlyPollStartLedgers) {
  constexpr uint64_t kAttemptEpoch = 7;
  constexpr uint64_t kAllocationGeneration = 11;
  const std::string first_request_id = "worker-failure-first";
  const std::string second_request_id = "worker-failure-second";
  const std::string new_request_id = "worker-failure-new";
  const DecodeKVContributionKey first_key =
      make_readiness_key(/*logical_block_ordinal=*/0,
                         /*destination_physical_block_id=*/100);
  const DecodeKVContributionKey second_key =
      make_readiness_key(/*logical_block_ordinal=*/1,
                         /*destination_physical_block_id=*/101);
  const DecodeKVContributionKey new_key =
      make_readiness_key(/*logical_block_ordinal=*/2,
                         /*destination_physical_block_id=*/102);
  detail::DecodeKVReadinessCoordinator coordinator;
  ASSERT_TRUE(
      coordinator.register_ledger(std::make_shared<DecodeKVReadinessLedger>(
          make_readiness_manifest(first_request_id,
                                  kAttemptEpoch,
                                  kAllocationGeneration,
                                  {first_key}))));
  ASSERT_TRUE(
      coordinator.register_ledger(std::make_shared<DecodeKVReadinessLedger>(
          make_readiness_manifest(second_request_id,
                                  kAttemptEpoch + 1,
                                  kAllocationGeneration + 1,
                                  {second_key}))));
  std::vector<KVTransferNotificationDrainResult> drains = {
      make_readiness_drain({serialize_readiness_notification(
          "submission-new",
          {make_readiness_receipt(new_request_id,
                                  new_key,
                                  "submission-new",
                                  kAttemptEpoch + 2,
                                  kAllocationGeneration + 2)})}),
      make_readiness_drain(/*payloads=*/{},
                           /*more_available=*/false,
                           /*ok=*/false)};
  bool registered_during_drain = false;

  const DecodeKVReadinessPollResult poll = coordinator.poll(
      drains.size(),
      [&coordinator,
       &drains,
       &new_key,
       &new_request_id,
       &registered_during_drain,
       kAttemptEpoch,
       kAllocationGeneration](size_t worker_rank) {
        if (worker_rank == 0) {
          registered_during_drain = coordinator.register_ledger(
              std::make_shared<DecodeKVReadinessLedger>(
                  make_readiness_manifest(new_request_id,
                                          kAttemptEpoch + 2,
                                          kAllocationGeneration + 2,
                                          {new_key})));
        }
        return std::move(drains[worker_rank]);
      });

  EXPECT_TRUE(registered_during_drain);
  EXPECT_FALSE(poll.ok);
  EXPECT_FALSE(poll.complete);
  EXPECT_NE(poll.error.find("failed to drain"), std::string::npos);
  const DecodeKVReadinessSnapshot first =
      coordinator.snapshot(first_request_id);
  const DecodeKVReadinessSnapshot second =
      coordinator.snapshot(second_request_id);
  const DecodeKVReadinessSnapshot newly_registered =
      coordinator.snapshot(new_request_id);
  EXPECT_TRUE(first.poisoned);
  EXPECT_TRUE(second.poisoned);
  EXPECT_FALSE(newly_registered.poisoned);
  EXPECT_FALSE(first.ready);
  EXPECT_FALSE(second.ready);
  EXPECT_FALSE(newly_registered.ready);
  EXPECT_NE(first.failure_reason.find("failed to drain"), std::string::npos);
  EXPECT_NE(second.failure_reason.find("failed to drain"), std::string::npos);
  EXPECT_FALSE(coordinator.try_publish(first_request_id));
  EXPECT_FALSE(coordinator.try_publish(second_request_id));
  EXPECT_TRUE(coordinator.discard(first_request_id));
  EXPECT_TRUE(coordinator.discard(second_request_id));

  const DecodeKVReadinessPollResult next_poll = coordinator.poll(
      /*worker_count=*/2,
      [](size_t /*worker_rank*/) { return make_readiness_drain(); });
  EXPECT_TRUE(next_poll.ok);
  EXPECT_TRUE(next_poll.complete);
  EXPECT_TRUE(coordinator.snapshot(new_request_id).ready);
  EXPECT_TRUE(coordinator.try_publish(new_request_id));
  EXPECT_TRUE(coordinator.discard(new_request_id));
}

TEST(DisaggPDSchedulerTest, StrictReadinessTimesOutWithoutReceipt) {
  constexpr uint64_t kAttemptEpoch = 7;
  constexpr uint64_t kAllocationGeneration = 11;
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  engine.configure_strict_readiness(kAttemptEpoch, kAllocationGeneration);
  DisaggPDScheduler::Options options = make_decode_options();
  options.decode_kv_readiness_timeout_ms(0);
  TestDisaggPDScheduler scheduler(&engine, options);
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  int32_t callback_count = 0;
  std::optional<Status> callback_status;
  request->state().output_func =
      capture_failure_output(&callback_count, &callback_status);
  ASSERT_TRUE(
      engine.block_manager_pool()->allocate(request->sequences()[0].get()));
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));
  ASSERT_TRUE(recv_strict_first_generation(
      &scheduler, kAttemptEpoch, kAllocationGeneration));

  scheduler.poll_decode_kv_readiness_for_test();
  scheduler.wait_for_responses();

  EXPECT_TRUE(scheduler.is_quarantined_for_test("req"));
  EXPECT_EQ(scheduler.strict_pending_count_for_test(), 0u);
  EXPECT_EQ(engine.readiness_discard_count, 1);
  EXPECT_EQ(callback_count, 1);
  ASSERT_TRUE(callback_status.has_value());
  EXPECT_EQ(callback_status->code(), StatusCode::UNKNOWN);
  EXPECT_NE(callback_status->message().find("receiver receipts"),
            std::string::npos);
  std::shared_ptr<Request> queued;
  EXPECT_FALSE(scheduler.pop_decode_request_for_test(&queued));
}

TEST(DisaggPDSchedulerTest, StrictReadinessTimesOutWithoutFirstGeneration) {
  constexpr uint64_t kAttemptEpoch = 7;
  constexpr uint64_t kAllocationGeneration = 11;
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  engine.configure_strict_readiness(kAttemptEpoch, kAllocationGeneration);
  engine.mark_readiness_ready();
  DisaggPDScheduler::Options options = make_decode_options();
  options.decode_kv_readiness_timeout_ms(0);
  TestDisaggPDScheduler scheduler(&engine, options);
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  int32_t callback_count = 0;
  std::optional<Status> callback_status;
  request->state().output_func =
      capture_failure_output(&callback_count, &callback_status);
  ASSERT_TRUE(
      engine.block_manager_pool()->allocate(request->sequences()[0].get()));
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  scheduler.poll_decode_kv_readiness_for_test();
  scheduler.wait_for_responses();

  EXPECT_TRUE(scheduler.is_quarantined_for_test("req"));
  EXPECT_EQ(scheduler.strict_pending_count_for_test(), 0u);
  EXPECT_EQ(engine.readiness_discard_count, 1);
  EXPECT_EQ(callback_count, 1);
  ASSERT_TRUE(callback_status.has_value());
  EXPECT_EQ(callback_status->code(), StatusCode::UNKNOWN);
  EXPECT_NE(callback_status->message().find("FirstGeneration"),
            std::string::npos);
  std::shared_ptr<Request> queued;
  EXPECT_FALSE(scheduler.pop_decode_request_for_test(&queued));
}

TEST(DisaggPDSchedulerTest,
     QuarantineDrainsLateNotificationsAndRejectsRequestIdReuse) {
  constexpr uint64_t kAttemptEpoch = 7;
  constexpr uint64_t kAllocationGeneration = 11;
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  engine.configure_strict_readiness(kAttemptEpoch, kAllocationGeneration);
  DisaggPDScheduler::Options options = make_decode_options();
  options.decode_kv_readiness_timeout_ms(0);
  TestDisaggPDScheduler scheduler(&engine, options);
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  int32_t callback_count = 0;
  std::optional<Status> callback_status;
  request->state().output_func =
      capture_failure_output(&callback_count, &callback_status);
  ASSERT_TRUE(
      engine.block_manager_pool()->allocate(request->sequences()[0].get()));
  const size_t retained_free_block_count =
      first_free_block_count(*engine.block_manager_pool());
  const Request* quarantined_request = request.get();
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  scheduler.poll_decode_kv_readiness_for_test();
  scheduler.wait_for_responses();
  ASSERT_EQ(callback_count, 1);
  ASSERT_TRUE(callback_status.has_value());
  EXPECT_EQ(callback_status->code(), StatusCode::UNKNOWN);
  ASSERT_EQ(scheduler.quarantined_request_for_test("req"), quarantined_request);
  EXPECT_EQ(first_free_block_count(*engine.block_manager_pool()),
            retained_free_block_count);

  scheduler.poll_decode_kv_readiness_for_test();
  EXPECT_EQ(engine.readiness_poll_count, 2);
  scheduler.expire_quarantine_drain_for_test();
  scheduler.poll_decode_kv_readiness_for_test();
  EXPECT_EQ(engine.readiness_poll_count, 2);

  engine.configure_strict_readiness(kAttemptEpoch + 1,
                                    kAllocationGeneration + 1);
  std::shared_ptr<Request> retry = make_request({5, 6, 7, 8});
  ASSERT_TRUE(
      engine.block_manager_pool()->allocate(retry->sequences()[0].get()));
  EXPECT_LT(first_free_block_count(*engine.block_manager_pool()),
            retained_free_block_count);
  EXPECT_FALSE(scheduler.decode_schedule(retry, "prefill"));
  EXPECT_EQ(scheduler.quarantined_request_for_test("req"), quarantined_request);
  EXPECT_EQ(first_free_block_count(*engine.block_manager_pool()),
            retained_free_block_count);
}

TEST(DisaggPDSchedulerTest, UnlinkFailsAndQuarantinesStrictPendingRequest) {
  constexpr uint64_t kAttemptEpoch = 7;
  constexpr uint64_t kAllocationGeneration = 11;
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  engine.configure_strict_readiness(kAttemptEpoch, kAllocationGeneration);
  TestDisaggPDScheduler scheduler(&engine, make_decode_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  int32_t callback_count = 0;
  std::optional<Status> callback_status;
  request->state().output_func =
      capture_failure_output(&callback_count, &callback_status);
  ASSERT_TRUE(
      engine.block_manager_pool()->allocate(request->sequences()[0].get()));
  const size_t retained_free_block_count =
      first_free_block_count(*engine.block_manager_pool());
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  EXPECT_TRUE(scheduler.unlink_instance("prefill",
                                        /*cluster_ids=*/{101},
                                        /*addrs=*/{"source-a"},
                                        /*ports=*/{1234},
                                        /*dp_size=*/1,
                                        /*src_kv_split_size=*/1));
  scheduler.wait_for_responses();
  scheduler.poll_decode_kv_readiness_for_test();

  EXPECT_TRUE(scheduler.is_quarantined_for_test("req"));
  EXPECT_EQ(scheduler.strict_pending_count_for_test(), 0u);
  EXPECT_EQ(engine.readiness_discard_count, 1);
  EXPECT_EQ(engine.unlink_count, 1);
  EXPECT_EQ(engine.readiness_poll_count, 1);
  EXPECT_EQ(first_free_block_count(*engine.block_manager_pool()),
            retained_free_block_count);
  EXPECT_EQ(callback_count, 1);
  ASSERT_TRUE(callback_status.has_value());
  EXPECT_EQ(callback_status->code(), StatusCode::UNKNOWN);
  EXPECT_NE(callback_status->message().find("unlinked"), std::string::npos);
}

TEST(DisaggPDSchedulerTest, StrictReadinessRejectsStaleIdentity) {
  constexpr uint64_t kAttemptEpoch = 7;
  constexpr uint64_t kAllocationGeneration = 11;
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  engine.configure_strict_readiness(kAttemptEpoch, kAllocationGeneration);
  TestDisaggPDScheduler scheduler(&engine, make_decode_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  int32_t callback_count = 0;
  std::optional<Status> callback_status;
  request->state().output_func =
      capture_failure_output(&callback_count, &callback_status);
  ASSERT_TRUE(
      engine.block_manager_pool()->allocate(request->sequences()[0].get()));
  const size_t retained_free_block_count =
      first_free_block_count(*engine.block_manager_pool());
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  EXPECT_FALSE(recv_strict_first_generation(
      &scheduler, kAttemptEpoch - 1, kAllocationGeneration));
  scheduler.wait_for_responses();
  EXPECT_TRUE(scheduler.is_quarantined_for_test("req"));
  EXPECT_EQ(scheduler.strict_pending_count_for_test(), 0u);
  EXPECT_EQ(engine.readiness_discard_count, 1);
  EXPECT_EQ(first_free_block_count(*engine.block_manager_pool()),
            retained_free_block_count);
  EXPECT_EQ(callback_count, 1);
  ASSERT_TRUE(callback_status.has_value());
  EXPECT_EQ(callback_status->code(), StatusCode::UNKNOWN);
  std::shared_ptr<Request> queued;
  EXPECT_FALSE(scheduler.pop_decode_request_for_test(&queued));
}

TEST(DisaggPDSchedulerTest,
     StrictReadinessRejectsConflictingFirstGenerationMapping) {
  constexpr uint64_t kAttemptEpoch = 7;
  constexpr uint64_t kAllocationGeneration = 11;
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  engine.configure_strict_readiness(kAttemptEpoch, kAllocationGeneration);
  TestDisaggPDScheduler scheduler(&engine, make_decode_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  int32_t callback_count = 0;
  std::optional<Status> callback_status;
  request->state().output_func =
      capture_failure_output(&callback_count, &callback_status);
  ASSERT_TRUE(
      engine.block_manager_pool()->allocate(request->sequences()[0].get()));
  const size_t retained_free_block_count =
      first_free_block_count(*engine.block_manager_pool());
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.remote_ids = {101};
  mapping.logical_block_ordinals = {2};
  mapping.valid_tokens = {1};
  mapping.receipt_remote_ids = {101};
  ASSERT_TRUE(recv_strict_first_generation(
      &scheduler, kAttemptEpoch, kAllocationGeneration, 0.1, {mapping}));

  mapping.receipt_remote_ids = {102};
  EXPECT_FALSE(recv_strict_first_generation(
      &scheduler, kAttemptEpoch, kAllocationGeneration, 0.1, {mapping}));
  scheduler.wait_for_responses();
  EXPECT_TRUE(scheduler.is_quarantined_for_test("req"));
  EXPECT_EQ(engine.readiness_discard_count, 1);
  EXPECT_EQ(first_free_block_count(*engine.block_manager_pool()),
            retained_free_block_count);
  EXPECT_EQ(callback_count, 1);
  ASSERT_TRUE(callback_status.has_value());
  EXPECT_EQ(callback_status->code(), StatusCode::UNKNOWN);
}

TEST(DisaggPDSchedulerTest, StrictReadinessPollFailureQuarantinesRequest) {
  constexpr uint64_t kAttemptEpoch = 7;
  constexpr uint64_t kAllocationGeneration = 11;
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  engine.configure_strict_readiness(kAttemptEpoch, kAllocationGeneration);
  engine.fail_readiness_poll("Decode worker failed to drain notifications");
  TestDisaggPDScheduler scheduler(&engine, make_decode_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  int32_t callback_count = 0;
  std::optional<Status> callback_status;
  request->state().output_func =
      capture_failure_output(&callback_count, &callback_status);
  ASSERT_TRUE(
      engine.block_manager_pool()->allocate(request->sequences()[0].get()));
  const size_t retained_free_block_count =
      first_free_block_count(*engine.block_manager_pool());
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  scheduler.poll_decode_kv_readiness_for_test();
  scheduler.wait_for_responses();
  EXPECT_TRUE(scheduler.is_quarantined_for_test("req"));
  EXPECT_EQ(engine.readiness_poll_count, 1);
  EXPECT_EQ(engine.readiness_discard_count, 1);
  EXPECT_EQ(first_free_block_count(*engine.block_manager_pool()),
            retained_free_block_count);
  EXPECT_EQ(callback_count, 1);
  ASSERT_TRUE(callback_status.has_value());
  EXPECT_EQ(callback_status->code(), StatusCode::UNKNOWN);
  std::shared_ptr<Request> queued;
  EXPECT_FALSE(scheduler.pop_decode_request_for_test(&queued));
}

TEST(DisaggPDSchedulerTest, LegacyPushEnqueuesWithoutReadinessLedger) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_decode_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  ASSERT_TRUE(scheduler.try_allocate(request->sequences()[0].get()));
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  ASSERT_TRUE(recv_first_generation(&scheduler, torch::Tensor()));
  std::shared_ptr<Request> queued;
  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  EXPECT_EQ(engine.readiness_poll_count, 0);
  EXPECT_EQ(engine.readiness_publish_count, 0);
  engine.block_manager_pool()->deallocate(queued.get());
}

TEST(DisaggPDSchedulerTest,
     SingleDpPrefillAdmissionAssignsSourceRankBeforeDecodeRpc) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_EQ(sequence->dp_rank(), -1);

  ASSERT_TRUE(scheduler.enqueue_ready_request_for_test(request));

  EXPECT_EQ(sequence->dp_rank(), 0);
}

TEST(DisaggPDSchedulerTest, PromptAtDecodeBlockCapacityIsNotPermanent) {
  EXPECT_FALSE(exceeds_decode_capacity(
      /*num_prompt_tokens=*/6, /*block_size=*/2, /*num_blocks=*/4));
}

TEST(DisaggPDSchedulerTest, OnlyTemporaryDecodeCapacityResponseIsRetryable) {
  EXPECT_FALSE(is_permanent_rejection(kDecodeAddNewRequestSuccessStatusCode));
  EXPECT_FALSE(
      is_permanent_rejection(kDecodeAddNewTemporaryCapacityStatusCode));
  EXPECT_TRUE(is_permanent_rejection(kDecodeAddNewPromptTooLongStatusCode));
  EXPECT_TRUE(is_permanent_rejection(/*status_code=*/500));
}

TEST(DisaggPDSchedulerTest, TerminalDecodeAdmissionMapsStatusAndMessage) {
  struct TestCase {
    int32_t status_code;
    StatusCode expected_code;
    std::string expected_message;
  };
  const std::vector<TestCase> test_cases = {
      {kDecodeAddNewPromptTooLongStatusCode,
       StatusCode::RESOURCE_EXHAUSTED,
       "Request prompt exceeds decode KV cache capacity"},
      {/*status_code=*/500,
       StatusCode::UNKNOWN,
       "Decode rejected request during admission, status_code=500"}};

  for (const TestCase& test_case : test_cases) {
    FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
    TestDisaggPDScheduler scheduler(&engine, make_options());
    std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
    int32_t callback_count = 0;
    std::optional<Status> callback_status;
    request->state().output_func =
        capture_failure_output(&callback_count, &callback_status);
    ASSERT_TRUE(
        engine.block_manager_pool()->allocate(request->sequences()[0].get()));

    scheduler.do_permanent_rejection_for_test(request, test_case.status_code);
    scheduler.wait_for_responses();

    EXPECT_EQ(callback_count, 1);
    ASSERT_TRUE(callback_status.has_value());
    EXPECT_EQ(callback_status->code(), test_case.expected_code);
    EXPECT_EQ(callback_status->message(), test_case.expected_message);
  }
}

TEST(DisaggPDSchedulerTest, PromptBeyondDecodeBlockCapacityIsPermanent) {
  EXPECT_TRUE(exceeds_decode_capacity(
      /*num_prompt_tokens=*/7, /*block_size=*/2, /*num_blocks=*/4));
}

TEST(DisaggPDSchedulerTest, TemporaryDecodeBlockPressureIsNotPermanent) {
  FakeEngine engine(/*num_blocks=*/4, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  BlockManagerPool* block_manager = engine.block_manager_pool();
  std::shared_ptr<Request> holder = make_request({1, 2, 3, 4, 5, 6});
  ASSERT_TRUE(block_manager->try_allocate(holder->sequences()[0].get()));
  std::shared_ptr<Request> request = make_request({7, 8});
  Sequence* sequence = request->sequences()[0].get();

  EXPECT_FALSE(scheduler.try_allocate(sequence));
  EXPECT_FALSE(scheduler.exceeds_decode_capacity(sequence));

  block_manager->deallocate(holder.get());
}

TEST(DisaggPDSchedulerTest, OversizedDecodePromptIsPermanent) {
  FakeEngine engine(/*num_blocks=*/4, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  BlockManagerPool* block_manager = engine.block_manager_pool();
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4, 5, 6, 7});
  Sequence* sequence = request->sequences()[0].get();

  EXPECT_FALSE(scheduler.try_allocate(sequence));
  EXPECT_TRUE(scheduler.exceeds_decode_capacity(sequence));

  block_manager->deallocate(request.get());
}

TEST(DisaggPDSchedulerTest, InvalidPrefillCachedTokensFallBackToZero) {
  for (int32_t num_cached_tokens : {-1, 5}) {
    FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
    TestDisaggPDScheduler scheduler(&engine, make_options());
    std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
    Sequence* sequence = request->sequences()[0].get();
    ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
    sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
    ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

    ASSERT_TRUE(
        recv_first_generation(&scheduler, torch::Tensor(), num_cached_tokens));

    std::shared_ptr<Request> queued;
    ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
    EXPECT_EQ(queued->num_prefix_cache_tokens(), 0u);
  }
}

TEST(DisaggPDSchedulerTest, AmortizedTokenLatencyRoundsHalfUp) {
  // Amortized per-token latency is round(latency / n) via (latency + n/2) / n.
  EXPECT_EQ(TestDisaggPDScheduler::amortized_token_latency_for_test(100, 4),
            25);
  EXPECT_EQ(TestDisaggPDScheduler::amortized_token_latency_for_test(101, 4),
            25);
  EXPECT_EQ(TestDisaggPDScheduler::amortized_token_latency_for_test(102, 4),
            26);
  EXPECT_EQ(TestDisaggPDScheduler::amortized_token_latency_for_test(50, 5), 10);
  EXPECT_EQ(TestDisaggPDScheduler::amortized_token_latency_for_test(53, 5), 11);
  // With a single committed token amortized latency equals the raw latency.
  EXPECT_EQ(TestDisaggPDScheduler::amortized_token_latency_for_test(37, 1), 37);
}

TEST(DisaggPDSchedulerTest, SchedulerDoesNotOverwriteSpeculativeOutputGauge) {
  FakeEngine engine(/*num_blocks=*/8,
                    /*block_size=*/2,
                    /*num_speculative_tokens=*/1);
  TestDisaggPDScheduler scheduler(&engine, make_mtp_decode_options());
  GAUGE_SET(speculative_mean_tokens_per_decode_step, 4.25);

  std::shared_ptr<Request> first_request = make_request({1, 2, 3, 4});
  Sequence* first_sequence = first_request->sequences()[0].get();
  first_sequence->kv_state().set_kv_cache_tokens_num(
      first_sequence->num_prompt_tokens());
  for (int32_t token_id = 10; token_id < 15; ++token_id) {
    first_sequence->append_token(Token(token_id));
  }

  std::shared_ptr<Request> second_request = make_request({5, 6, 7, 8});
  Sequence* second_sequence = second_request->sequences()[0].get();
  second_sequence->kv_state().set_kv_cache_tokens_num(
      second_sequence->num_prompt_tokens());
  for (int32_t token_id = 20; token_id < 23; ++token_id) {
    second_sequence->append_token(Token(token_id));
  }

  std::vector<Sequence*> sequences = {first_sequence, second_sequence};
  scheduler.update_metrics(sequences);

  EXPECT_DOUBLE_EQ(GAUGE_speculative_mean_tokens_per_decode_step.get_value(),
                   4.25);
  EXPECT_EQ(first_sequence->generated_tokens_since_latency(), 0u);
  EXPECT_EQ(second_sequence->generated_tokens_since_latency(), 0u);
}

TEST(DisaggPDSchedulerTest, SpeculativeMetricsSilentWhenDisabled) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  // make_options() keeps num_speculative_tokens at its default of 0.
  TestDisaggPDScheduler scheduler(&engine, make_options());

  GAUGE_SET(speculative_mean_tokens_per_decode_step, -1.0);

  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  sequence->append_token(Token(10));
  sequence->append_token(Token(11));
  std::vector<Sequence*> sequences = {sequence};

  scheduler.update_metrics(sequences);

  EXPECT_DOUBLE_EQ(GAUGE_speculative_mean_tokens_per_decode_step.get_value(),
                   -1.0);
}

TEST(DisaggPDSchedulerTest, StructuredOutputFieldsPreserveWireTags) {
  proto::DisaggRequest request;
  request.set_include_stop_str_in_output(true);
  request.set_json_object(true);
  request.set_json_reasoning_enabled(true);

  std::string serialized;
  ASSERT_TRUE(request.SerializeToString(&serialized));

  proto::DisaggRequest decoded;
  ASSERT_TRUE(decoded.ParseFromString(serialized));
  EXPECT_TRUE(decoded.include_stop_str_in_output());
  EXPECT_TRUE(decoded.json_object());
  EXPECT_TRUE(decoded.json_reasoning_enabled());
  EXPECT_EQ(proto::DisaggRequest::kIncludeStopStrInOutputFieldNumber, 39);
  EXPECT_EQ(proto::DisaggRequest::kJsonObjectFieldNumber, 40);
  EXPECT_EQ(proto::DisaggRequest::kJsonReasoningEnabledFieldNumber, 41);
}

}  // namespace xllm
