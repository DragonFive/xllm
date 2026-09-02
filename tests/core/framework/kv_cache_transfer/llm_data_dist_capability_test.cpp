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

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "framework/kv_cache_transfer/kv_cache_transfer.h"

namespace xllm {
namespace {

detail::LlmDataDistCapability supported_prefill_capability() {
  detail::LlmDataDistCapability capability;
  capability.is_npu_backend = true;
  capability.instance_role = InstanceRole::PREFILL;
  capability.transfer_mode = "PUSH";
  capability.model_type = "glm_moe_dsa";
  capability.has_lightning_indexer = true;
  capability.kv_cache_dtype = "auto";
  capability.dp_size = 1;
  capability.cp_size = 4;
  capability.kv_split_size = 1;
  return capability;
}

void expect_rejected(const detail::LlmDataDistCapability& capability,
                     const std::string& expected_message) {
  const std::optional<std::string> error =
      detail::validate_llm_data_dist_capability(capability);
  ASSERT_TRUE(error.has_value());
  EXPECT_NE(error->find(expected_message), std::string::npos) << *error;
}

detail::KVTransferCoverageKey make_coverage_key(
    const std::string& request_id,
    int32_t owner_rank,
    int64_t layer_id,
    KVCacheTensorRole::Value cache_role,
    uint64_t destination_physical_block_id) {
  return detail::KVTransferCoverageKey{request_id,
                                       owner_rank,
                                       layer_id,
                                       cache_group_id(BlockType::KV),
                                       cache_role,
                                       destination_physical_block_id};
}

TEST(LlmDataDistCapabilityTest, AcceptsSupportedPrefillAndDecodeTopologies) {
  detail::LlmDataDistCapability capability = supported_prefill_capability();
  EXPECT_FALSE(
      detail::validate_llm_data_dist_capability(capability).has_value());

  capability.cp_size = 1;
  EXPECT_FALSE(
      detail::validate_llm_data_dist_capability(capability).has_value());

  capability.instance_role = InstanceRole::DECODE;
  EXPECT_FALSE(
      detail::validate_llm_data_dist_capability(capability).has_value());
}

TEST(LlmDataDistCapabilityTest, RejectsUnsupportedBackendAndMode) {
  detail::LlmDataDistCapability capability = supported_prefill_capability();
  capability.is_npu_backend = false;
  expect_rejected(capability, "requires an NPU build");

  capability = supported_prefill_capability();
  capability.transfer_mode = "PULL";
  expect_rejected(capability, "requires kv_cache_transfer_mode=PUSH");
}

TEST(LlmDataDistCapabilityTest, RejectsUnsupportedModelAndCacheLayouts) {
  detail::LlmDataDistCapability capability = supported_prefill_capability();
  capability.model_type = "deepseek_v32";
  expect_rejected(capability, "model_type=glm_moe_dsa");

  capability = supported_prefill_capability();
  capability.has_lightning_indexer = false;
  expect_rejected(capability, "requires Lightning Indexer cache");

  capability = supported_prefill_capability();
  capability.kv_cache_dtype = "int8";
  expect_rejected(capability, "requires kv_cache_dtype=auto");

  capability = supported_prefill_capability();
  capability.enable_xtensor = true;
  expect_rejected(capability, "does not support XTensor cache");

  capability = supported_prefill_capability();
  capability.has_linear_attention_cache = true;
  expect_rejected(capability, "does not support linear-attention cache");

  capability = supported_prefill_capability();
  capability.has_grouped_cache_layout = true;
  expect_rejected(capability, "does not support grouped cache layout");
}

TEST(LlmDataDistCapabilityTest, RejectsDraftAndUnsupportedTopologies) {
  detail::LlmDataDistCapability capability = supported_prefill_capability();
  capability.is_spec_draft = true;
  expect_rejected(capability, "does not support speculative or MTP draft");

  capability = supported_prefill_capability();
  capability.dp_size = 2;
  expect_rejected(capability, "PREFILL(dp=1,cp>=1,kv_split=1)");

  capability = supported_prefill_capability();
  capability.kv_split_size = 2;
  expect_rejected(capability, "PREFILL(dp=1,cp>=1,kv_split=1)");

  capability = supported_prefill_capability();
  capability.cp_size = 0;
  expect_rejected(capability, "PREFILL(dp=1,cp>=1,kv_split=1)");

  capability = supported_prefill_capability();
  capability.instance_role = InstanceRole::MIX;
  expect_rejected(capability, "got role=MIX");
}

TEST(KVTransferCoverageLedgerTest, AcceptsCompleteSourceSubmissionCoverage) {
  const detail::KVTransferCoverageKey request_a_first_block_key =
      make_coverage_key("request-a",
                        /*owner_rank=*/0,
                        /*layer_id=*/0,
                        KVCacheTensorRole::KEY,
                        /*destination_physical_block_id=*/10);
  const detail::KVTransferCoverageKey request_a_second_block_key =
      make_coverage_key("request-a",
                        /*owner_rank=*/0,
                        /*layer_id=*/0,
                        KVCacheTensorRole::KEY,
                        /*destination_physical_block_id=*/11);
  const detail::KVTransferCoverageKey request_a_index_key =
      make_coverage_key("request-a",
                        /*owner_rank=*/0,
                        /*layer_id=*/1,
                        KVCacheTensorRole::INDEX,
                        /*destination_physical_block_id=*/10);
  const detail::KVTransferCoverageKey request_b_key =
      make_coverage_key("request-b",
                        /*owner_rank=*/0,
                        /*layer_id=*/0,
                        KVCacheTensorRole::KEY,
                        /*destination_physical_block_id=*/10);
  detail::KVTransferCoverageLedger ledger({request_a_first_block_key,
                                           request_a_second_block_key,
                                           request_a_index_key,
                                           request_b_key});

  EXPECT_EQ(ledger.record(request_b_key),
            detail::KVTransferCoverageRecordResult::RECORDED);
  EXPECT_EQ(ledger.record(request_a_index_key),
            detail::KVTransferCoverageRecordResult::RECORDED);
  EXPECT_EQ(ledger.record(request_a_second_block_key),
            detail::KVTransferCoverageRecordResult::RECORDED);
  EXPECT_FALSE(ledger.is_ready());
  EXPECT_EQ(ledger.record(request_a_first_block_key),
            detail::KVTransferCoverageRecordResult::RECORDED);

  EXPECT_TRUE(ledger.is_ready());
  EXPECT_EQ(ledger.expected_count(), 4U);
  EXPECT_EQ(ledger.received_count(), 4U);
  EXPECT_TRUE(ledger.missing().empty());
}

TEST(KVTransferCoverageLedgerTest, ReportsMissingContribution) {
  const detail::KVTransferCoverageKey first_expected_contribution =
      make_coverage_key("missing-contribution",
                        /*owner_rank=*/0,
                        /*layer_id=*/0,
                        KVCacheTensorRole::KEY,
                        /*destination_physical_block_id=*/20);
  const detail::KVTransferCoverageKey missing_contribution =
      make_coverage_key("missing-contribution",
                        /*owner_rank=*/0,
                        /*layer_id=*/0,
                        KVCacheTensorRole::KEY,
                        /*destination_physical_block_id=*/21);
  detail::KVTransferCoverageLedger ledger(
      {first_expected_contribution, missing_contribution});

  EXPECT_EQ(ledger.record(first_expected_contribution),
            detail::KVTransferCoverageRecordResult::RECORDED);
  EXPECT_FALSE(ledger.is_ready());
  const std::vector<detail::KVTransferCoverageKey> missing = ledger.missing();
  ASSERT_EQ(missing.size(), 1U);
  EXPECT_EQ(missing[0].request_id, missing_contribution.request_id);
  EXPECT_EQ(missing[0].owner_rank, missing_contribution.owner_rank);
  EXPECT_EQ(missing[0].layer_id, missing_contribution.layer_id);
  EXPECT_EQ(missing[0].cache_role, missing_contribution.cache_role);
  EXPECT_EQ(missing[0].destination_physical_block_id,
            missing_contribution.destination_physical_block_id);
}

TEST(KVTransferCoverageLedgerTest, RejectsDuplicateContribution) {
  const detail::KVTransferCoverageKey contribution =
      make_coverage_key("duplicate",
                        /*owner_rank=*/0,
                        /*layer_id=*/2,
                        KVCacheTensorRole::INDEX_SCALE,
                        /*destination_physical_block_id=*/30);
  detail::KVTransferCoverageLedger ledger({contribution});

  EXPECT_EQ(ledger.record(contribution),
            detail::KVTransferCoverageRecordResult::RECORDED);
  EXPECT_EQ(ledger.record(contribution),
            detail::KVTransferCoverageRecordResult::DUPLICATE);
  EXPECT_FALSE(ledger.is_ready());
  EXPECT_EQ(ledger.received_count(), 1U);
  EXPECT_EQ(ledger.duplicate_count(), 1U);
}

TEST(KVTransferCoverageLedgerTest, RejectsUnexpectedContribution) {
  const detail::KVTransferCoverageKey expected_contribution =
      make_coverage_key("unexpected",
                        /*owner_rank=*/0,
                        /*layer_id=*/3,
                        KVCacheTensorRole::KEY,
                        /*destination_physical_block_id=*/40);
  const detail::KVTransferCoverageKey unexpected_contribution =
      make_coverage_key("unexpected",
                        /*owner_rank=*/0,
                        /*layer_id=*/3,
                        KVCacheTensorRole::KEY,
                        /*destination_physical_block_id=*/41);
  detail::KVTransferCoverageLedger ledger({expected_contribution});

  EXPECT_EQ(ledger.record(unexpected_contribution),
            detail::KVTransferCoverageRecordResult::UNEXPECTED);
  EXPECT_FALSE(ledger.is_ready());
  EXPECT_EQ(ledger.received_count(), 0U);
  EXPECT_EQ(ledger.unexpected_count(), 1U);

  EXPECT_EQ(ledger.record(expected_contribution),
            detail::KVTransferCoverageRecordResult::RECORDED);
  EXPECT_FALSE(ledger.is_ready());
}

TEST(KVTransferCoverageLedgerTest, RejectsDuplicateExpectedManifest) {
  const detail::KVTransferCoverageKey expected_contribution =
      make_coverage_key("duplicate-manifest",
                        /*owner_rank=*/0,
                        /*layer_id=*/3,
                        KVCacheTensorRole::KEY,
                        /*destination_physical_block_id=*/40);
  detail::KVTransferCoverageLedger ledger(
      {expected_contribution, expected_contribution});

  EXPECT_FALSE(ledger.is_ready());
  EXPECT_EQ(ledger.expected_count(), 1U);
  EXPECT_EQ(ledger.duplicate_count(), 1U);
  EXPECT_EQ(ledger.record(expected_contribution),
            detail::KVTransferCoverageRecordResult::RECORDED);
  EXPECT_FALSE(ledger.is_ready());
}

TEST(KVTransferCoverageLedgerTest, BecomesReadyAfterLastExpectedContribution) {
  const detail::KVTransferCoverageKey first_expected_contribution =
      make_coverage_key("last-expected-contribution",
                        /*owner_rank=*/0,
                        /*layer_id=*/3,
                        KVCacheTensorRole::KEY,
                        /*destination_physical_block_id=*/40);
  const detail::KVTransferCoverageKey last_expected_contribution =
      make_coverage_key("last-expected-contribution",
                        /*owner_rank=*/0,
                        /*layer_id=*/3,
                        KVCacheTensorRole::KEY,
                        /*destination_physical_block_id=*/41);
  detail::KVTransferCoverageLedger ledger(
      {first_expected_contribution, last_expected_contribution});

  EXPECT_EQ(ledger.record(first_expected_contribution),
            detail::KVTransferCoverageRecordResult::RECORDED);
  EXPECT_FALSE(ledger.is_ready());
  EXPECT_EQ(ledger.received_count(), 1U);

  EXPECT_EQ(ledger.record(last_expected_contribution),
            detail::KVTransferCoverageRecordResult::RECORDED);
  EXPECT_TRUE(ledger.is_ready());
}

}  // namespace
}  // namespace xllm
