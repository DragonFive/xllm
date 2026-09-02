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

#include "core/framework/config/disagg_pd_config.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <nlohmann/json.hpp>
#include <string>

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"

namespace xllm {
namespace {

class TransferTypeFlagGuard final {
 public:
  TransferTypeFlagGuard() : old_transfer_type_(FLAGS_kv_cache_transfer_type) {}
  ~TransferTypeFlagGuard() {
    FLAGS_kv_cache_transfer_type = old_transfer_type_;
  }

 private:
  std::string old_transfer_type_;
};

class ReadinessTimeoutFlagGuard final {
 public:
  ReadinessTimeoutFlagGuard()
      : old_timeout_ms_(FLAGS_decode_kv_readiness_timeout_ms) {}
  ~ReadinessTimeoutFlagGuard() {
    FLAGS_decode_kv_readiness_timeout_ms = old_timeout_ms_;
  }

 private:
  int64_t old_timeout_ms_;
};

TEST(DisaggPDConfigTest, DefaultsToMooncakeTransfer) {
  const DisaggPDConfig config;
  EXPECT_EQ(config.kv_cache_transfer_type(), "Mooncake");
  EXPECT_EQ(config.decode_kv_readiness_timeout_ms(), 30000);
}

TEST(DisaggPDConfigTest, ReadsTransferTypeFromFlag) {
  TransferTypeFlagGuard flag_guard;
  FLAGS_kv_cache_transfer_type = "LlmDataDist";

  DisaggPDConfig config;
  config.from_flags();

  EXPECT_EQ(config.kv_cache_transfer_type(), "LlmDataDist");
}

TEST(DisaggPDConfigTest, ExposesTransferTypeInOptionCategory) {
  const auto& option_names = DisaggPDConfig::option_category().option_names;
  EXPECT_NE(
      std::find(
          option_names.begin(), option_names.end(), "kv_cache_transfer_type"),
      option_names.end());
  EXPECT_NE(std::find(option_names.begin(),
                      option_names.end(),
                      "decode_kv_readiness_timeout_ms"),
            option_names.end());
}

TEST(DisaggPDConfigTest, ReadsAndDumpsTransferTypeInJson) {
  const JsonReader json = config::parse_json_string(
      R"json({"kv_cache_transfer_type":"LlmDataDist"})json");
  DisaggPDConfig config;
  config.from_json(json);

  nlohmann::ordered_json dumped;
  config.append_config_json(dumped);

  EXPECT_EQ(config.kv_cache_transfer_type(), "LlmDataDist");
  ASSERT_TRUE(dumped.contains("kv_cache_transfer_type"));
  EXPECT_EQ(dumped["kv_cache_transfer_type"], "LlmDataDist");
}

TEST(DisaggPDConfigTest, ReadsReadinessTimeoutFromFlag) {
  ReadinessTimeoutFlagGuard flag_guard;
  FLAGS_decode_kv_readiness_timeout_ms = 45000;

  DisaggPDConfig config;
  config.from_flags();

  EXPECT_EQ(config.decode_kv_readiness_timeout_ms(), 45000);
}

TEST(DisaggPDConfigTest, ReadsAndDumpsReadinessTimeoutInJson) {
  const JsonReader json = config::parse_json_string(
      R"json({"decode_kv_readiness_timeout_ms":45000})json");
  DisaggPDConfig config;
  config.from_json(json);

  nlohmann::ordered_json dumped;
  config.append_config_json(dumped);

  EXPECT_EQ(config.decode_kv_readiness_timeout_ms(), 45000);
  ASSERT_TRUE(dumped.contains("decode_kv_readiness_timeout_ms"));
  EXPECT_EQ(dumped["decode_kv_readiness_timeout_ms"], 45000);
}

TEST(DisaggPDConfigTest, NormalizesUnsupportedPlatformBackendsToMooncake) {
  KVCacheConfig kv_cache_config;
  SchedulerConfig scheduler_config;

  DisaggPDConfig mlu_config;
  mlu_config.kv_cache_transfer_type("LlmDataDist");
  mlu_config.normalize_mlu(kv_cache_config, scheduler_config);
  EXPECT_EQ(mlu_config.kv_cache_transfer_type(), "Mooncake");

  DisaggPDConfig dcu_config;
  dcu_config.kv_cache_transfer_type("LlmDataDist");
  dcu_config.normalize_dcu(scheduler_config);
  EXPECT_EQ(dcu_config.kv_cache_transfer_type(), "Mooncake");
}

}  // namespace
}  // namespace xllm
