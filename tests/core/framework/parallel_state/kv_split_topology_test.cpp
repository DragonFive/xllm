/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "framework/parallel_state/kv_split_topology.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "framework/parallel_state/parallel_args.h"

namespace xllm {
namespace parallel_state {
namespace {

TEST(KvSplitTopology, UsesSplitMajorRankAndGroupOrder) {
  struct TestCase {
    int32_t world_size;
    int32_t cp_size;
    int32_t kv_split_size;
  };
  const std::vector<TestCase> cases = {
      {/*world_size=*/4, /*cp_size=*/4, /*kv_split_size=*/1},
      {/*world_size=*/4, /*cp_size=*/4, /*kv_split_size=*/4},
      {/*world_size=*/8, /*cp_size=*/4, /*kv_split_size=*/2},
  };

  for (const TestCase& test_case : cases) {
    const int32_t group_count = test_case.world_size / test_case.kv_split_size;
    std::set<int32_t> seen_ranks;
    for (int32_t global_rank = 0; global_rank < test_case.world_size;
         ++global_rank) {
      SCOPED_TRACE(global_rank);
      const std::vector<int32_t> group_ranks =
          compute_kv_split_group_ranks(global_rank,
                                       test_case.world_size,
                                       test_case.cp_size,
                                       test_case.kv_split_size);
      ASSERT_EQ(group_ranks.size(), test_case.kv_split_size);
      EXPECT_EQ(compute_kv_split_rank(global_rank,
                                      test_case.world_size,
                                      test_case.cp_size,
                                      test_case.kv_split_size),
                global_rank / group_count);
      EXPECT_EQ(group_ranks[global_rank / group_count], global_rank);

      const int32_t group_id = global_rank % group_count;
      for (int32_t kv_rank = 0; kv_rank < test_case.kv_split_size; ++kv_rank) {
        EXPECT_EQ(group_ranks[kv_rank], group_id + kv_rank * group_count);
      }
      seen_ranks.insert(group_ranks.begin(), group_ranks.end());
    }
    EXPECT_EQ(seen_ranks.size(), static_cast<size_t>(test_case.world_size));
  }
}

TEST(KvSplitTopology, ParallelArgsFallbackUsesEffectiveSplitMajorRank) {
  const int32_t world_size = 8;
  const int32_t cp_size = 4;
  const int32_t kv_split_size = 2;
  for (int32_t global_rank = 0; global_rank < world_size; ++global_rank) {
    ParallelArgs args(global_rank, world_size, nullptr);
    args.cp_size(cp_size);
    args.kv_split_size(kv_split_size);
    EXPECT_EQ(
        args.kv_split_rank(),
        compute_kv_split_rank(global_rank, world_size, cp_size, kv_split_size));
  }

  ParallelArgs legacy_args(/*rank=*/6, world_size, nullptr);
  legacy_args.cp_size(cp_size);
  legacy_args.kv_split_size(/*value=*/0);
  EXPECT_EQ(legacy_args.kv_split_size_effective(), cp_size);
  EXPECT_EQ(
      legacy_args.kv_split_rank(),
      compute_kv_split_rank(/*global_rank=*/6, world_size, cp_size, cp_size));
}

TEST(KvSplitTopology, RejectsInvalidCpAndKvTopology) {
  EXPECT_DEATH(compute_kv_split_rank(/*global_rank=*/0,
                                     /*world_size=*/8,
                                     /*cp_size=*/4,
                                     /*kv_split_size=*/0),
               "must be positive");
  EXPECT_DEATH(compute_kv_split_rank(/*global_rank=*/0,
                                     /*world_size=*/12,
                                     /*cp_size=*/4,
                                     /*kv_split_size=*/3),
               "must be divisible");
  EXPECT_DEATH(compute_kv_split_group_ranks(/*global_rank=*/0,
                                            /*world_size=*/8,
                                            /*cp_size=*/4,
                                            /*kv_split_size=*/8),
               "must not exceed");
  EXPECT_DEATH(compute_kv_split_rank(/*global_rank=*/0,
                                     /*world_size=*/7,
                                     /*kv_split_size=*/2),
               "world_size");
}

TEST(KvSplitTopology, OwnerShardedKvRequiresDisaggregatedPrefill) {
  const std::optional<std::string> supported =
      validate_owner_sharded_kv_execution(/*kv_split_size=*/4,
                                          /*enable_disagg_pd=*/true,
                                          /*is_prefill_role=*/true);
  EXPECT_FALSE(supported.has_value());

  const std::optional<std::string> standalone =
      validate_owner_sharded_kv_execution(/*kv_split_size=*/4,
                                          /*enable_disagg_pd=*/false,
                                          /*is_prefill_role=*/false);
  ASSERT_TRUE(standalone.has_value());
  EXPECT_NE(standalone->find("standalone Decode cannot reconstruct KV"),
            std::string::npos);

  const std::optional<std::string> role_only =
      validate_owner_sharded_kv_execution(/*kv_split_size=*/4,
                                          /*enable_disagg_pd=*/false,
                                          /*is_prefill_role=*/true);
  EXPECT_TRUE(role_only.has_value());

  const std::optional<std::string> pd_default =
      validate_owner_sharded_kv_execution(/*kv_split_size=*/4,
                                          /*enable_disagg_pd=*/true,
                                          /*is_prefill_role=*/false);
  EXPECT_TRUE(pd_default.has_value());
}

TEST(KvSplitTopology, ReplicatedKvAllowsStandaloneExecution) {
  const std::optional<std::string> result =
      validate_owner_sharded_kv_execution(/*kv_split_size=*/1,
                                          /*enable_disagg_pd=*/false,
                                          /*is_prefill_role=*/false);
  EXPECT_FALSE(result.has_value());
}

}  // namespace
}  // namespace parallel_state
}  // namespace xllm
