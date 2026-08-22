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

#include "core/framework/parallel_state/kv_split_topology.h"

#include <glog/logging.h>

namespace xllm {
namespace parallel_state {

namespace {

void validate_rank_inputs(int32_t global_rank,
                          int32_t world_size,
                          int32_t kv_split_size) {
  CHECK_GT(world_size, 0) << "world_size must be positive.";
  CHECK_GT(kv_split_size, 0) << "kv_split_size must be positive.";
  CHECK_EQ(world_size % kv_split_size, 0)
      << "world_size (" << world_size
      << ") must be divisible by kv_split_size (" << kv_split_size << ").";
  CHECK_GE(global_rank, 0);
  CHECK_LT(global_rank, world_size);
}

void validate_cp_topology(int32_t cp_size, int32_t kv_split_size) {
  CHECK_GT(cp_size, 0) << "cp_size must be positive.";
  CHECK_GT(kv_split_size, 0) << "kv_split_size must be positive.";
  CHECK_LE(kv_split_size, cp_size)
      << "kv_split_size (" << kv_split_size << ") must not exceed cp_size ("
      << cp_size << ").";
  CHECK_EQ(cp_size % kv_split_size, 0)
      << "cp_size (" << cp_size << ") must be divisible by kv_split_size ("
      << kv_split_size << ").";
}

}  // namespace

int32_t compute_kv_split_rank(int32_t global_rank,
                              int32_t world_size,
                              int32_t kv_split_size) {
  validate_rank_inputs(global_rank, world_size, kv_split_size);
  return global_rank / (world_size / kv_split_size);
}

std::vector<int32_t> compute_kv_split_group_ranks(int32_t global_rank,
                                                  int32_t world_size,
                                                  int32_t kv_split_size) {
  validate_rank_inputs(global_rank, world_size, kv_split_size);
  const int32_t group_count = world_size / kv_split_size;
  const int32_t group_id = global_rank % group_count;

  std::vector<int32_t> ranks;
  ranks.reserve(kv_split_size);
  for (int32_t kv_rank = 0; kv_rank < kv_split_size; ++kv_rank) {
    ranks.emplace_back(group_id + kv_rank * group_count);
  }
  return ranks;
}

int32_t compute_kv_split_rank(int32_t global_rank,
                              int32_t world_size,
                              int32_t cp_size,
                              int32_t kv_split_size) {
  validate_cp_topology(cp_size, kv_split_size);
  return compute_kv_split_rank(global_rank, world_size, kv_split_size);
}

std::vector<int32_t> compute_kv_split_group_ranks(int32_t global_rank,
                                                  int32_t world_size,
                                                  int32_t cp_size,
                                                  int32_t kv_split_size) {
  validate_cp_topology(cp_size, kv_split_size);
  return compute_kv_split_group_ranks(global_rank, world_size, kv_split_size);
}

std::optional<std::string> validate_owner_sharded_kv_execution(
    int32_t kv_split_size,
    bool enable_disagg_pd,
    bool is_prefill_role) {
  if (kv_split_size <= 1) {
    return std::nullopt;
  }
  if (!enable_disagg_pd || !is_prefill_role) {
    return "Owner-sharded persistent KV requires --enable_disagg_pd=true and "
           "--instance_role=PREFILL; standalone Decode cannot reconstruct KV "
           "from multiple Prefill owners";
  }
  return std::nullopt;
}

}  // namespace parallel_state
}  // namespace xllm
