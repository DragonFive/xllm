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

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace xllm {
namespace parallel_state {

// Return the split-major owner rank for a global rank. The split-major layout
// is rank = group_id + kv_split_rank * group_count, where group_count is
// world_size / kv_split_size.
int32_t compute_kv_split_rank(int32_t global_rank,
                              int32_t world_size,
                              int32_t kv_split_size);

// Return the global ranks in the KV-split group containing global_rank,
// ordered by KV-split rank.
std::vector<int32_t> compute_kv_split_group_ranks(int32_t global_rank,
                                                  int32_t world_size,
                                                  int32_t kv_split_size);

// Validate the CP/KV topology before applying the split-major mapping.
int32_t compute_kv_split_rank(int32_t global_rank,
                              int32_t world_size,
                              int32_t cp_size,
                              int32_t kv_split_size);

std::vector<int32_t> compute_kv_split_group_ranks(int32_t global_rank,
                                                  int32_t world_size,
                                                  int32_t cp_size,
                                                  int32_t kv_split_size);

// Persistent KV split across Prefill owners cannot be consumed by standalone
// Decode. Require the production PD Prefill path whenever KV is owner-sharded.
std::optional<std::string> validate_owner_sharded_kv_execution(
    int32_t kv_split_size,
    bool enable_disagg_pd,
    bool is_prefill_role);

}  // namespace parallel_state
}  // namespace xllm
