/* Copyright 2025 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================*/

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <tuple>

namespace xllm::kernel::cuda {

// TopK selection on the last dimension for 2D input [B, L].
//
// Algorithm selection:
// - k <= min(FLAGS_warp_topk_threshold, 32): Warp-level reduction (fastest)
// - 32 < k <= FLAGS_air_topk_large_k_threshold (default 128): AIR radix TopK
// - k > FLAGS_air_topk_large_k_threshold: Fallback to torch::topk
//
// If sorted_by_value is true, outputs are sorted by value (descending for
// largest=true, ascending for largest=false).
// Otherwise, outputs are in an unspecified order.
//
// Note: stable ordering for equal values is NOT guaranteed (performance first).
//
// Returns:
// - values: [B, k], same dtype as input
// - indices: [B, k], int32_t (NOT int64_t for better performance!)
std::tuple<torch::Tensor, torch::Tensor> air_topk_last_dim(
    const torch::Tensor& input,
    int32_t k,
    bool largest = true,
    bool sorted_by_value = false);

}  // namespace xllm::kernel::cuda
