/*
 * Copyright 2025 The xLLM Authors. All Rights Reserved.
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
 *
 * This file includes optimized TopK implementations:
 * - Warp-level TopK for k <= 32 (fastest path)
 * - Radix TopK (derived from TensorRT-LLM) for k > 32 (fast path for k=128)
 * - torch::topk fallback for very large k
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda/std/limits>
#include <limits>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "air_topk_last_dim.h"
#include "air_topk_stable_radix_11bits.cuh"
#include "core/common/global_flags.h"

namespace xllm::kernel::cuda {

namespace {

// ============================================================================
// Constants
// ============================================================================
constexpr int WARP_SIZE = 32;
constexpr unsigned FULL_WARP_MASK = 0xffffffff;

// ============================================================================
// Thread-local Workspace Cache (bytes)
// ============================================================================
struct WorkspaceCache {
  torch::Tensor buffer;
  size_t capacity = 0;
  int device_index = -1;
};

torch::Tensor get_workspace(size_t required_bytes,
                            const torch::Device& device) {
  thread_local std::unordered_map<int, WorkspaceCache> caches;

  int dev_idx = device.index();
  auto& cache = caches[dev_idx];

  if (cache.capacity < required_bytes || cache.device_index != dev_idx) {
    cache.buffer = torch::empty(
        {static_cast<int64_t>(required_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(device));
    cache.capacity = required_bytes;
    cache.device_index = dev_idx;
  }

  return cache.buffer;
}

// ============================================================================
// Thread-local Workspace Size Cache (avoid per-step CUB workspace size query)
// ============================================================================
struct WorkspaceSizeKey {
  int32_t batch;
  int32_t len;
  int32_t k;
  int dtype_code;  // 0=float, 1=bf16, 2=fp16
  bool sorted;

  bool operator==(const WorkspaceSizeKey& other) const {
    return batch == other.batch && len == other.len && k == other.k &&
           dtype_code == other.dtype_code && sorted == other.sorted;
  }
};

struct WorkspaceSizeKeyHash {
  size_t operator()(const WorkspaceSizeKey& key) const {
    // A simple hash combine
    size_t h = std::hash<int32_t>{}(key.batch);
    h ^= std::hash<int32_t>{}(key.len) << 1;
    h ^= std::hash<int32_t>{}(key.k) << 2;
    h ^= std::hash<int>{}(key.dtype_code) << 3;
    h ^= std::hash<bool>{}(key.sorted) << 4;
    return h;
  }
};

static inline std::
    unordered_map<WorkspaceSizeKey, size_t, WorkspaceSizeKeyHash>&
    get_workspace_size_cache() {
  thread_local std::
      unordered_map<WorkspaceSizeKey, size_t, WorkspaceSizeKeyHash>
          size_cache;
  return size_cache;
}

// Cache queried workspace sizes.
size_t get_cached_workspace_size(int32_t batch,
                                 int32_t len,
                                 int32_t k,
                                 int dtype_code,
                                 bool sorted) {
  WorkspaceSizeKey key{batch, len, k, dtype_code, sorted};
  auto& size_cache = get_workspace_size_cache();
  auto it = size_cache.find(key);
  if (it != size_cache.end()) {
    return it->second;  // cache hit
  }

  // Cache miss. Return 0 to trigger a size query.
  return 0;
}

void cache_workspace_size(int32_t batch,
                          int32_t len,
                          int32_t k,
                          int dtype_code,
                          bool sorted,
                          size_t size) {
  WorkspaceSizeKey key{batch, len, k, dtype_code, sorted};
  auto& size_cache = get_workspace_size_cache();
  size_cache[key] = size;
}

// ============================================================================
// Radix TopK helper: with cached workspace size.
// ============================================================================
template <typename T, bool sorted>
void run_radix_topk_with_cache(const T* in_ptr,
                               T* out_val,
                               int32_t* out_idx,
                               int32_t batch,
                               int32_t len,
                               int32_t k,
                               int dtype_code,
                               bool largest,
                               const torch::Device& device,
                               cudaStream_t stream) {
  // Try getting the workspace size from cache.
  size_t workspace_bytes =
      get_cached_workspace_size(batch, len, k, dtype_code, sorted);

  if (workspace_bytes == 0) {
    // Cache miss. Query workspace size.
    standalone_stable_radix_11bits<T, int32_t, sorted>(nullptr,
                                                       workspace_bytes,
                                                       in_ptr,
                                                       batch,
                                                       len,
                                                       k,
                                                       out_val,
                                                       out_idx,
                                                       largest,
                                                       stream,
                                                       /*stable=*/false);
    // Cache the result.
    cache_workspace_size(batch, len, k, dtype_code, sorted, workspace_bytes);
  }

  auto workspace = get_workspace(workspace_bytes, device);
  standalone_stable_radix_11bits<T, int32_t, sorted>(workspace.data_ptr(),
                                                     workspace_bytes,
                                                     in_ptr,
                                                     batch,
                                                     len,
                                                     k,
                                                     out_val,
                                                     out_idx,
                                                     largest,
                                                     stream,
                                                     /*stable=*/false);
}

// ============================================================================
// Thread-local Output Cache (for int32 indices)
// ============================================================================
struct OutputCache {
  torch::Tensor values;
  torch::Tensor indices;
  int64_t batch = 0;
  int64_t k = 0;
  int device_index = -1;
  torch::ScalarType dtype = torch::kFloat32;
};

std::pair<torch::Tensor, torch::Tensor> get_cached_output(
    int64_t batch,
    int64_t k,
    const torch::Device& device,
    torch::ScalarType dtype) {
  thread_local std::unordered_map<int, OutputCache> caches;

  int dev_idx = device.index();
  auto& cache = caches[dev_idx];

  bool need_realloc = (cache.batch < batch || cache.k < k ||
                       cache.device_index != dev_idx || cache.dtype != dtype);

  if (need_realloc) {
    cache.values = torch::empty(
        {batch, k}, torch::TensorOptions().dtype(dtype).device(device));
    // 使用 int32 索引（性能优化！）
    cache.indices = torch::empty(
        {batch, k}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    cache.batch = batch;
    cache.k = k;
    cache.device_index = dev_idx;
    cache.dtype = dtype;
  }

  if (cache.batch == batch && cache.k == k) {
    return {cache.values, cache.indices};
  } else {
    return {cache.values.slice(0, 0, batch).slice(1, 0, k),
            cache.indices.slice(0, 0, batch).slice(1, 0, k)};
  }
}

// ============================================================================
// Helper: Type traits for CUDA scalar types
// ============================================================================
template <typename T>
struct NumericLimits;

template <>
struct NumericLimits<float> {
  __device__ static float lowest() { return -INFINITY; }
  __device__ static float highest() { return INFINITY; }
};

template <>
struct NumericLimits<half> {
  __device__ static half lowest() { return __float2half(-INFINITY); }
  __device__ static half highest() { return __float2half(INFINITY); }
};

template <>
struct NumericLimits<__nv_bfloat16> {
  __device__ static __nv_bfloat16 lowest() {
    return __float2bfloat16(-INFINITY);
  }
  __device__ static __nv_bfloat16 highest() {
    return __float2bfloat16(INFINITY);
  }
};

// ============================================================================
// Helper: Convert to float for comparison
// ============================================================================
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

// ============================================================================
// Warp TopK Kernel: k <= WARP_SIZE (32)
// Each warp processes one batch element using warp shuffle reduction
// ============================================================================

template <typename T, int K>
__device__ __forceinline__ void warp_topk_insert(T* values,
                                                 int32_t* indices,
                                                 T new_val,
                                                 int32_t new_idx,
                                                 bool largest) {
  // 在已排序的数组中插入新元素（保持有序）
  // 使用简单的冒泡插入，因为 K 很小
  float new_val_f = to_float(new_val);

#pragma unroll
  for (int i = K - 1; i >= 0; --i) {
    float cur_val_f = to_float(values[i]);
    bool should_insert =
        largest ? (new_val_f > cur_val_f) : (new_val_f < cur_val_f);

    if (should_insert) {
      // 将当前位置及之后的元素后移
      if (i < K - 1) {
        values[i + 1] = values[i];
        indices[i + 1] = indices[i];
      }
      values[i] = new_val;
      indices[i] = new_idx;
      return;
    }
  }
}

template <typename T, int K>
__device__ __forceinline__ void warp_merge_topk(T* local_values,
                                                int32_t* local_indices,
                                                int lane_id,
                                                bool largest) {
  // Warp-level reduction: 合并所有 lane 的 top-k
  // 使用 butterfly reduction 模式
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < K; ++i) {
      // 从另一个 lane 获取值和索引
      float other_val_f =
          __shfl_xor_sync(FULL_WARP_MASK, to_float(local_values[i]), offset);
      int32_t other_idx =
          __shfl_xor_sync(FULL_WARP_MASK, local_indices[i], offset);

      // 将 float 转回 T 类型
      T other_val;
      if constexpr (std::is_same_v<T, float>) {
        other_val = other_val_f;
      } else if constexpr (std::is_same_v<T, half>) {
        other_val = __float2half(other_val_f);
      } else {
        other_val = __float2bfloat16(other_val_f);
      }

      // 尝试插入
      warp_topk_insert<T, K>(
          local_values, local_indices, other_val, other_idx, largest);
    }
  }
}

template <typename T, int K, int BLOCK_SIZE = 256>
__global__ void warp_topk_kernel(const T* __restrict__ input,
                                 T* __restrict__ out_values,
                                 int32_t* __restrict__ out_indices,
                                 int32_t batch_size,
                                 int32_t len,
                                 bool largest) {
  // 每个 warp 处理一个 batch
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;

  if (warp_id >= batch_size) return;

  const T* batch_input = input + warp_id * len;
  T* batch_out_values = out_values + warp_id * K;
  int32_t* batch_out_indices = out_indices + warp_id * K;

  // 每个 thread 维护本地的 top-K
  T local_values[K];
  int32_t local_indices[K];

  // 初始化为极值
  T init_val =
      largest ? NumericLimits<T>::lowest() : NumericLimits<T>::highest();
#pragma unroll
  for (int i = 0; i < K; ++i) {
    local_values[i] = init_val;
    local_indices[i] = -1;
  }

  // 每个 lane 处理 len / WARP_SIZE 个元素（strided access）
  for (int32_t i = lane_id; i < len; i += WARP_SIZE) {
    T val = batch_input[i];
    warp_topk_insert<T, K>(local_values, local_indices, val, i, largest);
  }

  // Warp-level merge
  warp_merge_topk<T, K>(local_values, local_indices, lane_id, largest);

  // Lane 0 写出结果
  if (lane_id == 0) {
#pragma unroll
    for (int i = 0; i < K; ++i) {
      batch_out_values[i] = local_values[i];
      batch_out_indices[i] = local_indices[i];
    }
  }
}

// 动态 K 版本（K 在运行时确定，但 <= 32）
template <typename T, int BLOCK_SIZE = 256>
__global__ void warp_topk_kernel_dynamic_k(const T* __restrict__ input,
                                           T* __restrict__ out_values,
                                           int32_t* __restrict__ out_indices,
                                           int32_t batch_size,
                                           int32_t len,
                                           int32_t k,
                                           bool largest) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;

  if (warp_id >= batch_size) return;

  const T* batch_input = input + warp_id * len;
  T* batch_out_values = out_values + warp_id * k;
  int32_t* batch_out_indices = out_indices + warp_id * k;

  // 使用 shared memory 存储 top-k 候选
  extern __shared__ char shared_mem[];
  const int warp_in_block = threadIdx.x / WARP_SIZE;
  T* warp_values = reinterpret_cast<T*>(shared_mem) + warp_in_block * WARP_SIZE;
  int32_t* warp_indices =
      reinterpret_cast<int32_t*>(shared_mem + (blockDim.x / WARP_SIZE) *
                                                  WARP_SIZE * sizeof(T)) +
      warp_in_block * WARP_SIZE;

  // 初始化
  T init_val =
      largest ? NumericLimits<T>::lowest() : NumericLimits<T>::highest();
  warp_values[lane_id] = init_val;
  warp_indices[lane_id] = -1;
  __syncwarp();

  // 处理输入数据
  for (int32_t i = lane_id; i < len; i += WARP_SIZE) {
    T val = batch_input[i];
    float val_f = to_float(val);

    // 找到插入位置
    int insert_pos = -1;
    for (int j = k - 1; j >= 0; --j) {
      float cur_val_f = to_float(warp_values[j]);
      bool should_insert = largest ? (val_f > cur_val_f) : (val_f < cur_val_f);
      if (should_insert) {
        insert_pos = j;
      } else {
        break;
      }
    }

    if (insert_pos >= 0) {
      // 需要同步插入（简化版：lane 0 负责所有插入）
      for (int src_lane = 0; src_lane < WARP_SIZE; ++src_lane) {
        float bcast_val_f = __shfl_sync(FULL_WARP_MASK, val_f, src_lane);
        int32_t bcast_idx = __shfl_sync(FULL_WARP_MASK, i, src_lane);
        int bcast_insert = __shfl_sync(FULL_WARP_MASK, insert_pos, src_lane);

        if (lane_id == 0 && bcast_insert >= 0) {
          // 后移元素
          for (int j = k - 1; j > bcast_insert; --j) {
            warp_values[j] = warp_values[j - 1];
            warp_indices[j] = warp_indices[j - 1];
          }
          // 插入新元素
          if constexpr (std::is_same_v<T, float>) {
            warp_values[bcast_insert] = bcast_val_f;
          } else if constexpr (std::is_same_v<T, half>) {
            warp_values[bcast_insert] = __float2half(bcast_val_f);
          } else {
            warp_values[bcast_insert] = __float2bfloat16(bcast_val_f);
          }
          warp_indices[bcast_insert] = bcast_idx;
        }
        __syncwarp();
      }
    }
  }

  // Lane 0 写出结果
  if (lane_id == 0) {
    for (int i = 0; i < k; ++i) {
      batch_out_values[i] = warp_values[i];
      batch_out_indices[i] = warp_indices[i];
    }
  }
}

// ============================================================================
// Simple Block TopK Kernel: 简化版，使用 partial sort
// ============================================================================
template <typename T>
__global__ void block_topk_simple_kernel(const T* __restrict__ input,
                                         T* __restrict__ out_values,
                                         int32_t* __restrict__ out_indices,
                                         int32_t batch_size,
                                         int32_t len,
                                         int32_t k,
                                         bool largest) {
  const int batch_id = blockIdx.x;
  if (batch_id >= batch_size) return;

  const T* batch_input = input + batch_id * len;
  T* batch_out_values = out_values + batch_id * k;
  int32_t* batch_out_indices = out_indices + batch_id * k;

  // 使用 shared memory 存储 top-k 候选
  extern __shared__ char shared_mem[];
  T* topk_values = reinterpret_cast<T*>(shared_mem);
  int32_t* topk_indices =
      reinterpret_cast<int32_t*>(shared_mem + k * sizeof(T));

  // 初始化 top-k
  T init_val =
      largest ? NumericLimits<T>::lowest() : NumericLimits<T>::highest();
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    topk_values[i] = init_val;
    topk_indices[i] = -1;
  }
  __syncthreads();

  // 每个 thread 处理一部分输入
  for (int32_t i = threadIdx.x; i < len; i += blockDim.x) {
    T val = batch_input[i];
    float val_f = to_float(val);

    // 检查是否应该插入（只检查最小的候选）
    float min_topk_f = to_float(topk_values[k - 1]);
    bool should_check = largest ? (val_f > min_topk_f) : (val_f < min_topk_f);

    if (should_check) {
      // 使用原子操作更新 top-k（简化实现）
      // 注意：这不是最优的，但对于中等 k 够用
      for (int j = 0; j < k; ++j) {
        float cur_val_f = to_float(topk_values[j]);
        bool should_insert =
            largest ? (val_f > cur_val_f) : (val_f < cur_val_f);

        if (should_insert) {
          // 原子比较交换（简化版：使用全局内存原子操作）
          // 实际实现中应该使用更复杂的无锁数据结构
          T old_val = topk_values[j];
          int32_t old_idx = topk_indices[j];

          // 简单的串行插入（单线程）
          if (threadIdx.x == 0) {
            for (int jj = k - 1; jj > j; --jj) {
              topk_values[jj] = topk_values[jj - 1];
              topk_indices[jj] = topk_indices[jj - 1];
            }
            topk_values[j] = val;
            topk_indices[j] = i;
          }
          __syncthreads();
          break;
        }
      }
    }
  }
  __syncthreads();

  // 写出结果
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    batch_out_values[i] = topk_values[i];
    batch_out_indices[i] = topk_indices[i];
  }
}

// ============================================================================
// Dispatch Warp TopK by K value (compile-time specialization)
// ============================================================================
template <typename T>
void dispatch_warp_topk(const T* input,
                        T* out_values,
                        int32_t* out_indices,
                        int32_t batch_size,
                        int32_t len,
                        int32_t k,
                        bool largest,
                        cudaStream_t stream) {
  // 每个 warp 处理一个 batch，计算需要多少 warps
  const int warps_needed = batch_size;
  const int threads_per_block = 256;
  const int warps_per_block = threads_per_block / WARP_SIZE;
  const int num_blocks = (warps_needed + warps_per_block - 1) / warps_per_block;

  // 根据 K 值选择模板特化版本（编译时优化）
  switch (k) {
    case 1:
      warp_topk_kernel<T, 1, threads_per_block>
          <<<num_blocks, threads_per_block, 0, stream>>>(
              input, out_values, out_indices, batch_size, len, largest);
      break;
    case 2:
      warp_topk_kernel<T, 2, threads_per_block>
          <<<num_blocks, threads_per_block, 0, stream>>>(
              input, out_values, out_indices, batch_size, len, largest);
      break;
    case 4:
      warp_topk_kernel<T, 4, threads_per_block>
          <<<num_blocks, threads_per_block, 0, stream>>>(
              input, out_values, out_indices, batch_size, len, largest);
      break;
    case 8:
      warp_topk_kernel<T, 8, threads_per_block>
          <<<num_blocks, threads_per_block, 0, stream>>>(
              input, out_values, out_indices, batch_size, len, largest);
      break;
    case 16:
      warp_topk_kernel<T, 16, threads_per_block>
          <<<num_blocks, threads_per_block, 0, stream>>>(
              input, out_values, out_indices, batch_size, len, largest);
      break;
    case 20:
      warp_topk_kernel<T, 20, threads_per_block>
          <<<num_blocks, threads_per_block, 0, stream>>>(
              input, out_values, out_indices, batch_size, len, largest);
      break;
    case 32:
      warp_topk_kernel<T, 32, threads_per_block>
          <<<num_blocks, threads_per_block, 0, stream>>>(
              input, out_values, out_indices, batch_size, len, largest);
      break;
    default: {
      // 对于其他 K 值，使用动态版本 + shared memory
      const int smem_size =
          warps_per_block * WARP_SIZE * (sizeof(T) + sizeof(int32_t));
      warp_topk_kernel_dynamic_k<T, threads_per_block>
          <<<num_blocks, threads_per_block, smem_size, stream>>>(
              input, out_values, out_indices, batch_size, len, k, largest);
      break;
    }
  }
}

// ============================================================================
// Fallback: Use torch::topk for large k (> warp_topk_threshold)
// This is a simple and robust fallback that leverages PyTorch's optimized impl
// ============================================================================
std::tuple<torch::Tensor, torch::Tensor> torch_topk_fallback(
    const torch::Tensor& input,
    int32_t k,
    bool largest,
    bool sorted) {
  // 使用 PyTorch 的 topk，然后转换索引为 int32
  auto [values, indices_i64] = input.topk(k, /*dim=*/-1, largest, sorted);

  // 转换 int64 -> int32
  auto indices = indices_i64.to(torch::kInt32);

  return {values, indices};
}

}  // namespace

// ============================================================================
// Public API: air_topk_last_dim
// ============================================================================
std::tuple<torch::Tensor, torch::Tensor> air_topk_last_dim(
    const torch::Tensor& input,
    int32_t k,
    bool largest,
    bool sorted_by_value) {
  TORCH_CHECK(input.is_cuda(), "air_topk_last_dim: input must be CUDA");
  TORCH_CHECK(input.dim() == 2, "air_topk_last_dim: input must be 2D [B, L]");
  TORCH_CHECK(k > 0, "air_topk_last_dim: k must be > 0");

  const int64_t batch64 = input.size(0);
  const int64_t len64 = input.size(1);
  TORCH_CHECK(batch64 >= 0 && batch64 <= INT32_MAX,
              "air_topk_last_dim: batch too large");
  TORCH_CHECK(len64 > 0 && len64 <= INT32_MAX,
              "air_topk_last_dim: len too large");
  TORCH_CHECK(k <= len64, "air_topk_last_dim: k must be <= len");

  const int32_t batch = static_cast<int32_t>(batch64);
  const int32_t len = static_cast<int32_t>(len64);

  c10::cuda::CUDAGuard device_guard(input.device());
  auto in = input.contiguous();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // 根据 k 选择算法
  const int32_t warp_threshold = FLAGS_warp_topk_threshold;

  if (k <= warp_threshold && k <= WARP_SIZE) {
    // 快速路径：Warp TopK
    auto [values, indices] = get_cached_output(
        batch64, static_cast<int64_t>(k), in.device(), in.scalar_type());

    const auto dtype = in.scalar_type();
    if (dtype == torch::kFloat32) {
      dispatch_warp_topk<float>(in.data_ptr<float>(),
                                values.data_ptr<float>(),
                                indices.data_ptr<int32_t>(),
                                batch,
                                len,
                                k,
                                largest,
                                stream);
    } else if (dtype == torch::kBFloat16) {
      dispatch_warp_topk<__nv_bfloat16>(
          reinterpret_cast<const __nv_bfloat16*>(in.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(values.data_ptr<at::BFloat16>()),
          indices.data_ptr<int32_t>(),
          batch,
          len,
          k,
          largest,
          stream);
    } else if (dtype == torch::kFloat16) {
      dispatch_warp_topk<half>(
          reinterpret_cast<const half*>(in.data_ptr<at::Half>()),
          reinterpret_cast<half*>(values.data_ptr<at::Half>()),
          indices.data_ptr<int32_t>(),
          batch,
          len,
          k,
          largest,
          stream);
    } else {
      TORCH_CHECK(false, "air_topk_last_dim: unsupported dtype");
    }

    return {values, indices};
  }

  const int32_t large_k_threshold = FLAGS_air_topk_large_k_threshold;
  if (k <= large_k_threshold) {
    auto [values, indices] = get_cached_output(
        batch64, static_cast<int64_t>(k), in.device(), in.scalar_type());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const auto dtype = in.scalar_type();
    if (dtype == torch::kFloat32) {
      using T = float;
      auto* out_val = reinterpret_cast<T*>(values.data_ptr<float>());
      auto* out_idx = reinterpret_cast<int32_t*>(indices.data_ptr<int32_t>());
      const auto* in_ptr = reinterpret_cast<const T*>(in.data_ptr<float>());

      constexpr int dtype_code = 0;  // float
      if (sorted_by_value) {
        run_radix_topk_with_cache<T, true>(in_ptr,
                                           out_val,
                                           out_idx,
                                           batch,
                                           len,
                                           k,
                                           dtype_code,
                                           largest,
                                           in.device(),
                                           stream);
      } else {
        run_radix_topk_with_cache<T, false>(in_ptr,
                                            out_val,
                                            out_idx,
                                            batch,
                                            len,
                                            k,
                                            dtype_code,
                                            largest,
                                            in.device(),
                                            stream);
      }

      return {values, indices};
    }

    if (dtype == torch::kBFloat16) {
      using T = __nv_bfloat16;
      auto* out_val = reinterpret_cast<T*>(values.data_ptr<at::BFloat16>());
      auto* out_idx = reinterpret_cast<int32_t*>(indices.data_ptr<int32_t>());
      const auto* in_ptr =
          reinterpret_cast<const T*>(in.data_ptr<at::BFloat16>());

      constexpr int dtype_code = 1;  // bf16
      if (sorted_by_value) {
        run_radix_topk_with_cache<T, true>(in_ptr,
                                           out_val,
                                           out_idx,
                                           batch,
                                           len,
                                           k,
                                           dtype_code,
                                           largest,
                                           in.device(),
                                           stream);
      } else {
        run_radix_topk_with_cache<T, false>(in_ptr,
                                            out_val,
                                            out_idx,
                                            batch,
                                            len,
                                            k,
                                            dtype_code,
                                            largest,
                                            in.device(),
                                            stream);
      }

      return {values, indices};
    }

    if (dtype == torch::kFloat16) {
      using T = half;
      auto* out_val = reinterpret_cast<T*>(values.data_ptr<at::Half>());
      auto* out_idx = reinterpret_cast<int32_t*>(indices.data_ptr<int32_t>());
      const auto* in_ptr = reinterpret_cast<const T*>(in.data_ptr<at::Half>());

      constexpr int dtype_code = 2;  // fp16
      if (sorted_by_value) {
        run_radix_topk_with_cache<T, true>(in_ptr,
                                           out_val,
                                           out_idx,
                                           batch,
                                           len,
                                           k,
                                           dtype_code,
                                           largest,
                                           in.device(),
                                           stream);
      } else {
        run_radix_topk_with_cache<T, false>(in_ptr,
                                            out_val,
                                            out_idx,
                                            batch,
                                            len,
                                            k,
                                            dtype_code,
                                            largest,
                                            in.device(),
                                            stream);
      }

      return {values, indices};
    }

    TORCH_CHECK(false, "air_topk_last_dim: unsupported dtype");
    return {values, indices};
  }

  // Fallback 路径：使用 torch::topk（例如超大 k）
  return torch_topk_fallback(in, k, largest, sorted_by_value);
}

}  // namespace xllm::kernel::cuda
