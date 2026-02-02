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

#include "core/platform/shared_vmm_allocator.h"

#include <gtest/gtest.h>

#include "core/platform/device.h"

#if defined(USE_CUDA) || defined(USE_ILU)
#include "core/platform/vmm_torch_allocator.h"
#endif

namespace {

bool HasDevice() { return xllm::Device::device_count() > 0; }

void InitDevice() {
  xllm::Device device(0);
  device.set_device();
  device.init_device_context();
}

}  // namespace

TEST(SharedVMMAllocatorTest, BasicAllocateAndSwitch) {
#if defined(USE_CUDA) || defined(USE_ILU) || defined(USE_NPU) || \
    defined(USE_MLU)
  if (!HasDevice()) {
    GTEST_SKIP() << "No accelerator device available";
  }

  InitDevice();
  xllm::SharedVMMAllocator allocator;
  const size_t reserve_size = 64 * 1024 * 1024;
  allocator.init(/*device_id=*/0, reserve_size);

  EXPECT_TRUE(allocator.is_initialized());
  EXPECT_GE(allocator.reserved_size(), reserve_size);
  EXPECT_EQ(allocator.current_offset(), 0u);

  void* first = allocator.allocate(1024 * 1024);
  EXPECT_NE(first, nullptr);
  allocator.deallocate(first);

  const size_t offset_after_first = allocator.current_offset();
  EXPECT_GT(offset_after_first, 0u);
  EXPECT_GE(allocator.mapped_size(), offset_after_first);

  void* second = allocator.allocate(1024 * 1024);
  EXPECT_NE(second, nullptr);
  EXPECT_NE(first, second);
  EXPECT_GE(allocator.current_offset(), offset_after_first);
  EXPECT_GE(allocator.high_water_mark(), allocator.current_offset());

  allocator.switch_to_new_virtual_space();
  EXPECT_EQ(allocator.current_offset(), 0u);

  void* third = allocator.allocate(1024 * 1024);
  EXPECT_NE(third, nullptr);
  EXPECT_NE(third, first);
#else
  GTEST_SKIP() << "VMM allocator not supported in this build";
#endif
}

#if defined(USE_CUDA) || defined(USE_ILU)
TEST(VMMTorchAllocatorTest, RawAllocAndDelete) {
  if (!HasDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }

  InitDevice();
  xllm::SharedVMMAllocator allocator;
  allocator.init(/*device_id=*/0, 64 * 1024 * 1024);
  xllm::VMMTorchAllocator torch_allocator(&allocator);

  void* first = torch_allocator.raw_alloc(1024 * 1024);
  EXPECT_NE(first, nullptr);
  torch_allocator.raw_delete(first);

  cudaStream_t stream = nullptr;
  void* second = torch_allocator.raw_alloc_with_stream(1024 * 1024, stream);
  EXPECT_NE(second, nullptr);
  torch_allocator.raw_delete(second);
}
#endif
