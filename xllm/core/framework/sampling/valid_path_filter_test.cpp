#include "valid_path_filter.h"

#include <ATen/ops/equal.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorImpl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <fstream>
#include <iostream>

namespace xllm {

TEST(ValidPathFilterTest, Vector) {
  // 修改为3个元素的序列，符合实现的期望
  std::vector<std::vector<int32_t>> tokens_list = {{1, 2, 3},
                                                   {1, 2, 2},
                                                   {1, 3, 3},
                                                   {1, 4, 4},
                                                   {2, 2, 3},
                                                   {3, 2, 3}};
  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(torch::kCPU);
  int32_t vocab_size = 5;
  ValidPathFilter filter =
      ValidPathFilter(tokens_list, vocab_size, dtype, device);
  std::vector<std::vector<int32_t>> candidate_tokens = {{1, 2, 3}, {1}, {}};

  const auto options = torch::dtype(dtype).device(device);

  torch::Tensor mask = filter.forward(candidate_tokens);

  EXPECT_EQ(mask.sizes(),
            torch::IntArrayRef({candidate_tokens.size(), vocab_size}));

  // 重新分析mask逻辑：
  // tokens_list = {{1,2,3}, {1,2,2}, {1,3,3}, {1,4,4}, {2,2,3}, {3,2,3}}
  // candidate_tokens = {{1,2,3}, {1}, {}}
  //
  // 对于 {1,2,3}: 查找以[1,2,3]开头的序列，找到{1,2,3}，下一个token无(序列结束)
  // 对于 {1}: 查找以[1]开头的序列，找到{1,2,3},{1,2,2},{1,3,3},{1,4,4}，下一个token是2,2,3,4
  // 对于 {}: 第一个token，从所有序列的第一个token：1,1,1,1,2,3
  std::vector<std::vector<int32_t>> desired_mask_vec = {
      {-10000, -10000, -10000, -10000, -10000},  // [1,2,3]后没有下一个token，全部mask
      {-10000, -10000, 0, 0, 0},                 // [1]后可以是2,3,4
      {-10000, 0, 0, 0, -10000}};                // []时第一个token可以是1,2,3
  for (int i = 0; i < candidate_tokens.size(); i++) {
    torch::Tensor desired_mask = torch::tensor(desired_mask_vec[i], options);
    EXPECT_TRUE(torch::equal(mask[i], desired_mask));
  }
}

TEST(ValidPathFilterTest, File) {
  // 修改为3个元素的序列，符合实现的期望
  std::vector<std::vector<int32_t>> tokens_list = {{1, 2, 3},
                                                   {1, 2, 2},
                                                   {1, 3, 3},
                                                   {1, 4, 4},
                                                   {2, 2, 3},
                                                   {3, 2, 3}};
  const std::string rec_tokens_file = "./test_data.bin";
  if (std::ifstream(rec_tokens_file)) {
    std::remove(rec_tokens_file.c_str());
  }
  std::ofstream outfile(rec_tokens_file, std::ios::binary);
  if (!outfile) {
    LOG(INFO) << " Fail to create : " << rec_tokens_file;
    return;
  }
  // 按照实现期望的格式写入：int64_t item_id + 3个int32_t
  int64_t item_id = 0;
  for (const auto& row : tokens_list) {
    outfile.write(reinterpret_cast<const char*>(&item_id), sizeof(int64_t));
    outfile.write(reinterpret_cast<const char*>(row.data()),
                  row.size() * sizeof(int32_t));
    item_id++;
  }
  outfile.close();

  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(torch::kCPU);
  int32_t vocab_size = 5;
  ValidPathFilter filter =
      ValidPathFilter(rec_tokens_file, vocab_size, dtype, device);
  std::vector<std::vector<int32_t>> candidate_tokens = {{1, 2, 3}, {1}, {}};

  const auto options = torch::dtype(dtype).device(device);

  torch::Tensor mask = filter.forward(candidate_tokens);

  EXPECT_EQ(mask.sizes(),
            torch::IntArrayRef({candidate_tokens.size(), vocab_size}));

  // 使用与Vector测试相同的期望mask
  std::vector<std::vector<int32_t>> desired_mask_vec = {
      {-10000, -10000, -10000, -10000, -10000},  // [1,2,3]后没有下一个token，全部mask
      {-10000, -10000, 0, 0, 0},                 // [1]后可以是2,3,4
      {-10000, 0, 0, 0, -10000}};                // []时第一个token可以是1,2,3
  for (int i = 0; i < candidate_tokens.size(); i++) {
    torch::Tensor desired_mask = torch::tensor(desired_mask_vec[i], options);
    EXPECT_TRUE(torch::equal(mask[i], desired_mask));
  }
  if (std::ifstream(rec_tokens_file)) {
    std::remove(rec_tokens_file.c_str());
  }
}

}  // namespace xllm
