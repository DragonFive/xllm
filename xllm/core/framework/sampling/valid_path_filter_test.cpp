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
  std::vector<std::vector<int32_t>> tokens_list = {{1, 2, 3, 4},
                                                   {1, 2, 3, 2},
                                                   {1, 3, 4, 3},
                                                   {1, 4, 3, 4},
                                                   {2, 2, 3, 3},
                                                   {3, 2, 3, 3}};
  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(torch::kCPU);
  int32_t vocab_size = 5;
  ValidPathFilter filter =
      ValidPathFilter(tokens_list, 0, vocab_size, dtype, device);
  std::vector<std::vector<int32_t>> candidate_tokens = {{1, 2, 3}, {1}, {}};

  const auto options = torch::dtype(dtype).device(device);

  torch::Tensor mask = filter.forward(candidate_tokens);

  EXPECT_EQ(mask.sizes(),
            torch::IntArrayRef({candidate_tokens.size(), vocab_size}));

  std::vector<std::vector<int32_t>> desired_mask_vec = {
      {-10000, -10000, 0, -10000, 0},
      {-10000, -10000, 0, 0, 0},
      {-10000, 0, 0, 0, -10000}};
  for (int i = 0; i < candidate_tokens.size(); i++) {
    torch::Tensor desired_mask = torch::tensor(desired_mask_vec[i], options);
    EXPECT_TRUE(torch::equal(mask[i], desired_mask));
  }
}

TEST(ValidPathFilterTest, File) {
  std::vector<std::vector<int32_t>> tokens_list = {{1, 2, 3, 4},
                                                   {1, 2, 3, 2},
                                                   {1, 3, 4, 3},
                                                   {1, 4, 3, 4},
                                                   {2, 2, 3, 3},
                                                   {3, 2, 3, 3}};
  const std::string rec_tokens_file = "./test_data.bin";
  if (std::ifstream(rec_tokens_file)) {
    std::remove(rec_tokens_file.c_str());
  }
  std::ofstream outfile(rec_tokens_file, std::ios::binary);
  if (!outfile) {
    LOG(INFO) << " Fail to create : " << rec_tokens_file;
    return;
  }
  for (const auto& row : tokens_list) {
    outfile.write(reinterpret_cast<const char*>(row.data()),
                  row.size() * sizeof(int32_t));
  }
  outfile.close();

  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(torch::kCPU);
  int32_t vocab_size = 5;
  ValidPathFilter filter =
      ValidPathFilter(rec_tokens_file, 0, vocab_size, dtype, device);
  std::vector<std::vector<int32_t>> candidate_tokens = {{1, 2, 3}, {1}, {}};

  const auto options = torch::dtype(dtype).device(device);

  torch::Tensor mask = filter.forward(candidate_tokens);

  EXPECT_EQ(mask.sizes(),
            torch::IntArrayRef({candidate_tokens.size(), vocab_size}));

  std::vector<std::vector<int32_t>> desired_mask_vec = {
      {-10000, -10000, 0, -10000, 0},
      {-10000, -10000, 0, 0, 0},
      {-10000, 0, 0, 0, -10000}};
  for (int i = 0; i < candidate_tokens.size(); i++) {
    torch::Tensor desired_mask = torch::tensor(desired_mask_vec[i], options);
    EXPECT_TRUE(torch::equal(mask[i], desired_mask));
  }
  if (std::ifstream(rec_tokens_file)) {
    std::remove(rec_tokens_file.c_str());
  }
}

}  // namespace xllm
