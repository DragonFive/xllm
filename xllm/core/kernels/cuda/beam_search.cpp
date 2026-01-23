#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "air_topk_last_dim.h"
#include "common/global_flags.h"
#include "cuda.h"

namespace xllm::kernel::cuda {

namespace {
// Thread-local cache for torch::arange results to avoid repeated allocations
// This is critical for CUDA Graph compatibility
struct ArangeCache {
  torch::Tensor cached_tensor;
  int32_t cached_size = 0;
  torch::ScalarType cached_dtype = torch::kInt32;
  torch::Device cached_device = torch::kCPU;

  void clear() {
    cached_tensor = torch::Tensor();
    cached_size = 0;
    cached_dtype = torch::kInt32;
    cached_device = torch::kCPU;
  }
};

thread_local ArangeCache g_arange_cache;

// Get cached arange tensor or create new one if cache miss
torch::Tensor get_cached_arange(int32_t size,
                                torch::ScalarType dtype,
                                const torch::Device& device) {
  // Check if cache is valid
  if (g_arange_cache.cached_size >= size &&
      g_arange_cache.cached_dtype == dtype &&
      g_arange_cache.cached_device == device &&
      g_arange_cache.cached_tensor.defined()) {
    // Return slice of cached tensor
    return g_arange_cache.cached_tensor.slice(/*dim=*/0,
                                              /*start=*/0,
                                              /*end=*/size);
  }

  // Cache miss or insufficient size - create new tensor
  // Allocate slightly larger size to reduce future reallocations
  const int32_t alloc_size = ((size + 15) / 16) * 16;  // Round up to 16
  g_arange_cache.cached_tensor = torch::arange(
      alloc_size, torch::TensorOptions().dtype(dtype).device(device));
  g_arange_cache.cached_size = alloc_size;
  g_arange_cache.cached_dtype = dtype;
  g_arange_cache.cached_device = device;

  // Return slice of newly created tensor
  return g_arange_cache.cached_tensor.slice(/*dim=*/0,
                                            /*start=*/0,
                                            /*end=*/size);
}
}  // namespace

void beam_search(torch::Tensor acc_logprob,
                 torch::Tensor in_sequence_group,
                 torch::Tensor top_tokens,
                 torch::Tensor top_logprobs,
                 torch::Tensor out_acc_logprob,
                 torch::Tensor out_token_ids,
                 torch::Tensor out_token_index,
                 torch::Tensor out_beam_count_prefix_sums,
                 torch::Tensor out_sequence_group,
                 uint32_t batch_size,
                 uint32_t current_step) {
  torch::Device device = acc_logprob.device();

  uint32_t beam_size = in_sequence_group.size(1);

  uint32_t top_k = top_tokens.size(1);
  uint32_t total_rounds = in_sequence_group.size(2);

  CHECK_EQ(beam_size, top_k) << "beam_size must be equal with top_k.";

  if (current_step == 0) {
    auto tokens_view =
        top_tokens.view({batch_size, top_k}).slice(1, 0, beam_size);
    auto init_probs_view =
        top_logprobs.view({batch_size, top_k}).slice(1, 0, beam_size);

    out_token_ids.view({batch_size, beam_size}).copy_(tokens_view);
    out_acc_logprob.view({batch_size, beam_size}).copy_(init_probs_view);

    // Use cached arange instead of torch::arange for CUDA Graph compatibility
    auto indices = get_cached_arange(beam_size, torch::kInt32, device)
                       .unsqueeze(0)
                       .expand({batch_size, -1})
                       .reshape({-1, 1});
    out_token_index.copy_(indices);

    auto sequence_view =
        out_sequence_group.view({batch_size, beam_size, total_rounds});
    sequence_view.slice(2, 0, 1).squeeze(2).copy_(tokens_view);

  } else {
    auto combined_probs =
        (acc_logprob + top_logprobs).view({batch_size, beam_size * top_k});

    const bool sorted_by_value = FLAGS_enable_topk_sorted;
    torch::Tensor new_probs;
    torch::Tensor new_indices;
    if (FLAGS_enable_air_topk && combined_probs.is_cuda()) {
      std::tie(new_probs, new_indices) =
          air_topk_last_dim(combined_probs,
                            static_cast<int32_t>(beam_size),
                            /*largest=*/true,
                            /*sorted_by_value=*/sorted_by_value);
    } else {
      auto topk_result = torch::topk(combined_probs,
                                     beam_size,
                                     /*dim=*/-1,
                                     /*largest=*/true,
                                     /*sorted=*/sorted_by_value);
      new_probs = std::get<0>(topk_result);    // [batch_size, beam_size]
      new_indices = std::get<1>(topk_result);  // [batch_size, beam_size]
    }

    // cache_select performs an in-place two-pass copy and assumes a safe
    // ordering of beam_index:
    // - cache_select is called in intermediate steps (current_step <
    //   total_rounds - 1)
    // - reorder new_indices (beam_index) by ascending index to avoid
    //   overwriting a source beam that is still needed later
    // The last step does not need cache_select, so keep the topk output order
    // (optionally sorted by value).
    if (current_step < total_rounds - 1) {
      auto ordered_indices =
          new_indices.argsort(static_cast<int64_t>(1), /*descending=*/false);
      new_probs = new_probs.gather(1, ordered_indices);
      new_indices = new_indices.gather(1, ordered_indices);
    }

    const auto top_k_i64 = static_cast<int64_t>(top_k);
    auto new_indices_i64 = new_indices.to(torch::kLong);
    // NOTE: In some PyTorch versions/configurations, `/` may perform
    // true_divide and return a floating tensor. Using it as an
    // advanced-indexing tensor triggers:
    // "tensors used as indices must be long, int, byte or bool tensors".
    // We need explicit integer division here; trunc mode is sufficient since
    // indices are non-negative.
    auto parent_beam = torch::div(new_indices_i64, top_k_i64, "trunc");
    auto token_in_beam = (new_indices_i64 % top_k_i64);

    auto top_tokens_reshaped = top_tokens.view({batch_size, beam_size, top_k});

    // Use cached arange instead of torch::arange for CUDA Graph compatibility
    auto batch_idx = get_cached_arange(batch_size, torch::kLong, device)
                         .unsqueeze(1)
                         .expand_as(parent_beam);

    using torch::indexing::TensorIndex;
    auto new_tokens = top_tokens_reshaped.index({TensorIndex(batch_idx),
                                                 TensorIndex(parent_beam),
                                                 TensorIndex(token_in_beam)});

    out_acc_logprob.view({batch_size, beam_size}).copy_(new_probs);
    out_token_index.view({batch_size, beam_size})
        .copy_(new_indices.to(torch::kInt32));
    out_token_ids.view({batch_size, beam_size}).copy_(new_tokens);

    // Use cached arange instead of torch::arange for CUDA Graph compatibility
    auto batch_range = get_cached_arange(batch_size, torch::kLong, device)
                           .unsqueeze(1)
                           .expand({-1, beam_size});

    using torch::indexing::Slice;
    using torch::indexing::TensorIndex;
    out_sequence_group.slice(2, 0, current_step) =
        in_sequence_group.index({TensorIndex(batch_range),
                                 TensorIndex(parent_beam),
                                 Slice(0, current_step)});

    out_sequence_group.slice(2, current_step, current_step + 1) =
        new_tokens.unsqueeze(2);
  }
}

}  // namespace xllm::kernel::cuda
