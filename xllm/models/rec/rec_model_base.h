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

#include <glog/logging.h>
#include <torch/torch.h>

#include <cmath>
#include <memory>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/model_loader.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/common/word_embedding.h"

namespace xllm {

template <typename ModelType>
class RecForCausalLMImplBase : public torch::nn::Module {
 public:
  explicit RecForCausalLMImplBase(const ModelContext& context)
      : context_(context) {
    const auto& args = context.get_model_args();
    tie_word_embeddings_ = args.tie_word_embeddings();
    const float denom =
        std::sqrt(static_cast<float>(std::max<int64_t>(1, args.hidden_size())));
    scale_factor_ = denom > 0.0f ? (1.0f / denom) : 1.0f;

    model_ = register_module("model", ModelType(context));
    lm_head_ = register_module("lm_head", layer::LmHead(context));
  }

  virtual ModelOutput forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) {
    return model_->forward(tokens, positions, kv_caches, input_params);
  }

  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& selected_idxes) {
    auto h = hidden_states;
    if (tie_word_embeddings_) {
      h = hidden_states * scale_factor_;
    }
    if (selected_idxes.defined()) {
      h = h.index_select(/*dim=*/0, selected_idxes);
    }
    return lm_head_(h);
  }

  // OneRec split lm_head: at decode step N, use only lm_head_segments_[N]
  // (shape [seg_width_N, hidden]) to compute that segment's logits, then
  // scatter back into a full [rows, vocab_size] tensor at seg_offsets_[N]
  // (other positions filled with a very negative value). This keeps the full
  // 25000-wide, global-token-id contract for all downstream consumers
  // (sampler / beam search / constrained topk / embedding) while cutting the
  // matmul to ~1/num_segments. Falls back to the plain full-vocab logits when
  // the model has no split heads or step is out of range.
  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& selected_idxes,
                               int32_t step) {
    if (!use_split_lm_head_ || step < 0 ||
        step >= static_cast<int32_t>(lm_head_segments_.size())) {
      return logits(hidden_states, selected_idxes);
    }
    auto h = hidden_states;
    if (tie_word_embeddings_) {
      h = hidden_states * scale_factor_;
    }
    if (selected_idxes.defined()) {
      h = h.index_select(/*dim=*/0, selected_idxes);
    }
    auto seg = lm_head_segments_[step](h);  // [rows, seg_width_step]
    const int64_t rows = seg.size(0);
    auto full = torch::full(
        {rows, full_vocab_size_}, kSplitLmHeadNegInf, seg.options());
    full.slice(/*dim=*/1,
               /*start=*/seg_offsets_[step],
               /*end=*/seg_offsets_[step] + seg_widths_[step]) = seg;
    return full;
  }

  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& selected_idxes) {
    if (selected_idxes.defined()) {
      return hidden_states.index_select(/*dim=*/0, selected_idxes);
    }
    return hidden_states;
  }

  virtual void load_model(std::unique_ptr<ModelLoader> loader,
                          std::string prefix = "model.") {
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix(prefix));
      if (tie_word_embeddings_) {
        lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix(prefix + "shared."));
      } else {
        lm_head_->load_state_dict(state_dict->get_dict_with_prefix("lm_head."));
      }
    }
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    (void)layer_id;
    (void)expert_ids;
  }

  virtual void update_expert_weight(int32_t layer_id) { (void)layer_id; }

  virtual layer::LmHead get_lm_head() { return lm_head_; }

  virtual void set_lm_head(layer::LmHead& head) { lm_head_ = head; }

  // Detect split lm_head weights (lm_head_0/1/2/...) in the checkpoint and, if
  // present, build one LmHead per segment (each [seg_width, hidden]) and load
  // its weight. Segment count / widths are inferred from the weights; offsets
  // are the prefix sum. Returns true if split heads were built. Called by the
  // model's load_model before falling back to the single-head path.
  bool try_build_split_lm_head(const StateDict& state_dict) {
    // Collect consecutive lm_head_<i>.weight entries.
    std::vector<torch::Tensor> seg_weights;
    for (int32_t i = 0;; ++i) {
      const std::string key = "lm_head_" + std::to_string(i) + ".weight";
      auto t = state_dict.get_tensor(key);
      if (!t.defined()) {
        break;
      }
      seg_weights.push_back(t);
    }
    if (seg_weights.empty()) {
      return false;
    }

    full_vocab_size_ = context_.get_model_args().vocab_size();
    seg_widths_.clear();
    seg_offsets_.clear();
    lm_head_segments_.clear();
    int64_t offset = 0;
    for (size_t i = 0; i < seg_weights.size(); ++i) {
      const int64_t width = seg_weights[i].size(0);
      seg_widths_.push_back(width);
      seg_offsets_.push_back(offset);
      offset += width;
      auto head = register_module("lm_head_" + std::to_string(i),
                                  layer::LmHead(context_, width));
      const std::string prefix = "lm_head_" + std::to_string(i) + ".";
      head->load_state_dict(state_dict.get_dict_with_prefix(prefix));
      lm_head_segments_.push_back(head);
    }
    CHECK_LE(offset, full_vocab_size_)
        << "OneRec split lm_head total width " << offset
        << " exceeds vocab_size " << full_vocab_size_;
    use_split_lm_head_ = true;
    LOG(INFO) << "OneRec split lm_head enabled: " << seg_weights.size()
              << " segments, widths sum=" << offset
              << ", vocab_size=" << full_vocab_size_;
    return true;
  }

  virtual layer::WordEmbedding get_word_embedding() {
    return model_->get_word_embedding();
  }

  virtual void set_word_embedding(layer::WordEmbedding& embedding) {
    model_->set_word_embedding(embedding);
  }

 protected:
  float scale_factor_ = 1.0f;
  bool tie_word_embeddings_ = false;
  ModelContext context_;

  // OneRec split lm_head state. When use_split_lm_head_ is true, per-step heads
  // in lm_head_segments_ replace the single lm_head_ for logits computation.
  // Segment count / widths / offsets are inferred from the checkpoint weights
  // (not hardcoded); offsets are the prefix sum of widths. full_vocab_size_ is
  // the scatter target width (= config vocab_size).
  static constexpr float kSplitLmHeadNegInf = -1e30f;
  bool use_split_lm_head_ = false;
  int64_t full_vocab_size_ = 0;
  std::vector<layer::LmHead> lm_head_segments_;
  std::vector<int64_t> seg_widths_;
  std::vector<int64_t> seg_offsets_;

  ModelType model_{nullptr};
  layer::LmHead lm_head_{nullptr};
};

}  // namespace xllm
