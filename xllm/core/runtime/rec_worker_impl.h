/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/rec_model_utils.h"
#include "platform/stream.h"
#include "runtime/llm_worker_impl.h"
#include "util/threadpool.h"

namespace xllm {

class ModelLoader;

class RecWorkerImpl : public LLMWorkerImpl {
 public:
  RecWorkerImpl(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options);

  ~RecWorkerImpl() override;

  bool init_model(ModelContext& context) override;

  void load_model(std::unique_ptr<ModelLoader> loader) override;

  ForwardInput prepare_inputs(Batch& batch) override;

  void prepare_work_before_execute(const ForwardInput& inputs,
                                   ForwardInput& processed_inputs) override;

  folly::SemiFuture<std::optional<ForwardOutput>> step_async(
      const ForwardInput& inputs) override;

  std::optional<ForwardOutput> step(const ForwardInput& input) override;

 protected:
  std::shared_ptr<ThreadPool> input_builder_thread_pool_;

 private:
  class RecWorkPipeline {
   public:
    virtual ~RecWorkPipeline() = default;

    virtual ForwardInput prepare_inputs(Batch& batch) = 0;

    virtual void prepare_work_before_execute(
        const ForwardInput& inputs,
        ForwardInput& processed_inputs) = 0;

    virtual std::optional<ForwardOutput> step(const ForwardInput& input) = 0;
  };

  class LlmRecWorkPipeline final : public RecWorkPipeline {
   public:
    explicit LlmRecWorkPipeline(RecWorkerImpl& worker);

    ForwardInput prepare_inputs(Batch& batch) override;

    void prepare_work_before_execute(const ForwardInput& inputs,
                                     ForwardInput& processed_inputs) override;

    std::optional<ForwardOutput> step(const ForwardInput& input) override;

   private:
    RecWorkerImpl& worker_;
  };

  class OneRecWorkPipeline final : public RecWorkPipeline {
   public:
    explicit OneRecWorkPipeline(RecWorkerImpl& worker);

    ForwardInput prepare_inputs(Batch& batch) override;

    void prepare_work_before_execute(const ForwardInput& inputs,
                                     ForwardInput& processed_inputs) override;

    std::optional<ForwardOutput> step(const ForwardInput& input) override;

   private:
    RecWorkerImpl& worker_;
  };

  torch::Tensor merge_embeddings_by_indices(
      const torch::Tensor& input_tokens_embedding,
      const torch::Tensor& input_embedding,
      const std::vector<int64_t>& input_indices);

  std::unique_ptr<RecWorkPipeline> work_pipeline_;

  RecModelKind rec_model_kind_ = RecModelKind::kNone;

  uint32_t max_concurrency_ = 1;
  bool concurrent_llmrec_enabled_ = false;

  std::vector<std::unique_ptr<CausalLM>> model_instances_;
  std::vector<std::unique_ptr<Executor>> executor_instances_;
  std::vector<std::unique_ptr<Stream>> execute_streams_;
  std::vector<ModelContext> context_instances_;

  std::unique_ptr<ThreadPool> step_threadpool_;

  std::unordered_map<std::thread::id, size_t> thread_id_to_instance_id_;
  std::set<size_t> allocated_instance_ids_;
  std::mutex allocation_mutex_;

  void get_thread_model_instance(CausalLM*& model,
                                 Executor*& executor,
                                 Stream*& execute_stream,
                                 ModelContext*& context);
  void allocate_instance_id_for_current_thread();

  void update_last_step_output(const std::optional<ForwardOutput>& output);
};

}  // namespace xllm
