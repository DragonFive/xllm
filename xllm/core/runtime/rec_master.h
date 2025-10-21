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

#include <atomic>
#include <thread>

#include "framework/chat_template/jinja_chat_template.h"
#include "common/rate_limiter.h"
#include "common/threadpool.h"
#include "runtime/master.h"
#include "framework/model/model_args.h"
#include "runtime/rec_engine.h"
#include "scheduler/continuous_scheduler.h"

namespace xllm {

class RecMaster : public Master {
 public:
  explicit RecMaster(const Options& options);
  ~RecMaster() = default;

  // schedule a request, the engine will execute the request asynchronously
  // and call the callback with output when the request is done
  // the callback will be called multiple times if the request is a streaming
  // request
  void schedule_async(
      std::string prompt,
      std::optional<std::vector<int>> prompt_tokens,
      std::optional<std::vector<proto::InferInputTensor>> input_tensors,
      RequestParams sp,
      Priority priority,
      bool stream,
      OutputCallback callback) override;

  void schedule_chat_async(
      std::vector<Message> messages,
      std::optional<std::vector<int>> prompt_tokens,
      std::optional<std::vector<proto::InferInputTensor>> input_tensors,
      RequestParams sp,
      Priority priority,
      bool stream,
      OutputCallback callback) override;

  // Additional virtual methods for API service compatibility
  void get_cache_info(std::vector<uint64_t>& cluster_ids,
                      std::vector<int64_t>& k_cache_ids,
                      std::vector<int64_t>& v_cache_ids) override {
    // OmniRec does not support cache info, provide empty implementation
    cluster_ids.clear();
    k_cache_ids.clear();
    v_cache_ids.clear();
  }

  bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                    const std::vector<std::string>& device_ips,
                    const std::vector<uint16_t>& ports,
                    const int32_t dp_size) override {
    // OmniRec does not support cluster linking, return false
    return false;
  }

  bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                      const std::vector<std::string>& device_ips,
                      const std::vector<uint16_t>& ports,
                      const int32_t dp_size) override {
    // OmniRec does not support cluster unlinking, return false
    return false;
  }

  // Inherit all public methods from Master
  void start() override;
  void stop() override;

  const TokensItemConverter* tokens_item_converter() const {
    return dynamic_cast<OmniRecEngine*>(engine_.get())->tokens_item_converter();
  }

 private:
  std::shared_ptr<Request> create_request(
      std::string prompt,
      std::optional<std::vector<int>> prompt_tokens,
      std::optional<std::vector<proto::InferInputTensor>> input_tensors,
      const RequestParams& sp,
      Priority priority,
      bool stream,
      OutputCallback callback);

  std::shared_ptr<Request> create_chat_request(
      const std::vector<Message>& messages,
      std::optional<std::vector<int>> prompt_tokens,
      const RequestParams& sp,
      Priority priority,
      bool stream,
      OutputCallback callback);

  void schedule(
      std::string prompt,
      std::optional<std::vector<int>> prompt_tokens,
      std::optional<std::vector<proto::InferInputTensor>> input_tensors,
      RequestParams sp,
      Priority priority,
      bool stream,
      OutputCallback callback);

  void schedule(
      std::vector<Message> messages,
      std::optional<std::vector<int>> prompt_tokens,
      std::optional<std::vector<proto::InferInputTensor>> input_tensors,
      RequestParams sp,
      Priority priority,
      bool stream,
      OutputCallback callback);

  std::unique_ptr<Scheduler> scheduler_;
  ModelArgs model_args_;
  std::unique_ptr<ThreadPool> threadpool_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<JinjaChatTemplate> chat_template_;
  // Thread management
  std::thread loop_thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stopped_{false};
};

}  // namespace llm