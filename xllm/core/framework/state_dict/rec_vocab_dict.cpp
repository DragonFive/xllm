#include "rec_vocab_dict.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>

#include "common/global_flags.h"
#include "util/timer.h"

namespace xllm {
namespace {

bool rec_token_triple_less(const RecTokenTriple& lhs,
                           const RecTokenTriple& rhs) {
  if (lhs[0] != rhs[0]) {
    return lhs[0] < rhs[0];
  }
  if (lhs[1] != rhs[1]) {
    return lhs[1] < rhs[1];
  }
  return lhs[2] < rhs[2];
}

void check_token_id(int32_t token_id, int32_t vocab_size, const char* name) {
  CHECK_GE(token_id, 0) << "Invalid OneRec " << name
                        << " token id: " << token_id;
  CHECK_LT(token_id, vocab_size)
      << "OneRec " << name << " token id " << token_id << " exceeds vocab_size "
      << vocab_size;
}

}  // namespace

bool RecVocabDict::initialize(const std::string& vocab_file) {
  if (initialized_) {
    return true;
  }

  Timer timer;

  if (vocab_file.empty()) {
    LOG(ERROR) << "Content data file is empty, file: " << vocab_file;
    return false;
  }
  if (!std::filesystem::exists(vocab_file)) {
    LOG(ERROR) << "Fail to find content data file: " << vocab_file;
    return false;
  }
  std::ifstream ifs(vocab_file.data(), std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) {
    LOG(ERROR) << "Fail to load content data file: " << vocab_file;
    return false;
  }

  const size_t file_size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  // Each line of content : 1 * int64_t(item id) + REC_TOKEN_SIZE *
  //  int32_t(token id);
  const size_t itemid_size = sizeof(int64_t);
  const size_t tokens_size = REC_TOKEN_SIZE * sizeof(int32_t);
  const size_t line_size = tokens_size + itemid_size;
  const size_t estimated_lines = (file_size + line_size - 1) / line_size;

  // 2 and 4 are only empirical values
  item_to_tokens_map_.reserve(estimated_lines);
  tokens_to_items_map_.reserve(estimated_lines / 2);
  prefix_tokens_to_next_tokens_map_.reserve(estimated_lines / 4);

  int64_t item_id = 0;
  RecTokenTriple tokens;

  while (ifs.read(reinterpret_cast<char*>(&item_id), itemid_size) &&
         ifs.read(reinterpret_cast<char*>(tokens.data()), tokens_size)) {
    if (FLAGS_enable_constrained_decoding) {
      for (int i = 0; i < tokens.size(); i++) {
        std::vector<int32_t> prefix_tokens;

        for (int j = 0; j < i; j++) {
          prefix_tokens.emplace_back(tokens[j]);
        }

        prefix_tokens_to_next_tokens_map_[prefix_tokens].insert(tokens[i]);
      }
    }

    item_to_tokens_map_[item_id] = tokens;

    tokens_to_items_map_[tokens].emplace_back(item_id);
  }

  if (ifs.gcount() != 0 && ifs.gcount() != line_size) {
    LOG(ERROR) << "Possibly containing incomplete lines : " << vocab_file;
    item_to_tokens_map_.clear();
    tokens_to_items_map_.clear();
    prefix_tokens_to_next_tokens_map_.clear();
    return false;
  }

  initialized_ = true;
  LOG(INFO) << "Total line size:" << estimated_lines
            << ",parse tokens to item id map size: "
            << tokens_to_items_map_.size()
            << ", parse item to tokens map size:" << item_to_tokens_map_.size()
            << ", parse prefix tokens to next tokens map size:"
            << prefix_tokens_to_next_tokens_map_.size()
            << ", cost: " << timer.elapsed_seconds() << " seconds";

  return true;
}

bool RecVocabDict::get_items_by_tokens(const RecTokenTriple& rec_token_triple,
                                       std::vector<int64_t>* item_ids) const {
  CHECK_EQ(initialized_, true);
  CHECK_NE(item_ids, nullptr);

  auto iter = tokens_to_items_map_.find(rec_token_triple);
  if (iter == tokens_to_items_map_.end()) {
    return false;
  }

  std::copy(
      iter->second.begin(), iter->second.end(), std::back_inserter(*item_ids));

  return true;
}

bool RecVocabDict::get_tokens_by_item(int64_t item_id,
                                      std::vector<int32_t>* token_ids) const {
  CHECK_EQ(initialized_, true);
  CHECK_NE(token_ids, nullptr);

  auto iter = item_to_tokens_map_.find(item_id);
  if (iter == item_to_tokens_map_.end()) {
    return false;
  }

  std::copy(
      iter->second.begin(), iter->second.end(), std::back_inserter(*token_ids));

  return true;
}

const std::unordered_set<int32_t>&
RecVocabDict::get_next_tokens_by_prefix_tokens(
    const Slice<int32_t>& prefix_token_ids) const {
  CHECK_EQ(initialized_, true);
  CHECK_LT(prefix_token_ids.size(), REC_TOKEN_SIZE);

  std::vector<int32_t> prefix_tokens_ids_vec = prefix_token_ids;
  auto iter = prefix_tokens_to_next_tokens_map_.find(prefix_tokens_ids_vec);
  if (iter == prefix_tokens_to_next_tokens_map_.end()) {
    static std::unordered_set<int32_t> empty_set;
    return empty_set;
  }

  return iter->second;
}

RecConstraintTables RecVocabDict::build_constraint_tables(
    int32_t vocab_size) const {
  CHECK_EQ(initialized_, true);
  CHECK_GT(vocab_size, 0);

  RecConstraintTables tables;
  tables.vocab_size = vocab_size;
  tables.prefix1_offsets.assign(static_cast<size_t>(vocab_size) + 1, 0);
  tables.prefix2_value_offsets.emplace_back(0);

  if (tokens_to_items_map_.empty()) {
    return tables;
  }

  std::vector<RecTokenTriple> triples;
  triples.reserve(tokens_to_items_map_.size());
  for (const auto& entry : tokens_to_items_map_) {
    const RecTokenTriple& tokens = entry.first;
    check_token_id(tokens[0], vocab_size, "t0");
    check_token_id(tokens[1], vocab_size, "t1");
    check_token_id(tokens[2], vocab_size, "t2");
    triples.emplace_back(tokens);
  }

  std::sort(triples.begin(), triples.end(), rec_token_triple_less);

  int32_t previous_t0 = -1;
  for (const RecTokenTriple& tokens : triples) {
    if (tokens[0] != previous_t0) {
      tables.first_token_ids.emplace_back(tokens[0]);
      previous_t0 = tokens[0];
    }
  }
  tables.max_first_degree = static_cast<int32_t>(tables.first_token_ids.size());

  tables.prefix1_values.reserve(triples.size());
  tables.prefix2_values.reserve(triples.size());
  tables.prefix2_value_offsets.reserve(triples.size() + 1);

  size_t triple_idx = 0;
  for (int32_t t0 = 0; t0 < vocab_size; ++t0) {
    const int32_t prefix1_begin =
        static_cast<int32_t>(tables.prefix1_values.size());
    tables.prefix1_offsets[static_cast<size_t>(t0)] = prefix1_begin;

    while (triple_idx < triples.size() && triples[triple_idx][0] == t0) {
      const int32_t t1 = triples[triple_idx][1];
      tables.prefix1_values.emplace_back(t1);

      const int32_t prefix2_begin =
          static_cast<int32_t>(tables.prefix2_values.size());
      int32_t previous_t2 = -1;
      while (triple_idx < triples.size() && triples[triple_idx][0] == t0 &&
             triples[triple_idx][1] == t1) {
        const int32_t t2 = triples[triple_idx][2];
        if (t2 != previous_t2) {
          tables.prefix2_values.emplace_back(t2);
          previous_t2 = t2;
        }
        ++triple_idx;
      }

      const int32_t prefix2_end =
          static_cast<int32_t>(tables.prefix2_values.size());
      tables.max_prefix2_degree =
          std::max(tables.max_prefix2_degree, prefix2_end - prefix2_begin);
      tables.prefix2_value_offsets.emplace_back(prefix2_end);
    }

    const int32_t prefix1_end =
        static_cast<int32_t>(tables.prefix1_values.size());
    tables.prefix1_offsets[static_cast<size_t>(t0) + 1] = prefix1_end;
    tables.max_prefix1_degree =
        std::max(tables.max_prefix1_degree, prefix1_end - prefix1_begin);
  }

  CHECK_EQ(triple_idx, triples.size());
  CHECK_EQ(tables.prefix2_value_offsets.size(),
           tables.prefix1_values.size() + 1);
  return tables;
}

}  // namespace xllm
