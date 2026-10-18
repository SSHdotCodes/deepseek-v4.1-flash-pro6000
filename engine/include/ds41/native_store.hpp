#pragma once
#include "ds41/engine.hpp"
#include <filesystem>
#include <map>
namespace ds41e {
struct Tensor {
  std::string name, path, dtype;
  std::vector<uint64_t> shape;
  uint64_t offset = 0, bytes = 0, file_bytes = 0;
};
class Store {
  std::filesystem::path root_;
  std::map<std::string, Tensor> tensors_;
  std::map<std::string, uint64_t> file_sizes_;

public:
  explicit Store(const std::string &root, bool require_complete = true);
  const Tensor &tensor(const std::string &name) const;
  const std::map<std::string, Tensor> &tensors() const { return tensors_; }
  void verify_receipt(const std::string &receipt) const;
  std::vector<uint8_t> read(const Tensor &tensor) const;
  Json inventory() const;
  static std::string normalize(std::string name);
};
Json read_json(const std::string &path);
std::string sha256_file(const std::string &path);
std::string sha256_bytes(const void *data, size_t bytes);
Json authenticate_files(const std::string &receipt,
                        const std::string &receipt_sha,
                        const std::map<std::string, std::string> &role_paths);
void validate_spans(uint64_t record_bytes,
                    std::vector<std::pair<uint64_t, uint64_t>> spans);
// CPU-only hash metadata is exported from unchanged official engram.py. History
// uses absolute positions, so rejected future tokens can simply be overwritten.
class EngramHash {
  Json meta_;
  std::vector<int64_t> history_;

public:
  explicit EngramHash(const std::string &path);
  std::vector<std::vector<int64_t>> rows(int token, int64_t pos,
                                         bool text = true);
  const std::vector<int64_t>& history() const { return history_; }
  void reset() { history_.clear(); }
  void truncate(size_t n) { if (n < history_.size()) history_.resize(n); }
};
} // namespace ds41e
