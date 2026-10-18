#include "ds41/native_store.hpp"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <openssl/evp.h>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
namespace ds41e {
Json read_json(const std::string &p) {
  std::ifstream f(p);
  if (!f)
    throw std::runtime_error("cannot open " + p);
  Json j;
  f >> j;
  return j;
}
std::string sha256_bytes(const void *data, size_t bytes) {
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned int count = 0;
  if (EVP_Digest(data, bytes, digest, &count, EVP_sha256(), nullptr) != 1 ||
      count != 32)
    throw std::runtime_error("SHA256 memory digest failed");
  static constexpr char hex[] = "0123456789abcdef";
  std::string out;
  for (unsigned int i = 0; i < count; ++i) {
    out += hex[digest[i] >> 4];
    out += hex[digest[i] & 15];
  }
  return out;
}
std::string sha256_file(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("cannot authenticate " + path);
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(),
                                                              EVP_MD_CTX_free);
  if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("SHA256 initialization failed");
  char chunk[65536];
  while (file.read(chunk, sizeof(chunk)) || file.gcount())
    if (EVP_DigestUpdate(ctx.get(), chunk, size_t(file.gcount())) != 1)
      throw std::runtime_error("SHA256 update failed");
  if (!file.eof())
    throw std::runtime_error("authentication read failed: " + path);
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned int bytes = 0;
  if (EVP_DigestFinal_ex(ctx.get(), digest, &bytes) != 1 || bytes != 32)
    throw std::runtime_error("SHA256 finalization failed");
  static constexpr char hex[] = "0123456789abcdef";
  std::string out;
  for (unsigned int i = 0; i < bytes; ++i) {
    out += hex[digest[i] >> 4];
    out += hex[digest[i] & 15];
  }
  return out;
}
Json authenticate_files(const std::string &receipt,
                        const std::string &receipt_sha,
                        const std::map<std::string, std::string> &role_paths) {
  if (receipt_sha.size() != 64 || sha256_file(receipt) != receipt_sha)
    throw std::runtime_error(
        "auxiliary receipt differs from compiled identity");
  auto manifest = read_json(receipt);
  if (manifest.at("schema") != "ds41-auxiliary-v1")
    throw std::runtime_error("auxiliary schema mismatch");
  for (const auto &item : role_paths) {
    const auto &identity = manifest.at("files").at(item.first);
    if (std::filesystem::file_size(item.second) !=
            identity.at("bytes").get<uint64_t>() ||
        sha256_file(item.second) != identity.at("sha256").get<std::string>())
      throw std::runtime_error("auxiliary identity mismatch: " + item.first);
  }
  return manifest;
}
void validate_spans(uint64_t size,
                    std::vector<std::pair<uint64_t, uint64_t>> spans) {
  if (!size || spans.empty())
    throw std::runtime_error("empty expert record/spans");
  std::sort(spans.begin(), spans.end());
  uint64_t end = 0;
  for (auto span : spans) {
    if (!span.second || span.first < end || span.first > size ||
        span.second > size - span.first)
      throw std::runtime_error("overlapping/out-of-bounds expert read spans");
    end = span.first + span.second;
  }
}
static uint64_t checked_add(uint64_t a, uint64_t b) {
  if (a > UINT64_MAX - b)
    throw std::runtime_error("tensor offset overflow");
  return a + b;
}
static uint64_t checked_mul(uint64_t a, uint64_t b) {
  if (b && a > UINT64_MAX / b)
    throw std::runtime_error("tensor shape overflow");
  return a * b;
}
static void pread_all(int fd, void *out, size_t n, uint64_t offset) {
  size_t done = 0;
  while (done < n) {
    auto r = pread(fd, static_cast<char *>(out) + done,
                   std::min(n - done, size_t(16 << 20)), offset + done);
    if (r < 0 && errno == EINTR)
      continue;
    if (r <= 0)
      throw std::runtime_error("short tensor read: " +
                               std::string(r < 0 ? strerror(errno) : "EOF"));
    done += r;
  }
}
struct Fd {
  int fd;
  explicit Fd(const std::string &p)
      : fd(open(p.c_str(), O_RDONLY | O_CLOEXEC)) {
    if (fd < 0)
      throw std::runtime_error("open failed: " + p);
  }
  ~Fd() { close(fd); }
};
std::string Store::normalize(std::string n) {
  if (n.compare(0, 6, "model.") == 0)
    n.erase(0, 6);
  auto replace = [&](const std::string &a, const std::string &b) {
    size_t i = 0;
    while ((i = n.find(a, i)) != std::string::npos) {
      n.replace(i, a.size(), b);
      i += b.size();
    }
  };
  replace("self_attn", "attn");
  if (n.compare(0, 7, "vision.") != 0)
    replace("mlp", "ffn");
  replace("weight_scale_inv", "scale");
  replace("e_score_correction_bias", "bias");
  return n;
}
Store::Store(const std::string &root, bool complete)
    : root_(std::filesystem::canonical(root)) {
  auto index = read_json((root_ / "model.safetensors.index.json").string());
  std::map<std::string, Json> headers;
  std::map<std::string, uint64_t> starts;
  for (auto it = index.at("weight_map").begin();
       it != index.at("weight_map").end(); ++it) {
    std::string shard = it.value();
    std::filesystem::path relative(shard);
    if (relative.is_absolute() || relative.filename() != relative)
      throw std::runtime_error("unsafe shard path");
    auto path = (root_ / relative).string();
    if (!headers.count(shard)) {
      Fd fd(path);
      uint8_t length_bytes[8];
      pread_all(fd.fd, length_bytes, 8, 0);
      uint64_t length = 0;
      for (int k = 0; k < 8; ++k)
        length |= uint64_t(length_bytes[k]) << (8 * k);
      if (!length || length > 128ull << 20)
        throw std::runtime_error("invalid safetensors header length");
      std::string raw(length, '\0');
      pread_all(fd.fd, raw.data(), length, 8);
      auto header = Json::parse(raw);
      uint64_t data_start = checked_add(8, length), end = 0;
      std::vector<std::pair<uint64_t, uint64_t>> intervals;
      const std::map<std::string, int> sizes = {
          {"BOOL", 1},    {"U8", 1},        {"I8", 1},      {"I16", 2},
          {"I32", 4},     {"I64", 8},       {"F16", 2},     {"BF16", 2},
          {"F32", 4},     {"F64", 8},       {"F8_E4M3", 1}, {"F8_E4M3FN", 1},
          {"F8_E8M0", 1}, {"F8_E8M0FNU", 1}};
      for (auto t = header.begin(); t != header.end(); ++t) {
        if (t.key() == "__metadata__")
          continue;
        auto d = t.value();
        auto range = d.at("data_offsets").get<std::vector<uint64_t>>();
        if (range.size() != 2 || range[1] < range[0])
          throw std::runtime_error("invalid tensor range");
        auto found = sizes.find(d.at("dtype"));
        if (found == sizes.end())
          throw std::runtime_error("unsupported native dtype " +
                                   d.at("dtype").get<std::string>());
        uint64_t bytes = found->second;
        for (uint64_t dim : d.at("shape").get<std::vector<uint64_t>>())
          bytes = checked_mul(bytes, dim);
        if (bytes != range[1] - range[0])
          throw std::runtime_error("tensor byte/shape mismatch: " + t.key());
        if (bytes)
          intervals.push_back({range[0], range[1]});
        end = std::max(end, range[1]);
      }
      std::sort(intervals.begin(), intervals.end());
      uint64_t cursor = 0;
      for (auto span : intervals) {
        if (span.first != cursor)
          throw std::runtime_error("overlapping or gapped tensor data");
        cursor = span.second;
      }
      struct stat st{};
      if (fstat(fd.fd, &st))
        throw std::runtime_error("fstat failed");
      uint64_t expected = checked_add(data_start, end);
      if (complete && uint64_t(st.st_size) != expected)
        throw std::runtime_error("incomplete/changed shard: " + shard +
                                 " expected " + std::to_string(expected) +
                                 " got " + std::to_string(st.st_size));
      headers[shard] = std::move(header);
      starts[shard] = data_start;
      file_sizes_[path] = expected;
    }
    auto d = headers.at(shard).at(it.key());
    auto range = d.at("data_offsets").get<std::vector<uint64_t>>();
    auto name = normalize(it.key());
    Tensor t{name,
             path,
             d.at("dtype"),
             d.at("shape").get<std::vector<uint64_t>>(),
             checked_add(starts.at(shard), range[0]),
             range[1] - range[0],
             file_sizes_.at(path)};
    if (!tensors_.emplace(name, std::move(t)).second)
      throw std::runtime_error("duplicate normalized name: " + name);
  }
}
const Tensor &Store::tensor(const std::string &n) const {
  return tensors_.at(n);
}
void Store::verify_receipt(const std::string &path) const {
  auto r = read_json(path);
  if (!r.value("complete", false) ||
      r.at("root").get<std::string>() != root_.string())
    throw std::runtime_error("complete matching integrity receipt required");
  std::map<std::string, Json> files;
  for (auto f : r.at("files"))
    if (f.value("ok", false))
      files[f.at("path")] = f;
  for (auto file : file_sizes_) {
    auto base = std::filesystem::path(file.first).filename().string();
    auto it = files.find(base);
    if (it == files.end())
      throw std::runtime_error("unverified shard: " + base);
    struct stat st{};
    if (stat(file.first.c_str(), &st))
      throw std::runtime_error("missing shard: " + base);
    int64_t mtime =
        int64_t(st.st_mtim.tv_sec) * 1000000000 + st.st_mtim.tv_nsec;
    if (uint64_t(st.st_size) != file.second ||
        it->second.at("size").get<uint64_t>() != file.second ||
        it->second.at("mtime_ns").get<int64_t>() != mtime)
      throw std::runtime_error("stale weight verification: " + base);
  }
}
std::vector<uint8_t> Store::read(const Tensor &t) const {
  Fd fd(t.path);
  struct stat st{};
  if (fstat(fd.fd, &st) || uint64_t(st.st_size) != t.file_bytes)
    throw std::runtime_error("shard changed before read");
  std::vector<uint8_t> out(t.bytes);
  pread_all(fd.fd, out.data(), out.size(), t.offset);
  return out;
}
Json Store::inventory() const {
  Json j = {{"tensors", tensors_.size()}, {"shards", file_sizes_.size()},
            {"tensor_bytes", 0},          {"routed_bytes", 0},
            {"engram_table_bytes", 0},    {"resident_bytes", 0}};
  for (auto item : tensors_) {
    auto &t = item.second;
    std::string group = t.name.find(".experts.") != std::string::npos
                            ? "routed_bytes"
                        : t.name.find(".engram.embed.") != std::string::npos
                            ? "engram_table_bytes"
                            : "resident_bytes";
    j[group] = j[group].get<uint64_t>() + t.bytes;
    j["tensor_bytes"] = j["tensor_bytes"].get<uint64_t>() + t.bytes;
  }
  return j;
}
EngramHash::EngramHash(const std::string &path) : meta_(read_json(path)) {
  if (meta_.at("schema") != "ds41-engram-hash-v1")
    throw std::runtime_error("invalid Engram metadata schema");
  if (meta_.at("token_map").size() != 129280 ||
      meta_.at("compressed_vocab_size") != 99092)
    throw std::runtime_error("Engram vocabulary mismatch");
  constexpr int64_t vocab = 99092;
  for (const auto &id : meta_.at("token_map"))
    if (!id.is_number_integer() || id.get<int64_t>() < 0 ||
        id.get<int64_t>() >= vocab)
      throw std::runtime_error("invalid compressed token map");
  auto pad = meta_.at("pad_id").get<int64_t>();
  if (pad < 0 || pad >= vocab || meta_.at("layers").size() != 2)
    throw std::runtime_error("invalid Engram pad/layer count");
  for (size_t i = 0; i < 2; ++i) {
    const auto &layer = meta_.at("layers").at(i);
    if (layer.at("layer") != (i == 0 ? 1 : 14) ||
        layer.at("primes").size() != 24 || layer.at("offsets").size() != 24 ||
        layer.at("multipliers").size() != 4)
      throw std::runtime_error("Engram layer order/geometry mismatch");
    int64_t offset = 0, previous = 0;
    for (int c = 0; c < 24; ++c) {
      auto prime = layer.at("primes").at(c).get<int64_t>();
      if (prime <= previous || prime > INT32_MAX ||
          layer.at("offsets").at(c).get<int64_t>() != offset)
        throw std::runtime_error("Engram primes/offsets mismatch");
      previous = prime;
      offset += prime;
    }
    for (const auto &m : layer.at("multipliers"))
      if (!m.is_number_integer() || m.get<int64_t>() <= 0 ||
          m.get<int64_t>() > INT64_MAX / (vocab - 1))
        throw std::runtime_error("Engram multiplier overflow/range");
  }
}
std::vector<std::vector<int64_t>> EngramHash::rows(int token, int64_t pos,
                                                   bool text) {
  if (token < 0 || token >= int(meta_.at("token_map").size()) || pos < 0 ||
      size_t(pos) > history_.size())
    throw std::invalid_argument("hash token or position");
  if (size_t(pos) == history_.size())
    history_.push_back(0);
  history_[pos] = text ? meta_.at("token_map").at(token).get<int64_t>() : -1;
  int64_t values[4];
  bool blocked = false;
  for (int k = 0; k < 4; ++k) {
    blocked = blocked || pos < k || (pos >= k && history_[pos - k] == -1);
    values[k] = blocked ? meta_.at("pad_id").get<int64_t>() : history_[pos - k];
  }
  std::vector<std::vector<int64_t>> result;
  for (auto layer : meta_.at("layers")) {
    std::vector<int64_t> rows;
    int64_t rolling = values[0] * layer.at("multipliers").at(0).get<int64_t>();
    for (int k = 1; k < 4; ++k) {
      rolling ^= values[k] * layer.at("multipliers").at(k).get<int64_t>();
      for (int h = 0; h < 8; ++h) {
        int col = (k - 1) * 8 + h;
        rows.push_back(rolling % layer.at("primes").at(col).get<int64_t>() +
                       layer.at("offsets").at(col).get<int64_t>());
      }
    }
    result.push_back(std::move(rows));
  }
  return result;
}
} // namespace ds41e
