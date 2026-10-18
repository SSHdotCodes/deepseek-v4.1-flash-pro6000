#include "ds41/engine.hpp"
#include "ds41/spark.h"
#include <algorithm>
#include <cstring>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
using namespace ds41e;
namespace {
void ck(cudaError_t rc) {
  if (rc != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(rc));
}
void sp(int rc) {
  if (rc)
    throw std::runtime_error(ds41_spark_error());
}
std::vector<char> tensor(const std::string &path, const char *name) {
  std::ifstream f(path, std::ios::binary);
  uint64_t n = 0;
  f.read(reinterpret_cast<char *>(&n), 8);
  if (!f || n > 16 * 1024 * 1024)
    throw std::runtime_error("invalid fixture header");
  std::string header(n, '\0');
  f.read(header.data(), header.size());
  auto entry = Json::parse(header).at(name);
  auto offsets = entry.at("data_offsets").get<std::vector<uint64_t>>();
  if (offsets.size() != 2 || offsets[1] < offsets[0])
    throw std::runtime_error("invalid fixture offsets");
  std::vector<char> data(offsets[1] - offsets[0]);
  f.seekg(8 + n + offsets[0]);
  f.read(data.data(), data.size());
  if (!f)
    throw std::runtime_error("short fixture tensor");
  return data;
}
struct Device {
  void *p = nullptr;
  explicit Device(size_t n) { ck(cudaMalloc(&p, n)); }
  ~Device() { cudaFree(p); }
};
// Substitute only the target's captured-input stream. All weights, routed
// experts, stream and BLAS handle are the authenticated native target's views.
class CaptureTarget final : public Forward {
  std::unique_ptr<Forward> native_;
  ds41_target_capture_sink sink_ = nullptr;
  void *user_ = nullptr;
  int64_t committed_ = 0;
  ds41_target_capture_v1 capture_{};
  const void *prefill_data_ = nullptr;
public:
  explicit CaptureTarget(std::unique_ptr<Forward> native)
      : native_(std::move(native)) {}
  int vocab() const override { return native_->vocab(); }
  int eos() const override { return native_->eos(); }
  void reset() override {
    native_->reset();
    committed_ = 0;
  }
  std::vector<std::vector<float>> verify(const std::vector<int> &) override {
    throw std::logic_error("fixture target accepts explicit captures only");
  }
  void push(const void *data, int rows) {
    capture_ = {1, rows, 15360, rows, committed_, 30720, data,
                shared_weights().cuda_stream};
    commit(rows);
  }
  void set_prefill_data(const void *data) { prefill_data_ = data; }
  bool supports_prefill(PrefillMode, size_t rows) const override {
    return rows > 0;
  }
  PrefillResult prefill(const std::vector<int> &ids,
                       const PrefillOptions &) override {
    if (!prefill_data_ || ids.empty())
      throw std::logic_error("missing fixture prefill captures");
    PrefillResult result;
    result.begin_position = committed_;
    result.consumed_rows = ids.size();
    const size_t tail = std::min<size_t>(128, ids.size());
    committed_ += ids.size() - tail;
    for (size_t i = 0; i < tail;) {
      const int rows = int(std::min<size_t>(6, tail - i));
      push(static_cast<const char *>(prefill_data_) +
               (ids.size() - tail + i) * 30720, rows);
      i += rows;
    }
    return result;
  }
  void commit(size_t n) override {
    if (n != size_t(capture_.committed_rows) ||
        !sink_ || sink_(user_, &capture_))
      throw std::runtime_error("fixture capture sink failed");
    committed_ += n;
  }
  ds41_target_weights_v2 shared_weights() const override {
    return native_->shared_weights();
  }
  ds41_target_services_v1 services() override { return native_->services(); }
  ds41_target_tensor_v1 resident_tensor(const std::string &s) const override {
    return native_->resident_tensor(s);
  }
  std::vector<std::string> resident_names(const std::string &s) const override {
    return native_->resident_names(s);
  }
  void set_capture_sink(ds41_target_capture_sink sink, void *user) override {
    sink_ = sink;
    user_ = user;
  }
};
std::unique_ptr<ds41_spark, decltype(&ds41_spark_destroy)>
reference(Forward &native) {
  auto w = native.shared_weights();
  auto s = native.services();
  ds41_spark_config cfg{};
  cfg.abi_version = DS41_SPARK_ABI;
  cfg.stream = w.cuda_stream;
  cfg.blas_handle = w.blas_handle;
  cfg.user = s.user;
  cfg.experts = s.experts;
  ds41_spark *raw = nullptr;
  sp(ds41_spark_create(&cfg, &raw));
  std::unique_ptr<ds41_spark, decltype(&ds41_spark_destroy)> ctx(
      raw, ds41_spark_destroy);
  auto names = native.resident_names("mtp.");
  names.insert(names.end(), {"embed.weight", "head.weight", "rope"});
  for (const auto &name : names) {
    auto v = native.resident_tensor(name);
    sp(ds41_spark_bind(ctx.get(), name.c_str(),
                       {v.data, v.dtype, v.rows, v.columns}));
  }
  return ctx;
}
} // namespace
int main(int argc, char **argv) {
  try {
    const bool policy_q0 = argc == 8 && std::string(argv[7]) == "--policy-q0";
    if (argc != 7 && !policy_q0)
      throw std::invalid_argument("test_native_spark_seed WEIGHTS RECEIPT ENGRAM "
                                  "ROPE EXPERTS FORWARD_FIXTURE [--policy-q0]");
    Config c;
    c.weights = argv[1]; c.receipt = argv[2]; c.engram = argv[3];
    c.rope = argv[4]; c.experts = argv[5]; c.max_context = 8192;
    c.expert_slots = 64; c.load_mtp = c.allow_unqualified = true;
    auto capture = std::make_unique<CaptureTarget>(make_native(c));
    auto *source = capture.get();
    auto stream = static_cast<cudaStream_t>(source->shared_weights().cuda_stream);
    auto adapter = make_spark_target(std::move(capture), policy_q0 ? 0 : 6,
                                     policy_q0);
    auto direct = reference(*source);
    auto seed = tensor(argv[6], "target_capture_seed_bf16");
    auto last = tensor(argv[6], "target_capture_r_bf16");
    auto gold = tensor(argv[6], "q_logits_f32");
    auto anchor_bytes = tensor(argv[6], "anchor_token_i64");
    auto previous_bytes = tensor(argv[6], "proposal_token_ids_i64");
    if (seed.size() != 47 * 30720 || last.size() != 30720 ||
        gold.size() != 5 * 129280 * 4 || anchor_bytes.size() != 8 ||
        previous_bytes.size() != 6 * 8)
      throw std::runtime_error("unexpected real-chat fixture shapes");
    int64_t anchor = 0, previous[6]{};
    std::memcpy(&anchor, anchor_bytes.data(), 8);
    std::memcpy(previous, previous_bytes.data(), sizeof(previous));
    std::vector<char> repeated(129 * 30720);
    for (int i = 0; i < 129; ++i)
      std::memcpy(repeated.data() + i * 30720,
                  seed.data() + (i % 47) * 30720, 30720);
    Device input(repeated.size());
    source->set_prefill_data(input.p);
    auto upload = [&](const std::vector<char> &data) {
      ck(cudaMemcpyAsync(input.p, data.data(), data.size(),
                          cudaMemcpyHostToDevice, stream));
    };
    auto push = [&](int begin, int rows) {
      source->push(static_cast<const char *>(input.p) + begin * 30720, rows);
    };
    uint64_t values = 0;
    auto equal = [&](const std::vector<float> &x, const void *y) {
      if (x.size() != 129280 || std::memcmp(x.data(), y, x.size() * 4))
        throw std::runtime_error("Spark adapter differs from frozen/native oracle");
      values += x.size();
    };
    adapter->reset(); upload(seed); adapter->begin_prompt(47, false);
    // A speculative target slab may contain six rows, but only its accepted
    // prefix enters this sink. Vary accepted widths to exercise chunk edges.
    for (int begin = 0, width = 1; begin < 47; width = width % 6 + 1) {
      int rows = std::min(width, 47 - begin); push(begin, rows); begin += rows;
    }
    adapter->end_prompt(true);
    upload(last); push(0, 1);
    if (policy_q0) {
      // Nonce15 is the frozen policy's deterministic16th-attempt exploration.
      // No current proposal, target score, or residency is supplied to admission.
      DraftContext context{int(anchor), 6, 48, 15, 1.0, 1.0};
      auto *draft = adapter->drafter();
      const int preflight = draft->prepare(context);
      if (preflight != 3)
        throw std::runtime_error("policy fixture did not admit exploration3");
      draft->begin(int(anchor));
      if (draft->finalize_width(preflight) != 3)
        throw std::runtime_error("policy fixture changed deterministic exploration");
      equal(draft->logits(0, int(anchor)), gold.data());
      // Every later Markov call overwrites the borrowed device logits scratch.
      // q4's numerical fixture remains a separate gate; only q0 retention is
      // asserted here after all five conditional heads have used that scratch.
      for (int k = 1; k < 5; ++k) {
        auto conditional = draft->logits(k, int(previous[k]));
        if (k < 4) equal(conditional, gold.data() + k * 129280 * 4);
        equal(draft->logits(0, int(anchor)), gold.data());
      }
      bool refused_anchor = false;
      try { draft->logits(0, int(anchor) + 1); }
      catch (const std::logic_error &) { refused_anchor = true; }
      if (!refused_anchor)
        throw std::runtime_error("retained q0 accepted a different known anchor");
      draft->abort(context.nonce);
      const auto aborted = adapter->counters().at("spark_policy_v2");
      if (aborted.at("generation") != 1 || aborted.at("completed") != 0 ||
          aborted.at("width_history") != Json::array({-3}))
        throw std::runtime_error("native policy abort did not retire its attempt");
      context.nonce = 16;
      if (draft->prepare(context) != 1)
        throw std::runtime_error("unknown-mode preflight did not skip forward");
      draft->abort(context.nonce);
      adapter->begin_request();
      const auto restarted = adapter->counters().at("spark_policy_v2");
      if (restarted.at("generation") != 0 || restarted.at("completed") != 0 ||
          restarted.at("attempts") != 0)
        throw std::runtime_error("policy request namespace did not reset");
      context.nonce = 15;
      if (draft->prepare(context) != 3)
        throw std::runtime_error("new request retained the previous nonce namespace");
      draft->abort(context.nonce);
      ck(cudaStreamSynchronize(stream));
      std::cout << Json({{"schema", "ds41-engine-native-policy-q0-v1"},
          {"execution_scope", "native q0 content/retention and adapter lifecycle only; no q4, target-width or S2/X4 qualification"},
          {"q0_after_markov_heads", {0, 1, 2, 3, 4}},
          {"compared_values_bitwise", values},
          {"q4_fixture_compared", false},
          {"wrong_anchor_refused", true}, {"abort_and_request_reset", true}}).dump() << '\n';
      return 0;
    }
    adapter->drafter()->begin(int(anchor));
    for (int k = 0; k < 5; ++k)
      equal(adapter->drafter()->logits(k, int(previous[k])),
             gold.data() + k * 129280 * 4);
    for (int n : {1, 16, 128, 129}) {
      adapter->reset(); upload(repeated); adapter->begin_prompt(n, false);
      for (int begin = 0; begin < n;) {
        int rows = std::min(6, n - begin); push(begin, rows); begin += rows;
      }
      adapter->end_prompt(true);
      sp(ds41_spark_seed(direct.get(),
                          static_cast<char *>(input.p) + std::max(0, n - 128) * 30720,
                          n));
      if (adapter->counters().at("spark_committed_inputs") != n ||
          adapter->counters().at("spark_seed_mode") != "full-M-cold")
        throw std::runtime_error("cold prompt lifecycle counters differ");
      // Equal-prompt reuse performs no seed; appended rows use cached M1 norms.
      adapter->begin_prompt(n, true); adapter->end_prompt(true);
      adapter->begin_prompt(n + 3, true); push(0, 3); adapter->end_prompt(true);
      sp(ds41_spark_ingest(direct.get(), input.p, n, 3));
      const float *u = nullptr, *q = nullptr, *confidence = nullptr;
      const void *hd = nullptr;
      sp(ds41_spark_forward(direct.get(), int(anchor), &u, &hd));
      sp(ds41_spark_markov(direct.get(), 0, int(anchor), &q, &confidence));
      std::vector<float> expected(129280);
      ck(cudaMemcpyAsync(expected.data(), q, expected.size() * 4,
                          cudaMemcpyDeviceToHost, stream));
      ck(cudaStreamSynchronize(stream));
      adapter->drafter()->begin(int(anchor));
      equal(adapter->drafter()->logits(0, int(anchor)), expected.data());
    }
    uint64_t prefill_values = 0;
    for (int n : {1, 16, 128, 129}) {
      adapter->reset(); upload(repeated); adapter->begin_prompt(n, false);
      PrefillOptions options;
      adapter->prefill(std::vector<int>(n, 0), options);
      adapter->end_prompt(true);
      sp(ds41_spark_seed(direct.get(),
          static_cast<char *>(input.p) + std::max(0, n - 128) * 30720, n));
      if (adapter->supports_prefill(PrefillMode::Append, 129) ||
          !adapter->supports_prefill(PrefillMode::Append, 128))
        throw std::runtime_error("cached append tail capability differs");
      adapter->begin_prompt(n + 3, true);
      options.mode = PrefillMode::Append;
      adapter->prefill(std::vector<int>(3, 0), options);
      adapter->end_prompt(true);
      sp(ds41_spark_ingest(direct.get(), input.p, n, 3));
      const float *u = nullptr, *q = nullptr, *confidence = nullptr;
      const void *hd = nullptr;
      sp(ds41_spark_forward(direct.get(), int(anchor), &u, &hd));
      sp(ds41_spark_markov(direct.get(), 0, int(anchor), &q, &confidence));
      std::vector<float> expected(129280);
      ck(cudaMemcpyAsync(expected.data(), q, expected.size() * 4,
                          cudaMemcpyDeviceToHost, stream));
      ck(cudaStreamSynchronize(stream));
      adapter->drafter()->begin(int(anchor));
      equal(adapter->drafter()->logits(0, int(anchor)), expected.data());
      prefill_values += expected.size();
    }
    adapter->reset(); adapter->begin_prompt(129, false); push(0, 6);
    adapter->end_prompt(false);
    bool refused = false;
    try { adapter->drafter()->begin(int(anchor)); }
    catch (const std::logic_error &) { refused = true; }
    if (!refused) throw std::runtime_error("incomplete seed allowed drafting");
    adapter->reset(); adapter->begin_prompt(1, false); push(0, 1);
    adapter->end_prompt(true);
    ck(cudaStreamSynchronize(stream));
    std::cout << Json({{"schema", "ds41-engine-spark-seed-v1"},
                       {"execution_scope", "correctness-only; no timing claim"},
                       {"frozen_chat_q_values_bitwise", 5 * 129280},
                       {"cold_append_q_values_bitwise", values - 5 * 129280},
                       {"cold_rows", {1, 16, 128, 129}},
                       {"equal_prompt_and_append", true},
                       {"explicit_prefill_tail_q_values_bitwise", prefill_values},
                       {"long_cached_append_uses_verify", true},
                       {"incomplete_prompt_guard_and_reset", true}}).dump() << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
