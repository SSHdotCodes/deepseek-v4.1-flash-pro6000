#pragma once
#include <iosfwd>
#include <limits>
#include "ds41/target_shared.h"
#include "ds41/prefill/tiers.hpp"
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>
namespace ds41e {
using Json = nlohmann::json;
struct Config {
  std::string weights, receipt, engram, rope, experts, placement, trace_dir;
  std::string engram_library, fuse_library;
  // Draft-off only: keep head.weight in native BF16 (no FP32 promotion) and compute every target head
  // row with this library's ds41_fuse_head_decode (bitwise-gated against the promoted cuBLAS head).
  std::string fused_head_library;
  std::string auxiliary;
  int max_context = 1048576;
  size_t cuda_stack_bytes = 8192; // 0 leaves automatic driver growth unchanged
  size_t expert_slots = 64, host_bytes = 0;
  bool allow_unqualified = false;
  // E0 memory candidates, each opt-in and independently gated.
  bool host_embedding = false, reuse_woa = false, report_vram = false;
  // Class N: keep wo_a in native FP8 (implies reuse_woa) and run decode/verify rows through
  // ds41_dense_wo_a_compact (register dequant, FP32 FMA) instead of cuBLAS on a BF16 copy.
  bool compact_woa = false;
  bool graphs = false;
  bool load_mtp = false;
  bool prefix_cache = false, prefix_any_conversation = false;
  // Served cold/append prefill: FP8 projection rows per launch (PrefillOptions::dense_batch_rows, 1..8).
  size_t prefill_dense_batch = 1;
  // Class-N batched prefill: rows per dense/attention launch (0 = off, per-row M1 geometry).
  size_t prefill_chunk_rows = 0;
  bool spark = false;
  bool spark_policy_v2 = false;
  int spark_width = 0; // 0 adaptive;1..6 fixed candidate baseline
  bool batched_verify = false;
  bool verify_lanes = false; // pipelined two-lane verify (row 0 ahead of rows 1..)
  bool streamed_experts = false, prefetch_routes = false, hotpath_profile = false;
  // Width-1 decode: score layer l+1's router on layer l's FFN input and prefetch the predicted
  // RAM-resident experts into VRAM. A transfer hint only; routing and outputs are unchanged.
  bool prefetch_lookahead = false;
  // After a served prefill of >= prefill_warm_rows rows, read the experts routed for its last
  // prefill_warm_rows rows (most frequent first, up to prefill_warm_gib) from NVMe into RAM.
  size_t prefill_warm_rows = 0;
  double prefill_warm_gib = 16;
  bool dynamic_host = false, async_retire = false;
  int host_recent_window = -1, host_decay_interval = -1;
  bool reset_cache_history = false;
  bool hotpath_gate = false; // test-only full row evidence, never enabled by serving flags
  // Development-only explicit cold/append entrypoint. Never changes verify.
  bool prefill = false;
  ds41e::prefill::HostLookup prefill_host_view;
  size_t prefill_budget_bytes = size_t(20) << 30;
};
enum class PrefillMode { Cold, Append };
// One image of a prompt, as official inference/image_processor.py lays it out: the API resizes and pads
// the picture (PIL, like load_image) and sends its RGB bytes, n_vit_h*14 rows of n_vit_w*14 pixels. The
// prompt holds image_token_id at every position of the span
//   [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]
// and the IMAGE slots take the aligner rows in reading order.
struct ImageSpan {
  static constexpr int token_id = 129264, patch = 14, downsample = 3, max_tokens = 1024;
  int64_t start = 0; // absolute position of IMAGE_START
  int n_vit_h = 0, n_vit_w = 0, n_llm_h = 0, n_llm_w = 0;
  std::vector<uint8_t> pixels;
  std::string digest; // sha256 of the geometry and pixels: prefix-cache identity
  int64_t tokens() const { return int64_t(n_llm_h) * (n_llm_w + 1) + 2; }
  int64_t end() const { return start + tokens(); }
  // -1 text, 0 IMAGE_START, 1 IMAGE (aligner row in *row), 2 IMAGE_NEW_LINE, 3 IMAGE_END.
  int kind(int64_t pos, int *row) const {
    if (pos < start || pos >= end()) return -1;
    const int64_t i = pos - start;
    if (i == 0) return 0;
    if (pos == end() - 1) return 3;
    const int64_t r = (i - 1) / (n_llm_w + 1), c = (i - 1) % (n_llm_w + 1);
    if (c == n_llm_w) return 2;
    if (row) *row = int(r * n_llm_w + c);
    return 1;
  }
};
struct PrefillOptions {
  PrefillMode mode = PrefillMode::Cold;
  size_t logits_tail = 1;
  // Row scheduling boundary; every row retains M1 dense reduction geometry.
  size_t chunk_rows = 4096;
  size_t dense_batch_rows = 1; //1..8, fixed-M1 reductions and cuBLAS
  bool warm = true; // allow the configured post-prefill RAM warm (only the final segment of a prompt)
  // The prompt's images at absolute positions, sorted and disjoint; rows outside [committed, committed+n)
  // are ignored, so the whole list can be passed with every segment.
  const std::vector<ImageSpan> *images = nullptr;
};
struct PrefillResult {
  int64_t begin_position = 0;
  size_t consumed_rows = 0;
  double elapsed_ms = 0;
  std::vector<std::vector<float>> logits;
  Json layers = Json::array();
};
enum class PrefetchIntent { NextLayer, Draft, Session };
// A forward consumes positions [committed, committed+ids.size()). The caller
// commits an accepted input prefix, never the correction/bonus token it emits.
class DraftProvider;
// Demand-only snapshots are independent of diagnostic history retention.
// Double fields preserve existing policy feedback arithmetic for all backends.
struct DemandCounters {
  double host_hits = 0, nvme_misses = 0;
};
class Forward {
public:
  virtual ~Forward() = default;
  virtual int vocab() const = 0;
  virtual int eos() const = 0;
  // Reset request-local metrics while retaining committed model state.
  virtual void begin_request() {}
  // Target prompt scheduling and drafter cold-seed geometry are independent.
  virtual void begin_prompt(int64_t, bool) {}
  virtual void end_prompt(bool) {}
  virtual void reset() = 0;
  virtual std::vector<std::vector<float>>
  verify(const std::vector<int> &ids) = 0;
  virtual void commit(size_t accepted_input_rows) = 0;
  // Consumes and commits all input rows. Append retains the live exact state;
  // cold requires position zero. Errors after mutation poison the target.
  virtual bool supports_prefill(PrefillMode, size_t) const { return false; }
  virtual PrefillResult prefill(const std::vector<int>&, const PrefillOptions&) {
    throw std::logic_error("prefill unsupported");
  }
  virtual void rollback(int64_t) {
    throw std::logic_error("rollback unsupported");
  }

  // Session snapshot at the committed frontier (no pending verify). It holds the
  // bytes that later positions overwrite in place (SWA rings, partial compressor
  // groups); position-addressed global KV beyond the frontier stays masked.
  struct Snapshot { virtual ~Snapshot() = default; int64_t position = 0; };
  virtual std::shared_ptr<Snapshot> snapshot() { return nullptr; }
  virtual void restore(const Snapshot &) { throw std::logic_error("restore unsupported"); }
  virtual std::vector<int> cached_prefix() const { return {}; }
  // Diagnostic only: position-addressed, logically live bytes for X3/X3b.
  virtual std::map<std::string,std::vector<uint8_t>> live_state() {
    throw std::logic_error("live state inspection unsupported");
  }
  virtual ds41_target_weights_v2 shared_weights() const { return {}; }
  virtual ds41_target_services_v1 services() { return {}; }
  virtual ds41_target_tensor_v1 resident_tensor(const std::string &) const {
    return {};
  }
  virtual std::vector<std::string> resident_names(const std::string &) const {
    return {};
  }
  virtual ds41_target_capture_v1 captures() const { return {}; }
  virtual void set_capture_sink(ds41_target_capture_sink, void *) {
    throw std::logic_error("capture sink unsupported");
  }
  virtual uint64_t prefetch_experts(const std::vector<uint32_t> &, PrefetchIntent) {
    throw std::logic_error("expert prefetch unsupported");
  }
  virtual void cancel_prefetch(uint64_t) {
    throw std::logic_error("expert prefetch unsupported");
  }
  virtual Json counters() const { return Json::object(); }
  virtual DemandCounters demand_counters() const;
  virtual DraftProvider *drafter() { return nullptr; }
};
std::unique_ptr<Forward> make_native(const Config &config);
std::unique_ptr<Forward> make_spark_target(std::unique_ptr<Forward>,
                                           int fixed_width,
                                           bool policy_v2 = false);
std::unique_ptr<Forward> make_replay(const std::string &path);
struct Sampling {
  double temperature = 1, top_p = 1;
  int max_tokens = 4096, logprobs = 0;
  uint64_t seed = 0;
  std::vector<int> stop;
};
enum class Purpose : uint32_t {
  Target = 0,
  Draft = 1,
  Accept = 2,
  Residual = 3
};
struct Rng {
  uint64_t seed;
  uint32_t word(uint64_t step, uint32_t slot, Purpose purpose,
                uint64_t draw = 0) const;
  double uniform(uint64_t step, uint32_t slot, Purpose purpose,
                 uint64_t draw = 0) const;
};
class AttemptCounter {
  uint64_t next_;
  bool exhausted_ = false;

public:
  explicit AttemptCounter(uint64_t first = 0) : next_(first) {}
  AttemptCounter(const AttemptCounter &) = delete;
  AttemptCounter &operator=(const AttemptCounter &) = delete;
  uint64_t take();
};
using Distribution = std::vector<double>;
Distribution probabilities(const std::vector<float> &logits, double temperature,
                           double top_p);
std::vector<Distribution> probabilities_rows(const std::vector<std::vector<float>> &rows,
                                             double temperature, double top_p);
int sample(const Distribution &, const Rng &, uint64_t step, uint32_t slot,
           Purpose);
int argmax(const std::vector<float> &logits);
struct Verification {
  std::vector<int> emitted;
  size_t accepted = 0;
  size_t accepted_input_rows = 0;
  bool rejected = false;
};
Verification reject_sample(const std::vector<int> &proposals,
                           const std::vector<Distribution> &q,
                           const std::vector<Distribution> &p, const Rng &,
                           uint64_t transaction_step, double temperature);
struct DraftObservation {
  int width = 1;
  uint64_t nonce = 0;
  size_t accepted_drafts = 0, tested_drafts = 0;
  double draft_ms = 0, verify_ms = 0, other_ms = 0;
  double host_misses = 0, nvme_misses = 0;
};
struct DraftContext {
  int anchor, maximum_width;
  int64_t committed_inputs;
  uint64_t nonce;
  double temperature, top_p;
};
class DraftProvider {
public:
  virtual ~DraftProvider() = default;
  // Raw drafter confidence-head output for the last logits() call (NaN if unavailable).
  virtual float last_confidence() const { return std::numeric_limits<float>::quiet_NaN(); }
  // Selection must use only previously completed observations. It is called
  // before begin/logits and cannot inspect this step's sampled proposals.
  virtual int choose_width(int maximum) = 0;
  virtual int prepare(const DraftContext &context) {
    return choose_width(context.maximum_width);
  }
  virtual void begin(int anchor) = 0;
  // Optional q0-only refinement. This runs before any current proposal draw.
  virtual int finalize_width(int preflight) { return preflight; }
  virtual std::vector<float> logits(int block_position, int previous) = 0;
  virtual void observe(const DraftObservation &) = 0;
  virtual void complete(const DraftObservation &observation, bool censored) {
    if (!censored) observe(observation);
  }
  virtual void abort(uint64_t) {}
  virtual void rollback(int64_t committed_inputs) = 0;
};
// Owns one pending target verify. The server decides how much of the sampled
// burst it delivered before finishing, so stop/cancel never commit future rows.
class SpeculativeStep {
  Forward &target_;
  DraftProvider *draft_;
  int64_t begin_;
  bool pending_ = false;
  bool draft_attempt_ = false;
  double started_ms_ = 0;
  void abort_draft() noexcept;
  std::vector<int> inputs_;
  std::vector<std::vector<float>> rows_;
  Verification result_;
  DraftObservation observed_;
  double stop_floor_ = 0;
  const std::vector<double> *stop_model_ = nullptr, *stop_scale_ = nullptr;
  std::ofstream *stop_log_ = nullptr;
  std::vector<double> confidence_;
  std::vector<float> head_confidence_;

public:
  SpeculativeStep(Forward &, DraftProvider *, int anchor,
                  int64_t committed_inputs, const Sampling &,
                  uint64_t first_output_position, uint64_t transaction_step,
                  int maximum_width);
  ~SpeculativeStep();
  SpeculativeStep(const SpeculativeStep &) = delete;
  SpeculativeStep &operator=(const SpeculativeStep &) = delete;
  const Verification &result() const { return result_; }
  const std::vector<std::vector<float>> &rows() const { return rows_; }
  const std::vector<int> &inputs() const { return inputs_; }
  const DraftObservation &observation() const { return observed_; }
  // terminal=true censors policy observation (stop, output cap, cancellation).
  // Returned rows include the already-emitted anchor and accepted input prefix.
  size_t finish(size_t delivered_outputs, bool terminal, bool cancelled = false);
};
void serve(const std::string &socket, std::unique_ptr<Forward> model,
           int max_context, bool prefix_cache = false, bool prefix_any_conversation = false,
           size_t prefill_dense_batch = 1);
} // namespace ds41e
