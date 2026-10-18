#include "ds41/engine.hpp"
#include <cassert>
#include <iostream>
#include <limits>
#include <numeric>
using namespace ds41e;
namespace {
std::vector<float> row(const std::vector<int> &prefix) {
  int next = (std::accumulate(prefix.begin(), prefix.end(), 0) +
              int(prefix.size()) + 1) %
             7;
  std::vector<float> out(7, -10);
  out[next] = 10;
  return out;
}
class Target : public Forward {
public:
  std::vector<int> committed, pending, captured;
  int width = 0;
  int vocab() const override { return 7; }
  int eos() const override { return 6; }
  void reset() override {
    committed.clear();
    pending.clear();
    captured.clear();
  }
  std::vector<std::vector<float>> verify(const std::vector<int> &ids) override {
    assert(pending.empty());
    pending = ids;
    width = int(ids.size());
    auto prefix = committed;
    std::vector<std::vector<float>> out;
    for (int id : ids) {
      prefix.push_back(id);
      out.push_back(row(prefix));
    }
    return out;
  }
  void commit(size_t count) override {
    assert(!pending.empty() && count <= pending.size());
    committed.insert(committed.end(), pending.begin(), pending.begin() + count);
    captured.insert(captured.end(), pending.begin(), pending.begin() + count);
    pending.clear();
  }
  Json counters() const override {
    return {{"expert_host_hits", committed.size() + pending.size()},
            {"expert_nvme_misses", 0}};
  }
};
class FastCounters : public Target {
public:
  mutable int snapshots = 0;
  Json counters() const override {
    throw std::logic_error("transaction copied full diagnostic history");
  }
  DemandCounters demand_counters() const override {
    ++snapshots;
    const double count = committed.size() + pending.size();
    return {count, 2 * count};
  }
};
class InvalidCounters : public Target {
public:
  mutable int snapshots = 0;
  bool nonfinite = false;
  DemandCounters demand_counters() const override {
    if (!snapshots++) return {2, 0};
    return {nonfinite ? std::numeric_limits<double>::quiet_NaN() : 1, 0};
  }
};
class Draft : public DraftProvider {
  Target &target;
  std::vector<int> prefix;

public:
  int width = 6, bad = -1, begins = 0, observes = 0, calls = 0;
  int64_t rolled = -1;
  bool chosen = false;
  DraftObservation observation;
  explicit Draft(Target &t) : target(t) {}
  int choose_width(int cap) override {
    assert(!chosen);
    chosen = true;
    return std::min(cap, width);
  }
  void begin(int anchor) override {
    assert(chosen);
    ++begins;
    prefix = target.committed;
    prefix.push_back(anchor);
  }
  std::vector<float> logits(int k, int previous) override {
    assert(chosen);
    ++calls;
    if (k)
      prefix.push_back(previous);
    else
      assert(prefix.back() == previous);
    auto out = row(prefix);
    if (k == bad) {
      int next = (argmax(out) + 1) % 7;
      out.assign(7, -10);
      out[next] = 10;
    }
    return out;
  }
  void observe(const DraftObservation &o) override {
    ++observes;
    observation = o;
  }
  void rollback(int64_t n) override {
    assert(n == int64_t(target.committed.size()));
    rolled = n;
    chosen = false;
  }
};
class Admission : public Draft {
  uint64_t last_nonce_ = 0, nonce_ = 0;
  bool live_ = false, frozen_ = false;
public:
  int final_width = 0, finalizations = 0, completions = 0, aborts = 0;
  bool fail_begin = false;
  explicit Admission(Target &target) : Draft(target) {}
  int prepare(const DraftContext &context) override {
    assert(!live_ && context.nonce > last_nonce_);
    nonce_ = context.nonce; live_ = true; frozen_ = false;
    calls = 0;
    return choose_width(context.maximum_width);
  }
  void begin(int anchor) override {
    Draft::begin(anchor);
    if (fail_begin) throw std::runtime_error("forward failed after admission");
  }
  int finalize_width(int preflight) override {
    assert(live_ && !frozen_ && calls == 0);
    frozen_ = true; ++finalizations;
    return final_width ? final_width : preflight;
  }
  std::vector<float> logits(int k, int previous) override {
    assert(live_ && frozen_);
    return Draft::logits(k, previous);
  }
  void complete(const DraftObservation &o, bool) override {
    assert(live_ && o.nonce == nonce_);
    assert(o.other_ms >= 0 && o.verify_ms > 0);
    ++completions; last_nonce_ = nonce_; live_ = false;
    Draft::observe(o);
  }
  void abort(uint64_t nonce) override {
    assert(live_ && nonce == nonce_);
    ++aborts; last_nonce_ = nonce_; live_ = false;
  }
};
void seed(Target &t) {
  t.verify({1, 2, 3});
  t.commit(3);
}
} // namespace
int main() {
  Sampling sampling;
  sampling.temperature = 0;
  sampling.seed = 42;
  // Per-transaction demand feedback must not request retained diagnostics.
  // RAM and NVMe are separate counters; host demand includes both tiers.
  for (int width : {1, 6}) {
    FastCounters t; seed(t); Draft d(t); d.width = width;
    SpeculativeStep step(t, &d, 0, 3, sampling, 4, 1, 6);
    assert(t.snapshots == 2 && step.observation().nvme_misses == 2 * width &&
           step.observation().host_misses == 3 * width);
    step.finish(step.result().emitted.size(), false);
    assert(t.snapshots == 2);
  }
  // Invalid or regressing snapshots abort the pending target transaction.
  for (bool nonfinite : {false, true}) {
    InvalidCounters t; seed(t); t.nonfinite = nonfinite;
    bool failed = false;
    try { SpeculativeStep step(t, nullptr, 0, 3, sampling, 4, 1, 1); }
    catch (const std::runtime_error &) { failed = true; }
    assert(failed && t.snapshots == 2 && t.pending.empty() &&
           t.committed.size() == 3);
  }
  // Admission may skip forward, or pay for forward/q0 and then narrow to one.
  for (bool paid : {false, true}) {
    Target t; seed(t); Admission d(t);
    d.width = paid ? 6 : 1; d.final_width = 1;
    SpeculativeStep step(t, &d, 0, 3, sampling, 4, 1, 6);
    assert(step.inputs().size() == 1 && d.calls == 0);
    assert(d.begins == int(paid) && d.finalizations == int(paid));
    step.finish(step.result().emitted.size(), false);
    assert(d.completions == 1 && d.aborts == 0);
    assert(d.observation.width == 1 && d.observation.tested_drafts == 0);
    assert(paid ? d.observation.draft_ms > 0 : d.observation.draft_ms == 0);
    assert(t.committed.size() == 4 && d.rolled == 4);
  }
  // A stop censors all labels beyond the delivered accepted prefix.
  {
    Target t; seed(t); Admission d(t);
    SpeculativeStep step(t, &d, 0, 3, sampling, 4, 1, 6);
    step.finish(2, true);
    assert(d.completions == 1 && d.observation.accepted_drafts == 2 &&
           d.observation.tested_drafts == 2 && d.aborts == 0);
  }
  // Cancellation retires the nonce without feeding observed acceptance labels.
  {
    Target t; seed(t); Admission d(t);
    SpeculativeStep step(t, &d, 0, 3, sampling, 4, 1, 6);
    step.finish(1, true, true);
    assert(d.completions == 0 && d.aborts == 1 && t.committed.size() == 5);
  }
  // Failure after forward admission still rolls back and retires its nonce.
  {
    Target t; seed(t); Admission d(t); d.fail_begin = true;
    bool failed = false;
    try { SpeculativeStep step(t, &d, 0, 3, sampling, 4, 1, 6); }
    catch (const std::runtime_error &) { failed = true; }
    assert(failed && d.aborts == 1 && d.rolled == 3 && t.pending.empty());
    d.fail_begin = false;
    { SpeculativeStep abandoned(t, &d, 0, 3, sampling, 4, 2, 6); }
    assert(d.aborts == 2 && t.committed.size() == 3 && t.pending.empty());
  }
  for (int width = 1; width <= 6; ++width)
    for (int bad = -1; bad < width - 1; ++bad) {
      Target t;
      seed(t);
      Draft d(t);
      d.width = width;
      d.bad = bad;
      auto expected = t.committed;
      int anchor = argmax(row(expected));
      SpeculativeStep step(t, &d, anchor, 3, sampling, 4, 1, 6);
      assert(t.width == width && d.begins == (width > 1));
      auto result = step.result();
      assert(result.accepted == size_t(bad < 0 ? width - 1 : bad));
      expected.push_back(anchor);
      for (int output : result.emitted) {
        assert(output == argmax(row(expected)));
        expected.push_back(output);
      }
      size_t inputs = step.finish(result.emitted.size(), false);
      assert(inputs == result.accepted_input_rows);
      expected.resize(3 + inputs);
      assert(t.committed == expected && t.captured == expected &&
             t.pending.empty());
      assert(d.observes == 1 && d.observation.width == width);
      assert(d.observation.host_misses == width &&
             d.observation.nvme_misses == 0);
    }
  // A stop or cancelled write can occur anywhere in an all-accepted burst.
  // Only the anchor and delivered accepted inputs may become capture history.
  for (size_t delivered = 0; delivered <= 6; ++delivered) {
    Target t;
    seed(t);
    Draft d(t);
    SpeculativeStep step(t, &d, 0, 3, sampling, 4, 1, 6);
    size_t inputs = step.finish(delivered, true);
    assert(inputs == 1 + std::min(delivered, size_t(5)));
    assert(t.committed.size() == 3 + inputs && t.captured == t.committed);
    assert(d.observes == 0 && t.pending.empty());
  }
  // Abandoned transactions do not advance target or capture state, but their
  // nonce is consumed and can never be reused by a later attempt.
  {
    Target t;
    seed(t);
    Draft d(t);
    AttemptCounter attempts;
    assert(attempts.take() == 0); // initial anchor
    {
      SpeculativeStep abandoned(t, &d, 0, 3, sampling, 4, attempts.take(), 6);
    }
    assert(attempts.take() == 2);
    assert(t.committed == std::vector<int>({1, 2, 3}) &&
           t.captured == t.committed);
    assert(t.pending.empty() && d.rolled == 3 && d.observes == 0);
  }
  // Output/context cap applies before any proposal identity can be inspected.
  {
    Target t;
    seed(t);
    Draft d(t);
    SpeculativeStep capped(t, &d, 0, 3, sampling, 4, 1, 1);
    assert(t.width == 1 && d.begins == 0 && d.calls == 0);
    capped.finish(1, true);
  }
  // No draft invokes the same owner verifier with zero proposals.
  {
    Target t;
    seed(t);
    SpeculativeStep plain(t, nullptr, 0, 3, sampling, 4, 1, 6);
    assert(t.width == 1 && plain.result().accepted == 0);
    plain.finish(1, false);
  }
  std::cout
      << "speculation: greedy correction/bonus, accepted captures, stop/cancel "
         "prefixes and abandonment passed (owner v2 sampler)\n";
}
