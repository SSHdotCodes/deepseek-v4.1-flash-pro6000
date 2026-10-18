#include "ds41/engine.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>

namespace ds41e {
namespace {
double milliseconds() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
double counter(const Json &j, const char *name) {
  double value = j.value(name, 0.0);
  if (!std::isfinite(value) || value < 0)
    throw std::runtime_error("invalid target tier counter");
  return value;
}
double delta(double before, double after) {
  if (!std::isfinite(before) || !std::isfinite(after) || before < 0 || after < 0)
    throw std::runtime_error("invalid target tier counter");
  if (after < before)
    throw std::runtime_error("target tier counter moved backwards");
  return after - before;
}
std::mutex &stop_mutex() { static std::mutex m; return m; }
double &last_finish_ms() { static double v = 0; return v; }
std::ofstream *open_stop_log() {
  const char *path = std::getenv("DS41_SPARK_STOP_LOG");
  if (!path || !*path) return nullptr;
  static std::ofstream out(path, std::ios::app);
  if (!out) throw std::runtime_error("DS41_SPARK_STOP_LOG open");
  return &out;
}
// Opt-in DS41_SPARK_STOP_MODEL="b0,b1,b2,b3,b4": acceptance estimate
// sigmoid(b0 + b1*c + b2*log(c) + b3*r + b4*c*r) from max q_k (c) and the drafter confidence head (r).
std::vector<double> stop_model() {
  std::vector<double> b;
  if (const char *v = std::getenv("DS41_SPARK_STOP_MODEL")) {
    std::string s(v); size_t at = 0;
    while (at <= s.size()) { size_t e = s.find(',', at); if (e == std::string::npos) e = s.size();
      b.push_back(std::stod(s.substr(at, e - at))); at = e + 1; }
    if (b.size() != 5) throw std::invalid_argument("DS41_SPARK_STOP_MODEL needs 5 coefficients");
  }
  return b;
}
// Opt-in DS41_SPARK_STOP_SCALE="s0,s1,...": per-draft multipliers of the stop floor (draft k uses
// floor * s_k; missing entries reuse the last). Scales with each extra verify row's marginal cost.
std::vector<double> stop_scale() {
  std::vector<double> v;
  if (const char *e = std::getenv("DS41_SPARK_STOP_SCALE")) {
    std::string s(e); size_t at = 0;
    while (at <= s.size()) { size_t x = s.find(',', at); if (x == std::string::npos) x = s.size();
      v.push_back(std::stod(s.substr(at, x - at))); at = x + 1; }
    for (double d : v) if (!std::isfinite(d) || d <= 0) throw std::invalid_argument("DS41_SPARK_STOP_SCALE");
  }
  return v;
}
double stop_floor() {
  const char *v = std::getenv("DS41_SPARK_STOP_Q");
  double f = v ? std::stod(v) : 0;
  if (!std::isfinite(f) || f < 0 || f >= 1) throw std::invalid_argument("DS41_SPARK_STOP_Q in [0,1)");
  return f;
}
} // namespace

DemandCounters Forward::demand_counters() const {
  const auto j = counters();
  return {counter(j, "expert_host_hits"), counter(j, "expert_nvme_misses")};
}

SpeculativeStep::SpeculativeStep(Forward &target, DraftProvider *draft,
                                 int anchor, int64_t committed,
                                 const Sampling &sampling, uint64_t position,
                                 uint64_t step, int maximum)
    : target_(target), draft_(draft), begin_(committed) {
  static const double floor = stop_floor();
  static const std::vector<double> model = stop_model();
  static const std::vector<double> scales = stop_scale();
  stop_model_ = &model; stop_scale_ = &scales;
  static std::ofstream *log = open_stop_log();
  stop_floor_ = floor; stop_log_ = log;
  started_ms_ = milliseconds();
  observed_.nonce = step;
  if (committed < 0 || position != uint64_t(committed) + 1 || maximum < 1 ||
      maximum > 6 || anchor < 0 || anchor >= target.vocab())
    throw std::invalid_argument("speculative step position/width/anchor");
  try {
    draft_attempt_ = draft != nullptr;
    int width = draft ? draft->prepare({anchor, maximum, committed, step,
                                       sampling.temperature, sampling.top_p}) : 1;
    if (width < 1 || width > maximum)
      throw std::runtime_error("draft policy returned invalid width");
    observed_.width = width;
    inputs_.push_back(anchor);
    std::vector<int> proposals;
    std::vector<Distribution> q;
    const Rng rng{sampling.seed};
    if (width > 1) {
      double started = milliseconds();
      draft->begin(anchor);
      observed_.draft_ms += milliseconds() - started;
      width = draft->finalize_width(width);
      if (width < 1 || width > maximum)
        throw std::runtime_error("final draft width outside negotiated capacity");
      observed_.width = width;
      started = milliseconds();
      // Opt-in stopping rule (DS41_SPARK_STOP_Q): before sampling draft k, stop if the running
      // product of draft confidences (max q_j, j <= k) falls below the floor. The decision uses only
      // the prefix, earlier drafts and q_k before d_k is drawn, so T=1 verification stays exact.
      double reach = 1;
      for (int k = 0; k < width - 1; ++k) {
        auto logits = draft->logits(k, inputs_.back());
        if (logits.size() != size_t(target.vocab()))
          throw std::runtime_error("draft vocabulary mismatch");
        if (stop_floor_ > 0 || stop_log_) {
          double c = sampling.temperature == 0 ? 0 : 1;
          if (sampling.temperature == 0) {
            auto d = probabilities(logits, 1, 1);
            c = *std::max_element(d.begin(), d.end());
          } else {
            q.push_back(probabilities(logits, sampling.temperature, sampling.top_p));
            c = *std::max_element(q.back().begin(), q.back().end());
          }
          confidence_.push_back(c);
          const float head = draft->last_confidence();
          head_confidence_.push_back(head);
          double estimate = c;
          if (!stop_model_->empty() && std::isfinite(head)) {
            const auto &b = *stop_model_;
            const double z = b[0] + b[1] * c + b[2] * std::log(std::max(c, 1e-6)) + b[3] * head + b[4] * c * head;
            estimate = 1 / (1 + std::exp(-z));
          }
          c = estimate;
          const double scale = stop_scale_->empty() ? 1.0
              : (*stop_scale_)[std::min<size_t>(size_t(k), stop_scale_->size() - 1)];
          if (reach * c < stop_floor_ * scale) {
            if (sampling.temperature != 0) q.pop_back();
            break;
          }
          reach *= c;
        }
        int token;
        if (sampling.temperature == 0) {
          token = argmax(logits);
        } else {
          if (q.size() == size_t(k))
            q.push_back(
                probabilities(logits, sampling.temperature, sampling.top_p));
          token = sample(q.back(), rng, step, uint32_t(k), Purpose::Draft);
        }
        proposals.push_back(token);
        inputs_.push_back(token);
      }
      width = int(inputs_.size());
      observed_.width = width;
      observed_.draft_ms += milliseconds() - started;
    }
    // Draft expert work is deliberately outside these target-only deltas.
    const auto before = target.demand_counters();
    const double started = milliseconds();
    rows_ = target.verify(inputs_);
    pending_ = true;
    observed_.verify_ms = milliseconds() - started;
    const auto after = target.demand_counters();
    observed_.nvme_misses = delta(before.nvme_misses, after.nvme_misses);
    observed_.host_misses =
        observed_.nvme_misses + delta(before.host_hits, after.host_hits);
    if (rows_.size() != inputs_.size())
      throw std::runtime_error("target verify returned wrong row count");
    const double probs_start = milliseconds();
    for (const auto &row : rows_)
      if (row.size() != size_t(target.vocab()))
        throw std::runtime_error("target vocabulary mismatch");
    std::vector<Distribution> p =
        probabilities_rows(rows_, sampling.temperature, sampling.top_p);
    const double reject_start = milliseconds();
    result_ = reject_sample(proposals, q, p, rng, step, sampling.temperature);
    const double reject_end = milliseconds();
    observed_.accepted_drafts = result_.accepted;
    observed_.tested_drafts = result_.accepted + size_t(result_.rejected);
    if (stop_log_) {
      std::lock_guard<std::mutex> lock(stop_mutex());
      *stop_log_ << width << ' ' << result_.accepted;
      for (double c : confidence_) *stop_log_ << ' ' << c;
      *stop_log_ << " R";
      for (float c : head_confidence_) *stop_log_ << ' ' << c;
      *stop_log_ << " T " << observed_.draft_ms << ' ' << observed_.verify_ms << ' '
                 << reject_start - probs_start << ' ' << reject_end - reject_start << ' ' << last_finish_ms();
      *stop_log_ << '\n' << std::flush;
    }
  } catch (...) {
    if (pending_) {
      try {
        target_.commit(0);
        pending_ = false;
      } catch (...) {
        // Native forward and its owner adapter poison state on failed work.
      }
    }
    if (draft_) {
      try { draft_->rollback(begin_); } catch (...) {}
    }
    abort_draft();
    throw;
  }
}

SpeculativeStep::~SpeculativeStep() {
  if (pending_) {
    try {
      target_.commit(0);
      if (draft_)
        draft_->rollback(begin_);
    } catch (...) {
      // Destruction cannot hide the original exception. No further request may
      // use a native target whose commit/capture or callback failed.
    }
  }
  abort_draft();
}

void SpeculativeStep::abort_draft() noexcept {
  if (draft_attempt_) {
    draft_attempt_ = false;
    try { draft_->abort(observed_.nonce); } catch (...) {}
  }
}

size_t SpeculativeStep::finish(size_t delivered, bool terminal, bool cancelled) {
  struct Timer { double t0 = milliseconds(); ~Timer() { last_finish_ms() = milliseconds() - t0; } } timer;
  if (!pending_ || delivered > result_.emitted.size() ||
      (!terminal && delivered != result_.emitted.size()) || (cancelled && !terminal))
    throw std::invalid_argument("speculative delivered prefix/transaction");
  size_t rows = 1 + std::min(delivered, result_.accepted);
  try {
    target_.commit(rows);
    pending_ = false;
    if (draft_) {
      draft_->rollback(begin_ + int64_t(rows));
      observed_.accepted_drafts = std::min(delivered, result_.accepted);
      observed_.tested_drafts = observed_.accepted_drafts;
      if (delivered == result_.emitted.size() && result_.rejected)
        ++observed_.tested_drafts;
      observed_.other_ms = std::max(0.0, milliseconds() - started_ms_ -
                                        observed_.draft_ms - observed_.verify_ms);
      if (cancelled) abort_draft();
      else {
        draft_->complete(observed_, terminal);
        draft_attempt_ = false;
      }
    }
  } catch (...) {
    abort_draft();
    throw;
  }
  return rows;
}
} // namespace ds41e
