#include "ds41/engine.hpp"
#include "ds41/spark.h"
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
namespace ds41e {
namespace {
// Small persistent pool for the per-token vocabulary loops. run(n, f) calls f(piece, begin, end) on the
// contiguous pieces of [0, n) (the caller takes piece 0) and returns after every piece finished.
class VocabPool {
public:
  static VocabPool &get() { static VocabPool pool; return pool; }
  size_t parts() const { return workers_.size() + 1; }
  using Body = std::function<void(size_t, size_t, size_t)>;
  void run(size_t n, const Body &f) {
    std::unique_lock<std::mutex> lock(mutex_);
    job_ = &f; n_ = n; pending_ = workers_.size(); error_ = nullptr; ++generation_;
    lock.unlock();
    wake_.notify_all();
    std::exception_ptr mine;
    try { piece(0, f); } catch (...) { mine = std::current_exception(); }
    lock.lock();
    done_.wait(lock, [&] { return pending_ == 0; });
    job_ = nullptr;
    if (mine) std::rethrow_exception(mine);
    if (error_) std::rethrow_exception(error_);
  }
  ~VocabPool() {
    { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; ++generation_; }
    wake_.notify_all();
    for (auto &t : workers_) t.join();
  }
private:
  VocabPool() {
    for (size_t i = 1; i < 8; ++i) workers_.emplace_back([this, i] { loop(i); });
  }
  void piece(size_t index, const Body &f) {
    const size_t k = parts(), begin = n_ * index / k, end = n_ * (index + 1) / k;
    if (begin < end) f(index, begin, end);
  }
  void loop(size_t index) {
    uint64_t seen = 0;
    for (;;) {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [&] { return generation_ != seen; });
      seen = generation_;
      if (stop_) return;
      const auto *f = job_;
      lock.unlock();
      std::exception_ptr e;
      try { piece(index, *f); } catch (...) { e = std::current_exception(); }
      lock.lock();
      if (e && !error_) error_ = e;
      if (--pending_ == 0) done_.notify_one();
    }
  }
  std::mutex mutex_;
  std::condition_variable wake_, done_;
  std::vector<std::thread> workers_;
  const Body *job_ = nullptr;
  size_t n_ = 0, pending_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
  std::exception_ptr error_;
};
// Exact fast path of ds41_spark_probabilities for t > 0, top_p == 1: the same first-maximum argmax, the same
// per-element exp((double(x[i]) - x[b]) / t), the same sequential normalizer and the same p[i] /= z. Only the
// independent per-element loops run in parallel; exp is a pure function, so every probability is bitwise equal.
void parallel_probabilities(const float *x, double *p, size_t n, double t) {
  auto &pool = VocabPool::get();
  std::vector<size_t> best(pool.parts(), SIZE_MAX);
  std::vector<char> bad(pool.parts(), 0);
  pool.run(n, [&](size_t piece, size_t begin, size_t end) {
    size_t b = begin;
    for (size_t i = begin; i < end; ++i) {
      if (!std::isfinite(x[i])) { bad[piece] = 1; return; }
      if (x[i] > x[b]) b = i;
    }
    best[piece] = b;
  });
  if (std::any_of(bad.begin(), bad.end(), [](char c) { return c; }))
    throw std::runtime_error("libds41_spark sampler: nonfinite logits");
  size_t b = SIZE_MAX;
  for (size_t c : best)
    if (c != SIZE_MAX && (b == SIZE_MAX || x[c] > x[b])) b = c;
  const float top = x[b];
  pool.run(n, [&](size_t, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) p[i] = std::exp((double(x[i]) - top) / t);
  });
  double z = 0;
  for (size_t i = 0; i < n; ++i) z += p[i];
  pool.run(n, [&](size_t, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) p[i] /= z;
  });
}
void spark_check(int rc) {
  if (rc)
    throw std::runtime_error(std::string("libds41_spark sampler: ") +
                             ds41_spark_error());
}
int width(size_t n) {
  if (!n || n > size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("empty/oversized sampler vocabulary");
  return int(n);
}
} // namespace
uint64_t AttemptCounter::take() {
  if (exhausted_)
    throw std::overflow_error("transaction nonce exhausted");
  const uint64_t out = next_;
  if (next_ == UINT64_MAX)
    exhausted_ = true;
  else
    ++next_;
  return out;
}
uint32_t Rng::word(uint64_t step, uint32_t slot, Purpose purpose,
                   uint64_t draw) const {
  if (draw / 4 > UINT32_MAX || slot > 5 || uint32_t(purpose) > 3)
    throw std::overflow_error("Philox draw counter");
  return ds41_spark_rng_word_v2(seed, step, slot, uint32_t(purpose), draw);
}
double Rng::uniform(uint64_t step, uint32_t slot, Purpose purpose,
                    uint64_t draw) const {
  if (draw > 8589934591ULL || slot > 5 || uint32_t(purpose) > 3)
    throw std::overflow_error("Philox draw counter");
  return ds41_spark_uniform_v2(seed, step, slot, uint32_t(purpose), draw);
}
int argmax(const std::vector<float> &x) {
  width(x.size());
  for (float f : x)
    if (!std::isfinite(f))
      throw std::runtime_error("non-finite target logits");
  return int(std::max_element(x.begin(), x.end()) - x.begin());
}
// Every verify row at once: one pool dispatch per pass over the flattened rows and the per-row sequential
// normalizers run concurrently. Per element and per row this is the same arithmetic as probabilities().
std::vector<Distribution> probabilities_rows(const std::vector<std::vector<float>> &rows,
                                             double temp, double top_p) {
  const double t = temp > 0 ? std::max(temp, 1e-5) : temp;
  const size_t m = rows.size(), n = m ? rows[0].size() : 0;
  bool fast = m > 1 && t > 0 && std::isfinite(t) && top_p == 1.0 && n >= 4096 &&
              n <= size_t(std::numeric_limits<int>::max());
  for (const auto &r : rows) fast = fast && r.size() == n;
  std::vector<Distribution> p;
  if (!fast) {
    for (const auto &r : rows) p.push_back(probabilities(r, temp, top_p));
    return p;
  }
  p.assign(m, Distribution(n));
  auto &pool = VocabPool::get();
  const size_t parts = pool.parts();
  std::vector<size_t> best(m * parts, SIZE_MAX);
  std::vector<char> bad(parts, 0);
  pool.run(m * n, [&](size_t piece, size_t begin, size_t end) {
    for (size_t i = begin; i < end;) {
      const size_t r = i / n, stop = std::min(end, (r + 1) * n);
      const float *x = rows[r].data();
      size_t b = i - r * n;
      for (size_t j = i - r * n; j < stop - r * n; ++j) {
        if (!std::isfinite(x[j])) { bad[piece] = 1; return; }
        if (x[j] > x[b]) b = j;
      }
      best[r * parts + piece] = b;
      i = stop;
    }
  });
  if (std::any_of(bad.begin(), bad.end(), [](char c) { return c; }))
    throw std::runtime_error("libds41_spark sampler: nonfinite logits");
  std::vector<float> top(m);
  for (size_t r = 0; r < m; ++r) {
    size_t b = SIZE_MAX;
    for (size_t k = 0; k < parts; ++k) {
      const size_t c = best[r * parts + k];
      if (c != SIZE_MAX && (b == SIZE_MAX || rows[r][c] > rows[r][b])) b = c;
    }
    top[r] = rows[r][b];
  }
  pool.run(m * n, [&](size_t, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      const size_t r = i / n, j = i - r * n;
      p[r][j] = std::exp((double(rows[r][j]) - top[r]) / t);
    }
  });
  std::vector<double> z(m, 0);
  pool.run(m, [&](size_t, size_t begin, size_t end) {
    for (size_t r = begin; r < end; ++r) {
      double sum = 0;
      for (size_t j = 0; j < n; ++j) sum += p[r][j];
      z[r] = sum;
    }
  });
  pool.run(m * n, [&](size_t, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      const size_t r = i / n;
      p[r][i - r * n] /= z[r];
    }
  });
  return p;
}
Distribution probabilities(const std::vector<float> &x, double temp,
                           double top_p) {
  Distribution p(x.size());
  // Preserve the official tiny-temperature floor at the adapter boundary.
  if (temp > 0)
    temp = std::max(temp, 1e-5);
  if (temp > 0 && std::isfinite(temp) && top_p == 1.0 && x.size() >= 4096 &&
      x.size() <= size_t(std::numeric_limits<int>::max())) {
    parallel_probabilities(x.data(), p.data(), x.size(), temp);
    return p;
  }
  spark_check(ds41_spark_probabilities(x.data(), p.data(), width(x.size()),
                                       temp, top_p));
  return p;
}
int sample(const Distribution &p, const Rng &rng, uint64_t step, uint32_t slot,
           Purpose purpose) {
  int32_t token = -1;
  spark_check(ds41_spark_sample_v2(p.data(), width(p.size()), rng.seed, step,
                                   slot, uint32_t(purpose), &token));
  return token;
}
Verification reject_sample(const std::vector<int> &proposals,
                           const std::vector<Distribution> &q,
                           const std::vector<Distribution> &p, const Rng &rng,
                           uint64_t step, double temperature) {
  bool greedy = temperature == 0;
  if (!std::isfinite(temperature) || temperature < 0 || proposals.size() > 5 ||
      ((!greedy || !q.empty()) && q.size() != proposals.size()) ||
      p.size() != proposals.size() + 1)
    throw std::invalid_argument("speculative row count");
  int vocab = width(p[0].size());
  if (proposals.empty()) {
    // No draft rows: verify the single target row in place instead of copying it.
    if (p[0].size() != size_t(vocab))
      throw std::invalid_argument("target vocabulary mismatch");
    ds41_spark_result result{};
    spark_check(ds41_spark_verify_v2(nullptr, nullptr, p[0].data(), 0, vocab,
                                     greedy ? 1 : 0, rng.seed, step, &result));
    Verification out;
    out.emitted.assign(result.tokens, result.tokens + result.emitted);
    out.accepted = result.accepted_drafts;
    out.accepted_input_rows = result.accepted_input_rows;
    out.rejected = result.rejected;
    return out;
  }
  std::vector<double> target, draft;
  target.reserve(p.size() * vocab);
  draft.reserve(q.size() * vocab);
  for (const auto &row : p) {
    if (row.size() != size_t(vocab))
      throw std::invalid_argument("target vocabulary mismatch");
    target.insert(target.end(), row.begin(), row.end());
  }
  for (const auto &row : q) {
    if (row.size() != size_t(vocab))
      throw std::invalid_argument("draft vocabulary mismatch");
    draft.insert(draft.end(), row.begin(), row.end());
  }
  std::vector<int32_t> ids(proposals.begin(), proposals.end());
  ds41_spark_result result{};
  spark_check(ds41_spark_verify_v2(ids.data(), draft.data(), target.data(),
                                   int(ids.size()), vocab, greedy ? 1 : 0,
                                   rng.seed, step, &result));
  Verification out;
  out.emitted.assign(result.tokens, result.tokens + result.emitted);
  out.accepted = result.accepted_drafts;
  out.accepted_input_rows = result.accepted_input_rows;
  out.rejected = result.rejected;
  return out;
}
} // namespace ds41e
