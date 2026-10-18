// Bitwise equivalence of the parallel default-sampler path and the in-place no-draft verify
// against libds41_spark's sequential reference.
#include "ds41/engine.hpp"
#include "ds41/spark.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
using namespace ds41e;
static void require(bool ok, const char *what) { if (!ok) throw std::runtime_error(what); }
int main() {
  std::mt19937_64 gen(20261018);
  const int sizes[] = {4096, 4097, 65537, 129280, 129280, 129281};
  const double temps[] = {1.0, 0.7, 1e-7, 2.5, 1.0};
  size_t rows = 0;
  for (int n : sizes)
    for (double t : temps)
      for (int trial = 0; trial < 4; ++trial) {
        std::vector<float> x(n);
        std::normal_distribution<float> d(0.f, trial == 3 ? 40.f : 4.f);
        for (auto &v : x) v = d(gen);
        if (trial == 1) {                       // ties at the maximum, first occurrence must win
          float m = *std::max_element(x.begin(), x.end()) + 1.f;
          x[n / 3] = m; x[n / 2] = m; x[n - 1] = m;
        }
        if (trial == 2) for (int i = 0; i < n; i += 7) x[i] = std::round(x[i]);
        auto fast = probabilities(x, t, 1.0);
        std::vector<double> ref(n);
        require(ds41_spark_probabilities(x.data(), ref.data(), n, std::max(t, 1e-5), 1.0) == 0, "reference failed");
        require(std::memcmp(fast.data(), ref.data(), n * sizeof(double)) == 0, "probability row differs");
        for (uint64_t step = 0; step < 3; ++step) {
          Rng rng{0x5eedULL + uint64_t(trial)};
          auto v = reject_sample({}, {}, {fast}, rng, step, t);
          ds41_spark_result r{};
          require(ds41_spark_verify_v2(nullptr, nullptr, ref.data(), 0, n, 0, rng.seed, step, &r) == 0, "reference verify failed");
          require(v.emitted.size() == size_t(r.emitted) && v.emitted[0] == r.tokens[0] &&
                  v.accepted == r.accepted_drafts && v.accepted_input_rows == r.accepted_input_rows &&
                  v.rejected == r.rejected, "no-draft verify differs");
        }
        ++rows;
      }
  std::vector<float> bad(129280, 0.f); bad[77] = NAN;
  bool threw = false;
  try { probabilities(bad, 1.0, 1.0); } catch (const std::runtime_error &) { threw = true; }
  require(threw, "nonfinite logits accepted");
  std::vector<float> x(129280);
  std::normal_distribution<float> d(0.f, 4.f);
  for (auto &v : x) v = d(gen);
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 200; ++i) probabilities(x, 1.0, 1.0);
  auto t1 = std::chrono::steady_clock::now();
  std::vector<double> ref(x.size());
  for (int i = 0; i < 200; ++i) ds41_spark_probabilities(x.data(), ref.data(), int(x.size()), 1.0, 1.0);
  auto t2 = std::chrono::steady_clock::now();
  auto us = [](auto a, auto b) { return std::chrono::duration<double, std::micro>(b - a).count() / 200; };
  std::printf("sampling_parallel: %zu rows bitwise equal; parallel %.1f us, reference %.1f us per 129280-row\n",
              rows, us(t0, t1), us(t1, t2));
}
