#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ds41e::hotpath {
constexpr int max_rows = 6, topk = 6, max_routes = max_rows * topk;
// Compact expert-major rows are independent of slot residency and completion
// order. destination maps back to the original [token, topk-slot] before the
// existing ascending-logical-ID reduction.
struct ExpertRows {
  uint32_t id;
  int begin, count;
};
struct RoutePlan {
  std::vector<uint32_t> ids;
  std::vector<ExpertRows> experts;
  std::array<int, max_routes> source{}, destination{};
  int count = 0;
};
inline RoutePlan plan_routes(const int *selected, int rows, int layer) {
  if (rows < 1 || rows > max_rows || layer < 0 || layer >= 40)
    throw std::invalid_argument("target route geometry");
  RoutePlan p;
  std::array<int, max_routes> order{};
  p.count = rows * topk;
  for (int i = 0; i < p.count; ++i) {
    if (selected[i] < 0 || selected[i] >= 384)
      throw std::invalid_argument("target route expert id");
    for (int j = i / topk * topk; j < i; ++j)
      if (selected[i] == selected[j])
        throw std::invalid_argument("duplicate per-token route");
    order[i] = i;
  }
  std::stable_sort(order.begin(), order.begin() + p.count,
                   [&](int a, int b) { return selected[a] < selected[b]; });
  for (int i = 0; i < p.count; ++i) {
    int slot = order[i];
    uint32_t id = uint32_t(layer * 384 + selected[slot]);
    if (p.ids.empty() || p.ids.back() != id) {
      p.ids.push_back(id);
      p.experts.push_back({id, i, 0});
    }
    ++p.experts.back().count;
    p.source[i] = slot / topk;
    p.destination[i] = slot;
  }
  return p;
}
} // namespace ds41e::hotpath
